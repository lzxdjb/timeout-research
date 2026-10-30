# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CPU regressions for exact-token external teacher scoring and resource isolation."""

import asyncio
import copy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import torch
from aiohttp import web

from verl.experimental.teacher_loop.errors import TeacherScoringError
from verl.experimental.teacher_loop.external_teacher import (
    ExternalTeacherClient,
    ExternalTeacherScorer,
    parse_prompt_logprobs,
)
from verl.utils.async_rpc import RPCCancellationError, rollout_request_scope
from verl.workers.config.distillation import (
    DistillationConfig,
    DistillationLossConfig,
    DistillationTeacherModelConfig,
    ExternalTeacherConfig,
)
from verl.workers.config.rollout import RolloutConfig
from verl.workers.utils.padding import no_padding_2_padding


def _external(**kwargs):
    return ExternalTeacherConfig(base_url="http://127.0.0.1:1", model_name="teacher", max_model_len=16, **kwargs)


def _teacher(backend="external", key=None):
    return DistillationTeacherModelConfig(
        key=key,
        backend=backend,
        external=_external(),
        model_path="managed/checkpoint" if backend == "managed" else None,
        inference=RolloutConfig(name="vllm", prompt_length=3, response_length=4, tensor_model_parallel_size=2),
        num_replicas=2,
    )


def _payload(tokens):
    return {
        "choices": [
            {
                "prompt_token_ids": list(tokens),
                "prompt_logprobs": [None]
                + [{str(token): {"logprob": -float(i), "rank": 470}} for i, token in enumerate(tokens[1:], 1)],
            }
        ]
    }


def test_supplied_low_rank_tokens_use_managed_next_token_alignment():
    result = parse_prompt_logprobs(_payload([20, 100, 200]), [20, 100, 200])
    assert result == {"prompt_ids": [[100], [200], [0]], "prompt_logprobs": [[-1.0], [-2.0], [0.0]]}


def test_http_scores_align_with_response_tokens_through_actual_loss_padding_helper():
    tokens = [10, 11, 12, 13, 14]
    result = parse_prompt_logprobs(_payload(tokens), tokens)
    data = {
        "prompts": torch.tensor([[10, 11]]),
        "responses": torch.tensor([[12, 13, 14]]),
        "attention_mask": torch.ones(1, 5, dtype=torch.long),
    }
    logprobs = no_padding_2_padding(torch.tensor(result["prompt_logprobs"]), data).squeeze(-1)
    ids = no_padding_2_padding(torch.tensor(result["prompt_ids"]), data).squeeze(-1)
    torch.testing.assert_close(ids, data["responses"])
    torch.testing.assert_close(logprobs, torch.tensor([[-2.0, -3.0, -4.0]]))


@pytest.mark.parametrize("failure", ["changed_ids", "missing_ids", "short_scores", "missing_token", "nan", "positive"])
def test_malformed_teacher_targets_fail_closed(failure):
    payload = _payload([20, 100])
    choice = payload["choices"][0]
    if failure == "changed_ids":
        choice["prompt_token_ids"] = [20, 101]
    elif failure == "missing_ids":
        del choice["prompt_token_ids"]
    elif failure == "short_scores":
        choice["prompt_logprobs"].pop()
    elif failure == "missing_token":
        choice["prompt_logprobs"][1] = {}
    else:
        choice["prompt_logprobs"][1]["100"]["logprob"] = float("nan") if failure == "nan" else 0.1
    with pytest.raises(ValueError):
        parse_prompt_logprobs(payload, [20, 100])


def test_external_teacher_needs_no_gpu_pool_or_model_path():
    config = DistillationConfig(
        enabled=True,
        teacher_models={"teacher_model": _teacher()},
        distillation_loss=DistillationLossConfig(loss_mode="k1"),
    )
    assert not config.has_managed_teachers
    teacher = config.teacher_models["default"]
    assert teacher.world_size == 0
    assert teacher.inference.prompt_length == 7
    assert teacher.inference.response_length == 1


def test_managed_teacher_replica_calculation_is_preserved():
    config = DistillationConfig(
        enabled=True,
        n_gpus_per_node=8,
        nnodes=1,
        teacher_models={"teacher_model": _teacher(backend="managed")},
    )
    assert config.has_managed_teachers
    assert config.teacher_models["default"].num_replicas == 4
    assert config.teacher_models["default"].world_size == 8


def test_mixed_teacher_pool_accounts_only_for_managed_gpus():
    config = DistillationConfig(
        enabled=True,
        n_gpus_per_node=4,
        nnodes=1,
        teacher_models={
            "teacher_model": _teacher(),
            "local": _teacher(backend="managed", key="local"),
            "remote": _teacher(key="remote"),
        },
    )
    assert config.has_managed_teachers
    assert sum(teacher.world_size for teacher in config.teacher_models.values()) == 4
    assert set(config.teacher_models) == {"local", "remote"}


def test_external_config_reserves_one_scoring_token():
    teacher = _teacher()
    teacher.inference.prompt_length = 12
    teacher.inference.response_length = 4
    with pytest.raises(ValueError, match="one generated token"):
        DistillationConfig(enabled=True, teacher_models={"teacher_model": teacher})


def test_external_topk_is_rejected_without_affecting_managed_topk():
    with pytest.raises(NotImplementedError, match="sampled-token"):
        DistillationConfig(
            enabled=True,
            teacher_models={"teacher_model": _teacher()},
            distillation_loss=DistillationLossConfig(loss_mode="forward_kl_topk", use_policy_gradient=False),
        )


def test_http_exact_ids_retries_and_shared_concurrency():
    async def run():
        state = {"calls": 0, "active": 0, "peak": 0}
        received = []

        async def models(request):
            return web.json_response({"data": [{"id": "teacher", "max_model_len": 16}]})

        async def completions(request):
            payload = await request.json()
            received.append(copy.deepcopy(payload))
            state["calls"] += 1
            if state["calls"] == 1:
                return web.Response(status=503)
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
            try:
                await asyncio.sleep(0.01)
                return web.json_response(_payload(payload["prompt"][0]))
            finally:
                state["active"] -= 1

        app = web.Application()
        app.router.add_get("/v1/models", models)
        app.router.add_post("/v1/completions", completions)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        scorer = ExternalTeacherScorer(
            ExternalTeacherConfig(
                base_url=f"http://127.0.0.1:{port}/v1",
                model_name="teacher",
                max_model_len=16,
                max_concurrent_requests=2,
                retry_backoff_s=0,
            ),
            phase_timing=True,
        )
        try:
            assert (await scorer.validate_connection())["model_name"] == "teacher"
            results = await asyncio.gather(*(scorer.score([20, 100, 200], {"prompt_logprobs": 0}) for _ in range(5)))
            assert len(results) == 5
            assert state["calls"] == 6
            assert state["peak"] == 2
            assert all(result["prompt_logprobs"] == [[-1.0], [-2.0], [0.0]] for result in results)
            assert all(payload["prompt"] == [[20, 100, 200]] for payload in received)
            assert all(payload["add_special_tokens"] is False and payload["return_token_ids"] for payload in received)
            assert all(payload["max_tokens"] == 1 and payload["prompt_logprobs"] == 0 for payload in received)
            metrics = await scorer.pop_opd_phase_metrics()
            # The transient retry is part of one HTTP scoring phase. The
            # startup connection check does not contaminate rollout counters.
            for phase in ("teacher_queue", "teacher_http", "teacher_score"):
                assert metrics[phase]["count"] == 5
                assert metrics[phase]["completed_count"] == 5
                assert metrics[phase]["failed_count"] == metrics[phase]["cancelled_count"] == 0
        finally:
            await scorer.close()
            await runner.cleanup()

    asyncio.run(run())


def test_exhausted_retries_produce_no_targets():
    async def run():
        scorer = ExternalTeacherScorer(_external(max_retries=0, request_timeout_s=0.1))
        try:
            with pytest.raises(RuntimeError, match="no targets produced"):
                await scorer.score([20, 100], {"prompt_logprobs": 0})
        finally:
            await scorer.close()

    asyncio.run(run())


def test_server_context_must_cover_full_student_budget_and_scoring_reserve():
    async def run():
        scorer = ExternalTeacherScorer(
            ExternalTeacherConfig(base_url="http://127.0.0.1:1", model_name="teacher", max_model_len=262145)
        )
        scorer._request = AsyncMock(return_value={"data": [{"id": "teacher", "max_model_len": 262144}]})
        with pytest.raises(ValueError, match="Increase the actual teacher serving limit"):
            await scorer.validate_connection()
        scorer._request.return_value["data"][0]["max_model_len"] = 262145
        assert (await scorer.validate_connection())["max_model_len"] == 262145

    asyncio.run(run())


def test_unsupported_audio_ids_and_context_are_rejected_before_http():
    async def run():
        scorer = ExternalTeacherScorer(_external(unsupported_token_ids=[248070]))
        with pytest.raises(ValueError, match="unsupported"):
            await scorer.score([20, 248070], {})
        with pytest.raises(ValueError, match="no room"):
            await scorer.score([20] * 16, {})
        assert scorer._session is None

    asyncio.run(run())


def test_client_returns_existing_token_output_and_marks_scoring_errors_fatal():
    async def run():
        async def score(tokens, params):
            return parse_prompt_logprobs(_payload(tokens), tokens)

        client = ExternalTeacherClient(config=None, scoring_handle=SimpleNamespace(score=SimpleNamespace(remote=score)))
        output = await client.generate("test", prompt_ids=[20, 100], sampling_params={})
        assert output.token_ids == []
        assert output.extra_fields["prompt_logprobs"] == [[-1.0], [0.0]]

        async def fail(*_args):
            raise ValueError("bad scoring alignment")

        client._scoring_handle.score.remote = fail
        with pytest.raises(TeacherScoringError, match="bad scoring alignment"):
            await client.generate("test", prompt_ids=[20, 100], sampling_params={})
        with pytest.raises(TeacherScoringError, match="text-only"):
            await client.generate("test", prompt_ids=[20, 100], sampling_params={}, audio_data=[b"audio"])

    asyncio.run(run())


def test_phase_timing_records_cancelled_queue_without_starting_http():
    async def run():
        scorer = ExternalTeacherScorer(_external(max_concurrent_requests=1), phase_timing=True)
        http_entered = asyncio.Event()
        release_http = asyncio.Event()

        async def request(*_args):
            http_entered.set()
            await release_http.wait()
            return _payload([20, 100])

        scorer._request = AsyncMock(side_effect=request)
        active = asyncio.create_task(scorer.score([20, 100], {}))
        await http_entered.wait()
        queued = asyncio.create_task(scorer.score([20, 100], {}))
        await asyncio.sleep(0)
        queued.cancel()
        with pytest.raises(asyncio.CancelledError):
            await queued
        release_http.set()
        await active
        assert scorer._request.await_count == 1
        metrics = await scorer.pop_opd_phase_metrics()
        assert metrics["teacher_score"]["count"] == 2
        assert metrics["teacher_score"]["completed_count"] == 1
        assert metrics["teacher_score"]["cancelled_count"] == 1
        assert metrics["teacher_queue"]["count"] == 2
        assert metrics["teacher_queue"]["cancelled_count"] == 1
        assert metrics["teacher_http"]["count"] == 1
        assert metrics["teacher_http"]["completed_count"] == 1
        assert metrics["teacher_queue"]["seconds_sum"] > 0
        assert await scorer.pop_opd_phase_metrics() == {}

    asyncio.run(run())


def test_cancelled_active_score_releases_capacity_and_actor_is_reusable():
    async def run():
        scorer = ExternalTeacherScorer(_external(max_concurrent_requests=1), phase_timing=True)
        http_entered = asyncio.Event()
        active_http = 0

        async def request(*_args):
            nonlocal active_http
            active_http += 1
            try:
                http_entered.set()
                await asyncio.Event().wait()
            finally:
                active_http -= 1

        scorer._request = AsyncMock(side_effect=request)
        active = asyncio.create_task(scorer.score([20, 100], {}))
        await http_entered.wait()
        active.cancel()
        with pytest.raises(asyncio.CancelledError):
            await active
        assert active_http == 0
        scorer._request = AsyncMock(return_value=_payload([20, 100]))
        result = await asyncio.wait_for(scorer.score([20, 100], {}), timeout=1)
        assert result["prompt_ids"] == [[100], [0]]
        metrics = await scorer.pop_opd_phase_metrics()
        for phase in ("teacher_http", "teacher_score"):
            assert metrics[phase]["count"] == 2
            assert metrics[phase]["cancelled_count"] == 1
            assert metrics[phase]["completed_count"] == 1
            assert metrics[phase]["failed_count"] == 0

    asyncio.run(run())


def test_real_http_cancellation_closes_response_without_retry_and_next_score_succeeds():
    async def run():
        entered, disconnected = asyncio.Event(), asyncio.Event()
        calls = 0

        async def complete(request):
            nonlocal calls
            data = await request.json()
            calls += 1
            if calls != 1:
                return web.json_response(_payload(data["prompt"][0]))
            response = web.StreamResponse(headers={"Content-Type": "application/json"})
            await response.prepare(request)
            await response.write(b"{")
            entered.set()
            for _ in range(200):
                if request.transport is None or request.transport.is_closing():
                    disconnected.set()
                    break
                await asyncio.sleep(0.01)
            return response

        app = web.Application()
        app.router.add_post("/v1/completions", complete)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        config = ExternalTeacherConfig(
            base_url=f"http://127.0.0.1:{port}",
            model_name="teacher",
            max_model_len=16,
            max_concurrent_requests=1,
            max_retries=2,
            retry_backoff_s=0,
        )
        scorer = ExternalTeacherScorer(config, phase_timing=True)
        try:
            active = asyncio.create_task(scorer.score([20, 100], {}))
            await asyncio.wait_for(entered.wait(), 3)
            active.cancel()
            with pytest.raises(asyncio.CancelledError):
                await active
            await asyncio.wait_for(disconnected.wait(), 3)
            result = await asyncio.wait_for(scorer.score([20, 100], {}), 3)
            assert result["prompt_ids"] == [[100], [0]]
            assert calls == 2  # One cancelled call, one new call; cancellation never retries.
            metrics = await scorer.pop_opd_phase_metrics()
            for phase in ("teacher_http", "teacher_score"):
                assert metrics[phase]["count"] == 2
                assert metrics[phase]["cancelled_count"] == 1
                assert metrics[phase]["completed_count"] == 1
        finally:
            await scorer.close()
            await runner.cleanup()

    asyncio.run(run())


def test_phase_timing_failure_preserves_error_and_default_is_disabled():
    async def run():
        for enabled in (False, True):
            scorer = ExternalTeacherScorer(_external(), phase_timing=enabled)
            scorer._request = AsyncMock(side_effect=RuntimeError("teacher unavailable"))
            with pytest.raises(RuntimeError, match="teacher unavailable"):
                await scorer.score([20, 100], {})
            metrics = await scorer.pop_opd_phase_metrics()
            if not enabled:
                assert metrics == {}
                continue
            assert metrics["teacher_score"]["failed_count"] == 1
            assert metrics["teacher_http"]["failed_count"] == 1
            assert metrics["teacher_queue"]["completed_count"] == 1
            for phase in metrics.values():
                assert phase["count"] == sum(
                    phase[f"{outcome}_count"] for outcome in ("completed", "cancelled", "failed")
                )
                assert phase["seconds_sum"] >= phase["seconds_max"] >= 0

    asyncio.run(run())


def test_client_propagates_rpc_cleanup_failure_without_reward_fallback(monkeypatch):
    async def run():
        async def score(*_args):
            return {}

        failure = RPCCancellationError("teacher cleanup timed out")
        await_ref = AsyncMock(side_effect=failure)
        monkeypatch.setattr("verl.experimental.teacher_loop.external_teacher.await_cancellable_ray_ref", await_ref)
        client = ExternalTeacherClient(config=None, scoring_handle=SimpleNamespace(score=SimpleNamespace(remote=score)))
        with pytest.raises(RPCCancellationError) as exc_info:
            await client.generate("test", prompt_ids=[20, 100], sampling_params={})
        assert exc_info.value is failure
        # The mocked awaiter deliberately never consumes the coroutine.
        await_ref.call_args.args[0].close()

    asyncio.run(run())


def test_client_cutoff_cancels_actor_score_and_settles_metrics_before_return(monkeypatch):
    async def run():
        scorer = ExternalTeacherScorer(_external(max_concurrent_requests=1), phase_timing=True)
        entered = asyncio.Event()
        cleaned = asyncio.Event()

        async def request(*_args):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaned.set()

        scorer._request = AsyncMock(side_effect=request)
        refs = []

        class Ref:
            def __init__(self, tokens, params):
                self.task = asyncio.create_task(scorer.score(tokens, params))

            def __await__(self):
                return self.task.__await__()

        def score(tokens, params):
            ref = Ref(tokens, params)
            refs.append(ref)
            return ref

        def cancel(ref, *, force):
            assert not force
            ref.task.cancel()

        monkeypatch.setattr("verl.utils.async_rpc.ray.cancel", cancel)
        client = ExternalTeacherClient(config=None, scoring_handle=SimpleNamespace(score=SimpleNamespace(remote=score)))
        with rollout_request_scope(cancel_on_cutoff=True, cleanup_timeout_s=1):
            task = asyncio.create_task(client.generate("test", prompt_ids=[20, 100], sampling_params={}))
            await asyncio.wait_for(entered.wait(), timeout=1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert cleaned.is_set()
        assert refs[0].task.done()
        metrics = await scorer.pop_opd_phase_metrics()
        assert metrics["teacher_http"]["cancelled_count"] == 1
        assert metrics["teacher_score"]["cancelled_count"] == 1
        scorer._request = AsyncMock(return_value=_payload([20, 100]))
        output = await asyncio.wait_for(client.generate("next", prompt_ids=[20, 100], sampling_params={}), timeout=1)
        assert output.extra_fields["prompt_ids"] == [[100], [0]]

    asyncio.run(run())

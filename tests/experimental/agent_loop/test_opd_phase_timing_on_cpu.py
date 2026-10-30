"""Phase accounting includes failed and cancelled work without validation drift."""

import asyncio
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from omegaconf import OmegaConf

from verl.experimental.agent_loop import agent_loop as module
from verl.experimental.agent_loop.agent_loop import AgentLoopOutput, AgentLoopWorker
from verl.utils.async_rpc import PhaseMetrics, RPCCancellationError, await_cancellable_ray_ref, rollout_request_scope


def _worker():
    worker = object.__new__(AgentLoopWorker)
    worker.config = OmegaConf.create(
        {
            "trainer": {"use_v1": True},
            "data": {},
            "distillation": {
                "enabled": True,
                "phase_timing": True,
                "cancel_pending_on_cutoff": True,
                "cutoff_cleanup_timeout_s": 1,
                "distillation_loss": {"use_task_rewards": False},
            },
        }
    )
    worker._opd_phase_metrics = PhaseMetrics()
    worker.llm_client = worker.tokenizer = worker.processor = worker.dataset_cls = None
    worker.tools = []
    return worker


@pytest.mark.parametrize("outcome", ["completed", "cancelled", "failed"])
def test_agent_run_phase_includes_all_outcomes(monkeypatch, outcome):
    worker = _worker()
    output = AgentLoopOutput(
        prompt_ids=[1], response_ids=[2], response_mask=[1], num_turns=1, metrics=module.AgentLoopMetrics()
    )

    async def run(*_args, **_kwargs):
        if outcome == "cancelled":
            raise asyncio.CancelledError
        if outcome == "failed":
            raise ValueError("agent failed")
        return output

    monkeypatch.setitem(module._agent_loop_registry, "fake", {})
    monkeypatch.setattr(module.hydra.utils, "instantiate", lambda **_kwargs: SimpleNamespace(run=run))
    monkeypatch.setattr(module, "rollout_trace_attr", lambda **_kwargs: nullcontext())
    worker._agent_loop_postprocess = AsyncMock(return_value=output)

    async def exercise():
        operation = worker._run_agent_loop(
            {}, {"step": 1, "sample_index": 0, "rollout_n": 1, "validate": False}, agent_name="fake"
        )
        if outcome == "cancelled":
            with pytest.raises(asyncio.CancelledError):
                await operation
        elif outcome == "failed":
            with pytest.raises(ValueError):
                await operation
        else:
            assert await operation is output
        row = (await worker.pop_opd_phase_metrics())["agent_loop"]
        assert row["count"] == row[f"{outcome}_count"] == 1
        assert row["seconds_sum"] >= 0
        assert await worker.pop_opd_phase_metrics() == {}

    asyncio.run(exercise())


def test_swallowed_cleanup_failure_stops_before_teacher_postprocessing(monkeypatch):
    worker = _worker()
    worker.config.distillation.cutoff_cleanup_timeout_s = 0.01

    async def run(*_args, **_kwargs):
        unsettled = asyncio.get_running_loop().create_future()
        request = asyncio.create_task(await_cancellable_ray_ref(unsettled))
        await asyncio.sleep(0)
        request.cancel()
        try:
            await request
        except RPCCancellationError:
            unsettled.set_result(None)
            return "source swallowed fatal cleanup error"

    monkeypatch.setitem(module._agent_loop_registry, "fake", {})
    monkeypatch.setattr(module.hydra.utils, "instantiate", lambda **_kwargs: SimpleNamespace(run=run))
    monkeypatch.setattr(module, "rollout_trace_attr", lambda **_kwargs: nullcontext())
    monkeypatch.setattr("verl.utils.async_rpc.ray.cancel", lambda *_a, **_k: None)
    worker._agent_loop_postprocess = AsyncMock()

    async def exercise():
        with pytest.raises(RPCCancellationError, match="did not settle"):
            await worker._run_agent_loop(
                {}, {"step": 1, "sample_index": 0, "rollout_n": 1, "validate": False}, agent_name="fake"
            )
        assert (await worker.pop_opd_phase_metrics())["agent_loop"]["failed_count"] == 1

    asyncio.run(exercise())
    worker._agent_loop_postprocess.assert_not_awaited()


@pytest.mark.parametrize("outcome", ["completed", "cancelled", "failed"])
def test_teacher_target_phase_includes_all_outcomes(outcome):
    worker = _worker()
    worker.distillation_enabled = True
    worker.teacher_key = "teacher"
    failure = asyncio.CancelledError() if outcome == "cancelled" else ValueError("bad target")
    worker.teacher_server_manager = SimpleNamespace(
        compute_teacher_logprobs_single=AsyncMock(
            side_effect=failure if outcome != "completed" else None, return_value=([1], [-1.0])
        )
    )
    output = AgentLoopOutput(
        prompt_ids=[1], response_ids=[2], response_mask=[1], num_turns=1, metrics=module.AgentLoopMetrics()
    )

    async def exercise():
        with rollout_request_scope(False, 1, worker._opd_phase_metrics.record):
            operation = worker._compute_teacher_logprobs(output, [1], [2], validate=False)
            if outcome == "completed":
                await operation
            else:
                with pytest.raises(type(failure)):
                    await operation
        row = (await worker.pop_opd_phase_metrics())["teacher_target"]
        assert row["count"] == row[f"{outcome}_count"] == 1
        if outcome == "completed":
            assert output.extra_fields["teacher_ids"] == [1]
        else:
            assert "teacher_ids" not in output.extra_fields

    asyncio.run(exercise())


def test_validation_does_not_score_teacher_or_record_phase():
    worker = _worker()
    worker.distillation_enabled = True
    worker.teacher_server_manager = SimpleNamespace(compute_teacher_logprobs_single=AsyncMock())
    output = AgentLoopOutput(
        prompt_ids=[1], response_ids=[2], response_mask=[1], num_turns=1, metrics=module.AgentLoopMetrics()
    )

    async def exercise():
        await worker._compute_teacher_logprobs(output, [1], [2], validate=True)
        assert await worker.pop_opd_phase_metrics() == {}

    asyncio.run(exercise())
    worker.teacher_server_manager.compute_teacher_logprobs_single.assert_not_awaited()

"""Network-free contracts for repeat-only deferred reward scheduling."""
import asyncio
import copy
import logging
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from recipe.swe_agent import repetition_reward as repeat
from verl.trainer.ppo.v1.repeat_judge_queue import (
    RepeatJudgeCoordinator, deferred_enabled, finalize_output, validate_deferred_config,
)


def make_job(config, base=1., validation=False):
    return dict(config=config, metadata=dict(repeat_reward_outcome_bucket="submitted_success",
        repeat_reward_judge_called=0, repeat_reward_stage=0), validation=validation, valid=True,
        raw_score=1., existing_score=base, transcript="public task", events=[], actor_segments=[])


@pytest.mark.parametrize("validation", [False, True])
@pytest.mark.parametrize("base", [1., 0., -.1])
@pytest.mark.parametrize("verdict", ["clean", "uncertain", "repetitive"])
def test_deferred_and_inline_finalization_identical(monkeypatch, validation, base, verdict):
    async def run():
        config = repeat.RepeatConfig(mode="apply", reward_scope="all_valid", deferred_enabled=True)
        queue = RepeatJudgeCoordinator(config)
        class Client:
            async def judge_consensus(self, *args, **kwargs):
                return {"verdict": verdict, "reason": "fixture", "queue_seconds": 2., "service_seconds": 3.}
        queue.client = Client()
        job = make_job(config, base, validation)
        inline_job = copy.deepcopy(job)
        inline_job["metadata"].update(repeat_reward_judge_called=1, repeat_reward_stage=2)
        inline_score, inline_meta = await repeat.finish_repeat_reward(inline_job, await queue.client.judge_consensus())
        identity, score, meta = await queue.submit("sample", job)
        assert identity == "sample" and score == inline_score
        for key, value in inline_meta.items():
            assert meta[key] == value
        assert meta["repeat_reward_pending"] == 0
        assert meta["repeat_reward_deferred_request_queue_seconds"] == 2
        assert not queue.pending and not queue.running
    monkeypatch.setattr(repeat, "save_audit", lambda *args, **kwargs: {})
    asyncio.run(run())


def test_bounded_admission_duplicates_and_failure_release(monkeypatch):
    async def run():
        config = repeat.RepeatConfig(max_pending=1)
        queue = RepeatJudgeCoordinator(config)
        entered, release = asyncio.Event(), asyncio.Event()
        class Client:
            async def judge_consensus(self, *args, **kwargs):
                entered.set()
                await release.wait()
                raise RuntimeError("coordinator processing failed")
        queue.client = Client()
        first = asyncio.create_task(queue.submit("first", make_job(config)))
        await entered.wait()
        with pytest.raises(RuntimeError, match="Duplicate"):
            await queue.submit("first", make_job(config))
        result = await asyncio.wait_for(queue.submit("second", make_job(config)), timeout=2)
        assert result[1] == 1. and result[2]["repeat_reward_queue_full"] == 1
        assert queue.snapshot()["pending"] == 1
        assert queue.snapshot()["dropped"] == 1
        release.set()
        with pytest.raises(RuntimeError, match="processing failed"):
            await first
        assert queue.snapshot()["capacity"] == 1 and queue.snapshot()["failed"] == 1
        with pytest.raises(RuntimeError, match="processing failed"):
            await queue.submit("third", make_job(config))
        assert queue.snapshot()["failed"] == 2
    monkeypatch.setattr(repeat, "save_audit", lambda *args, **kwargs: {})
    asyncio.run(run())


@pytest.mark.parametrize("early_exit", [False, True])
def test_one_request_semaphore_across_independent_jobs(monkeypatch, early_exit):
    async def run():
        config = repeat.RepeatConfig(concurrency=2, max_pending=10, vote_count=3, vote_quorum=2,
                                     early_exit_enabled=early_exit)
        queue = RepeatJudgeCoordinator(config)
        queue.client = repeat.JudgeClient(config)
        active = peak = 0
        async def fake_request(*args, **kwargs):
            nonlocal active, peak
            async with queue.client.semaphore:
                active += 1
                peak = max(peak, active)
                await asyncio.sleep(.001)
                active -= 1
            return {"verdict": "clean", "reason": "fixture"}
        monkeypatch.setattr(queue.client, "judge", fake_request)
        async def job(i):
            key = str(i)
            return await queue.submit(key, make_job(config))
        results = await asyncio.gather(*(job(i) for i in range(8)))
        assert len(results) == 8 and peak == 2
        assert queue.snapshot()["completed"] == 8 and queue.snapshot()["pending"] == 0
        assert all(result[1] == 1. for result in results)
        assert all(result[2]["repeat_reward_judge_vote_count"] == (2 if early_exit else 3) for result in results)
        assert all(result[2]["repeat_reward_judge_votes_skipped"] == int(early_exit) for result in results)
    monkeypatch.setattr(repeat, "save_audit", lambda *args, **kwargs: {})
    asyncio.run(run())


@pytest.mark.parametrize("validation", [False, True])
@pytest.mark.parametrize("base", [1., 0., -.1])
def test_queue_full_preserves_base_and_audits_uncertain(monkeypatch, base, validation):
    async def run():
        config = repeat.RepeatConfig(mode="apply", reward_scope="all_valid", max_pending=1)
        queue = RepeatJudgeCoordinator(config)
        entered, release = asyncio.Event(), asyncio.Event()
        class Client:
            async def judge_consensus(self, *args, **kwargs):
                entered.set()
                await release.wait()
                return {"verdict": "repetitive", "reason": "fixture"}
        queue.client = Client()
        first = asyncio.create_task(queue.submit("first", make_job(config)))
        await entered.wait()
        job = make_job(config, base, validation)
        before = copy.deepcopy(job)
        _, score, metadata = await asyncio.wait_for(queue.submit("full", job), timeout=2)
        assert job == before
        assert score == base and metadata["repeat_reward_queue_full"] == 1
        assert metadata["repeat_reward_queue_accepted"] == 0
        assert metadata["repeat_reward_pending"] == 0 and metadata["repeat_reward_judge_called"] == 0
        assert metadata["repeat_reward_stage"] == 0 and metadata["repeat_reward_uncertain"] == 1
        assert metadata["repeat_reward_applied"] == metadata["repeat_reward_would_apply"] == 0
        assert "repeat_queue_full" in metadata["repeat_reward_verdict_json"]
        release.set()
        await first
        assert queue.snapshot()["pending"] == 0 and queue.snapshot()["completed"] == 1
    monkeypatch.setattr(repeat, "save_audit", lambda *args, **kwargs: {})
    asyncio.run(run())


def test_job_queue_cancel_and_failure_do_not_leak_capacity(monkeypatch):
    async def run():
        config = repeat.RepeatConfig(max_pending=2, concurrency=1)
        queue = RepeatJudgeCoordinator(config)
        entered, release = asyncio.Event(), asyncio.Event()
        calls = 0
        class Client:
            async def judge_consensus(self, *args, **kwargs):
                nonlocal calls
                calls += 1
                entered.set()
                await release.wait()
                return {"verdict": "clean", "reason": "fixture"}
        queue.client = Client()
        running = asyncio.create_task(queue.submit("running", make_job(config)))
        await entered.wait()
        queued = asyncio.create_task(queue.submit("queued", make_job(config)))
        await asyncio.sleep(0)
        snapshot = queue.snapshot()
        assert snapshot["active"] == 1 and snapshot["queued"] == 1 and snapshot["capacity"] == 0
        assert calls == 1
        queued.cancel()
        with pytest.raises(asyncio.CancelledError):
            await queued
        assert queue.snapshot()["capacity"] == 1 and queue.snapshot()["queued"] == 0
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
        assert queue.snapshot()["capacity"] == 2 and queue.snapshot()["cancelled"] == 2
        release.set()
        await queue.submit("replacement", make_job(config))
        assert calls == 2 and queue.snapshot()["completed"] == 1
    monkeypatch.setattr(repeat, "save_audit", lambda *args, **kwargs: {})
    asyncio.run(run())


def test_accepted_job_waits_internally_then_runs(monkeypatch):
    async def run():
        config = repeat.RepeatConfig(max_pending=2, concurrency=1)
        queue = RepeatJudgeCoordinator(config)
        entered, release = asyncio.Event(), asyncio.Event()
        class Client:
            async def judge_consensus(self, *args, **kwargs):
                entered.set()
                await release.wait()
                return {"verdict": "clean", "reason": "fixture"}
        queue.client = Client()
        first = asyncio.create_task(queue.submit("first", make_job(config)))
        await entered.wait()
        second = asyncio.create_task(queue.submit("second", make_job(config)))
        await asyncio.sleep(0)
        assert not second.done() and queue.snapshot()["queued"] == 1
        release.set()
        results = await asyncio.gather(first, second)
        assert [result[0] for result in results] == ["first", "second"]
        assert all(result[2]["repeat_reward_queue_accepted"] == 1 for result in results)
        assert results[1][2]["repeat_reward_deferred_job_queue_seconds"] > 0
        assert queue.snapshot()["capacity"] == 2 and queue.snapshot()["completed"] == 2
    monkeypatch.setattr(repeat, "save_audit", lambda *args, **kwargs: {})
    asyncio.run(run())


def test_score_copies_raw_reward_and_double_finalization():
    output = SimpleNamespace(reward_score=1., extra_fields={
        "reward_extra_info": {"score": 1., "raw_score": 1., "noninfra_shaped_score": 1.},
        "swe_hidden_reward": {"ok": True, "score": 1., "raw_score": 1., "shaped_score": 1.,
                              "metadata": {"noninfra_shaped_score": 1.}},
    })
    finalize_output(output, .1, {"repeat_reward_pending": 0})
    assert output.reward_score == .1
    assert output.extra_fields["reward_extra_info"]["noninfra_shaped_score"] == .1
    assert output.extra_fields["swe_hidden_reward"]["metadata"]["noninfra_shaped_score"] == .1
    assert output.extra_fields["swe_hidden_reward"]["score"] == 1.
    with pytest.raises(RuntimeError, match="twice"):
        finalize_output(output, .01, {})


def test_feature_default_config_validation_and_forwarding(monkeypatch):
    from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
    monkeypatch.delenv("SWE_AGENT_REPEAT_JUDGE_DEFERRED_ENABLED", raising=False)
    assert not deferred_enabled() and not repeat.RepeatConfig.from_env().deferred_enabled
    config = OmegaConf.create({"trainer": {"use_v1": True, "v1": {"trainer_mode": "sync"}}, "actor_rollout_ref": {"rollout": {}}})
    monkeypatch.setenv("SWE_AGENT_REPEAT_JUDGE_DEFERRED_ENABLED", "1")
    assert repeat.RepeatConfig.from_env().deferred_enabled
    assert get_ppo_ray_runtime_env()["env_vars"]["SWE_AGENT_REPEAT_JUDGE_DEFERRED_ENABLED"] == "1"
    monkeypatch.setenv("SWE_AGENT_REPEAT_JUDGE_MAX_PENDING", "1024")
    assert repeat.RepeatConfig.from_env().max_pending == 1024
    assert get_ppo_ray_runtime_env()["env_vars"]["SWE_AGENT_REPEAT_JUDGE_MAX_PENDING"] == "1024"
    validate_deferred_config(config)
    config.trainer.v1.trainer_mode = "async"
    with pytest.raises(ValueError, match="sync"):
        validate_deferred_config(config)
    config.trainer.use_v1 = False
    with pytest.raises(ValueError):
        validate_deferred_config(config)
    monkeypatch.setenv("SWE_AGENT_REPEAT_JUDGE_MAX_PENDING", "0")
    with pytest.raises(ValueError):
        repeat.RepeatConfig.from_env()


def worker_harness(monkeypatch, n=1):
    """Exercise production prompt/postprocessing methods without model or Ray startup."""
    import torch
    from verl.trainer.ppo.v1 import agent_loop_tq as module
    cls = module.AgentLoopWorkerTQ.__ray_metadata__.modified_class
    worker = object.__new__(cls)
    worker.repeat_coordinator = None
    worker.repeat_finalizers = {}
    worker.repeat_job_refs = {}
    worker.dapo_cutoff_uids = set()
    worker.dapo_cutoff_enabled = False
    worker.dapo_scoring_sessions = {}
    worker.dapo_cleanup_timeouts = {}
    worker.dapo_publications = {}
    worker.repeat_generation_sessions = {}
    worker.repeat_debug = module._repeat_debug_enabled()
    worker.repeat_debug_slow_seconds = module._repeat_debug_slow_seconds()
    worker.repeat_debug_prompts = {}
    worker.prompt_tasks = {"train": {}, "val": {}}
    worker.cutoff_session_states = {}
    worker.config = OmegaConf.create({"actor_rollout_ref": {"rollout": {"n": n, "val_kwargs": {"n": n}}}})
    store, publications = {}, []
    async def put(key, partition_id, tag):
        store.setdefault(partition_id, {}).setdefault(key, {}).update(tag)
    async def batch_put(keys, fields, tags, partition_id):
        for key, tag in zip(keys, tags):
            store.setdefault(partition_id, {})[key] = tag
        publications.append((keys, fields, tags))
    def sync_batch_put(keys, partition_id, tags):
        for key, tag in zip(keys, tags):
            store[partition_id][key].update(tag)
    monkeypatch.setattr(module.tq, "async_kv_put", put)
    monkeypatch.setattr(module.tq, "async_kv_batch_put", batch_put)
    monkeypatch.setattr(module.tq, "kv_batch_put", sync_batch_put)
    monkeypatch.setattr(module.tq, "kv_list", lambda partition_id=None: store)
    worker._ready_trajectory_keys = lambda partition_id, keys: keys
    async def noop(*args, **kwargs):
        pass
    worker._compute_score = noop
    worker._compute_teacher_logprobs = noop
    worker._compute_multi_modal_inputs = lambda *args: None
    worker._compute_position_ids = lambda input_ids, *args: torch.arange(input_ids.shape[-1]).unsqueeze(0)
    return worker, store, publications


class RemoteMethod:
    def __init__(self, call):
        self.call = call
    def remote(self, *args):
        return asyncio.create_task(self.call(*args))


def fake_output(config):
    from verl.experimental.agent_loop import AgentLoopOutput
    job = make_job(config)
    return AgentLoopOutput(prompt_ids=[1], response_ids=[2, 3], response_mask=[1, 1], reward_score=1., metrics={},
        extra_fields={"_swe_repeat_job": job,
            "reward_extra_info": {"score": 1., "raw_score": 1.},
            "swe_hidden_reward": {"ok": True, "score": 1., "shaped_score": 1., "metadata": {}}})


def test_worker_returns_after_handoff_and_publishes_only_final_reward(monkeypatch):
    async def run():
        worker, store, publications = worker_harness(monkeypatch)
        config = repeat.RepeatConfig(deferred_enabled=True)
        complete = asyncio.Event()
        async def submit(job_id, job):
            await complete.wait()
            return job_id, .1, {"repeat_reward_pending": 0}
        worker.repeat_coordinator = SimpleNamespace(submit=RemoteMethod(submit))
        worker.repeat_finalizers[("train", "u")] = []
        worker.repeat_generation_sessions[("train", "u")] = (1, set())
        await worker._agent_loop_postprocess(fake_output(config), False, uid="u", session_id=0, global_steps=0)
        assert store["train"]["u"]["generation_complete"]
        assert not publications
        complete.set()
        await worker._join_repeat("train", "u")
        fields = publications[0][1]
        assert fields["rm_scores"].sum().item() == pytest.approx(.1)
        assert fields["extra_fields"][0]["reward_extra_info"]["score"] == .1
    asyncio.run(run())


def test_cutoff_cancels_unfinished_generation_but_drains_retained_judgment(monkeypatch):
    async def run():
        worker, store, publications = worker_harness(monkeypatch, n=2)
        config = repeat.RepeatConfig(deferred_enabled=True)
        judging = asyncio.Event()
        finish_judge = asyncio.Event()
        long_generation_started = asyncio.Event()
        generation_cancelled = asyncio.Event()
        async def submit(job_id, job):
            judging.set()
            await finish_judge.wait()
            return job_id, .1, {"repeat_reward_pending": 0}
        worker.repeat_coordinator = SimpleNamespace(submit=RemoteMethod(submit))
        async def session(params, *, session_id, **kwargs):
            if session_id == 1:
                long_generation_started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    generation_cancelled.set()
            else:
                await worker._agent_loop_postprocess(fake_output(config), False,
                    uid="u", session_id=0, global_steps=0)
        worker._run_agent_loop = session
        prompt_task = asyncio.create_task(worker._run_prompt({"uid": "u"}, {}, {"validate": False}))
        worker.prompt_tasks["train"]["u"] = prompt_task
        await judging.wait()
        await long_generation_started.wait()
        cutoff = asyncio.create_task(worker.cancel_sequences(["u"]))
        await generation_cancelled.wait()
        await asyncio.sleep(0)
        assert not cutoff.done() and not publications
        finish_judge.set()
        assert await cutoff == 1
        assert store["train"]["u"]["status"] == "failure"
        assert store["train"]["u"]["completion_ratio_cutoff"]
        assert publications[0][1]["rm_scores"].sum().item() == pytest.approx(.1)
        assert all(t.done() for t in worker.repeat_finalizers.get(("train", "u"), []))
    asyncio.run(run())


@pytest.mark.parametrize("n", [1, 2])
def test_dapo_cutoff_cancels_generation_but_drains_deferred_rewards(monkeypatch, n):
    async def run():
        worker, store, publications = worker_harness(monkeypatch, n=n)
        worker.dapo_cutoff_enabled = True
        judging, judge_cancelled, generation_cleanup = asyncio.Event(), asyncio.Event(), asyncio.Event()
        finish_judge = asyncio.Event()
        release_generation = asyncio.Event()

        async def submit(job_id, job):
            judging.set()
            try:
                await finish_judge.wait()
                return job_id, .1, {"repeat_reward_pending": 0}
            except asyncio.CancelledError:
                judge_cancelled.set()
                raise

        worker.repeat_coordinator = SimpleNamespace(submit=RemoteMethod(submit))

        async def session(params, *, session_id, **kwargs):
            if session_id == 0:
                await worker._agent_loop_postprocess(fake_output(repeat.RepeatConfig(deferred_enabled=True)), False,
                                                    uid="u", session_id=0, global_steps=0)
            else:
                try:
                    await asyncio.Event().wait()
                finally:
                    await release_generation.wait()
                    generation_cleanup.set()

        worker._run_agent_loop = session
        prompt = asyncio.create_task(worker._run_prompt({"uid": "u"}, {}, {"validate": False}))
        worker.prompt_tasks["train"]["u"] = prompt
        await asyncio.wait_for(judging.wait(), timeout=2)
        if n == 1:
            assert store["train"]["u"]["generation_complete"]
        cutoff = asyncio.create_task(worker.cancel_sequences(["u"], dapo_cutoff=True))
        await asyncio.sleep(.01)
        if n == 2:
            assert not cutoff.done()
            release_generation.set()
        await asyncio.sleep(.01)
        assert not cutoff.done() and not judge_cancelled.is_set()
        finish_judge.set()
        assert await asyncio.wait_for(cutoff, timeout=2) == 1
        assert not judge_cancelled.is_set()
        assert n == 1 or generation_cleanup.is_set()
        assert store["train"]["u"]["dapo_cutoff"]
        assert not store["train"]["u"].get("completion_ratio_cutoff")
        await asyncio.sleep(.01)
        assert len(publications) == 1 and not worker.repeat_finalizers and not worker.repeat_job_refs
        assert not worker.dapo_cutoff_uids
        assert not worker.dapo_scoring_sessions and not worker.dapo_cleanup_timeouts

    asyncio.run(run())


def test_dapo_cutoff_retains_judge_even_if_finalizer_has_not_started(monkeypatch):
    async def run():
        worker, _, publications = worker_harness(monkeypatch)
        cancelled = asyncio.Event()
        finished = asyncio.Event()

        async def submit():
            try:
                await finished.wait()
                return "job", .1, {"repeat_reward_pending": 0}
            except asyncio.CancelledError:
                cancelled.set()
                raise

        ref = asyncio.create_task(submit())
        await asyncio.sleep(0)
        output = fake_output(repeat.RepeatConfig())
        output.extra_fields.pop("_swe_repeat_job")
        finalizer = asyncio.create_task(worker._finalize_repeat(
            ref, "job", output, False, {"uid": "u", "session_id": 0, "global_steps": 0}
        ))
        # Finalized outputs must skip the generation handoff and publish directly.
        worker.repeat_generation_sessions[("train", "u")] = (1, {0})
        worker.repeat_job_refs[("train", "u")] = [ref]
        worker.repeat_finalizers[("train", "u")] = [finalizer]
        join = asyncio.create_task(worker._join_repeat("train", "u"))
        await asyncio.sleep(0)
        assert not join.done() and not cancelled.is_set()
        finished.set()
        await asyncio.wait_for(join, timeout=2)
        assert not cancelled.is_set() and not finalizer.cancelled() and len(publications) == 1

    asyncio.run(run())


@pytest.mark.parametrize("deferred", [False, True])
@pytest.mark.parametrize("publication_timeout", [False, True])
def test_dapo_cutoff_waits_for_publication_already_sent_to_transfer_queue(monkeypatch, deferred, publication_timeout):
    from verl.trainer.ppo.v1 import agent_loop_tq as module
    from verl.utils.async_rpc import mark_scoring_started

    async def run():
        worker, store, publications = worker_harness(monkeypatch)
        worker.dapo_cutoff_enabled = True
        entered, release = asyncio.Event(), asyncio.Event()
        original_put = module.tq.async_kv_batch_put

        async def slow_put(**kwargs):
            entered.set()
            await release.wait()
            return await original_put(**kwargs)

        monkeypatch.setattr(module.tq, "async_kv_batch_put", slow_put)
        if deferred:
            async def submit(job_id, job):
                return job_id, .1, {"repeat_reward_pending": 0}
            worker.repeat_coordinator = SimpleNamespace(submit=RemoteMethod(submit))

        async def session(*args, **kwargs):
            mark_scoring_started()
            output = fake_output(repeat.RepeatConfig(deferred_enabled=True))
            if not deferred:
                output.extra_fields.pop("_swe_repeat_job")
            await worker._agent_loop_postprocess(output, False, uid="u", session_id=0, global_steps=0)

        worker._run_agent_loop = session
        prompt = asyncio.create_task(worker._run_prompt({"uid": "u"}, {}, {"validate": False}))
        worker.prompt_tasks["train"]["u"] = prompt
        await asyncio.wait_for(entered.wait(), timeout=2)
        cutoff = asyncio.create_task(worker.cancel_sequences(
            ["u"], dapo_cutoff=True, cleanup_timeout_seconds=.01 if publication_timeout else 300
        ))
        if publication_timeout:
            try:
                with pytest.raises(RuntimeError, match="DAPO .*cleanup"):
                    await asyncio.wait_for(cutoff, timeout=2)
                assert not publications
            finally:
                release.set()
                await asyncio.gather(prompt, return_exceptions=True)
                await asyncio.gather(*worker.dapo_publications.get(("train", "u"), []), return_exceptions=True)
            return
        await asyncio.sleep(.01)
        assert not cutoff.done() and not publications
        release.set()
        assert await asyncio.wait_for(cutoff, timeout=2) == 1
        assert len(publications) == 1 and store["train"]["u"]["dapo_cutoff"]
        assert not worker.dapo_publications

    asyncio.run(run())


@pytest.mark.parametrize("n", [1, 2])
@pytest.mark.parametrize("opd_enabled", [False, True])
def test_dapo_cutoff_keeps_started_hidden_rewards_and_bounds_only_generation_cleanup(monkeypatch, n, opd_enabled):
    from verl.utils.async_rpc import mark_scoring_started

    async def run():
        worker, store, publications = worker_harness(monkeypatch, n=n)
        worker.dapo_cutoff_enabled = True
        if opd_enabled:
            worker.config.trainer = {"use_v1": True}
            worker.config.distillation = {"enabled": True, "cancel_pending_on_cutoff": True,
                                          "wait_for_scoring_on_cutoff": True}
            worker.opd_cutoff_enabled = True
        scoring_started, finish_scoring = asyncio.Event(), asyncio.Event()
        generation_cancelled, scoring_cancelled = asyncio.Event(), asyncio.Event()

        async def session(params, *, session_id, **kwargs):
            if session_id == 0:
                mark_scoring_started()
                scoring_started.set()
                try:
                    await finish_scoring.wait()
                except asyncio.CancelledError:
                    scoring_cancelled.set()
                    raise
                output = fake_output(repeat.RepeatConfig())
                output.extra_fields.pop("_swe_repeat_job")
                await worker._agent_loop_postprocess(output, False, uid="u", session_id=0, global_steps=0)
            else:
                try:
                    await asyncio.Event().wait()
                finally:
                    generation_cancelled.set()

        worker._run_agent_loop = session
        prompt = asyncio.create_task(worker._run_prompt({"uid": "u"}, {}, {"validate": False}))
        worker.prompt_tasks["train"]["u"] = prompt
        await asyncio.wait_for(scoring_started.wait(), timeout=2)
        if opd_enabled:
            assert worker.opd_scoring_sessions[("train", "u")] is worker.dapo_scoring_sessions[("train", "u")]
        cutoff = asyncio.create_task(worker.cancel_sequences(["u"], dapo_cutoff=True, cleanup_timeout_seconds=.01))
        await asyncio.sleep(.04)  # Normal hidden scoring may exceed the generation cleanup deadline.
        assert not cutoff.done() and not scoring_cancelled.is_set()
        assert n == 1 or generation_cancelled.is_set()
        finish_scoring.set()
        assert await asyncio.wait_for(cutoff, timeout=2) == 1
        assert len(publications) == 1 and store["train"]["u"]["dapo_cutoff"]
        assert not worker.dapo_scoring_sessions and not worker.dapo_publications

    asyncio.run(run())


def test_dapo_cutoff_generation_cleanup_still_has_a_deadline(monkeypatch):
    async def run():
        worker, _, _ = worker_harness(monkeypatch)
        worker.dapo_cutoff_enabled = True
        started, release_cleanup = asyncio.Event(), asyncio.Event()

        async def session(*args, **kwargs):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                await release_cleanup.wait()

        worker._run_agent_loop = session
        prompt = asyncio.create_task(worker._run_prompt({"uid": "u"}, {}, {"validate": False}))
        worker.prompt_tasks["train"]["u"] = prompt
        await asyncio.wait_for(started.wait(), timeout=2)
        try:
            with pytest.raises(RuntimeError, match="DAPO generation cleanup exceeded"):
                await worker.cancel_sequences(["u"], dapo_cutoff=True, cleanup_timeout_seconds=.01)
            assert not prompt.done()
        finally:
            release_cleanup.set()
            await asyncio.gather(prompt, return_exceptions=True)

    asyncio.run(run())


def test_dapo_cutoff_tag_publication_still_has_a_deadline(monkeypatch):
    from verl.trainer.ppo.v1 import agent_loop_tq as module

    async def run():
        worker, _, _ = worker_harness(monkeypatch)
        worker.dapo_cutoff_enabled = True
        started, publishing, release_publication = asyncio.Event(), asyncio.Event(), asyncio.Event()
        original_put = module.tq.async_kv_put

        async def put(**kwargs):
            if kwargs["tag"].get("dapo_cutoff"):
                publishing.set()
                await release_publication.wait()
            return await original_put(**kwargs)

        monkeypatch.setattr(module.tq, "async_kv_put", put)

        async def session(*args, **kwargs):
            started.set()
            await asyncio.Event().wait()

        worker._run_agent_loop = session
        prompt = asyncio.create_task(worker._run_prompt({"uid": "u"}, {}, {"validate": False}))
        worker.prompt_tasks["train"]["u"] = prompt
        await asyncio.wait_for(started.wait(), timeout=2)
        try:
            with pytest.raises(RuntimeError, match="DAPO cutoff tag publication cleanup exceeded"):
                await worker.cancel_sequences(["u"], dapo_cutoff=True, cleanup_timeout_seconds=.01)
            assert publishing.is_set()
        finally:
            release_publication.set()
            await asyncio.sleep(0)
            await asyncio.gather(prompt, return_exceptions=True)

    asyncio.run(run())


def test_dapo_cutoff_keeps_retained_reward_errors_fatal(monkeypatch):
    async def run():
        worker, store, _ = worker_harness(monkeypatch)
        worker.dapo_cutoff_enabled = True
        judging, release = asyncio.Event(), asyncio.Event()

        async def submit(job_id, job):
            judging.set()
            await release.wait()
            raise RuntimeError("retained judge failed")

        worker.repeat_coordinator = SimpleNamespace(submit=RemoteMethod(submit))

        async def session(*args, **kwargs):
            await worker._agent_loop_postprocess(fake_output(repeat.RepeatConfig(deferred_enabled=True)), False,
                                                uid="u", session_id=0, global_steps=0)

        worker._run_agent_loop = session
        prompt = asyncio.create_task(worker._run_prompt({"uid": "u"}, {}, {"validate": False}))
        worker.prompt_tasks["train"]["u"] = prompt
        await asyncio.wait_for(judging.wait(), timeout=2)
        cutoff = asyncio.create_task(worker.cancel_sequences(["u"], dapo_cutoff=True))
        await asyncio.sleep(0)
        release.set()
        with pytest.raises(RuntimeError, match="DAPO rollout cancellation cleanup failed"):
            await asyncio.wait_for(cutoff, timeout=2)
        assert store["train"]["u"]["status"] == "reward_error"
        assert not store["train"]["u"].get("dapo_cutoff")

    asyncio.run(run())


def test_dapo_cutoff_does_not_overwrite_a_reward_error_when_parent_returns_normally(monkeypatch):
    async def run():
        worker, store, _ = worker_harness(monkeypatch)
        worker.dapo_scoring_sessions[("train", "u")] = []
        release = asyncio.Event()

        async def parent():
            await release.wait()
            store.setdefault("train", {})["u"] = {"status": "reward_error"}

        prompt = asyncio.create_task(parent())
        worker.prompt_tasks["train"]["u"] = prompt
        cutoff = asyncio.create_task(worker.cancel_sequences(["u"], dapo_cutoff=True))
        await asyncio.sleep(0)
        release.set()
        with pytest.raises(RuntimeError, match="DAPO retained reward processing failed"):
            await asyncio.wait_for(cutoff, timeout=2)
        assert store["train"]["u"]["status"] == "reward_error"

    asyncio.run(run())


def test_pending_submission_does_not_block_generation_and_is_protected_from_cutoff(monkeypatch):
    async def run():
        worker, store, publications = worker_harness(monkeypatch)
        capacity, entered, generation_returned = asyncio.Event(), asyncio.Event(), asyncio.Event()
        config = repeat.RepeatConfig(deferred_enabled=True)
        async def submit(job_id, job):
            entered.set()
            await capacity.wait()
            return job_id, .1, {"repeat_reward_pending": 0}
        worker.repeat_coordinator = SimpleNamespace(submit=RemoteMethod(submit))
        async def session(*args, **kwargs):
            await worker._agent_loop_postprocess(fake_output(config), False, uid="u", session_id=0, global_steps=0)
            generation_returned.set()
        worker._run_agent_loop = session
        prompt_task = asyncio.create_task(worker._run_prompt({"uid": "u"}, {}, {"validate": False}))
        worker.prompt_tasks["train"]["u"] = prompt_task
        await asyncio.wait_for(generation_returned.wait(), timeout=2)
        await entered.wait()
        assert store["train"]["u"]["generation_complete"]
        assert not prompt_task.done() and not publications
        assert await worker.cancel_sequences(["u"]) == 0
        capacity.set()
        await prompt_task
        assert store["train"]["u"]["status"] == "finished"
        assert len(publications) == 1
    asyncio.run(run())


def test_strict_validation_publishes_official_evaluator_evidence(monkeypatch):
    from verl.experimental.agent_loop import AgentLoopOutput
    from verl.utils.swe_validation import evaluation_reason

    monkeypatch.setenv("SWE_AGENT_VALIDATION_STRICT_ACCOUNTING", "1")

    async def run():
        worker, _store, publications = worker_harness(monkeypatch)
        info = {
            "evaluation_valid": True,
            "hidden_pass": 1,
            "evaluator_invalid_reason": "",
            "partial_hidden_reward_counts_complete": False,
        }
        output = AgentLoopOutput(
            prompt_ids=[1],
            response_ids=[2],
            response_mask=[1],
            reward_score=1.0,
            metrics={},
            extra_fields={"reward_extra_info": info},
        )
        await worker._agent_loop_postprocess(output, True, uid="u", session_id=0, global_steps=0)
        tags = publications[0][2][0]
        assert tags["evaluation_valid"] and tags["hidden_pass"] == 1
        assert evaluation_reason(tags) is None

    asyncio.run(run())


def test_strict_validation_can_cancel_reward_pending_prompt(monkeypatch):
    monkeypatch.setenv("SWE_AGENT_VALIDATION_STRICT_ACCOUNTING", "1")

    async def run():
        worker, store, publications = worker_harness(monkeypatch)
        entered, release = asyncio.Event(), asyncio.Event()

        async def submit(job_id, job):
            entered.set()
            await release.wait()
            return job_id, 0.1, {"repeat_reward_pending": 0}

        worker.repeat_coordinator = SimpleNamespace(submit=RemoteMethod(submit))

        async def session(*args, **kwargs):
            await worker._agent_loop_postprocess(
                fake_output(repeat.RepeatConfig(deferred_enabled=True)), True, uid="u", session_id=0, global_steps=0
            )

        worker._run_agent_loop = session
        prompt = asyncio.create_task(worker._run_prompt({"uid": "u"}, {}, {"validate": True}))
        worker.prompt_tasks["val"]["u"] = prompt
        await entered.wait()
        assert store["val"]["u"]["generation_complete"]
        cancelling = asyncio.create_task(worker.cancel_sequences(["u"], validate=True))
        await asyncio.sleep(0)
        assert prompt.cancelling() and not cancelling.done()
        release.set()
        assert await asyncio.wait_for(cancelling, timeout=2) == 1
        assert prompt.done() and len(publications) == 1

    asyncio.run(run())


@pytest.mark.parametrize("deferred", [False, True])
@pytest.mark.parametrize("stage1", [False, True])
def test_real_swe_preparation_keeps_stage1_inline_and_releases_before_queue(monkeypatch, deferred, stage1):
    from recipe.swe_agent.agent_loop import SWEToolAgentLoop
    async def run():
        loop = object.__new__(SWEToolAgentLoop)
        loop._swe_repeat_config = repeat.RepeatConfig(mode="apply", deferred_enabled=deferred)
        loop._swe_repeat_events = []
        loop._swe_repeat_actor_segments = []
        loop._swe_agent_data = SimpleNamespace(prompt_ids=[1], extra_fields={"submission_signal_seen": True})
        loop._swe_force_release = False
        loop.tokenizer = SimpleNamespace(decode=lambda *args, **kwargs: "public prompt")
        released = []
        async def release(**kwargs):
            released.append(True)
            return {"ok": True}
        loop._release_remote_session = release
        monkeypatch.setattr(repeat, "conservative_detect", lambda *args, **kwargs:
                            {"verdict": "repetitive" if stage1 else "uncertain", "reason": "fixture"})
        class Client:
            async def judge_consensus(self, *args, **kwargs):
                assert released
                assert not deferred
                return {"verdict": "repetitive", "reason": "fixture"}
        monkeypatch.setattr(repeat, "get_client", lambda config: Client())
        output = SimpleNamespace(extra_fields={})
        score, metadata = await loop._apply_success_repeat_reward(output, raw_score=1., existing_score=1., valid=True)
        assert released
        if deferred and not stage1:
            assert score == 1. and output.extra_fields["_swe_repeat_job"]["transcript"] == "public prompt"
            assert metadata["repeat_reward_pending"] == 1
        else:
            assert score == .1 and "_swe_repeat_job" not in output.extra_fields
            assert metadata["repeat_reward_stage"] == (1 if stage1 else 2)
    monkeypatch.setattr(repeat, "save_audit", lambda *args, **kwargs: {})
    asyncio.run(run())


def test_worker_queue_loss_is_terminal_error(monkeypatch):
    async def run():
        worker, store, publications = worker_harness(monkeypatch)
        async def submit(job_id, job):
            raise RuntimeError("actor unavailable")
        worker.repeat_coordinator = SimpleNamespace(submit=RemoteMethod(submit))
        async def session(*args, **kwargs):
            await worker._agent_loop_postprocess(fake_output(repeat.RepeatConfig()), False,
                                                uid="u", session_id=0, global_steps=0)
        worker._run_agent_loop = session
        await worker._run_prompt({"uid": "u"}, {}, {"validate": False})
        assert store["train"]["u"]["status"] == "reward_error"
        assert not publications
    asyncio.run(run())


def test_debug_queue_reports_full_fallback_and_parsed_startup(monkeypatch, caplog):
    async def run():
        config = repeat.RepeatConfig(max_pending=1, concurrency=1)
        queue = RepeatJudgeCoordinator(config)
        entered, release = asyncio.Event(), asyncio.Event()
        class Client:
            async def judge_consensus(self, *args, **kwargs):
                entered.set()
                await release.wait()
                return {"verdict": "clean", "reason": "fixture"}
        queue.client = Client()
        task = asyncio.create_task(queue.submit("first", make_job(config)))
        await entered.wait()
        await queue.submit("second", make_job(config))
        release.set()
        await task

    monkeypatch.setenv("SWE_AGENT_REPEAT_JUDGE_DEBUG", "1")
    monkeypatch.delenv("SWE_AGENT_REPEAT_JUDGE_MODEL", raising=False)
    monkeypatch.setattr(repeat, "save_audit", lambda *args, **kwargs: {})
    caplog.set_level(logging.INFO, logger="verl.trainer.ppo.v1.repeat_judge_queue")
    asyncio.run(run())
    messages = [record.getMessage() for record in caplog.records]
    assert any("model=swe-repeat-judge" in message for message in messages)
    assert any("reason=queue_full" in message and "dropped=1" in message for message in messages)
    assert any("capacity=0" in message for message in messages)


def test_debug_queue_reports_judge_rpc_and_verdict(monkeypatch, caplog):
    async def run():
        config = repeat.RepeatConfig(max_pending=1)
        queue = RepeatJudgeCoordinator(config)
        started, release = asyncio.Event(), asyncio.Event()

        class Client:
            async def judge_consensus(self, *args, **kwargs):
                started.set()
                await release.wait()
                return {"verdict": "clean", "reason": "fixture", "queue_seconds": 0.5, "service_seconds": 1.5}

        queue.client = Client()
        task = asyncio.create_task(queue.submit("judge", make_job(config)))
        await started.wait()
        release.set()
        await task

    monkeypatch.setenv("SWE_AGENT_REPEAT_JUDGE_DEBUG", "1")
    monkeypatch.setenv("SWE_AGENT_REPEAT_JUDGE_DEBUG_SLOW_SECONDS", "0")
    monkeypatch.setattr(repeat, "save_audit", lambda *args, **kwargs: {})
    caplog.set_level(logging.INFO, logger="verl.trainer.ppo.v1.repeat_judge_queue")
    asyncio.run(run())
    messages = [record.getMessage() for record in caplog.records]
    assert any("RPC submitted job_id=judge" in message for message in messages)
    assert any("verdict job_id=judge verdict=clean" in message for message in messages)
    assert any("phase=judge_rpc job_id=judge" in message for message in messages)


def test_debug_lifecycle_identifies_delayed_finalization(monkeypatch, caplog):
    async def run():
        worker, _store, _publications = worker_harness(monkeypatch)
        config = repeat.RepeatConfig(deferred_enabled=True)
        release = asyncio.Event()

        async def submit(job_id, job):
            await release.wait()
            return job_id, 0.1, {"repeat_reward_pending": 0}

        worker.repeat_coordinator = SimpleNamespace(
            submit=RemoteMethod(submit)
        )
        worker.repeat_finalizers[("train", "u")] = []
        worker.repeat_generation_sessions[("train", "u")] = (1, set())
        await worker._agent_loop_postprocess(fake_output(config), False, uid="u", session_id=0, global_steps=0)
        await asyncio.sleep(0)
        assert worker.repeat_finalizers[("train", "u")]
        assert not worker.repeat_finalizers[("train", "u")][0].done()
        release.set()
        await worker._join_repeat("train", "u")

    monkeypatch.setenv("SWE_AGENT_REPEAT_JUDGE_DEBUG", "1")
    monkeypatch.setenv("SWE_AGENT_REPEAT_JUDGE_DEBUG_SLOW_SECONDS", "0")
    monkeypatch.setattr(repeat, "save_audit", lambda *args, **kwargs: {})
    caplog.set_level(logging.INFO, logger="verl.trainer.ppo.v1.agent_loop_tq")
    asyncio.run(run())
    messages = [record.getMessage() for record in caplog.records]
    assert any("event=finalizer_start partition=train uid=u" in message for message in messages)
    assert any("event=join_repeat_start partition=train uid=u" in message for message in messages)
    assert any("event=finalizer_end partition=train uid=u" in message for message in messages)


def test_debug_lifecycle_identifies_delayed_transfer_queue_publication(monkeypatch, caplog):
    async def run():
        worker, _store, _publications = worker_harness(monkeypatch)
        from verl.trainer.ppo.v1 import agent_loop_tq as module

        config = repeat.RepeatConfig(deferred_enabled=True)
        entered, release = asyncio.Event(), asyncio.Event()

        async def submit(job_id, job):
            return job_id, 0.1, {"repeat_reward_pending": 0}

        async def delayed_batch_put(*args, **kwargs):
            entered.set()
            await release.wait()
            return None

        monkeypatch.setattr(module.tq, "async_kv_batch_put", delayed_batch_put)
        worker.repeat_coordinator = SimpleNamespace(
            submit=RemoteMethod(submit)
        )
        worker.repeat_finalizers[("train", "u")] = []
        worker.repeat_generation_sessions[("train", "u")] = (1, set())
        await worker._agent_loop_postprocess(fake_output(config), False, uid="u", session_id=0, global_steps=0)
        await entered.wait()
        task = worker.repeat_finalizers[("train", "u")][0]
        assert not task.done()
        release.set()
        await worker._join_repeat("train", "u")

    monkeypatch.setenv("SWE_AGENT_REPEAT_JUDGE_DEBUG", "1")
    monkeypatch.setenv("SWE_AGENT_REPEAT_JUDGE_DEBUG_SLOW_SECONDS", "0")
    monkeypatch.setattr(repeat, "save_audit", lambda *args, **kwargs: {})
    caplog.set_level(logging.INFO, logger="verl.trainer.ppo.v1.agent_loop_tq")
    asyncio.run(run())
    messages = [record.getMessage() for record in caplog.records]
    assert any("event=trajectory_published partition=train uid=u" in message for message in messages)
    assert any("event=finalizer_end partition=train uid=u" in message for message in messages)


def test_reward_drain_heartbeat_preserves_publication_barrier(monkeypatch, caplog):
    async def run():
        worker, _store, _publications = worker_harness(monkeypatch)
        release = asyncio.Event()
        finalizer = asyncio.create_task(release.wait())
        worker.repeat_finalizers[("train", "u")] = [finalizer]
        join = asyncio.create_task(worker._join_repeat("train", "u"))
        try:
            async with asyncio.timeout(2):
                while "event=join_repeat_progress" not in caplog.text:
                    await asyncio.sleep(.002)
            assert not join.done() and not finalizer.done()
            release.set()
            await join
            assert finalizer.done()
        finally:
            release.set()
            await join
    monkeypatch.setenv("SWE_AGENT_REPEAT_JUDGE_DEBUG", "1")
    monkeypatch.setenv("SWE_AGENT_REPEAT_JUDGE_PROGRESS_INTERVAL_SECONDS", ".005")
    caplog.set_level(logging.INFO, logger="verl.trainer.ppo.v1.agent_loop_tq")
    asyncio.run(run())


def test_manager_cancellation_heartbeat_does_not_block_on_snapshot(monkeypatch, caplog):
    from verl.trainer.ppo.v1 import agent_loop_tq as module
    manager = object.__new__(module.AgentLoopManagerTQ)
    manager.agent_loop_workers = [SimpleNamespace(cancel_sequences=SimpleNamespace(remote=lambda *a, **kw: "worker"))]
    manager.repeat_coordinator = SimpleNamespace(snapshot=SimpleNamespace(remote=lambda: "snapshot"))
    tick = [0.]
    worker_polls = [0]
    def wait(refs, *, num_returns=1, timeout):
        if refs == ["snapshot"]:
            assert timeout == 0
            return [], refs  # A slow snapshot must never stall cancellation.
        worker_polls[0] += 1
        tick[0] += 2
        return ([], refs) if worker_polls[0] == 1 else (refs, [])
    def get(refs):
        assert refs != "snapshot"
        return [1] if refs else []
    monkeypatch.setenv("SWE_AGENT_REPEAT_JUDGE_PROGRESS_INTERVAL_SECONDS", "1")
    monkeypatch.setattr(module.ray, "wait", wait)
    monkeypatch.setattr(module.ray, "get", get)
    monkeypatch.setattr(module.time, "monotonic", lambda: tick[0])
    caplog.set_level(logging.INFO, logger=module.__name__)
    assert manager.cancel_sequences(["u"]) == 1
    assert "cancellation/drain progress" in caplog.text
    assert "cancellation/drain end" in caplog.text


def test_dapo_manager_waits_for_retained_scoring_beyond_cleanup_deadline(monkeypatch):
    from verl.trainer.ppo.v1 import agent_loop_tq as module

    manager = object.__new__(module.AgentLoopManagerTQ)
    forwarded = []

    def remote(*args, **kwargs):
        forwarded.append(kwargs)
        return "worker"

    manager.agent_loop_workers = [SimpleNamespace(cancel_sequences=SimpleNamespace(remote=remote))]
    manager.repeat_coordinator = None
    tick = [0.]

    def wait(refs, **kwargs):
        tick[0] += 1
        return (refs, []) if tick[0] == 4 else ([], refs)

    monkeypatch.setattr(module.ray, "wait", wait)
    monkeypatch.setattr(module.ray, "get", lambda refs: [1] if refs else [])
    monkeypatch.setattr(module.time, "monotonic", lambda: tick[0])
    assert manager.cancel_sequences(["u"], dapo_cutoff=True, cleanup_timeout_seconds=2.) == 1
    assert tick[0] == 4
    assert forwarded == [{"validate": False, "dapo_cutoff": True, "cleanup_timeout_seconds": 2.}]

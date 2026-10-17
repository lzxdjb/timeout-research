"""Network-free contracts for repeat-only deferred reward scheduling."""
import asyncio
import copy
from dataclasses import replace
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
        admission = await queue.reserve("sample")
        identity, score, meta = await queue.evaluate("sample", job, admission)
        assert identity == "sample" and score == inline_score
        for key, value in inline_meta.items():
            assert meta[key] == value
        assert meta["repeat_reward_pending"] == 0
        assert meta["repeat_reward_deferred_request_queue_seconds"] == 2
        assert not queue.reservations and not queue.running
    monkeypatch.setattr(repeat, "save_audit", lambda *args, **kwargs: {})
    asyncio.run(run())


def test_bounded_admission_duplicates_and_failure_release(monkeypatch):
    async def run():
        config = repeat.RepeatConfig(max_pending=1)
        queue = RepeatJudgeCoordinator(config)
        class Client:
            async def judge_consensus(self, *args, **kwargs):
                raise RuntimeError("coordinator processing failed")
        queue.client = Client()
        await queue.reserve("first")
        pending = asyncio.create_task(queue.reserve("second"))
        await asyncio.sleep(0)
        assert not pending.done()
        with pytest.raises(RuntimeError, match="processing failed"):
            await queue.evaluate("first", make_job(config))
        await pending
        await queue.abandon("second")
        await queue.reserve("first")
        await queue.abandon("first")
        with pytest.raises(RuntimeError, match="Missing or duplicate"):
            await queue.evaluate("first", make_job(config))
        config2 = replace(config, max_pending=2)
        other = RepeatJudgeCoordinator(config2)
        await other.reserve("same")
        with pytest.raises(RuntimeError, match="Duplicate"):
            await other.reserve("same")
        await other.abandon("same")
    asyncio.run(run())


def test_one_request_semaphore_across_independent_jobs(monkeypatch):
    async def run():
        config = repeat.RepeatConfig(concurrency=2, max_pending=10, vote_count=3, vote_quorum=2)
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
            await queue.reserve(key)
            return await queue.evaluate(key, make_job(config))
        results = await asyncio.gather(*(job(i) for i in range(8)))
        assert len(results) == 8 and peak == 2
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
    worker.repeat_generation_sessions = {}
    worker.prompt_tasks = {"train": {}, "val": {}}
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
        async def reserve(job_id):
            return 0.
        async def evaluate(job_id, job, admission):
            await complete.wait()
            return job_id, .1, {"repeat_reward_pending": 0}
        worker.repeat_coordinator = SimpleNamespace(reserve=RemoteMethod(reserve), evaluate=RemoteMethod(evaluate))
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
        async def reserve(job_id):
            return 0.
        async def evaluate(job_id, job, admission):
            judging.set()
            await finish_judge.wait()
            return job_id, .1, {"repeat_reward_pending": 0}
        worker.repeat_coordinator = SimpleNamespace(reserve=RemoteMethod(reserve), evaluate=RemoteMethod(evaluate))
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


def test_backpressure_completed_generation_is_protected_from_cutoff(monkeypatch):
    async def run():
        worker, store, publications = worker_harness(monkeypatch)
        capacity = asyncio.Event()
        entered = asyncio.Event()
        config = repeat.RepeatConfig(deferred_enabled=True)
        async def reserve(job_id):
            entered.set()
            await capacity.wait()
            return 1.
        async def evaluate(job_id, job, admission):
            return job_id, .1, {"repeat_reward_pending": 0}
        worker.repeat_coordinator = SimpleNamespace(reserve=RemoteMethod(reserve), evaluate=RemoteMethod(evaluate))
        async def session(*args, **kwargs):
            await worker._agent_loop_postprocess(fake_output(config), False, uid="u", session_id=0, global_steps=0)
        worker._run_agent_loop = session
        prompt_task = asyncio.create_task(worker._run_prompt({"uid": "u"}, {}, {"validate": False}))
        worker.prompt_tasks["train"]["u"] = prompt_task
        await entered.wait()
        assert store["train"]["u"]["generation_complete"]
        assert await worker.cancel_sequences(["u"]) == 0
        capacity.set()
        await prompt_task
        assert store["train"]["u"]["status"] == "finished"
        assert len(publications) == 1
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
        async def reserve(job_id): return 0.
        async def evaluate(job_id, job, admission):
            raise RuntimeError("actor unavailable")
        worker.repeat_coordinator = SimpleNamespace(reserve=RemoteMethod(reserve), evaluate=RemoteMethod(evaluate))
        async def session(*args, **kwargs):
            await worker._agent_loop_postprocess(fake_output(repeat.RepeatConfig()), False,
                                                uid="u", session_id=0, global_steps=0)
        worker._run_agent_loop = session
        await worker._run_prompt({"uid": "u"}, {}, {"validate": False})
        assert store["train"]["u"]["status"] == "reward_error"
        assert not publications
    asyncio.run(run())

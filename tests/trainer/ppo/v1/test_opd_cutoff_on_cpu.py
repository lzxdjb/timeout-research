"""Whole-group OPD cutoff, publication settlement, and all-work phase metrics."""

import asyncio
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from verl.trainer.distillation.config_utils import (
    opd_cancel_on_cutoff,
    opd_cleanup_timeout_s,
    opd_phase_timing_enabled,
)
from verl.trainer.ppo.v1 import agent_loop_tq as module
from verl.trainer.ppo.v1.utils import MetricsAggregator
from verl.utils.async_rpc import RPCCancellationError

Worker = module.AgentLoopWorkerTQ.__ray_metadata__.modified_class


def _config(**overrides):
    return OmegaConf.create(
        {
            "trainer": {"use_v1": True},
            "actor_rollout_ref": {"rollout": {"n": 2, "val_kwargs": {"n": 1}}},
            "distillation": {
                "enabled": True,
                "distillation_loss": {"use_task_rewards": False},
                "cancel_pending_on_cutoff": True,
                "phase_timing": True,
                "cutoff_cleanup_timeout_s": 1,
                **overrides,
            },
        }
    )


def _worker(monkeypatch, *, task_rewards=False):
    worker = object.__new__(Worker)
    worker.config = _config(distillation_loss={"use_task_rewards": task_rewards})
    worker.repeat_coordinator = None
    worker.dapo_cutoff_enabled = False
    worker.opd_cutoff_enabled = True
    worker.dapo_cutoff_uids = set()
    worker.repeat_debug = False
    for name in (
        "repeat_finalizers",
        "repeat_job_refs",
        "dapo_publications",
        "repeat_generation_sessions",
        "repeat_debug_prompts",
        "cutoff_session_states",
    ):
        setattr(worker, name, {})
    worker.prompt_tasks = {"train": {}, "val": {}}
    snapshot = {"train": {"g": {"status": "pending"}}}

    async def put(*, key, partition_id, tag):
        snapshot[partition_id].setdefault(key, {}).update(tag)

    def put_batch(*, keys, partition_id, tags):
        for key, tag in zip(keys, tags, strict=True):
            snapshot[partition_id][key].update(tag)

    monkeypatch.setattr(module.tq, "async_kv_put", put)
    monkeypatch.setattr(module.tq, "kv_batch_put", put_batch)
    monkeypatch.setattr(module.tq, "kv_list", lambda *_a: snapshot)
    monkeypatch.setattr(module.tq, "kv_batch_get", lambda **_k: {})
    return worker, snapshot


@pytest.mark.parametrize("sibling_failed", [False, True])
@pytest.mark.parametrize("task_rewards", [False, True])
def test_cutoff_cancels_teacher_pending_group_and_excludes_completed_sibling(monkeypatch, sibling_failed, task_rewards):
    worker, snapshot = _worker(monkeypatch, task_rewards=task_rewards)

    async def run():
        teacher_started = asyncio.Event()
        teacher_cleaned = asyncio.Event()

        async def session(_params, *, session_id, **_kwargs):
            if session_id == 0:
                if sibling_failed:
                    raise ValueError("ordinary agent/environment failure")
                snapshot["train"]["g_0_0"] = {"status": "success"}
                return
            teacher_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                await asyncio.sleep(0)
                teacher_cleaned.set()

        worker._run_agent_loop = session
        task = asyncio.create_task(worker._run_prompt({"uid": "g"}, {}, {"validate": False}))
        worker.prompt_tasks["train"]["g"] = task
        await asyncio.wait_for(teacher_started.wait(), 1)
        assert snapshot["train"]["g"]["status"] == "running"
        assert await worker.cancel_sequences(["g"]) == 1
        assert teacher_cleaned.is_set()
        assert task.cancelled()

    asyncio.run(run())
    assert snapshot["train"]["g"]["completion_ratio_cutoff"]
    if not sibling_failed:
        assert snapshot["train"]["g_0_0"]["completion_ratio_cutoff"]


@pytest.mark.parametrize("publication_fails", [False, True])
@pytest.mark.parametrize("task_rewards", [False, True])
def test_cutoff_ack_waits_for_dispatched_publication(monkeypatch, publication_fails, task_rewards):
    worker, snapshot = _worker(monkeypatch, task_rewards=task_rewards)

    async def run():
        publishing, release, published = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def write():
            publishing.set()
            await release.wait()
            if publication_fails:
                raise RuntimeError("queue write failed")
            snapshot["train"]["g_0_0"] = {"status": "success"}
            published.set()

        async def session(_params, *, session_id, **_kwargs):
            if session_id == 0:
                await worker._await_tq_publication("train", "g", write())
            else:
                await asyncio.Event().wait()

        worker._run_agent_loop = session
        prompt = asyncio.create_task(worker._run_prompt({"uid": "g"}, {}, {"validate": False}))
        worker.prompt_tasks["train"]["g"] = prompt
        await asyncio.wait_for(publishing.wait(), 1)
        cancellation = asyncio.create_task(worker.cancel_sequences(["g"]))
        await asyncio.sleep(0.01)
        assert not cancellation.done()
        assert not published.is_set()
        release.set()
        if publication_fails:
            with pytest.raises(RPCCancellationError, match="cleanup failed"):
                await cancellation
        else:
            assert await cancellation == 1
            assert published.is_set()
            assert snapshot["train"]["g_0_0"]["completion_ratio_cutoff"]
            final_snapshot = repr(snapshot)
            await asyncio.sleep(0)
            assert repr(snapshot) == final_snapshot

    asyncio.run(run())


@pytest.mark.parametrize("task_rewards", [False, True])
def test_opd_manager_has_bounded_cleanup_deadline(monkeypatch, task_rewards):
    manager = object.__new__(module.AgentLoopManagerTQ)
    manager.config = _config(distillation_loss={"use_task_rewards": task_rewards})
    manager.agent_loop_workers = [SimpleNamespace(cancel_sequences=SimpleNamespace(remote=lambda *a, **k: "worker"))]
    manager.repeat_coordinator = None
    tick = [0.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: tick[0])

    def wait(refs, *, timeout, **_kwargs):
        tick[0] += timeout
        return [], refs

    monkeypatch.setattr(module.ray, "wait", wait)
    monkeypatch.setattr(module.ray, "get", lambda refs: [])
    with pytest.raises(RuntimeError, match="OPD cancellation.*cleanup deadline"):
        manager.cancel_sequences(["g"])
    assert tick[0] == 1


def test_snapshot_merging_and_iteration_aggregation_count_cancelled_work():
    snapshots = [
        {
            "teacher_http": {
                "seconds_sum": 10,
                "seconds_max": 8,
                "count": 2,
                "completed_count": 1,
                "cancelled_count": 1,
                "failed_count": 0,
            }
        },
        {
            "teacher_http": {
                "seconds_sum": 3,
                "seconds_max": 3,
                "count": 1,
                "completed_count": 0,
                "cancelled_count": 0,
                "failed_count": 1,
            }
        },
    ]
    metrics = module._summarize_opd_phase_metrics(snapshots)
    prefix = "opd_phase/teacher_http"
    assert metrics[f"{prefix}/seconds_total"] == 13
    assert metrics[f"{prefix}/seconds_max"] == 8
    assert metrics[f"{prefix}/count"] == 3
    assert metrics[f"{prefix}/seconds_mean"] == pytest.approx(13 / 3)
    agg = MetricsAggregator()
    agg.add_step_metrics(metrics, sample_count=64)
    agg.add_step_metrics(module._summarize_opd_phase_metrics(snapshots[:1]), sample_count=1)
    result = agg.get_aggregated_metrics()
    assert result[f"{prefix}/seconds_total"] == 23
    assert result[f"{prefix}/seconds_max"] == 8
    assert result[f"{prefix}/count"] == 5
    assert result[f"{prefix}/cancelled_count"] == 2
    assert result[f"{prefix}/seconds_mean"] == pytest.approx(23 / 5)


def test_manager_collects_worker_and_shared_teacher_snapshots(monkeypatch):
    manager = object.__new__(module.AgentLoopManagerTQ)
    manager.config = _config()
    payload = {"teacher_http": {"seconds_sum": 2, "count": 1, "cancelled_count": 1}}
    remote = SimpleNamespace(pop_opd_phase_metrics=SimpleNamespace(remote=lambda: payload))
    manager.agent_loop_workers = [remote]
    manager.teacher_client = {"teacher": SimpleNamespace(_scoring_handle=remote)}
    monkeypatch.setattr(module.ray, "get", lambda refs, timeout: refs)
    result = manager.pop_opd_phase_metrics()
    assert result["opd_phase/teacher_http/count"] == 2
    assert result["opd_phase/teacher_http/cancelled_count"] == 2
    manager.config.distillation.phase_timing = False
    assert manager.pop_opd_phase_metrics() == {}


def test_options_are_default_off_and_training_only():
    assert not opd_cancel_on_cutoff({})
    assert not opd_phase_timing_enabled({})
    assert opd_cancel_on_cutoff(_config())
    assert opd_phase_timing_enabled(_config())
    assert not opd_cancel_on_cutoff(_config(), validate=True)
    assert not opd_phase_timing_enabled(_config(), validate=True)
    hybrid = _config(distillation_loss={"use_task_rewards": True})
    assert opd_cancel_on_cutoff(hybrid)
    assert not opd_cancel_on_cutoff(hybrid, validate=True)
    with pytest.raises(ValueError, match="requires enabled distillation"):
        opd_cancel_on_cutoff(_config(enabled=False))


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_cleanup_timeout_must_be_positive_finite(timeout):
    with pytest.raises(ValueError, match="positive and finite"):
        opd_cleanup_timeout_s(_config(cutoff_cleanup_timeout_s=timeout))

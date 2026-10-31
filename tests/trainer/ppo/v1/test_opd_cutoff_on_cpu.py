"""Whole-group OPD cutoff, publication settlement, and all-work phase metrics."""

import asyncio
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from verl.experimental.teacher_loop.errors import TeacherScoringError
from verl.trainer.distillation.config_utils import (
    opd_cancel_on_cutoff,
    opd_cleanup_timeout_s,
    opd_phase_timing_enabled,
)
from verl.trainer.ppo.v1 import agent_loop_tq as module
from verl.trainer.ppo.v1.utils import MetricsAggregator
from verl.utils.async_rpc import RPCCancellationError, mark_scoring_started

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
@pytest.mark.parametrize("scoring_phase", ["teacher_queue", "teacher_http", "hidden_then_teacher"])
def test_generation_only_cutoff_drains_scoring_pipeline(monkeypatch, task_rewards, scoring_phase):
    worker, snapshot = _worker(monkeypatch, task_rewards=task_rewards)
    worker.config.distillation.wait_for_scoring_on_cutoff = True

    async def run():
        scoring_started = asyncio.Event()
        generation_started = asyncio.Event()
        generation_cancelled = asyncio.Event()
        scoring_finished = asyncio.Event()
        release_first_phase = asyncio.Event()
        teacher_started = asyncio.Event()
        release_teacher = asyncio.Event()

        async def session(_params, *, session_id, **_kwargs):
            if session_id == 0:
                generation_started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    generation_cancelled.set()
                return
            # The marker protects both queued work and phases that have not
            # started yet, including teacher scoring after hidden evaluation.
            mark_scoring_started()
            scoring_started.set()
            await release_first_phase.wait()
            if scoring_phase == "hidden_then_teacher":
                teacher_started.set()
                await release_teacher.wait()
            snapshot["train"]["g_1_0"] = {"status": "success"}
            scoring_finished.set()

        worker._run_agent_loop = session
        prompt = asyncio.create_task(worker._run_prompt({"uid": "g"}, {}, {"validate": False}))
        worker.prompt_tasks["train"]["g"] = prompt
        await asyncio.wait_for(scoring_started.wait(), 1)
        await asyncio.wait_for(generation_started.wait(), 1)
        cutoff = asyncio.create_task(worker.cancel_sequences(["g"]))
        await asyncio.wait_for(generation_cancelled.wait(), 1)
        assert not cutoff.done()
        assert not scoring_finished.is_set()
        release_first_phase.set()
        if scoring_phase == "hidden_then_teacher":
            await asyncio.wait_for(teacher_started.wait(), 1)
            assert not cutoff.done()
            assert not scoring_finished.is_set()
            release_teacher.set()
        assert await asyncio.wait_for(cutoff, 1) == 1
        assert scoring_finished.is_set()
        assert prompt.cancelled()
        assert not worker.opd_scoring_sessions

    asyncio.run(run())
    assert snapshot["train"]["g"]["completion_ratio_cutoff"]
    assert snapshot["train"]["g_1_0"]["completion_ratio_cutoff"]


@pytest.mark.parametrize("task_rewards", [False, True])
def test_generation_only_cutoff_retains_group_when_all_sessions_are_scoring(monkeypatch, task_rewards):
    worker, snapshot = _worker(monkeypatch, task_rewards=task_rewards)
    worker.config.distillation.wait_for_scoring_on_cutoff = True

    async def run():
        started = [asyncio.Event(), asyncio.Event()]
        finish = asyncio.Event()

        async def session(_params, *, session_id, **_kwargs):
            mark_scoring_started()
            started[session_id].set()
            await finish.wait()
            snapshot["train"][f"g_{session_id}_0"] = {"status": "success"}

        worker._run_agent_loop = session
        prompt = asyncio.create_task(worker._run_prompt({"uid": "g"}, {}, {"validate": False}))
        worker.prompt_tasks["train"]["g"] = prompt
        await asyncio.wait_for(asyncio.gather(*(event.wait() for event in started)), 1)
        cutoff = asyncio.create_task(worker.cancel_sequences(["g"]))
        await asyncio.sleep(0)
        assert not cutoff.done()
        finish.set()
        assert await asyncio.wait_for(cutoff, 1) == 1
        assert not prompt.cancelled()
        assert not worker.opd_scoring_sessions

    asyncio.run(run())
    assert snapshot["train"]["g"]["status"] == "finished"
    assert not snapshot["train"]["g"].get("completion_ratio_cutoff")
    assert not snapshot["train"]["g_0_0"].get("completion_ratio_cutoff")
    assert not snapshot["train"]["g_1_0"].get("completion_ratio_cutoff")


def test_generation_selected_for_cutoff_cannot_start_scoring_after_swallowing_cancel(monkeypatch):
    worker, snapshot = _worker(monkeypatch)
    worker.config.distillation.wait_for_scoring_on_cutoff = True

    async def run():
        started = [asyncio.Event(), asyncio.Event()]
        scoring_started = []

        async def session(_params, *, session_id, **_kwargs):
            started[session_id].set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                pass
            mark_scoring_started()
            scoring_started.append(session_id)

        worker._run_agent_loop = session
        prompt = asyncio.create_task(worker._run_prompt({"uid": "g"}, {}, {"validate": False}))
        worker.prompt_tasks["train"]["g"] = prompt
        await asyncio.wait_for(asyncio.gather(*(event.wait() for event in started)), 1)
        assert await asyncio.wait_for(worker.cancel_sequences(["g"]), 1) == 1
        assert scoring_started == []
        assert prompt.cancelled()

    asyncio.run(run())
    assert snapshot["train"]["g"]["completion_ratio_cutoff"]


def test_scoring_drain_preserves_teacher_failure_channel(monkeypatch):
    worker, snapshot = _worker(monkeypatch)
    worker.config.distillation.wait_for_scoring_on_cutoff = True

    async def run():
        started = [asyncio.Event(), asyncio.Event()]
        finish = asyncio.Event()

        async def session(_params, *, session_id, **_kwargs):
            mark_scoring_started()
            started[session_id].set()
            await finish.wait()
            if session_id == 0:
                raise TeacherScoringError("malformed teacher target")

        worker._run_agent_loop = session
        prompt = asyncio.create_task(worker._run_prompt({"uid": "g"}, {}, {"validate": False}))
        worker.prompt_tasks["train"]["g"] = prompt
        await asyncio.wait_for(asyncio.gather(*(event.wait() for event in started)), 1)
        cutoff = asyncio.create_task(worker.cancel_sequences(["g"]))
        await asyncio.sleep(0)
        finish.set()
        assert await asyncio.wait_for(cutoff, 1) == 1

    asyncio.run(run())
    assert snapshot["train"]["g"]["status"] == "reward_error"
    assert "malformed teacher target" in snapshot["train"]["g"]["teacher_error"]
    assert not snapshot["train"]["g"].get("completion_ratio_cutoff")


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


def test_scoring_drain_can_exceed_cancellation_cleanup_deadline(monkeypatch):
    manager = object.__new__(module.AgentLoopManagerTQ)
    manager.config = _config(wait_for_scoring_on_cutoff=True)
    manager.agent_loop_workers = [SimpleNamespace(cancel_sequences=SimpleNamespace(remote=lambda *a, **k: "worker"))]
    manager.repeat_coordinator = None
    tick = [0.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: tick[0])
    monkeypatch.setattr(module, "progress_interval_seconds", lambda: 2.0)

    def wait(refs, *, timeout, **_kwargs):
        tick[0] += timeout
        return ([], refs) if tick[0] < 6 else (refs, [])

    monkeypatch.setattr(module.ray, "wait", wait)
    monkeypatch.setattr(module.ray, "get", lambda refs: [1 for _ in refs])
    assert manager.cancel_sequences(["g"]) == 1
    assert tick[0] == 6


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

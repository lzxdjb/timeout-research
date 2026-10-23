"""Exercise actual prompt cancellation without Ray actors or inference."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from verl.trainer.ppo.v1 import agent_loop_tq as module

Worker = module.AgentLoopWorkerTQ.__ray_metadata__.modified_class


def make_worker(monkeypatch):
    worker = object.__new__(Worker)
    worker.config = SimpleNamespace(actor_rollout_ref=SimpleNamespace(
        rollout=SimpleNamespace(n=2, val_kwargs=SimpleNamespace(n=1))))
    worker.repeat_coordinator = None
    worker.repeat_finalizers = {}
    worker.repeat_generation_sessions = {}
    worker.repeat_debug_prompts = {}
    worker.repeat_debug = False
    worker.cutoff_session_states = {}
    worker.prompt_tasks = {"train": {}, "val": {}}
    snapshot = {"train": {"group": {"status": "pending"}}}
    reads = []

    async def put(*, key, partition_id, tag):
        snapshot[partition_id].setdefault(key, {}).update(tag)

    def batch_put(*, keys, partition_id, tags):
        for key, tag in zip(keys, tags, strict=True):
            snapshot[partition_id][key].update(tag)

    def get(*, keys, **_kwargs):
        reads.extend(keys)
        return {}

    monkeypatch.setattr(module.tq, "async_kv_put", put)
    monkeypatch.setattr(module.tq, "kv_list", lambda *_args: snapshot)
    monkeypatch.setattr(module.tq, "kv_batch_put", batch_put)
    monkeypatch.setattr(module.tq, "kv_batch_get", get)
    return worker, snapshot, reads


@pytest.mark.parametrize("enabled", [False, True])
def test_partial_sibling_cutoff_keeps_queue_behavior_and_counts_sessions(tmp_path, monkeypatch, enabled):
    monkeypatch.setenv("SWE_AGENT_CUTOFF_DEBUG", str(int(enabled)))
    monkeypatch.setenv("SWE_AGENT_CUTOFF_DEBUG_DIR", str(tmp_path))
    worker, snapshot, reads = make_worker(monkeypatch)

    async def run():
        started = asyncio.Event()

        async def session(_params, *, session_id, **kwargs):
            context = kwargs.get("_swe_cutoff_context")
            if session_id == 0:
                snapshot["train"]["group_0_0"] = {"status": "success"}
                if context is not None:
                    context["published"].add(0)
                    context["published_trajectory_count"] = 1
                return
            started.set()
            await asyncio.Event().wait()

        worker._run_agent_loop = session
        task = asyncio.create_task(worker._run_prompt(
            {"uid": "group"}, {}, {"validate": False, "step": 3, "sample_index": 4}))
        worker.prompt_tasks["train"]["group"] = task
        await asyncio.wait_for(started.wait(), 2)
        assert await worker.cancel_sequences(["group"]) == 1
        assert task.cancelled()
        assert not worker.cutoff_session_states

    asyncio.run(run())
    assert snapshot["train"]["group"]["completion_ratio_cutoff"] is True
    assert snapshot["train"]["group_0_0"]["completion_ratio_cutoff"] is True
    assert reads == ["group_0_0"]  # No extra mutating readiness check for terminal groups.
    rows = [json.loads(line) for path in tmp_path.glob("*.jsonl") for line in path.read_text().splitlines()]
    if enabled:
        census, = [row for row in rows if row["event"] == "prompt_group_cutoff_census"]
        assert census["expected_sessions"] == 2
        assert census["started_sessions"] == 2
        assert census["returned_sessions"] == 1
        assert census["cancelled_sessions"] == 1
        assert census["published_sessions"] == 1
        assert census["cutoff_requested"] == 1
        assert census["cutoff_applied"] == 1
    else:
        assert rows == []


def test_never_started_group_is_counted_and_cleaned_up(tmp_path, monkeypatch):
    monkeypatch.setenv("SWE_AGENT_CUTOFF_DEBUG", "1")
    monkeypatch.setenv("SWE_AGENT_CUTOFF_DEBUG_DIR", str(tmp_path))
    worker, snapshot, _reads = make_worker(monkeypatch)
    worker.cutoff_session_states[("train", "group")] = {"expected": 2}

    async def run():
        task = asyncio.create_task(worker._run_prompt({"uid": "group"}, {}, {"validate": False}))
        worker.prompt_tasks["train"]["group"] = task
        assert await worker.cancel_sequences(["group"]) == 1
        assert task.cancelled()
        assert not worker.cutoff_session_states

    asyncio.run(run())
    row = json.loads(next(tmp_path.glob("*.jsonl")).read_text())
    assert row["not_started_sessions"] == 2
    assert row["cancelled_sessions"] == 0
    assert row["cutoff_requested"] == 1
    assert row["cutoff_applied"] == 1
    assert snapshot["train"]["group"]["status"] == "failure"


def test_deferred_completion_race_preserves_finished_group(tmp_path, monkeypatch):
    monkeypatch.setenv("SWE_AGENT_CUTOFF_DEBUG", "1")
    monkeypatch.setenv("SWE_AGENT_CUTOFF_DEBUG_DIR", str(tmp_path))
    worker, snapshot, reads = make_worker(monkeypatch)
    worker.repeat_coordinator = object()

    async def run():
        ready = asyncio.Event()

        async def session(_params, *, session_id, **_kwargs):
            worker.repeat_generation_sessions[("train", "group")][1].add(session_id)
            if session_id == 1:
                ready.set()
                await asyncio.Event().wait()

        worker._run_agent_loop = session
        task = asyncio.create_task(worker._run_prompt(
            {"uid": "group"}, {}, {"validate": False, "step": 2, "sample_index": 1}))
        worker.prompt_tasks["train"]["group"] = task
        await asyncio.wait_for(ready.wait(), 2)
        assert await worker.cancel_sequences(["group"]) == 1
        assert task.done() and not task.cancelled()

    asyncio.run(run())
    assert snapshot["train"]["group"]["status"] == "finished"
    assert not snapshot["train"]["group"].get("completion_ratio_cutoff")
    assert reads == []
    row = json.loads(next(tmp_path.glob("*.jsonl")).read_text())
    assert row["terminal_status"] == "finished"
    assert row["cutoff_requested"] == 1
    assert row["cutoff_applied"] == 0


def test_diagnostic_failure_cannot_mask_cancellation(monkeypatch):
    monkeypatch.setenv("SWE_AGENT_CUTOFF_DEBUG", "1")
    worker, snapshot, _reads = make_worker(monkeypatch)

    def broken_writer(*_args, **_kwargs):
        raise RuntimeError("diagnostics failed")

    monkeypatch.setattr(module, "write_cutoff_debug_event", broken_writer)
    worker.cutoff_session_states[("train", "group")] = {"expected": 2}

    async def run():
        task = asyncio.create_task(asyncio.sleep(100))
        worker.prompt_tasks["train"]["group"] = task
        assert await worker.cancel_sequences(["group"]) == 1
        assert task.cancelled()

    asyncio.run(run())
    assert snapshot["train"]["group"]["status"] == "failure"

"""Cancellation must reach remote work and settle before cutoff is acknowledged."""

import asyncio

import pytest
import ray

from verl.utils import async_rpc
from verl.utils.async_rpc import (
    PhaseMetrics,
    RPCCancellationError,
    await_cancellable_ray_ref,
    record_rpc_phase,
    rollout_request_scope,
)


def test_default_await_preserves_existing_behavior(monkeypatch):
    monkeypatch.setattr(async_rpc.ray, "cancel", lambda *_a, **_k: pytest.fail("unexpected remote cancellation"))

    async def run():
        future = asyncio.get_running_loop().create_future()
        task = asyncio.create_task(await_cancellable_ray_ref(future))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert future.cancelled()

    asyncio.run(run())


def test_cutoff_cancels_remote_ref_and_waits_for_acknowledgment(monkeypatch):
    async def run():
        future = asyncio.get_running_loop().create_future()
        requested = asyncio.Event()

        def cancel(ref, *, force):
            assert ref is future and force is False
            requested.set()

        monkeypatch.setattr(async_rpc.ray, "cancel", cancel)
        with rollout_request_scope(True, 1):
            task = asyncio.create_task(await_cancellable_ray_ref(future))
        await asyncio.sleep(0)
        task.cancel()
        await requested.wait()
        assert not future.cancelled()
        assert not task.done()
        future.set_exception(ray.exceptions.TaskCancelledError())
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())


def test_completion_racing_cancellation_is_harmless(monkeypatch):
    async def run():
        future = asyncio.get_running_loop().create_future()
        monkeypatch.setattr(async_rpc.ray, "cancel", lambda *_a, **_k: future.set_result("completed"))
        with rollout_request_scope(True, 1):
            task = asyncio.create_task(await_cancellable_ray_ref(future))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert future.result() == "completed"

    asyncio.run(run())


def test_cleanup_timeout_is_fatal_and_does_not_cancel_settlement_waiter(monkeypatch):
    monkeypatch.setattr(async_rpc.ray, "cancel", lambda *_a, **_k: None)

    async def run():
        future = asyncio.get_running_loop().create_future()
        with rollout_request_scope(True, 0.01):
            task = asyncio.create_task(await_cancellable_ray_ref(future))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(RPCCancellationError, match="did not settle"):
            await task
        assert not future.cancelled()
        future.set_result(None)

    asyncio.run(run())


def test_remote_cancellation_submission_error_is_fatal(monkeypatch):
    def fail(*_a, **_k):
        raise ValueError("disconnected")

    monkeypatch.setattr(async_rpc.ray, "cancel", fail)

    async def run():
        future = asyncio.get_running_loop().create_future()
        with rollout_request_scope(True, 1):
            task = asyncio.create_task(await_cancellable_ray_ref(future))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(RPCCancellationError, match="Unable to request"):
            await task
        future.set_result(None)

    asyncio.run(run())


def test_real_rpc_errors_are_not_replaced_by_fake_results():
    async def run():
        future = asyncio.get_running_loop().create_future()
        future.set_exception(ValueError("bad teacher output"))
        with rollout_request_scope(True, 1):
            with pytest.raises(ValueError, match="bad teacher output"):
                await await_cancellable_ray_ref(future)

    asyncio.run(run())


@pytest.mark.parametrize("returned_output", ["successful output", "failed rollout"])
@pytest.mark.parametrize("failure_mode", ["timeout", "submission"])
@pytest.mark.parametrize("child_task", [False, True])
def test_source_cannot_swallow_unsettled_rpc_cleanup_failure(monkeypatch, returned_output, failure_mode, child_task):
    async def run():
        future = asyncio.get_running_loop().create_future()
        started = asyncio.Event()
        swallowed = []

        def cancel(*_args, **_kwargs):
            if failure_mode == "submission":
                raise ValueError("disconnected")

        monkeypatch.setattr(async_rpc.ray, "cancel", cancel)

        async def rpc_operation():
            started.set()
            return await await_cancellable_ray_ref(future)

        async def source_run():
            try:
                operation = rpc_operation()
                await (asyncio.create_task(operation) if child_task else operation)
            except Exception as error:
                swallowed.append(error)
                return returned_output

        async def worker_run():
            with rollout_request_scope(True, 0.01):
                return await source_run()

        task = asyncio.create_task(worker_run())
        await started.wait()
        task.cancel()
        with pytest.raises(RPCCancellationError):
            await task
        assert len(swallowed) == 1 and isinstance(swallowed[0], RPCCancellationError)
        future.set_result(None)

    asyncio.run(run())


def test_second_cancellation_cannot_skip_remote_settlement(monkeypatch):
    async def run():
        future = asyncio.get_running_loop().create_future()
        started, requested = asyncio.Event(), asyncio.Event()
        monkeypatch.setattr(async_rpc.ray, "cancel", lambda *_args, **_kwargs: requested.set())

        async def worker_run():
            with rollout_request_scope(True, 1):
                started.set()
                await await_cancellable_ray_ref(future)

        task = asyncio.create_task(worker_run())
        await started.wait()
        task.cancel()
        await requested.wait()
        task.cancel()
        with pytest.raises(RPCCancellationError, match="interrupted"):
            await task
        assert not future.cancelled()
        future.set_result(None)

    asyncio.run(run())


def test_nested_source_scope_cannot_hide_fatal_cleanup_failure(monkeypatch):
    async def run():
        future = asyncio.get_running_loop().create_future()
        started = asyncio.Event()
        monkeypatch.setattr(async_rpc.ray, "cancel", lambda *_args, **_kwargs: None)

        async def worker_run():
            with rollout_request_scope(True, 0.01):
                try:
                    with rollout_request_scope(True, 0.01):
                        started.set()
                        await await_cancellable_ray_ref(future)
                except RPCCancellationError:
                    return "incorrectly recovered"

        task = asyncio.create_task(worker_run())
        await started.wait()
        task.cancel()
        with pytest.raises(RPCCancellationError, match="did not settle"):
            await task
        future.set_result(None)

    asyncio.run(run())


def test_nested_scopes_and_phase_snapshots_are_isolated():
    metrics = PhaseMetrics()
    with rollout_request_scope(False, 1, metrics.record):
        record_rpc_phase("teacher_target", 2, "completed")
        with rollout_request_scope(False, 1):
            record_rpc_phase("teacher_target", 100, "failed")
        record_rpc_phase("teacher_target", 3, "cancelled")
        record_rpc_phase("teacher_target", 1, "failed")
    record_rpc_phase("teacher_target", 200, "completed")
    row = metrics.pop()["teacher_target"]
    assert row == {
        "seconds_sum": 6,
        "seconds_max": 3,
        "count": 3,
        "completed_count": 1,
        "cancelled_count": 1,
        "failed_count": 1,
    }
    assert metrics.pop() == {}


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_cleanup_timeout_must_be_finite_and_positive(timeout):
    with pytest.raises(ValueError):
        with rollout_request_scope(True, timeout):
            pass

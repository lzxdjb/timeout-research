"""Student cancellation must release routing only after its RPC has settled."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import ray
from omegaconf import OmegaConf

from verl.utils import async_rpc
from verl.utils.async_rpc import PhaseMetrics, RPCCancellationError, rollout_request_scope
from verl.workers.rollout.llm_server import LLMServerClient
from verl.workers.rollout.replica import TokenOutput


@pytest.mark.parametrize("outcome", ["completed", "cancelled", "failed"])
def test_student_rpc_phase_and_routing_cleanup(monkeypatch, outcome):
    async def exercise():
        future = asyncio.get_running_loop().create_future()
        submitted = asyncio.Event()

        def remote(**_kwargs):
            submitted.set()
            return future

        server = SimpleNamespace(generate=SimpleNamespace(remote=remote))
        client = LLMServerClient(
            OmegaConf.create({"actor_rollout_ref": {"rollout": {"name": "vllm", "full_determinism": False}}})
        )
        client._acquire_server = AsyncMock(return_value=("server", server))
        client._release_server = Mock()
        metrics = PhaseMetrics()
        cancelled_refs = []

        def cancel(ref, *, force):
            assert ref is future and force is False
            cancelled_refs.append(ref)
            future.set_exception(ray.exceptions.TaskCancelledError())

        monkeypatch.setattr(async_rpc.ray, "cancel", cancel)
        with rollout_request_scope(True, 1, metrics.record):
            task = asyncio.create_task(client.generate("request", prompt_ids=[1], sampling_params={}))
        await submitted.wait()
        if outcome == "completed":
            future.set_result(TokenOutput(token_ids=[2], extra_fields={"global_steps": 3}))
            result = await task
            assert result.token_ids == [2]
            assert result.extra_fields["min_global_steps"] == result.extra_fields["max_global_steps"] == 3
        elif outcome == "failed":
            future.set_exception(ValueError("student failed"))
            with pytest.raises(ValueError, match="student failed"):
                await task
        else:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert len(cancelled_refs) == int(outcome == "cancelled")
        client._release_server.assert_called_once_with("server")
        row = metrics.pop()["student_rpc"]
        assert row["count"] == row[f"{outcome}_count"] == 1

    asyncio.run(exercise())


def test_cutoff_during_routing_acquire_drains_and_releases_counter(monkeypatch):
    async def exercise():
        acquired = asyncio.get_running_loop().create_future()
        submitted = asyncio.Event()
        released = []

        def acquire(**_kwargs):
            submitted.set()
            return acquired

        load_balancer = SimpleNamespace(
            acquire_server=SimpleNamespace(remote=acquire),
            release_server=SimpleNamespace(remote=lambda *, server_id: released.append(server_id)),
        )
        client = LLMServerClient(OmegaConf.create({}), load_balancer_handle=load_balancer)
        monkeypatch.setattr(async_rpc.ray, "cancel", lambda *_a, **_k: pytest.fail("LB acquisition must drain"))
        with rollout_request_scope(True, 1):
            task = asyncio.create_task(client.generate("request", prompt_ids=[1], sampling_params={}))
        await submitted.wait()
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done() and not acquired.cancelled()
        acquired.set_result(("server", SimpleNamespace()))
        with pytest.raises(asyncio.CancelledError):
            await task
        assert released == ["server"]

    asyncio.run(exercise())


def test_routing_acquire_cleanup_deadline_remains_fatal(monkeypatch):
    async def exercise():
        acquired = asyncio.get_running_loop().create_future()
        submitted = asyncio.Event()

        def acquire(**_kwargs):
            submitted.set()
            return acquired

        load_balancer = SimpleNamespace(acquire_server=SimpleNamespace(remote=acquire))
        client = LLMServerClient(OmegaConf.create({}), load_balancer_handle=load_balancer)
        monkeypatch.setattr(async_rpc.ray, "cancel", lambda *_a, **_k: pytest.fail("LB acquisition must drain"))
        with rollout_request_scope(True, 0.01):
            task = asyncio.create_task(client.generate("request", prompt_ids=[1], sampling_params={}))
        await submitted.wait()
        task.cancel()
        with pytest.raises(RPCCancellationError, match="did not settle"):
            await task
        acquired.set_result(("server", None))

    asyncio.run(exercise())

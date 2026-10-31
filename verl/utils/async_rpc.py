"""Opt-in cancellation and phase accounting for rollout RPCs."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

import ray


class RPCCancellationError(RuntimeError):
    """A cancelled rollout could not establish that remote work had settled."""


@dataclass
class GenerationCutoffState:
    """Worker-local phase shared with one session, including its child tasks."""

    scoring_started: bool = False
    cutoff_requested: bool = False


_generation_cutoff: ContextVar[GenerationCutoffState | None] = ContextVar("generation_cutoff", default=None)


@contextmanager
def generation_cutoff_scope(state: GenerationCutoffState | None):
    token = _generation_cutoff.set(state)
    try:
        yield
    finally:
        _generation_cutoff.reset(token)


def mark_scoring_started() -> None:
    """Protect the whole scoring pipeline before its first await.

    Cutoff selection and this transition run on the same worker event loop.
    A session already selected for cancellation must not start scoring, even
    if custom agent code swallowed the original CancelledError.
    """
    state = _generation_cutoff.get()
    if state is not None:
        if state.cutoff_requested:
            raise asyncio.CancelledError
        state.scoring_started = True


class PhaseMetrics:
    """Small actor-local accumulator; snapshots preserve cancelled work too."""

    def __init__(self):
        self._phases: dict[str, dict[str, float]] = {}

    def record(self, phase: str, seconds: float, outcome: str) -> None:
        if outcome not in {"completed", "cancelled", "failed"}:
            raise ValueError(f"Unknown RPC outcome: {outcome}")
        seconds = float(seconds)
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("Phase duration must be finite and nonnegative")
        row = self._phases.setdefault(
            phase,
            {
                "seconds_sum": 0.0,
                "seconds_max": 0.0,
                "count": 0,
                "completed_count": 0,
                "cancelled_count": 0,
                "failed_count": 0,
            },
        )
        row["seconds_sum"] += seconds
        row["seconds_max"] = max(row["seconds_max"], seconds)
        row["count"] += 1
        row[f"{outcome}_count"] += 1

    def pop(self) -> dict[str, dict[str, float]]:
        snapshot, self._phases = self._phases, {}
        return snapshot


@dataclass
class _RequestScope:
    cancel_on_cutoff: bool = False
    cleanup_timeout_s: float = 120.0
    record_phase: Callable[[str, float, str], None] | None = None
    fatal_cleanup_error: RPCCancellationError | None = None


_request_scope: ContextVar[_RequestScope | None] = ContextVar("rollout_request_scope", default=None)


@contextmanager
def rollout_request_scope(
    cancel_on_cutoff: bool,
    cleanup_timeout_s: float,
    record_phase: Callable[[str, float, str], None] | None = None,
):
    """Limit cancellation changes to the selected training trajectory's task."""
    if cancel_on_cutoff and (not math.isfinite(cleanup_timeout_s) or cleanup_timeout_s <= 0):
        raise ValueError("RPC cancellation cleanup timeout must be positive and finite")
    scope = _RequestScope(cancel_on_cutoff, cleanup_timeout_s, record_phase)
    token = _request_scope.set(scope)
    try:
        yield
    finally:
        _request_scope.reset(token)
        # Custom agent loops may catch ordinary Exception and return a failed
        # or even successful output. An unsettled remote request remains fatal
        # across that boundary, including when the helper ran in a child task.
        if scope.fatal_cleanup_error is not None:
            parent = _request_scope.get()
            if parent is not None and parent.fatal_cleanup_error is None:
                parent.fatal_cleanup_error = scope.fatal_cleanup_error
            raise scope.fatal_cleanup_error


def record_rpc_phase(phase: str, seconds: float, outcome: str) -> None:
    scope = _request_scope.get()
    if scope is not None and scope.record_phase is not None:
        scope.record_phase(phase, seconds, outcome)


def raise_if_rpc_cleanup_failed() -> None:
    """Check custom-agent boundaries before starting more work or publishing."""
    scope = _request_scope.get()
    if scope is not None and scope.fatal_cleanup_error is not None:
        raise scope.fatal_cleanup_error


def _consume_result(future: asyncio.Future) -> None:
    if not future.cancelled():
        future.exception()


def _cleanup_failure(scope: _RequestScope, message: str) -> RPCCancellationError:
    error = RPCCancellationError(message)
    if scope.fatal_cleanup_error is None:
        scope.fatal_cleanup_error = error
    return scope.fatal_cleanup_error


async def await_cancellable_ray_ref(
    ref, *, cancel_remote: bool = True, on_cancel_result: Callable[[Any], None] | None = None
):
    """Cancel the actual async actor task and settle it before acknowledging cutoff.

    A cancelled local await does not cancel remote Ray work. Shield the waiter
    so it can still confirm completion after an explicit ``ray.cancel``. The
    default path deliberately retains ordinary Ray await behavior. Set
    ``cancel_remote=False`` for short side-effecting operations such as routing
    acquisition: retain their result and undo the side effect with the callback
    if the caller was cancelled, rather than interrupting the operation.
    """
    scope = _request_scope.get()
    if scope is None or not scope.cancel_on_cutoff:
        return await ref
    waiter = asyncio.ensure_future(ref)
    try:
        return await asyncio.shield(waiter)
    except asyncio.CancelledError:
        if cancel_remote:
            try:
                ray.cancel(ref, force=False)
            except Exception as error:
                waiter.add_done_callback(_consume_result)
                raise _cleanup_failure(scope, "Unable to request remote rollout RPC cancellation") from error
        try:
            result = await asyncio.wait_for(asyncio.shield(waiter), timeout=scope.cleanup_timeout_s)
        except ray.exceptions.TaskCancelledError:
            pass  # The remote async actor acknowledged cancellation.
        except asyncio.CancelledError as error:
            if not waiter.done():
                waiter.add_done_callback(_consume_result)
                raise _cleanup_failure(scope, "RPC cancellation cleanup was interrupted") from error
            if not waiter.cancelled():
                raise
        except TimeoutError as error:
            waiter.add_done_callback(_consume_result)
            raise _cleanup_failure(
                scope, f"Remote rollout RPC did not settle within {scope.cleanup_timeout_s:g}s after cancellation"
            ) from error
        else:
            if on_cancel_result is not None:
                try:
                    on_cancel_result(result)
                except Exception as error:
                    raise _cleanup_failure(scope, "Unable to release cancelled rollout RPC result") from error
        raise

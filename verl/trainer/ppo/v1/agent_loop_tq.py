# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

# TODO: move this file to verl.experimental.agent_loop after V1 is stable
"""TransferQueue adapter for AgentLoopManager and AgentLoopWorker"""

import asyncio
import logging
import math
import os
import time
from typing import Any
from uuid import uuid4

import ray
import torch
import transfer_queue as tq
from tensordict import NonTensorData, NonTensorStack, TensorDict

from verl.experimental.agent_loop import (
    AgentLoopManager,
    AgentLoopOutput,
    AgentLoopWorker,
    get_trajectory_info,
)
from verl.experimental.teacher_loop.errors import TeacherScoringError
from verl.trainer.distillation.config_utils import (
    opd_cancel_on_cutoff,
    opd_cleanup_timeout_s,
    opd_phase_timing_enabled,
    opd_wait_for_scoring_on_cutoff,
    should_skip_hidden_reward,
)
from verl.trainer.ppo.v1.repeat_judge_queue import RepeatQueueError, progress_interval_seconds
from verl.trainer.ppo.v1.utils import hidden_reward_was_skipped
from verl.utils.async_rpc import GenerationCutoffState, RPCCancellationError, generation_cutoff_scope
from verl.utils.ray_utils import auto_await
from verl.utils.swe_cutoff_debug import cutoff_debug_enabled, reconcile_cutoff_sessions, write_cutoff_debug_event
from verl.utils.swe_validation import positive_seconds, strict_validation_enabled
from verl.utils.tensordict_utils import get_non_tensor_data, list_of_dict_to_tensordict

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


def _repeat_debug_enabled() -> bool:
    value = os.getenv("SWE_AGENT_REPEAT_JUDGE_DEBUG", "0").lower()
    if value not in {"0", "1", "false", "true", "no", "yes", "off", "on"}:
        raise ValueError("SWE_AGENT_REPEAT_JUDGE_DEBUG must be a boolean")
    return value in {"1", "true", "yes", "on"}


def _repeat_debug_slow_seconds() -> float:
    value = float(os.getenv("SWE_AGENT_REPEAT_JUDGE_DEBUG_SLOW_SECONDS", "30"))
    if value < 0:
        raise ValueError("SWE_AGENT_REPEAT_JUDGE_DEBUG_SLOW_SECONDS must be nonnegative")
    return value


def apply_greedy_sampling_params(params: dict[str, Any]) -> None:
    params["top_p"] = 1.0
    params["top_k"] = -1
    params["temperature"] = 0


async def _settle_session_tasks(tasks: list[asyncio.Task[Any]]) -> list[BaseException]:
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return [result for result in results if isinstance(result, BaseException)]


def _summarize_opd_phase_metrics(snapshots: list[dict]) -> dict[str, float]:
    """Reduce actor-local snapshots without dropping cancelled/discarded work."""
    phases: dict[str, dict[str, float]] = {}
    for snapshot in snapshots:
        for phase, values in snapshot.items():
            merged = phases.setdefault(phase, {})
            for name, value in values.items():
                merged[name] = (
                    max(merged.get(name, 0.0), value) if name == "seconds_max" else merged.get(name, 0.0) + value
                )
    metrics = {}
    for phase, values in phases.items():
        prefix = f"opd_phase/{phase}"
        for name, value in values.items():
            metrics[f"{prefix}/{'seconds_total' if name == 'seconds_sum' else name}"] = float(value)
        metrics[f"{prefix}/seconds_mean"] = values.get("seconds_sum", 0.0) / max(values.get("count", 0), 1)
    return metrics


def _infrastructure_failure_from_extra_fields(extra_fields) -> bool:
    """Read the SWE infrastructure marker from either supported metadata layout."""
    extra_fields = extra_fields if isinstance(extra_fields, dict) else {}
    reward_extra_info = extra_fields.get("reward_extra_info", {})
    reward_extra_info = reward_extra_info if isinstance(reward_extra_info, dict) else {}
    value = reward_extra_info.get(
        "observed_infrastructure_failure",
        extra_fields.get(
            "observed_infrastructure_failure",
            reward_extra_info.get("infrastructure_failure", extra_fields.get("infrastructure_failure", 0)),
        ),
    )
    try:
        return int(value) == 1
    except (TypeError, ValueError, OverflowError):
        return False


@ray.remote
class AgentLoopWorkerTQ(AgentLoopWorker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        tq.init()
        self.repeat_coordinator = None
        self.repeat_finalizers = {}
        self.repeat_job_refs = {}
        self.dapo_cutoff_uids = set()
        self.dapo_cutoff_enabled = bool(
            self.config.get("algorithm", {}).get("filter_groups", {}).get("cutoff_when_full", False)
        )
        self.dapo_scoring_sessions: dict[tuple[str, str], list[tuple[asyncio.Task, GenerationCutoffState]]] = {}
        self.dapo_cleanup_timeouts: dict[tuple[str, str], float] = {}
        self.opd_cutoff_enabled = opd_cancel_on_cutoff(self.config)
        self.opd_scoring_sessions: dict[tuple[str, str], list[tuple[asyncio.Task, GenerationCutoffState]]] = {}
        self.dapo_publications = {}
        self.repeat_generation_sessions = {}
        self.repeat_debug = _repeat_debug_enabled()
        self.repeat_debug_slow_seconds = _repeat_debug_slow_seconds()
        self.repeat_debug_prompts: dict[tuple[str, str], float] = {}
        self.background_tasks = set()
        self.prompt_tasks: dict[str, dict[str, asyncio.Task[Any]]] = {"train": {}, "val": {}}
        self.cutoff_session_states: dict[tuple[str, str], dict[str, Any]] = {}

    async def configure_repeat_coordinator(self, coordinator):
        self.repeat_coordinator = coordinator

    def _repeat_debug_event(self, event: str, partition_id: str, uid: str, **fields) -> None:
        if not self.repeat_debug:
            return
        seconds = fields.get("seconds")
        slow = isinstance(seconds, (int, float)) and seconds >= self.repeat_debug_slow_seconds
        logger.log(
            logging.WARNING if slow else logging.INFO,
            "Repeat lifecycle event=%s partition=%s uid=%s slow=%s %s",
            event,
            partition_id,
            uid,
            slow,
            " ".join(f"{key}={value}" for key, value in fields.items()),
        )

    async def _finalize_repeat(self, ref, job_id, output, validate, kwargs):
        from verl.trainer.ppo.v1.repeat_judge_queue import finalize_output

        partition_id = "val" if validate else "train"
        uid = str(kwargs["uid"])
        started = time.monotonic()
        self._repeat_debug_event("finalizer_start", partition_id, uid, job_id=job_id)
        result_id, score, metadata = await ref
        if result_id != job_id:
            raise RuntimeError("Repeat result trajectory identity mismatch")
        finalize_output(output, score, metadata)
        await self._agent_loop_postprocess(output, validate, **kwargs)
        self._repeat_debug_event("finalizer_end", partition_id, uid, job_id=job_id,
                                 seconds=time.monotonic() - started, score=score)

    async def _join_repeat(self, partition_id, uid):
        tasks = self.repeat_finalizers.get((partition_id, uid), [])
        if not tasks:
            self._repeat_debug_event("join_repeat_no_jobs", partition_id, uid)
            return
        started = time.monotonic()
        self._repeat_debug_event("join_repeat_start", partition_id, uid, jobs=len(tasks))
        # Shield retained judgments from generation-cutoff cancellation. Publication
        # must settle before ReplayBuffer is allowed to materialize or clear a group.
        # asyncio.wait never cancels the settlement task if this caller is
        # cancelled, preserving the original shield/publication barrier.
        settlement = asyncio.ensure_future(_settle_session_tasks(tasks))
        interval = progress_interval_seconds() if self.repeat_debug else None
        while True:
            done, _ = await asyncio.wait([settlement], timeout=interval)
            if done:
                errors = settlement.result()
                break
            self._repeat_debug_event("join_repeat_progress", partition_id, uid,
                                     jobs=len(tasks), pending=sum(not task.done() for task in tasks),
                                     seconds=time.monotonic() - started)
        logger.info("Repeat reward barrier partition=%s uid=%s jobs=%d wait_seconds=%.3f",
                    partition_id, uid, len(tasks), time.monotonic() - started)
        self._repeat_debug_event("join_repeat_end", partition_id, uid, jobs=len(tasks),
                                 seconds=time.monotonic() - started, errors=len(errors))
        if errors:
            raise RepeatQueueError("Deferred repeat reward failed: " + str(errors[0])) from errors[0]

    def _discard_prompt_task(self, partition_id: str, uid: str, task: asyncio.Task[Any]) -> None:
        if self.prompt_tasks[partition_id].get(uid) is task:
            del self.prompt_tasks[partition_id][uid]

    def _ready_trajectory_keys(self, partition_id: str, keys: list[str]) -> list[str]:
        """Return published trajectory keys whose token fields are materialized.

        TransferQueue can expose a key/tag before an async field publication
        finishes. Cancellation must not mark such metadata-only keys as valid
        partial trajectories because the trainer cannot consume them.
        """
        ready, unready = [], []
        for key in keys:
            try:
                tq.kv_batch_get(keys=[key], partition_id=partition_id, select_fields=["input_ids"])
            except (KeyError, TypeError, ValueError, RuntimeError):
                unready.append(key)
            else:
                ready.append(key)
        if unready:
            tq.kv_clear(keys=unready, partition_id=partition_id)
            logger.warning(
                "Removed %d incomplete trajectory publication(s) after cancellation; sample keys: %s",
                len(unready),
                unready[:5],
            )
        return ready

    async def generate_sequences(self, batch: TensorDict) -> None:
        """Spawn agent loop for each sample in the batch without waiting for the results."""
        validate = batch["validate"] if "validate" in batch else False
        batch.pop("validate", None)
        config = self.config.actor_rollout_ref.rollout
        sampling_params = dict(
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repetition_penalty=1.0,
            logprobs=config.calculate_log_probs,
        )

        # override sampling params for validation
        if validate:
            sampling_params["top_p"] = config.val_kwargs.top_p
            sampling_params["top_k"] = config.val_kwargs.top_k
            sampling_params["temperature"] = config.val_kwargs.temperature

        if cutoff_debug_enabled():
            write_cutoff_debug_event(
                "rollout_debug_config", partition="val" if validate else "train",
                temperature=sampling_params["temperature"], top_p=sampling_params["top_p"],
                top_k=sampling_params["top_k"], rollout_n=config.val_kwargs.n if validate else config.n,
                training_completion_ratio=os.environ.get("SWE_AGENT_ROLLOUT_TRAINING_COMPLETION_RATIO_THRESHOLD"),
                validation_completion_ratio=os.environ.get("SWE_AGENT_ROLLOUT_VALIDATION_COMPLETION_RATIO_THRESHOLD"),
            )

        # by default, we assume it's a single turn agent
        if "agent_name" not in batch:
            default_agent_loop = config.agent.default_agent_loop
            batch["agent_name"] = NonTensorData(default_agent_loop)

        trajectory_info = await get_trajectory_info(batch["global_steps"], batch["index"], validate)

        # create background tasks for each sample in the batch
        for i in range(len(batch)):
            # TODO(wuxibin): add trace support
            trace_this_sample = False
            prompt = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    prompt[k] = v[i]
                elif isinstance(v, NonTensorStack):
                    prompt[k] = v[i].data
                elif isinstance(v, NonTensorData):
                    prompt[k] = v.data
                else:
                    logger.exception(f"Unsupported type {type(v)} for key {k}")

            uid = str(prompt["uid"])
            partition_id = "val" if validate else "train"
            if cutoff_debug_enabled():
                config = self.config.actor_rollout_ref.rollout
                expected = prompt.get("__rollout_n__", config.val_kwargs.n if validate else config.n)
                self.cutoff_session_states[(partition_id, uid)] = {
                    "expected": int(expected),
                    "started": set(),
                    "returned": set(),
                    "cancelled": set(),
                    "failed": set(),
                    "published": set(),
                    "published_trajectory_count": 0,
                    "session_summaries": {},
                    "cancel_reason": None,
                    "step": trajectory_info[i].get("step"),
                    "sample_index": trajectory_info[i].get("sample_index"),
                    "started_at": time.time(),
                }
            # “fire-and-forget” background tasks
            task = asyncio.create_task(
                self._run_prompt(prompt, sampling_params, trajectory=trajectory_info[i], trace=trace_this_sample)
            )
            self.background_tasks.add(task)
            task.add_done_callback(self.background_tasks.discard)
            self.prompt_tasks[partition_id][uid] = task
            task.add_done_callback(
                lambda completed, partition_id=partition_id, uid=uid: self._discard_prompt_task(
                    partition_id, uid, completed
                )
            )

    async def _await_tq_publication(self, partition_id: str, uid: str, operation):
        if partition_id != "train" or not (
            self.dapo_cutoff_enabled or getattr(self, "opd_cutoff_enabled", False)
        ):
            return await operation
        # Cancelling a local await does not stop a TransferQueue RPC already sent
        # to Ray. Shield and retain that operation until the cutoff cleanup barrier.
        task = asyncio.create_task(operation)
        self.dapo_publications.setdefault((partition_id, str(uid)), []).append(task)
        return await asyncio.shield(task)

    async def _settle_opd_publications(self, partition_id: str, uid: str) -> None:
        """An OPD cancellation acknowledgement is also a queue-write barrier."""
        if partition_id != "train" or not getattr(self, "opd_cutoff_enabled", False):
            return
        tasks = self.dapo_publications.get((partition_id, str(uid)), [])
        if not tasks:
            return
        settlement = asyncio.ensure_future(_settle_session_tasks(tasks))
        try:
            errors = await asyncio.wait_for(asyncio.shield(settlement), timeout=opd_cleanup_timeout_s(self.config))
        except asyncio.TimeoutError as error:
            # Keep observing dispatched writes, but never acknowledge an unsafe
            # cutoff or allow the trainer to proceed after the deadline.
            settlement.add_done_callback(lambda task: task.exception() if not task.cancelled() else None)
            raise RPCCancellationError("OPD TransferQueue publication cleanup exceeded its deadline") from error
        if errors:
            raise RPCCancellationError("OPD TransferQueue publication cleanup failed") from errors[0]

    async def _put_prompt_tag(self, *, key, partition_id, tag):
        return await self._await_tq_publication(
            partition_id, key, tq.async_kv_put(key=key, partition_id=partition_id, tag=tag)
        )

    @staticmethod
    async def _settle_dapo_cleanup(tasks, timeout_seconds: float, phase: str, *, allow_cancelled: bool = False):
        """Bound cancelled generation and dispatched writes, separately from normal scoring."""
        if not tasks:
            return
        _, pending = await asyncio.wait(tasks, timeout=timeout_seconds)
        if pending:
            raise RuntimeError(f"DAPO {phase} cleanup exceeded its deadline; refusing to acknowledge cutoff")
        errors = await _settle_session_tasks(tasks)
        if allow_cancelled:
            errors = [error for error in errors if not isinstance(error, asyncio.CancelledError)]
        if errors:
            raise RuntimeError(f"DAPO {phase} cleanup failed") from errors[0]

    async def _cancel_dapo_sequences(self, uids: list[str], cleanup_timeout_seconds: float = 300.0) -> int:
        tasks = {uid: self.prompt_tasks["train"][uid] for uid in uids
                 if uid in self.prompt_tasks["train"] and not self.prompt_tasks["train"][uid].done()}
        keys = {("train", str(uid)) for uid in tasks}
        self.dapo_cutoff_uids.update(keys)
        self.dapo_cleanup_timeouts.update(dict.fromkeys(keys, cleanup_timeout_seconds))
        try:
            generation_tasks = []
            draining_sessions = 0
            for uid, task in tasks.items():
                sessions = self.dapo_scoring_sessions.get(("train", str(uid)))
                if sessions is None:
                    # Not yet registered: no generation or scoring can have started.
                    task.cancel()
                    generation_tasks.append(task)
                    continue
                for session_task, state in sessions:
                    if session_task.done():
                        continue
                    if state.scoring_started:
                        draining_sessions += 1
                    else:
                        state.cutoff_requested = True
                        session_task.cancel()
                        generation_tasks.append(session_task)
            logger.info("DAPO generation cutoff groups=%d cancelled_generation_sessions=%d "
                        "draining_scoring_sessions=%d",
                        len(tasks), len(generation_tasks), draining_sessions)
            # Observe parent exceptions even if generation cleanup times out first.
            settlement = asyncio.ensure_future(_settle_session_tasks(list(tasks.values())))
            await self._settle_dapo_cleanup(
                generation_tasks, cleanup_timeout_seconds, "generation", allow_cancelled=True
            )
            # Parent tasks remain alive to finish started hidden rewards, deferred
            # judging, and publication. Their normal request timeouts still apply.
            # Watch dispatched writes while parents drain: a blocked publication
            # can itself hold a repeat finalizer (and therefore its parent) open.
            while not settlement.done():
                publications = [publication for key in keys
                                for publication in self.dapo_publications.get(key, []) if not publication.done()]
                if publications:
                    await self._settle_dapo_cleanup(
                        publications, cleanup_timeout_seconds, "TransferQueue publication"
                    )
                else:
                    await asyncio.wait([settlement], timeout=min(.1, cleanup_timeout_seconds))
            for result in await settlement:
                if isinstance(result, BaseException) and not isinstance(result, asyncio.CancelledError):
                    raise RuntimeError("DAPO rollout cancellation cleanup failed") from result
            items = (tq.kv_list("train") or {}).get("train", {})
            reward_errors = [uid for uid in tasks if items.get(uid, {}).get("status") == "reward_error"]
            if reward_errors:
                raise RuntimeError(f"DAPO retained reward processing failed for groups: {reward_errors[:5]}")
            await self._settle_dapo_cleanup(
                [asyncio.create_task(tq.async_kv_put(
                    key=uid, partition_id="train", tag={"status": "failure", "dapo_cutoff": True}
                )) for uid in tasks],
                cleanup_timeout_seconds, "cutoff tag publication",
            )
            return len(tasks)
        finally:
            self.dapo_cutoff_uids.difference_update(keys)
            for key in keys:
                self.dapo_cleanup_timeouts.pop(key, None)

    async def cancel_sequences(
        self, uids: list[str], validate: bool = False, *, dapo_cutoff: bool = False,
        cleanup_timeout_seconds: float = 300.0,
    ) -> int:
        """Cancel prompt groups and wait until their sessions release resources."""
        if dapo_cutoff:
            if validate:
                raise ValueError("DAPO cutoff applies only to training")
            return await self._cancel_dapo_sequences(uids, cleanup_timeout_seconds)
        partition_id = "val" if validate else "train"
        items = (tq.kv_list(partition_id) or {}).get(partition_id, {}) if self.repeat_coordinator is not None else {}
        tasks_by_uid = {
            uid: self.prompt_tasks[partition_id][uid]
            for uid in uids
            if uid in self.prompt_tasks[partition_id] and not self.prompt_tasks[partition_id][uid].done()
            and ((validate and strict_validation_enabled()) or (
                items.get(uid, {}).get("status") != "reward_pending"
                and not items.get(uid, {}).get("generation_complete", False)))
        }
        cutoff_states = {}
        for uid in tasks_by_uid:
            state = self.cutoff_session_states.get((partition_id, str(uid)))
            if state is not None:
                state["cancel_reason"] = "completion_ratio_cutoff"
                state["cutoff_requested_at"] = time.time()
                cutoff_states[uid] = state
        drain_scoring = opd_wait_for_scoring_on_cutoff(getattr(self, "config", None), validate=validate)
        for uid, task in tasks_by_uid.items():
            sessions = getattr(self, "opd_scoring_sessions", {}).get((partition_id, str(uid)))
            if drain_scoring and sessions is not None:
                # Do not cancel the parent gather: that would propagate into
                # hidden evaluation and queued/running teacher requests.
                # Phase transition and selection have no await between them.
                cancelled_sessions = 0
                draining_sessions = 0
                for session_task, state in sessions:
                    if session_task.done():
                        continue
                    if state.scoring_started:
                        draining_sessions += 1
                    else:
                        state.cutoff_requested = True
                        session_task.cancel()
                        cancelled_sessions += 1
                logger.info("OPD generation cutoff uid=%s cancelled_sessions=%d draining_scoring_sessions=%d",
                            uid, cancelled_sessions, draining_sessions)
            else:
                # Includes prompts not yet started/registered; no scoring can
                # exist before registration on this worker's event loop.
                task.cancel()
        if tasks_by_uid:
            results = await asyncio.gather(*tasks_by_uid.values(), return_exceptions=True)
            if getattr(self, "opd_cutoff_enabled", False) and not validate:
                for result in results:
                    if isinstance(result, BaseException) and not isinstance(result, asyncio.CancelledError):
                        raise RPCCancellationError("OPD rollout cancellation cleanup failed") from result

        # Preserve trajectories that were published before cancellation. They are valid
        # partial rollouts for completion-ratio reward prediction; groups with no
        # materialized sessions are evicted by ReplayBuffer.
        items = (tq.kv_list(partition_id) or {}).get(partition_id, {})
        for uid in tasks_by_uid:
            tag = items.get(uid, {})
            trajectory_keys = [key for key in items if key.startswith(f"{uid}_")]
            if tag.get("status") in {"pending", "running"}:
                trajectory_keys = self._ready_trajectory_keys(partition_id, trajectory_keys)
            state = cutoff_states.get(uid)
            if state is not None:
                # Observe the existing queue snapshot. Readiness checks below may
                # delete incomplete fields, so never add one for a terminal group.
                published_sessions = set()
                for key in trajectory_keys:
                    suffix = key[len(f"{uid}_"):].rsplit("_", 1)[0]
                    try:
                        published_sessions.add(int(suffix))
                    except (TypeError, ValueError):
                        continue
                self._record_cutoff_group_census(
                    partition_id, str(uid), state,
                    published=published_sessions,
                    trajectory_count=len(trajectory_keys),
                    terminal_status=tag.get("status"),
                    cutoff_applied=(
                        bool(tag.get("completion_ratio_cutoff")) or tag.get("status") in {"pending", "running"}
                    ),
                )
            self.cutoff_session_states.pop((partition_id, str(uid)), None)
            if tag.get("status") not in {"pending", "running"}:
                continue
            if trajectory_keys:
                trajectory_tags = []
                for key in trajectory_keys:
                    trajectory_tag = dict(items.get(key, {}))
                    trajectory_tag["completion_ratio_cutoff"] = True
                    trajectory_tags.append(trajectory_tag)
                tq.kv_batch_put(
                    keys=trajectory_keys,
                    partition_id=partition_id,
                    tags=trajectory_tags,
                )
            await tq.async_kv_put(
                key=uid,
                partition_id=partition_id,
                tag={"status": "failure", "completion_ratio_cutoff": True},
            )
        return len(tasks_by_uid)

    async def _run_prompt(self, prompt: dict, sampling_params: dict, trajectory: dict, trace: bool = False) -> None:
        """Spawn multiple agent loops in parallel according to rollout.n or rollout.val_kwargs.n."""
        uid, partition_id = prompt["uid"], "train" if not trajectory["validate"] else "val"
        prompt_started = time.monotonic()
        self.repeat_debug_prompts[(partition_id, str(uid))] = prompt_started
        self._repeat_debug_event("prompt_start", partition_id, str(uid))
        tasks = []
        self.repeat_finalizers[(partition_id, str(uid))] = []
        self.repeat_job_refs[(partition_id, str(uid))] = []
        config = self.config.actor_rollout_ref.rollout
        n = prompt.pop("__rollout_n__", config.n if not trajectory["validate"] else config.val_kwargs.n)
        do_sample = prompt.pop("__do_sample__", True)
        state_key = (partition_id, str(uid))
        dapo_drain_scoring = self.dapo_cutoff_enabled and not trajectory["validate"]
        opd_drain_scoring = opd_wait_for_scoring_on_cutoff(
            self.config, validate=trajectory["validate"]
        )
        drain_scoring = dapo_drain_scoring or opd_drain_scoring
        scoring_sessions = [] if drain_scoring else None
        scoring_controls = [GenerationCutoffState() for _ in range(n)] if drain_scoring else None
        session_state = self.cutoff_session_states.get(state_key) if cutoff_debug_enabled() else None
        if cutoff_debug_enabled() and session_state is None:
            session_state = {
                "expected": n,
                "started": set(),
                "returned": set(),
                "cancelled": set(),
                "failed": set(),
                "published": set(),
                "published_trajectory_count": 0,
                "session_summaries": {},
                "cancel_reason": None,
                "step": trajectory.get("step"),
                "sample_index": trajectory.get("sample_index"),
                "started_at": time.time(),
            }
            self.cutoff_session_states[state_key] = session_state
        self.repeat_generation_sessions[state_key] = (n, set())
        try:
            await self._put_prompt_tag(key=uid, partition_id=partition_id, tag={"status": "running"})
            self._repeat_debug_event("prompt_running", partition_id, str(uid))
            run_sampling_params = dict(sampling_params)
            if not trajectory["validate"] and not do_sample:
                apply_greedy_sampling_params(run_sampling_params)

            async def run_session(session_id: int):
                if session_state is not None:
                    session_state["started"].add(session_id)
                try:
                    control = scoring_controls[session_id] if scoring_controls is not None else None
                    with generation_cutoff_scope(control):
                        result = await self._run_agent_loop(
                            run_sampling_params,
                            trajectory=trajectory,
                            trace=trace,
                            session_id=session_id,
                            **({"_swe_cutoff_context": session_state} if session_state is not None else {}),
                            **({"_swe_repeat_deferred_supported": True} if self.repeat_coordinator is not None else {}),
                            **prompt,
                        )
                except asyncio.CancelledError:
                    if session_state is not None:
                        session_state["cancelled"].add(session_id)
                    raise
                except BaseException:
                    if session_state is not None:
                        session_state["failed"].add(session_id)
                    raise
                else:
                    if session_state is not None:
                        session_state["returned"].add(session_id)
                    return result

            tasks = []
            if scoring_sessions is not None:
                if dapo_drain_scoring:
                    self.dapo_scoring_sessions[state_key] = scoring_sessions
                if opd_drain_scoring:
                    if not hasattr(self, "opd_scoring_sessions"):
                        self.opd_scoring_sessions = {}
                    self.opd_scoring_sessions[state_key] = scoring_sessions
            for i in range(n):
                task = asyncio.create_task(run_session(i))
                tasks.append(task)
                if scoring_sessions is not None:
                    scoring_sessions.append((task, scoring_controls[i]))

            # Publish a terminal status only after every session settles, so no sibling can write after
            # ReplayBuffer clears a failed group.
            session_errors = await _settle_session_tasks(tasks)
            generation_seconds = time.monotonic() - prompt_started
            self._repeat_debug_event("generation_complete", partition_id, str(uid),
                                     seconds=generation_seconds, sessions=n, errors=len(session_errors))
            queue_error = next((error for error in session_errors if isinstance(error, RepeatQueueError)), None)
            if queue_error is not None:
                raise queue_error
            teacher_error = next((error for error in session_errors if isinstance(error, TeacherScoringError)), None)
            if teacher_error is not None:
                raise teacher_error
            cleanup_error = next((error for error in session_errors if isinstance(error, RPCCancellationError)), None)
            if cleanup_error is not None:
                raise cleanup_error
            if scoring_sessions is not None and any(state.cutoff_requested for _, state in scoring_sessions):
                # All retained scoring has settled. Reuse the existing partial
                # group/publication path without changing sample eligibility.
                raise asyncio.CancelledError
            await self._settle_opd_publications(partition_id, str(uid))
            if self.repeat_coordinator is not None:
                await self._put_prompt_tag(key=uid, partition_id=partition_id,
                                      tag={"status": "reward_pending", "generation_complete": True})
                self._repeat_debug_event("prompt_reward_pending", partition_id, str(uid),
                                         seconds=time.monotonic() - prompt_started)
                await self._join_repeat(partition_id, str(uid))
            if session_errors:
                for error in session_errors:
                    logger.error(
                        f"Error in _run_prompt for uid={uid}",
                        exc_info=(type(error), error, error.__traceback__),
                    )
                status = "failure"
            else:
                status = "finished"
            await self._put_prompt_tag(key=uid, partition_id=partition_id, tag={"status": status})
            if session_state is not None:
                session_state["terminal_status"] = status
            self._repeat_debug_event("prompt_finished", partition_id, str(uid), status=status,
                                     seconds=time.monotonic() - prompt_started)
        except asyncio.CancelledError:
            self._repeat_debug_event("prompt_cancelled", partition_id, str(uid),
                                     seconds=time.monotonic() - prompt_started)
            for task in tasks:
                task.cancel()
            cleanup_errors = await _settle_session_tasks(tasks) if tasks else []

            if getattr(self, "opd_cutoff_enabled", False) and partition_id == "train":
                cleanup_error = next(
                    (error for error in cleanup_errors
                     if isinstance(error, (TeacherScoringError, RPCCancellationError))),
                    None,
                )
                if cleanup_error is not None:
                    raise RPCCancellationError("OPD generation/teacher session cleanup failed") from cleanup_error
                # Settle writes that raced cancellation before inspecting or
                # marking partial output. No teacher-pending group is protected.
                await self._settle_opd_publications(partition_id, str(uid))

            if (partition_id, str(uid)) in self.dapo_cutoff_uids:
                cleanup_error = next(
                    (error for error in cleanup_errors if not isinstance(error, asyncio.CancelledError)), None
                )
                if cleanup_error is not None:
                    raise RuntimeError("DAPO generation session cleanup failed") from cleanup_error
                # The original input will be regenerated under the next policy.
                # Retained judgments/publications are settled in finally below.
                raise
            try:
                await self._join_repeat(partition_id, str(uid))
            except Exception as error:
                await self._put_prompt_tag(key=uid, partition_id=partition_id,
                                      tag={"status": "reward_error", "repeat_error": str(error)[:1000]})
                self._repeat_debug_event("prompt_terminal", partition_id, str(uid), status="reward_error",
                                         seconds=time.monotonic() - prompt_started,
                                         error_type=type(error).__name__)
                raise
            progress = self.repeat_generation_sessions.get((partition_id, str(uid)))
            if self.repeat_coordinator is not None and progress and len(progress[1]) == progress[0]:
                # Cutoff raced the final generation completion. Every rollout was
                # complete before cancellation; preserve the fully finalized group.
                await self._put_prompt_tag(key=uid, partition_id=partition_id, tag={"status": "finished"})
                self._repeat_debug_event("prompt_terminal", partition_id, str(uid), status="finished",
                                         seconds=time.monotonic() - prompt_started)
                return
            items = (tq.kv_list(partition_id) or {}).get(partition_id, {})
            trajectory_keys = [key for key in items if key.startswith(f"{uid}_")]
            trajectory_keys = self._ready_trajectory_keys(partition_id, trajectory_keys)
            if trajectory_keys:
                trajectory_tags = []
                for key in trajectory_keys:
                    trajectory_tag = dict(items.get(key, {}))
                    trajectory_tag["completion_ratio_cutoff"] = True
                    trajectory_tags.append(trajectory_tag)
                tq.kv_batch_put(
                    keys=trajectory_keys,
                    partition_id=partition_id,
                    tags=trajectory_tags,
                )
            await self._put_prompt_tag(
                key=uid,
                partition_id=partition_id,
                tag={"status": "failure", "completion_ratio_cutoff": True},
            )
            self._repeat_debug_event("prompt_terminal", partition_id, str(uid), status="failure",
                                     seconds=time.monotonic() - prompt_started,
                                     completion_ratio_cutoff=True)
            raise
        except Exception as e:
            logger.exception(f"Error in _run_prompt: {e}")
            self._repeat_debug_event("prompt_error", partition_id, str(uid),
                                     seconds=time.monotonic() - prompt_started,
                                     error_type=type(e).__name__)
            if tasks:
                await _settle_session_tasks(tasks)
            try:
                await self._join_repeat(partition_id, str(uid))
            except Exception as reward_error:
                e = reward_error
            # Reuse the existing fatal postprocessing-error channel. A missing
            # teacher target must stop training rather than silently refill the
            # prompt with an easier or shorter trajectory.
            status = "reward_error" if isinstance(
                e, (RepeatQueueError, TeacherScoringError, RPCCancellationError)
            ) else "failure"
            if session_state is not None:
                session_state["terminal_status"] = status
            tag = {"status": status}
            if isinstance(e, TeacherScoringError):
                tag["teacher_error"] = str(e)[:1000]
            elif isinstance(e, RPCCancellationError):
                tag["teacher_error"] = "OPD cancellation cleanup failed: " + str(e)[:1000]
            elif status == "reward_error":
                tag["repeat_error"] = str(e)[:1000]
            await self._put_prompt_tag(key=uid, partition_id=partition_id, tag=tag)
            self._repeat_debug_event("prompt_terminal", partition_id, str(uid), status=status,
                                     seconds=time.monotonic() - prompt_started,
                                     error_type=type(e).__name__)
        finally:
            if (partition_id, str(uid)) in self.dapo_cutoff_uids:
                # Deferred repeat judging belongs to retained scoring, not generation
                # cleanup. Never interrupt it or charge it to the cleanup deadline.
                await self._join_repeat(partition_id, str(uid))
                await self._settle_dapo_cleanup(
                    self.dapo_publications.get((partition_id, str(uid)), []),
                    self.dapo_cleanup_timeouts.get(state_key, 300.0), "TransferQueue publication",
                )
            await self._settle_opd_publications(partition_id, str(uid))
            if session_state is not None and session_state.get("cancel_reason") is None:
                self._record_cutoff_group_census(
                    partition_id, str(uid), session_state,
                    published=session_state["published"],
                    trajectory_count=session_state["published_trajectory_count"],
                    terminal_status=session_state.get("terminal_status"),
                )
            self.repeat_debug_prompts.pop((partition_id, str(uid)), None)
            self.repeat_finalizers.pop((partition_id, str(uid)), None)
            self.repeat_job_refs.pop((partition_id, str(uid)), None)
            self.dapo_publications.pop((partition_id, str(uid)), None)
            self.repeat_generation_sessions.pop((partition_id, str(uid)), None)
            self.cutoff_session_states.pop((partition_id, str(uid)), None)
            getattr(self, "opd_scoring_sessions", {}).pop(state_key, None)
            getattr(self, "dapo_scoring_sessions", {}).pop(state_key, None)

    @staticmethod
    def _record_cutoff_group_census(
        partition_id: str,
        uid: str,
        state: dict[str, Any],
        *,
        published: set[int],
        trajectory_count: int,
        terminal_status: str | None = None,
        cutoff_applied: bool = False,
    ) -> None:
        if state.get("census_written"):
            return
        try:
            state["census_written"] = True
            counts = reconcile_cutoff_sessions(
                state.get("expected", 0),
                started=state.get("started", ()),
                returned=state.get("returned", ()),
                cancelled=state.get("cancelled", ()),
                failed=state.get("failed", ()),
                published=published,
            )
            threshold_name = (
                "SWE_AGENT_ROLLOUT_VALIDATION_COMPLETION_RATIO_THRESHOLD"
                if partition_id == "val"
                else "SWE_AGENT_ROLLOUT_TRAINING_COMPLETION_RATIO_THRESHOLD"
            )
            cutoff_requested = state.get("cancel_reason") == "completion_ratio_cutoff"
            cancelled = bool(counts["cancelled_sessions"])
            if cutoff_requested:
                event = "prompt_group_cutoff_census"
            elif cancelled:
                event = "prompt_group_cancelled_census"
            else:
                event = "prompt_group_settled_census"
            write_cutoff_debug_event(
                event,
                partition=partition_id,
                uid=uid,
                global_step=state.get("step"),
                sample_index=state.get("sample_index"),
                cutoff_reason=state.get("cancel_reason"),
                cutoff_requested=int(cutoff_requested),
                cutoff_applied=int(cutoff_applied),
                terminal_status=terminal_status,
                threshold=os.environ.get(threshold_name),
                cutoff_requested_at_unix_seconds=state.get("cutoff_requested_at"),
                group_elapsed_seconds=round(time.time() - state.get("started_at", time.time()), 3),
                published_trajectory_count=int(trajectory_count),
                **counts,
            )
            for session_id, summary in state.get("session_summaries", {}).items():
                write_cutoff_debug_event(
                    "session_behavior_summary",
                    partition=partition_id,
                    uid=uid,
                    global_step=state.get("step"),
                    sample_index=state.get("sample_index"),
                    session_id=session_id,
                    cutoff_requested=int(cutoff_requested),
                    cutoff_group=int(cutoff_applied),
                    **summary,
                )
            logger.info(
                "Cutoff session census partition=%s uid=%s expected=%d started=%d returned=%d "
                "cancelled=%d failed=%d published=%d not_started=%d unpublished=%d trajectories=%d",
                partition_id, uid, counts["expected_sessions"], counts["started_sessions"],
                counts["returned_sessions"], counts["cancelled_sessions"], counts["failed_sessions"],
                counts["published_sessions"], counts["not_started_sessions"],
                counts["returned_without_publication_sessions"], trajectory_count,
            )
        except Exception:
            logger.warning("Unable to summarize cutoff group; rollout behavior is unchanged", exc_info=True)
            state["census_written"] = True

    async def _agent_loop_postprocess(
        self, output: AgentLoopOutput | list[AgentLoopOutput], validate, **kwargs
    ) -> None:
        """Put agent loop outputs into TransferQueue."""
        kwargs.pop("_swe_repeat_deferred_supported", None)
        kwargs.pop("_swe_cutoff_context", None)
        uid, session_id = kwargs["uid"], kwargs["session_id"]
        partition_id = "val" if validate else "train"
        postprocess_started = time.monotonic()
        self._repeat_debug_event("postprocess_entered", partition_id, str(uid), session=session_id)
        outputs = output if isinstance(output, list) else [output]
        if (outputs and self.repeat_coordinator is not None
                and not outputs[-1].extra_fields.get("_swe_repeat_finalized")):
            expected, completed = self.repeat_generation_sessions[(partition_id, str(uid))]
            completed.add(session_id)
            if len(completed) == expected:
                # Mark completion before bounded admission: a full queue is not a
                # generation straggler and must not bias completion-ratio cutoff.
                await self._put_prompt_tag(key=uid, partition_id=partition_id,
                                      tag={"status": "running", "generation_complete": True})
                self._repeat_debug_event("prompt_generation_complete", partition_id, str(uid),
                                         sessions=len(completed), expected=expected)
        jobs = [o.extra_fields.get("_swe_repeat_job") for o in outputs]
        if any(job is not None for job in jobs):
            if len(outputs) != 1 or self.repeat_coordinator is None:
                raise RepeatQueueError("Deferred repeat rewards require a configured coordinator and one SWE output")
            job = outputs[0].extra_fields.pop("_swe_repeat_job")
            job_id = f"{kwargs['global_steps']}:{'val' if validate else 'train'}:{uid}:{session_id}:{uuid4().hex}"
            # Admission and inference both belong to the coordinator. Do not await
            # either here: even a full queue must release this generation session.
            try:
                ref = self.repeat_coordinator.submit.remote(job_id, job)
            except Exception as error:
                raise RepeatQueueError("Repeat queue handoff failed") from error
            self.repeat_job_refs.setdefault((partition_id, str(uid)), []).append(ref)
            self._repeat_debug_event("repeat_job_submitted", partition_id, str(uid),
                                     session=session_id, job_id=job_id)
            task = asyncio.create_task(self._finalize_repeat(ref, job_id, outputs[0], validate, kwargs))
            self.repeat_finalizers[("val" if validate else "train", str(uid))].append(task)
            self._repeat_debug_event("postprocess_handoff_complete", partition_id, str(uid),
                                     session=session_id, seconds=time.monotonic() - postprocess_started)
            return
        if not outputs:
            logger.warning(f"Empty output for prompt {uid}_{session_id}")
            return

        if should_skip_hidden_reward(self.config, validate=validate):
            # The SWE loop must have skipped the evaluator itself. Never fall
            # back to a reward worker if an outdated agent omitted the marker.
            if any(not hidden_reward_was_skipped(o.extra_fields) or o.reward_score != 0.0 for o in outputs):
                raise RuntimeError(
                    "skip_hidden_reward requires SWE outputs marked hidden_reward_skipped with a zero "
                    "transport placeholder; update the SWE source checkout"
                )
        else:
            await self._compute_score(outputs, kwargs=kwargs)

        final_output = outputs[-1]
        # TODO: Support output:list[AgentLoopOutput]
        await self._compute_teacher_logprobs(
            final_output,
            prompt_ids=final_output.prompt_ids,
            response_ids=final_output.response_ids,
            validate=validate,
            sample_kwargs=kwargs,
        )

        if final_output.reward_score is not None:
            for output in outputs[:-1]:
                output.reward_score = final_output.reward_score
                output.extra_fields["reward_extra_info"] = final_output.extra_fields["reward_extra_info"]

        # NOTE: agent loop may has multiple outputs, put each output into TransferQueue.
        # key format: {uid}_{session_id}_{index}
        # - uid: raw prompt uid from dataset
        # - session_id: session id for rollout.n sampling
        # - index: index of agent loop output
        keys, fields, tags = [], [], []
        infrastructure_metadata_count = 0
        infrastructure_failure_count = 0
        for i, output in enumerate(outputs):
            prompts = torch.tensor(output.prompt_ids, dtype=torch.int64)
            responses = torch.tensor(output.response_ids, dtype=torch.int64)
            input_ids = torch.cat([prompts, responses], dim=0)
            attention_mask = torch.ones_like(input_ids, dtype=torch.int64)
            multi_modal_inputs = self._compute_multi_modal_inputs(output, input_ids)
            position_ids = self._compute_position_ids(
                input_ids.unsqueeze(0), attention_mask.unsqueeze(0), multi_modal_inputs
            ).squeeze(0)

            keys.append(f"{uid}_{session_id}_{i}")
            field = output.as_dict()
            field.update(kwargs)
            # do not store raw image/video
            field.pop("multi_modal_data", None)
            # TODO: uniform response_mask and loss_mask
            field["loss_mask"] = field["response_mask"]
            field["input_ids"] = input_ids
            field["position_ids"] = position_ids
            field["multi_modal_inputs"] = multi_modal_inputs
            fields.append(field)
            prompt_len, response_len = field["prompts"].size(0), field["responses"].size(0)
            extra_fields = field["extra_fields"]
            if isinstance(extra_fields, dict):
                reward_extra_info = extra_fields.get("reward_extra_info", {})
                if (
                    "observed_infrastructure_failure" in extra_fields
                    or "infrastructure_failure" in extra_fields
                    or (
                        isinstance(reward_extra_info, dict)
                        and (
                            "observed_infrastructure_failure" in reward_extra_info
                            or "infrastructure_failure" in reward_extra_info
                        )
                    )
                ):
                    infrastructure_metadata_count += 1
            infrastructure_failure = _infrastructure_failure_from_extra_fields(extra_fields)
            reward_info = extra_fields.get("reward_extra_info", {}) if isinstance(extra_fields, dict) else {}
            evaluator_group_excluded = bool(
                isinstance(reward_info, dict) and reward_info.get("evaluator_group_excluded", False)
            )
            infrastructure_failure_count += int(infrastructure_failure)
            tags.append(
                {
                    "status": "success",
                    "prompt_len": prompt_len,
                    "response_len": response_len,
                    "seq_len": prompt_len + response_len,
                    # These tags are used for off-policy staleness control, if a trajectory
                    # spans too many global steps, we need to filter it out.
                    # global_steps: which global steps this sample is from dataloader
                    "global_steps": kwargs["global_steps"],
                    # min_global_steps: start generation model weights version of this trajectory
                    "min_global_steps": field["extra_fields"].get("min_global_steps"),
                    # max_global_steps: end generation model weights version of this trajectory
                    "max_global_steps": field["extra_fields"].get("max_global_steps"),
                    # ReplayBuffer needs this before sampling so invalid groups
                    # do not contribute to completion-ratio accounting.
                    "infrastructure_failure": infrastructure_failure,
                    "evaluator_group_excluded": evaluator_group_excluded,
                    "unverified_trajectory": evaluator_group_excluded,
                }
            )
            if validate and strict_validation_enabled():
                # Publish the same strict evidence the final metric uses, before
                # the sampler counts a task toward the completion threshold.
                from verl.trainer.ppo.v1.utils import validation_reward_metadata

                info = validation_reward_metadata(extra_fields)
                for name in ("evaluation_valid", "hidden_pass", "evaluator_invalid_reason",
                             "observed_infrastructure_failure", "evaluator_failure_excluded",
                             "partial_hidden_reward_counts_complete", "rebench_failure_kind",
                             "hidden_failure_phase", "hidden_verifier_staging_ok", "hidden_verifier_setup_ok"):
                    tags[-1][name] = info.get(name)

        if infrastructure_metadata_count or infrastructure_failure_count:
            logger.info(
                "V1 infrastructure marker propagation: uid=%s session=%s validation=%s outputs=%d "
                "metadata_outputs=%d failure_outputs=%d",
                uid,
                session_id,
                validate,
                len(outputs),
                infrastructure_metadata_count,
                infrastructure_failure_count,
            )

        await self._await_tq_publication(partition_id, str(uid), tq.async_kv_batch_put(
            keys=keys,
            fields=list_of_dict_to_tensordict(fields),
            tags=tags,
            partition_id="train" if not validate else "val",
        ))
        cutoff_state = self.cutoff_session_states.get((partition_id, str(uid)))
        if cutoff_state is not None:
            cutoff_state["published"].add(int(session_id))
            cutoff_state["published_trajectory_count"] += len(keys)
        self._repeat_debug_event("trajectory_published", partition_id, str(uid),
                                 session=session_id, outputs=len(outputs),
                                 seconds=time.monotonic() - postprocess_started)


class AgentLoopManagerTQ(AgentLoopManager):
    def __init__(self, *args, **kwargs):
        self.agent_loop_workers_class = AgentLoopWorkerTQ
        super().__init__(*args, **kwargs)

    @classmethod
    @auto_await
    async def create(cls, *args, **kwargs):
        """Create agent loop manager."""
        instance = cls(*args, **kwargs)
        from verl.trainer.ppo.v1.repeat_judge_queue import (
            RepeatJudgeCoordinatorActor,
            deferred_enabled,
            validate_deferred_config,
        )
        instance.repeat_coordinator = None
        if deferred_enabled():
            from recipe.swe_agent.repetition_reward import RepeatConfig
            validate_deferred_config(instance.config)
            instance.repeat_coordinator = RepeatJudgeCoordinatorActor.remote(RepeatConfig.from_env())
            try:
                await instance.repeat_coordinator.ready.remote()
            except BaseException:
                ray.kill(instance.repeat_coordinator)
                raise
        try:
            await instance._init_agent_loop_workers()
            if instance.repeat_coordinator is not None:
                await asyncio.gather(*[worker.configure_repeat_coordinator.remote(instance.repeat_coordinator)
                                       for worker in instance.agent_loop_workers])
        except BaseException:
            instance.close_repeat_coordinator()
            raise
        return instance

    def generate_sequences(self, prompts: TensorDict) -> None:
        """
        Dispatch input batch to agent loop workers without blocking. Workers should put agent loop outputs
        into TransferQueue once an agent loop finished.

        Args:
            prompts (TensorDict): Input batch from train or validation dataset.
        """
        chunkes = prompts.chunk(len(self.agent_loop_workers))
        refs = [worker.generate_sequences.remote(chunk)
                for worker, chunk in zip(self.agent_loop_workers, chunkes, strict=False)]
        if strict_validation_enabled() and get_non_tensor_data(prompts, "validate", False):
            try:
                ray.get(refs, timeout=positive_seconds("SWE_AGENT_VALIDATION_CLEANUP_TIMEOUT_SECONDS", 300))
            except ray.exceptions.GetTimeoutError as error:
                raise RuntimeError("Validation dispatch was not acknowledged; refusing to resume training") from error
        else:
            ray.get(refs)

    def pop_opd_phase_metrics(self) -> dict[str, float]:
        """Collect interval counters from workers and the shared scoring actors.

        Collect after sampling and cutoff cleanup, independently of selected
        trajectory rows. Thus cancelled requests and infrastructure refills also
        contribute. Async training reports work settled since the last snapshot.
        """
        if not opd_phase_timing_enabled(self.config):
            return {}
        refs = [worker.pop_opd_phase_metrics.remote() for worker in self.agent_loop_workers]
        for client in (self.teacher_client or {}).values():
            handle = getattr(client, "_scoring_handle", None)
            if handle is not None:
                refs.append(handle.pop_opd_phase_metrics.remote())
        try:
            snapshots = ray.get(refs, timeout=opd_cleanup_timeout_s(self.config))
        except ray.exceptions.GetTimeoutError as error:
            raise RuntimeError("OPD phase timing snapshot exceeded its deadline") from error
        metrics = _summarize_opd_phase_metrics(snapshots)
        logger.info("OPD phase timings (concurrent seconds; includes cancelled/failed work): %s", metrics)
        return metrics

    def cancel_sequences(
        self, uids: list[str], validate: bool = False, *, dapo_cutoff: bool = False,
        cleanup_timeout_seconds: float = 300.0,
    ) -> int:
        """Cancel the requested prompt groups on whichever workers own them."""
        if not uids:
            return 0
        if dapo_cutoff and (validate or not math.isfinite(cleanup_timeout_seconds) or cleanup_timeout_seconds <= 0):
            raise ValueError("DAPO training cutoff requires a positive finite cleanup timeout")
        started = time.monotonic()
        interval = progress_interval_seconds()
        cancellation_kwargs = (
            {"dapo_cutoff": True, "cleanup_timeout_seconds": cleanup_timeout_seconds} if dapo_cutoff else {}
        )
        pending = [worker.cancel_sequences.remote(uids, validate=validate, **cancellation_kwargs)
                   for worker in self.agent_loop_workers]
        logger.info("Rollout cancellation/drain start partition=%s groups=%d workers=%d",
                    "val" if validate else "train", len(uids), len(pending))
        cancelled = 0
        snapshot_ref = None
        last_progress = started
        cleanup_timeout = (positive_seconds("SWE_AGENT_VALIDATION_CLEANUP_TIMEOUT_SECONDS", 300)
                           if validate and strict_validation_enabled() else None)
        if dapo_cutoff:
            # Workers bound generation cancellation and publication cleanup.
            # The manager also waits for retained rewards under their own timeouts.
            cleanup_timeout = None
        opd_cutoff = opd_cancel_on_cutoff(getattr(self, "config", None), validate=validate)
        if opd_cutoff and not dapo_cutoff:
            cleanup_timeout = opd_cleanup_timeout_s(self.config)
        if opd_wait_for_scoring_on_cutoff(getattr(self, "config", None), validate=validate) and not dapo_cutoff:
            # Hidden evaluation and teacher queue/HTTP requests retain their
            # own timeouts. The RPC cancellation helper still bounds generation
            # acknowledgement; scoring/publication are normal work, not cleanup.
            cleanup_timeout = None
        cutoff_name = "OPD" if opd_cutoff else "DAPO" if dapo_cutoff else "Validation"
        while pending:
            remaining = cleanup_timeout - (time.monotonic() - started) if cleanup_timeout else interval
            if remaining <= 0:
                raise RuntimeError(
                    f"{cutoff_name} cancellation was not acknowledged "
                    "within the cleanup deadline; refusing to resume training while workers may still publish"
                )
            done, pending = ray.wait(pending, num_returns=1, timeout=min(interval, remaining))
            cancelled += sum(ray.get(done))
            now = time.monotonic()
            if now - last_progress >= interval:
                logger.info("Rollout cancellation/drain progress partition=%s pending_workers=%d "
                            "cancelled_groups=%d wait_seconds=%.3f",
                            "val" if validate else "train", len(pending), cancelled, now - started)
                last_progress = now
                if self.repeat_coordinator is not None:
                    # Snapshot diagnostics must never introduce another blocking
                    # ray.get or obscure errors from the cancellation workers.
                    try:
                        if snapshot_ref is None:
                            snapshot_ref = self.repeat_coordinator.snapshot.remote()
                        ready, _ = ray.wait([snapshot_ref], timeout=0)
                        if ready:
                            logger.info("Repeat judge cancellation/drain queue snapshot %s", ray.get(snapshot_ref))
                            snapshot_ref = None
                    except Exception as error:
                        logger.warning("Repeat judge drain snapshot unavailable type=%s", type(error).__name__)
                        snapshot_ref = None
        logger.info("Rollout cancellation/drain end partition=%s cancelled_groups=%d wait_seconds=%.3f",
                    "val" if validate else "train", cancelled, time.monotonic() - started)
        return cancelled

    def close_repeat_coordinator(self):
        if self.repeat_coordinator is not None:
            ray.kill(self.repeat_coordinator)
            self.repeat_coordinator = None

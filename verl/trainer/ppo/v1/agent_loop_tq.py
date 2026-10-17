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
from verl.trainer.ppo.v1.repeat_judge_queue import RepeatQueueError
from verl.utils.ray_utils import auto_await
from verl.utils.tensordict_utils import list_of_dict_to_tensordict

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
        self.repeat_generation_sessions = {}
        self.repeat_debug = _repeat_debug_enabled()
        self.repeat_debug_slow_seconds = _repeat_debug_slow_seconds()
        self.repeat_debug_prompts: dict[tuple[str, str], float] = {}
        self.background_tasks = set()
        self.prompt_tasks: dict[str, dict[str, asyncio.Task[Any]]] = {"train": {}, "val": {}}

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
        errors = await asyncio.shield(_settle_session_tasks(tasks))
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

            # “fire-and-forget” background tasks
            task = asyncio.create_task(
                self._run_prompt(prompt, sampling_params, trajectory=trajectory_info[i], trace=trace_this_sample)
            )
            self.background_tasks.add(task)
            task.add_done_callback(self.background_tasks.discard)
            uid = str(prompt["uid"])
            partition_id = "val" if validate else "train"
            self.prompt_tasks[partition_id][uid] = task
            task.add_done_callback(
                lambda completed, partition_id=partition_id, uid=uid: self._discard_prompt_task(
                    partition_id, uid, completed
                )
            )

    async def cancel_sequences(self, uids: list[str], validate: bool = False) -> int:
        """Cancel prompt groups and wait until their sessions release resources."""
        partition_id = "val" if validate else "train"
        items = (tq.kv_list(partition_id) or {}).get(partition_id, {}) if self.repeat_coordinator is not None else {}
        tasks_by_uid = {
            uid: self.prompt_tasks[partition_id][uid]
            for uid in uids
            if uid in self.prompt_tasks[partition_id] and not self.prompt_tasks[partition_id][uid].done()
            and items.get(uid, {}).get("status") != "reward_pending"
            and not items.get(uid, {}).get("generation_complete", False)
        }
        for task in tasks_by_uid.values():
            task.cancel()
        if tasks_by_uid:
            await asyncio.gather(*tasks_by_uid.values(), return_exceptions=True)

        # Preserve trajectories that were published before cancellation. They are valid
        # partial rollouts for completion-ratio reward prediction; groups with no
        # materialized sessions are evicted by ReplayBuffer.
        items = (tq.kv_list(partition_id) or {}).get(partition_id, {})
        for uid in tasks_by_uid:
            tag = items.get(uid, {})
            if tag.get("status") not in {"pending", "running"}:
                continue
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
        await tq.async_kv_put(key=uid, partition_id=partition_id, tag={"status": "running"})
        self._repeat_debug_event("prompt_running", partition_id, str(uid))
        tasks = []
        self.repeat_finalizers[(partition_id, str(uid))] = []
        try:
            # NOTE: user can dynamically adjust n for each sample here, e.g according to task difficulty.
            config = self.config.actor_rollout_ref.rollout
            n = prompt.pop("__rollout_n__", config.n if not trajectory["validate"] else config.val_kwargs.n)
            do_sample = prompt.pop("__do_sample__", True)
            self.repeat_generation_sessions[(partition_id, str(uid))] = (n, set())

            run_sampling_params = dict(sampling_params)
            if not trajectory["validate"] and not do_sample:
                apply_greedy_sampling_params(run_sampling_params)

            tasks = []
            for i in range(n):
                task = asyncio.create_task(
                    self._run_agent_loop(
                        run_sampling_params, trajectory=trajectory, trace=trace, session_id=i,
                        **({"_swe_repeat_deferred_supported": True} if self.repeat_coordinator is not None else {}),
                        **prompt
                    )
                )
                tasks.append(task)

            # Publish a terminal status only after every session settles, so no sibling can write after
            # ReplayBuffer clears a failed group.
            session_errors = await _settle_session_tasks(tasks)
            generation_seconds = time.monotonic() - prompt_started
            self._repeat_debug_event("generation_complete", partition_id, str(uid),
                                     seconds=generation_seconds, sessions=n, errors=len(session_errors))
            queue_error = next((error for error in session_errors if isinstance(error, RepeatQueueError)), None)
            if queue_error is not None:
                raise queue_error
            if self.repeat_coordinator is not None:
                await tq.async_kv_put(key=uid, partition_id=partition_id,
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
            await tq.async_kv_put(key=uid, partition_id=partition_id, tag={"status": status})
            self._repeat_debug_event("prompt_finished", partition_id, str(uid), status=status,
                                     seconds=time.monotonic() - prompt_started)
        except asyncio.CancelledError:
            self._repeat_debug_event("prompt_cancelled", partition_id, str(uid),
                                     seconds=time.monotonic() - prompt_started)
            for task in tasks:
                task.cancel()
            if tasks:
                await _settle_session_tasks(tasks)

            try:
                await self._join_repeat(partition_id, str(uid))
            except Exception as error:
                await tq.async_kv_put(key=uid, partition_id=partition_id,
                                      tag={"status": "reward_error", "repeat_error": str(error)[:1000]})
                self._repeat_debug_event("prompt_terminal", partition_id, str(uid), status="reward_error",
                                         seconds=time.monotonic() - prompt_started,
                                         error_type=type(error).__name__)
                raise
            progress = self.repeat_generation_sessions.get((partition_id, str(uid)))
            if self.repeat_coordinator is not None and progress and len(progress[1]) == progress[0]:
                # Cutoff raced the final generation completion. Every rollout was
                # complete before cancellation; preserve the fully finalized group.
                await tq.async_kv_put(key=uid, partition_id=partition_id, tag={"status": "finished"})
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
            await tq.async_kv_put(
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
            status = "reward_error" if isinstance(e, RepeatQueueError) else "failure"
            tag = {"status": status}
            if status == "reward_error":
                tag["repeat_error"] = str(e)[:1000]
            await tq.async_kv_put(key=uid, partition_id=partition_id, tag=tag)
            self._repeat_debug_event("prompt_terminal", partition_id, str(uid), status=status,
                                     seconds=time.monotonic() - prompt_started,
                                     error_type=type(e).__name__)
        finally:
            self.repeat_debug_prompts.pop((partition_id, str(uid)), None)
            self.repeat_finalizers.pop((partition_id, str(uid)), None)
            self.repeat_generation_sessions.pop((partition_id, str(uid)), None)

    async def _agent_loop_postprocess(
        self, output: AgentLoopOutput | list[AgentLoopOutput], validate, **kwargs
    ) -> None:
        """Put agent loop outputs into TransferQueue."""
        kwargs.pop("_swe_repeat_deferred_supported", None)
        uid, session_id = kwargs["uid"], kwargs["session_id"]
        partition_id = "val" if validate else "train"
        postprocess_started = time.monotonic()
        self._repeat_debug_event("postprocess_entered", partition_id, str(uid), session=session_id)
        outputs = output if isinstance(output, list) else [output]
        if outputs and self.repeat_coordinator is not None and not outputs[-1].extra_fields.get("_swe_repeat_finalized"):
            expected, completed = self.repeat_generation_sessions[(partition_id, str(uid))]
            completed.add(session_id)
            if len(completed) == expected:
                # Mark completion before bounded admission: a full queue is not a
                # generation straggler and must not bias completion-ratio cutoff.
                await tq.async_kv_put(key=uid, partition_id=partition_id,
                                      tag={"status": "running", "generation_complete": True})
                self._repeat_debug_event("prompt_generation_complete", partition_id, str(uid),
                                         sessions=len(completed), expected=expected)
        jobs = [o.extra_fields.get("_swe_repeat_job") for o in outputs]
        if any(job is not None for job in jobs):
            if len(outputs) != 1 or self.repeat_coordinator is None:
                raise RepeatQueueError("Deferred repeat rewards require a configured coordinator and one SWE output")
            job = outputs[0].extra_fields.pop("_swe_repeat_job")
            job_id = f"{kwargs['global_steps']}:{'val' if validate else 'train'}:{uid}:{session_id}:{uuid4().hex}"
            self._repeat_debug_event("repeat_reservation_requested", partition_id, str(uid),
                                     session=session_id, job_id=job_id)
            # Shield admission so cancellation cannot leak a reservation. If cutoff
            # arrives during backpressure, finish handoff and retain this completed output.
            reservation = asyncio.ensure_future(self.repeat_coordinator.reserve.remote(job_id))
            cancelled = False
            try:
                try:
                    admission = await asyncio.shield(reservation)
                except asyncio.CancelledError:
                    admission = await reservation
                    cancelled = True
            except Exception as error:
                raise RepeatQueueError("Repeat queue admission failed") from error
            self._repeat_debug_event("repeat_reservation_acquired", partition_id, str(uid),
                                     session=session_id, job_id=job_id, seconds=admission)
            try:
                ref = self.repeat_coordinator.evaluate.remote(job_id, job, admission)
            except Exception as error:
                try:
                    await self.repeat_coordinator.abandon.remote(job_id)
                except Exception:
                    logger.exception("Failed to release repeat reservation %s", job_id)
                raise RepeatQueueError("Repeat queue handoff failed") from error
            self._repeat_debug_event("judge_rpc_submitted", partition_id, str(uid),
                                     session=session_id, job_id=job_id)
            task = asyncio.create_task(self._finalize_repeat(ref, job_id, outputs[0], validate, kwargs))
            self.repeat_finalizers[("val" if validate else "train", str(uid))].append(task)
            if cancelled:
                raise asyncio.CancelledError
            self._repeat_debug_event("postprocess_handoff_complete", partition_id, str(uid),
                                     session=session_id, seconds=time.monotonic() - postprocess_started)
            return
        if not outputs:
            logger.warning(f"Empty output for prompt {uid}_{session_id}")
            return

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

        await tq.async_kv_batch_put(
            keys=keys,
            fields=list_of_dict_to_tensordict(fields),
            tags=tags,
            partition_id="train" if not validate else "val",
        )
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
            RepeatJudgeCoordinatorActor, deferred_enabled, validate_deferred_config,
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
        ray.get(
            [
                worker.generate_sequences.remote(chunk)
                for worker, chunk in zip(self.agent_loop_workers, chunkes, strict=False)
            ]
        )

    def cancel_sequences(self, uids: list[str], validate: bool = False) -> int:
        """Cancel the requested prompt groups on whichever workers own them."""
        if not uids:
            return 0
        return sum(
            ray.get(
                [
                    worker.cancel_sequences.remote(uids, validate=validate)
                    for worker in self.agent_loop_workers
                ]
            )
        )

    def close_repeat_coordinator(self):
        if self.repeat_coordinator is not None:
            ray.kill(self.repeat_coordinator)
            self.repeat_coordinator = None

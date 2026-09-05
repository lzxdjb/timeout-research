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
import logging
import math
import os
import time
from collections import Counter, defaultdict

import numpy as np
import transfer_queue as tq
from omegaconf import DictConfig
from transfer_queue import KVBatchMeta

from verl.utils.skip import SkipManager

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

VERL_REPLAY_BUFFER_DEBUG_INTERVAL_SECONDS = int(os.getenv("VERL_REPLAY_BUFFER_DEBUG_INTERVAL_SECONDS", "60"))

DAPO_FILTERED_REWARD_COUNTS_KEY = "_dapo_filtered_reward_counts"

COMPLETION_RATIO_ENV = {
    "train": "SWE_AGENT_ROLLOUT_TRAINING_COMPLETION_RATIO_THRESHOLD",
    "val": "SWE_AGENT_ROLLOUT_VALIDATION_COMPLETION_RATIO_THRESHOLD",
}

INFRA_FAILURE_RATIO_ENV = {
    "train": "SWE_AGENT_ROLLOUT_TRAINING_INFRA_FAILURE_GROUP_RATIO_THRESHOLD",
    "val": "SWE_AGENT_ROLLOUT_VALIDATION_INFRA_FAILURE_GROUP_RATIO_THRESHOLD",
}


def completion_ratio_threshold(partition_id: str) -> float | None:
    """Return the opt-in completion ratio for a TransferQueue partition."""
    name = COMPLETION_RATIO_ENV[partition_id]
    raw_value = os.getenv(name)
    if raw_value is None:
        return None
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(value) or not 0.0 < value <= 1.0:
        raise ValueError(f"{name} must be greater than 0 and at most 1")
    return value


def infrastructure_failure_ratio_threshold(partition_id: str) -> float | None:
    """Return the opt-in infrastructure-failure ratio for a partition."""
    name = INFRA_FAILURE_RATIO_ENV[partition_id]
    raw_value = os.getenv(name)
    if raw_value is None:
        return None
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return value


def _accumulate_eviction_metrics(acc: dict, new: dict, stale_count: int) -> None:
    """Merge one poll iteration's eviction metrics into ``acc`` in place.

    ``stale_count`` weights the staleness mean so it stays a true per-sample average across iterations.
    """
    stale_count_key = next((k for k in new if k.endswith("/off_policy/evicted_samples")), None)
    prev_stale_total = acc.get(stale_count_key, 0) if stale_count_key else 0
    infrastructure_count_key = next(
        (k for k in new if k.endswith("/infrastructure_group_filter/discarded_groups")), None
    )
    previous_infrastructure_total = acc.get(infrastructure_count_key, 0) if infrastructure_count_key else 0
    new_infrastructure_total = new.get(infrastructure_count_key, 0) if infrastructure_count_key else 0

    for key, value in new.items():
        if key.endswith("/evicted_samples_staleness/mean"):
            denom = prev_stale_total + stale_count
            acc[key] = (acc.get(key, 0.0) * prev_stale_total + value * stale_count) / denom if denom else value
        elif key.endswith("/evicted_samples_staleness/max"):
            acc[key] = max(acc.get(key, value), value)
        elif key.endswith("/evicted_samples_staleness/min"):
            acc[key] = min(acc.get(key, value), value)
        elif key.endswith("/infrastructure_group_filter/failure_ratio_mean"):
            denominator = previous_infrastructure_total + new_infrastructure_total
            acc[key] = (
                acc.get(key, 0.0) * previous_infrastructure_total + value * new_infrastructure_total
            ) / denominator
        elif key.endswith("/infrastructure_group_filter/failure_ratio_max"):
            acc[key] = max(acc.get(key, value), value)
        elif key == DAPO_FILTERED_REWARD_COUNTS_KEY:
            # Dict-valued diagnostic: merge {metric_value: count} across poll iterations.
            merged = Counter(acc.get(key, {}))
            merged.update(value)
            acc[key] = dict(merged)
        else:
            acc[key] = acc.get(key, 0) + value


# TODO: Pass custom sampler to TransferQueue:
# https://github.com/Ascend/TransferQueue/blob/main/tutorial/05_custom_sampler.py


class ReplayBuffer:
    """ReplayBuffer is used by trainer to sample trajectories produced during rollout.

    We use [TransferQueue](https://github.com/Ascend/TransferQueue) as kv store to store trajectories.

    ### [Trajectories storage format]
    The key format is `{uid}_{session_id}_{index}`, where:
    - uid: Auto generated unique id when prompt is sampled from dataset.
    - session_id: Session id for GRPO group sampling: [0, n).
    - index: Index of output trajectory in a session.

    There're two types of data associated with each key: tag and value. The tag are arbitrary metadata:
    `{"status": "running", ...}` used to track the status of the trajectory.

    The value is a dictionary containing the following fields:
    - messages/datasource/reward_model/...: fields from dataset.
    - prompt_ids/response_ids/response_mask/...: fields from AgentLoopOutput.

    TransferQueue store tag and value separately, the tag are stored in meta server, while the value is stored
    in storage units.

    ### [GRPO group sampling control]
    Except trajectories, we also store raw prompts in TransferQueue with key `{uid}`, with `status` tag to track
    status of GRPO group sampling.
    - pending: the prompt is sampled from dataset but its sessions are not yet started.
    - running: all sessions of the prompt are running.
    - finished: all sessions of the prompt are finished without error.
    - failure: all sessions of the prompt are finished, but at least one session failed.
    Only prompts with status `finished` or `failure` enter terminal-group handling.

    ### [Terminal-group eviction/refill matrix]
    ``drop`` means off-policy staleness dropping. Both off-policy strategies (``drop`` and the dropless
    ``wait``, which blocks until stale in-flight prompts finish) are only for async trainers; sync sampling
    is on-policy, so ``max_off_policy_strategy`` is a NO-OP there.
    ``DAPO`` means filtering groups whose configured reward metric is identical across all trajectories,
             for async reward-computation path.
    ``failure`` is the group status described above.
    The matrix applies to the training partition, in which ``k`` is the number of prompts to evict.
    Validation treats all terminal groups as sampleable

    |   trainer mode   |   drop   |               DAPO               |            failure            |
    | ---------------- | -------- | -------------------------------- | ----------------------------- |
    |       sync       |   NO-OP  |    Evict ``k``; refill ``2k``    |  NO-OP or opt-in refill ``k`` |
    |      async       |              All the same: Evict ``k``; refill ``k``.                       |

    In sync mode, DAPO is opt-in and trades generation time for training stability. Each ``k`` evictions add
    ``2k`` logical refill credits, but prompts are fetched only as bounded pending/running slots become available.
    Terminal groups are filtered while other requests remain in flight. Once enough groups are sampleable, inflight
    requests are drained and discarded. By default, failed groups remain sampleable and missing trajectories are
    padded downstream. Setting ``sync_refill_failed_groups=True`` allows refilling failed samples.
    In async mode, ``num_warmup_batches`` absorbs retry cost, so all three paths refill exactly ``k`` prompts.

    Args:
        trainer_mode (str): Trainer mode.
        trainer_config (DictConfig): Trainer configuration.
        max_off_policy_threshold (int): Maximum number of model versions that trajectory can span.
        max_off_policy_strategy (str): How to handle trajectory that exceeds the maximum number of model versions.
        sampler_kwargs (dict): Additional kwargs for the custom sampler.
        poll_interval (float, optional): Poll interval in seconds. Defaults to 2.0.
        refill_fn (callable, optional): Trainer-injected function that submits an exact number of fresh prompts.
        filter_groups_metric (str, optional): DAPO group-filtering metric read from each trajectory's
            ``extra_fields.reward_extra_info``. ``None`` disables DAPO filtering.
        train_batch_size (int, optional): Prompt count represented by one Sync DAPO in-flight batch.
        gen_batch_size (int, optional): Dataloader fetch granularity for refill dispatches.
        max_inflight_gen_batches (int): Maximum Sync DAPO prompt batches concurrently pending or running.
        sync_refill_failed_groups (bool): Whether sync sampling replaces failed groups with no trajectories.
        cancel_fn (callable, optional): UID-scoped rollout cancellation used by completion-ratio cutoff.
    """

    def __init__(
        self,
        trainer_mode: str,
        trainer_config: DictConfig,
        max_off_policy_threshold: int,
        max_off_policy_strategy: str,
        sampler_kwargs: DictConfig,
        poll_interval: float = 2.0,
        refill_fn=None,
        filter_groups_metric: str | None = None,
        train_batch_size: int | None = None,
        gen_batch_size: int | None = None,
        max_inflight_gen_batches: int = 1,
        sync_refill_failed_groups: bool = False,
        cancel_fn=None,
        expected_rollout_n: int | None = None,
        expected_validation_rollout_n: int | None = None,
        validation_refill_fn=None,
    ):
        self.trainer_mode = trainer_mode
        self.trainer_config = trainer_config
        self.max_off_policy_threshold = max_off_policy_threshold
        self.max_off_policy_strategy = max_off_policy_strategy
        self.sampler_kwargs = sampler_kwargs
        self.poll_interval = poll_interval
        self.refill_fn = refill_fn
        self.filter_groups_metric = filter_groups_metric
        self.train_batch_size = train_batch_size
        self.gen_batch_size = gen_batch_size
        self.max_inflight_gen_batches = max_inflight_gen_batches
        self.sync_refill_failed_groups = sync_refill_failed_groups
        self.cancel_fn = cancel_fn
        self.completion_ratio_thresholds = {
            partition_id: completion_ratio_threshold(partition_id) for partition_id in COMPLETION_RATIO_ENV
        }
        self.infrastructure_failure_ratio_thresholds = {
            partition_id: infrastructure_failure_ratio_threshold(partition_id)
            for partition_id in INFRA_FAILURE_RATIO_ENV
        }
        self.expected_rollout_counts = {
            "train": expected_rollout_n,
            "val": expected_validation_rollout_n,
        }
        self.validation_refill_fn = validation_refill_fn

        logger.info(
            "V1 replay threshold config: mode=%s completion(train/val)=%s/%s "
            "infra_failure(train/val)=%s/%s rollout_n(train/val)=%s/%s "
            "refill(train/val)=%s/%s",
            self.trainer_mode,
            self.completion_ratio_thresholds["train"],
            self.completion_ratio_thresholds["val"],
            self.infrastructure_failure_ratio_thresholds["train"],
            self.infrastructure_failure_ratio_thresholds["val"],
            expected_rollout_n,
            expected_validation_rollout_n,
            self.refill_fn is not None,
            self.validation_refill_fn is not None,
        )

        assert isinstance(self.max_off_policy_threshold, int) and self.max_off_policy_threshold > 0, (
            f"Invalid max off policy threshold: {self.max_off_policy_threshold}, must be an integer greater than 0"
        )
        assert self.max_off_policy_strategy in ["drop", "wait"], (
            f"Invalid max off policy strategy: {self.max_off_policy_strategy}, must be one of ['drop', 'wait']"
        )
        if self.filter_groups_metric is not None and self.refill_fn is None:
            raise ValueError("Group filtering (filter_groups_metric) requires refill_fn to replace evicted groups")
        if self.sync_refill_failed_groups and self.refill_fn is None:
            raise ValueError("sync_refill_failed_groups requires refill_fn to replace failed groups")
        if any(value is not None for value in self.completion_ratio_thresholds.values()):
            if self.trainer_mode != "sync":
                raise ValueError("Completion-ratio rollout cutoff is supported only in V1 synchronous trainer mode")
            if self.filter_groups_metric is not None:
                raise ValueError("Completion-ratio rollout cutoff cannot be combined with synchronous group filtering")
        if any(value is not None for value in self.infrastructure_failure_ratio_thresholds.values()):
            if self.trainer_mode != "sync":
                raise ValueError("Infrastructure group filtering is supported only in V1 synchronous trainer mode")
            for partition_id, count in self.expected_rollout_counts.items():
                if self.infrastructure_failure_ratio_thresholds[partition_id] is None:
                    continue
                if self.completion_ratio_thresholds[partition_id] is None:
                    name = COMPLETION_RATIO_ENV[partition_id]
                    raise ValueError(f"Infrastructure group filtering requires {name}")
                if not isinstance(count, int) or count <= 0:
                    raise ValueError(
                        f"Infrastructure group filtering requires a positive rollout count for {partition_id}"
                    )
                if partition_id == "train" and self.refill_fn is None:
                    raise ValueError("Training infrastructure group filtering requires refill_fn")
                if partition_id == "val" and self.validation_refill_fn is None:
                    raise ValueError("Validation infrastructure group filtering requires validation_refill_fn")
        self._validate_mode_config()
        # partition_id => {key: tag}
        self.partitions: dict[str, dict[str, dict]] = defaultdict(dict)
        self.pending_keys: dict[str, set] = defaultdict(set)
        self.running_keys: dict[str, set] = defaultdict(set)
        self.finished_keys: dict[str, set] = defaultdict(set)
        self.failure_keys: dict[str, set] = defaultdict(set)
        self.completion_ratio_cutoff_keys: dict[str, set] = defaultdict(set)
        # partition_id => {prompt_key: global_steps}, used to prioritize older samples.
        self.prompt_global_steps: dict[str, dict[str, int]] = defaultdict(dict)
        # Finished groups are immutable, so their DAPO classification can be reused across polling iterations.
        self._dapo_classification_cache: dict[str, dict[str, float | None]] = defaultdict(dict)
        self._infrastructure_classification_cache: dict[str, dict[str, float]] = defaultdict(dict)

    def _validate_mode_config(self) -> None:
        if self.filter_groups_metric is not None:
            if not isinstance(self.max_inflight_gen_batches, int) or self.max_inflight_gen_batches <= 0:
                raise ValueError("max_inflight_gen_batches must be a positive integer")
        if self.sync_refill_failed_groups and self.gen_batch_size != 1:
            raise ValueError("sync_refill_failed_groups requires gen_batch_size=1")

    def _sync_metadata_from_transfer_queue(self):
        """Sync the metadata from TransferQueue."""
        self.partitions.clear()
        self.pending_keys.clear()
        self.running_keys.clear()
        self.finished_keys.clear()
        self.failure_keys.clear()
        self.completion_ratio_cutoff_keys.clear()
        self.prompt_global_steps.clear()

        data = tq.kv_list()
        if data is None:
            return

        for partition_id, items in data.items():
            partition = self.partitions[partition_id]
            for key, tag in items.items():
                if tag.get("is_prompt", False):
                    # see: [GRPO group sampling control]
                    self.prompt_global_steps[partition_id][key] = tag["global_steps"]
                    match tag["status"]:
                        case "pending":
                            self.pending_keys[partition_id].add(key)
                        case "running":
                            self.running_keys[partition_id].add(key)
                        case "finished":
                            self.finished_keys[partition_id].add(key)
                        case "failure":
                            self.failure_keys[partition_id].add(key)
                            if tag.get("completion_ratio_cutoff", False):
                                self.completion_ratio_cutoff_keys[partition_id].add(key)
                        case _:
                            raise ValueError(f"Unknown status: {tag['status']}")
                else:
                    # see: [Trajectories storage format]
                    if key not in partition:
                        partition[key] = {}
                    partition[key].update(tag)

        for partition_id, cache in self._infrastructure_classification_cache.items():
            live_uids = self.finished_keys[partition_id] | self.failure_keys[partition_id]
            for uid in cache.keys() - live_uids:
                del cache[uid]

    @staticmethod
    def _metrics_prefix(partition_id: str) -> str:
        return "training" if partition_id == "train" else "validation"

    def _clear_groups(self, partition_id: str, uids: set[str]) -> None:
        """Remove prompt groups from TransferQueue and the active metadata snapshot."""
        if not uids:
            return

        trajectory_keys = {key for key in self.partitions[partition_id] if key.split("_")[0] in uids}
        tq.kv_clear(
            partition_id=partition_id,
            keys=[*uids, *trajectory_keys],
        )

        # Keep same-poll decisions consistent with tq.
        for key in trajectory_keys:
            del self.partitions[partition_id][key]
        for status_keys in (self.pending_keys, self.running_keys, self.finished_keys, self.failure_keys):
            status_keys[partition_id].difference_update(uids)
        self.completion_ratio_cutoff_keys[partition_id].difference_update(uids)
        for uid in uids:
            self.prompt_global_steps[partition_id].pop(uid, None)
            self._dapo_classification_cache[partition_id].pop(uid, None)

    @staticmethod
    def _infrastructure_flag(tag: dict) -> bool:
        """Read the pre-sampling infrastructure marker published by the agent loop."""
        value = tag.get("infrastructure_failure", False)
        try:
            return int(value) == 1
        except (TypeError, ValueError, OverflowError):
            return False

    def _infrastructure_filtered_uids(self, partition_id: str) -> tuple[set[str], dict[str, float]]:
        """Classify terminal groups before they participate in completion accounting."""
        threshold = self.infrastructure_failure_ratio_thresholds.get(partition_id)
        if threshold is None:
            return set(), {}

        terminal_uids = self.finished_keys[partition_id] | self.failure_keys[partition_id]
        cache = self._infrastructure_classification_cache[partition_id]
        for uid in cache.keys() - terminal_uids:
            del cache[uid]

        new_uids = terminal_uids - cache.keys()
        final_key_by_session: dict[tuple[str, str], tuple[int, str]] = {}
        for key in self.partitions[partition_id]:
            parts = key.rsplit("_", 2)
            if len(parts) != 3 or parts[0] not in new_uids:
                continue
            try:
                output_index = int(parts[2])
            except ValueError:
                output_index = 0
            session = (parts[0], parts[1])
            if session not in final_key_by_session or output_index > final_key_by_session[session][0]:
                final_key_by_session[session] = (output_index, key)

        failures_by_uid: dict[str, int] = defaultdict(int)
        for (_uid, _session), (_index, key) in final_key_by_session.items():
            if self._infrastructure_flag(self.partitions[partition_id][key]):
                failures_by_uid[_uid] += 1

        expected_count = self.expected_rollout_counts[partition_id]
        assert expected_count is not None
        for uid in new_uids:
            cache[uid] = failures_by_uid[uid] / expected_count

        ratios = {uid: cache[uid] for uid in terminal_uids}
        discarded_uids = {uid for uid, ratio in ratios.items() if ratio > threshold}
        if new_uids:
            logger.info(
                "V1 infrastructure group scan: partition=%s terminal_groups=%d newly_classified=%d "
                "materialized_sessions=%d marked_failures=%d expected_sessions=%d threshold=%.4f "
                "discarded_groups=%d",
                partition_id,
                len(terminal_uids),
                len(new_uids),
                len(final_key_by_session),
                sum(failures_by_uid.values()),
                expected_count,
                threshold,
                len(discarded_uids),
            )
        return discarded_uids, ratios

    def _discard_infrastructure_groups(self, partition_id: str) -> tuple[set[str], dict[str, float]]:
        discarded_uids, ratios = self._infrastructure_filtered_uids(partition_id)
        if not discarded_uids:
            return set(), {}
        prefix = self._metrics_prefix(partition_id)
        discarded_ratios = [ratios[uid] for uid in discarded_uids]
        metrics = {
            f"{prefix}/infrastructure_group_filter/discarded_groups": float(len(discarded_uids)),
            f"{prefix}/infrastructure_group_filter/failure_ratio_mean": float(np.mean(discarded_ratios)),
            f"{prefix}/infrastructure_group_filter/failure_ratio_max": float(np.max(discarded_ratios)),
        }
        logger.info(
            "V1 infrastructure group filter discarded partition=%s groups=%d ratio_mean=%.4f ratio_max=%.4f",
            partition_id,
            len(discarded_uids),
            float(np.mean(discarded_ratios)),
            float(np.max(discarded_ratios)),
        )
        self._clear_groups(partition_id, discarded_uids)
        return discarded_uids, metrics

    def _dapo_filtered_keys(self, partition_id: str) -> tuple[set[str], Counter]:
        """Finished groups whose configured DAPO metric is identical across all trajectories.

        Returns the filtered uids and a ``{shared_metric_value: group_count}`` breakdown built in the
        same scope, so the diagnostic (which reward level the no-signal groups collapse to) travels
        with the uids through the return value instead of via hidden instance state.
        """
        if partition_id == "val" or self.filter_groups_metric is None:
            return set(), Counter()

        finished_uids = self.finished_keys[partition_id]
        classification_cache = self._dapo_classification_cache[partition_id]
        for uid in classification_cache.keys() - finished_uids:
            del classification_cache[uid]

        new_finished_uids = finished_uids - classification_cache.keys()
        trajectory_keys = [key for key in self.partitions[partition_id] if key.split("_")[0] in new_finished_uids]
        metrics_by_uid: dict[str, list[float]] = defaultdict(list)
        missing_metric_uids = new_finished_uids - {key.split("_")[0] for key in trajectory_keys}

        if trajectory_keys:
            data = tq.kv_batch_get(
                keys=trajectory_keys,
                partition_id=partition_id,
                select_fields=["extra_fields"],
            )
            extra_fields_list = list(data["extra_fields"])
        else:
            extra_fields_list = []

        for key, extra_fields in zip(trajectory_keys, extra_fields_list, strict=True):
            uid = key.split("_")[0]
            extra_fields = getattr(extra_fields, "data", extra_fields)
            reward_extra_info = extra_fields.get("reward_extra_info", {}) if isinstance(extra_fields, dict) else {}
            if self.filter_groups_metric not in reward_extra_info:
                missing_metric_uids.add(uid)
            else:
                metrics_by_uid[uid].append(float(reward_extra_info[self.filter_groups_metric]))

        if missing_metric_uids:
            raise RuntimeError(
                f"Finished groups are missing DAPO metric {self.filter_groups_metric!r}: "
                f"{sorted(missing_metric_uids)[:5]}"
            )

        for uid in new_finished_uids:
            values = metrics_by_uid[uid]
            classification_cache[uid] = float(values[0]) if len(values) > 1 and float(np.std(values)) == 0.0 else None

        filtered_rewards = {uid: reward for uid, reward in classification_cache.items() if reward is not None}
        return set(filtered_rewards), Counter(filtered_rewards.values())

    def _terminal_eviction_reasons(
        self, global_steps: int, partition_id: str
    ) -> tuple[set[str], set[str], set[str], Counter]:
        """Return stale, DAPO-filtered, and failed groups (plus the DAPO value->count breakdown).

        The three sets may overlap. Callers clear and refill their union, so one prompt is never handled
        twice. ``dapo_counts`` is the {shared_metric_value: group_count} diagnostic for ``dapo_uids``; it
        rides along in the return value so no hidden state is needed between production and consumption.
        """
        cutoff_uids = set(self.completion_ratio_cutoff_keys[partition_id])
        usable_cutoff_uids = set()
        for uid in cutoff_uids:
            session_ids = {
                key.rsplit("_", 2)[1]
                for key in self.partitions[partition_id]
                if key.split("_")[0] == uid
            }
            if session_ids:
                usable_cutoff_uids.add(uid)
        cutoff_uids -= usable_cutoff_uids
        if partition_id == "val":
            return set(), set(), cutoff_uids, Counter()

        dapo_uids, dapo_counts = self._dapo_filtered_keys(partition_id)
        failed_uids = cutoff_uids
        if self.sync_refill_failed_groups:
            materializable_uids = {key.split("_")[0] for key in self.partitions[partition_id]}
            failed_uids |= self.failure_keys[partition_id] - materializable_uids
        return set(), dapo_uids, failed_uids, dapo_counts

    def _sampleable_terminal_keys(
        self,
        partition_id: str,
        eviction_reasons: tuple[set[str], set[str], set[str], Counter],
    ) -> set[str]:
        terminal_uids = self.finished_keys[partition_id] | self.failure_keys[partition_id]
        stale_uids, dapo_uids, failed_uids, _dapo_counts = eviction_reasons
        return terminal_uids - (stale_uids | dapo_uids | failed_uids)

    def _evict_terminal_groups(
        self,
        global_steps: int,
        partition_id: str,
        eviction_reasons: tuple[set[str], set[str], set[str], Counter],
    ) -> tuple[set[str], int, int, dict]:
        """Evict terminal groups selected by any active policy exactly once."""
        stale_uids, dapo_uids, failed_uids, dapo_counts = eviction_reasons
        evicted_uids = stale_uids | dapo_uids | failed_uids
        if not evicted_uids:
            return set(), 0, 0, {}

        prefix = self._metrics_prefix(partition_id)
        metrics: dict = {}
        if stale_uids:
            prompt_global_steps = self.prompt_global_steps[partition_id]
            spans = np.array(
                [global_steps - prompt_global_steps.get(uid, global_steps) + 1 for uid in stale_uids],
                dtype=float,
            )
            metrics.update(
                {
                    f"{prefix}/off_policy/evicted_samples": len(stale_uids),
                    f"{prefix}/off_policy/evicted_samples_staleness/mean": spans.mean(),
                    f"{prefix}/off_policy/evicted_samples_staleness/max": spans.max(),
                    f"{prefix}/off_policy/evicted_samples_staleness/min": spans.min(),
                }
            )
        if dapo_uids:
            metrics[f"{prefix}/filter_groups/evicted_samples"] = len(dapo_uids)
            # Non-scalar diagnostic: how many filtered (no-signal) groups collapsed to each metric value.
            metrics[DAPO_FILTERED_REWARD_COUNTS_KEY] = dict(dapo_counts)
        if failed_uids:
            metrics[f"{prefix}/rollout_failure/evicted_samples"] = len(failed_uids)

        self._clear_groups(partition_id, evicted_uids)
        return evicted_uids, len(stale_uids), len(dapo_uids), metrics

    def _select_prompt_uids(
        self, partition_id: str, sampleable_keys: set[str], batch_size: int
    ) -> tuple[list[str], dict[str, dict], dict[str, int]]:
        prompt_global_steps_snapshot = dict(self.prompt_global_steps[partition_id])
        partition_snapshot = dict(self.partitions[partition_id])
        ordered_keys = sorted(
            sampleable_keys,
            key=lambda key: prompt_global_steps_snapshot.get(key, 0),
        )
        return ordered_keys[:batch_size], partition_snapshot, prompt_global_steps_snapshot

    def _materialize_batch(
        self, partition_id: str, selected_prompt_uids: list[str], partition_snapshot: dict[str, dict]
    ) -> KVBatchMeta:
        tq.kv_clear(partition_id=partition_id, keys=selected_prompt_uids)

        keys, tags = [], []
        selected = set(selected_prompt_uids)
        for key, tag in partition_snapshot.items():
            uid = key.split("_")[0]
            if uid in selected:
                keys.append(key)
                tags.append(tag)
        return KVBatchMeta(partition_id=partition_id, keys=keys, tags=tags)

    def _wait_for_next_poll(self, partition_id: str, last_debug_time: float) -> float:
        time.sleep(self.poll_interval)
        now = time.time()
        if now - last_debug_time > VERL_REPLAY_BUFFER_DEBUG_INTERVAL_SECONDS:
            logger.info(
                f"pending: {len(self.pending_keys[partition_id])}, "
                f"running: {len(self.running_keys[partition_id])}, "
                f"finished: {len(self.finished_keys[partition_id])}, "
                f"failure: {len(self.failure_keys[partition_id])}"
            )
            return now
        return last_debug_time

    def _maybe_apply_completion_ratio_cutoff(
        self,
        partition_id: str,
        batch_size: int,
        cutoff_requested: bool,
    ) -> tuple[bool, dict[str, float]]:
        threshold = self.completion_ratio_thresholds.get(partition_id)
        if threshold is None or cutoff_requested:
            return cutoff_requested, {}

        terminal_uids = self.finished_keys[partition_id] | self.failure_keys[partition_id]
        inflight_uids = self.pending_keys[partition_id] | self.running_keys[partition_id]
        total_groups = len(terminal_uids) + len(inflight_uids)
        required_groups = math.ceil(total_groups * threshold)
        if total_groups < batch_size or len(terminal_uids) < required_groups or not inflight_uids:
            return False, {}
        if self.cancel_fn is None:
            raise RuntimeError("Completion-ratio rollout cutoff requires an agent-loop cancellation callback")

        cancelled = int(self.cancel_fn(sorted(inflight_uids), validate=partition_id == "val"))
        prefix = self._metrics_prefix(partition_id)
        logger.info(
            "Applied %s completion-ratio cutoff: terminal=%s total=%s threshold=%s requested=%s cancelled=%s",
            partition_id,
            len(terminal_uids),
            total_groups,
            threshold,
            len(inflight_uids),
            cancelled,
        )
        return True, {
            f"{prefix}/completion_ratio/threshold": threshold,
            f"{prefix}/completion_ratio/terminal_groups": float(len(terminal_uids)),
            f"{prefix}/completion_ratio/total_groups": float(total_groups),
            f"{prefix}/completion_ratio/cutoff_groups": float(len(inflight_uids)),
            f"{prefix}/completion_ratio/cancelled_groups": float(cancelled),
        }

    @SkipManager.annotate_tq(role="rollout_tq", phase="sample")
    def sample(self, global_steps: int, partition_id: str, batch_size: int) -> tuple[KVBatchMeta, dict]:
        """Sample a batch using synchronous rollout semantics.

        NOTE: user can customize sampling strategy by setting:
        ```bash
        trainer.v1.sampler.custom_sampler.path = "path/to/your/sampler.py"
        trainer.v1.sampler.custom_sampler.name = "UserCustomReplayBuffer"
        ```

        Args:
            global_steps (int): Global steps of the current training.
            partition_id (str): Partition of TransferQueue, e.g. "train" or "val".
            batch_size (int, optional): Batch size.

        Returns:
            KVBatchMeta: A batch of data.
            dict: Auxiliary metrics.
        """
        last_debug_time = time.time()
        eviction_metrics: dict = {}
        dapo_enabled = partition_id != "val" and self.filter_groups_metric is not None
        refill_credit = 0
        draining = False
        max_inflight_prompts = 0
        cutoff_requested = False
        if dapo_enabled:
            max_inflight_prompts = self.max_inflight_gen_batches * self.train_batch_size

        while True:
            # Eviction, gating, and selection below must all use this snapshot.
            self._sync_metadata_from_transfer_queue()

            discarded_uids, infrastructure_metrics = self._discard_infrastructure_groups(partition_id)
            if discarded_uids:
                _accumulate_eviction_metrics(eviction_metrics, infrastructure_metrics, stale_count=0)
                if not cutoff_requested:
                    if partition_id == "val":
                        self.validation_refill_fn(sorted(discarded_uids))
                    else:
                        self.refill_fn(len(discarded_uids))
                    prefix = self._metrics_prefix(partition_id)
                    refill_key = f"{prefix}/infrastructure_group_filter/refilled_groups"
                    eviction_metrics[refill_key] = eviction_metrics.get(refill_key, 0) + len(discarded_uids)
                    logger.info(
                        "V1 infrastructure group filter refilled partition=%s groups=%d",
                        partition_id,
                        len(discarded_uids),
                    )
                continue

            cutoff_requested, completion_ratio_metrics = self._maybe_apply_completion_ratio_cutoff(
                partition_id, batch_size, cutoff_requested
            )
            eviction_metrics.update(completion_ratio_metrics)
            if completion_ratio_metrics:
                continue

            eviction_reasons = self._terminal_eviction_reasons(global_steps, partition_id)
            failed_count = len(eviction_reasons[2])
            evicted_uids, stale_count, dapo_count, metrics = self._evict_terminal_groups(
                global_steps, partition_id, eviction_reasons
            )
            if evicted_uids:
                _accumulate_eviction_metrics(eviction_metrics, metrics, stale_count)

            sampleable_keys = self._sampleable_terminal_keys(partition_id, eviction_reasons)
            has_enough_samples = len(sampleable_keys) >= batch_size
            inflight_count = len(self.pending_keys[partition_id]) + len(self.running_keys[partition_id])

            if not dapo_enabled and not cutoff_requested and failed_count > 0 and not has_enough_samples:
                self.refill_fn(failed_count)
                continue

            if dapo_enabled:
                if has_enough_samples:
                    # Stop speculative dispatch, then drain requests already running under this policy version.
                    draining = True
                    refill_credit = 0
                elif not draining:
                    refill_credit += 2 * dapo_count + failed_count

                if not draining and refill_credit > 0:
                    available_slots = max(0, max_inflight_prompts - inflight_count)
                    dispatch_count = min(refill_credit, available_slots)
                    assert self.gen_batch_size is not None
                    dispatch_count -= dispatch_count % self.gen_batch_size
                    if dispatch_count > 0:
                        assert self.refill_fn is not None
                        self.refill_fn(dispatch_count)
                        refill_credit -= dispatch_count
                        continue

            can_select = (
                (has_enough_samples or (cutoff_requested and bool(sampleable_keys)))
                and (not dapo_enabled or inflight_count == 0)
            )
            if can_select:
                selected_prompt_uids, partition_snapshot, _prompt_global_steps_snapshot = self._select_prompt_uids(
                    partition_id, sampleable_keys, batch_size
                )

                # Sync remains bufferless: all speculative requests are drained, then surplus is discarded.
                if dapo_enabled:
                    surplus_uids = sampleable_keys - set(selected_prompt_uids)
                    if surplus_uids:
                        self._clear_groups(partition_id, surplus_uids)
                        key = f"{self._metrics_prefix(partition_id)}/filter_groups/discarded_surplus_samples"
                        eviction_metrics[key] = eviction_metrics.get(key, 0) + len(surplus_uids)
                break

            last_debug_time = self._wait_for_next_poll(partition_id, last_debug_time)

        selected_uids = set(selected_prompt_uids)
        if partition_id != "val" and not any(key.split("_")[0] in selected_uids for key in partition_snapshot):
            message = "Sync replay buffer selected terminal groups with no materializable trajectories."
            if not self.sync_refill_failed_groups:
                message += " Enable trainer.v1.sampler.sync_refill_failed_groups to replace failed groups."
            raise RuntimeError(message)
        return self._materialize_batch(partition_id, selected_prompt_uids, partition_snapshot), eviction_metrics


class ReplayBufferAsync(ReplayBuffer):
    """Async sampling policy over the shared TransferQueue and dynamic-filter implementation."""

    def _validate_mode_config(self) -> None:
        pass

    def _stale_terminal_keys(self, global_steps: int, partition_id: str) -> set[str]:
        if partition_id == "val" or self.max_off_policy_strategy != "drop":
            return set()
        prompt_global_steps = self.prompt_global_steps[partition_id]
        terminal_keys = self.finished_keys[partition_id]
        return {
            uid
            for uid in terminal_keys
            if global_steps - prompt_global_steps.get(uid, global_steps) + 1 > self.max_off_policy_threshold
        }

    def _terminal_eviction_reasons(
        self, global_steps: int, partition_id: str
    ) -> tuple[set[str], set[str], set[str], Counter]:
        if partition_id == "val":
            return set(), set(), set(), Counter()

        stale_uids = self._stale_terminal_keys(global_steps, partition_id)
        dapo_uids, dapo_counts = self._dapo_filtered_keys(partition_id)
        return stale_uids, dapo_uids, set(self.failure_keys[partition_id]), dapo_counts

    def _has_enough_samples(
        self,
        global_steps: int,
        partition_id: str,
        batch_size: int,
        sampleable_keys: set[str],
    ) -> bool:
        # Dropless off-policy control: block sampling while any in-flight prompt has reached the staleness
        # threshold, so it can finish and be trained on instead of dropped.
        if self.max_off_policy_strategy == "wait":
            for key in self.pending_keys[partition_id] | self.running_keys[partition_id]:
                prompt_global_steps = self.prompt_global_steps[partition_id][key]
                if (global_steps - prompt_global_steps + 1) >= self.max_off_policy_threshold:
                    return False

        return len(sampleable_keys) >= batch_size

    @SkipManager.annotate_tq(role="rollout_tq", phase="sample")
    def sample(self, global_steps: int, partition_id: str, batch_size: int) -> tuple[KVBatchMeta, dict]:
        """Sample a batch while evicting and replacing stale, DAPO-filtered, or failed groups."""
        last_debug_time = time.time()
        eviction_metrics: dict = {}

        while True:
            # Eviction and selection share one snapshot so newly terminal stale groups wait for the next eviction pass.
            self._sync_metadata_from_transfer_queue()

            eviction_reasons = self._terminal_eviction_reasons(global_steps, partition_id)
            evicted_uids, stale_count, _dapo_count, metrics = self._evict_terminal_groups(
                global_steps, partition_id, eviction_reasons
            )
            if evicted_uids:
                _accumulate_eviction_metrics(eviction_metrics, metrics, stale_count)
                if self.refill_fn is not None:
                    self.refill_fn(len(evicted_uids))
                continue

            sampleable_keys = self._sampleable_terminal_keys(partition_id, eviction_reasons)
            if self._has_enough_samples(global_steps, partition_id, batch_size, sampleable_keys):
                selected_prompt_uids, partition_snapshot, prompt_global_steps_snapshot = self._select_prompt_uids(
                    partition_id, sampleable_keys, batch_size
                )
                break

            last_debug_time = self._wait_for_next_poll(partition_id, last_debug_time)

        if partition_id != "val" and self.max_off_policy_strategy == "drop":
            selected_spans = [
                global_steps - prompt_global_steps_snapshot.get(uid, global_steps) + 1 for uid in selected_prompt_uids
            ]
            assert all(span <= self.max_off_policy_threshold for span in selected_spans), (
                f"drop strategy selected stale prompts: spans={selected_spans}, "
                f"threshold={self.max_off_policy_threshold}"
            )

        return self._materialize_batch(partition_id, selected_prompt_uids, partition_snapshot), eviction_metrics

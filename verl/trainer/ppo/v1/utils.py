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
from collections import Counter, defaultdict
from typing import Any

import numpy as np
import torch

from verl.protocol import DataProto
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.ray_trainer import compute_advantage
from verl.trainer.ppo.v1.replay_buffer import DAPO_FILTERED_REWARD_COUNTS_KEY


_SUCCESS_RATIO_VARIANTS = (
    "excluding_completion_cutoff",
    "excluding_completion_cutoff_and_infrastructure",
)


def _optional_finite_float(value: Any) -> float | None:
    """Return a finite scalar, or ``None`` for missing/non-numeric metadata."""
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        value = value.item()
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return numeric if np.isfinite(numeric) else None


def v1_success_values(extra_fields: list[Any], fallback_scores: list[Any]) -> list[float | None]:
    """Use V1's validation-core success convention for trajectory metrics.

    ``acc`` is preferred when the reward manager emitted it for at least one
    trajectory. Otherwise, the trajectory reward is used, matching validation's
    existing ``acc``-then-``reward`` core-variable selection.
    """
    if len(extra_fields) != len(fallback_scores):
        raise ValueError(
            f"extra_fields and fallback_scores must have equal length, got {len(extra_fields)} and "
            f"{len(fallback_scores)}"
        )

    accuracy_values: list[float | None] = []
    has_accuracy = False
    for extra in extra_fields:
        extra = getattr(extra, "data", extra)
        extra = extra if isinstance(extra, dict) else {}
        reward_info = extra.get("reward_extra_info", {})
        reward_info = reward_info if isinstance(reward_info, dict) else {}
        has_accuracy = has_accuracy or "acc" in reward_info
        accuracy_values.append(_optional_finite_float(reward_info.get("acc")))

    if has_accuracy:
        return accuracy_values
    return [_optional_finite_float(score) for score in fallback_scores]


def flatten_v1_extra_fields(extra_fields: list[Any]) -> dict[str, np.ndarray]:
    """Flatten per-trajectory V1 reward metadata for ``compute_data_metrics``.

    V0 places reward metadata directly in ``DataProto.non_tensor_batch``. V1
    stores the same values in TransferQueue ``extra_fields``; this adapter
    restores the V0 representation without changing the queue schema.
    """
    rows: list[dict[str, Any]] = []
    for extra in extra_fields:
        extra = getattr(extra, "data", extra)
        extra = extra if isinstance(extra, dict) else {}
        reward_info = extra.get("reward_extra_info", {})
        reward_info = reward_info if isinstance(reward_info, dict) else {}
        values = dict(extra)
        values.update(reward_info)
        values.pop("reward_extra_info", None)
        rows.append(values)
    keys = set().union(*(row.keys() for row in rows)) if rows else set()
    return {
        key: np.asarray([row.get(key) for row in rows], dtype=object)
        for key in keys
    }


def compute_v1_success_ratio_metrics(
    *,
    batch_keys: list[str],
    batch_tags: list[dict[str, Any]],
    success_values: list[Any],
    prefix: str,
    infrastructure_failure_ratio_threshold: float | None,
    expected_rollout_count: int,
    metric_sources: list[Any] | None = None,
) -> dict[str, float]:
    """Compute trajectory success ratios after excluding whole V1 prompt groups.

    Only the final output of each agent-loop session participates. Completion
    cutoff is a group property when any row carries the cutoff tag. The
    infrastructure classification intentionally mirrors ``ReplayBuffer``: the
    final-session failure count is divided by the configured rollout count and
    compared to the threshold with a strict ``>``.
    """
    if not (len(batch_keys) == len(batch_tags) == len(success_values)):
        raise ValueError(
            "batch_keys, batch_tags, and success_values must have equal length, got "
            f"{len(batch_keys)}, {len(batch_tags)}, and {len(success_values)}"
        )
    if expected_rollout_count <= 0:
        raise ValueError(f"expected_rollout_count must be positive, got {expected_rollout_count}")
    if metric_sources is not None and len(metric_sources) != len(batch_keys):
        raise ValueError(
            f"metric_sources must have length {len(batch_keys)}, got {len(metric_sources)}"
        )

    final_by_session: dict[tuple[str, str], tuple[int, int]] = {}
    group_cutoff: dict[str, bool] = defaultdict(bool)
    for row_index, (key, tag) in enumerate(zip(batch_keys, batch_tags, strict=True)):
        if tag.get("is_padding", False):
            continue
        parts = key.rsplit("_", 2)
        if len(parts) == 3:
            uid, session_id = parts[0], parts[1]
            try:
                output_index = int(parts[2])
            except ValueError:
                output_index = 0
        else:
            uid, session_id, output_index = key, "0", 0
        group_cutoff[uid] = group_cutoff[uid] or bool(tag.get("completion_ratio_cutoff", False))
        session = (uid, session_id)
        if session not in final_by_session or output_index > final_by_session[session][0]:
            final_by_session[session] = (output_index, row_index)

    final_rows: list[tuple[str, float | None, bool, Any]] = []
    failures_by_group: Counter[str] = Counter()
    for (uid, _session_id), (_output_index, row_index) in final_by_session.items():
        success = _optional_finite_float(success_values[row_index])
        tag = batch_tags[row_index]
        try:
            infrastructure_failure = int(tag.get("infrastructure_failure", False)) == 1
        except (TypeError, ValueError, OverflowError):
            infrastructure_failure = False
        failures_by_group[uid] += int(infrastructure_failure)
        source = metric_sources[row_index] if metric_sources is not None else None
        final_rows.append((uid, success, group_cutoff[uid], source))

    infrastructure_groups: set[str] = set()
    if infrastructure_failure_ratio_threshold is not None:
        infrastructure_groups = {
            uid
            for uid, failure_count in failures_by_group.items()
            if failure_count / expected_rollout_count > infrastructure_failure_ratio_threshold
        }

    cutoff_rows = [(uid, success) for uid, success, cutoff, _source in final_rows if not cutoff and success is not None]
    valid_rows = [
        (uid, success) for uid, success in cutoff_rows if uid not in infrastructure_groups and success is not None
    ]

    def ratio(rows: list[tuple[str, float]]) -> tuple[float, float, float]:
        eligible_count = len(rows)
        successful_count = float(sum(success for _uid, success in rows))
        value = successful_count / eligible_count if eligible_count else 0.0
        return value, successful_count, float(eligible_count)

    cutoff_ratio, cutoff_successes, cutoff_eligible = ratio(cutoff_rows)
    valid_ratio, valid_successes, valid_eligible = ratio(valid_rows)
    metric_root = f"{prefix}/success_ratio"
    cutoff_name, valid_name = _SUCCESS_RATIO_VARIANTS
    metrics = {
        f"{metric_root}/{cutoff_name}": cutoff_ratio,
        f"{metric_root}/{cutoff_name}_successful_count": cutoff_successes,
        f"{metric_root}/{cutoff_name}_eligible_count": cutoff_eligible,
        f"{metric_root}/{valid_name}": valid_ratio,
        f"{metric_root}/{valid_name}_successful_count": valid_successes,
        f"{metric_root}/{valid_name}_eligible_count": valid_eligible,
        f"{metric_root}/excluded_completion_cutoff_group_count": float(
            sum(uid in group_cutoff and group_cutoff[uid] for uid in {row[0] for row in final_rows})
        ),
        f"{metric_root}/excluded_infrastructure_group_count": float(len(infrastructure_groups)),
    }
    if metric_sources is None:
        return metrics

    # Emit the same metrics under each benchmark/source.  This is deliberately
    # additive: existing aggregate keys remain unchanged for dashboards that
    # already consume them.
    source_names = {
        str(source) if source not in (None, "") else "unknown"
        for _uid, _success, _cutoff, source in final_rows
    }
    for source in source_names:
        source_rows = [
            (uid, success, cutoff)
            for uid, success, cutoff, row_source in final_rows
            if (str(row_source) if row_source not in (None, "") else "unknown") == source
            and success is not None
        ]
        source_cutoff_rows = [(uid, success) for uid, success, cutoff in source_rows if not cutoff]
        source_valid_rows = [
            (uid, success) for uid, success in source_cutoff_rows if uid not in infrastructure_groups
        ]
        source_cutoff_ratio, source_cutoff_successes, source_cutoff_eligible = ratio(source_cutoff_rows)
        source_valid_ratio, source_valid_successes, source_valid_eligible = ratio(source_valid_rows)
        source_root = f"{prefix}/{source}/success_ratio"
        metrics.update(
            {
                f"{source_root}/{cutoff_name}": source_cutoff_ratio,
                f"{source_root}/{cutoff_name}_successful_count": source_cutoff_successes,
                f"{source_root}/{cutoff_name}_eligible_count": source_cutoff_eligible,
                f"{source_root}/{valid_name}": source_valid_ratio,
                f"{source_root}/{valid_name}_successful_count": source_valid_successes,
                f"{source_root}/{valid_name}_eligible_count": source_valid_eligible,
                f"{source_root}/excluded_completion_cutoff_group_count": float(
                    len({uid for uid, _success, cutoff in source_rows if cutoff})
                ),
                f"{source_root}/excluded_infrastructure_group_count": float(
                    len({uid for uid, _success in source_cutoff_rows if uid in infrastructure_groups})
                ),
            }
        )
    return metrics


class MetricsAggregator:
    """
    Combine per-iteration training metrics collected within a single ``parameter_sync_step`` cycle.
    Adapted from ``verl.experimental.fully_async_policy.detach_utils.MetricsAggregator`.
    """

    def __init__(self):
        self.metric_values: dict[str, list[float]] = defaultdict(list)
        self.metric_weights: dict[str, list[int]] = defaultdict(list)
        self.dict_metrics: dict[str, Counter] = defaultdict(Counter)
        self.step_count = 0
        self.aggregation_rules = self._init_aggregation_rules()

    def _init_aggregation_rules(self) -> dict[str, list[str]]:
        return {
            "sum": [
                "training/off_policy/evicted_samples",
                "validation/off_policy/evicted_samples",
                "training/filter_groups/evicted_samples",
                "validation/filter_groups/evicted_samples",
                "training/filter_groups/discarded_surplus_samples",
                "validation/filter_groups/discarded_surplus_samples",
                "training/rollout_failure/evicted_samples",
                "validation/rollout_failure/evicted_samples",
                "training/infrastructure_group_filter/discarded_groups",
                "validation/infrastructure_group_filter/discarded_groups",
                "training/infrastructure_group_filter/refilled_groups",
                "validation/infrastructure_group_filter/refilled_groups",
            ],
            "last": [
                "training/global_step",
                "training/rollout_probs_diff_valid",
            ],
        }

    def add_step_metrics(self, metrics: dict[str, Any], sample_count: int = 0):
        """Record one iteration's metrics."""
        self.step_count += 1
        for key, value in metrics.items():
            if isinstance(value, bool):
                continue
            if key == DAPO_FILTERED_REWARD_COUNTS_KEY and isinstance(value, dict):
                self.dict_metrics[key].update(value)
                continue
            if isinstance(value, int | float | np.number):
                self.metric_values[key].append(float(value))
                self.metric_weights[key].append(self._get_metric_weight(key, metrics, sample_count))
            elif isinstance(value, torch.Tensor) and value.numel() == 1:
                self.metric_values[key].append(float(value.item()))
                self.metric_weights[key].append(self._get_metric_weight(key, metrics, sample_count))

    def _get_metric_weight(self, metric_name: str, metrics: dict[str, Any], sample_count: int) -> int:
        """Return the sample weight used when reducing per-iteration average metrics."""
        if "/success_ratio/" in metric_name and metric_name.rsplit("/", 1)[-1] in _SUCCESS_RATIO_VARIANTS:
            eligible_count = metrics.get(f"{metric_name}_eligible_count", sample_count)
            if isinstance(eligible_count, torch.Tensor):
                return int(eligible_count.item()) if eligible_count.numel() == 1 else sample_count
            if isinstance(eligible_count, int | float | np.number):
                return int(eligible_count)
        if metric_name.endswith("/off_policy/evicted_samples_staleness/mean"):
            prefix = metric_name.rsplit("_staleness/mean", 1)[0]
            evicted_samples = metrics.get(prefix, sample_count)
            if isinstance(evicted_samples, torch.Tensor):
                return int(evicted_samples.item()) if evicted_samples.numel() == 1 else sample_count
            if isinstance(evicted_samples, int | float | np.number):
                return int(evicted_samples)
        if metric_name.endswith("/infrastructure_group_filter/failure_ratio_mean"):
            prefix = metric_name.rsplit("/failure_ratio_mean", 1)[0]
            discarded_groups = metrics.get(f"{prefix}/discarded_groups", sample_count)
            if isinstance(discarded_groups, torch.Tensor):
                return int(discarded_groups.item()) if discarded_groups.numel() == 1 else sample_count
            if isinstance(discarded_groups, int | float | np.number):
                return int(discarded_groups)
        return sample_count

    def _get_aggregation_type(self, metric_name: str) -> str:
        for agg_type, metric_list in self.aggregation_rules.items():
            if metric_name in metric_list:
                return agg_type

        metric_lower = metric_name.lower()
        if "/success_ratio/" in metric_lower and metric_lower.endswith("_count"):
            return "sum"
        if metric_lower.endswith("/lr") or metric_lower.endswith("_lr") or metric_lower == "lr":
            return "last"
        if "timing_s/" in metric_lower or "timing_per_token_ms/" in metric_lower:
            return "time_sum"
        if any(keyword in metric_lower for keyword in ["max", "maximum"]):
            return "max"
        if any(keyword in metric_lower for keyword in ["min", "minimum"]):
            return "min"
        if any(keyword in metric_lower for keyword in ["sum", "total"]):
            return "sum"
        if any(keyword in metric_lower for keyword in ["weighted_avg", "mean", "avg", "average"]):
            return "weighted_avg"
        return "weighted_avg"

    def _aggregate_single_metric(self, metric_name: str, values: list[float]) -> float:
        if not values:
            return 0.0

        agg_type = self._get_aggregation_type(metric_name)
        if agg_type == "last":
            return values[-1]
        if agg_type == "weighted_avg":
            weights = self.metric_weights[metric_name]
            if len(values) != len(weights) or sum(weights) == 0:
                return sum(values) / len(values)
            weighted_sum = sum(v * c for v, c in zip(values, weights, strict=False))
            return weighted_sum / sum(weights)
        if agg_type in ("sum", "time_sum"):
            return sum(values)
        if agg_type == "max":
            return max(values)
        if agg_type == "min":
            return min(values)
        return sum(values) / len(values)

    def get_aggregated_metrics(self) -> dict[str, Any]:
        if self.step_count == 0:
            return {}
        aggregated = {name: self._aggregate_single_metric(name, values) for name, values in self.metric_values.items()}
        for name, counter in self.dict_metrics.items():
            if counter:
                aggregated[name] = dict(counter)
        return self._special_metrics_aggregate(aggregated)

    def _special_metrics_aggregate(self, aggregated: dict[str, Any]) -> dict[str, Any]:
        """Recompute derived metrics that cannot be reduced from their per-iteration values."""
        if {"global_seqlen/minmax_diff", "global_seqlen/max", "global_seqlen/min"}.issubset(aggregated):
            aggregated["global_seqlen/minmax_diff"] = aggregated["global_seqlen/max"] - aggregated["global_seqlen/min"]

        return aggregated

    def reset(self):
        self.metric_values.clear()
        self.metric_weights.clear()
        self.dict_metrics.clear()
        self.step_count = 0


def compute_advantage_for_multi_trajectories(
    data: DataProto,
    batch_keys: list[str],
    adv_estimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Any = None,
) -> DataProto:
    """Compute GRPO advantages from each session's final output. For non-GRPO
    estimators, such as GAE, are delegated to the original compute_advantage() unchanged.

    For GRPO, only the final output in each ``{uid}_{session_id}`` group participates
    in advantage computation, and the result is broadcast to the other outputs in
    the same session. Sessions whose AgentLoop returns ``None`` simply do not appear
    in ``batch_keys``. Non-GRPO estimators, such as GAE, are delegated to the
    original ``compute_advantage()`` unchanged.
    """
    if adv_estimator != core_algos.AdvantageEstimator.GRPO:
        return compute_advantage(
            data,
            adv_estimator=adv_estimator,
            gamma=gamma,
            lam=lam,
            num_repeat=num_repeat,
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            config=config,
        )

    # final session of each agent loop: {uid}_{session_id} => (index, row_index)
    final_sessions: dict[str, tuple[int, int]] = {}
    row_session_keys = []
    for i, key in enumerate(batch_keys):
        fields = key.rsplit("_", 2)
        assert len(fields) == 3, f"Unexpected key format: {key}"
        uid, session_id, index = fields[0], fields[1], int(fields[2])
        session_key = f"{uid}_{session_id}"
        row_session_keys.append(session_key)
        if session_key not in final_sessions or final_sessions[session_key][0] < index:
            final_sessions[session_key] = (index, i)

    # final session indices in batch data
    final_indices = []
    session_key_to_local_index = {}
    for session_key, (_, row_index) in final_sessions.items():
        final_indices.append(row_index)
        session_key_to_local_index[session_key] = len(final_indices) - 1
    row_to_local_index = [session_key_to_local_index[session_key] for session_key in row_session_keys]

    # select final sessions from batch data for group relative advantage computation
    final_data = compute_advantage(
        data.select_idxs(final_indices),
        adv_estimator=adv_estimator,
        gamma=gamma,
        lam=lam,
        num_repeat=num_repeat,
        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        config=config,
    )
    first_nnz_indices = final_data.batch["response_mask"].argmax(dim=1)
    final_scores = final_data.batch["advantages"][torch.arange(len(final_data)), first_nnz_indices]

    # scatter final scores to all rows in batch data
    scores = final_scores[row_to_local_index]
    scores = scores.unsqueeze(-1) * data.batch["response_mask"]

    data.batch["advantages"] = scores
    data.batch["returns"] = scores
    return data

# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch
from tensordict import TensorDict

from verl.base_config import BaseConfig
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils import tensordict_utils as tu
from verl.utils.metric import AggregationType, Metric
from verl.workers.config import ActorConfig, DistillationConfig, DistillationLossConfig
from verl.workers.utils.losses import ppo_loss
from verl.workers.utils.padding import no_padding_2_padding

DistillationLossFn = Callable[
    [
        ActorConfig,  # actor_config
        DistillationConfig,  # distillation_config
        dict,  # model_output
        TensorDict,  # micro batch input
    ],
    tuple[torch.Tensor, dict[str, Any]],
]


def is_distillation_enabled(config: Optional[DistillationConfig]) -> bool:
    """Check if distillation is enabled based on the provided configuration."""
    if config is None:
        return False
    return config.enabled


@dataclass
class DistillationLossSettings(BaseConfig):
    """
    Settings for a distillation loss function to be registered.

    Args:
        names (str | list[str]): Name(s) to register the distillation loss function under.
        use_topk (bool): Whether the loss function uses top-k log probabilities.
        use_estimator (bool): Whether the loss function uses single-sample KL estimators.
    """

    names: str | list[str] = field(default_factory=list)
    use_topk: bool = False
    use_estimator: bool = False

    _mutable_fields = {"names"}

    def __post_init__(self):
        self.names = [self.names] if isinstance(self.names, str) else self.names
        if sum([self.use_topk, self.use_estimator]) != 1:
            raise ValueError(
                f"Expected only one of use_estimator, use_topk, but got {self.use_estimator=}, {self.use_topk=}."
            )


DISTILLATION_LOSS_REGISTRY: dict[str, DistillationLossFn] = {}
DISTILLATION_SETTINGS_REGISTRY: dict[str, DistillationLossSettings] = {}


def register_distillation_loss(
    loss_settings: DistillationLossSettings,
) -> Callable[[DistillationLossFn], DistillationLossFn]:
    """Register a distillation loss function with the given name."""

    def decorator(func: DistillationLossFn) -> DistillationLossFn:
        for name in loss_settings.names:
            if name in DISTILLATION_LOSS_REGISTRY:
                raise ValueError(f"Distillation loss function with name '{name}' is already registered.")
            DISTILLATION_LOSS_REGISTRY[name] = func
            DISTILLATION_SETTINGS_REGISTRY[name] = loss_settings
        return func

    return decorator


def get_distillation_loss_fn(loss_name: str) -> DistillationLossFn:
    """Get the distillation loss function with a given name."""
    if loss_name not in DISTILLATION_LOSS_REGISTRY:
        raise ValueError(
            f"Unsupported loss mode: {loss_name}. Supported modes are: {list(DISTILLATION_LOSS_REGISTRY.keys())}"
        )
    return DISTILLATION_LOSS_REGISTRY[loss_name]


def get_distillation_loss_settings(loss_name: str) -> DistillationLossSettings:
    """Get the distillation loss settings with a given name."""
    if loss_name not in DISTILLATION_SETTINGS_REGISTRY:
        raise ValueError(
            f"Unsupported loss mode: {loss_name}. Supported modes are: {list(DISTILLATION_SETTINGS_REGISTRY.keys())}"
        )
    return DISTILLATION_SETTINGS_REGISTRY[loss_name]


def _distillation_response_mask(data: TensorDict) -> torch.Tensor:
    """Return the padded assistant-token mask for rows accepted for training.

    ``response_mask`` also serves rollout diagnostics, so infrastructure filtering
    deliberately leaves it unchanged. Distillation must apply the same optional
    sequence-level filter as PPO without modifying the original data.
    """
    response_mask = data["response_mask"]
    if response_mask.is_nested:
        response_mask = response_mask.to_padded_tensor(False)
    response_mask = response_mask.bool()
    train_sample_mask = data.get("train_sample_mask")
    if train_sample_mask is not None:
        if train_sample_mask.ndim != 1 or train_sample_mask.shape[0] != response_mask.shape[0]:
            raise ValueError("train_sample_mask must have shape [batch_size] for distillation")
        row_mask = train_sample_mask.to(device=response_mask.device, dtype=torch.bool).unsqueeze(-1)
        response_mask = response_mask & row_mask
    return response_mask


def _teacher_sequence_mask(data: TensorDict, response_mask: torch.Tensor) -> torch.Tensor:
    """Return explicit sequence-score routing; never infer it from zero targets."""
    route = data.get("teacher_sequence_mask")
    if route is None or route.ndim != 1 or route.shape[0] != response_mask.shape[0]:
        raise ValueError("teacher_sequence_mask must have shape [batch_size] for audio sequence distillation")
    if route.dtype != torch.bool:
        raise ValueError("teacher_sequence_mask must be boolean")
    return route.to(device=response_mask.device)


def _opd_tool_failure_weights(config: ActorConfig, loss_config: DistillationLossConfig, data: TensorDict):
    """Reuse V1's verified local costs only when task PPO is absent.

    Mixed distillation already adds this cost through ``ppo_loss``. Keep that
    existing path unchanged, including its task-advantage credit gating.
    """
    if loss_config.use_task_rewards or not getattr(config, "tool_failure_penalty_enabled", False):
        return None
    if not loss_config.use_policy_gradient:
        raise ValueError("Tool-failure penalty in pure OPD requires use_policy_gradient=True")
    if config.policy_loss.get("loss_mode", "vanilla") != "vanilla":
        raise ValueError("Tool-failure penalty currently supports vanilla PPO only")
    active_global_count = tu.get_non_tensor_data(data, "tool_failure_active_global_count", default=None)
    if active_global_count is None or "tool_failure_weights" not in data:
        raise ValueError("Enabled tool-failure penalty requires V1 minibatch cost fields")
    weights = data["tool_failure_weights"]
    weights = weights.to_padded_tensor(0.0) if weights.is_nested else weights
    if loss_config.audio_sequence_surrogate:
        # Costs describe SWE tool calls. Stale or padding metadata must never
        # penalize a single-turn audio response or gate its teacher reward.
        route = _teacher_sequence_mask(data, _distillation_response_mask(data))
        weights = weights.masked_fill(route.unsqueeze(-1), 0.0)
    return weights


def compute_distillation_loss_range(
    distillation_losses: torch.Tensor, response_mask: torch.Tensor
) -> dict[str, Metric]:
    """Compute min and max distillation loss over valid response tokens."""
    if response_mask.is_nested:
        distillation_losses_response = distillation_losses[response_mask.bool().to_padded_tensor(False)]
    else:
        distillation_losses_response = distillation_losses[response_mask.bool()]
    if distillation_losses_response.numel() == 0:
        # Keep metric counts aligned across DP ranks for filtered microbatches.
        return {
            "distillation/loss_min": Metric(AggregationType.MIN, 0.0),
            "distillation/loss_max": Metric(AggregationType.MAX, 0.0),
        }
    return {
        "distillation/loss_min": Metric(AggregationType.MIN, distillation_losses_response.min()),
        "distillation/loss_max": Metric(AggregationType.MAX, distillation_losses_response.max()),
    }


def compute_topk_loss(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    data: TensorDict,
    student_logits: torch.Tensor,
    data_format: str,
) -> torch.Tensor:
    """Compute the topk loss in logit processor.

    Returns:
    - distillation_losses: (bsz, seqlen/cp_size)
    - student_mass: (bsz, seqlen/cp_size)
    - teacher_mass: (bsz, seqlen/cp_size)
    """
    match config.strategy:
        # VeOmni uses FSDP2 internally, so its loss computation is identical to FSDP.
        case "fsdp" | "veomni":
            import verl.trainer.distillation.fsdp.losses as fsdp_losses

            distillation_loss_fn = fsdp_losses.compute_forward_kl_topk
        case "megatron":
            import verl.trainer.distillation.megatron.losses as megatron_losses

            distillation_loss_fn = megatron_losses.compute_forward_kl_topk
        case _:
            raise NotImplementedError(f"Unsupported strategy: {config.strategy=}")

    outputs = distillation_loss_fn(
        student_logits=student_logits,
        teacher_topk_log_probs=data["teacher_logprobs"],
        teacher_topk_ids=data["teacher_ids"],
        config=distillation_config,
        data_format=data_format,
    )

    expected_shape = student_logits.shape[:2]
    for k, v in outputs.items():
        assert v.shape == expected_shape, f"Expected shape {expected_shape}, but got {v.shape} for {k=}."

    return outputs


def distillation_ppo_loss(
    config: ActorConfig,
    distillation_config: Optional[DistillationConfig],
    model_output: dict = None,
    data: TensorDict = None,
    dp_group=None,
    student_logits: torch.Tensor = None,
    data_format: str = "thd",
):
    """Loss function used both for logit processor and final policy loss.
    - student_logits is not None, compute the topk loss in logit processor.
    - student_logits is None, compute final policy loss.

    [split sequence across sp/cp groups]
                   |
    [model forward and output logits: (bsz, seqlen/cp_size, vocab_size/tp_size)]
                   |
    [logits processor compute topk loss: (bsz, seqlen/cp_size)]
                   |
    [all gather topk loss across sp/cp groups: (bsz, seqlen)]
                   |
    [combine topk loss with policy loss]

    Args:
        config: Actor configuration.
        distillation_config: Distillation configuration.
        model_output: Model output, including log_probs, entropy.
        data: Micro input batch, contains
          - teacher_logprobs: (bsz, seqlen, topk)
          - teacher_ids: (bsz, seqlen, topk)
        student_logits: (bsz, seqlen/cp_size, vocab_size/tp_size).
        data_format: "thd" or "bshd", models not support THD format, e.g GPT-OSS, Qwen3.5

    Returns:
    - student_logits is not None, return the topk loss tensor (bsz, seqlen/cp_size).
    - student_logits is None, return the final policy loss scalar and metrics.
    """

    # Called as logits processor
    if student_logits is not None:
        return compute_topk_loss(config, distillation_config, data, student_logits, data_format)

    # Called as final policy loss
    distillation_loss_config = distillation_config.distillation_loss
    distill_loss, distill_metrics = distillation_loss(config, distillation_config, model_output, data)
    if not distillation_loss_config.use_task_rewards:
        # PG distillation supplies the policy objective. The optional local cost
        # stays separate from task rewards and uses the existing V1 normalizer.
        policy_loss = 0.0
        policy_metrics = {}
        tool_failure_weights = _opd_tool_failure_weights(config, distillation_loss_config, data)
        if tool_failure_weights is not None:
            from verl.utils.swe_tool_failure import local_ppo_loss

            log_prob = no_padding_2_padding(model_output["log_probs"], data)
            old_log_prob = data["old_log_probs"]
            if old_log_prob.is_nested:
                old_log_prob = old_log_prob.to_padded_tensor(0.0)
            rollout_is_weights = data.get("rollout_is_weights")
            if rollout_is_weights is not None and rollout_is_weights.is_nested:
                rollout_is_weights = rollout_is_weights.to_padded_tensor(0.0)
            cost_mask = _distillation_response_mask(data)
            if distillation_loss_config.audio_sequence_surrogate:
                cost_mask = cost_mask & ~_teacher_sequence_mask(data, cost_mask).unsqueeze(-1)
                log_prob = torch.where(cost_mask, log_prob, 0.0)
                old_log_prob = torch.where(cost_mask, old_log_prob, 0.0)
                if rollout_is_weights is not None:
                    rollout_is_weights = torch.where(cost_mask, rollout_is_weights, 0.0)
            policy_loss = local_ppo_loss(
                log_prob,
                old_log_prob,
                tool_failure_weights,
                cost_mask,
                config,
                active_global_count=tu.get_non_tensor_data(data, "tool_failure_active_global_count", default=None),
                dp_size=data["dp_size"],
                rollout_is_weights=rollout_is_weights,
            )
            policy_metrics["actor/tool_failure_loss"] = Metric(value=policy_loss, aggregation=AggregationType.SUM)
    else:
        policy_loss, policy_metrics = ppo_loss(config, model_output, data, dp_group)

    # Combine distillation with policy loss
    policy_metrics.update(distill_metrics)
    distillation_loss_coef = (
        distillation_loss_config.distillation_loss_coef if distillation_loss_config.use_task_rewards else 1.0
    )
    policy_loss += distill_loss * distillation_loss_coef
    policy_metrics["distillation/loss"] = Metric(value=distill_loss, aggregation=AggregationType.SUM)

    return policy_loss, policy_metrics


def _audio_sequence_distillation_loss(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output: dict,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Combine SWE token k1 with an explicitly different audio sequence objective.

    For completed audio response y, T is the teacher's native-token assistant
    log-likelihood including its verified termination token. P_old and P_theta
    sum the student's assistant-token log-probabilities, including the emitted
    termination token. The stopped reward is R = scale * (T - P_old). We apply
    sequence PPO to R, with ratio exp(P_theta - P_old), and no reward baseline.
    This estimates a teacher-score plus student-entropy surrogate; different
    tokenizers mean it is not a token-aligned KL or exact text-distribution KL.

    Both routes use the accepted global sequence denominator. Consequently the
    combined loss equals each route's mean weighted by its realized prompt
    share when rollout n is the same for both routes. Audio is never divided by
    token length; SWE retains seq-mean-token-mean and its existing local costs.
    The optional SWE tool cost remains a separate auxiliary objective with its
    existing SWE active-trajectory denominator; it is not part of this mixture.
    """
    loss_config = distillation_config.distillation_loss
    if config.loss_agg_mode != "seq-mean-token-mean":
        raise ValueError("audio_sequence_surrogate requires loss_agg_mode=seq-mean-token-mean")
    response_mask = _distillation_response_mask(data)
    route = _teacher_sequence_mask(data, response_mask)
    audio_mask = response_mask & route.unsqueeze(-1)
    text_mask = response_mask & ~route.unsqueeze(-1)
    audio_rows = audio_mask.any(dim=-1)
    student = no_padding_2_padding(model_output["log_probs"], data).float()
    old = data["old_log_probs"]
    old = (old.to_padded_tensor(0.0) if old.is_nested else old).detach().float()
    if student.shape != response_mask.shape or old.shape != response_mask.shape:
        raise ValueError("Student current/old log probabilities must match the assistant response mask")
    if not torch.isfinite(student[response_mask]).all() or not torch.isfinite(old[response_mask]).all():
        raise ValueError("Audio sequence distillation requires finite accepted student log probabilities")
    student = torch.where(response_mask, student, 0.0)
    old = torch.where(response_mask, old, 0.0)

    # These are optimizer-minibatch normalizers prepared before microbatching,
    # including DP reduction. Do not replace them with a microbatch row count.
    denominator = tu.get_non_tensor_data(data, "global_batch_size", default=None)
    dp_size = tu.get_non_tensor_data(data, "dp_size", default=None)
    if denominator is None or dp_size is None or denominator <= 0:
        raise ValueError("Audio sequence distillation requires positive global minibatch normalization")
    global_info = dict(
        dp_size=dp_size,
        batch_num_tokens=data["batch_num_tokens"],
        global_batch_size=denominator,
        loss_scale_factor=config.loss_scale_factor,
    )
    loss_config.global_batch_info.update(global_info)

    # Index only accepted SWE tokens: the audio route's padded token targets
    # are transport placeholders, and must not be interpreted as evidence.
    text_losses = torch.zeros_like(student)
    if text_mask.any():
        if "teacher_logprobs" not in data:
            raise ValueError("SWE rows require their token-aligned teacher_logprobs")
        teacher = no_padding_2_padding(data["teacher_logprobs"], data).squeeze(-1).float()
        if teacher.shape != student.shape or not torch.isfinite(teacher[text_mask]).all():
            raise ValueError("SWE teacher log probabilities must be finite and aligned with accepted tokens")
        text_losses[text_mask] = student[text_mask] - teacher[text_mask]
    metrics = compute_distillation_loss_range(text_losses, text_mask)
    metrics["distillation/abs_loss"] = Metric(
        AggregationType.MEAN, text_losses[text_mask].abs().mean().detach() if text_mask.any() else 0.0
    )
    if loss_config.loss_max_clamp is not None:
        text_losses = text_losses.clamp(-loss_config.loss_max_clamp, loss_config.loss_max_clamp)
    text_advantages = -text_losses.detach()
    tool_weights = _opd_tool_failure_weights(config, loss_config, data)
    if tool_weights is not None:
        from verl.utils.swe_tool_failure import gate_positive_credit

        text_advantages = gate_positive_credit(text_advantages, tool_weights)
    importance = data.get("rollout_is_weights")
    if importance is not None and importance.is_nested:
        importance = importance.to_padded_tensor(0.0)
    if importance is not None:
        if importance.shape != response_mask.shape or not torch.isfinite(importance[response_mask]).all():
            raise ValueError("Rollout importance weights must be finite and aligned for accepted tokens")
        if not torch.all(importance[audio_mask] == 1):
            raise ValueError("The audio sequence surrogate pilot requires no rollout importance correction for audio")
        importance = torch.where(response_mask, importance, 0.0)
    if text_mask.any():
        text_loss, text_metrics = get_policy_loss_fn(loss_config.policy_loss_mode)(
            old_log_prob=old,
            log_prob=student,
            advantages=text_advantages,
            response_mask=text_mask,
            loss_agg_mode=config.loss_agg_mode,
            config=loss_config,
            rollout_is_weights=importance,
        )
        metrics.update({f"distillation/{key.removeprefix('actor/')}": value for key, value in text_metrics.items()})
    else:
        text_loss = student.masked_select(text_mask).sum()
        metrics.update({f"distillation/{key}": 0.0 for key in ("pg_clipfrac", "ppo_kl", "pg_clipfrac_lower")})

    # torch.where excludes masked padding/observations rather than multiplying
    # them by zero (0 * NaN would contaminate even an excluded trajectory).
    current_sequence = torch.where(audio_mask, student, 0.0).sum(dim=-1)[audio_rows]
    old_sequence = torch.where(audio_mask, old, 0.0).sum(dim=-1)[audio_rows]
    teacher_sequence = data.get("teacher_sequence_logprob")
    if teacher_sequence is None or teacher_sequence.ndim != 1 or teacher_sequence.shape != route.shape:
        raise ValueError("teacher_sequence_logprob must have shape [batch_size]")
    teacher_sequence = teacher_sequence.to(device=student.device).float()[audio_rows].detach()
    if not torch.isfinite(teacher_sequence).all():
        raise ValueError("Accepted audio teacher sequence scores must be finite")
    reward = (teacher_sequence - old_sequence).detach() * loss_config.audio_sequence_reward_scale
    log_ratio = current_sequence - old_sequence
    # The same numerical guard as vanilla token PPO; PPO's configured clip is
    # applied separately below to the ratio of whole response probabilities.
    ratio = log_ratio.clamp(-20.0, 20.0).exp()
    clipped_ratio = ratio.clamp(1 - loss_config.audio_sequence_clip_ratio, 1 + loss_config.audio_sequence_clip_ratio)
    unclipped_losses = -reward * ratio
    clipped_losses = -reward * clipped_ratio
    clipped_active = clipped_losses > unclipped_losses
    sequence_losses = torch.maximum(unclipped_losses, clipped_losses)
    audio_loss = sequence_losses.sum() * dp_size / denominator
    # Empty selections retain the model's graph, including on audio-only and
    # fully excluded microbatches/ranks.
    total_loss = text_loss + audio_loss
    audio_count = tu.get_non_tensor_data(data, "audio_sequence_global_count", default=denominator)
    for key, values in {
        "reward": reward,
        "teacher_sequence_logprob": teacher_sequence,
        "old_student_sequence_logprob": old_sequence,
        "sequence_ratio": ratio,
        "clip_fraction": clipped_active.float(),
    }.items():
        metrics[f"distillation/audio/{key}"] = Metric(
            AggregationType.SUM, values.detach().sum() * dp_size / max(1, audio_count)
        )
    metrics["distillation/audio/accepted_sequences"] = Metric(
        AggregationType.SUM, int(audio_rows.sum().item()) * dp_size
    )
    metrics["distillation/swe/accepted_sequences"] = Metric(
        AggregationType.SUM, int(text_mask.any(dim=-1).sum().item()) * dp_size
    )
    metrics["distillation/audio/finite_teacher_sequences"] = Metric(
        AggregationType.SUM, int(torch.isfinite(teacher_sequence).sum().item()) * dp_size
    )
    metrics["distillation/audio/clipped_sequences"] = Metric(
        AggregationType.SUM, int(clipped_active.sum().item()) * dp_size
    )
    metrics["distillation/audio/loss"] = Metric(AggregationType.SUM, audio_loss.detach())
    metrics["distillation/swe/loss"] = Metric(AggregationType.SUM, text_loss.detach())
    return total_loss, metrics


def distillation_loss(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output: dict,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the distillation loss and related metrics.

    Returns:
    - distillation_loss: Aggregated distillation loss scalar.
    - distillation_metrics: Dictionary of metrics.
    """
    assert distillation_config is not None
    loss_config: DistillationLossConfig = distillation_config.distillation_loss
    if loss_config.audio_sequence_surrogate:
        return _audio_sequence_distillation_loss(config, distillation_config, model_output, data)
    sequence_route = data.get("teacher_sequence_mask")
    if sequence_route is not None and sequence_route.any():
        raise ValueError("Audio sequence teacher scores require audio_sequence_surrogate=True")
    tool_failure_weights = _opd_tool_failure_weights(config, loss_config, data)
    distillation_loss_fn = get_distillation_loss_fn(loss_config.loss_mode)
    distillation_losses, distillation_metrics = distillation_loss_fn(
        config=config,
        distillation_config=distillation_config,
        model_output=model_output,
        data=data,
    )
    response_mask = _distillation_response_mask(data)
    loss_agg_mode = config.loss_agg_mode

    distillation_metrics.update(
        compute_distillation_loss_range(distillation_losses=distillation_losses, response_mask=response_mask)
    )
    if loss_config.loss_max_clamp is not None:
        # clamping min is for k1 loss which can be negative
        distillation_losses = distillation_losses.clamp(min=-loss_config.loss_max_clamp, max=loss_config.loss_max_clamp)

    if not response_mask.any():
        # Preserve the autograd graph while avoiding zero local denominators and
        # min/max reductions on empty tensors. Other microbatches/ranks may train.
        if loss_config.use_policy_gradient:
            distillation_metrics.update(
                {f"distillation/{key}": 0.0 for key in ("pg_clipfrac", "ppo_kl", "pg_clipfrac_lower")}
            )
        return distillation_losses.masked_select(response_mask).sum(), distillation_metrics

    # global batch info for loss aggregation
    config.global_batch_info["dp_size"] = data["dp_size"]
    config.global_batch_info["batch_num_tokens"] = data["batch_num_tokens"]
    config.global_batch_info["global_batch_size"] = data["global_batch_size"]
    config.global_batch_info["loss_scale_factor"] = config.loss_scale_factor

    if loss_config.use_policy_gradient:
        # Use negative distillation loss as reward, as done by https://thinkingmachines.ai/blog/on-policy-distillation/.
        policy_loss_fn = get_policy_loss_fn(loss_config.policy_loss_mode)
        for k, v in config.global_batch_info.items():
            loss_config.global_batch_info[k] = v
        log_prob = no_padding_2_padding(model_output["log_probs"], data)
        old_log_prob = data["old_log_probs"]
        if old_log_prob.is_nested:
            old_log_prob = data["old_log_probs"].to_padded_tensor(0.0)
        rollout_is_weights = data.get("rollout_is_weights", None)
        if rollout_is_weights is not None and rollout_is_weights.is_nested:
            rollout_is_weights = rollout_is_weights.to_padded_tensor(0.0)
        advantages = -distillation_losses.detach()
        if tool_failure_weights is not None:
            from verl.utils.swe_tool_failure import gate_positive_credit

            advantages = gate_positive_credit(advantages, tool_failure_weights)
        distillation_loss, pg_metrics = policy_loss_fn(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=response_mask,
            loss_agg_mode=loss_agg_mode,
            config=loss_config,
            rollout_is_weights=rollout_is_weights,
        )
        pg_metrics = {f"distillation/{k[len('actor/') :]}": v for k, v in pg_metrics.items()}
        distillation_metrics.update(pg_metrics)
    else:
        # Directly backpropagate distillation loss as a supervised loss, as in https://arxiv.org/abs/2306.13649.
        distillation_loss = agg_loss(
            loss_mat=distillation_losses,
            loss_mask=response_mask,
            loss_agg_mode=loss_agg_mode,
            **config.global_batch_info,
        )

    return distillation_loss, distillation_metrics


@register_distillation_loss(DistillationLossSettings(names=["forward_kl_topk"], use_topk=True))  # type: ignore[arg-type]
def compute_forward_kl_topk(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output: dict,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute forward KL distillation loss and related metrics using top-k log probabilities.

    Returns:
    - distillation_losses: (bsz, resp_len)
    - distillation_metrics: Dictionary of metrics.
    """
    # topk loss has been computed in logits processor
    distillation_losses = no_padding_2_padding(model_output["distillation_losses"], data)
    student_mass = no_padding_2_padding(model_output["student_mass"], data)
    teacher_mass = no_padding_2_padding(model_output["teacher_mass"], data)
    overlap_count = model_output.get("overlap_count")
    overlap_token_advantage = model_output.get("overlap_token_advantage")
    if overlap_count is not None and overlap_token_advantage is not None:
        overlap_count = no_padding_2_padding(overlap_count, data)
        overlap_token_advantage = no_padding_2_padding(overlap_token_advantage, data)
    response_mask_bool = _distillation_response_mask(data)
    assert distillation_losses.shape == student_mass.shape == teacher_mass.shape == response_mask_bool.shape

    if not response_mask_bool.any():
        metrics = {
            "distillation/student_mass": 0.0,
            "distillation/student_mass_min": Metric(AggregationType.MIN, 0.0),
            "distillation/student_mass_max": Metric(AggregationType.MAX, 0.0),
            "distillation/teacher_mass": 0.0,
            "distillation/teacher_mass_min": Metric(AggregationType.MIN, 0.0),
            "distillation/teacher_mass_max": Metric(AggregationType.MAX, 0.0),
        }
        if overlap_count is not None and overlap_token_advantage is not None:
            metrics.update({"distillation/overlap_ratio": 0.0, "distillation/overlap_token_advantage": 0.0})
        return distillation_losses.clamp_min(0.0), metrics

    overlap_metrics = {}
    if overlap_count is not None and overlap_token_advantage is not None:
        assert overlap_count.shape == overlap_token_advantage.shape == response_mask_bool.shape
        valid_overlap_count = overlap_count[response_mask_bool]
        k = distillation_config.distillation_loss.topk
        assert k is not None
        # Diagnostics for tracking teacher/student top-k overlap in OPD, following
        # "Rethinking On-Policy Distillation of Large Language Models" (arXiv:2604.13016):
        # overlap ratio and average teacher-token KL contribution on overlapped tokens.
        overlap_metrics["distillation/overlap_ratio"] = (valid_overlap_count.float().mean() / k).item()
        overlap_position_mask = response_mask_bool & (overlap_count > 0)
        if overlap_position_mask.any():
            overlap_metrics["distillation/overlap_token_advantage"] = (
                overlap_token_advantage[overlap_position_mask].mean().item()
            )
        else:
            overlap_metrics["distillation/overlap_token_advantage"] = 0.0

    # Log amount of mass in the top-k log probabilities for both student and teacher.
    student_mass = student_mass[response_mask_bool]
    teacher_mass = teacher_mass[response_mask_bool]
    distillation_metrics = {
        "distillation/student_mass": student_mass.mean().item(),
        "distillation/student_mass_min": Metric(AggregationType.MIN, student_mass.min()),
        "distillation/student_mass_max": Metric(AggregationType.MAX, student_mass.max()),
        "distillation/teacher_mass": teacher_mass.mean().item(),
        "distillation/teacher_mass_min": Metric(AggregationType.MIN, teacher_mass.min()),
        "distillation/teacher_mass_max": Metric(AggregationType.MAX, teacher_mass.max()),
        **overlap_metrics,
    }

    # Due to use of top-k, student and teacher distributions don't sum to 1 -> divergences can be negative.
    distillation_losses = distillation_losses.clamp_min(0.0)

    return distillation_losses, distillation_metrics


@register_distillation_loss(
    DistillationLossSettings(names=["kl", "k1", "abs", "mse", "k2", "low_var_kl", "k3"], use_estimator=True)
)  # type: ignore[arg-type]
def compute_distillation_loss_reverse_kl_estimator(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the distillation loss and related metrics using single-sample KL estimators.

    Uses the kl_penalty function from core_algos which supports various KL divergence
    estimators: "kl", "k1", "abs", "mse", "k2", "low_var_kl", "k3".

    Returns:
    - distillation_losses: (bsz, resp_len)
    - distillation_metrics: Dictionary of metrics.
    """
    student_log_probs = no_padding_2_padding(model_output["log_probs"], data)
    teacher_log_probs = no_padding_2_padding(data["teacher_logprobs"], data).squeeze(-1)
    response_mask_bool = _distillation_response_mask(data)
    assert teacher_log_probs.shape == student_log_probs.shape == response_mask_bool.shape

    loss_config: DistillationLossConfig = distillation_config.distillation_loss
    distillation_losses = kl_penalty(
        logprob=student_log_probs, ref_logprob=teacher_log_probs, kl_penalty=loss_config.loss_mode
    )
    # Since k1 can be negative, log the mean absolute loss.
    mean_abs_loss = distillation_losses[response_mask_bool].abs().mean() if response_mask_bool.any() else 0.0
    metrics = {"distillation/abs_loss": Metric(AggregationType.MEAN, mean_abs_loss)}
    return distillation_losses, metrics

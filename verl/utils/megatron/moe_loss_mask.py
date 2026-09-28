# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""Sequence exclusions for native Megatron router losses.

Keep the existing router reductions and microbatch averaging. A rejected
sequence owns its whole padded THD segment, including its alignment padding.
Retained segments keep the previous padding policy. No attention mask, route,
loss coefficient, or process-global auxiliary backward scale is modified.
"""

import inspect
from types import MethodType

import torch


def validate_train_sample_mask(mask: torch.Tensor, batch_size: int) -> torch.Tensor:
    if mask.ndim != 1 or mask.shape[0] != batch_size:
        raise ValueError(f"train_sample_mask must have shape [{batch_size}], got {tuple(mask.shape)}")
    if not torch.all((mask == 0) | (mask == 1)).item():
        raise ValueError("train_sample_mask must be binary")
    return mask.to(torch.bool)


def packed_moe_router_mask(
    train_sample_mask: torch.Tensor | None,
    cu_seqlens_padded: torch.Tensor,
    *,
    cp_size: int,
    cp_rank: int,
    cp_layout: str = "zigzag",
    existing_mask: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Build [1, CP-local tokens] exclusions from the actual packing metadata.

    SP scattering is left to Megatron, exactly as for its native padding mask.
    In zigzag CP each rank owns an equal-sized portion of every padded sequence;
    in contiguous CP each rank owns one interval of the global packed buffer.
    """
    if train_sample_mask is None:
        return existing_mask
    keep = validate_train_sample_mask(train_sample_mask, cu_seqlens_padded.numel() - 1)
    if keep.all().item():
        return existing_mask  # Preserve the all-retained path, including padding semantics.
    if cp_size < 1 or not 0 <= cp_rank < cp_size:
        raise ValueError(f"Invalid context parallel coordinates: size={cp_size}, rank={cp_rank}")
    lengths = cu_seqlens_padded.diff().to(torch.long)
    if lengths.lt(0).any().item() or cu_seqlens_padded[0].item() != 0:
        raise ValueError("Invalid padded sequence offsets")
    excluded = (~keep).to(device=lengths.device)
    if cp_layout == "zigzag":
        if lengths.remainder(cp_size).any().item():
            raise ValueError("Each padded sequence must be divisible by CP size")
        mask = torch.repeat_interleave(excluded, lengths // cp_size)
    elif cp_layout == "contiguous":
        mask = torch.repeat_interleave(excluded, lengths)
        if mask.numel() % cp_size:
            raise ValueError("Padded token count must be divisible by CP size")
        local_size = mask.numel() // cp_size
        mask = mask[cp_rank * local_size : (cp_rank + 1) * local_size]
    else:
        raise ValueError(f"Unsupported CP layout: {cp_layout}")
    mask = mask.unsqueeze(0)
    if existing_mask is not None:
        if existing_mask.shape != mask.shape:
            raise ValueError(f"Router mask shape mismatch: {tuple(existing_mask.shape)} vs {tuple(mask.shape)}")
        mask = mask | existing_mask.to(device=mask.device, dtype=torch.bool)
    return mask


def validate_moe_loss_mask_config(engine_config, model_config, tf_config=None) -> None:
    """Fail before training on combinations whose mask/scaling path is untested."""
    if not getattr(engine_config, "moe_loss_respects_train_sample_mask", False):
        return
    if engine_config.dynamic_context_parallel or not engine_config.use_remove_padding:
        raise ValueError("MoE train-sample masking requires static CP and use_remove_padding=True")
    if engine_config.pipeline_model_parallel_size != 1:
        raise ValueError("MoE train-sample masking currently requires pipeline_model_parallel_size=1")
    if model_config.model_type != "language_model" or model_config.mtp.enable:
        raise ValueError("MoE train-sample masking requires a language model with MTP disabled")
    if tf_config is None:
        return
    if hasattr(getattr(model_config, "hf_config", None), "vision_config") and (
        getattr(tf_config, "fp8", None) not in (None, False) or getattr(engine_config, "pad_to_length", False)
    ):
        raise ValueError("MoE train-sample masking does not support VLM repacking with FP8 or bucket padding")
    if tf_config.calculate_per_token_loss:
        raise ValueError("MoE train-sample masking does not yet support calculate_per_token_loss=True")
    if tf_config.moe_token_dispatcher_type != "alltoall" or tf_config.moe_expert_capacity_factor is not None:
        raise ValueError("MoE train-sample masking requires the dropless alltoall dispatcher")
    if getattr(tf_config, "moe_expert_rank_capacity_factor", None) is not None:
        raise ValueError("MoE train-sample masking does not support expert rank capacity limits")
    if getattr(tf_config, "moe_router_enable_expert_bias", False) or getattr(tf_config, "moe_n_hash_layers", 0):
        raise ValueError("MoE train-sample masking does not support expert-bias updates or hash routing")
    modes = tf_config.moe_router_load_balancing_type
    modes = [modes] if isinstance(modes, str) else modes
    if not modes or any(mode not in {"none", "aux_loss"} for mode in modes):
        raise ValueError("MoE train-sample masking currently supports router modes 'none' and 'aux_loss'")
    if getattr(tf_config, "moe_router_fusion", False):
        raise ValueError("MoE train-sample masking currently requires moe_router_fusion=False")


def _apply_masked_aux_loss(
    self, probs, scores_for_aux_loss, routing_map, with_padding_mask=False, packed_seq_params=None
):
    """Native standard balancing loss, with a safe empty-domain denominator.

    Keep all native reductions/logging even when this rank has no eligible
    tokens. Clamping the reduced denominator avoids 0/0 when every rank in the
    auxiliary-loss domain is empty; masked scores and expert counts are zero.
    """
    if not with_padding_mask:
        return self._verl_original_aux_loss(
            probs, scores_for_aux_loss, routing_map, with_padding_mask, packed_seq_params
        )
    from megatron.core.transformer.moe.router import (
        get_tokens_per_expert_and_token_count,
        switch_load_balancing_loss_func,
    )

    coeff = self.get_aux_loss_coeff("aux_loss")
    if coeff == 0:
        return probs
    groups = self._get_aux_loss_groups(packed_seq_params)
    counts, local_tokens, total_tokens = get_tokens_per_expert_and_token_count(
        routing_map=routing_map,
        reduce_group=groups.loss_reduce_groups[0],
        reduce_groups=groups.loss_reduce_groups,
        topk=self.topk,
        with_padding_mask=True,
    )
    denominator = total_tokens.clamp_min(1) if torch.is_tensor(total_tokens) else max(total_tokens, 1)
    loss = switch_load_balancing_loss_func(
        probs=scores_for_aux_loss,
        tokens_per_expert=counts,
        total_num_tokens=denominator,
        topk=self.topk,
        num_experts=self.config.num_moe_experts,
        moe_aux_loss_coeff=coeff,
        fused=False,
    )
    return self.attach_and_log_load_balancing_loss(
        probs,
        coeff,
        loss,
        "load_balancing_loss",
        groups.metric_reduce_group,
        avg_group=groups.metric_avg_group,
        needs_dp_avg=groups.metric_needs_dp_avg,
        valid_token_count=local_tokens,
        aux_loss_logging_reduce_groups=groups.metric_pre_reduce_groups,
        aux_loss_scale_reduce_groups=groups.loss_reduce_groups,
        aux_loss_scale_num_tokens=total_tokens,
    )


def install_moe_loss_mask_support(modules) -> None:
    """Validate native support and install the balancing guard on this actor only."""
    from megatron.core.transformer.moe.moe_layer import MoELayer
    from megatron.core.transformer.moe.router import TopKRouter

    if "padding_mask" not in inspect.signature(MoELayer.forward).parameters:
        raise RuntimeError("Installed Megatron MoELayer does not support router padding masks")
    if "padding_mask" not in inspect.signature(TopKRouter.apply_z_loss).parameters:
        raise RuntimeError("Installed Megatron z-loss does not support router padding masks")
    routers = [m for module in modules for m in module.modules() if isinstance(m, TopKRouter)]
    if not routers:
        raise RuntimeError("MoE train-sample masking was enabled, but no supported TopKRouter was found")
    for router in routers:
        if router.get_aux_loss_coeff("aux_loss") > 0 and not hasattr(router, "_verl_original_aux_loss"):
            required_logging_args = {
                "avg_group",
                "needs_dp_avg",
                "valid_token_count",
                "aux_loss_logging_reduce_groups",
                "aux_loss_scale_reduce_groups",
                "aux_loss_scale_num_tokens",
            }
            if (
                not hasattr(router, "_get_aux_loss_groups")
                or not {"with_padding_mask", "packed_seq_params"}.issubset(
                    inspect.signature(router._apply_aux_loss).parameters
                )
                or not required_logging_args.issubset(
                    inspect.signature(router.attach_and_log_load_balancing_loss).parameters
                )
            ):
                raise RuntimeError("Installed Megatron balancing loss lacks the required masked-reduction API")
            router._verl_original_aux_loss = router._apply_aux_loss
            router._apply_aux_loss = MethodType(_apply_masked_aux_loss, router)


def moe_mask_batch_counts(data, *, device, dp_group) -> torch.Tensor:
    """Globally reduce eligible/total sequences and input tokens over DP.

    All ranks participate, including ranks whose local mask is entirely false.
    TP/CP replicas have the same sequence batch; reducing over DP avoids counting
    those replicas twice. Missing masks mean that every sequence is eligible.
    """
    mask = data.get("train_sample_mask")
    lengths = data["input_ids"].offsets().diff().to(device=device, dtype=torch.long)
    keep = torch.ones(len(data), device=device, dtype=torch.bool)
    if mask is not None:
        keep = validate_train_sample_mask(mask, len(data)).to(device)
    counts = torch.stack((keep.sum(), lengths.new_tensor(len(data)), lengths[keep].sum(), lengths.sum()))
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(counts, group=dp_group)
    return counts

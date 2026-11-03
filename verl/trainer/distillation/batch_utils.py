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
"""Prepare OPD normalizers once per optimizer minibatch, before microbatching."""

import torch

from verl.trainer.distillation.losses import _distillation_response_mask, _teacher_sequence_mask
from verl.trainer.distillation.reasoning_utils import reasoning_row_mask
from verl.utils import tensordict_utils as tu
from verl.workers.utils.padding import response_to_nested


def prepare_distillation_minibatch(data, dp_group=None):
    """Use accepted assistant tokens for engine token and sequence normalizers.

    V1 leaves the diagnostic response_mask unchanged. Updating loss_mask here
    ensures the engine's existing global token count matches the OPD objective.
    Sequence means likewise need the accepted count across the actual minibatch,
    rather than the nominal configured size. Every DP rank participates even
    when none of its local rows is accepted.
    """
    effective_mask = _distillation_response_mask(data)
    active_count = int(effective_mask.any(dim=-1).sum().item())
    route_counts = None
    if "teacher_sequence_mask" in data:
        route = _teacher_sequence_mask(data, effective_mask)
        audio_count = int((effective_mask.any(dim=-1) & route).sum().item())
        route_counts = (audio_count, active_count - audio_count)
        non_swe = route
        if "general_reasoning_mask" in data:
            reasoning = reasoning_row_mask(data, effective_mask)
            if (reasoning & route).any():
                raise ValueError("A reasoning row cannot use the audio sequence teacher")
            general_count = int((effective_mask.any(dim=-1) & reasoning).sum().item())
            route_counts = (audio_count, active_count - audio_count - general_count, general_count)
            non_swe = route | reasoning
        if "tool_failure_normalizer" in data:
            # A shared SWE launcher can enable local tool costs. Audio has no
            # SWE calls and must not dilute that cost's trajectory denominator.
            data["tool_failure_normalizer"] = data["tool_failure_normalizer"].masked_fill(non_swe, 0.0)
    if data["response_mask"].is_nested:
        data["loss_mask"] = response_to_nested(effective_mask, data["response_mask"])
    else:
        data["loss_mask"] = effective_mask
    if torch.distributed.is_initialized():
        counts = [None] * torch.distributed.get_world_size(dp_group)
        if route_counts is None:
            torch.distributed.all_gather_object(counts, active_count, dp_group)
            active_count = sum(counts)
        else:
            torch.distributed.all_gather_object(counts, route_counts, dp_group)
            route_counts = tuple(sum(values[i] for values in counts) for i in range(len(route_counts)))
            active_count = sum(route_counts)
    # Fully rejected batches are skipped by the existing engine safeguards; the
    # finite denominator also protects zero-loss microbatches on other backends.
    tu.assign_non_tensor(data, global_batch_size=max(1, active_count))
    if route_counts is not None:
        tu.assign_non_tensor(
            data,
            audio_sequence_global_count=route_counts[0],
            swe_sequence_global_count=route_counts[1],
        )
        if len(route_counts) == 3:
            tu.assign_non_tensor(data, general_sequence_global_count=route_counts[2])

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

from verl.trainer.distillation.losses import _distillation_response_mask
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
    if data["response_mask"].is_nested:
        data["loss_mask"] = response_to_nested(effective_mask, data["response_mask"])
    else:
        data["loss_mask"] = effective_mask
    if torch.distributed.is_initialized():
        counts = [None] * torch.distributed.get_world_size(dp_group)
        torch.distributed.all_gather_object(counts, active_count, dp_group)
        active_count = sum(counts)
    # Fully rejected batches are skipped by the existing engine safeguards; the
    # finite denominator also protects zero-loss microbatches on other backends.
    tu.assign_non_tensor(data, global_batch_size=max(1, active_count))

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

import numpy as np
import pytest
import torch

from verl import DataProto
from verl.trainer.ppo.ray_trainer import (
    _capture_grpo_boundary,
    _check_grpo_boundary,
    _validate_grpo_response_batch,
)


def test_grpo_boundary_detects_mocked_log_prob_mutation(monkeypatch):
    monkeypatch.setenv("VERL_GRPO_DIAGNOSTICS", "0")
    data = DataProto.from_dict(
        tensors={
            "token_level_scores": torch.tensor([[0.0, 1.0], [0.0, 0.0]]),
            "response_mask": torch.ones(2, 2),
        },
        non_tensors={"uid": np.array(["prompt-a", "prompt-b"], dtype=object)},
    )
    baseline = _capture_grpo_boundary(data)

    def mocked_old_log_prob(batch):
        batch.batch["token_level_scores"][0, 0] = 1.0

    mocked_old_log_prob(data)
    metrics = {}
    _check_grpo_boundary("after_old_log_prob", _capture_grpo_boundary(data), baseline, metrics)

    assert metrics["grpo/boundary_after_old_log_prob_changed"] == 1
    assert metrics["grpo/boundary_mutations"] == 1


def _make_response_batch(response_mask: torch.Tensor) -> tuple[DataProto, torch.Tensor]:
    batch_size, response_width = response_mask.shape
    responses = torch.zeros(batch_size, response_width, dtype=torch.long)
    prompt_attention = torch.ones(batch_size, 2, dtype=torch.long)
    response_attention = (response_mask != 0).to(dtype=torch.long)
    data = DataProto.from_dict(
        tensors={
            "responses": responses,
            "response_mask": response_mask,
            "attention_mask": torch.cat([prompt_attention, response_attention], dim=-1),
        }
    )
    return data, torch.zeros(batch_size, response_width)


def test_grpo_response_batch_accepts_valid_empty_trajectory():
    data, rewards = _make_response_batch(torch.tensor([[0, 0], [1, 0]], dtype=torch.long))

    metrics = _validate_grpo_response_batch(data, rewards)

    assert metrics["grpo/nonbinary_mask_tokens"] == 0
    assert metrics["grpo/empty_response_trajectories"] == 1
    assert metrics["grpo/zero_generation_trajectories"] == 1


def test_grpo_response_batch_rejects_token_pad_id_in_mask():
    data, rewards = _make_response_batch(torch.tensor([[248044, 0]], dtype=torch.long))

    with pytest.raises(ValueError, match="must be binary"):
        _validate_grpo_response_batch(data, rewards)


def test_grpo_response_batch_rejects_reward_on_empty_response():
    data, rewards = _make_response_batch(torch.tensor([[0, 0]], dtype=torch.long))
    rewards[0, -1] = 1.0

    with pytest.raises(ValueError, match="nonzero rewards on empty responses"):
        _validate_grpo_response_batch(data, rewards)

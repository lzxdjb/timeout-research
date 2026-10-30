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
"""Hidden-test GRPO and teacher PG share the existing mixed loss and local cost."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from verl.trainer.distillation.losses import distillation_loss, distillation_ppo_loss
from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage
from verl.utils import tensordict_utils as tu
from verl.utils.swe_tool_failure import gate_positive_credit
from verl.workers.config import ActorConfig, DistillationLossConfig
from verl.workers.utils.losses import ppo_loss


def _hybrid_batch(*, nested, dp_size=1, penalty=True, outcome=None):
    # The second response is shorter. The other masks represent an observation
    # token, an infrastructure-filtered row, and an entirely cut-off row.
    response_lengths = [3, 2, 3, 3]
    mask = torch.tensor([[1, 0, 1], [1, 1, 0], [1, 1, 1], [0, 0, 0]], dtype=torch.bool)
    keep = torch.tensor([True, True, False, True])
    effective_mask = mask & keep.unsqueeze(-1)
    rewards = torch.zeros(4, 3)
    rewards[:, 0] = torch.tensor([1.0, 0.0, 1.0, 1.0] if outcome is None else [outcome] * 4)
    advantages, _ = compute_grpo_outcome_advantage(
        rewards,
        effective_mask,
        np.array(["kept", "kept", "infra", "cutoff"]),
        norm_adv_by_std_in_grpo=False,
        valid_sample_mask=effective_mask.any(-1),
    )
    weights = torch.tensor([[0.02, 0.0, 0.0], [0.0, 0.04, 0.0], [9.0, 9.0, 9.0], [9.0, 9.0, 9.0]])
    if penalty:
        # V1 applies this after hidden-reward advantage construction. Preserve
        # that task-credit policy without gating the mixed teacher objective.
        advantages = gate_positive_credit(advantages, weights)
    prompts = torch.ones(4, 1, dtype=torch.long)
    responses = torch.arange(12).reshape(4, 3)
    old_log_probs = torch.full((4, 3), -2.0)
    importance = torch.tensor([[0.5, 1.0, 1.5], [1.0, 0.5, 1.0], [4.0, 4.0, 4.0], [4.0, 4.0, 4.0]])
    fields = {
        "prompts": prompts,
        "responses": responses,
        "response_mask": mask,
        "old_log_probs": old_log_probs,
        "advantages": advantages,
        "rollout_is_weights": importance,
        "tool_failure_weights": weights,
        "train_sample_mask": keep,
        "teacher_logprobs": torch.nested.as_nested_tensor(
            [torch.full((length + 1, 1), -1.0) for length in response_lengths], layout=torch.jagged
        ),
        "attention_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0], [1, 1, 1, 1], [1, 1, 1, 1]]),
    }
    if nested:
        fields["prompts"] = torch.nested.as_nested_tensor(list(prompts.unbind()), layout=torch.jagged)
        for key in (
            "responses",
            "response_mask",
            "old_log_probs",
            "advantages",
            "rollout_is_weights",
            "tool_failure_weights",
        ):
            fields[key] = torch.nested.as_nested_tensor(
                [row[:length] for row, length in zip(fields[key], response_lengths, strict=True)], layout=torch.jagged
            )
    data = TensorDict(fields, batch_size=[4])
    # Simulate identical data on every DP rank: the global normalizers and DP
    # multiplier must cancel for task, teacher, and local cost alike.
    tu.assign_non_tensor(
        data,
        dp_size=dp_size,
        batch_num_tokens=4 * dp_size,
        global_batch_size=2 * dp_size,
        tool_failure_active_global_count=2 * dp_size,
    )
    student = torch.full((sum(length + 1 for length in response_lengths),), -2.0, requires_grad=True)
    return data, student


def _configs(*, coef=1.0, penalty=True):
    actor = ActorConfig(
        strategy="fsdp2",
        rollout_n=2,
        use_dynamic_bsz=True,
        tool_failure_penalty_enabled=penalty,
        loss_agg_mode="token-mean",
    )
    teacher = SimpleNamespace(
        distillation_loss=DistillationLossConfig(
            loss_mode="k1",
            use_policy_gradient=True,
            use_task_rewards=True,
            distillation_loss_coef=coef,
            loss_max_clamp=None,
        )
    )
    return actor, teacher


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("dp_size", [1, 2])
@pytest.mark.parametrize("coef", [0.0, 0.4, 1.7])
def test_real_mixed_loss_adds_hidden_teacher_and_one_local_cost(nested, dp_size, coef):
    data, student = _hybrid_batch(nested=nested, dp_size=dp_size)
    actor, teacher = _configs(coef=coef)
    loss, metrics = distillation_ppo_loss(actor, teacher, {"log_probs": student}, data)
    gradient = torch.autograd.grad(loss, student, retain_graph=True)[0]

    task_with_cost, task_metrics = ppo_loss(actor, {"log_probs": student}, data)
    teacher_loss, _ = distillation_loss(actor, teacher, {"log_probs": student}, data)
    expected_sum_gradient = torch.autograd.grad(task_with_cost + coef * teacher_loss, student)[0]
    torch.testing.assert_close(gradient, expected_sum_gradient)

    # At ratio=1, the manual gradient is -advantage * importance / 4,
    # plus weights * importance / 2 for the single local-cost term.
    hidden_gradient = torch.zeros_like(student)
    hidden_gradient[[2, 4, 5]] = torch.tensor([-0.1875, 0.125, 0.0625])
    teacher_gradient = torch.zeros_like(student)
    teacher_gradient[[0, 2, 4, 5]] = torch.tensor([-0.125, -0.375, -0.25, -0.125])
    cost_gradient = torch.zeros_like(student)
    cost_gradient[[0, 5]] = torch.tensor([0.005, 0.01])
    torch.testing.assert_close(gradient, hidden_gradient + coef * teacher_gradient + cost_gradient)
    assert hidden_gradient.count_nonzero() > 0
    assert teacher_gradient.count_nonzero() > 0
    assert loss.item() == pytest.approx(coef * -0.875 + 0.015, abs=1e-6)
    assert metrics["actor/tool_failure_loss"].aggregate() == pytest.approx(0.015)
    assert task_metrics["actor/pg_loss"].aggregate() == pytest.approx(0.0)
    assert metrics["distillation/loss"].aggregate() == pytest.approx(-0.875)
    # Observations, trailing predictions, infrastructure rows, and cutoff rows
    # remain excluded from every branch even with large stale local weights.
    excluded = torch.ones_like(student, dtype=torch.bool)
    excluded[[0, 2, 4, 5]] = False
    assert gradient[excluded].eq(0).all()


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("outcome", [0.0, 1.0])
def test_constant_hidden_outcomes_leave_teacher_learning_active(nested, outcome):
    data, student = _hybrid_batch(nested=nested, penalty=False, outcome=outcome)
    actor, teacher = _configs(coef=0.7, penalty=False)
    task_loss, _ = ppo_loss(actor, {"log_probs": student}, data)
    task_gradient = torch.autograd.grad(task_loss, student, retain_graph=True)[0]
    loss, metrics = distillation_ppo_loss(actor, teacher, {"log_probs": student}, data)
    loss.backward()
    assert task_loss.item() == 0.0
    assert task_gradient.eq(0).all()
    assert student.grad.count_nonzero() == 4
    assert loss.item() == pytest.approx(-0.7 * 0.875)
    assert "actor/tool_failure_loss" not in metrics


@pytest.mark.parametrize("nested", [False, True])
def test_mixed_teacher_coefficient_scales_only_teacher_gradient(nested):
    data, student = _hybrid_batch(nested=nested)
    actor, teacher = _configs(coef=0.25)
    low_loss, low_metrics = distillation_ppo_loss(actor, teacher, {"log_probs": student}, data)
    low_gradient = torch.autograd.grad(low_loss, student, retain_graph=True)[0]
    _, teacher = _configs(coef=1.25)
    high_loss, high_metrics = distillation_ppo_loss(actor, teacher, {"log_probs": student}, data)
    high_gradient = torch.autograd.grad(high_loss, student)[0]
    expected_difference = torch.zeros_like(student)
    expected_difference[[0, 2, 4, 5]] = torch.tensor([-0.125, -0.375, -0.25, -0.125])
    torch.testing.assert_close(high_gradient - low_gradient, expected_difference)
    assert high_metrics["actor/tool_failure_loss"].aggregate() == low_metrics["actor/tool_failure_loss"].aggregate()


@pytest.mark.parametrize("nested", [False, True])
def test_fully_rejected_mixed_microbatch_has_zero_finite_loss_and_gradient(nested):
    data, student = _hybrid_batch(nested=nested)
    data["train_sample_mask"] = torch.zeros(4, dtype=torch.bool)
    actor, teacher = _configs()
    loss, metrics = distillation_ppo_loss(actor, teacher, {"log_probs": student}, data)
    loss.backward()
    assert loss.item() == 0.0
    assert student.grad.eq(0).all()
    assert metrics["actor/tool_failure_loss"].aggregate() == 0.0
    assert metrics["distillation/loss"].aggregate() == 0.0

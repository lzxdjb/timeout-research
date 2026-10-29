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
"""Pure OPD retains verified tool costs without using hidden-test advantages."""

from unittest.mock import patch

import pytest
import torch

from tests.trainer.test_distillation_sample_mask_on_cpu import _batch, _config
from verl.trainer.distillation.losses import distillation_ppo_loss
from verl.workers.config import ActorConfig


def _penalty_batch(*, nested=False, all_masked=False, enabled=True, task_rewards=False):
    data, student = _batch(nested=nested, all_masked=all_masked)
    actor = ActorConfig(strategy="fsdp2", rollout_n=1, use_dynamic_bsz=True, tool_failure_penalty_enabled=enabled)
    _, teacher = _config(True, use_task_rewards=task_rewards)
    weights = torch.tensor([[0.02, 0.0], [0.02, 0.02]])
    if nested:
        weights = torch.nested.as_nested_tensor(list(weights.unbind()), layout=torch.jagged)
    data["tool_failure_weights"] = weights
    data["tool_failure_active_global_count"] = 0 if all_masked else 1
    return actor, teacher, data, student


@pytest.mark.parametrize("nested", [False, True])
def test_opd_gates_positive_teacher_credit_and_adds_local_cost_without_task_ppo(nested):
    actor, teacher, data, student = _penalty_batch(nested=nested)
    with patch("verl.trainer.distillation.losses.ppo_loss", side_effect=AssertionError("unexpected task PPO")):
        loss, metrics = distillation_ppo_loss(actor, teacher, {"log_probs": student}, data)
    loss.backward()

    # Only the proven bad token loses positive teacher credit. Its local cost
    # lowers its probability; the healthy token still receives the OPD signal.
    assert loss.item() == pytest.approx(-0.98)
    assert metrics["distillation/loss"].aggregate() == pytest.approx(-1.0)
    assert metrics["actor/tool_failure_loss"].aggregate() == pytest.approx(0.02)
    torch.testing.assert_close(student.grad, torch.tensor([0.02, -1.0, 0.0, 0.0, 0.0, 0.0]))


def test_opd_preserves_negative_teacher_credit_on_bad_tool_tokens():
    actor, teacher, data, student = _penalty_batch()
    data["teacher_logprobs"][0] = -3.0
    loss, _ = distillation_ppo_loss(actor, teacher, {"log_probs": student}, data)
    loss.backward()

    assert loss.item() == pytest.approx(-0.48)
    torch.testing.assert_close(student.grad, torch.tensor([0.52, -1.0, 0.0, 0.0, 0.0, 0.0]))


@pytest.mark.parametrize("nested", [False, True])
def test_opd_tool_cost_respects_fully_filtered_microbatch(nested):
    actor, teacher, data, student = _penalty_batch(nested=nested, all_masked=True)
    loss, metrics = distillation_ppo_loss(actor, teacher, {"log_probs": student}, data)
    loss.backward()
    assert loss.item() == 0
    assert metrics["actor/tool_failure_loss"].aggregate() == 0
    torch.testing.assert_close(student.grad, torch.zeros_like(student))


def test_disabled_tool_cost_preserves_pure_opd_even_if_stale_cost_fields_are_present():
    actor, teacher, data, student = _penalty_batch(enabled=False)
    loss, metrics = distillation_ppo_loss(actor, teacher, {"log_probs": student}, data)
    loss.backward()
    assert loss.item() == pytest.approx(-1.5)
    assert "actor/tool_failure_loss" not in metrics
    torch.testing.assert_close(student.grad, torch.tensor([-0.5, -1.0, 0.0, 0.0, 0.0, 0.0]))


def test_mixed_task_distillation_leaves_existing_ppo_tool_cost_path_unchanged():
    actor, teacher, data, student = _penalty_batch(task_rewards=True)
    with patch("verl.trainer.distillation.losses.ppo_loss", return_value=(torch.tensor(3.0), {})) as task:
        loss, _ = distillation_ppo_loss(actor, teacher, {"log_probs": student}, data)
    task.assert_called_once()
    assert loss.item() == pytest.approx(1.5)


@pytest.mark.parametrize("missing", ["tool_failure_weights", "tool_failure_active_global_count"])
def test_enabled_opd_tool_cost_fails_closed_without_v1_cost_fields(missing):
    actor, teacher, data, student = _penalty_batch()
    del data[missing]
    with pytest.raises(ValueError, match="V1 minibatch cost fields"):
        distillation_ppo_loss(actor, teacher, {"log_probs": student}, data)


def test_enabled_opd_tool_cost_requires_pg_advantage_for_credit_gating():
    actor, _, data, student = _penalty_batch()
    _, teacher = _config(False)
    with pytest.raises(ValueError, match="requires use_policy_gradient=True"):
        distillation_ppo_loss(actor, teacher, {"log_probs": student}, data)


def test_opd_tool_cost_uses_existing_trajectory_normalizer_and_importance_weights():
    actor, teacher, data, student = _penalty_batch(nested=True)
    data["dp_size"] = 2
    data["tool_failure_active_global_count"] = 4
    data["rollout_is_weights"] = torch.nested.as_nested_tensor(
        [torch.tensor([0.5, 1.0]), torch.tensor([1.0, 1.0])], layout=torch.jagged
    )
    loss, metrics = distillation_ppo_loss(actor, teacher, {"log_probs": student}, data)
    loss.backward()
    # Cost: 0.02 * importance 0.5 * DP 2 / global active trajectories 4.
    assert metrics["actor/tool_failure_loss"].aggregate() == pytest.approx(0.005)
    assert student.grad[0].item() == pytest.approx(0.005)
    assert student.grad[3:].eq(0).all()


def test_zero_cost_does_not_gate_teacher_credit():
    actor, teacher, data, student = _penalty_batch()
    data["tool_failure_weights"].zero_()
    loss, metrics = distillation_ppo_loss(actor, teacher, {"log_probs": student}, data)
    loss.backward()
    assert loss.item() == pytest.approx(-1.5)
    assert metrics["actor/tool_failure_loss"].aggregate() == 0
    torch.testing.assert_close(student.grad, torch.tensor([-0.5, -1.0, 0.0, 0.0, 0.0, 0.0]))

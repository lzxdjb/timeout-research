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
"""Infrastructure-filtered SWE rows must not train through the OPD objective."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from tensordict import TensorDict

from verl.trainer.distillation.batch_utils import prepare_distillation_minibatch
from verl.trainer.distillation.losses import compute_forward_kl_topk, distillation_ppo_loss
from verl.utils.metric import Metric
from verl.workers.config import DistillationLossConfig


def _batch(*, nested=False, all_masked=False, filtered=True):
    # Packed model outputs predict the two response tokens at positions 0/1
    # and 3/4 respectively. The second row is intentionally much worse.
    student_logprobs = torch.tensor([-2.0, -3.0, 0.0, -8.0, -9.0, 0.0], requires_grad=True)
    data = {
        "prompts": torch.tensor([[10], [20]]),
        "responses": torch.tensor([[11, 12], [21, 22]]),
        "attention_mask": torch.ones(2, 3, dtype=torch.long),
        "response_mask": torch.ones(2, 2, dtype=torch.bool),
        "teacher_logprobs": torch.tensor([[-1.0], [-1.0], [0.0], [-1.0], [-1.0], [0.0]]),
        "old_log_probs": student_logprobs.detach().reshape(2, 3)[:, :2],
        "dp_size": 1,
        "batch_num_tokens": 0 if all_masked else (2 if filtered else 4),
        "global_batch_size": 0 if all_masked else (1 if filtered else 2),
    }
    if filtered:
        data["train_sample_mask"] = torch.tensor([not all_masked, False])
    if nested:
        for key in ("prompts", "responses", "response_mask", "old_log_probs"):
            data[key] = torch.nested.as_nested_tensor(list(data[key].unbind()), layout=torch.jagged)
    return data, student_logprobs


def _config(use_policy_gradient, *, use_task_rewards=False):
    actor = SimpleNamespace(loss_agg_mode="token-mean", loss_scale_factor=None, global_batch_info={})
    loss = DistillationLossConfig(
        loss_mode="k1" if use_policy_gradient else "k2",
        use_policy_gradient=use_policy_gradient,
        use_task_rewards=use_task_rewards,
        loss_max_clamp=None,
    )
    return actor, SimpleNamespace(distillation_loss=loss)


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("use_policy_gradient", [False, True])
def test_filtered_row_has_zero_opd_gradient_and_is_excluded_from_metrics(nested, use_policy_gradient):
    data, student = _batch(nested=nested)
    actor, teacher = _config(use_policy_gradient)
    original_mask = data["response_mask"].clone()

    # No task advantages are supplied: pure PG OPD must not invoke task PPO.
    with patch("verl.trainer.distillation.losses.ppo_loss", side_effect=AssertionError("unexpected task PPO")):
        loss, metrics = distillation_ppo_loss(actor, teacher, {"log_probs": student}, data)

    assert loss.item() == pytest.approx(-1.5 if use_policy_gradient else 1.25)
    loss.backward()
    torch.testing.assert_close(student.grad, torch.tensor([-0.5, -1.0, 0.0, 0.0, 0.0, 0.0]))
    assert metrics["distillation/abs_loss"].aggregate() == pytest.approx(1.5 if use_policy_gradient else 1.25)
    assert metrics["distillation/loss_min"].aggregate() == pytest.approx(-2.0 if use_policy_gradient else 0.5)
    if nested:
        torch.testing.assert_close(data["response_mask"].values(), original_mask.values())
    else:
        torch.testing.assert_close(data["response_mask"], original_mask)


@pytest.mark.parametrize("use_policy_gradient", [False, True])
def test_fully_filtered_microbatch_has_finite_zero_loss_and_gradients(use_policy_gradient):
    data, student = _batch(all_masked=True)
    actor, teacher = _config(use_policy_gradient)
    loss, metrics = distillation_ppo_loss(actor, teacher, {"log_probs": student}, data)

    assert loss.item() == 0.0
    loss.backward()
    torch.testing.assert_close(student.grad, torch.zeros_like(student))
    for value in metrics.values():
        assert torch.isfinite(torch.tensor(value.aggregate() if isinstance(value, Metric) else value))


def test_unfiltered_distillation_preserves_existing_objective():
    data, student = _batch(filtered=False)
    actor, teacher = _config(False)
    loss, _ = distillation_ppo_loss(actor, teacher, {"log_probs": student}, data)
    assert loss.item() == pytest.approx((0.5 + 2.0 + 24.5 + 32.0) / 4)
    loss.backward()
    torch.testing.assert_close(student.grad, torch.tensor([-0.25, -0.5, 0.0, -1.75, -2.0, 0.0]))


def test_mixed_distillation_still_calls_task_ppo():
    data, student = _batch()
    actor, teacher = _config(True, use_task_rewards=True)
    task_result = (torch.tensor(3.0), {"task_metric": 2.0})
    with patch("verl.trainer.distillation.losses.ppo_loss", return_value=task_result) as task:
        loss, metrics = distillation_ppo_loss(actor, teacher, {"log_probs": student}, data)
    task.assert_called_once()
    assert loss.item() == pytest.approx(1.5)
    assert metrics["task_metric"] == 2.0


def test_topk_diagnostics_exclude_filtered_rows():
    data, _ = _batch()
    output = {
        "distillation_losses": torch.tensor([0.1, 0.2, 0.0, 4.0, 5.0, 0.0]),
        "student_mass": torch.tensor([0.9, 0.8, 0.0, 0.1, 0.2, 0.0]),
        "teacher_mass": torch.tensor([0.95, 0.85, 0.0, 0.3, 0.4, 0.0]),
        "overlap_count": torch.tensor([2.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
        "overlap_token_advantage": torch.tensor([-0.2, -0.4, 0.0, -9.0, -9.0, 0.0]),
    }
    teacher = SimpleNamespace(distillation_loss=SimpleNamespace(topk=2))
    _, metrics = compute_forward_kl_topk(SimpleNamespace(), teacher, output, data)
    assert metrics["distillation/student_mass"] == pytest.approx(0.85)
    assert metrics["distillation/teacher_mass"] == pytest.approx(0.9)
    assert metrics["distillation/overlap_ratio"] == pytest.approx(0.75)
    assert metrics["distillation/overlap_token_advantage"] == pytest.approx(-0.3)


def test_invalid_sequence_mask_shape_fails_before_training():
    data, student = _batch()
    data["train_sample_mask"] = torch.ones(2, 1)
    actor, teacher = _config(True)
    with pytest.raises(ValueError, match="shape"):
        distillation_ppo_loss(actor, teacher, {"log_probs": student}, data)


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("all_masked", [False, True])
def test_minibatch_normalizers_use_accepted_rows_and_tokens(nested, all_masked):
    response_mask = torch.tensor([[1, 0, 1], [1, 1, 0]], dtype=torch.bool)
    if nested:
        response_mask = torch.nested.as_nested_tensor(list(response_mask.unbind()), layout=torch.jagged)
    data = TensorDict(
        {"response_mask": response_mask, "train_sample_mask": torch.tensor([not all_masked, False])},
        batch_size=[2],
    )
    prepare_distillation_minibatch(data)
    mask = data["loss_mask"].to_padded_tensor(False) if nested else data["loss_mask"]
    assert mask.sum().item() == (0 if all_masked else 2)
    assert data["global_batch_size"] == 1
    assert not mask[1].any()


def test_minibatch_sequence_normalizer_reduces_across_dp_before_microbatching():
    data = TensorDict(
        {"response_mask": torch.ones(2, 3, dtype=torch.bool), "train_sample_mask": torch.tensor([True, False])},
        batch_size=[2],
    )

    def gather(counts, local_count, group):
        assert local_count == 1
        assert group == "dp-group"
        counts[:] = [local_count, 2]

    with (
        patch("torch.distributed.is_initialized", return_value=True),
        patch("torch.distributed.get_world_size", return_value=2),
        patch("torch.distributed.all_gather_object", side_effect=gather),
    ):
        prepare_distillation_minibatch(data, "dp-group")
    assert data["global_batch_size"] == 3

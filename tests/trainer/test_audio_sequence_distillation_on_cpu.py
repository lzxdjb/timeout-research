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
"""Cross-tokenizer audio uses whole-response PG, while SWE keeps token k1."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from tensordict import TensorDict

from verl.trainer.distillation.batch_utils import prepare_distillation_minibatch
from verl.trainer.distillation.losses import distillation_ppo_loss
from verl.utils import tensordict_utils as tu
from verl.utils.metric import Metric
from verl.workers.config import ActorConfig, DistillationLossConfig
from verl.workers.utils.padding import left_right_2_no_padding


def _batch(*, nested=False, dp_size=1):
    # Each row has one prompt token and two assistant tokens; the second
    # assistant token is the emitted EOS and must receive the audio gradient.
    student = torch.tensor([-2.0, -3.0, 0.0, -4.0, -5.0, 0.0], requires_grad=True)
    data = {
        "prompts": torch.tensor([[10], [20]]),
        "responses": torch.tensor([[11, 12], [21, 22]]),
        "attention_mask": torch.ones(2, 3, dtype=torch.long),
        "response_mask": torch.ones(2, 2, dtype=torch.bool),
        "old_log_probs": torch.tensor([[-2.0, -3.0], [-4.0, -5.0]]),
        # Deliberate NaNs prove audio placeholders are never token targets.
        "teacher_logprobs": torch.tensor([[-1.0], [-1.0], [0.0], [float("nan")], [float("nan")], [float("nan")]]),
        "teacher_sequence_logprob": torch.tensor([float("nan"), -3.0], requires_grad=True),
        "teacher_sequence_mask": torch.tensor([False, True]),
        "teacher_sequence_token_count": torch.tensor([0, 3]),
        "train_sample_mask": torch.ones(2, dtype=torch.bool),
        "dp_size": dp_size,
        "batch_num_tokens": 4 * dp_size,
        "global_batch_size": 2 * dp_size,
        "audio_sequence_global_count": dp_size,
        "swe_sequence_global_count": dp_size,
    }
    if nested:
        for key in ("prompts", "responses", "response_mask", "old_log_probs"):
            data[key] = torch.nested.as_nested_tensor(list(data[key].unbind()), layout=torch.jagged)
    return data, student


def _config(*, enabled=True, scale=1.0, tool_penalty=False):
    actor = ActorConfig(
        strategy="fsdp2",
        rollout_n=2,
        use_dynamic_bsz=True,
        loss_agg_mode="seq-mean-token-mean",
        tool_failure_penalty_enabled=tool_penalty,
    )
    teacher = SimpleNamespace(
        distillation_loss=DistillationLossConfig(
            loss_mode="k1",
            use_policy_gradient=True,
            use_task_rewards=False,
            loss_max_clamp=None,
            audio_sequence_surrogate=enabled,
            audio_sequence_reward_scale=scale,
        )
    )
    return actor, teacher


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("dp_size", [1, 2])
def test_real_mixed_loss_and_gradients_use_whole_audio_response_and_stop_teacher_reward(nested, dp_size):
    data, student = _batch(nested=nested, dp_size=dp_size)
    actor, teacher = _config()
    with patch("verl.trainer.distillation.losses.ppo_loss", side_effect=AssertionError("no hidden/task PPO")):
        loss, metrics = distillation_ppo_loss(actor, teacher, {"log_probs": student}, data)
    # SWE mean advantages=(1+2)/2, audio R=-3-(-4-5)=6.
    assert loss.item() == pytest.approx(-3.75)
    loss.backward()
    torch.testing.assert_close(student.grad, torch.tensor([-0.25, -0.5, 0.0, -3.0, -3.0, 0.0]))
    assert data["teacher_sequence_logprob"].grad is None
    assert metrics["distillation/audio/reward"].aggregate() == pytest.approx(6)
    assert metrics["distillation/audio/loss"].aggregate() == pytest.approx(-3)
    assert metrics["distillation/swe/loss"].aggregate() == pytest.approx(-0.75)
    assert metrics["distillation/audio/accepted_sequences"].aggregate() == dp_size
    assert metrics["distillation/audio/finite_teacher_sequences"].aggregate() == dp_size
    assert metrics["distillation/audio/clipped_sequences"].aggregate() == 0


@pytest.mark.parametrize(
    ("reward", "ratio", "expected_audio_gradient"),
    [(6.0, 1.5, 0.0), (6.0, 0.5, -1.5), (-6.0, 1.5, 4.5), (-6.0, 0.5, 0.0)],
)
def test_clipping_applies_to_the_ratio_of_whole_sequence_probabilities(reward, ratio, expected_audio_gradient):
    data, student = _batch()
    data["teacher_sequence_logprob"] = torch.tensor([0.0, -9.0 + reward])
    with torch.no_grad():
        student[3:5].add_(torch.tensor(ratio).log() / 2)
    actor, teacher = _config()
    loss, metrics = distillation_ppo_loss(actor, teacher, {"log_probs": student}, data)
    loss.backward()
    torch.testing.assert_close(student.grad[3:5], torch.full((2,), expected_audio_gradient))
    assert metrics["distillation/audio/clipped_sequences"].aggregate() == int(expected_audio_gradient == 0)


def test_audio_reward_scale_does_not_change_swe_gradient():
    data, student = _batch()
    actor, teacher = _config(scale=0.25)
    loss, _ = distillation_ppo_loss(actor, teacher, {"log_probs": student}, data)
    loss.backward()
    torch.testing.assert_close(student.grad, torch.tensor([-0.25, -0.5, 0.0, -0.75, -0.75, 0.0]))


def test_sequence_sums_exclude_nonassistant_positions():
    data, student = _batch()
    data["response_mask"][1, 0] = False
    actor, teacher = _config()
    loss, _ = distillation_ppo_loss(actor, teacher, {"log_probs": student}, data)
    loss.backward()
    # P_old=-5, not -9: the excluded position cannot change R or get credit.
    assert loss.item() == pytest.approx(-1.75)
    torch.testing.assert_close(student.grad, torch.tensor([-0.25, -0.5, 0.0, 0.0, -1.0, 0.0]))


@pytest.mark.parametrize("tool_penalty", [False, True])
def test_nonfinite_excluded_student_positions_cannot_contaminate_swe_loss(tool_penalty):
    data, student = _batch()
    data["train_sample_mask"][1] = False
    data["global_batch_size"] = 1
    data["audio_sequence_global_count"] = 0
    data["old_log_probs"][1] = float("nan")
    with torch.no_grad():
        student[3:5] = float("nan")
    if tool_penalty:
        data["tool_failure_weights"] = torch.zeros(2, 2)
        data["tool_failure_active_global_count"] = 1
    actor, teacher = _config(tool_penalty=tool_penalty)
    loss, _ = distillation_ppo_loss(actor, teacher, {"log_probs": student}, data)
    loss.backward()
    assert loss.item() == pytest.approx(-1.5)
    torch.testing.assert_close(student.grad, torch.tensor([-0.5, -1.0, 0.0, 0.0, 0.0, 0.0]))


@pytest.mark.parametrize("nested", [False, True])
def test_filtered_audio_does_not_read_invalid_scores_or_train(nested):
    data, student = _batch(nested=nested)
    data["train_sample_mask"][1] = False
    data["teacher_sequence_logprob"] = torch.full((2,), float("nan"))
    data["global_batch_size"] = 1
    data["audio_sequence_global_count"] = 0
    actor, teacher = _config()
    loss, _ = distillation_ppo_loss(actor, teacher, {"log_probs": student}, data)
    loss.backward()
    assert loss.item() == pytest.approx(-1.5)
    torch.testing.assert_close(student.grad, torch.tensor([-0.5, -1.0, 0.0, 0.0, 0.0, 0.0]))


@pytest.mark.parametrize("nested", [False, True])
def test_fully_filtered_microbatch_has_finite_zero_loss_and_gradients(nested):
    data, student = _batch(nested=nested)
    data["train_sample_mask"].zero_()
    data["teacher_sequence_logprob"] = torch.full((2,), float("nan"))
    actor, teacher = _config()
    loss, metrics = distillation_ppo_loss(actor, teacher, {"log_probs": student}, data)
    loss.backward()
    assert loss.item() == 0
    torch.testing.assert_close(student.grad, torch.zeros_like(student))
    assert all(torch.isfinite(torch.as_tensor(v.aggregate() if isinstance(v, Metric) else v)) for v in metrics.values())


def test_audio_only_microbatch_does_not_require_token_teacher_targets():
    data, student = _batch()
    data["teacher_sequence_mask"][:] = True
    data["teacher_sequence_logprob"] = torch.tensor([-3.0, -3.0])
    data.pop("teacher_logprobs")
    actor, teacher = _config()
    loss, _ = distillation_ppo_loss(actor, teacher, {"log_probs": student}, data)
    loss.backward()
    torch.testing.assert_close(student.grad, torch.tensor([-1.0, -1.0, 0.0, -3.0, -3.0, 0.0]))


def test_text_only_batches_retain_exact_legacy_loss_and_gradients():
    legacy_data, legacy_student = _batch()
    legacy_data["teacher_sequence_mask"].zero_()
    legacy_data["teacher_logprobs"][3:5] = -1
    new_data, new_student = _batch()
    new_data["teacher_sequence_mask"].zero_()
    new_data["teacher_logprobs"][3:5] = -1
    old_actor, old_teacher = _config(enabled=False)
    new_actor, new_teacher = _config()
    old_loss, _ = distillation_ppo_loss(old_actor, old_teacher, {"log_probs": legacy_student}, legacy_data)
    new_loss, _ = distillation_ppo_loss(new_actor, new_teacher, {"log_probs": new_student}, new_data)
    torch.testing.assert_close(new_loss, old_loss, rtol=0, atol=0)
    old_loss.backward()
    new_loss.backward()
    torch.testing.assert_close(new_student.grad, legacy_student.grad, rtol=0, atol=0)


def test_sequence_scores_fail_closed_when_the_new_loss_is_disabled():
    data, student = _batch()
    actor, teacher = _config(enabled=False)
    with pytest.raises(ValueError, match="audio_sequence_surrogate=True"):
        distillation_ppo_loss(actor, teacher, {"log_probs": student}, data)


@pytest.mark.parametrize("missing", ["teacher_sequence_logprob", "teacher_sequence_mask"])
def test_audio_dispatch_fails_closed_without_scalar_or_explicit_route(missing):
    data, student = _batch()
    data.pop(missing)
    actor, teacher = _config()
    with pytest.raises(ValueError, match=missing):
        distillation_ppo_loss(actor, teacher, {"log_probs": student}, data)


def test_nonfinite_accepted_teacher_score_fails_before_backward():
    data, student = _batch()
    data["teacher_sequence_logprob"] = torch.full((2,), float("nan"))
    actor, teacher = _config()
    with pytest.raises(ValueError, match="teacher sequence scores must be finite"):
        distillation_ppo_loss(actor, teacher, {"log_probs": student}, data)


def test_audio_refuses_inherited_token_importance_weights():
    data, student = _batch()
    data["rollout_is_weights"] = torch.full((2, 2), 0.5)
    actor, teacher = _config()
    with pytest.raises(ValueError, match="no rollout importance correction"):
        distillation_ppo_loss(actor, teacher, {"log_probs": student}, data)


def test_local_tool_cost_and_positive_credit_gating_apply_only_to_swe():
    data, student = _batch()
    data["tool_failure_weights"] = torch.tensor([[0.02, 0.0], [100.0, 100.0]])
    data["tool_failure_active_global_count"] = 1
    actor, teacher = _config(tool_penalty=True)
    loss, metrics = distillation_ppo_loss(actor, teacher, {"log_probs": student}, data)
    loss.backward()
    assert metrics["actor/tool_failure_loss"].aggregate() == pytest.approx(0.02)
    assert loss.item() == pytest.approx(-3.48)
    torch.testing.assert_close(student.grad, torch.tensor([0.02, -0.5, 0.0, -3.0, -3.0, 0.0]))


def _tensor_batch():
    fields, student = _batch(nested=True)
    non_tensor = {}
    for key in (
        "dp_size",
        "batch_num_tokens",
        "global_batch_size",
        "audio_sequence_global_count",
        "swe_sequence_global_count",
    ):
        non_tensor[key] = fields.pop(key)
    fields["teacher_logprobs"] = torch.nested.as_nested_tensor(
        list(fields["teacher_logprobs"].reshape(2, 3, 1).unbind()), layout=torch.jagged
    )
    data = TensorDict(fields, batch_size=[2])
    tu.assign_non_tensor(data, **non_tensor)
    return data, student


def test_realized_route_normalization_is_invariant_to_dynamic_microbatch_splitting():
    whole, whole_student = _tensor_batch()
    split, split_student = _tensor_batch()
    prepare_distillation_minibatch(whole)
    prepare_distillation_minibatch(split)
    actor, teacher = _config()
    whole_loss, _ = distillation_ppo_loss(actor, teacher, {"log_probs": whole_student}, whole)
    split_loss = sum(
        distillation_ppo_loss(
            actor,
            teacher,
            {"log_probs": split_student[i * 3 : (i + 1) * 3]},
            tu.index_select_tensor_dict(split, [i]),
        )[0]
        for i in range(2)
    )
    torch.testing.assert_close(split_loss, whole_loss)
    whole_loss.backward()
    split_loss.backward()
    torch.testing.assert_close(split_student.grad, whole_student.grad)


def test_route_counts_reduce_across_dp_and_tool_normalizer_excludes_audio():
    data, _ = _tensor_batch()
    data["tool_failure_normalizer"] = torch.ones(2)

    def gather(counts, local_counts, group):
        assert local_counts == (1, 1)
        assert group == "dp-group"
        counts[:] = [(1, 1), (0, 2)]

    with (
        patch("torch.distributed.is_initialized", return_value=True),
        patch("torch.distributed.get_world_size", return_value=2),
        patch("torch.distributed.all_gather_object", side_effect=gather),
    ):
        prepare_distillation_minibatch(data, "dp-group")
    assert data["global_batch_size"] == 4
    assert data["audio_sequence_global_count"] == 1
    assert data["swe_sequence_global_count"] == 3
    torch.testing.assert_close(data["tool_failure_normalizer"], torch.tensor([1.0, 0.0]))


def test_unequal_route_counts_across_dp_preserve_global_objective_and_audio_metrics():
    first, first_student = _batch(dp_size=2)
    second, second_student = _batch(dp_size=2)
    second["teacher_sequence_mask"].zero_()
    second["teacher_logprobs"][3:5] = -1.0
    for data in (first, second):
        data["audio_sequence_global_count"] = 1
        data["swe_sequence_global_count"] = 3
    actor, teacher = _config()
    first_loss, first_metrics = distillation_ppo_loss(actor, teacher, {"log_probs": first_student}, first)
    second_loss, second_metrics = distillation_ppo_loss(actor, teacher, {"log_probs": second_student}, second)
    # Simulate DDP averaging. Three SWE means 1.5,1.5,3.5 and one audio R=6.
    global_loss = (first_loss + second_loss) / 2
    assert global_loss.item() == pytest.approx(-(1.5 + 1.5 + 3.5 + 6) / 4)
    global_loss.backward()
    torch.testing.assert_close(first_student.grad, torch.tensor([-0.125, -0.25, 0.0, -1.5, -1.5, 0.0]))
    assert (
        Metric.aggregate_dp([first_metrics["distillation/audio/reward"], second_metrics["distillation/audio/reward"]])
        == 6
    )
    assert (
        Metric.aggregate_dp(
            [
                first_metrics["distillation/audio/accepted_sequences"],
                second_metrics["distillation/audio/accepted_sequences"],
            ]
        )
        == 1
    )


def test_padding_conversion_preserves_sequence_fields_as_regular_batch_tensors():
    fields = {
        "input_ids": torch.tensor([[10, 11, 12], [20, 21, 22]]),
        "attention_mask": torch.ones(2, 3, dtype=torch.long),
        "position_ids": torch.tensor([[0, 1, 2], [0, 1, 2]]),
        "response_mask": torch.ones(2, 2, dtype=torch.bool),
        "teacher_sequence_logprob": torch.tensor([0.0, -3.0]),
        "teacher_sequence_mask": torch.tensor([False, True]),
        "teacher_sequence_token_count": torch.tensor([0, 3]),
    }
    data = TensorDict(fields, batch_size=[2])
    left_right_2_no_padding(data)
    for key in ("teacher_sequence_logprob", "teacher_sequence_mask", "teacher_sequence_token_count"):
        assert not data[key].is_nested
        assert data[key].shape == (2,)
        torch.testing.assert_close(data[key], fields[key])


@pytest.mark.parametrize(
    "overrides",
    [
        {"use_task_rewards": True},
        {"use_policy_gradient": False},
        {"loss_mode": "k3"},
        {"audio_sequence_reward_scale": 0},
        {"audio_sequence_reward_scale": float("inf")},
        {"audio_sequence_clip_ratio": 1.0},
        {"audio_sequence_baseline": "batch_mean"},
    ],
)
def test_invalid_audio_objective_settings_fail_during_configuration(overrides):
    values = dict(loss_mode="k1", use_policy_gradient=True, use_task_rewards=False, audio_sequence_surrogate=True)
    values.update(overrides)
    with pytest.raises(ValueError):
        DistillationLossConfig(**values)

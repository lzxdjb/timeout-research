"""Reasoning gradients coexist with unchanged SWE/audio objectives and costs."""

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from transfer_queue import KVBatchMeta

from verl.trainer.distillation.batch_utils import prepare_distillation_minibatch
from verl.trainer.distillation.losses import _opd_tool_failure_weights, distillation_ppo_loss
from verl.trainer.distillation.reasoning_utils import (
    image_reasoning_trajectory,
    skip_reasoning_task_reward,
)
from verl.trainer.ppo.v1.utils import v1_success_values
from verl.utils import tensordict_utils as tu
from verl.utils.swe_tool_failure import rejection_metadata, token_digest
from verl.workers.config import ActorConfig, DistillationLossConfig


def _profile(enabled=True):
    return {
        "trainer": {"general_reasoning": {"enabled": enabled}},
        "distillation": {"enabled": True, "distillation_loss": {"use_task_rewards": False}},
    }


def _outputs():
    return [SimpleNamespace(reward_score=None, response_mask=[1, 1], extra_fields={})]


def test_training_bypass_is_opt_in_and_never_infers_correctness_from_zero():
    outputs = _outputs()
    sample = {"is_general_reasoning": True}
    assert not skip_reasoning_task_reward(_profile(False), outputs, sample)
    assert outputs[0].reward_score is None
    assert not skip_reasoning_task_reward(_profile(), outputs, sample, validate=True)
    assert outputs[0].reward_score is None
    assert not skip_reasoning_task_reward(_profile(), outputs, {"is_general_reasoning": False})
    assert skip_reasoning_task_reward(_profile(), outputs, sample)
    assert outputs[0].reward_score == 0.0
    assert v1_success_values([outputs[0].extra_fields], [0.0]) == [None]


def test_reward_bypass_rejects_task_reward_training():
    profile = _profile()
    profile["distillation"]["distillation_loss"]["use_task_rewards"] = True
    with pytest.raises(ValueError, match="pure OPD"):
        skip_reasoning_task_reward(profile, _outputs(), {"is_general_reasoning": True})


def _batch(nested=False):
    # One SWE row, one audio row, and one reasoning row, each with two
    # assistant tokens. Audio token targets are deliberately nonfinite.
    student = torch.tensor([-2.0, -3.0, 0.0, -4.0, -5.0, 0.0, -6.0, -7.0, 0.0], requires_grad=True)
    data = {
        "prompts": torch.tensor([[10], [20], [30]]),
        "responses": torch.tensor([[11, 12], [21, 22], [31, 32]]),
        "attention_mask": torch.ones(3, 3, dtype=torch.long),
        "response_mask": torch.ones(3, 2, dtype=torch.bool),
        "old_log_probs": torch.tensor([[-2.0, -3.0], [-4.0, -5.0], [-6.0, -7.0]]),
        "teacher_logprobs": torch.tensor([
            [-1.0], [-1.0], [0.0], [float("nan")], [float("nan")], [float("nan")],
            [-5.0], [-5.0], [0.0],
        ]),
        "teacher_sequence_logprob": torch.tensor([0.0, -3.0, 0.0]),
        "teacher_sequence_mask": torch.tensor([False, True, False]),
        "general_reasoning_mask": torch.tensor([False, False, True]),
        "train_sample_mask": torch.ones(3, dtype=torch.bool),
        "dp_size": 1,
        "batch_num_tokens": 6,
    }
    if nested:
        for key in ("prompts", "responses", "response_mask", "old_log_probs"):
            data[key] = torch.nested.as_nested_tensor(list(data[key].unbind()), layout=torch.jagged)
    # Teacher evidence is a sequence field before engine packing; unlike the
    # flattened model output, it must retain the TensorDict batch dimension.
    data["teacher_logprobs"] = torch.nested.as_nested_tensor(
        list(data["teacher_logprobs"].reshape(3, 3, 1).unbind()), layout=torch.jagged
    )
    metadata = {key: data.pop(key) for key in ("dp_size", "batch_num_tokens")}
    data = TensorDict(data, batch_size=[3])
    tu.assign_non_tensor(data, **metadata)
    return data, student


def _config(tool_penalty=False):
    actor = ActorConfig(
        strategy="fsdp2", rollout_n=1, use_dynamic_bsz=True,
        loss_agg_mode="seq-mean-token-mean", tool_failure_penalty_enabled=tool_penalty,
    )
    teacher = SimpleNamespace(distillation_loss=DistillationLossConfig(
        loss_mode="k1", use_policy_gradient=True, use_task_rewards=False,
        loss_max_clamp=None, audio_sequence_surrogate=True,
    ))
    return actor, teacher


@pytest.mark.parametrize("nested", [False, True])
def test_all_domains_train_with_separate_metrics_and_one_shared_sequence_denominator(nested):
    data, student = _batch(nested)
    prepare_distillation_minibatch(data)
    assert tu.get_non_tensor_data(data, "global_batch_size", default=None) == 3
    for key in ("audio", "swe", "general"):
        assert tu.get_non_tensor_data(data, f"{key}_sequence_global_count", default=None) == 1
    actor, teacher = _config()
    loss, metrics = distillation_ppo_loss(actor, teacher, {"log_probs": student}, data)
    assert loss.item() == pytest.approx(-3.0)
    loss.backward()
    torch.testing.assert_close(student.grad, torch.tensor([
        -1 / 6, -1 / 3, 0.0, -2.0, -2.0, 0.0, -1 / 6, -1 / 3, 0.0,
    ]))
    assert metrics["distillation/swe/loss"].aggregate() == pytest.approx(-0.5)
    assert metrics["distillation/audio/loss"].aggregate() == pytest.approx(-2.0)
    assert metrics["distillation/general/loss"].aggregate() == pytest.approx(-0.5)
    for key in ("swe", "audio", "general"):
        assert metrics[f"distillation/{key}/accepted_sequences"].aggregate() == 1


def test_general_rows_cannot_dilute_or_receive_swe_tool_penalties_even_with_stale_cost_fields():
    data, _ = _batch()
    data["tool_failure_normalizer"] = torch.ones(3)
    data["tool_failure_weights"] = torch.ones(3, 2)
    prepare_distillation_minibatch(data)
    torch.testing.assert_close(data["tool_failure_normalizer"], torch.tensor([1.0, 0.0, 0.0]))
    tu.assign_non_tensor(data, tool_failure_active_global_count=1)
    actor, teacher = _config(tool_penalty=True)
    weights = _opd_tool_failure_weights(actor, teacher.distillation_loss, data)
    torch.testing.assert_close(weights, torch.tensor([[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]]))
    assert data["train_sample_mask"].all()


def test_rejected_reasoning_rows_stay_out_of_normalizers_and_gradients():
    data, student = _batch()
    data["train_sample_mask"][2] = False
    prepare_distillation_minibatch(data)
    assert tu.get_non_tensor_data(data, "global_batch_size", default=None) == 2
    assert tu.get_non_tensor_data(data, "general_sequence_global_count", default=None) == 0
    actor, teacher = _config()
    loss, metrics = distillation_ppo_loss(actor, teacher, {"log_probs": student}, data)
    loss.backward()
    torch.testing.assert_close(student.grad[6:], torch.zeros(3))
    assert metrics["distillation/general/accepted_sequences"].aggregate() == 0


def test_image_trajectory_preserves_thinking_and_accepts_truncated_candidates():
    tokenizer = SimpleNamespace(decode=lambda ids, **kwargs: {
        (1,): "user<|im_end|>\n<|im_start|>assistant\n<think>\n",
        (2,): "reasoning ",
    }[tuple(ids)])
    for ids, terminated in (([2], False), ([2, 3], True)):
        trajectory = image_reasoning_trajectory(tokenizer, [1], ids, [{"role": "user", "content": "x"}], {3})
        assert trajectory["response_text"] == "reasoning "
        assert trajectory["assistant_prefix"] == "<think>\n"
        assert trajectory["terminated"] is terminated


def test_real_trainer_retains_swe_costs_and_keeps_reasoning_active(monkeypatch):
    from verl.trainer.ppo.v1 import trainer_base

    class Trainer(trainer_base.PPOTrainer):
        def on_step_end(self):
            pass

        def on_sample_end(self):
            pass

    monkeypatch.setenv("SWE_AGENT_TASK_FILTER_TRAINING_ENABLED", "0")
    monkeypatch.setenv("SWE_AGENT_REPEAT_REWARD_MODE", "off")
    monkeypatch.setenv("VERL_GRPO_GOLDEN_REPLAY_ENABLED", "0")
    trainer = Trainer.__new__(Trainer)
    trainer.config = OmegaConf.create({
        "trainer": {"use_v1": True, "general_reasoning": {"enabled": True}},
        "distillation": {
            "enabled": True, "skip_hidden_reward": True,
            "distillation_loss": {"use_task_rewards": False},
        },
        "algorithm": {"adv_estimator": "grpo", "use_kl_in_reward": False},
        "actor_rollout_ref": {
            "model": {}, "rollout": {"n": 1},
            "actor": {
                "tool_failure_penalty_enabled": True,
                "tool_failure_penalty_cost": 0.02,
                "tool_failure_penalty_cap": 0.30,
                "tool_failure_penalty_warmup_steps": 0,
            },
        },
    })
    trainer.timeout_prediction_enabled = False
    trainer.global_steps = 1
    ids = [10, 11, 12, 13]
    event = {
        "version": 1, "start": 1, "end": 3,
        "token_sha256": token_digest(ids[1:3]),
        "prefix_sha256": token_digest(ids[:3]),
        "proof": rejection_metadata("unknown_tool")["tool_contract_rejection"],
    }
    batch = KVBatchMeta(partition_id="train", keys=["swe_0_0", "general_0_0"], tags=[{}, {}])
    queue = tu.get_tensordict({
        "uid": torch.arange(2),
        "responses": torch.nested.as_nested_tensor([torch.tensor(ids)] * 2, layout=torch.jagged),
        "response_mask": torch.nested.as_nested_tensor([torch.ones(4)] * 2, layout=torch.jagged),
        "rm_scores": torch.nested.as_nested_tensor([torch.zeros(4)] * 2, layout=torch.jagged),
        "general_reasoning_mask": torch.tensor([False, True]),
        "extra_fields": [
            {"tool_failure_events": [event], "reward_extra_info": {"hidden_reward_skipped": 1}},
            {"tool_failure_events": [event], "general_opd_task_reward_skipped": True},
        ],
    })
    captured = {}

    def get(**kwargs):
        fields = kwargs["select_fields"]
        if len(fields) == 1:
            raise KeyError(fields[0])
        assert "general_reasoning_mask" in fields
        return queue.select(*(field for field in fields if field in queue)).clone()

    def put(**kwargs):
        captured.update(kwargs["fields"])
        return batch

    monkeypatch.setattr(trainer_base.tq, "kv_batch_get", get)
    monkeypatch.setattr(trainer_base.tq, "kv_batch_put", put)
    trainer._compute_advantage(batch, {})
    assert captured["tool_failure_active"].tolist() == [True, False]
    torch.testing.assert_close(captured["tool_failure_normalizer"], torch.tensor([1.0, 0.0]))
    assert captured["tool_failure_weights"].unbind()[0].sum().item() == pytest.approx(0.02)
    assert captured["tool_failure_weights"].unbind()[1].eq(0).all()
    assert "train_sample_mask" not in captured or captured["train_sample_mask"].all()

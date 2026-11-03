"""Capped audio candidates must reach the actor as explicitly excluded rows."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopOutput
from verl.experimental.teacher_loop.errors import TeacherScoringError
from verl.trainer.distillation.batch_utils import prepare_distillation_minibatch
from verl.trainer.distillation.losses import distillation_ppo_loss
from verl.trainer.ppo.v1 import agent_loop_tq
from verl.trainer.ppo.v1.trainer_base import _v1_train_sample_mask_from_tags
from verl.utils import tensordict_utils as tu
from verl.workers.config import ActorConfig, DistillationLossConfig


def _worker(monkeypatch, *, reasoning=True):
    worker_class = agent_loop_tq.AgentLoopWorkerTQ.__ray_metadata__.modified_class
    worker = object.__new__(worker_class)
    worker.config = OmegaConf.create({
        "trainer": {"general_reasoning": {"enabled": reasoning}},
        "distillation": {
            "enabled": True,
            "teacher_key": "teacher_route",
            "distillation_loss": {"use_task_rewards": False},
            "teacher_models": {
                "swe": {"key": "swe", "backend": "external", "external": {"protocol": "text_token_ids"}},
                "audio": {"key": "audio", "backend": "external", "external": {
                    "protocol": "audio_candidate_score", "audio_response_length": 2,
                }},
            },
        },
    })
    worker.distillation_enabled = True
    worker.teacher_key = "teacher_route"
    worker.model_config = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=77))

    class Tokenizer:
        eos_token_id = 55

        def decode(self, ids, **kwargs):
            return "<|im_start|>assistant\n" if ids == [1] else "Birdsong."

    worker.tokenizer = Tokenizer()
    worker.teacher_server_manager = SimpleNamespace(
        uses_audio_sequence_score=lambda key: key == "audio",
        has_audio_teacher=True,
        compute_teacher_sequence_score_single=AsyncMock(return_value={
            "teacher_sequence_mask": True,
            "teacher_sequence_logprob": -3.0,
            "teacher_sequence_token_count": 2,
        }),
    )
    worker._compute_teacher_logprobs = AsyncMock(wraps=worker._compute_teacher_logprobs)
    worker.repeat_coordinator = None
    worker._repeat_debug_event = lambda *args, **kwargs: None
    worker.cutoff_session_states = {}
    worker._compute_score = AsyncMock(side_effect=AssertionError("Task rewards must stay bypassed"))
    worker._compute_multi_modal_inputs = lambda *args: {}
    worker._compute_position_ids = lambda inputs, *args: torch.arange(inputs.shape[-1]).unsqueeze(0)
    published = {}

    async def publish(**kwargs):
        published.update(kwargs)

    async def await_publish(partition, uid, operation):
        return await operation

    worker._await_tq_publication = await_publish
    monkeypatch.setattr(agent_loop_tq.tq, "async_kv_batch_put", publish)
    return worker, published


def _output(*, eos=None):
    return AgentLoopOutput(
        prompt_ids=[1], response_ids=[2, 3 if eos is None else eos], response_mask=[1, 1],
        metrics=AgentLoopMetrics(), extra_fields={"audio_student_truncated": False},
    )


def _postprocess(worker, output):
    return worker._agent_loop_postprocess(
        output, False, uid="capped", session_id=0, global_steps=0, teacher_route="audio",
        raw_prompt=[{"role": "user", "content": "Listen"}],
    )


def test_capped_candidate_is_published_masked_without_any_teacher_or_task_reward(monkeypatch):
    worker, published = _worker(monkeypatch)
    output = _output()
    asyncio.run(_postprocess(worker, output))
    worker._compute_teacher_logprobs.assert_not_awaited()
    worker.teacher_server_manager.compute_teacher_sequence_score_single.assert_not_awaited()
    worker._compute_score.assert_not_awaited()
    assert published["tags"][0]["train_sample_mask"] is False
    assert published["tags"][0]["audio_opd_candidate_excluded"] is True
    assert published["fields"]["teacher_sequence_mask"].tolist() == [False]
    assert output.response_ids == [2, 3]
    assert output.extra_fields["reward_extra_info"]["audio_response_tokens"] == 2

    mask = _v1_train_sample_mask_from_tags(published["keys"], published["tags"])
    assert mask.tolist() == [False]


@pytest.mark.parametrize("reasoning", [False, True])
@pytest.mark.parametrize("eos", [55, 77])
def test_complete_audio_uses_existing_teacher_and_reward_bypass(monkeypatch, reasoning, eos):
    worker, published = _worker(monkeypatch, reasoning=reasoning)
    asyncio.run(_postprocess(worker, _output(eos=eos)))
    worker._compute_teacher_logprobs.assert_awaited_once()
    worker.teacher_server_manager.compute_teacher_sequence_score_single.assert_awaited_once()
    worker._compute_score.assert_not_awaited()
    assert published["fields"]["teacher_sequence_mask"].tolist() == [True]
    assert published["tags"][0].get("train_sample_mask", True) is True


def test_legacy_profile_keeps_strict_incomplete_audio_failure(monkeypatch):
    worker, published = _worker(monkeypatch, reasoning=False)
    with pytest.raises(TeacherScoringError, match="complete, assistant-only"):
        asyncio.run(_postprocess(worker, _output()))
    assert not published


def test_real_teacher_failure_remains_fatal_in_reasoning_profile(monkeypatch):
    worker, published = _worker(monkeypatch)
    worker.teacher_server_manager.compute_teacher_sequence_score_single.side_effect = TeacherScoringError(
        "Teacher endpoint failed"
    )
    with pytest.raises(TeacherScoringError, match="Teacher endpoint failed"):
        asyncio.run(_postprocess(worker, _output(eos=55)))
    assert not published


def test_published_exclusion_has_zero_gradient_and_does_not_dilute_normalizers(monkeypatch):
    worker, published = _worker(monkeypatch)
    asyncio.run(_postprocess(worker, _output()))
    row_mask = _v1_train_sample_mask_from_tags(
        ["swe_0_0", "audio_0_0", *published["keys"]], [{}, {}, *published["tags"]],
    )
    normalizers = TensorDict({
        "response_mask": torch.ones(3, 2, dtype=torch.bool),
        "train_sample_mask": row_mask,
        "teacher_sequence_mask": torch.tensor([False, True, False]),
    }, batch_size=[3])
    prepare_distillation_minibatch(normalizers)
    assert tu.get_non_tensor_data(normalizers, "global_batch_size", None) == 2
    assert tu.get_non_tensor_data(normalizers, "audio_sequence_global_count", None) == 1
    assert tu.get_non_tensor_data(normalizers, "swe_sequence_global_count", None) == 1
    assert not normalizers["loss_mask"][2].any()

    student = torch.tensor([-2., -3., 0., -4., -5., 0., -8., -9., 0.], requires_grad=True)
    data = {
        "prompts": torch.tensor([[1], [1], [1]]),
        "responses": torch.tensor([[2, 55], [2, 55], [2, 3]]),
        "attention_mask": torch.ones(3, 3, dtype=torch.long),
        "response_mask": normalizers["response_mask"],
        "train_sample_mask": row_mask,
        "teacher_sequence_mask": normalizers["teacher_sequence_mask"],
        "teacher_sequence_logprob": torch.tensor([0., -3., 0.]),
        "old_log_probs": student.detach().reshape(3, 3)[:, :2],
        "teacher_logprobs": torch.tensor([[-1.], [-1.], [0.], [0.], [0.], [0.], [0.], [0.], [0.]]),
        "dp_size": 1, "batch_num_tokens": 4,
        **{key: tu.get_non_tensor_data(normalizers, key, None) for key in (
            "global_batch_size", "audio_sequence_global_count", "swe_sequence_global_count",
        )},
    }
    actor = ActorConfig(strategy="fsdp2", rollout_n=8, use_dynamic_bsz=True, loss_agg_mode="seq-mean-token-mean")
    teacher = SimpleNamespace(distillation_loss=DistillationLossConfig(
        loss_mode="k1", use_policy_gradient=True, use_task_rewards=False,
        loss_max_clamp=None, audio_sequence_surrogate=True,
    ))
    loss, metrics = distillation_ppo_loss(actor, teacher, {"log_probs": student}, data)
    loss.backward()
    assert loss.item() == pytest.approx(-3.75)
    torch.testing.assert_close(student.grad, torch.tensor([-.25, -.5, 0., -3., -3., 0., 0., 0., 0.]))
    assert metrics["distillation/audio/accepted_sequences"].aggregate() == 1
    assert metrics["distillation/swe/accepted_sequences"].aggregate() == 1

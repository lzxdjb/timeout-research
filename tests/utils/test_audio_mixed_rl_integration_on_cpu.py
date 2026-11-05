"""Teacher-free audio RL must preserve SWE rewards and exclude incomplete rows."""

import asyncio
import copy
import json
from collections import Counter
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
import torch
from datasets import Dataset
from omegaconf import OmegaConf
from tensordict import TensorDict
from torchdata.stateful_dataloader import StatefulDataLoader
from transfer_queue import KVBatchMeta

from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopOutput
from verl.protocol import DataProto
from verl.trainer.ppo.core_algos import AdvantageEstimator
from verl.trainer.ppo.v1 import agent_loop_tq
from verl.trainer.ppo.v1.audio_rl import audio_rl_grpo_valid_mask, audio_rl_source_metrics
from verl.trainer.ppo.v1.trainer_base import PPOTrainer, _v1_train_sample_mask_from_tags
from verl.trainer.ppo.v1.utils import compute_advantage_for_multi_trajectories, v1_success_values
from verl.utils import tensordict_utils as tu
from verl.utils.dataset.audio_opd_sampler import AudioOPDQuotaSampler
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.reward_score.audio import compute_score
from verl.workers.config import ActorConfig
from verl.workers.utils.losses import ppo_loss


def _config(*, enabled=True):
    return OmegaConf.create(
        {
            "trainer": {
                "audio_rl": {
                    "enabled": enabled,
                    "response_length": 2,
                    "enable_thinking": False,
                    "exclude_incomplete_sequences": True,
                }
            },
            "distillation": {"enabled": False},
            "data": {"max_response_length": 100},
            "actor_rollout_ref": {"rollout": {"response_length": 100, "n": 8}},
            "algorithm": {"adv_estimator": "grpo", "use_kl_in_reward": False, "gamma": 1, "lam": 1},
        }
    )


def _worker(monkeypatch, *, enabled=True, solution="A bell rings."):
    worker_class = agent_loop_tq.AgentLoopWorkerTQ.__ray_metadata__.modified_class
    worker = object.__new__(worker_class)
    worker.config = _config(enabled=enabled)
    worker.distillation_enabled = False
    worker.teacher_key = "teacher_route"
    worker.model_config = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=77))
    worker.tokenizer = SimpleNamespace(eos_token_id=55, decode=lambda *args, **kwargs: solution)
    # A teacher must never be needed, even when audio metadata retains an OPD route.
    worker.teacher_server_manager = SimpleNamespace(
        compute_teacher_logprobs_single=AsyncMock(side_effect=AssertionError("Unexpected teacher request")),
        compute_teacher_sequence_score_single=AsyncMock(side_effect=AssertionError("Unexpected teacher request")),
    )
    worker.repeat_coordinator = None
    worker._repeat_debug_event = lambda *args, **kwargs: None
    worker.cutoff_session_states = {}
    worker._compute_multi_modal_inputs = lambda *args: {}
    worker._get_mm_processor_kwargs = lambda *args: {}
    worker._compute_position_ids = lambda inputs, *args: torch.arange(inputs.shape[-1]).unsqueeze(0)

    async def score(data):
        specification = data.non_tensor_batch["reward_model"][0]["ground_truth"]
        reward = compute_score(str(data.non_tensor_batch["data_source"][0]), solution, specification)
        return {"reward_score": reward, "reward_extra_info": {"acc": reward, "audio_rule_scored": True}}

    reward_call = AsyncMock(side_effect=score)
    worker.reward_loop_worker_handles = [SimpleNamespace(compute_score=SimpleNamespace(remote=reward_call))]
    worker._compute_score = AsyncMock(wraps=worker._compute_score)
    published = {}

    async def publish(**kwargs):
        published.update(kwargs)

    async def await_publish(partition, uid, operation):
        return await operation

    worker._await_tq_publication = await_publish
    monkeypatch.setattr(agent_loop_tq.tq, "async_kv_batch_put", publish)
    monkeypatch.setattr(agent_loop_tq, "strict_validation_enabled", lambda: False)
    return worker, published, reward_call


def _output(*, eos=55, reward=None):
    return AgentLoopOutput(
        prompt_ids=[1],
        response_ids=[2, eos],
        response_mask=[1, 1],
        reward_score=reward,
        metrics=AgentLoopMetrics(),
        extra_fields={},
    )


def _postprocess(worker, output, *, validate=False, source="audio_librispeech", specification=None):
    specification = specification or {"task": "asr", "language": "en", "reference": "a bell rings"}
    return worker._agent_loop_postprocess(
        output,
        validate,
        uid="same_prompt_uid",
        session_id=0,
        global_steps=0,
        data_source=source,
        teacher_route="audio" if source.startswith("audio_") else "swe",
        raw_prompt=[{"role": "user", "content": "Listen"}],
        reward_model={"ground_truth": json.dumps(specification)},
    )


def _assert_no_teacher_fields(worker, published):
    assert not any(str(key).startswith("teacher_") for key in published["fields"] if key != "teacher_route")
    extras = published["fields"]["extra_fields"][0]
    assert not any(str(key).startswith("teacher_") for key in extras)
    worker.teacher_server_manager.compute_teacher_logprobs_single.assert_not_awaited()
    worker.teacher_server_manager.compute_teacher_sequence_score_single.assert_not_awaited()


@pytest.mark.parametrize(
    "source,solution,specification,expected",
    [
        ("audio_librispeech", "A bell rings.", {"task": "asr", "language": "en", "reference": "a bell rings"}, 1.0),
        ("audio_aishell1", "你 好。", {"task": "asr", "language": "zh", "reference": "你好"}, 1.0),
        (
            "audio_clotho_aqa",
            "A bell.",
            {
                "task": "closed_qa",
                "language": "en",
                "reference": "bell",
                "reference_aliases": ["a bell"],
            },
            1.0,
        ),
    ],
)
@pytest.mark.parametrize("eos", [55, 77])
def test_completed_audio_reaches_actual_rule_reward_without_teacher(
    monkeypatch, source, solution, specification, expected, eos
):
    worker, published, reward_call = _worker(monkeypatch, solution=solution)
    output = _output(eos=eos)
    asyncio.run(_postprocess(worker, output, source=source, specification=specification))
    reward_call.assert_awaited_once()
    assert output.reward_score == pytest.approx(expected)
    assert published["fields"]["rm_scores"].tolist() == [[0.0, expected]]
    assert published["tags"][0].get("train_sample_mask", True)
    assert not published["tags"][0].get("audio_rl_candidate_excluded", False)
    assert published["tags"][0]["mixed_rl_source"] == source
    _assert_no_teacher_fields(worker, published)


@pytest.mark.parametrize("source", ["swe_agent_verl", "rebench"])
def test_mixed_rl_preserves_embedded_swe_reward_without_reward_worker(monkeypatch, source):
    worker, published, reward_call = _worker(monkeypatch)
    output = _output(eos=3, reward=0.625)
    output.extra_fields["reward_extra_info"] = {"acc": 1, "hidden_pass": 1, "tool_failure_count": 2}
    before = copy.deepcopy(output.extra_fields)
    asyncio.run(_postprocess(worker, output, source=source))
    reward_call.assert_not_awaited()
    assert output.reward_score == 0.625
    assert output.extra_fields == before
    assert published["fields"]["rm_scores"].tolist() == [[0.0, 0.625]]
    assert published["tags"][0].get("train_sample_mask", True)
    assert published["tags"][0]["mixed_rl_source"] == "swe"
    _assert_no_teacher_fields(worker, published)


def test_incomplete_audio_is_published_with_loss_mask_and_no_reward_or_teacher(monkeypatch):
    worker, published, reward_call = _worker(monkeypatch)
    output = _output(eos=3)
    original = (list(output.response_ids), list(output.response_mask))
    asyncio.run(_postprocess(worker, output))
    worker._compute_score.assert_not_awaited()
    reward_call.assert_not_awaited()
    assert output.reward_score == 0.0
    assert (output.response_ids, output.response_mask) == original
    assert published["tags"][0]["train_sample_mask"] is False
    assert published["tags"][0]["audio_rl_candidate_excluded"] is True
    assert _v1_train_sample_mask_from_tags(published["keys"], published["tags"]).tolist() == [False]
    assert published["fields"]["rm_scores"].count_nonzero().item() == 0
    _assert_no_teacher_fields(worker, published)


@pytest.mark.parametrize("enabled,validate", [(False, False), (True, True), (False, True)])
def test_disabled_profile_and_validation_preserve_existing_scoring(monkeypatch, enabled, validate):
    worker, published, reward_call = _worker(monkeypatch, enabled=enabled)
    output = _output(eos=3)
    asyncio.run(_postprocess(worker, output, validate=validate))
    reward_call.assert_awaited_once()
    assert published["partition_id"] == ("val" if validate else "train")
    assert published["tags"][0].get("train_sample_mask", True)
    assert "audio_rl_candidate_excluded" not in published["tags"][0]
    if not enabled:
        assert "mixed_rl_source" not in published["tags"][0]
    _assert_no_teacher_fields(worker, published)


def test_audio_rl_mask_intersects_evaluator_mask_without_changing_legacy_exclusions():
    keys = [f"original_uid_{index}_0" for index in range(5)]
    tags = [
        {"audio_rl_candidate_excluded": True, "train_sample_mask": False},
        {},
        {"train_sample_mask": False},
        {"audio_opd_candidate_excluded": True, "train_sample_mask": False},
        {},
    ]
    evaluator_mask = torch.tensor([True, False, True, True, True])
    result = audio_rl_grpo_valid_mask(_config(), keys, tags, evaluator_mask)
    assert result.tolist() == [False, False, True, True, True]
    assert evaluator_mask.tolist() == [True, False, True, True, True]
    assert (
        audio_rl_grpo_valid_mask(_config(enabled=False), keys, tags, evaluator_mask).tolist() == evaluator_mask.tolist()
    )
    opd = _config(enabled=False)
    opd.distillation.enabled = True
    assert audio_rl_grpo_valid_mask(opd, keys, tags, evaluator_mask) is evaluator_mask
    absent = _config()
    del absent.trainer.audio_rl
    assert audio_rl_grpo_valid_mask(absent, keys, tags, evaluator_mask) is evaluator_mask


def test_audio_rl_validity_excludes_every_output_of_the_excluded_session_only():
    keys = ["same_uid_0_0", "same_uid_0_1", "same_uid_1_0", "other_uid_0_0"]
    tags = [{}, {"audio_rl_candidate_excluded": True}, {}, {}]
    evaluator_mask = torch.ones(4, dtype=torch.bool)
    keep = audio_rl_grpo_valid_mask(_config(), keys, tags, evaluator_mask)
    assert keep.tolist() == [False, False, True, True]


def test_audio_rl_exclusion_is_absent_from_success_metrics_but_zero_reward_remains_valid():
    assert v1_success_values([{"audio_rl_candidate_excluded": True}, {}, {}], [0.0, 0.0, 1.0]) == [None, 0.0, 1.0]


def test_source_metrics_count_prompts_and_sessions_without_double_counting_outputs():
    keys = ["audio_uid_0_0", "audio_uid_0_1", "audio_uid_1_0", "swe_uid_0_0", "padding_0_0"]
    tags = [
        {"mixed_rl_source": "audio_librispeech"},
        {"mixed_rl_source": "audio_librispeech", "audio_rl_candidate_excluded": True},
        {"mixed_rl_source": "audio_librispeech"},
        {"mixed_rl_source": "swe"},
        {"mixed_rl_source": "swe", "is_padding": True},
    ]
    metrics = audio_rl_source_metrics(
        _config(), keys, tags, [False, False, True, True, True], {"swe": 16, "audio_librispeech": 18}
    )
    prefix = "training/mixed_rl/audio_librispeech/"
    assert metrics[prefix + "requested_prompts_total"] == 18
    assert metrics[prefix + "admitted_prompts"] == 1
    assert metrics[prefix + "admitted_trajectories"] == 2
    assert metrics[prefix + "eligible_trajectories"] == 1
    assert metrics[prefix + "excluded_incomplete_trajectories"] == 1
    assert metrics["training/mixed_rl/swe/admitted_trajectories"] == 1
    assert audio_rl_source_metrics(_config(enabled=False), keys, tags, [True] * 5, {}) == {}


@pytest.mark.parametrize("excluded_reward", [-1e9, 1e9, float("nan")])
def test_grpo_excludes_incomplete_audio_from_its_original_prompt_group(excluded_reward):
    keys = [f"original_uid_{index}_0" for index in range(4)] + ["other_uid_0_0", "other_uid_1_0"]
    tags = [{"audio_rl_candidate_excluded": True, "train_sample_mask": False}, {}, {}, {}, {}, {}]
    keep = audio_rl_grpo_valid_mask(_config(), keys, tags, torch.ones(6, dtype=torch.bool))
    data = DataProto.from_dict(
        tensors={
            "token_level_rewards": torch.tensor(
                [[0.0, excluded_reward], [0.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 0.0], [0.0, 1.0]]
            ),
            "response_mask": torch.ones(6, 2),
            "grpo_valid_sample_mask": keep,
            "train_sample_mask": _v1_train_sample_mask_from_tags(keys, tags),
        },
        non_tensors={"uid": np.array(["original_uid"] * 4 + ["other_uid"] * 2)},
    )
    output = compute_advantage_for_multi_trajectories(data, keys, AdvantageEstimator.GRPO, num_repeat=4)
    expected = torch.tensor([-2 / 3, 1 / 3, 1 / 3]) / (torch.tensor([0.0, 1.0, 1.0]).std() + 1e-6)
    torch.testing.assert_close(output.batch["advantages"][1:4, 0], expected)
    assert output.batch["advantages"][0].count_nonzero().item() == 0
    assert output.batch["advantages"][4, 0] < 0 < output.batch["advantages"][5, 0]
    assert data.non_tensor_batch["uid"].tolist() == ["original_uid"] * 4 + ["other_uid"] * 2


@pytest.mark.parametrize("remaining", [0, 1])
def test_grpo_with_fewer_than_two_valid_audio_rollouts_has_zero_advantage(remaining):
    keys = [f"audio_prompt_{index}_0" for index in range(4)]
    tags = [{} if index < remaining else {"audio_rl_candidate_excluded": True} for index in range(4)]
    keep = audio_rl_grpo_valid_mask(_config(), keys, tags, torch.ones(4, dtype=torch.bool))
    data = DataProto.from_dict(
        tensors={
            "token_level_rewards": torch.tensor([[0.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 1.0]]),
            "response_mask": torch.ones(4, 2),
            "grpo_valid_sample_mask": keep,
        },
        non_tensors={"uid": np.array(["audio_prompt"] * 4)},
    )
    output = compute_advantage_for_multi_trajectories(data, keys, AdvantageEstimator.GRPO, num_repeat=4)
    assert output.batch["advantages"].count_nonzero().item() == 0


def test_exclusion_reaches_real_trainer_and_actor_with_zero_gradient(monkeypatch):
    monkeypatch.delenv("SWE_AGENT_TASK_FILTER_TRAINING_ENABLED", raising=False)
    keys = [f"audio_prompt_{index}_0" for index in range(4)]
    tags = [{"audio_rl_candidate_excluded": True, "train_sample_mask": False}, {}, {}, {}]
    batch = KVBatchMeta(partition_id="train", keys=keys, tags=tags)
    queue_data = tu.get_tensordict(
        {
            "uid": ["audio_prompt"] * 4,
            "response_mask": torch.nested.nested_tensor([torch.tensor([1, 1])] * 4, layout=torch.jagged),
            "rm_scores": torch.nested.nested_tensor(
                [
                    torch.tensor([0.0, 1e9]),
                    torch.tensor([0.0, 0.0]),
                    torch.tensor([0.0, 1.0]),
                    torch.tensor([0.0, 1.0]),
                ],
                layout=torch.jagged,
            ),
            "extra_fields": [{"reward_extra_info": {}}] * 4,
        }
    )
    trainer = SimpleNamespace(
        config=_config(),
        timeout_prediction_enabled=False,
        _apply_v1_task_filter=lambda *args: None,
    )
    with (
        patch("verl.trainer.ppo.v1.trainer_base.tq.kv_batch_get", return_value=queue_data),
        patch("verl.trainer.ppo.v1.trainer_base.tq.kv_batch_put", return_value=batch) as publish,
    ):
        PPOTrainer._compute_advantage(trainer, batch, {})
    fields = publish.call_args.kwargs["fields"]
    assert fields["train_sample_mask"].tolist() == [False, True, True, True]
    assert fields["train_exclusion_code"].tolist() == [4 | 64, 0, 0, 0]
    advantages = fields["advantages"].to_padded_tensor(0)
    expected = torch.tensor([-2 / 3, 1 / 3, 1 / 3]) / (torch.tensor([0.0, 1.0, 1.0]).std() + 1e-6)
    torch.testing.assert_close(advantages[1:, 0], expected)
    assert advantages[0].count_nonzero().item() == 0

    actor_data = TensorDict(
        {
            "prompts": torch.ones(4, 1, dtype=torch.long),
            "responses": torch.tensor([[2, 55]] * 4),
            "attention_mask": torch.ones(4, 3, dtype=torch.long),
            "response_mask": torch.ones(4, 2, dtype=torch.bool),
            "train_sample_mask": fields["train_sample_mask"],
            "old_log_probs": torch.zeros(4, 2),
            "advantages": advantages,
        },
        batch_size=[4],
    )
    tu.assign_non_tensor(actor_data, dp_size=1, batch_num_tokens=6, global_batch_size=3)
    student_logprobs = torch.zeros(12, requires_grad=True)
    actor = ActorConfig(strategy="fsdp2", rollout_n=8, use_dynamic_bsz=True, loss_agg_mode="token-mean")
    loss, _ = ppo_loss(actor, {"log_probs": student_logprobs}, actor_data)
    loss.backward()
    assert student_logprobs.grad[:3].count_nonzero().item() == 0
    assert student_logprobs.grad[3:].count_nonzero().item() > 0


def test_pure_rl_source_quota_and_mid_quota_loader_resume_match_original_recipe():
    sources = {"swe": 16, "audio_librispeech": 18, "audio_aishell1": 18, "audio_clotho_aqa": 12}
    schedule = [{"steps": 100, "quotas": sources}]

    class RLPool:
        def __init__(self):
            self.dataframe = Dataset.from_dict({"opd_source": [source for source in sources for _ in range(100)]})

        def __len__(self):
            return len(self.dataframe)

        def __getitem__(self, index):
            return {"index": index, "source": self.dataframe[index]["opd_source"], "raw_prompt": "unchanged"}

    pool = RLPool()
    sampler = AudioOPDQuotaSampler(pool, schedule, 64, 42)
    loader = StatefulDataLoader(pool, batch_size=1, sampler=sampler, num_workers=0, collate_fn=collate_fn)
    stream = iter(loader)
    initial = [next(stream) for _ in range(13)]
    state = copy.deepcopy(loader.state_dict())
    expected = [next(stream) for _ in range(115)]
    assert Counter(row["source"][0] for row in initial + expected[:51]) == sources
    restored = StatefulDataLoader(
        pool,
        batch_size=1,
        sampler=AudioOPDQuotaSampler(pool, schedule, 64, 42),
        num_workers=0,
        collate_fn=collate_fn,
    )
    restored.load_state_dict(state)
    resumed_stream = iter(restored)
    actual = [next(resumed_stream) for _ in range(115)]
    assert [row["index"][0] for row in actual] == [row["index"][0] for row in expected]
    assert all(row["raw_prompt"] == ["unchanged"] for row in actual)
    assert dict(sampler.draw_counts) == {source: quota * 2 for source, quota in sources.items()}
    metrics = audio_rl_source_metrics(_config(), [], [], [], sampler.draw_counts)
    for source, quota in sources.items():
        assert metrics[f"training/mixed_rl/{source}/requested_prompts_total"] == quota * 2

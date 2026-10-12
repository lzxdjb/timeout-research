import numpy as np
import pytest
import torch

from verl.protocol import DataProto
from verl.trainer.ppo.core_algos import AdvantageEstimator
from verl.trainer.ppo.metric_utils import usable_validation_metrics
from verl.trainer.ppo.v1.replay_buffer import ReplayBuffer, unverified_group_decision
from verl.trainer.ppo.v1.trainer_base import _v1_train_sample_mask_from_tags
from verl.trainer.ppo.v1.utils import compute_advantage_for_multi_trajectories


@pytest.mark.parametrize("bad,drop", [(0, False), (1, False), (2, False), (3, True), (8, True)])
def test_ratio_boundary(bad, drop):
    tags = [{"unverified_trajectory": i < bad} for i in range(8)]
    assert unverified_group_decision(tags, 8, 0.25) == (drop, bad / 8, 8 - bad)


def test_missing_terminal_sessions_and_minimum_valid_are_not_silently_admitted():
    assert unverified_group_decision([{}] * 5, 8, 0.25)[0]
    assert unverified_group_decision([{}, {"unverified_trajectory": True}], 2, 1.0)[0]


def test_final_session_ratio_is_independent_of_infrastructure_gate():
    rb = ReplayBuffer.__new__(ReplayBuffer)
    rb.partitions = {"train": {
        **{f"group_{i}_0": {"unverified_trajectory": i < 2, "infrastructure_failure": i < 2}
           for i in range(8)},
        "group_2_1": {"unverified_trajectory": False},
    }}
    rb.finished_keys, rb.failure_keys = {"train": {"group"}}, {"train": set()}
    rb.completion_ratio_cutoff_keys = {"train": set()}
    rb.unverified_ratio_thresholds = {"train": 0.25}
    rb.infrastructure_failure_ratio_thresholds = {"train": 0.0}
    rb.expected_rollout_counts = {"train": 8}
    rb._infrastructure_classification_cache = {"train": {}}
    assert rb._infrastructure_filtered_uids("train")[0] == set()
    rb.partitions["train"]["group_2_1"]["unverified_trajectory"] = True
    assert rb._infrastructure_filtered_uids("train")[0] == {"group"}


def test_uncertain_reward_cannot_change_valid_siblings_advantages(monkeypatch):
    monkeypatch.setenv("SWE_AGENT_ROLLOUT_TRAINING_UNVERIFIED_TRAJECTORY_RATIO_THRESHOLD", "0.25")
    keys = [f"group_{i}_0" for i in range(4)] + ["group_0_1"]
    tags = [{"unverified_trajectory": True}, {}, {}, {}, {"unverified_trajectory": True}]
    keep = _v1_train_sample_mask_from_tags(keys, tags)
    assert keep.tolist() == [False, True, True, True, False]
    for uncertain_reward in (-1e6, 1e6, float("nan")):
        data = DataProto.from_dict(tensors={
            "token_level_rewards": torch.tensor([[0.0, uncertain_reward], [0.0, 0.0],
                                                  [0.0, 1.0], [0.0, 1.0], [0.0, uncertain_reward]]),
            "response_mask": torch.ones(5, 2), "train_sample_mask": keep,
            "grpo_valid_sample_mask": keep,
        }, non_tensors={"uid": np.array(["group"] * 5)})
        output = compute_advantage_for_multi_trajectories(data, keys, AdvantageEstimator.GRPO, num_repeat=4)
        expected = torch.tensor([-2 / 3, 1 / 3, 1 / 3]) / (torch.tensor([0., 1., 1.]).std() + 1e-6)
        torch.testing.assert_close(output.batch["advantages"][1:4, 0], expected)
        assert torch.count_nonzero(output.batch["advantages"][[0, 4]]) == 0


def test_validation_ratio_uses_only_confirmed_siblings():
    sources = ["bench"] * 16
    uids = ["boundary"] * 8 + ["drop"] * 8
    uncertain = [1, 1, 0, 0, 0, 0, 0, 0] + [1, 1, 1, 0, 0, 0, 0, 0]
    _, retained, values, coverage = usable_validation_metrics(
        sources, uids, {"reward": [0] * 16, "evaluator_group_excluded": uncertain,
                        "observed_infrastructure_failure": uncertain},
        unverified_threshold=0.25, expected_rollout_count=8)
    assert retained == ["boundary"] * 6
    assert values["reward"] == [0] * 6
    assert coverage["bench"]["valid_count"] == 6
    assert coverage["bench"]["unverified_excluded_groups"] == 1


def test_success_metrics_exclude_uncertain_rewards_and_keep_confirmed_siblings():
    from verl.trainer.ppo.v1.utils import compute_v1_success_ratio_metrics
    metrics = compute_v1_success_ratio_metrics(
        batch_keys=[f"group_{i}_0" for i in range(8)],
        batch_tags=[{"unverified_trajectory": True, "infrastructure_failure": True}] * 2 + [{}] * 6,
        success_values=[-100, 100, 0, 0, 1, 1, 1, 1], prefix="training",
        infrastructure_failure_ratio_threshold=0.0, expected_rollout_count=8,
        unverified_threshold=0.25, metric_sources=["rebench"] * 8)
    assert metrics["training/success_ratio/excluding_completion_cutoff_and_infrastructure_eligible_count"] == 6
    assert metrics["training/success_ratio/excluding_completion_cutoff_and_infrastructure"] == pytest.approx(4 / 6)
    assert metrics["training/rebench/success_ratio/excluding_completion_cutoff_and_infrastructure"] == pytest.approx(4 / 6)


def test_legacy_task_filter_does_not_double_count_evaluator_uncertainty(monkeypatch):
    from types import SimpleNamespace
    from omegaconf import OmegaConf
    from verl.trainer.ppo.v1.trainer_base import PPOTrainer
    monkeypatch.setenv("SWE_AGENT_TASK_FILTER_TRAINING_ENABLED", "1")
    monkeypatch.setenv("SWE_AGENT_TASK_FILTER_INFRA_RATIO_THRESHOLD", "0.25")
    monkeypatch.setenv("SWE_AGENT_TASK_FILTER_MIN_GROUP_ATTEMPTS", "8")
    monkeypatch.setenv("SWE_AGENT_TASK_FILTER_INCLUDE_QUEUE_FAILURES", "1")
    trainer = SimpleNamespace(
        config=OmegaConf.create({"algorithm": {"adv_estimator": "grpo"},
                                 "actor_rollout_ref": {"rollout": {"n": 8}}}),
        _extract_v1_infrastructure_metadata=PPOTrainer._extract_v1_infrastructure_metadata)
    keep = torch.tensor([False, False, True, True, True, True, True, True])
    data = DataProto.from_dict(tensors={"response_mask": torch.ones(8, 2),
                                       "train_sample_mask": keep, "grpo_valid_sample_mask": keep},
                               non_tensors={"uid": np.array(["group"] * 8)})
    extra = [{"reward_extra_info": {"observed_infrastructure_failure": int(i < 3),
                                     "observed_infrastructure_failure_code": 2}} for i in range(8)]
    metrics = {}
    PPOTrainer._apply_v1_task_filter(trainer, data, extra, metrics)
    assert metrics["task_filter/groups_excluded"] == 0
    assert data.batch["train_sample_mask"].tolist() == keep.tolist()
    assert data.non_tensor_batch["task_filter_ratio"].tolist() == [1 / 8] * 8


def test_validation_refill_preserves_prompt_and_stops_after_bounded_retries(monkeypatch):
    from types import SimpleNamespace
    from tensordict import NonTensorData, NonTensorStack, TensorDict
    from verl.trainer.ppo.v1.trainer_base import PPOTrainer
    monkeypatch.setenv("SWE_AGENT_ROLLOUT_VALIDATION_MAX_GROUP_RETRIES", "2")
    template = TensorDict({"uid": NonTensorStack.from_list([NonTensorData("original")]),
                           "prompt": torch.tensor([42])}, batch_size=[1])
    trainer = SimpleNamespace(_validation_prompt_templates={"original": template})
    submitted = []
    def submit(batch):
        uid = str(batch["uid"][0])
        assert batch["prompt"].item() == 42
        trainer._validation_prompt_templates[uid] = batch.clone()
        submitted.append(uid)
        return len(batch)
    trainer._submit_validation_batch = submit
    assert PPOTrainer._refill_validation_groups(trainer, ["original"]) == 1
    assert PPOTrainer._refill_validation_groups(trainer, [submitted[-1]]) == 1
    assert len(set(["original", *submitted])) == 3
    with pytest.raises(RuntimeError, match="no complete validation score"):
        PPOTrainer._refill_validation_groups(trainer, [submitted[-1]])

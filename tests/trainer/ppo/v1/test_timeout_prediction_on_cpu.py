from types import SimpleNamespace

import numpy as np
import torch
from tensordict import TensorDict

from verl.protocol import DataProto
from verl.trainer.ppo.timeout_prediction import TimeoutRewardPredictor
from verl.trainer.ppo.v1.trainer_base import PPOTrainer


class _TestTrainer(PPOTrainer):
    def on_step_end(self):
        pass

    def on_sample_end(self):
        pass


def test_v1_completion_cutoff_prediction_excludes_infrastructure_rows() -> None:
    trainer = object.__new__(_TestTrainer)
    trainer.timeout_prediction_cfg = {"confidence_delta": 0.5, "reward_assignment": "terminal"}
    trainer.timeout_predictor = TimeoutRewardPredictor()
    with torch.no_grad():
        for parameter in trainer.timeout_predictor.parameters():
            parameter.zero_()
        trainer.timeout_predictor.net[-1].bias.copy_(torch.tensor([5.0, 0.0, 0.0]))

    data = DataProto(
        batch=TensorDict(
            {
                "response_mask": torch.ones(3, 2, dtype=torch.long),
                "old_log_probs": torch.zeros(3, 2),
                "rm_scores": torch.tensor([[0.0, -0.1], [0.0, 0.0], [0.0, 0.0]]),
            },
            batch_size=3,
        )
    )
    tags = [{}, {"completion_ratio_cutoff": True}, {"completion_ratio_cutoff": True}]
    extra_fields = [
        {"reward_extra_info": {"observed_infrastructure_failure": 0}},
        {"reward_extra_info": {"observed_infrastructure_failure": 0}},
        {"reward_extra_info": {"observed_infrastructure_failure": 1}},
    ]

    scores, train_sample_mask, calibration_mask = trainer._apply_v1_timeout_prediction(
        data, tags, extra_fields
    )

    assert torch.isclose(scores[1, -1], torch.tensor(-0.1))
    assert scores[2].eq(0).all()
    assert train_sample_mask.tolist() == [True, True, False]
    assert calibration_mask.tolist() == [True, False, False]


def _filter_trainer(adv_estimator: str = "grpo", rollout_n: int = 8):
    trainer = object.__new__(_TestTrainer)
    trainer.config = SimpleNamespace(
        algorithm=SimpleNamespace(adv_estimator=adv_estimator),
        actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(n=rollout_n)),
    )
    return trainer


def test_v1_task_filter_reads_nested_infrastructure_metadata(monkeypatch) -> None:
    monkeypatch.setenv("SWE_AGENT_TASK_FILTER_TRAINING_ENABLED", "1")
    monkeypatch.setenv("SWE_AGENT_TASK_FILTER_INFRA_RATIO_THRESHOLD", "0.25")
    monkeypatch.setenv("SWE_AGENT_TASK_FILTER_MIN_GROUP_ATTEMPTS", "8")
    data = DataProto.from_single_dict(
        {
            "response_mask": torch.ones(16, 2, dtype=torch.long),
            "uid": np.asarray(["bad"] * 8 + ["good"] * 8, dtype=object),
        }
    )
    extra_fields = [
        {
            "reward_extra_info": {
                "observed_infrastructure_failure": int(i < 3),
                "observed_infrastructure_failure_code": 3,
            }
        }
        for i in range(8)
    ] + [{"reward_extra_info": {"observed_infrastructure_failure": 0}} for _ in range(8)]

    trainer = _filter_trainer()
    metrics = {}
    trainer._apply_v1_task_filter(data, extra_fields, metrics)

    assert data.batch["train_sample_mask"].tolist() == [False] * 8 + [True] * 8
    assert metrics["task_filter/groups_excluded"] == 1.0


def test_v1_task_filter_intersects_existing_timeout_mask(monkeypatch) -> None:
    monkeypatch.setenv("SWE_AGENT_TASK_FILTER_TRAINING_ENABLED", "1")
    monkeypatch.setenv("SWE_AGENT_TASK_FILTER_INFRA_RATIO_THRESHOLD", "0.25")
    monkeypatch.setenv("SWE_AGENT_TASK_FILTER_MIN_GROUP_ATTEMPTS", "8")
    data = DataProto.from_single_dict(
        {
            "response_mask": torch.ones(16, 2, dtype=torch.long),
            "uid": np.asarray(["bad"] * 8 + ["good"] * 8, dtype=object),
            "train_sample_mask": torch.tensor([True] * 8 + [False] * 8),
        }
    )
    extra_fields = [
        {"reward_extra_info": {"observed_infrastructure_failure": int(i < 3)}} for i in range(8)
    ] + [{"reward_extra_info": {"observed_infrastructure_failure": 0}} for _ in range(8)]

    trainer = _filter_trainer()
    trainer._apply_v1_task_filter(data, extra_fields, {})

    assert data.batch["train_sample_mask"].tolist() == [False] * 16
    assert data.meta_info["task_filter_all_excluded"] is True


def test_v1_task_filter_prefers_direct_metadata_when_present() -> None:
    failure, code = PPOTrainer._extract_v1_infrastructure_metadata(
        [{"reward_extra_info": {"observed_infrastructure_failure": 0}}],
        {
            "observed_infrastructure_failure": [1],
            "observed_infrastructure_failure_code": [3],
        },
    )

    assert failure.tolist() == [1]
    assert code.tolist() == [3]


def test_v1_task_filter_is_a_noop_by_default(monkeypatch) -> None:
    monkeypatch.delenv("SWE_AGENT_TASK_FILTER_TRAINING_ENABLED", raising=False)
    data = DataProto.from_single_dict({"response_mask": torch.ones(1, 2, dtype=torch.long)})

    _filter_trainer()._apply_v1_task_filter(data, [{}], {})

    assert "train_sample_mask" not in data.batch

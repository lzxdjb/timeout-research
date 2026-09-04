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

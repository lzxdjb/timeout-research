import torch

from verl.trainer.ppo.timeout_prediction import (
    TimeoutRewardPredictor,
    build_timeout_rewards,
    conformal_radius,
    reward_classes,
)


def test_reward_classes_cover_all_three_values():
    assert reward_classes(torch.tensor([-0.1, 0.0, 1.0])).tolist() == [0, 1, 2]


def test_conformal_radius_uses_true_class_for_negative_and_zero_rewards():
    probabilities = torch.tensor(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.7, 0.2],
            [0.1, 0.2, 0.7],
        ]
    )
    rewards = torch.tensor([-0.1, 0.0, 1.0])
    # Residuals are 0.2, 0.3, and 0.3; with delta=0.25 the calibrated rank is 3.
    assert conformal_radius(probabilities, rewards, delta=0.25) == 0.3


def test_build_timeout_rewards_keeps_only_confident_timeout_rows():
    reward_tensor = torch.zeros(2, 4)
    reward_tensor[0, 2] = 1.0
    response_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.bool)
    timeout_mask = torch.tensor([False, True])
    probabilities = torch.tensor([[0.1, 0.2, 0.7], [0.8, 0.1, 0.1]])
    keep_mask = torch.tensor([True, True])
    scores, train_mask = build_timeout_rewards(
        reward_tensor,
        response_mask,
        timeout_mask,
        probabilities,
        keep_mask,
        assignment="terminal",
    )
    assert scores[1].tolist() == [0.0, 0.0, 0.0, -0.1]
    assert train_mask.tolist() == [[True, True, True, False], [True, True, True, True]]


def test_predictor_outputs_three_class_probabilities():
    predictor = TimeoutRewardPredictor()
    features = torch.zeros(4, 2)
    probabilities, values = predictor.predict(features)
    assert probabilities.shape == (4, 3)
    assert torch.allclose(probabilities.sum(dim=-1), torch.ones(4))
    assert values.shape == (4,)

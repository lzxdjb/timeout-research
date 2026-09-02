"""Small, checkpoint-independent utilities for timeout reward imputation."""

from __future__ import annotations

import torch
from torch import nn


class TimeoutRewardPredictor(nn.Module):
    """Predict one of the three task-reward classes for a trajectory.

    This sidecar deliberately consumes tensors already produced by PPO.  It is
    kept outside the actor model so existing model and optimizer checkpoints
    remain compatible.
    """

    class_values = (-0.1, 0.0, 1.0)

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 3))

    @staticmethod
    def features(old_log_probs: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
        mask = response_mask.to(dtype=old_log_probs.dtype)
        lengths = mask.sum(dim=-1).clamp_min(1.0)
        mean_log_prob = (old_log_probs * mask).sum(dim=-1) / lengths
        normalized_length = lengths / float(max(1, old_log_probs.shape[-1]))
        return torch.stack((mean_log_prob, normalized_length), dim=-1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)

    def predict(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        probabilities = self(features).softmax(dim=-1)
        values = torch.as_tensor(self.class_values, device=probabilities.device, dtype=probabilities.dtype)
        return probabilities, probabilities @ values


def reward_classes(rewards: torch.Tensor) -> torch.Tensor:
    """Map numerical rewards to class indices ``{-0.1, 0, 1} -> {0, 1, 2}``."""
    values = rewards.to(torch.float32)
    distances = torch.stack(((values + 0.1).abs(), values.abs(), (values - 1.0).abs()), dim=-1)
    return distances.argmin(dim=-1)


def conformal_radius(probabilities: torch.Tensor, rewards: torch.Tensor, delta: float) -> float:
    """Compute a split-conformal radius using all three class coordinates.

    For a complete trajectory with class label ``y``, the score is
    ``max_c |p(c)-1{c=y}|``.  This is class-agnostic and reduces to
    ``1 - p(y)`` for a probability simplex.
    """
    if probabilities.ndim != 2 or probabilities.shape[-1] != 3:
        raise ValueError(f"Expected probabilities with shape (n, 3), got {tuple(probabilities.shape)}")
    labels = reward_classes(rewards).to(device=probabilities.device)
    if labels.numel() != probabilities.shape[0]:
        raise ValueError("Calibration rewards and probabilities must have the same number of rows")
    if labels.numel() == 0:
        return 1.0
    true_probability = probabilities.gather(1, labels.unsqueeze(-1)).squeeze(-1)
    residuals = (1.0 - true_probability).abs().flatten()
    delta = min(max(float(delta), 1e-8), 1.0 - 1e-8)
    rank = int(torch.ceil(torch.tensor((residuals.numel() + 1) * (1.0 - delta))).item())
    if rank > residuals.numel():
        return 1.0
    return float(torch.kthvalue(residuals, rank).values.item())


def build_timeout_rewards(
    reward_tensor: torch.Tensor,
    response_mask: torch.Tensor,
    timeout_mask: torch.Tensor,
    probabilities: torch.Tensor,
    keep_mask: torch.Tensor,
    assignment: str = "terminal",
    normalize_broadcast: bool = False,
    infrastructure_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Impute retained timeout rewards while preserving the existing layout."""
    if assignment not in {"terminal", "broadcast"}:
        raise ValueError(f"Unsupported timeout reward assignment: {assignment}")
    scores = reward_tensor.clone()
    mask_bool = response_mask.to(torch.bool)
    lengths = mask_bool.sum(dim=-1).clamp_min(1)
    last_positions = lengths - 1
    predicted_classes = probabilities.argmax(dim=-1)
    predicted_rewards = torch.as_tensor(
        TimeoutRewardPredictor.class_values,
        device=probabilities.device,
        dtype=probabilities.dtype,
    ).index_select(0, predicted_classes)
    eligible = timeout_mask.to(torch.bool) & keep_mask.to(torch.bool)

    for row in eligible.nonzero(as_tuple=False).flatten().tolist():
        if not mask_bool[row].any():
            continue
        if assignment == "terminal":
            scores[row, last_positions[row]] = predicted_rewards[row].detach().to(scores.dtype)
        else:
            value = predicted_rewards[row].detach().to(scores.dtype)
            if normalize_broadcast:
                value = value / lengths[row].to(scores.dtype)
            scores[row] = torch.where(mask_bool[row], value, scores[row])

    base_keep = (
        ~infrastructure_mask.to(torch.bool)
        if infrastructure_mask is not None
        else torch.ones_like(timeout_mask, dtype=torch.bool)
    )
    train_mask = mask_bool & (base_keep & ((~timeout_mask.to(torch.bool)) | eligible)).unsqueeze(-1)
    return scores, train_mask

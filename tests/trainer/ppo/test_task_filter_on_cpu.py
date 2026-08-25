from __future__ import annotations

import numpy as np
import torch

from verl import DataProto
from verl.trainer.ppo.ray_trainer import apply_training_task_filter
from verl.workers.utils.padding import left_right_2_no_padding


def _batch() -> DataProto:
    return DataProto.from_single_dict(
        {
            "response_mask": torch.ones((16, 3), dtype=torch.int64),
            "uid": np.asarray(["group-a"] * 8 + ["group-b"] * 8, dtype=object),
            "infrastructure_failure": np.asarray([1, 1, 1, 0, 0, 0, 0, 0] + [0] * 8),
            "infrastructure_failure_code": np.asarray([3, 3, 3, 0, 0, 0, 0, 0] + [0] * 8),
        }
    )


def test_filter_masks_whole_prompt_group(monkeypatch) -> None:
    monkeypatch.setenv("SWE_AGENT_TASK_FILTER_TRAINING_ENABLED", "1")
    monkeypatch.setenv("SWE_AGENT_TASK_FILTER_INFRA_RATIO_THRESHOLD", "0.25")
    monkeypatch.setenv("SWE_AGENT_TASK_FILTER_MIN_GROUP_ATTEMPTS", "8")
    batch = _batch()

    metrics = apply_training_task_filter(batch)

    assert metrics["task_filter/groups_excluded"] == 1.0
    assert batch.batch["train_sample_mask"].tolist() == [False] * 8 + [True] * 8
    assert batch.non_tensor_batch["task_filter_excluded"].tolist() == [1] * 8 + [0] * 8
    assert metrics["task_filter/active_token_fraction"] == 0.5


def test_queue_failures_are_not_counted_by_default(monkeypatch) -> None:
    monkeypatch.setenv("SWE_AGENT_TASK_FILTER_TRAINING_ENABLED", "1")
    monkeypatch.setenv("SWE_AGENT_TASK_FILTER_INFRA_RATIO_THRESHOLD", "0.25")
    monkeypatch.setenv("SWE_AGENT_TASK_FILTER_MIN_GROUP_ATTEMPTS", "8")
    batch = _batch()
    batch.non_tensor_batch["infrastructure_failure_code"][:3] = 1

    metrics = apply_training_task_filter(batch)

    assert metrics["task_filter/groups_excluded"] == 0.0
    assert batch.batch["train_sample_mask"].all()


def test_filtered_loss_mask_preserves_response_mask() -> None:
    batch = DataProto.from_single_dict(
        {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "attention_mask": torch.ones((2, 3), dtype=torch.int64),
            "response_mask": torch.tensor([[0, 1, 1], [0, 1, 1]], dtype=torch.int64),
            "position_ids": torch.arange(3).repeat(2, 1),
            "train_sample_mask": torch.tensor([True, False]),
        }
    )

    td = left_right_2_no_padding(batch.to_tensordict())

    assert td["response_mask"].tolist() == [[0, 1, 1], [0, 1, 1]]
    assert td["loss_mask"].tolist() == [[0, 1, 1], [0, 0, 0]]


def test_all_excluded_batch_is_marked_for_noop_update(monkeypatch) -> None:
    monkeypatch.setenv("SWE_AGENT_TASK_FILTER_TRAINING_ENABLED", "1")
    monkeypatch.setenv("SWE_AGENT_TASK_FILTER_INFRA_RATIO_THRESHOLD", "0.25")
    monkeypatch.setenv("SWE_AGENT_TASK_FILTER_MIN_GROUP_ATTEMPTS", "8")
    batch = _batch()
    batch.non_tensor_batch["infrastructure_failure"][:] = 1
    batch.non_tensor_batch["infrastructure_failure_code"][:] = 3

    metrics = apply_training_task_filter(batch)

    assert metrics["task_filter/all_groups_excluded"] == 1.0
    assert batch.meta_info["task_filter_all_excluded"] is True
    assert not batch.batch["train_sample_mask"].any()


def test_per_sample_filter_masks_only_infrastructure_failures(monkeypatch) -> None:
    monkeypatch.setenv("SWE_AGENT_TASK_FILTER_TRAINING_ENABLED", "1")
    batch = DataProto.from_single_dict({"response_mask": torch.ones((3, 3), dtype=torch.int64)})
    batch.non_tensor_batch = {
        "uid": np.asarray(["a", "b", "c"], dtype=object),
        "infrastructure_failure": np.asarray([0, 1, 1]),
        "infrastructure_failure_code": np.asarray([0, 3, 1]),
    }

    metrics = apply_training_task_filter(batch, per_sample_infrastructure_filter=True)

    assert metrics["task_filter/mode_per_sample"] == 1.0
    assert batch.batch["train_sample_mask"].tolist() == [True, False, False]
    assert batch.non_tensor_batch["task_filter_excluded"].tolist() == [0, 1, 1]


def test_per_sample_filter_marks_queue_failures_as_infrastructure(monkeypatch) -> None:
    monkeypatch.setenv("SWE_AGENT_TASK_FILTER_TRAINING_ENABLED", "1")
    batch = DataProto.from_single_dict({"response_mask": torch.ones((2, 3), dtype=torch.int64)})
    batch.non_tensor_batch = {
        "uid": np.asarray(["a", "b"], dtype=object),
        "infrastructure_failure": np.asarray([1, 1]),
        "infrastructure_failure_code": np.asarray([1, 3]),
    }

    apply_training_task_filter(batch, per_sample_infrastructure_filter=True)

    assert batch.batch["train_sample_mask"].tolist() == [False, False]
    assert batch.meta_info["task_filter_all_excluded"] is True

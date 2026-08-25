from __future__ import annotations

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from verl import DataProto
from verl.experimental.fully_async_policy.detach_utils import _compute_partial_rollout_stats, addition_process
from verl.experimental.fully_async_policy.fully_async_rollouter import FullyAsyncRollouter
from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer
from verl.trainer.ppo.core_algos import AdvantageEstimator


def _rollouter_class():
    return FullyAsyncRollouter.__ray_metadata__.modified_class


def _trainer_class():
    return FullyAsyncTrainer.__ray_metadata__.modified_class


def test_addition_process_stores_timing_columns_as_numpy_arrays():
    output = DataProto.from_dict(tensors={"obs": torch.tensor([1, 2])})
    output.meta_info["metrics"] = [
        {"generate_sequences": 1.25, "tool_calls": 2.5},
        {"generate_sequences": 3.75, "tool_calls": 4.0},
    ]

    result = addition_process(output)

    assert np.array_equal(result.non_tensor_batch["processing_times"], np.array([1.25, 3.75]))
    assert np.array_equal(result.non_tensor_batch["tool_calls_times"], np.array([2.5, 4.0]))
    result.check_consistency()


def test_partial_rollout_stats_ignore_unknown_versions_on_cpu():
    stats = _compute_partial_rollout_stats(
        np.array([0, None, 1, None], dtype=object),
        np.array([0, None, 3, 2], dtype=object),
    )

    assert stats == {
        "fully_async/partial/total_partial_num": 1,
        "fully_async/partial/partial_ratio": 0.5,
        "fully_async/partial/max_partial_span": 2,
        "fully_async/partial/unknown_version_num": 2,
        "fully_async/partial/unknown_version_ratio": 0.5,
    }


def test_partial_rollout_stats_support_all_unknown_versions_on_cpu():
    stats = _compute_partial_rollout_stats(
        np.array([None, None], dtype=object),
        np.array([None, None], dtype=object),
    )

    assert stats["fully_async/partial/total_partial_num"] == 0
    assert stats["fully_async/partial/partial_ratio"] == 0.0
    assert stats["fully_async/partial/max_partial_span"] == 0
    assert stats["fully_async/partial/unknown_version_num"] == 2
    assert stats["fully_async/partial/unknown_version_ratio"] == 1.0


def test_stale_trajectory_metrics_ignore_unknown_versions_on_cpu():
    harness = SimpleNamespace(current_param_version=3, stale_trajectory_processed=4)
    batch = SimpleNamespace(
        meta_info={
            "trajectory_param_versions": np.array([1, None, 3], dtype=object),
            "fully_async/partial/unknown_version_num": 1,
        }
    )
    metrics = {}

    _trainer_class()._collect_metrics_from_samples(harness, batch, metrics)

    assert harness.stale_trajectory_processed == 5
    assert metrics["fully_async/count/stale_trajectory_processed"] == 5
    assert metrics["fully_async/count/current_param_version"] == 3
    assert metrics["fully_async/partial/unknown_version_num"] == 1


def test_fully_async_reward_path_captures_grpo_boundary(monkeypatch):
    monkeypatch.setenv("VERL_GRPO_DIAGNOSTICS", "0")
    batch = DataProto.from_dict(
        tensors={
            "responses": torch.tensor([[11, 12, 13], [21, 22, 23]]),
            "response_mask": torch.ones((2, 3), dtype=torch.long),
            "attention_mask": torch.ones((2, 5), dtype=torch.long),
            "rm_scores": torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]),
        },
        non_tensors={"uid": np.array(["group", "group"], dtype=object)},
    )
    harness = SimpleNamespace(
        timing_raw={},
        metrics={},
        use_rm=False,
        config=SimpleNamespace(
            algorithm=SimpleNamespace(adv_estimator=AdvantageEstimator.GRPO),
        ),
    )

    result = _trainer_class()._fit_compute_reward(harness, batch)

    assert result is batch
    assert torch.equal(harness.reward_tensor, batch.batch["rm_scores"])
    assert harness._grpo_boundary_baseline["reward"]
    assert harness._grpo_boundary_baseline["uid"]
    assert harness.metrics["grpo/boundary_reward_extraction_changed"] == 0.0


@pytest.mark.asyncio
async def test_rollouter_fit_propagates_streaming_failures():
    failure = RuntimeError("streaming failed")

    async def fail_generation():
        raise failure

    async def monitor_until_cancelled():
        await asyncio.Event().wait()

    harness = SimpleNamespace(
        message_queue_client=object(),
        lock=asyncio.Lock(),
        paused=False,
        running=False,
        _resume_event=asyncio.Event(),
        _streaming_generation_main=fail_generation,
        _async_monitor_loop=monitor_until_cancelled,
    )

    with pytest.raises(RuntimeError, match="streaming failed"):
        await _rollouter_class().fit(harness)


def test_close_swe_prefetchers_closes_both_scopes():
    closed = []
    harness = SimpleNamespace(
        _swe_training_image_prefetcher=SimpleNamespace(close=lambda: closed.append("training")),
        _swe_validation_image_prefetcher=SimpleNamespace(close=lambda: closed.append("validation")),
    )

    _rollouter_class().close_swe_prefetchers(harness)

    assert closed == ["training", "validation"]

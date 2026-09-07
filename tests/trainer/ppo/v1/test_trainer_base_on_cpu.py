# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest.mock import patch

import pytest
import torch
from omegaconf import OmegaConf
from transfer_queue import KVBatchMeta

from verl.trainer.ppo.v1.replay_buffer import ReplayBuffer, ReplayBufferAsync
from verl.trainer.ppo.v1.trainer_base import (
    PPOTrainer,
    _get_off_policy_step_arrays,
    _resolve_v1_sequence_lengths,
)


class _StubTrainer(PPOTrainer):
    def on_step_end(self):
        pass

    def on_sample_end(self):
        pass


class _CustomSampler:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _batch_with_tags(tags: list[dict]) -> KVBatchMeta:
    return KVBatchMeta(
        partition_id="train",
        keys=[f"row_{index}" for index in range(len(tags))],
        tags=tags,
    )


def test_v1_sequence_lengths_use_complete_tags_without_queue_read():
    batch = _batch_with_tags([{"seq_len": 3}, {"seq_len": 7}])

    with patch("verl.trainer.ppo.v1.trainer_base.tq.kv_batch_get") as kv_batch_get:
        lengths = _resolve_v1_sequence_lengths(batch)

    assert lengths.tolist() == [3, 7]
    kv_batch_get.assert_not_called()


def test_v1_sequence_lengths_recover_only_missing_rows_from_jagged_input_ids():
    batch = _batch_with_tags([{"seq_len": 3}, {"status": "success"}, {"seq_len": 7}])
    input_ids = torch.nested.nested_tensor([torch.arange(5)], layout=torch.jagged)

    with (
        patch(
            "verl.trainer.ppo.v1.trainer_base.tq.kv_batch_get",
            return_value={"input_ids": input_ids},
        ) as kv_batch_get,
        patch("verl.trainer.ppo.v1.trainer_base.logger.warning") as warning,
    ):
        lengths = _resolve_v1_sequence_lengths(batch)

    assert lengths.tolist() == [3, 5, 7]
    assert batch.tags[1]["seq_len"] == 5
    kv_batch_get.assert_called_once_with(
        keys=["row_1"], partition_id="train", select_fields=["input_ids"]
    )
    warning.assert_called_once()


def test_v1_sequence_lengths_recover_invalid_rows_in_key_order():
    batch = _batch_with_tags([{"seq_len": 0}, {"seq_len": "invalid"}])
    input_ids = torch.nested.nested_tensor([torch.arange(4), torch.arange(6)], layout=torch.jagged)

    with patch(
        "verl.trainer.ppo.v1.trainer_base.tq.kv_batch_get",
        return_value={"input_ids": input_ids},
    ):
        lengths = _resolve_v1_sequence_lengths(batch)

    assert lengths.tolist() == [4, 6]
    assert [tag["seq_len"] for tag in batch.tags] == [4, 6]


def test_v1_sequence_lengths_raise_contextual_error_when_recovery_fails():
    batch = _batch_with_tags([{}, {"seq_len": 3}])

    with (
        patch(
            "verl.trainer.ppo.v1.trainer_base.tq.kv_batch_get",
            side_effect=KeyError("input_ids"),
        ),
        pytest.raises(RuntimeError, match=r"could not recover.*row_0"),
    ):
        _resolve_v1_sequence_lengths(batch)


def test_off_policy_steps_sync_fall_back_to_sample_step():
    tags = [
        {"global_steps": 3, "min_global_steps": None, "max_global_steps": None},
        {"is_padding": True},
    ]
    min_steps, max_steps = _get_off_policy_step_arrays(tags, [True, False], "sync")

    assert min_steps.tolist() == [2]
    assert max_steps.tolist() == [2]


def test_off_policy_steps_async_require_rollout_versions():
    tags = [{"global_steps": 3, "min_global_steps": None, "max_global_steps": None}]

    min_steps, max_steps = _get_off_policy_step_arrays(tags, [True], "separate_async")

    assert min_steps is None
    assert max_steps is None


def test_off_policy_steps_preserve_async_rollout_versions():
    tags = [{"global_steps": 3, "min_global_steps": 1, "max_global_steps": 2}]

    min_steps, max_steps = _get_off_policy_step_arrays(tags, [True], "separate_async")

    assert min_steps.tolist() == [1]
    assert max_steps.tolist() == [2]


def _trainer_with_filter_groups(filter_groups: dict, trainer_mode: str = "sync") -> _StubTrainer:
    trainer = _StubTrainer.__new__(_StubTrainer)
    trainer.trainer_mode = trainer_mode
    trainer.config = OmegaConf.create(
        {
            "algorithm": {"filter_groups": filter_groups},
            "data": {"train_batch_size": 64, "gen_batch_size": 8},
            "reward": {"reward_model": {"enable": False, "enable_resource_pool": False}},
            "trainer": {
                "v1": {
                    trainer_mode: {},
                    "sampler": {
                        "custom_sampler": None,
                        "max_off_policy_threshold": 1,
                        "max_off_policy_strategy": "drop",
                        "sampler_kwargs": {},
                    },
                }
            },
        }
    )
    return trainer


def test_builtin_sampler_class_follows_trainer_mode():
    sync_sampler = _trainer_with_filter_groups({"enable": False}, trainer_mode="sync")._build_replay_buffer()
    async_samplers = [
        _trainer_with_filter_groups({"enable": True, "metric": "acc"}, trainer_mode=mode)._build_replay_buffer()
        for mode in ("colocate_async", "separate_async")
    ]

    assert type(sync_sampler) is ReplayBuffer
    assert all(type(sampler) is ReplayBufferAsync for sampler in async_samplers)
    assert all(sampler.filter_groups_metric == "acc" for sampler in async_samplers)
    assert all(sampler.train_batch_size is None for sampler in async_samplers)
    assert all(sampler.gen_batch_size is None for sampler in async_samplers)


def test_custom_sampler_skips_builtin_filter_groups_validation():
    trainer = _trainer_with_filter_groups({"enable": True, "metric": "acc"})
    trainer.config.trainer.v1.sampler.custom_sampler = {"path": "custom.py", "name": "CustomSampler"}

    with (
        patch("verl.trainer.ppo.v1.trainer_base.load_extern_type", return_value=_CustomSampler),
        patch.object(trainer, "_resolve_filter_groups_metric") as resolve_filter_groups_metric,
    ):
        sampler = trainer._build_replay_buffer()

    resolve_filter_groups_metric.assert_not_called()
    assert isinstance(sampler, _CustomSampler)
    assert "filter_groups_metric" not in sampler.kwargs
    assert "train_batch_size" not in sampler.kwargs
    assert "gen_batch_size" not in sampler.kwargs
    assert "max_inflight_gen_batches" not in sampler.kwargs
    assert "sync_refill_failed_groups" not in sampler.kwargs


def test_builtin_filter_groups_uses_default_inflight_limit():
    trainer = _trainer_with_filter_groups({"enable": True, "metric": "acc"})

    sampler = trainer._build_replay_buffer()

    assert sampler.filter_groups_metric == "acc"
    assert sampler.train_batch_size == 64
    assert sampler.gen_batch_size == 1
    assert sampler.max_inflight_gen_batches == 1


def test_builtin_filter_groups_forwards_configured_inflight_limit():
    trainer = _trainer_with_filter_groups({"enable": True, "metric": "acc", "max_inflight_gen_batches": 3})

    sampler = trainer._build_replay_buffer()

    assert sampler.max_inflight_gen_batches == 3


def test_builtin_sync_failure_refill_forces_single_prompt_generation():
    trainer = _trainer_with_filter_groups({"enable": False})
    trainer.config.trainer.v1.sampler.sync_refill_failed_groups = True

    sampler = trainer._build_replay_buffer()

    assert sampler.sync_refill_failed_groups is True
    assert sampler.gen_batch_size == 1


def test_validation_infrastructure_filter_does_not_change_training_generation_batch(monkeypatch):
    monkeypatch.setenv("SWE_AGENT_ROLLOUT_VALIDATION_COMPLETION_RATIO_THRESHOLD", "0.9")
    monkeypatch.setenv("SWE_AGENT_ROLLOUT_VALIDATION_INFRA_FAILURE_GROUP_RATIO_THRESHOLD", "0.25")
    trainer = _trainer_with_filter_groups({"enable": False})

    sampler = trainer._build_replay_buffer()

    assert sampler.gen_batch_size == 8


def test_sync_failure_refill_overrides_dataloader_generation_batch_size():
    trainer = _trainer_with_filter_groups({"enable": False})
    trainer.config.trainer.v1.sampler.sync_refill_failed_groups = True
    trainer.config.data.update(
        {
            "train_files": [],
            "val_files": [],
            "train_max_samples": -1,
            "val_max_samples": -1,
            "dataloader_num_workers": 0,
            "val_batch_size": 1,
            "validation_shuffle": False,
        }
    )
    trainer.config.trainer.total_epochs = 1
    trainer.config.trainer.total_training_steps = None
    trainer.parameter_sync_step = 1
    trainer.tokenizer = None
    trainer.processor = None

    with (
        patch("verl.trainer.ppo.v1.trainer_base.create_rl_dataset", side_effect=[[{}, {}], [{}]]),
        patch("verl.trainer.ppo.v1.trainer_base.create_rl_sampler", return_value=None),
        patch("verl.trainer.ppo.v1.trainer_base.StatefulDataLoader") as dataloader,
        patch("verl.trainer.ppo.v1.trainer_base.logger.warning") as warning,
    ):
        trainer._init_dataloader()

    assert trainer.config.data.gen_batch_size == 1
    assert dataloader.call_args_list[0].kwargs["batch_size"] == 1
    warning.assert_any_call("data.gen_batch_size=8 is overridden to 1.")


def test_builtin_filter_groups_warns_when_total_generation_limit_is_configured():
    trainer = _trainer_with_filter_groups({"enable": True, "metric": "acc", "max_num_gen_batches": 10})

    with patch("verl.trainer.ppo.v1.trainer_base.logger.warning") as warning:
        trainer._build_replay_buffer()

    warning.assert_called_once_with(
        "algorithm.filter_groups.max_num_gen_batches=%s is ignored by the built-in V1 ReplayBuffer; "
        "use max_inflight_gen_batches to bound concurrent Sync DAPO generation.",
        10,
    )

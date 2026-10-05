# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf
from tensordict.tensorclass import NonTensorData

from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync
from verl.trainer.sft_rl_trainer import SFTRLTrainer
from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode, SFTTensorCollator
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset
from verl.workers.utils.losses import sft_loss


def _trainer_for_batch_preparation(*, dp_size: int = 2) -> SFTRLTrainer:
    trainer = SFTRLTrainer.__new__(SFTRLTrainer)
    trainer.config = OmegaConf.create(
        {
            "data": {"train_batch_size": 4, "pad_mode": "no_padding"},
            "trainer": {"balance_batch": False},
            "actor_rollout_ref": {
                "actor": {
                    "ppo_mini_batch_size": 4,
                    "ppo_epochs": 1,
                    "data_loader_seed": 1,
                    "shuffle": False,
                }
            },
        }
    )
    trainer.tokenizer = SimpleNamespace(pad_token_id=0)
    trainer._get_actor_data_parallel_size = lambda: dp_size
    return trainer


def _ragged_batch(lengths: list[int]) -> dict[str, torch.Tensor]:
    examples = []
    for index, length in enumerate(lengths):
        examples.append(
            {
                "input_ids": torch.arange(length),
                "position_ids": torch.arange(length),
                "loss_mask": torch.full((length,), index + 1, dtype=torch.long),
            }
        )
    return SFTTensorCollator(pad_mode=DatasetPadMode.NO_PADDING)(examples)


def test_on_init_end_installs_sft_loss_without_synchronizing_rollout() -> None:
    trainer = SFTRLTrainer.__new__(SFTRLTrainer)
    trainer.actor_rollout_wg = MagicMock()
    trainer.checkpoint_manager = MagicMock()

    trainer.on_init_end()

    loss_fn = trainer.actor_rollout_wg.set_loss_fn.call_args.args[0]
    assert loss_fn.func is sft_loss
    trainer.checkpoint_manager.update_weights.assert_not_called()


@pytest.mark.parametrize("validation_error", [None, RuntimeError("validation failed")])
def test_rl_validation_synchronizes_then_always_sleeps(validation_error: Exception | None) -> None:
    trainer = SFTRLTrainer.__new__(SFTRLTrainer)
    trainer.global_steps = 7
    events = []
    trainer.on_validate_begin = lambda: events.append("begin")
    trainer.on_validate_end = lambda: events.append("end")
    trainer.checkpoint_manager = SimpleNamespace(
        update_weights=lambda step: events.append(("update", step)),
        sleep_replicas=lambda: events.append("sleep"),
    )

    expected = {"val/reward": 1.0}
    validate_kwargs = (
        {"return_value": expected} if validation_error is None else {"side_effect": validation_error}
    )
    with patch.object(PPOTrainerSync, "_validate", **validate_kwargs):
        if validation_error is None:
            assert trainer._run_rl_validation() == expected
        else:
            with pytest.raises(RuntimeError, match="validation failed"):
                trainer._run_rl_validation()

    assert events == ["begin", ("update", 7), "sleep", "end"]


def test_teacher_validation_padding_has_no_target_tokens() -> None:
    trainer = _trainer_for_batch_preparation(dp_size=2)

    batch, sequence_lengths, target_tokens = trainer._prepare_sft_batch(
        _ragged_batch([2, 3, 4]), training=False
    )

    assert sequence_lengths == [2, 3, 4, 2]
    assert [int(mask.sum()) for mask in batch["loss_mask"].unbind()] == [2, 6, 12, 0]
    assert target_tokens == 20
    assert isinstance(batch.get("global_token_num"), NonTensorData)
    assert tu.get(batch, "global_token_num") == sequence_lengths


def test_training_batch_must_be_divisible_by_actor_data_parallel_size() -> None:
    trainer = _trainer_for_batch_preparation(dp_size=2)

    with pytest.raises(ValueError, match="must be divisible"):
        trainer._prepare_sft_batch(_ragged_batch([2, 3, 4]), training=True)


def test_leading_system_and_user_messages_are_rendered_together() -> None:
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "request"},
        {"role": "assistant", "content": "answer"},
    ]
    dataset = MultiTurnSFTDataset.__new__(MultiTurnSFTDataset)
    dataset.dataframe = pd.DataFrame([{"messages": messages}])
    dataset.messages_key = "messages"
    dataset.image_key = "images"
    dataset.video_key = "videos"
    dataset.tools = None
    dataset.enable_thinking = None
    dataset.enable_thinking_default = None
    dataset.processor = None
    dataset.tokenizer = SimpleNamespace(pad_token_id=0)
    dataset.pad_mode = DatasetPadMode.NO_PADDING
    dataset.max_length = 16
    dataset.truncation = "error"
    dataset.sanity_check = MagicMock()

    process_message = MagicMock(
        side_effect=[
            (torch.tensor([10, 11]), torch.tensor([0, 0]), torch.tensor([1, 1]), {}),
            (torch.tensor([12]), torch.tensor([1]), torch.tensor([1]), {}),
        ]
    )
    dataset._process_single_message = process_message

    with patch("verl.utils.dataset.multiturn_sft_dataset.print_assembled_message"):
        example = dataset[0]

    assert process_message.call_args_list[0].kwargs["index"] == 0
    assert process_message.call_args_list[0].kwargs["message"] == messages[:2]
    assert process_message.call_args_list[1].kwargs["index"] == 2
    assert process_message.call_args_list[1].kwargs["message"] == messages[2]
    assert example["loss_mask"].tolist() == [0, 0, 1]

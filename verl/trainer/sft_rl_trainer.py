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

"""Supervised fine-tuning with the V1 RL generation validation stack."""

import logging
import os
import time
from functools import partial
from pprint import pprint

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict
from tensordict import TensorDict
from tensordict.tensorclass import NonTensorData
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl.trainer.ppo.utils import create_rl_dataset
from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync
from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode, SFTTensorCollator
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset
from verl.utils.dataset.rl_dataset import collate_fn as rl_collate_fn
from verl.utils.metric import reduce_metrics
from verl.utils.py_functional import rename_dict
from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions
from verl.utils.skip import SkipManager
from verl.utils.tensordict_utils import nested_tensor_from_tensor_list
from verl.utils.tracking import DapoFilteredRewardTableLogger, Tracking, ValidationGenerationsLogger
from verl.workers.utils.losses import sft_loss

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


class SFTRLTrainer(PPOTrainerSync):
    """Train the actor with SFT while retaining V1 RL generation validation.

    Megatron and vLLM remain colocated. Rollout replicas stay asleep for SFT
    updates and are synchronized only immediately before generation validation.
    """

    def __init__(self, config):
        super().__init__(config=config)
        if self.trainer_mode != "sync":
            raise ValueError("SFTRLTrainer supports only trainer.v1.trainer_mode=sync")
        if self.use_critic:
            raise ValueError("SFTRLTrainer requires critic.enable=False")
        if self.use_reference_policy:
            raise ValueError(
                "SFTRLTrainer does not use a reference policy; disable actor KL loss and reward KL"
            )
        if self.use_teacher_policy:
            raise ValueError("SFTRLTrainer does not support on-policy distillation")

        self.sft_config = config.get("sft", {})
        self._actor_data_parallel_size = None

    def _init_dataloader(self):
        """Build an SFT train loader and a separate RL prompt validation loader."""
        data_config = self.config.data
        pad_mode = data_config.get("pad_mode", DatasetPadMode.NO_PADDING)
        if pad_mode != DatasetPadMode.NO_PADDING:
            raise ValueError("SFTRLTrainer currently requires data.pad_mode=no_padding")

        self.train_dataset = MultiTurnSFTDataset(
            parquet_files=data_config.train_files,
            tokenizer=self.tokenizer,
            config=data_config,
            processor=self.processor,
            max_samples=data_config.get("train_max_samples", -1),
        )
        self.sft_collate_fn = SFTTensorCollator(pad_mode=pad_mode)
        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=data_config.train_batch_size,
            shuffle=data_config.get("shuffle", True),
            num_workers=data_config.get("dataloader_num_workers", 8),
            drop_last=True,
            collate_fn=self.sft_collate_fn,
            pin_memory=False,
        )
        self.train_dataloader_it = None
        self.steps_per_epoch = len(self.train_dataloader)
        if self.steps_per_epoch == 0:
            raise ValueError(
                "The SFT train loader has no complete batch; reduce data.train_batch_size "
                f"below the dataset size ({len(self.train_dataset)})"
            )

        self.val_dataset = create_rl_dataset(
            data_config.val_files,
            data_config,
            self.tokenizer,
            self.processor,
            is_train=False,
            max_samples=data_config.get("val_max_samples", -1),
        )
        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=data_config.val_batch_size or len(self.val_dataset),
            num_workers=data_config.get("dataloader_num_workers", 8),
            shuffle=data_config.get("validation_shuffle", False),
            drop_last=False,
            collate_fn=rl_collate_fn,
        )

        teacher_val_files = self.sft_config.get("teacher_forced_val_files", None)
        if teacher_val_files:
            self.sft_val_dataset = MultiTurnSFTDataset(
                parquet_files=teacher_val_files,
                tokenizer=self.tokenizer,
                config=data_config,
                processor=self.processor,
                max_samples=self.sft_config.get("teacher_forced_val_max_samples", -1),
            )
            teacher_val_batch_size = self.sft_config.get(
                "teacher_forced_val_batch_size", data_config.train_batch_size
            )
            self.sft_val_dataloader = StatefulDataLoader(
                dataset=self.sft_val_dataset,
                batch_size=teacher_val_batch_size,
                shuffle=False,
                num_workers=data_config.get("dataloader_num_workers", 8),
                drop_last=False,
                collate_fn=self.sft_collate_fn,
                pin_memory=False,
            )
        else:
            self.sft_val_dataset = None
            self.sft_val_dataloader = None

        total_training_steps = self.steps_per_epoch * self.config.trainer.total_epochs
        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps
        self.total_training_steps = int(total_training_steps)
        if self.total_training_steps <= 0:
            raise ValueError("trainer.total_training_steps must be positive")

        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = self.total_training_steps

        logger.info(
            "SFT/RL dataloaders initialized: sft_train=%d, rl_val=%d, sft_val=%s, steps_per_epoch=%d, total_steps=%d",
            len(self.train_dataset),
            len(self.val_dataset),
            len(self.sft_val_dataset) if self.sft_val_dataset is not None else "disabled",
            self.steps_per_epoch,
            self.total_training_steps,
        )

    def on_init_end(self):
        """Install SFT loss and deliberately leave rollout replicas asleep."""
        self.actor_rollout_wg.set_loss_fn(partial(sft_loss, config=None))

    def on_step_end(self):
        """SFT updates do not synchronize or wake vLLM."""
        return

    def on_sample_end(self):
        """SFT has no rollout sampling phase."""
        return

    def _get_actor_data_parallel_size(self) -> int:
        if self._actor_data_parallel_size is None:
            rank_mapping = self.actor_rollout_wg._query_dispatch_info("actor")
            self._actor_data_parallel_size = max(rank_mapping) + 1
        return self._actor_data_parallel_size

    @staticmethod
    def _batch_sequence_lengths(data: TensorDict) -> torch.Tensor:
        input_ids = data["input_ids"]
        if input_ids.is_nested:
            return input_ids.offsets().diff()
        attention_mask = data.get("attention_mask", None)
        if attention_mask is None:
            return torch.full((len(data),), input_ids.shape[-1], dtype=torch.long)
        return attention_mask.sum(dim=-1)

    @staticmethod
    def _batch_target_tokens(data: TensorDict) -> int:
        loss_mask = data["loss_mask"]
        values = loss_mask.values() if loss_mask.is_nested else loss_mask
        return int(values.sum().item())

    def _prepare_sft_batch(self, batch_dict: dict, *, training: bool) -> tuple[TensorDict, list[int], int]:
        batch = tu.get_tensordict(tensor_dict=batch_dict)
        sequence_lengths = self._batch_sequence_lengths(batch)
        dp_size = self._get_actor_data_parallel_size()

        if training and len(batch) % dp_size != 0:
            raise ValueError(f"SFT batch size {len(batch)} must be divisible by actor data parallel size {dp_size}")
        if not training and len(batch) % dp_size != 0:
            original_size = len(batch)
            padding_size = dp_size - original_size % dp_size
            indices = list(range(original_size)) + [index % original_size for index in range(padding_size)]
            batch = tu.index_select_tensor_dict(batch, indices)
            sequence_lengths = self._batch_sequence_lengths(batch)

            loss_masks = list(batch["loss_mask"].unbind())
            for index in range(original_size, len(loss_masks)):
                loss_masks[index] = torch.zeros_like(loss_masks[index])
            batch["loss_mask"] = nested_tensor_from_tensor_list(loss_masks)

        if training and self.config.trainer.balance_batch:
            workloads = calculate_workload(sequence_lengths.to(torch.float32))
            partitions = get_seqlen_balanced_partitions(
                workloads,
                k_partitions=dp_size,
                equal_size=True,
            )
            for index, partition in enumerate(partitions):
                partition.sort(key=lambda item: (workloads[item], item))
                partitions[index] = partition[::2] + partition[1::2][::-1]
            indices = torch.tensor([item for partition in partitions for item in partition])
            batch = tu.index_select_tensor_dict(batch, indices)
            sequence_lengths = sequence_lengths[indices]

        actor_config = self.config.actor_rollout_ref.actor
        metadata = {
            "temperature": 1.0,
            "global_batch_size": self.config.data.train_batch_size,
            "mini_batch_size": actor_config.ppo_mini_batch_size,
            "epochs": actor_config.ppo_epochs,
            "seed": actor_config.data_loader_seed,
            "dataloader_kwargs": {"shuffle": actor_config.shuffle},
            "pad_mode": self.config.data.pad_mode,
            "pad_token_id": self.tokenizer.pad_token_id,
            "global_token_num": NonTensorData(sequence_lengths.tolist()),
        }
        tu.assign_non_tensor(batch, **metadata)
        return batch, sequence_lengths.tolist(), self._batch_target_tokens(batch)

    @staticmethod
    def _reduce_worker_metrics(output: TensorDict, prefix: str) -> dict[str, float]:
        raw_metrics = tu.get(output, "metrics")
        metrics = reduce_metrics(dict(raw_metrics))
        return rename_dict(metrics, prefix)

    def _sft_update(self, batch_dict: dict) -> dict[str, float]:
        batch, sequence_lengths, target_tokens = self._prepare_sft_batch(batch_dict, training=True)
        start = time.perf_counter()
        output = self.actor_rollout_wg.update_actor(batch)
        duration = time.perf_counter() - start
        metrics = self._reduce_worker_metrics(output, "train/")

        input_tokens = int(sum(sequence_lengths))
        metrics.update(
            {
                "train/input_tokens": input_tokens,
                "train/target_tokens": target_tokens,
                "train/sequence_length/mean": float(np.mean(sequence_lengths)),
                "train/sequence_length/max": int(max(sequence_lengths)),
                "train/sequence_length/min": int(min(sequence_lengths)),
                "timing_s/sft_update": duration,
                "perf/sft_input_tokens_per_second_per_gpu": input_tokens
                / max(duration * self._get_n_gpus_for_throughput(), 1.0),
            }
        )
        return metrics

    def _run_teacher_forced_validation(self) -> dict[str, float]:
        if self.sft_val_dataloader is None:
            return {}

        weighted_loss = 0.0
        total_target_tokens = 0
        total_examples = 0
        for batch_dict in self.sft_val_dataloader:
            batch, _sequence_lengths, target_tokens = self._prepare_sft_batch(batch_dict, training=False)
            output = self.actor_rollout_wg.compute_log_prob(batch)
            metrics = self._reduce_worker_metrics(output, "")
            weighted_loss += float(metrics["loss"]) * target_tokens
            total_target_tokens += target_tokens
            total_examples += len(batch_dict["input_ids"])

        if total_target_tokens == 0:
            raise ValueError("Teacher-forced SFT validation contains no assistant target tokens")
        return {
            "sft-val/loss": weighted_loss / total_target_tokens,
            "sft-val/target_tokens": total_target_tokens,
            "sft-val/examples": total_examples,
        }

    def _run_rl_validation(self) -> dict[str, float]:
        """Wake/synchronize vLLM for validation and always put it back to sleep."""
        self.on_validate_begin()
        try:
            self.checkpoint_manager.update_weights(self.global_steps)
            return super()._validate()
        finally:
            try:
                self.checkpoint_manager.sleep_replicas()
            finally:
                self.on_validate_end()

    def _should_run_teacher_validation(self, *, is_last_step: bool) -> bool:
        frequency = int(self.sft_config.get("teacher_forced_test_freq", -1))
        return self.sft_val_dataloader is not None and (
            is_last_step or (frequency > 0 and self.global_steps % frequency == 0)
        )

    def fit(self, agent_loop_manager):
        """Run SFT updates and periodically invoke inherited RL validation."""
        self.agent_loop_manager = agent_loop_manager
        self.replay_buffer.cancel_fn = agent_loop_manager.cancel_sequences
        SkipManager.init(self.config)
        SkipManager.set_step(self.global_steps)
        self.logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )
        self.dapo_filtered_reward_logger = DapoFilteredRewardTableLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        try:
            initial_metrics = {}
            if self.sft_config.get("teacher_forced_val_before_train", False):
                initial_metrics.update(self._run_teacher_forced_validation())

            if self.config.trainer.get("val_before_train", True):
                val_metrics = self._run_rl_validation()
                if not val_metrics:
                    raise RuntimeError("RL generation validation returned no metrics")
                pprint(f"Initial validation metrics: {val_metrics}")
                initial_metrics.update(val_metrics)
                self.logger.log(data=initial_metrics, step=self.global_steps, commit=True)
                if self.config.trainer.get("val_only", False):
                    return
            elif initial_metrics:
                self.logger.log(data=initial_metrics, step=self.global_steps, commit=True)

            progress = tqdm(
                total=self.total_training_steps,
                initial=self.global_steps,
                desc="SFT Training Progress",
            )
            start_epoch = self.global_steps // self.steps_per_epoch
            last_val_metrics = None
            self.on_train_begin()

            for epoch in range(start_epoch, self.config.trainer.total_epochs):
                for batch_dict in self.train_dataloader:
                    if self.global_steps >= self.total_training_steps:
                        break

                    self.global_steps += 1
                    SkipManager.set_step(self.global_steps)
                    is_last_step = self.global_steps >= self.total_training_steps
                    metrics = self._sft_update(batch_dict)
                    metrics.update(
                        {
                            "training/global_step": self.global_steps,
                            "training/epoch": epoch,
                        }
                    )

                    if self.config.trainer.save_freq > 0 and (
                        is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                    ):
                        self._save_checkpoint()

                    if self._should_run_teacher_validation(is_last_step=is_last_step):
                        metrics.update(self._run_teacher_forced_validation())

                    if self.config.trainer.test_freq > 0 and (
                        is_last_step or self.global_steps % self.config.trainer.test_freq == 0
                    ):
                        last_val_metrics = self._run_rl_validation()
                        metrics.update(last_val_metrics)

                    self.logger.log(data=metrics, step=self.global_steps, commit=True)
                    progress.update(1)
                    if is_last_step:
                        pprint(f"Final validation metrics: {last_val_metrics}")
                        progress.close()
                        return

                if self.global_steps >= self.total_training_steps:
                    break

            self.on_train_end()
            progress.close()
        finally:
            # Initialization and every update leave replicas asleep. This also
            # restores that invariant after an interrupted validation.
            try:
                self.checkpoint_manager.sleep_replicas()
            finally:
                self._shutdown_dump_executor()

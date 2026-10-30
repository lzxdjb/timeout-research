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
"""Cutoff-induced synthetic queue rows must remain inert in the real OPD loss."""

from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict
from transfer_queue import KVBatchMeta
from transfer_queue.storage.managers.simple_storage_manager import AsyncSimpleStorageManager

from verl.trainer.distillation.batch_utils import prepare_distillation_minibatch
from verl.trainer.distillation.losses import distillation_ppo_loss
from verl.trainer.ppo import padding_utils
from verl.utils import tensordict_utils as tu
from verl.workers.config import ActorConfig, DistillationLossConfig


def _pack_rows(rows):
    # Use the installed queue's actual tensor packing, including teacher targets
    # whose ragged dimension differs from their top-k dimension.
    return TensorDict(
        {key: AsyncSimpleStorageManager._pack_field_values([row[key] for row in rows]) for key in rows[0]},
        batch_size=[len(rows)],
    )


def _source_row(prompt_len, response_mask):
    response_mask = torch.tensor(response_mask, dtype=torch.long)
    seq_len = prompt_len + response_mask.numel()
    input_ids = torch.arange(seq_len, dtype=torch.long) + 10
    return {
        "input_ids": input_ids,
        "prompts": input_ids[:prompt_len],
        "responses": input_ids[prompt_len:],
        "attention_mask": torch.ones(seq_len, dtype=torch.long),
        "position_ids": torch.arange(seq_len, dtype=torch.long),
        "response_mask": response_mask,
        "loss_mask": response_mask.clone(),
        "rm_scores": torch.zeros_like(response_mask, dtype=torch.float32),
        "rollout_log_probs": torch.full_like(response_mask, -2.0, dtype=torch.float32),
        "teacher_logprobs": torch.full((seq_len, 1), -1.0),
        "teacher_ids": input_ids.unsqueeze(-1),
    }


def _padded_queue_batch(monkeypatch, *, min_seq_len=8, real_count=2, multiple=4):
    originals = [_source_row(3, [1, 0, 1]), _source_row(2, [1, 1])]
    rows = {f"real_{index}": originals[index % 2] for index in range(real_count)}
    tags = [{"seq_len": rows[key]["input_ids"].numel(), "train_sample_mask": True, "is_padding": False} for key in rows]
    batch = KVBatchMeta(keys=list(rows), tags=tags, partition_id="train")

    def queue_get(*, keys, partition_id):
        assert partition_id == "train"
        return _pack_rows([rows[key] for key in keys])

    def queue_put(*, keys, partition_id, fields, tags):
        assert partition_id == "train"
        assert len(keys) == len(tags) == len(fields)
        for index, key in enumerate(keys):
            rows[key] = dict(fields[index].items())

    monkeypatch.setattr(padding_utils.tq, "kv_batch_get", queue_get)
    monkeypatch.setattr(padding_utils.tq, "kv_batch_put", queue_put)
    padded = padding_utils.upsample_batch_to_divisible_size(
        batch, batch_multiple=multiple, eos_token_id=2, min_seq_len=min_seq_len
    )
    data = queue_get(keys=padded.keys, partition_id=padded.partition_id)
    data["train_sample_mask"] = torch.tensor([tag.get("train_sample_mask", True) for tag in padded.tags])
    return data, padded


def _prepare_loss_batch(data):
    # These response fields are computed after queue upsampling by the trainer.
    response_lengths = data["responses"].offsets().diff().tolist()
    keep = data["train_sample_mask"].tolist()
    for key, make_row in {
        "old_log_probs": lambda index, length: torch.full((length,), -2.0),
        "advantages": lambda index, length: torch.full((length,), 0.5 if index % 2 == 0 else -0.5),
        "tool_failure_weights": lambda index, length: torch.tensor(
            [0.02 if keep[index] else 9.0] + [0.0] * (length - 1)
        ),
    }.items():
        data[key] = torch.nested.as_nested_tensor(
            [make_row(index, length) for index, length in enumerate(response_lengths)], layout=torch.jagged
        )
    prepare_distillation_minibatch(data)
    tu.assign_non_tensor(
        data,
        dp_size=1,
        batch_num_tokens=int(data["loss_mask"].values().sum().item()),
        tool_failure_active_global_count=int(data["train_sample_mask"].sum().item()),
    )
    return data


def _loss(data, *, hybrid):
    actor = ActorConfig(strategy="fsdp2", rollout_n=2, use_dynamic_bsz=True, tool_failure_penalty_enabled=True)
    teacher = SimpleNamespace(
        distillation_loss=DistillationLossConfig(
            loss_mode="k1",
            use_policy_gradient=True,
            use_task_rewards=hybrid,
            distillation_loss_coef=0.7,
            loss_max_clamp=None,
        )
    )
    student = torch.full((data["input_ids"].values().numel(),), -2.0, requires_grad=True)
    loss, metrics = distillation_ppo_loss(actor, teacher, {"log_probs": student}, data)
    loss.backward()
    return loss, metrics, student.grad


@pytest.mark.parametrize("hybrid", [False, True])
@pytest.mark.parametrize("min_seq_len", [2, 8])
def test_cutoff_padding_preserves_real_opd_loss_and_gradients(monkeypatch, hybrid, min_seq_len):
    data, metadata = _padded_queue_batch(monkeypatch, min_seq_len=min_seq_len)
    assert len(metadata) == 4
    assert sum(tag["is_padding"] for tag in metadata.tags) == 2
    sequence_lengths = data["input_ids"].offsets().diff()
    for field in ("teacher_logprobs", "teacher_ids"):
        assert torch.equal(data[field].offsets().diff(), sequence_lengths)
        assert data[field].unbind()[2].eq(0).all()

    data = _prepare_loss_batch(data)
    # Exercise the actual nested minibatch selector after a batch-size cutoff.
    real = tu.index_select_tensor_dict(data, [0, 1])
    loss, metrics, gradients = _loss(data, hybrid=hybrid)
    reference_loss, reference_metrics, reference_gradients = _loss(real, hybrid=hybrid)
    real_tokens = real["input_ids"].values().numel()
    torch.testing.assert_close(loss, reference_loss)
    torch.testing.assert_close(gradients[:real_tokens], reference_gradients)
    assert gradients[:real_tokens].count_nonzero() > 0
    assert gradients[real_tokens:].eq(0).all()
    # Two accepted rows, each with one 0.02 tool cost: no synthetic contribution.
    assert metrics["actor/tool_failure_loss"].aggregate() == pytest.approx(0.02)
    assert metrics["distillation/loss"].aggregate() == reference_metrics["distillation/loss"].aggregate()
    assert torch.isfinite(loss)

    # Engine microbatching can place only synthetic rows on a worker.
    padding_only = tu.index_select_tensor_dict(data, [3, 2])
    padding_loss, padding_metrics, padding_gradients = _loss(padding_only, hybrid=hybrid)
    assert padding_loss.item() == 0.0
    assert padding_gradients.eq(0).all()
    for metric in padding_metrics.values():
        assert torch.isfinite(torch.as_tensor(metric.aggregate() if hasattr(metric, "aggregate") else metric))


def test_observed_502_to_512_cutoff_aligns_every_teacher_target(monkeypatch):
    data, metadata = _padded_queue_batch(monkeypatch, real_count=502, multiple=512)
    assert len(metadata) == 512
    assert sum(tag["is_padding"] for tag in metadata.tags) == 10
    seq_lengths = data["prompts"].offsets().diff() + data["responses"].offsets().diff()
    for field in ("teacher_logprobs", "teacher_ids"):
        assert torch.equal(data[field].offsets().diff(), seq_lengths)


@pytest.mark.parametrize("hybrid", [False, True])
def test_malformed_real_teacher_target_still_fails_alignment_assert(monkeypatch, hybrid):
    data, _ = _padded_queue_batch(monkeypatch)
    data = _prepare_loss_batch(data)
    targets = list(data["teacher_logprobs"].unbind())
    targets[0] = targets[0][:-1]
    data["teacher_logprobs"] = torch.nested.as_nested_tensor(targets, layout=torch.jagged)
    with pytest.raises(AssertionError):
        _loss(data, hybrid=hybrid)

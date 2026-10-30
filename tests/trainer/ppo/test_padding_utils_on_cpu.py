from unittest.mock import patch

import pytest
import torch
from tensordict import TensorDict
from transfer_queue import KVBatchMeta

from verl.trainer.ppo.padding_utils import construct_minimal_padding_template, repair_v1_incomplete_rows
from verl.workers.utils.padding import no_padding_2_padding


def test_padding_template_uses_context_parallel_safe_sequence_length() -> None:
    source = {
        "position_ids": torch.tensor([0, 1]),
        "input_ids": torch.tensor([10, 11]),
        "response_mask": torch.ones(1, dtype=torch.long),
    }
    tag = {"status": "success", "prompt_len": 1, "response_len": 1, "seq_len": 2}

    sample, padding_tag = construct_minimal_padding_template(
        source,
        tag,
        eos_token_id=2,
        min_seq_len=8,  # TP=2, CP=4
    )

    assert sample["input_ids"].shape == (8,)
    assert sample["attention_mask"].eq(1).all()
    assert sample["response_mask"].eq(0).all()
    assert sample["loss_mask"].eq(0).all()
    assert sample["prompts"].shape == (7,)
    assert sample["responses"].shape == (1,)
    assert padding_tag["is_padding"] is True
    assert padding_tag["train_sample_mask"] is False
    assert padding_tag["prompt_len"] == 7
    assert padding_tag["response_len"] == 1
    assert padding_tag["seq_len"] == 8


def test_padding_template_default_remains_two_tokens() -> None:
    sample, padding_tag = construct_minimal_padding_template(
        {"position_ids": torch.tensor([0, 1])},
        {"status": "success"},
        eos_token_id=2,
    )

    assert sample["input_ids"].shape == (2,)
    assert padding_tag["seq_len"] == 2
    assert "teacher_logprobs" not in sample
    assert "teacher_ids" not in sample


@pytest.mark.parametrize("min_seq_len", [2, 8])
@pytest.mark.parametrize("topk", [1, 4])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_padding_template_rebuilds_teacher_targets(min_seq_len, topk, dtype) -> None:
    source = {
        "input_ids": torch.arange(19),
        "teacher_logprobs": torch.full((19, topk), -2.0, dtype=dtype),
        "teacher_ids": torch.arange(19 * topk, dtype=torch.int32).reshape(19, topk),
    }
    original = {key: value.clone() for key, value in source.items()}
    tag = {"status": "success", "seq_len": 19}
    sample, padding_tag = construct_minimal_padding_template(source, tag, eos_token_id=2, min_seq_len=min_seq_len)

    for key in ("teacher_logprobs", "teacher_ids"):
        assert sample[key].shape == (min_seq_len, topk)
        assert sample[key].dtype == source[key].dtype
        assert sample[key].device == source[key].device
        assert sample[key].eq(0).all()
        torch.testing.assert_close(source[key], original[key])
    torch.testing.assert_close(source["input_ids"], original["input_ids"])
    assert tag == {"status": "success", "seq_len": 19}
    assert padding_tag["train_sample_mask"] is False
    assert sample["loss_mask"].eq(0).all()

    # Exercise the exact conversion that asserted on full-length copied targets
    # in the first actor update after completion-ratio cutoff.
    data = TensorDict(
        {
            key: torch.nested.as_nested_tensor([sample[key]], layout=torch.jagged)
            for key in ("prompts", "responses", "teacher_logprobs", "teacher_ids")
        },
        batch_size=[1],
    )
    assert no_padding_2_padding(data["teacher_logprobs"], data).shape == (1, 1, topk)
    assert no_padding_2_padding(data["teacher_ids"], data).shape == (1, 1, topk)


@pytest.mark.parametrize("key", ["teacher_logprobs", "teacher_ids"])
def test_padding_template_preserves_independently_optional_teacher_fields(key) -> None:
    source = {key: torch.ones(19, 1)}
    sample, _ = construct_minimal_padding_template(source, {}, eos_token_id=2, min_seq_len=8)
    assert sample[key].shape == (8, 1)
    assert sample[key].eq(0).all()
    assert ("teacher_ids" if key == "teacher_logprobs" else "teacher_logprobs") not in sample


def test_repair_v1_incomplete_rows_recovers_seq_len_from_ready_input_ids() -> None:
    batch = KVBatchMeta(
        partition_id="train",
        keys=["uid_0_0"],
        tags=[{"status": "success"}],
    )

    with patch(
        "verl.trainer.ppo.padding_utils.tq.kv_batch_get",
        return_value={"input_ids": torch.arange(5)},
    ) as kv_batch_get:
        repaired, count = repair_v1_incomplete_rows(batch, eos_token_id=2)

    assert repaired is batch
    assert count == 0
    assert batch.tags[0]["seq_len"] == 5
    kv_batch_get.assert_called_once_with(keys=["uid_0_0"], partition_id="train", select_fields=["input_ids"])


@pytest.mark.parametrize("with_teacher", [False, True])
def test_repair_v1_incomplete_rows_replaces_unmaterialized_row(with_teacher) -> None:
    batch = KVBatchMeta(
        partition_id="train",
        keys=["uid_0_0", "uid_1_0"],
        tags=[{"status": "success", "seq_len": 3}, {"status": "running"}],
    )
    source = {"position_ids": torch.tensor([0, 1]), "input_ids": torch.tensor([10, 11])}
    if with_teacher:
        source.update(teacher_logprobs=torch.full((2, 1), -2.0), teacher_ids=torch.ones(2, 1, dtype=torch.int32))

    def get_row(*, keys, partition_id, select_fields=None):
        if select_fields is not None:
            raise ValueError("fields are not ready")
        return [source]

    with (
        patch("verl.trainer.ppo.padding_utils.tq.kv_batch_get", side_effect=get_row),
        patch("verl.trainer.ppo.padding_utils.tq.kv_clear") as kv_clear,
        patch("verl.trainer.ppo.padding_utils.tq.kv_batch_put") as kv_batch_put,
    ):
        repaired, count = repair_v1_incomplete_rows(batch, eos_token_id=2, min_seq_len=8)

    assert count == 1
    assert len(repaired) == 2
    assert repaired.keys[0] == "uid_0_0"
    assert repaired.tags[1]["is_padding"] is True
    assert repaired.tags[1]["train_sample_mask"] is False
    assert repaired.tags[1]["fill_reason"] == "incomplete_queue_row"
    kv_clear.assert_called_once_with(keys=["uid_1_0"], partition_id="train")
    kv_batch_put.assert_called_once()
    if with_teacher:
        fields = kv_batch_put.call_args.kwargs["fields"]
        for key in ("teacher_logprobs", "teacher_ids"):
            assert fields[key][0].shape == (8, 1)
            assert fields[key][0].eq(0).all()

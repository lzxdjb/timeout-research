from unittest.mock import patch

import torch
from transfer_queue import KVBatchMeta

from verl.trainer.ppo.padding_utils import construct_minimal_padding_template, repair_v1_incomplete_rows


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
    kv_batch_get.assert_called_once_with(
        keys=["uid_0_0"], partition_id="train", select_fields=["input_ids"]
    )


def test_repair_v1_incomplete_rows_replaces_unmaterialized_row() -> None:
    batch = KVBatchMeta(
        partition_id="train",
        keys=["uid_0_0", "uid_1_0"],
        tags=[{"status": "success", "seq_len": 3}, {"status": "running"}],
    )
    source = {"position_ids": torch.tensor([0, 1]), "input_ids": torch.tensor([10, 11])}

    def get_row(*, keys, partition_id, select_fields=None):
        if select_fields is not None:
            raise ValueError("fields are not ready")
        return [source]

    with (
        patch("verl.trainer.ppo.padding_utils.tq.kv_batch_get", side_effect=get_row),
        patch("verl.trainer.ppo.padding_utils.tq.kv_clear") as kv_clear,
        patch("verl.trainer.ppo.padding_utils.tq.kv_batch_put") as kv_batch_put,
    ):
        repaired, count = repair_v1_incomplete_rows(batch, eos_token_id=2)

    assert count == 1
    assert len(repaired) == 2
    assert repaired.keys[0] == "uid_0_0"
    assert repaired.tags[1]["is_padding"] is True
    assert repaired.tags[1]["train_sample_mask"] is False
    assert repaired.tags[1]["fill_reason"] == "incomplete_queue_row"
    kv_clear.assert_called_once_with(keys=["uid_1_0"], partition_id="train")
    kv_batch_put.assert_called_once()

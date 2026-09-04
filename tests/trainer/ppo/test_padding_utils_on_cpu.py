import torch

from verl.trainer.ppo.padding_utils import construct_minimal_padding_template


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

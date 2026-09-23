"""Preserve partial-reward diagnostic alignment in sorted rollout JSONL."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from verl.trainer.ppo.v1 import trainer_base
from verl.trainer.ppo.v1.trainer_base import PPOTrainer


@pytest.mark.parametrize("metadata_case", ["present", "missing", "key_error", "type_error", "value_error"])
def test_rollout_dump_keeps_counts_with_matching_trajectory(monkeypatch, tmp_path, metadata_case):
    with_metadata = metadata_case == "present"
    keys = ["b_0_0", "a_0_0", "c_0_0"]
    data = {
        "uid": np.array(keys),
        "prompts": MagicMock(),
        "responses": MagicMock(),
        "rm_scores": torch.tensor([[0.8], [0.0], [1.0]]),
        "reward_model": np.array([{"ground_truth": key} for key in keys], dtype=object),
    }
    data["prompts"].to_padded_tensor.return_value = keys
    data["responses"].to_padded_tensor.return_value = keys
    if with_metadata:
        data["extra_fields"] = np.array(
            [
                {
                    "reward_extra_info": {
                        "raw_score": 0,
                        "partial_hidden_reward_passed_count": 8,
                        "partial_hidden_reward_total_count": 10,
                        "partial_hidden_reward_reason": "partial_hidden_reward",
                    }
                },
                SimpleNamespace(
                    data={
                        "reward_extra_info": {
                            "raw_score": 0,
                            "partial_hidden_reward_passed_count": -1,
                            "partial_hidden_reward_total_count": -1,
                            "partial_hidden_reward_reason": "unsupported_contract",
                        }
                    }
                ),
                {},
            ],
            dtype=object,
        )

    def get(**kwargs):
        if kwargs["select_fields"] == ["extra_fields"]:
            error_type = {"key_error": KeyError, "type_error": TypeError, "value_error": ValueError}.get(metadata_case)
            if error_type is not None:
                raise error_type("Optional field is unavailable")
            return {"extra_fields": data["extra_fields"]} if with_metadata else {}
        assert "extra_fields" not in kwargs["select_fields"]
        return data

    monkeypatch.setattr(trainer_base.tq, "kv_batch_get", get)
    trainer = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0, decode=lambda ids, **kw: ids))

    def dump(**kwargs):
        PPOTrainer._write_generations(**kwargs, global_steps=7)

    trainer._dump_generations = dump
    PPOTrainer._log_rollout_data(trainer, SimpleNamespace(keys=keys, partition_id="train"), {}, str(tmp_path))
    rows = [json.loads(line) for line in (tmp_path / "7.jsonl").read_text().splitlines()]
    assert [row["uid"] for row in rows] == ["a_0_0", "b_0_0", "c_0_0"]
    assert rows[1]["score"] == pytest.approx(0.8)
    assert rows[1]["output"] == "b_0_0"
    if with_metadata:
        assert rows[0]["partial_hidden_reward_reason"] == "unsupported_contract"
        assert rows[1]["partial_hidden_reward_passed_count"] == 8
        assert rows[1]["partial_hidden_reward_total_count"] == 10
        assert rows[2]["partial_hidden_reward_passed_count"] is None
    else:
        assert "partial_hidden_reward_passed_count" not in rows[0]

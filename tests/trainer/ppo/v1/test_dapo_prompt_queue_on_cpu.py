"""Contracts for carrying original prompts across DAPO policy updates."""

import pytest
import torch

from verl.trainer.ppo.v1.dapo_prompt_queue import DAPOPromptQueue
from verl.utils import tensordict_utils as tu


def prompt_batch(uids):
    return tu.get_tensordict({
        "uid": uids,
        "input_ids": torch.tensor([[1, 2] for _ in uids]),
        "raw_prompt": [[{"role": "user", "content": uid}] for uid in uids],
        "extra_info": [{"task_id": uid, "nested": {"value": 1}} for uid in uids],
    })


def test_queue_copies_original_metadata_and_preserves_submission_order():
    queue = DAPOPromptQueue()
    batch = prompt_batch(["a", "b", "c", "d"])
    queue.register(batch)
    batch["input_ids"][0, 0] = 99
    batch["extra_info"][0]["nested"]["value"] = 99
    assert queue.requeue({"c", "a"}) == 2
    queue.release(["b", "d", "a", "c"])
    rows = queue.take(3)
    assert [row["extra_info"][0]["task_id"] for row in rows] == ["a", "c"]
    assert rows[0]["input_ids"].tolist() == [[1, 2]]
    assert rows[0]["extra_info"][0]["nested"]["value"] == 1
    assert len({str(row["uid"][0]) for row in rows} - {"a", "c"}) == 2
    assert not queue.active and len(queue) == 0
    with pytest.raises(RuntimeError, match="Missing original"):
        queue.requeue(["a"])


def test_queue_checkpoint_roundtrip_and_old_checkpoint(tmp_path):
    queue = DAPOPromptQueue()
    original = prompt_batch(["a", "b"])
    tu.assign_non_tensor_data(original, "global_steps", 3)
    queue.register(original)
    with pytest.raises(RuntimeError, match="active"):
        queue.save(str(tmp_path / "queue.pt"))
    queue.requeue({"b", "a"})
    queue.save(str(tmp_path / "queue.pt"))
    restored = DAPOPromptQueue()
    restored.load(str(tmp_path / "queue.pt"))
    assert [row["extra_info"][0]["task_id"] for row in restored.take(2)] == ["a", "b"]
    restored.load(str(tmp_path / "old_checkpoint_without_queue.pt"))
    assert len(restored) == 0 and not restored.active


def test_trainer_consumes_queued_prompts_before_dataset_and_stamps_current_step():
    from test_trainer_base_on_cpu import _trainer_with_filter_groups

    trainer = _trainer_with_filter_groups({
        "enable": True, "metric": "score", "cutoff_when_full": True, "requeue_unselected_prompts": True,
    })
    trainer._build_replay_buffer()
    trainer.config.data.gen_batch_size = 1
    trainer.global_steps = 7
    queue = trainer._dapo_prompt_queue
    original = prompt_batch(["a", "b"])
    tu.assign_non_tensor_data(original, "global_steps", 3)
    queue.register(original)
    queue.requeue({"b", "a"})
    fetched = []

    def fetch():
        fetched.append(True)
        return prompt_batch(["new"])

    trainer._fetch_one_gen_batch = fetch
    batch = trainer._next_train_batch(3)
    assert len(fetched) == 1
    assert [info["task_id"] for info in batch["extra_info"]] == ["a", "b", "new"]
    assert set(batch["uid"]) & {"a", "b"} == set()
    assert tu.get_non_tensor_data(batch, "global_steps", None) == 7

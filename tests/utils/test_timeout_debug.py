import json

import torch
from tensordict import TensorDict

from verl.utils import timeout_debug


def test_timeout_debug_is_disabled_by_default(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("SWE_AGENT_TIMEOUT_PREDICTION_DEBUG", raising=False)
    monkeypatch.setenv("SWE_AGENT_TIMEOUT_PREDICTION_DEBUG_DIR", str(tmp_path))

    timeout_debug.record("disabled", TensorDict({"response_mask": torch.ones(1, 1)}, batch_size=1))

    assert list(tmp_path.iterdir()) == []


def test_timeout_debug_writes_nested_lengths_and_row_tags(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("SWE_AGENT_TIMEOUT_PREDICTION_DEBUG", "1")
    monkeypatch.setenv("SWE_AGENT_TIMEOUT_PREDICTION_DEBUG_DIR", str(tmp_path))
    monkeypatch.setenv("SWE_AGENT_TIMEOUT_PREDICTION_DEBUG_MAX_ROWS", "1")
    input_ids = torch.nested.as_nested_tensor(
        [torch.tensor([1, 2]), torch.tensor([3])],
        layout=torch.jagged,
    )
    data = TensorDict(
        {
            "input_ids": input_ids,
            "attention_mask": torch.tensor([[1, 1], [1, 0]]),
            "response_mask": torch.tensor([[1, 1], [0, 0]]),
        },
        batch_size=2,
    )

    timeout_debug.record(
        "unit_test",
        data,
        tags=[{"is_padding": False}, {"is_padding": True}],
        row_keys=["trajectory-0", "trajectory-1"],
    )

    files = list(tmp_path.glob("*_unit_test.json"))
    assert len(files) == 1
    payload = json.loads(files[0].read_text())
    assert payload["batch_size"] == 2
    assert payload["tensor_meta"]["input_ids"]["offsets"] == [0, 2, 3]
    assert payload["rows"] == [
        {
            "attention_mask": 1,
            "index": 1,
            "input_ids": 1,
            "key": "trajectory-1",
            "response_mask": 0,
            "tag": {"is_padding": True},
        }
    ]


def test_timeout_debug_never_raises(monkeypatch) -> None:
    monkeypatch.setenv("SWE_AGENT_TIMEOUT_PREDICTION_DEBUG", "1")

    class BrokenBatch:
        def keys(self):
            raise RuntimeError("diagnostic failure")

    timeout_debug.record("broken", BrokenBatch())

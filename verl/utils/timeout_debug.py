"""Opt-in diagnostics for timeout/completion-ratio rollout batches.

The helpers in this module deliberately inspect shapes, masks, and stable row
metadata only.  They are inert unless ``SWE_AGENT_TIMEOUT_PREDICTION_DEBUG``
is enabled, so normal training has no additional TransferQueue reads or file
writes.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import time
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


def enabled() -> bool:
    value = os.getenv("SWE_AGENT_TIMEOUT_PREDICTION_DEBUG", "")
    return value.lower() in {"1", "true", "yes", "on"}


def _max_rows() -> int:
    try:
        return max(1, int(os.getenv("SWE_AGENT_TIMEOUT_PREDICTION_DEBUG_MAX_ROWS", "32")))
    except ValueError:
        return 32


def _rank() -> int | None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_rank())
    return None


def _json_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, torch.Tensor) and value.numel() == 1:
        return value.detach().cpu().item()
    return str(value)


def _row_values(value: Any, row_count: int) -> list[Any] | None:
    """Return compact per-row values for masks/lengths where possible."""
    if not isinstance(value, torch.Tensor):
        return None
    if value.is_nested:
        try:
            return [int(v) for v in value.offsets().diff().detach().cpu().tolist()]
        except (RuntimeError, IndexError):
            return None
    if value.ndim == 0 or value.shape[0] != row_count:
        return None
    if value.ndim == 1:
        return [_json_value(v) for v in value.detach().cpu().tolist()]
    try:
        return [int(v) for v in value.detach().to(torch.bool).sum(dim=tuple(range(1, value.ndim))).cpu().tolist()]
    except (RuntimeError, TypeError):
        return None


def summarize_batch(
    data: Any,
    *,
    tags: list[dict] | None = None,
    row_keys: list[str] | None = None,
    extra: dict | None = None,
) -> dict:
    """Build a JSON-safe summary of a TensorDict-like batch."""
    summary: dict[str, Any] = {
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "rank": _rank(),
        "timestamp_unix": time.time(),
    }
    if extra:
        summary["extra"] = {str(k): _json_value(v) for k, v in extra.items()}

    keys = list(data.keys()) if hasattr(data, "keys") else []
    summary["fields"] = keys
    row_count = 0
    for candidate in ("input_ids", "attention_mask", "response_mask", "loss_mask", "rm_scores"):
        if candidate not in keys:
            continue
        value = data[candidate]
        try:
            row_count = len(value)
        except TypeError:
            continue
        break
    summary["batch_size"] = row_count

    tensor_meta: dict[str, Any] = {}
    for key in keys:
        value = data[key]
        if not isinstance(value, torch.Tensor):
            continue
        item: dict[str, Any] = {"dtype": str(value.dtype), "device": str(value.device), "ndim": value.ndim}
        if value.is_nested:
            try:
                item["offsets"] = value.offsets().detach().cpu().tolist()
            except (RuntimeError, IndexError):
                item["offsets"] = "unavailable"
        else:
            item["shape"] = list(value.shape)
        tensor_meta[key] = item
    summary["tensor_meta"] = tensor_meta

    row_value_keys = {
        "input_ids",
        "attention_mask",
        "response_mask",
        "loss_mask",
        "train_sample_mask",
    }
    row_fields = {key: _row_values(data[key], row_count) for key in keys if key in row_value_keys}
    padding_rows = []
    empty_mask_rows = []
    cutoff_rows = []
    ordinary_rows = []
    for index in range(row_count):
        tag = tags[index] if tags is not None and index < len(tags) else {}
        has_empty_mask = any(
            values is not None and index < len(values) and values[index] == 0
            for key, values in row_fields.items()
            if key in {"attention_mask", "response_mask", "loss_mask"}
        )
        if tag.get("is_padding", False):
            padding_rows.append(index)
        elif has_empty_mask:
            empty_mask_rows.append(index)
        elif tag.get("completion_ratio_cutoff", False):
            cutoff_rows.append(index)
        else:
            ordinary_rows.append(index)

    selected_rows = (padding_rows + empty_mask_rows + cutoff_rows + ordinary_rows)[: _max_rows()]
    rows = []
    for index in selected_rows:
        row = {"index": index}
        if row_keys is not None and index < len(row_keys):
            row["key"] = str(row_keys[index])
        if tags is not None and index < len(tags):
            tag = tags[index]
            row["tag"] = {
                key: _json_value(tag.get(key))
                for key in ("status", "is_padding", "completion_ratio_cutoff", "prompt_len", "response_len", "seq_len")
                if key in tag
            }
        for key, values in row_fields.items():
            if values is not None and index < len(values):
                row[key] = _json_value(values[index])
        rows.append(row)
    summary["rows"] = rows
    summary["row_selection"] = {
        "reported": len(rows),
        "padding": len(padding_rows),
        "empty_mask": len(empty_mask_rows),
        "completion_ratio_cutoff": len(cutoff_rows),
    }
    return summary


def record(
    stage: str,
    data: Any,
    *,
    tags: list[dict] | None = None,
    row_keys: list[str] | None = None,
    extra: dict | None = None,
    force: bool = False,
) -> None:
    """Log and optionally persist a batch snapshot."""
    if not force and not enabled():
        return
    try:
        summary = summarize_batch(data, tags=tags, row_keys=row_keys, extra=extra)
        payload = {"stage": stage, **summary}
        logger.warning("SWE_AGENT_TIMEOUT_DEBUG %s", json.dumps(payload, sort_keys=True, default=str))

        dump_dir = os.getenv("SWE_AGENT_TIMEOUT_PREDICTION_DEBUG_DIR")
        if not dump_dir:
            return
        path = Path(dump_dir)
        path.mkdir(parents=True, exist_ok=True)
        filename = f"{int(summary['timestamp_unix'] * 1_000_000)}_{summary['pid']}_{summary['rank']}_{stage}.json"
        (path / filename).write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    except Exception:
        # Debugging must not replace the training error being investigated.
        logger.exception("Unable to collect timeout debug snapshot for stage=%s", stage)

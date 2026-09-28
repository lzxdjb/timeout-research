"""Restore the frozen Qwen3.5-Omni audio tower after actor weight updates."""

from __future__ import annotations

import json
import logging
import math
import time
from collections import defaultdict
from collections.abc import Iterable
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_ARCHITECTURE = "GageQwen3_5OmniMoeForConditionalGeneration"
_CHECKPOINT_PREFIX = "model.audio_tower."
_AUDIO_PREFIXES = (_CHECKPOINT_PREFIX, "audio_tower.")


def cuda_memory_snapshot(device: Any = None) -> dict[str, Any]:
    """Return process-visible CUDA memory counters without allocating memory."""
    if not torch.cuda.is_available():
        return {"cuda_available": False}
    try:
        free, total = torch.cuda.mem_get_info(device)
        return {
            "cuda_available": True,
            "device": str(device) if device is not None else str(torch.cuda.current_device()),
            "allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "reserved_bytes": int(torch.cuda.memory_reserved(device)),
            "max_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "free_bytes": int(free),
            "total_bytes": int(total),
        }
    except Exception as exc:  # pragma: no cover - accelerator-specific failures
        return {"cuda_available": True, "memory_query_error": repr(exc)}


def log_cuda_memory_snapshot(
    phase: str,
    *,
    device: Any = None,
    rank: int | None = None,
    global_steps: int | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Log one structured memory sample for correlating rollout phases."""
    snapshot = cuda_memory_snapshot(device)
    fields = {"phase": phase, "rank": rank, "step": global_steps, **snapshot, **(extra or {})}
    logger.info("vllm_cuda_memory %s", " ".join(f"{key}={value}" for key, value in fields.items()))
    return snapshot


def _dtype_itemsize(dtype: str) -> int:
    return {
        "BF16": 2,
        "F16": 2,
        "F32": 4,
        "F64": 8,
        "I64": 8,
        "I32": 4,
        "I16": 2,
        "I8": 1,
        "U8": 1,
        "BOOL": 1,
    }.get(dtype.upper(), 0)


def audio_weight_name(name: str) -> str | None:
    """Return a tower-relative name for a supported audio weight prefix."""
    for prefix in _AUDIO_PREFIXES:
        if name.startswith(prefix):
            return name[len(prefix) :]
    return None


def _weight_schema(tower: torch.nn.Module) -> tuple[dict[str, tuple[int, ...]], dict[str, set[str]]]:
    """Derive full HF tensor shapes, including every constituent of fused QKV."""
    shapes: dict[str, tuple[int, ...]] = {}
    targets: dict[str, set[str]] = {}
    for name, param in tower.named_parameters():
        module_name, _, leaf = name.rpartition(".")
        module = tower.get_submodule(module_name)
        if module_name.endswith(".self_attn.qkv") and leaf in {"weight", "bias"}:
            sources = set()
            for projection, output_size in zip(("q", "k", "v"), module.output_sizes, strict=True):
                source = name.replace(".self_attn.qkv.", f".self_attn.{projection}_proj.")
                shapes[source] = (output_size, module.input_size) if leaf == "weight" else (output_size,)
                sources.add(source)
            targets[name] = sources
        else:
            # vLLM LinearBase exposes the global dimensions, while its parameter
            # shapes may describe only this TP rank's row/column partition.
            if leaf in {"weight", "bias"} and hasattr(module, "input_size") and hasattr(module, "output_size"):
                shape = (module.output_size, module.input_size) if leaf == "weight" else (module.output_size,)
            else:
                shape = tuple(param.shape)
            shapes[name] = shape
            targets[name] = {name}
    if not shapes:
        raise RuntimeError("The configured Omni audio tower has no parameters")
    return shapes, targets


def _checkpoint_files(source: Path) -> dict[str, Path]:
    """Resolve only audio tensors in an indexed or single-file checkpoint."""
    index = source / "model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text(encoding="utf-8")).get("weight_map")
        if not isinstance(weight_map, dict):
            raise RuntimeError(f"Missing weight_map in audio source checkpoint index: {index}")
        return {
            name[len(_CHECKPOINT_PREFIX) :]: source / shard
            for name, shard in weight_map.items()
            if name.startswith(_CHECKPOINT_PREFIX)
        }
    single = source / "model.safetensors"
    if not single.is_file():
        raise FileNotFoundError(f"Frozen audio restoration requires a local safetensors checkpoint: {source}")
    with safe_open(single, framework="pt", device="cpu") as handle:
        return {
            name[len(_CHECKPOINT_PREFIX) :]: single for name in handle.keys() if name.startswith(_CHECKPOINT_PREFIX)
        }


@torch.no_grad()
def restore_frozen_audio_tower(
    model: torch.nn.Module,
    model_config: Any,
    received_names: Iterable[str],
    *,
    rank: int | None = None,
    global_steps: int | None = None,
) -> dict[str, Any] | None:
    """Complete a hybrid rollout update with checkpoint-backed audio weights.

    Megatron's Qwen3.5 actor exports vision/text parameters only. Dummy loading
    and level-2 sleep therefore require restoring the frozen tower on *every*
    update, before post-processing and generation resume. A complete audio
    update from another backend takes precedence; partial updates fail rather
    than mixing trained and frozen audio parameters.

    Args:
        model: The main vLLM model owned by this worker.
        model_config: vLLM model configuration containing the local source path.
        received_names: Tower-relative tensor names received across all buckets.
        rank: Worker rank included in restoration diagnostics.
        global_steps: Actor weight version included in restoration diagnostics.

    Returns:
        Restoration diagnostics, or None for unrelated model architectures.
    """
    hf_config = getattr(model_config, "hf_config", None)
    if _ARCHITECTURE not in (getattr(hf_config, "architectures", None) or []):
        return None
    tower = getattr(model, "audio_tower", None)
    if not isinstance(tower, torch.nn.Module):
        raise RuntimeError("Audio-capable rollout is missing its audio_tower module")

    parameter_bytes = sum(parameter.numel() * parameter.element_size() for parameter in tower.parameters())
    log_cuda_memory_snapshot(
        "audio_restore_before_schema",
        device=next(tower.parameters()).device,
        rank=rank,
        global_steps=global_steps,
        extra={"tower_parameter_bytes": parameter_bytes},
    )

    # STEP 1: Require complete coverage, including Q, K and V individually.
    shapes, targets = _weight_schema(tower)
    expected = set(shapes)
    received = set(received_names) - set(dict(tower.named_buffers()))
    covered: set[str] = set()
    for name in received:
        if name in shapes:
            covered.add(name)
        elif name in targets:
            covered.update(targets[name])
        else:
            raise RuntimeError(f"Unexpected synchronized audio tensor: {name}")
    if received:
        missing = expected - covered
        if missing:
            raise RuntimeError(f"Incomplete synchronized audio tower; missing tensors: {sorted(missing)}")
        logger.info("Omni audio supplied by actor: rank=%s step=%s tensors=%d", rank, global_steps, len(received))
        log_cuda_memory_snapshot(
            "audio_restore_after_actor_sync",
            device=next(tower.parameters()).device,
            rank=rank,
            global_steps=global_steps,
            extra={"received_tensors": len(received), "tower_parameter_bytes": parameter_bytes},
        )
        return {"source": "actor", "tensor_count": len(received)}

    # STEP 2: Validate headers before copying any checkpoint tensor into the model.
    started = time.monotonic()
    source_value = getattr(model_config, "model", None)
    if not source_value:
        raise RuntimeError("Frozen audio restoration requires model_config.model to identify its source checkpoint")
    source = Path(source_value).expanduser()
    files = _checkpoint_files(source)
    missing, unexpected = expected - files.keys(), files.keys() - expected
    if missing or unexpected:
        raise RuntimeError(
            f"Audio checkpoint coverage mismatch at {source}; "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    by_file: dict[Path, list[str]] = defaultdict(list)
    for name, file in files.items():
        by_file[file].append(name)

    checkpoint_bytes = 0
    with ExitStack() as stack:
        handles = {file: stack.enter_context(safe_open(file, framework="pt", device="cpu")) for file in sorted(by_file)}
        for file, names in by_file.items():
            handle = handles[file]
            available = set(handle.keys())
            for name in names:
                key = _CHECKPOINT_PREFIX + name
                if key not in available:
                    raise RuntimeError(f"Indexed audio tensor {key} is absent from {file}")
                tensor_slice = handle.get_slice(key)
                actual_shape = tuple(tensor_slice.get_shape())
                checkpoint_bytes += _dtype_itemsize(tensor_slice.get_dtype()) * math.prod(actual_shape)
                if actual_shape != shapes[name]:
                    raise RuntimeError(
                        f"Audio tensor shape mismatch for {key}: checkpoint={actual_shape}, expected={shapes[name]}"
                    )

        # STEP 3: Let the native loader perform TP sharding and QKV packing.
        # Stream CPU tensors; never read language/vision tensors or retain a
        # second complete GPU model. Reopen on each update after level-2 sleep.
        cpu_bytes_loaded = 0

        def checkpoint_weights():
            nonlocal cpu_bytes_loaded
            for file, names in by_file.items():
                for name in sorted(names):
                    tensor = handles[file].get_tensor(_CHECKPOINT_PREFIX + name)
                    cpu_bytes_loaded += tensor.nbytes
                    yield name, tensor

        try:
            loaded = set(tower.load_weights(checkpoint_weights()))
        except torch.cuda.OutOfMemoryError:
            log_cuda_memory_snapshot(
                "audio_restore_oom",
                device=next(tower.parameters()).device,
                rank=rank,
                global_steps=global_steps,
                extra={
                    "checkpoint_tensors": len(files),
                    "checkpoint_bytes": checkpoint_bytes,
                    "cpu_bytes_loaded": cpu_bytes_loaded,
                    "tower_parameter_bytes": parameter_bytes,
                },
            )
            raise
        if loaded != set(targets):
            raise RuntimeError(
                f"Audio loader coverage mismatch: missing={sorted(set(targets) - loaded)}, "
                f"unexpected={sorted(loaded - set(targets))}"
            )

    elapsed = time.monotonic() - started
    log_cuda_memory_snapshot(
        "audio_restore_after_load",
        device=next(tower.parameters()).device,
        rank=rank,
        global_steps=global_steps,
        extra={
            "checkpoint_tensors": len(files),
            "checkpoint_bytes": checkpoint_bytes,
            "cpu_bytes_loaded": cpu_bytes_loaded,
            "tower_parameter_bytes": parameter_bytes,
            "restore_seconds": round(elapsed, 3),
        },
    )
    logger.info(
        "Restored frozen Omni audio: rank=%s step=%s source=%s checkpoint_tensors=%d parameters=%d seconds=%.3f",
        rank,
        global_steps,
        source,
        len(files),
        len(loaded),
        elapsed,
    )
    return {"source": str(source), "tensor_count": len(files), "parameter_count": len(loaded), "seconds": elapsed}

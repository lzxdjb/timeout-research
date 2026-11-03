"""Verify a Stage 2 HF export changed the projector and preserved the audio encoder."""

import argparse
import json
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import SafetensorError, safe_open

AUDIO_PREFIX = "model.audio_tower."
PROJECTOR_KEYS = frozenset(
    f"{AUDIO_PREFIX}{layer}.{kind}" for layer in ("proj1", "proj2") for kind in ("weight", "bias")
)


def _audio_files(checkpoint):
    checkpoint = Path(checkpoint)
    index = checkpoint / "model.safetensors.index.json"
    single = checkpoint / "model.safetensors"
    if index.is_file():
        weight_map = json.loads(index.read_text())["weight_map"]
        files = {name: checkpoint / shard for name, shard in weight_map.items() if name.startswith(AUDIO_PREFIX)}
    elif single.is_file():
        with safe_open(single, framework="pt", device="cpu") as handle:
            files = {name: single for name in handle.keys() if name.startswith(AUDIO_PREFIX)}
    else:
        raise ValueError(f"No HF safetensors index or model.safetensors in {checkpoint}")
    if not PROJECTOR_KEYS <= files.keys() or not (files.keys() - PROJECTOR_KEYS):
        raise ValueError(f"Incomplete audio tower in {checkpoint}: encoder and all four projector tensors required")
    for shard in set(files.values()):
        if not shard.is_file():
            raise ValueError(f"Missing audio shard: {shard}")
    return files


def check_projector_update(source_hf, saved_hf):
    """Read one pair of audio tensors at a time, without loading other model weights."""
    source_files = _audio_files(source_hf)
    saved_files = _audio_files(saved_hf)
    if source_files.keys() != saved_files.keys():
        raise ValueError(
            f"Audio key schema mismatch: missing={sorted(source_files.keys() - saved_files.keys())}, "
            f"unexpected={sorted(saved_files.keys() - source_files.keys())}"
        )
    changed_total = 0
    encoder_count = 0
    with ExitStack() as stack:
        handles = {
            shard: stack.enter_context(safe_open(shard, framework="pt", device="cpu"))
            for shard in set(source_files.values()) | set(saved_files.values())
        }
        for name in sorted(source_files):
            before = handles[source_files[name]].get_tensor(name)
            after = handles[saved_files[name]].get_tensor(name)
            if before.shape != after.shape or before.dtype != after.dtype:
                raise ValueError(
                    f"Audio tensor schema mismatch for {name}: "
                    f"source={tuple(before.shape)}/{before.dtype}, saved={tuple(after.shape)}/{after.dtype}"
                )
            if name not in PROJECTOR_KEYS:
                if not torch.equal(before.reshape(-1).view(torch.uint8), after.reshape(-1).view(torch.uint8)):
                    raise ValueError(f"Frozen encoder tensor changed: {name}")
                encoder_count += 1
                continue
            if not torch.isfinite(before).all() or not torch.isfinite(after).all():
                raise ValueError(f"Nonfinite projector tensor: {name}")
            changed = int(torch.count_nonzero(before != after))
            max_abs = float((after.to(torch.float64) - before.to(torch.float64)).abs().max())
            print(f"{name}: changed={changed}/{before.numel()} max_abs_change={max_abs:.9g}")
            changed_total += changed
    if not changed_total:
        raise ValueError("Projector did not change; no updated projector elements in the saved checkpoint")
    print(
        f"PASS: {encoder_count} frozen encoder tensors are bitwise identical; "
        f"{changed_total} projector elements changed; all projector tensors are finite."
    )
    return changed_total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_hf", metavar="SOURCE_HF", type=Path)
    parser.add_argument("saved_hf", metavar="SAVED_HF", type=Path)
    args = parser.parse_args()
    try:
        check_projector_update(args.source_hf, args.saved_hf)
    except (ValueError, OSError, KeyError, RuntimeError, SafetensorError) as exc:
        parser.exit(1, f"FAIL: {exc}\n")


if __name__ == "__main__":
    main()

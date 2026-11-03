"""Run the full audio launcher sequentially for completed checkpoint exports."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _discover_checkpoints(root: Path) -> tuple[list[tuple[int, Path]], list[dict[str, Any]]]:
    if not root.is_dir():
        raise ValueError(f"Checkpoint directory not found: {root}")
    marker = root / "latest_checkpointed_iteration.txt"
    if not marker.is_file():
        raise ValueError(f"Completed checkpoint marker not found: {marker}; set CHECKPOINT_STEP explicitly.")
    value = marker.read_text().strip()
    if not value.isascii() or not value.isdigit():
        raise ValueError(f"Invalid completed-checkpoint marker {marker}: {value!r}")
    latest = int(value)

    candidates = []
    skipped = []
    for checkpoint in root.iterdir():
        match = re.fullmatch(r"global_step_(0|[1-9][0-9]*)", checkpoint.name)
        if not checkpoint.is_dir() or match is None:
            if checkpoint.is_dir() and checkpoint.name.startswith("global_step_"):
                print(f"Skipping {checkpoint.name}: invalid checkpoint step name", file=sys.stderr, flush=True)
            continue
        step = int(match[1])
        model = checkpoint / "actor" / "model" / "huggingface"
        reason = ""
        if step > latest:
            reason = f"newer than completed-checkpoint marker ({latest})"
        else:
            try:
                # Use the same export requirements as the single-step launcher.
                json.loads((model / "config.json").read_text())
                weights = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
                if not isinstance(weights, dict) or not weights:
                    raise ValueError("checkpoint index has no weight map")
                if not any(name.startswith("model.audio_tower.") for name in weights):
                    raise ValueError("checkpoint index has no frozen audio tower")
                shards = set(weights.values())
                if not all(isinstance(shard, str) and shard for shard in shards):
                    raise ValueError("checkpoint index has invalid shard names")
                missing = sorted(shard for shard in shards if not (model / shard).is_file())
                if missing:
                    raise ValueError(f"checkpoint export is missing weight shards: {missing}")
            except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
                reason = str(error)
        if reason:
            skipped.append({"step": step, "checkpoint_path": str(model), "reason": reason})
        else:
            candidates.append((step, model))
    candidates.sort()
    skipped.sort(key=lambda item: item["step"])
    for item in skipped:
        print(f"Skipping global_step_{item['step']}: {item['reason']}", file=sys.stderr, flush=True)
    if not candidates:
        raise ValueError(f"No complete audio checkpoints found under {root}")
    return candidates, skipped


def _write_summary(run_dir: Path, summary: dict[str, Any]) -> None:
    summary["updated_at"] = _timestamp()
    destination = run_dir / "checkpoint_summary.json"
    temporary = run_dir / ".checkpoint_summary.json.tmp"
    temporary.write_text(json.dumps(summary, indent=2) + "\n")
    temporary.replace(destination)


def main() -> int:
    """Discover a checkpoint snapshot and run one isolated audio suite per step."""
    launcher = Path(sys.argv[1]).resolve()
    root = Path(os.environ["CHECKPOINT_ROOT"]).expanduser().resolve()
    try:
        checkpoints, skipped = _discover_checkpoints(root)
    except (OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 2

    preview = os.environ.get("CONFIG_ONLY", "0") == "1"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    experiment = os.environ.get("EXPERIMENT_NAME") or f"qwen35_audio_full_all_{stamp}_{os.getpid()}"
    default_run_dir = Path(os.environ["TARGET_VERL"]) / "validation_data" / "audio_full" / experiment
    run_dir = Path(os.environ.get("AUDIO_FULL_RUN_DIR") or default_run_dir).expanduser().resolve()
    summary: dict[str, Any] = {
        "checkpoint_root": str(root),
        "run_dir": str(run_dir),
        "status": "running",
        "started_at": _timestamp(),
        "skipped_checkpoints": skipped,
        "checkpoints": [
            {
                "step": step,
                "checkpoint_path": str(model),
                "run_dir": str(run_dir / f"global_step_{step}"),
                "experiment_name": f"{experiment}_step{step}",
                "status": "pending",
                "exit_code": None,
            }
            for step, model in checkpoints
        ],
    }
    print(f"Audio checkpoint snapshot: {[step for step, _ in checkpoints]}", flush=True)
    print(f"All-checkpoint run directory: {run_dir}", flush=True)
    if not preview:
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            print(f"Refusing to reuse an existing evaluation run directory: {run_dir}", file=sys.stderr)
            return 2
        _write_summary(run_dir, summary)

    # Each child owns its Ray driver and exits fully before the next one starts.
    # Do not use global Ray/GPU cleanup, which would affect unrelated jobs.
    for item in summary["checkpoints"]:
        child_env = {
            **os.environ,
            "CHECKPOINT_ROOT": str(root),
            "CHECKPOINT_STEP": str(item["step"]),
            "AUDIO_FULL_RUN_DIR": item["run_dir"],
            "EXPERIMENT_NAME": item["experiment_name"],
        }
        item.update(status="running", started_at=_timestamp())
        if not preview:
            _write_summary(run_dir, summary)
        try:
            result = subprocess.run(["bash", str(launcher), *sys.argv[2:]], env=child_env, check=False)
            item["exit_code"] = result.returncode
            item["status"] = "completed" if result.returncode == 0 else "failed"
        except OSError as error:
            print(f"Cannot launch checkpoint step {item['step']}: {error}", file=sys.stderr, flush=True)
            item.update(status="failed", exit_code=127, error=str(error))
        except KeyboardInterrupt:
            item.update(status="interrupted", exit_code=130, completed_at=_timestamp())
            summary.update(status="interrupted", completed_at=_timestamp())
            if not preview:
                _write_summary(run_dir, summary)
            return 130
        item["completed_at"] = _timestamp()
        if not preview:
            _write_summary(run_dir, summary)
        outcome = "configuration previewed" if preview and item["exit_code"] == 0 else item["status"]
        print(f"Checkpoint step {item['step']}: {outcome} (exit={item['exit_code']})", flush=True)

    failed = [item["step"] for item in summary["checkpoints"] if item["status"] != "completed"]
    summary.update(status="completed_with_failures" if failed else "completed", completed_at=_timestamp())
    if not preview:
        _write_summary(run_dir, summary)
        print(f"Checkpoint summary: {run_dir / 'checkpoint_summary.json'}", flush=True)
    if failed:
        print(f"Failed checkpoint steps: {failed}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

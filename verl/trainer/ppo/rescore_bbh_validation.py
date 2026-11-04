"""Rescore saved BBH validation answers without generation or dashboard writes."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from verl.trainer.ppo.bbh_validation_scoring import score_bbh_validation


def _metadata(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    import pyarrow.parquet as pq

    records = pq.read_table(path).to_pylist()
    result = {}
    for row in records:
        uid = str(row["uid"])
        if uid in result:
            raise ValueError(f"Duplicate metadata UID: {uid}")
        result[uid] = row
    return result


def _mean(records: list[dict[str, Any]], key: str) -> float:
    return sum(float(row[key]) for row in records) / len(records)


def rescore_trajectory(
    samples_path: Path, output_dir: Path, *, bbh_parquet: Path | None = None
) -> dict[str, Any]:
    """Write a separate, reproducible BBH report while retaining legacy scores.

    Saved prompts and references are authoritative. Optional Parquet metadata
    must match them. Historical trajectories without extra_info recover their
    task name from the standard bbh:<task>:<index> UID, without modifying data.
    """
    samples_path = samples_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite an existing report directory: {output_dir}")
    metadata = _metadata(bbh_parquet)
    records = []
    seen = set()
    digest = hashlib.sha256()
    with samples_path.open("rb") as source:
        for line_number, raw in enumerate(source, 1):
            digest.update(raw)
            if not raw.strip():
                continue
            row = json.loads(raw)
            if row.get("benchmark") != "bbh":
                continue
            uid = str(row["uid"])
            if uid in seen:
                raise ValueError(f"Duplicate BBH trajectory UID: {uid}")
            seen.add(uid)
            info = dict(row.get("extra_info") or {})
            task_source = "trajectory"
            if bbh_parquet is not None:
                if uid not in metadata:
                    raise ValueError(f"BBH UID missing from supplied metadata: {uid}")
                prepared = metadata[uid]
                if prepared["prompt"] != row["prompt"] or (
                    str(prepared["reward_model"]["ground_truth"]) != str(row["ground_truth"])
                ):
                    raise ValueError(f"Saved prompt/reference differs from supplied metadata: {uid}")
                info = {**dict(prepared.get("extra_info") or {}), **info}
                task_source = "parquet"
            if not info.get("task") and not info.get("meta_json"):
                parts = uid.split(":")
                if len(parts) == 3 and parts[0] == "bbh" and parts[1]:
                    info["task"] = parts[1]
                    task_source = "uid"
                else:
                    task_source = "unknown"
            diagnostic = score_bbh_validation(row["response"], row["ground_truth"], row["prompt"], info)
            reward = row["reward"]
            record = {
                "uid": uid,
                "source_line": line_number,
                "ground_truth": row["ground_truth"],
                "original_acc": float(reward["acc"]),
                "original_deterministic_acc": float(reward["deterministic_acc"]),
                "original_format_valid": float(reward["format_valid"]),
                "original_prediction": reward.get("pred"),
                "finish_reason": row.get("finish_reason"),
                "response_tokens": row.get("response_tokens", 0),
                "task_source": task_source,
                **diagnostic,
            }
            records.append(record)
    if not records:
        raise ValueError(f"No BBH validation records found: {samples_path}")
    keys = (
        "original_acc",
        "original_deterministic_acc",
        "original_format_valid",
        "answer_acc",
        "representation_acc",
        "boxed_answer_acc",
        "boxed_representation_acc",
    )
    tasks = defaultdict(list)
    for row in records:
        tasks[row.get("task") or "unknown"].append(row)
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "samples_path": str(samples_path),
        "samples_sha256": digest.hexdigest(),
        "bbh_parquet": str(bbh_parquet.resolve()) if bbh_parquet is not None else None,
        "parser_versions": sorted({row["parser_version"] for row in records}),
        "count": len(records),
        "metrics": {key: _mean(records, key) for key in keys},
        "tasks": {
            task: {"count": len(rows), **{key: _mean(rows, key) for key in keys}}
            for task, rows in sorted(tasks.items())
        },
        "notes": [
            "Offline only: no model calls, source edits, or W&B writes.",
            "answer_acc permits an unboxed final answer; legacy acc is retained independently.",
            "representation_acc adds task-specific separator/LaTeX compatibility and is diagnostic, not official BBH.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / "samples.jsonl").open("x", encoding="utf-8") as destination:
        for row in records:
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    lines = [
        "# Offline BBH validation rescore",
        "",
        f"Source: `{samples_path}`; {len(records)} BBH records. All metrics are fractions of the same records.",
        "",
        "| Metric | Accuracy |",
        "| --- | ---: |",
        *[f"| {key} | {summary['metrics'][key]:.2%} |" for key in keys],
        "",
        *[f"- {note}" for note in summary["notes"]],
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("samples_path", type=Path, help="Existing reasoning validation samples.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory; existing targets are rejected")
    parser.add_argument("--bbh-parquet", type=Path, help="Optional matching prepared metadata, read without rewriting")
    args = parser.parse_args()
    summary = rescore_trajectory(args.samples_path, args.output_dir, bbh_parquet=args.bbh_parquet)
    print(json.dumps({"count": summary["count"], "metrics": summary["metrics"]}, indent=2))


if __name__ == "__main__":
    main()

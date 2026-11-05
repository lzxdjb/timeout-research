#!/usr/bin/env python3
"""Reaggregate saved audio ASR edit counts without inference or W&B writes.

Usage: python scripts/rescore_audio_asr.py SUITE_DIR [...] --output NEW_JSON

Persisted edit counts preserve the original transcript normalization. All saved
outputs contribute, including failed, repetitive, and token-limited outputs.
Missing counts, duplicate samples, and incomplete task coverage are errors.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

ASR_BENCHMARKS = {
    "aishell1": ("cer", "reference_char_count"),
    "wenetspeech": ("cer", "reference_char_count"),
    "librispeech": ("wer", "reference_word_count"),
}
REPEATED_BLOCK = re.compile(r"(.{1,32}?)\1{7,}", re.DOTALL)


def _records(value: Any, path: Path) -> Iterator[dict[str, Any]]:
    if isinstance(value, dict) and isinstance(value.get("samples"), list):
        value = value["samples"]
    values = value if isinstance(value, list) else [value]
    for record in values:
        if not isinstance(record, dict):
            raise ValueError(f"{path}: expected sample objects")
        yield record


def read_samples(path: Path) -> Iterator[dict[str, Any]]:
    """Read physical JSONL lines, or a pretty JSON object/array.

    Iterating the file preserves Unicode line/paragraph separators inside JSON
    strings, unlike ``str.splitlines()``. The benchmark root JSONL is preferred
    over per-task previews, which may contain only the last saved sample.
    """
    with path.open(encoding="utf-8") as handle:
        for first_line in handle:
            if first_line.strip():
                break
        else:
            return
        try:
            first_record = json.loads(first_line)
        except json.JSONDecodeError:
            handle.seek(0)
            yield from _records(json.load(handle), path)
            return
        yield from _records(first_record, path)
        for line_number, line in enumerate(handle, start=2):
            if line.strip():
                try:
                    yield from _records(json.loads(line), path)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSONL") from exc


def _count(metadata: dict[str, Any], key: str, sample_id: str) -> int:
    value = metadata.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"Sample {sample_id}: {key} must be a nonnegative integer; got {value!r}")
    return value


def _metric_metadata(record: dict[str, Any], reference_key: str) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for raw in (record.get("metrics"), record.get("evaluation", {}).get("raw_metrics")):
        if isinstance(raw, dict):
            candidates.extend(metric for metric in raw.values() if isinstance(metric, dict))
    candidates.extend(record.get("evaluation", {}).get("metrics", []))
    metadata_candidates = [
        metric["metadata"]
        for metric in candidates
        if isinstance(metric, dict) and isinstance(metric.get("metadata"), dict) and reference_key in metric["metadata"]
    ]
    sample_id = str(record.get("sample_id", "unknown"))
    if not metadata_candidates:
        raise ValueError(f"Sample {sample_id}: persisted ASR edit/reference counts are missing")
    counts = {
        (_count(metadata, "edit_distance", sample_id), _count(metadata, reference_key, sample_id))
        for metadata in metadata_candidates
    }
    if len(counts) != 1:
        raise ValueError(f"Sample {sample_id}: persisted ASR metrics disagree on edit/reference counts")
    return metadata_candidates[0]


@dataclass
class ASRCounts:
    """Accumulate corpus counts and explicitly defined output diagnostics."""

    sample_count: int = 0
    total_edit_distance: int = 0
    total_reference_units: int = 0
    utterance_percent_sum: float = 0.0
    diagnostics: Counter[str] = field(default_factory=Counter)

    def add(self, record: dict[str, Any], metadata: dict[str, Any], reference_key: str) -> None:
        sample_id = str(record.get("sample_id", "unknown"))
        edits = _count(metadata, "edit_distance", sample_id)
        reference_units = _count(metadata, reference_key, sample_id)
        self.sample_count += 1
        self.total_edit_distance += edits
        self.total_reference_units += reference_units
        # Match the historical per-utterance denominator for empty references.
        self.utterance_percent_sum += 100.0 * edits / (reference_units or 1)

        output = record.get("model_output") or {}
        prediction = metadata.get("prediction_normalized", output.get("answer", ""))
        prediction = str(prediction or "")
        raw_response = output.get("raw_response") or {}
        choices = raw_response.get("choices") or []
        finish_reason = output.get("finish_reason")
        if finish_reason is None and choices:
            finish_reason = choices[0].get("finish_reason")
        usage = output.get("usage") or raw_response.get("usage") or {}
        self.diagnostics["finish_reason_length_count"] += finish_reason == "length"
        self.diagnostics["truncated_flag_count"] += output.get("truncated") is True
        self.diagnostics["completion_tokens_ge_4096_count"] += (usage.get("completion_tokens") or 0) >= 4096
        self.diagnostics["empty_normalized_prediction_count"] += not prediction
        self.diagnostics["utterance_error_rate_gt_100_percent_count"] += edits > reference_units
        self.diagnostics["repeated_block_count"] += any(
            match.end() - match.start() >= 32 for match in REPEATED_BLOCK.finditer(prediction)
        )
        self.diagnostics["record_error_count"] += bool(record.get("error"))
        self.diagnostics["excluded_in_original_aggregate_count"] += (
            record.get("evaluation", {}).get("include_in_aggregate") is False
        )

    def merge(self, other: ASRCounts) -> None:
        self.sample_count += other.sample_count
        self.total_edit_distance += other.total_edit_distance
        self.total_reference_units += other.total_reference_units
        self.utterance_percent_sum += other.utterance_percent_sum
        self.diagnostics.update(other.diagnostics)

    def result(self, value_key: str) -> dict[str, Any]:
        if self.total_reference_units:
            corpus_percent = 100.0 * self.total_edit_distance / self.total_reference_units
        elif self.total_edit_distance:
            raise ValueError("Corpus error rate is undefined: edits exist but total reference units are zero")
        else:
            corpus_percent = 0.0
        return {
            "value_key": value_key,
            "unit": "percent",
            "corpus_percent": corpus_percent,
            "utterance_mean_percent": self.utterance_percent_sum / self.sample_count if self.sample_count else 0.0,
            "sample_count": self.sample_count,
            "total_edit_distance": self.total_edit_distance,
            "total_reference_units": self.total_reference_units,
            "diagnostics": dict(self.diagnostics),
        }


def rescore_suite(path: Path) -> dict[str, Any]:
    """Reaggregate every saved ASR sample and verify summary sample coverage."""
    path = path / "suite_summary.json" if path.is_dir() else path
    path = path.resolve()
    suite = json.loads(path.read_text(encoding="utf-8"))
    results: list[dict[str, Any]] = []
    for benchmark in suite.get("benchmarks", []):
        benchmark_id = benchmark.get("benchmark_id")
        if benchmark_id not in ASR_BENCHMARKS:
            continue
        value_key, reference_key = ASR_BENCHMARKS[benchmark_id]
        benchmark_dir = path.parent / benchmark_id
        root_samples = benchmark_dir / "samples.jsonl"
        sources = (
            [root_samples] if root_samples.is_file() else sorted((benchmark_dir / "samples").rglob("samples.jsonl"))
        )
        if not sources:
            raise ValueError(f"{benchmark_id}: no saved sample artifacts found below {benchmark_dir}")
        tasks: dict[str, ASRCounts] = {}
        seen: set[tuple[str, str]] = set()
        for source in sources:
            for record in read_samples(source):
                task_id = record.get("task_id") or record.get("sample", {}).get("_gage_task_id")
                sample_id = record.get("sample_id")
                if not task_id or not sample_id:
                    raise ValueError(f"{source}: sample is missing task_id or sample_id")
                identity = (task_id, sample_id)
                if identity in seen:
                    raise ValueError(f"{source}: duplicate saved sample {identity}")
                seen.add(identity)
                metadata = _metric_metadata(record, reference_key)
                tasks.setdefault(task_id, ASRCounts()).add(record, metadata, reference_key)
        expected_tasks = benchmark.get("summary", {}).get("tasks", [])
        expected_ids = {task["task_id"] for task in expected_tasks}
        if set(tasks) != expected_ids:
            raise ValueError(
                f"{benchmark_id}: saved task IDs {sorted(tasks)} differ from summary {sorted(expected_ids)}"
            )
        pooled = ASRCounts()
        task_results = []
        for task in expected_tasks:
            task_id = task["task_id"]
            counts = tasks[task_id]
            expected_count = task.get("sample_count")
            if counts.sample_count != expected_count:
                raise ValueError(
                    f"{benchmark_id}/{task_id}: saved {counts.sample_count} samples, expected {expected_count}; "
                    "refusing to publish an incomplete score"
                )
            task_results.append({"task_id": task_id, **counts.result(value_key)})
            pooled.merge(counts)
        results.append(
            {
                "benchmark_id": benchmark_id,
                "sample_sources": [str(source) for source in sources],
                "tasks": task_results,
                "pooled": pooled.result(value_key),
            }
        )
    if not results:
        raise ValueError(f"{path}: no supported ASR benchmarks in suite summary")
    return {"run_id": suite.get("run_id"), "suite_summary": str(path), "benchmarks": results}


def build_report(paths: list[Path]) -> dict[str, Any]:
    """Build a local report; source files and historical results stay untouched."""
    return {
        "schema_version": "audio_asr_rescore.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": {
            "counts_source": "persisted per-sample metric metadata; original normalization preserved",
            "corpus_percent": "100 * sum(edit_distance) / sum(reference_unit_count)",
            "utterance_mean_percent": "mean(100 * edit_distance / max(reference_unit_count, 1))",
            "inclusion": "all saved samples, including errors, repetitions, and truncated outputs; no rate clamp",
            "coverage": "task IDs and saved sample counts must equal the suite summary",
            "pooled": "sum edit/reference counts across benchmark tasks; never average split percentages",
            "diagnostics": {
                "finish_reason_length_count": "saved model finish_reason is exactly 'length'",
                "truncated_flag_count": "saved model_output.truncated is true",
                "completion_tokens_ge_4096_count": "saved completion token usage >= 4096 (diagnostic threshold only)",
                "repeated_block_count": (
                    "normalized prediction has >= 8 consecutive copies of a 1..32 character block "
                    "spanning >= 32 characters; heuristic only"
                ),
                "empty_normalized_prediction_count": "normalized prediction is empty",
                "utterance_error_rate_gt_100_percent_count": "edit_distance exceeds reference unit count",
                "record_error_count": "saved record.error is nonempty",
                "excluded_in_original_aggregate_count": "saved include_in_aggregate is false; still included here",
            },
        },
        "runs": [rescore_suite(path) for path in paths],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", type=Path, nargs="+", help="Saved suite directory or suite_summary.json")
    parser.add_argument("--output", type=Path, required=True, help="New local JSON report; existing files are refused")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    report = build_report(args.suite)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"Saved ASR rescore report: {args.output}")


if __name__ == "__main__":
    main()

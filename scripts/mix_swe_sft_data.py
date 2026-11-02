#!/usr/bin/env python3
"""Publish a reproducible, task-disjoint mixture of accepted SWE SFT snapshots.

Inputs are the core train/validation parquet files from export_swe_sft_snapshot.
By default every accepted teacher training task is retained, and existing examples
are sampled without replacement toward the requested teacher fraction. Explicit
oversampling is also available. Validation has one example per task and no repeats.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import random
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from scripts.make_swe_sft_data import _split_is_validation


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def task_key(row):
    benchmark, task_id = row.get("benchmark"), row.get("task_id")
    if not isinstance(benchmark, str) or not benchmark or not isinstance(task_id, str) or not task_id:
        raise ValueError("Every SFT row must have nonempty benchmark and task_id strings")
    return benchmark, task_id


def excluded_tasks(paths):
    import pyarrow.parquet as pq

    excluded = set()
    for path in paths:
        names = pq.read_schema(path).names
        if "extra_info" in names:
            rows = pq.read_table(path, columns=["extra_info"]).to_pylist()
            rows = [r["extra_info"] for r in rows]
        elif {"benchmark", "task_id"} <= set(names):
            rows = pq.read_table(path, columns=["benchmark", "task_id"]).to_pylist()
        else:
            raise ValueError(f"Held-out parquet has no task identity columns: {path}")
        excluded.update(task_key(row) for row in rows)
    return excluded


def make_token_counter(tokenizer_path, enable_thinking):
    import torch
    from transformers import AutoTokenizer

    from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset
    from verl.utils.tokenizer.chat_template import extract_system_prompt_and_generation

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), trust_remote_code=True)
    # Reuse the training dataset's per-turn rendering and consistency check.
    # A whole-transcript render alone can discard earlier assistant reasoning
    # with Qwen templates, so its length is not a safe training length bound.
    dataset = object.__new__(MultiTurnSFTDataset)
    dataset.tokenizer, dataset.processor = tokenizer, None
    dataset.apply_chat_template_kwargs = {"enable_thinking": enable_thinking}
    dataset.ignore_input_ids_mismatch = False
    dataset.system_prompt, dataset.generation_prompt = extract_system_prompt_and_generation(
        tokenizer, **dataset.apply_chat_template_kwargs
    )

    def count(row):
        messages = row["messages"]
        if not messages or messages[-1].get("role") != "assistant":
            raise ValueError("SFT conversation must end with an assistant response")
        thinking = row.get("enable_thinking", enable_thinking)
        if thinking is not None:
            thinking = bool(thinking)
        indexed = list(enumerate(messages))
        if messages[0]["role"] == "system":
            if len(messages) < 2 or messages[1]["role"] != "user":
                raise ValueError("A leading system message must be followed by a user message")
            indexed = [(0, messages[:2]), *list(enumerate(messages[2:], start=2))]
        pieces, supervised = [], 0
        for index, message in indexed:
            ids, loss_mask, _, _ = dataset._process_single_message(
                index=index,
                message=message,
                full_message=messages,
                tools=row.get("tools") if index == 0 else None,
                enable_thinking=thinking,
            )
            pieces.append(ids)
            supervised += int(loss_mask.sum().item())
        ids = torch.cat(pieces, dim=0)
        dataset.sanity_check(ids, messages, row.get("tools"), thinking)
        if not supervised:
            raise ValueError("SFT conversation has no supervised assistant tokens")
        return ids.numel()

    return count


def _write_jsonl(path, records):
    with path.open("w") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def _repeat_to(rows, count, rng):
    if not rows:
        return []
    result = rows * (count // len(rows))
    remainder = list(rows)
    rng.shuffle(remainder)
    result.extend(remainder[: count % len(rows)])
    return result


def mix(args, *, token_counter=None):
    import pyarrow as pa
    import pyarrow.parquet as pq

    if not 0 <= args.teacher_fraction <= 1:
        raise ValueError("teacher-fraction must be between zero and one")
    if args.teacher_fraction > 0 and not args.teacher_parquet:
        raise ValueError("teacher-parquet is required when teacher-fraction is greater than zero")
    if args.teacher_fraction < 1 and not args.existing_parquet:
        raise ValueError("existing-parquet is required when teacher-fraction is less than one")
    if not 0 < args.val_fraction < 1:
        raise ValueError("val-fraction must be strictly between zero and one")
    if args.max_length <= 0:
        raise ValueError("max-length must be positive")
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Mixture already exists: {output}; choose a new output directory")
    sources = []
    candidates = collections.defaultdict(list)
    schemas = []
    excluded = excluded_tasks(args.exclude_task_parquet)
    excluded_ids = {task_id for _, task_id in excluded}
    held_out_sources = [{"path": str(path.resolve()), "sha256": sha256(path)} for path in args.exclude_task_parquet]
    exclusions = []
    for origin, paths in (("existing", args.existing_parquet), ("teacher", args.teacher_parquet)):
        if (origin == "existing" and args.teacher_fraction == 1) or (
            origin == "teacher" and args.teacher_fraction == 0
        ):
            continue
        for path in paths:
            path = path.resolve()
            before = path.stat()
            file_hash = sha256(path)
            table = pq.read_table(path)
            schemas.append(table.schema)
            sources.append({"path": str(path), "sha256": file_hash, "rows": len(table), "origin": origin})
            for source_row, row in enumerate(table.to_pylist()):
                key = task_key(row)
                if key[1] in excluded_ids:
                    exclusions.append({"benchmark": key[0], "task_id": key[1], "reason": "held_out_task"})
                    continue
                if row.get("raw_score") != 1.0:
                    raise ValueError(f"Non-successful row in accepted snapshot: {path}:{source_row}")
                if not isinstance(row.get("messages"), list) or not row["messages"]:
                    raise ValueError(f"Missing conversation: {path}:{source_row}")
                row = {**row, "sft_origin": origin, "sft_source_parquet": str(path), "sft_source_row": source_row}
                # Teacher replaces an existing demonstration of the same task.
                # Retain alternatives until the complete student sequence has
                # been checked, so an oversized teacher cannot hide a usable
                # existing demonstration. Ties do not depend on input order.
                rank = (
                    origin != "teacher",
                    row.get("response_tokens") or 0,
                    row.get("transcript_hash") or "",
                    str(path),
                    source_row,
                )
                candidates[key[1]].append((rank, row))
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or sha256(path) != file_hash:
                raise ValueError(f"Input changed during mixture export: {path}")
    if not candidates:
        raise ValueError("No accepted tasks remain after held-out exclusions")
    token_counter = token_counter or make_token_counter(args.tokenizer_path, args.enable_thinking)
    train, validation = {"existing": [], "teacher": []}, []
    manifest = []
    for task_id, alternatives in sorted(candidates.items()):
        row = None
        for _, candidate in sorted(alternatives, key=lambda item: item[0]):
            length = token_counter(candidate)
            if length > args.max_length:
                exclusions.append(
                    {
                        "benchmark": candidate["benchmark"],
                        "task_id": task_id,
                        "reason": "student_context_limit",
                        "tokens": length,
                    }
                )
                continue
            row = candidate
            break
        if row is None:
            continue
        row["student_sequence_tokens"] = length
        key = task_key(row)
        is_val = _split_is_validation("", task_id, args.val_fraction)
        (validation if is_val else train[row["sft_origin"]]).append(row)
        manifest.append(
            {
                "benchmark": key[0],
                "task_id": key[1],
                "origin": row["sft_origin"],
                "split": "val" if is_val else "train",
                "student_sequence_tokens": length,
                "transcript_hash": row.get("transcript_hash"),
                "source_parquet": row["sft_source_parquet"],
                "source_row": row["sft_source_row"],
            }
        )
    if not validation:
        raise ValueError("Task-hash split produced no validation rows; collect more data or change val-fraction")
    fractions = {"teacher": args.teacher_fraction, "existing": 1 - args.teacher_fraction}
    if any(fraction > 0 and not train[origin] for origin, fraction in fractions.items()):
        raise ValueError("Each requested source needs at least one training task after filtering")
    eligible_counts = {origin: len(rows) for origin, rows in train.items()}
    rng = random.Random(args.seed)
    if args.sampling_mode == "oversample":
        total = max(len(train[origin]) / fraction for origin, fraction in fractions.items() if fraction > 0)
        targets = {origin: math.ceil(total * fraction) for origin, fraction in fractions.items()}
        mixed = _repeat_to(train["teacher"], targets["teacher"], rng) + _repeat_to(
            train["existing"], targets["existing"], rng
        )
    else:
        # Preserve teacher demonstrations. When too few existing tasks remain,
        # retain all available tasks and report the attainable actual fraction.
        targets = dict(eligible_counts)
        if 0 < args.teacher_fraction < 1:
            targets["existing"] = min(
                len(train["existing"]),
                round(len(train["teacher"]) * (1 - args.teacher_fraction) / args.teacher_fraction),
            )
        mixed = list(train["teacher"]) + rng.sample(train["existing"], targets["existing"])
    rng.shuffle(mixed)
    train_ids = {row["task_id"] for row in mixed}
    if train_ids & {row["task_id"] for row in validation}:
        raise AssertionError("Task leakage between training and teacher-forced validation")
    for record in manifest:
        if record["split"] == "train" and record["task_id"] not in train_ids:
            record.update(split="excluded", reason="mixture_downsample")
            exclusions.append(
                {"benchmark": record["benchmark"], "task_id": record["task_id"], "reason": "mixture_downsample"}
            )
    actual_unique = dict(collections.Counter(row["sft_origin"] for row in {r["task_id"]: r for r in mixed}.values()))
    actual_unique = {origin: actual_unique.get(origin, 0) for origin in train}
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        # Union input schemas so optional tools/reasoning fields survive Arrow
        # inference even when the first selected row does not contain them.
        schema = pa.unify_schemas(schemas, promote_options="permissive")
        schema = pa.schema(
            list(schema)
            + [
                pa.field("sft_origin", pa.string()),
                pa.field("sft_source_parquet", pa.string()),
                pa.field("sft_source_row", pa.int64()),
                pa.field("student_sequence_tokens", pa.int64()),
            ]
        )
        pq.write_table(pa.Table.from_pylist(mixed, schema=schema), stage / "train.parquet", compression="zstd")
        pq.write_table(pa.Table.from_pylist(validation, schema=schema), stage / "val.parquet", compression="zstd")
        _write_jsonl(stage / "task_manifest.jsonl", manifest)
        _write_jsonl(stage / "excluded_tasks.jsonl", exclusions)
        report = {
            "mixture_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "sources": sources,
            "tokenizer_path": str(args.tokenizer_path),
            "max_length": args.max_length,
            "held_out_sources": held_out_sources,
            "enable_thinking": args.enable_thinking,
            "seed": args.seed,
            "val_fraction": args.val_fraction,
            "teacher_fraction_requested": args.teacher_fraction,
            "teacher_fraction_actual": targets["teacher"] / len(mixed),
            "fraction_unit": "training rows",
            "task_identity": "task_id across benchmark aliases",
            "sampling_mode": args.sampling_mode,
            "eligible_train_unique_tasks": eligible_counts,
            "train_unique_tasks": actual_unique,
            "train_rows": targets,
            "repeated_training_rows": len(mixed) - len(train_ids),
            "val_unique_tasks": dict(collections.Counter(r["sft_origin"] for r in validation)),
            "held_out_task_count": len(excluded_ids),
            "train_val_task_overlap": 0,
            "excluded_counts": dict(collections.Counter(r["reason"] for r in exclusions)),
            "files": {p.name: sha256(p) for p in sorted(stage.iterdir())},
        }
        (stage / "mixture_report.json").write_text(json.dumps(report, indent=2) + "\n")
        if output.exists():
            raise FileExistsError(f"Mixture output appeared during export: {output}")
        stage.rename(output)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return report


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-parquet", type=Path, action="append", default=[])
    parser.add_argument("--existing-parquet", type=Path, action="append", default=[])
    parser.add_argument(
        "--exclude-task-parquet",
        type=Path,
        action="append",
        default=[],
        help="Exclude RL validation/test task identities",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True, help="Student tokenizer directory")
    parser.add_argument("--max-length", type=int, default=106496)
    parser.add_argument("--teacher-fraction", type=float, default=0.3)
    parser.add_argument(
        "--sampling-mode",
        choices=["downsample", "oversample"],
        default="downsample",
        help="Default retains teacher tasks and samples existing tasks without replacement",
    )
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main():
    print(json.dumps(mix(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()

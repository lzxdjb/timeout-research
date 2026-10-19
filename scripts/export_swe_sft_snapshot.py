#!/usr/bin/env python3
"""Export an immutable SFT checkpoint without interrupting a running quality filter.

Run as: python -m scripts.export_swe_sft_snapshot --help
"""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from scripts import make_swe_sft_data as sft
from scripts import swe_sft_quality as quality


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def capture_decisions(source, destination):
    """Capture only complete lines within the file size observed when opened."""
    digest = hashlib.sha256()
    records = []
    consumed = 0
    with source.open("rb") as stream, destination.open("xb") as target:
        cutoff = os.fstat(stream.fileno()).st_size
        while consumed < cutoff:
            line = stream.readline(cutoff - consumed)
            if not line.endswith(b"\n"):
                break
            record = json.loads(line)
            if not isinstance(record, dict) or type(record.get("accepted")) is not bool:
                raise ValueError("Malformed completed decision")
            records.append(record)
            target.write(line)
            digest.update(line)
            consumed += len(line)
        if os.fstat(stream.fileno()).st_size < cutoff:
            raise ValueError("Decision log was truncated during snapshot capture")
        # Detect a restart that replaced or rewrote the captured prefix.
        stream.seek(0)
        if hashlib.sha256(stream.read(consumed)).hexdigest() != digest.hexdigest():
            raise ValueError("Decision log changed during snapshot capture")
        current = source.stat()
        if (current.st_dev, current.st_ino) != (os.fstat(stream.fileno()).st_dev, os.fstat(stream.fileno()).st_ino):
            raise ValueError("Decision log was replaced during snapshot capture")
    return records, {
        "observed_bytes": cutoff,
        "captured_bytes": consumed,
        "ignored_partial_bytes": cutoff - consumed,
        "sha256": digest.hexdigest(),
        "completed_decisions": len(records),
    }


def selection_args(args):
    # Defaults match run_swe_sft_quality.sh. Overrides must reproduce the pool's
    # fingerprint; no source scanning or pool preparation is performed here.
    selected = sft.build_parser().parse_args(
        ["--output-dir", str(args.run_dir), "--max-response-tokens", "95536", "--max-tool-dispatches", "120"]
    )
    if args.selection_args_json:
        overrides = json.loads(args.selection_args_json.read_text())
        allowed = set(quality.candidate_kwargs(selected)) | {"relaxed_max_tool_dispatches"}
        allowed -= {"quality_mode", "max_penalized_repeated_tool_calls"}
        if not isinstance(overrides, dict) or set(overrides) - allowed:
            raise ValueError("Unsupported selection-args JSON fields")
        for key, value in overrides.items():
            setattr(selected, key, value)
    return selected


def verify_pool(db, run_dir, selected):
    sources = json.loads((run_dir / "sources.json").read_text())
    fingerprint = quality.repeat.digest(
        {
            "sources": [{k: source[k] for k in ("path", "size", "mtime_ns")} for source in sources["sources"]],
            "filters": quality.candidate_kwargs(selected),
            "core_max_tool_dispatches": selected.max_tool_dispatches,
            "source": quality.SOURCE_SHA256,
            "builder": sha256(Path(sft.__file__)),
        }
    )
    saved = db.execute("SELECT value FROM meta WHERE key='fingerprint'").fetchone()
    if not saved or saved[0] != fingerprint or sources["fingerprint"] != fingerprint:
        raise ValueError(
            "Pool fingerprint mismatch: selection settings or filtering code changed; do not alter the pool"
        )
    return fingerprint


def accepted_candidate(record, db, selected, run_dir):
    digest = record["transcript_hash"]
    packed = db.execute("SELECT task,raw,provenance FROM candidates WHERE hash=?", (digest,)).fetchone()
    if packed is None:
        raise ValueError(f"Accepted transcript missing from pool: {digest}")
    task, raw, provenance = packed
    provenance = json.loads(provenance)
    candidate = sft._candidate(json.loads(gzip.decompress(raw)), **provenance, **quality.candidate_kwargs(selected))
    artifact_path = run_dir / "quality_artifacts" / f"{digest}.json.gz"
    artifact_bytes = artifact_path.read_bytes()
    if hashlib.sha256(artifact_bytes).hexdigest() != record["artifact_sha256"]:
        raise ValueError(f"Artifact SHA256 mismatch: {digest}")
    artifact = json.loads(gzip.decompress(artifact_bytes))
    result = artifact["result"]
    if (
        candidate.transcript_hash != digest
        or record["task_key"] != task
        or tuple(json.loads(task)) != (candidate.benchmark, candidate.task_id)
        or record["task_id"] != candidate.task_id
        or any(record[k] != value for k, value in provenance.items())
        or artifact["transcript_hash"] != digest
        or artifact["source"] != provenance
        or artifact["public_task"] != candidate.input_text
        or artifact["policy_source_sha256"] != quality.SOURCE_SHA256
    ):
        raise ValueError(f"Decision/artifact/pool identity mismatch: {digest}")
    # The live process retains its imported reward-library version. Preserve its
    # recorded identity even if that external library was edited subsequently.
    reward_hash = artifact["provenance"]["policy_source_sha256"]
    if (
        not isinstance(reward_hash, str)
        or len(reward_hash) != 64
        or any(c not in "0123456789abcdef" for c in reward_hash)
    ):
        raise ValueError(f"Invalid saved reward-library identity: {digest}")
    config = quality.repeat.RepeatConfig(**artifact["config"])
    client = quality.SFTJudgeClient(config)
    votes = result["votes"]
    quorum = config.vote_quorum
    good = 0
    fields = ("verdict", "checks", "event_ids", "text_ids", "reason")
    for call in votes + [result["reviewer"]]:
        value = client.validate({k: call[k] for k in fields}, artifact["events"], artifact["actor_segments"])
        guarded = client.guard(value, artifact["events"], artifact["actor_segments"])
        if guarded["verdict"] == "good" and not call.get("error"):
            good += 1
    if (
        result.get("accepted") is not True
        or record["verdict"] != "good"
        or result["verdict"] != "good"
        or result["stage"] != "model"
        or record["stage"] != "model"
        or result.get("error")
        or record.get("unresolved_error_calls")
        or not config.reviewer_enabled
        or not 1 <= quorum <= config.vote_count
        or len(votes) != config.vote_count
        or result["vote_count"] != config.vote_count
        or record["vote_count"] != config.vote_count
        or result["vote_quorum"] != quorum
        or result["good_votes"] != record["good_votes"]
        or result["good_votes"] != good - 1
        or good - 1 < quorum
        or result.get("reviewer_called") is not True
        or record.get("reviewer_called") is not True
        or record.get("reviewer_verdict") != "good"
        or result["reviewer"]["verdict"] != "good"
        or result["reviewer"].get("error")
        or client.guard(result["reviewer"], artifact["events"], artifact["actor_segments"])["verdict"] != "good"
    ):
        raise ValueError(f"Unsupported accepted judgment: {digest}")
    return candidate, artifact_bytes, artifact["config"], artifact["provenance"]


def export_rl(stage, reference_path, reserved, scope, write_filtered):
    import pyarrow as pa
    import pyarrow.parquet as pq

    reference_path = reference_path.resolve()
    before = reference_path.stat()
    reference_sha256 = sha256(reference_path)
    identities = pq.read_table(
        reference_path, columns=["index", "extra_info.benchmark", "extra_info.task_id"]
    ).to_pylist()
    keep = [(r["benchmark"], r["task_id"]) not in reserved for r in identities]
    excluded = [r for r, remains in zip(identities, keep, strict=True) if not remains]
    sft._write_jsonl(stage / "rl_exclusion_manifest.jsonl", excluded)
    if write_filtered:
        table = pq.read_table(reference_path)
        if len(table) != len(keep):
            raise ValueError("RL reference row count changed")
        filtered = table.filter(pa.array(keep, type=pa.bool_()))
        pq.write_table(filtered, stage / "rl_train_filtered.parquet", compression="zstd")
        remaining = pq.read_table(
            stage / "rl_train_filtered.parquet", columns=["extra_info.benchmark", "extra_info.task_id"]
        ).to_pylist()
        if any((r["benchmark"], r["task_id"]) in reserved for r in remaining):
            raise ValueError("Filtered RL copy overlaps reserved SFT prompts")
    after = reference_path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or sha256(
        reference_path
    ) != reference_sha256:
        raise ValueError("RL reference changed during export")
    report = {
        "reference": str(reference_path),
        "reference_sha256": reference_sha256,
        "reference_rows": len(identities),
        "scope": scope,
        "reserved_tasks": len(reserved),
        "matched_reference_rows": len(excluded),
        "remaining_reference_rows": sum(keep),
        "remaining_task_overlap": 0,
        "reference_modified": False,
        "filtered_copy_written": write_filtered,
    }
    (stage / "rl_exclusion_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def export_snapshot(args):
    run_dir, output_dir = args.run_dir.resolve(), args.output_dir.resolve()
    if output_dir.is_relative_to(run_dir) or run_dir.is_relative_to(output_dir):
        raise ValueError("Snapshot output must be separate from the running filter directory")
    if output_dir.exists():
        raise FileExistsError(f"Snapshot already exists: {output_dir}; choose a new directory")
    if not 0 <= args.val_fraction < 1:
        raise ValueError("val-fraction must be in [0, 1)")
    selected = selection_args(args)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    lock = output_dir.with_name(f".{output_dir.name}.lock")
    lock.mkdir()
    stage = None
    try:
        stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
        records, cutoff = capture_decisions(run_dir / "quality_decisions.jsonl", stage / "decisions_at_cutoff.jsonl")
        accepted = [r for r in records if r["accepted"]]
        if not accepted:
            raise ValueError("No accepted trajectories at the captured checkpoint")
        core, relaxed, manifest = {}, {}, []
        judgment_configs, provenances = {}, {}
        with closing(sqlite3.connect((run_dir / "candidate_pool.sqlite").as_uri() + "?mode=ro", uri=True)) as db:
            fingerprint = verify_pool(db, run_dir, selected)
            possible = {tuple(json.loads(r[0])) for r in db.execute("SELECT DISTINCT task FROM candidates")}
            (stage / "quality_artifacts").mkdir()
            for record in accepted:
                candidate, artifact_bytes, config, provenance = accepted_candidate(record, db, selected, run_dir)
                key = (candidate.benchmark, candidate.task_id)
                if key in relaxed:
                    raise ValueError(f"Multiple accepted decisions for task: {key}")
                relaxed[key] = candidate
                if candidate.tool_dispatches <= selected.max_tool_dispatches:
                    core[key] = candidate
                artifact = f"quality_artifacts/{candidate.transcript_hash}.json.gz"
                (stage / artifact).write_bytes(artifact_bytes)
                manifest.append(
                    {**record, "original_artifact_path": record["artifact_path"], "artifact_path": artifact}
                )
                judgment_configs[quality.repeat.digest(config)] = config
                provenances[quality.repeat.digest(provenance)] = provenance
        gold = {k: c for k, c in core.items() if c.has_verification}
        tiers = {"core": core, "relaxed": relaxed, "gold": gold}
        counts = {}
        for tier, candidates in tiers.items():
            counts.update(sft._write_dataset_splits(stage, candidates, tier, args.val_fraction))
        quality.write_exclusions(argparse.Namespace(output_dir=stage, rl_reference_parquet=None), tiers)
        sft._write_jsonl(stage / "accepted_manifest.jsonl", manifest)
        shutil.copyfile(run_dir / "sources.json", stage / "sources.json")
        reserved = possible if args.rl_exclusion_scope == "candidate-pool" else set(relaxed)
        sft._write_jsonl(
            stage / "rl_reserved_prompt_ids.jsonl",
            [
                {"benchmark": b, "task_id": t, "prompt_id": json.dumps([b, t], separators=(",", ":"))}
                for b, t in sorted(reserved)
            ],
        )
        rl_report = export_rl(
            stage, args.rl_reference_parquet, reserved, args.rl_exclusion_scope, args.write_filtered_rl
        )
        # Validate files using their serialized identities, not only in-memory selection.
        import pyarrow.parquet as pq

        for tier, candidates in tiers.items():
            exported = []
            for split in ("train", "val"):
                exported.extend(
                    pq.read_table(stage / f"{tier}_{split}.parquet", columns=["benchmark", "task_id"]).to_pylist()
                )
            if len(exported) != len(candidates) or {(r["benchmark"], r["task_id"]) for r in exported} != set(
                candidates
            ):
                raise ValueError(f"Exported {tier} prompt identities differ from selection")
        report = {
            "snapshot_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_run_dir": str(run_dir),
            "source_fingerprint": fingerprint,
            "policy_source_sha256": quality.SOURCE_SHA256,
            "builder_source_sha256": sha256(Path(sft.__file__)),
            "cutoff": cutoff,
            "decision_verdict_counts": dict(collections.Counter(r["verdict"] for r in records)),
            "accepted_unique_tasks": len(relaxed),
            "candidate_pool_tasks": len(possible),
            "dataset_counts": counts,
            "val_fraction": args.val_fraction,
            "judgment_configs": list(judgment_configs.values()),
            "judge_provenances": list(provenances.values()),
            "exporter_reward_source_sha256": quality.repeat.POLICY_SOURCE_SHA256,
            "recorded_reward_source_differs_from_exporter": any(
                p["policy_source_sha256"] != quality.repeat.POLICY_SOURCE_SHA256 for p in provenances.values()
            ),
            "independent_audit_verified": False,
            "rl_exclusion": rl_report,
            "files": {str(p.relative_to(stage)): sha256(p) for p in sorted(stage.rglob("*")) if p.is_file()},
        }
        (stage / "snapshot_report.json").write_text(json.dumps(report, indent=2) + "\n")
        if output_dir.exists():
            raise FileExistsError(f"Snapshot output appeared during export: {output_dir}")
        stage.rename(output_dir)
        stage = None
        return report
    finally:
        if stage is not None:
            shutil.rmtree(stage)
        lock.rmdir()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="New immutable snapshot directory")
    parser.add_argument(
        "--selection-args-json", type=Path, help="Candidate filter overrides for non-default runs; fingerprint-checked"
    )
    parser.add_argument("--val-fraction", type=float, default=0)
    parser.add_argument("--rl-exclusion-scope", choices=("candidate-pool", "snapshot"), default="candidate-pool")
    parser.add_argument(
        "--rl-reference-parquet",
        type=Path,
        default=Path(sft.__file__).resolve().parents[2]
        / "stock-rl-reflect/data/swe_benchmarks/verl/swe_rebench_v2_reference_verified_snapshot_2000.parquet",
    )
    parser.add_argument(
        "--write-filtered-rl",
        action="store_true",
        help="Also export rl_train_filtered.parquet; original stays unchanged",
    )
    return parser


def main():
    report = export_snapshot(build_parser().parse_args())
    print(
        json.dumps(
            {k: report[k] for k in ("accepted_unique_tasks", "dataset_counts", "cutoff", "rl_exclusion")}, indent=2
        )
    )


if __name__ == "__main__":
    main()

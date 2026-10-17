"""Export real pilot evidence and validate independent labels before a full SFT run."""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import sqlite3
from pathlib import Path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sample_tasks(db, seed, count):
    tasks = [r[0] for r in db.execute("SELECT DISTINCT task FROM candidates")]
    tasks.sort(key=lambda t: hashlib.sha256(f"{seed}:{t}".encode()).hexdigest())
    return tasks[:count] if count else tasks


def export_evidence(pool_dir, output_dir, count, seed):
    from scripts import swe_sft_quality as quality

    output_dir.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(f"file:{pool_dir.resolve() / 'candidate_pool.sqlite'}?mode=ro", uri=True)
    stats = collections.Counter()
    records = []
    try:
        for task in sample_tasks(db, seed, count):
            digest, packed, provenance = db.execute(
                "SELECT hash,raw,provenance FROM candidates WHERE task=? ORDER BY rank LIMIT 1", (task,)
            ).fetchone()
            row = json.loads(gzip.decompress(packed))
            result = None
            events, segments, source = [], [], "unavailable"
            try:
                events, segments, source = quality.load_evidence(row)
                result = quality.failure_rule(row, events)
                if result is None:
                    rule = quality.repeat.conservative_detect(events, actor_segments=segments)
                    if rule["verdict"] == "repetitive":
                        result = {"stage": "rule", "verdict": "bad", "reason": rule["reason"], "evidence": rule}
            except (ValueError, OSError, KeyError, TypeError) as exc:
                result = {"stage": "evidence", "verdict": "uncertain", "reason": str(exc)[:600]}
            artifact = output_dir / f"{digest}.json.gz"
            artifact.write_bytes(
                gzip.compress(
                    json.dumps(
                        {
                            "transcript_hash": digest,
                            "task": task,
                            "source": json.loads(provenance),
                            "public_task": row.get("input"),
                            "rendered_output": row.get("output"),
                            "events": events,
                            "actor_segments": segments,
                            "evidence_source": source,
                            "offline_rule_result": result,
                            "model_judgment_performed": False,
                        },
                        ensure_ascii=False,
                    ).encode()
                )
            )
            record = {
                "transcript_hash": digest,
                "task": task,
                "artifact_path": str(artifact.resolve()),
                "artifact_sha256": sha256(artifact),
                "events": len(events),
                "offline_stage": result["stage"] if result else "needs_model",
                "source": json.loads(provenance),
            }
            stats[record["offline_stage"]] += 1
            records.append(record)
        (output_dir / "evidence_cases.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
        (output_dir / "evidence_report.json").write_text(
            json.dumps(
                {
                    "cases": len(records),
                    "stages": dict(stats),
                    "sample_seed": seed,
                    "source_manifest_sha256": sha256(pool_dir / "sources.json"),
                    "model_judgment_performed": False,
                    "full_filter_gate_satisfied": False,
                },
                indent=2,
            )
            + "\n"
        )
    finally:
        db.close()


def evaluate_audit(directory, labels_path):
    report_path = directory / "quality_report.json"
    decisions_path = directory / "quality_decisions.jsonl"
    pilot = json.loads(report_path.read_text())
    if pilot["pilot_cases"] < 50:
        raise ValueError("Pilot must cover at least 50 real tasks")
    decisions = [json.loads(line) for line in decisions_path.read_text().splitlines()]
    labels = [json.loads(line) for line in labels_path.read_text().splitlines()]
    by_hash = {r["transcript_hash"]: r for r in labels}
    if len(by_hash) != len(labels) or len(labels) < 50:
        raise ValueError("At least 50 unique independent labels are required")
    if set(by_hash) != {r["transcript_hash"] for r in decisions}:
        raise ValueError("Independently label every case in the pilot decisions")
    stats = collections.Counter()
    artifact_hashes = {}
    for decision in decisions:
        digest = decision["transcript_hash"]
        label = by_hash[digest]
        if (
            label.get("label") not in {"good", "bad", "uncertain"}
            or len(str(label.get("reason", "")).strip()) < 20
            or not label.get("auditor")
        ):
            raise ValueError(f"Missing independent label, auditor or substantive reason: {digest}")
        artifact_path = Path(decision["artifact_path"])
        artifact_hash = sha256(artifact_path)
        if artifact_hash != decision["artifact_sha256"] or artifact_hash != label.get("artifact_sha256"):
            raise ValueError(f"Audit artifact identity mismatch: {digest}")
        artifact = json.loads(gzip.decompress(artifact_path.read_bytes()))
        if artifact["transcript_hash"] != digest or artifact["result"]["accepted"] != decision["accepted"]:
            raise ValueError(f"Decision/artifact mismatch: {digest}")
        ids = label.get("event_ids")
        if not isinstance(ids, list) or any(type(i) is not int for i in ids):
            raise ValueError(f"Independent label needs event_ids list: {digest}")
        if not set(ids) <= {e["id"] for e in artifact["events"]}:
            raise ValueError(f"Independent label cites missing events: {digest}")
        if label["label"] == "good" and not ids:
            raise ValueError(f"GOOD independent labels need execution evidence: {digest}")
        if artifact["result"].get("error"):
            raise ValueError("Pilot has model errors; resolve and rerun before full filtering")
        if decision["accepted"] and label["label"] != "good":
            raise ValueError(f"False or unsubstantiated acceptance in pilot: {digest}")
        stats["reviewed_cases"] += 1
        stats["stage_" + decision["stage"]] += 1
        stats["independent_" + label["label"]] += 1
        stats["accepted"] += int(decision["accepted"])
        stats["false_rejections"] += int(not decision["accepted"] and label["label"] == "good")
        artifact_hashes[digest] = artifact_hash
    if not stats["stage_model"]:
        raise ValueError("Audit must include real model judgments")
    return {
        "passed": True,
        "source_fingerprint": pilot["source_fingerprint"],
        "config_key": pilot["config_key"],
        "pilot_report_sha256": sha256(report_path),
        "decisions_sha256": sha256(decisions_path),
        "labels_path": str(labels_path.resolve()),
        "labels_sha256": sha256(labels_path),
        "audit_logic_sha256": sha256(Path(__file__)),
        "artifact_hashes": artifact_hashes,
        "stats": dict(stats),
    }


def validate_audit(directory):
    saved = json.loads((directory / "audit_report.json").read_text())
    current = evaluate_audit(directory, Path(saved["labels_path"]))
    if current != saved:
        raise ValueError("Pilot audit changed since validation; repeat the audit check")
    return current


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    evidence = sub.add_parser("evidence", help="Export public real-case evidence without calling the model")
    evidence.add_argument("--pool-dir", type=Path, required=True)
    evidence.add_argument("--output-dir", type=Path, required=True)
    evidence.add_argument("--cases", type=int, default=60)
    evidence.add_argument("--sample-seed", default="sft-pilot-v1")
    audit = sub.add_parser("check", help="Compare independent labels with completed real model pilot")
    audit.add_argument("--pilot-dir", type=Path, required=True)
    audit.add_argument("--labels", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "evidence":
        if args.cases < 50:
            parser.error("Audit evidence needs at least 50 cases")
        export_evidence(args.pool_dir, args.output_dir, args.cases, args.sample_seed)
    else:
        report = evaluate_audit(args.pilot_dir, args.labels)
        (args.pilot_dir / "audit_report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report["stats"], indent=2))


if __name__ == "__main__":
    main()

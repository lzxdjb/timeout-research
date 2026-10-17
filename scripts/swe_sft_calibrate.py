"""Replay a small labeled diagnostic set through the unchanged SFT judge policy.

Labels and prior model judgments are never sent to the judge. This deliberately
does not create the audit_report.json required to start full filtering.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import dataclasses
import gzip
import json
from pathlib import Path

from scripts import swe_sft_quality as quality
from scripts.swe_sft_filter_audit import label_metrics, sha256


def load_cases(manifest):
    cases = []
    seen = set()
    for line in manifest.read_text().splitlines():
        label = json.loads(line)
        digest = label["transcript_hash"]
        if digest in seen or label["label"] not in {"good", "bad", "uncertain"}:
            raise ValueError("Duplicate case or invalid independent label")
        seen.add(digest)
        path = Path(label["artifact_path"])
        if not path.is_absolute():
            path = manifest.parent / path
        if sha256(path) != label["artifact_sha256"]:
            raise ValueError(f"Calibration artifact hash mismatch: {digest}")
        original = json.loads(gzip.decompress(path.read_bytes()))
        if original["transcript_hash"] != digest:
            raise ValueError(f"Calibration transcript mismatch: {digest}")
        # Allowlist public evidence; exclude old judgments, messages and labels.
        evidence = {k: original[k] for k in ("public_task", "events", "actor_segments")}
        if not set(label["event_ids"]) <= {e["id"] for e in evidence["events"]}:
            raise ValueError(f"Calibration label cites unavailable events: {digest}")
        cases.append((label, evidence))
    if not cases:
        raise ValueError("Empty calibration set")
    return cases


async def run(args):
    config = quality.judge_config(args)
    cases = load_cases(args.cases)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Use a new empty calibration output directory for each model/configuration")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    quality.write_run_status(args, "calibration_running")
    try:
        client = quality.SFTJudgeClient(config)
        await client.preflight()
        execution = collections.Counter()
        decisions = []
        with (args.output_dir / "calibration_decisions.jsonl").open("w") as stream:
            for label, evidence in cases:
                result = await client.judge_consensus(
                    evidence["public_task"], evidence["events"], actor_segments=evidence["actor_segments"]
                )
                execution.update(quality.execution_metrics(result))
                artifact = args.output_dir / f"{label['transcript_hash']}.json.gz"
                artifact.write_bytes(
                    gzip.compress(
                        json.dumps(
                            {
                                **evidence,
                                "transcript_hash": label["transcript_hash"],
                                "result": result,
                                "config": dataclasses.asdict(config),
                                "provenance": client.provenance(),
                                "policy_source_sha256": quality.SOURCE_SHA256,
                            },
                            ensure_ascii=False,
                        ).encode()
                    )
                )
                decision = {
                    "case": label["case"],
                    "task_id": label["task_id"],
                    "transcript_hash": label["transcript_hash"],
                    "expected_label": label["label"],
                    "verdict": result["verdict"],
                    "accepted": result["accepted"],
                    "good_votes": result["good_votes"],
                    "reason": result["reason"],
                    "artifact_path": str(artifact.resolve()),
                    "artifact_sha256": sha256(artifact),
                }
                decisions.append(decision)
                stream.write(json.dumps(decision, ensure_ascii=False) + "\n")
                stream.flush()
                print(json.dumps(decision, ensure_ascii=False), flush=True)
        report = {
            "diagnostic_only": True,
            "full_filter_gate_satisfied": False,
            "cases": len(cases),
            "manifest_sha256": sha256(args.cases),
            "policy_source_sha256": quality.SOURCE_SHA256,
            "config": dataclasses.asdict(config),
            "provenance": client.provenance(),
            "execution": dict(execution),
            "label_comparison": label_metrics(decisions, [label for label, _ in cases]),
            "verdict_matches": sum(d["verdict"] == d["expected_label"] for d in decisions),
        }
        (args.output_dir / "calibration_report.json").write_text(json.dumps(report, indent=2) + "\n")
        quality.write_run_status(args, "calibration_complete", full_filter_gate_satisfied=False)
        return report
    except BaseException as exc:
        quality.write_run_status(
            args,
            "interrupted" if isinstance(exc, (KeyboardInterrupt, asyncio.CancelledError)) else "failed",
            error_type=type(exc).__name__,
            error=str(exc)[:1000],
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    quality.add_arguments(parser)
    parser.add_argument("--cases", type=Path, required=True, help="Independent calibration labels JSONL")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(run(args)), indent=2))


if __name__ == "__main__":
    main()

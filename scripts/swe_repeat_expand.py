"""Freeze an expanded precision suite from reviewed captured trajectories.

No model predictions, legacy repetition counts, or hidden outcomes become labels.
The source manifest supplies independent reviews; all captured artifacts are hashed
before migration to the new public-event serializer. Never overwrite a frozen set.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
from pathlib import Path

from scripts.swe_repeat_regression import file_digest, partition, write_json
from scripts.swe_repeat_contracts import success_repeat_judge_contracts, independent_stage2_contracts
from recipe.swe_agent import repetition_reward as policy


def canonical_task_key(case: dict) -> str:
    """Unify historical repo/task keys with newer instance-only captures."""
    return case.get("task") or case["task_key"].rsplit("\x1f", 1)[-1]


def build(existing: Path, reviews: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError("Refusing to overwrite a frozen evaluation set")
    source_labels = json.loads((existing / "labels.json").read_text())
    source_manifest = json.loads((existing / "manifest.json").read_text())
    if (source_manifest["fixtures_sha256"] != file_digest(existing / "fixtures.jsonl.gz")
            or source_manifest["labels_sha256"] != file_digest(existing / "labels.json")):
        raise ValueError("Existing frozen set integrity failure")
    with gzip.open(existing / "fixtures.jsonl.gz", "rt") as stream:
        cases = [json.loads(line) for line in stream]
    labels = source_labels["cases"][:]
    sources = {str(existing / n): file_digest(existing / n) for n in ("fixtures.jsonl.gz", "labels.json")}
    sources[str(reviews)] = file_digest(reviews)
    for review in json.loads(reviews.read_text())["cases"]:
        if review["expected"] not in {"clean", "repetitive", "uncertain"} or not review["reason"]:
            raise ValueError("Every expansion case requires an independent review")
        path = Path(review["source"])
        if file_digest(path) != review["sha256"]:
            raise ValueError("Captured artifact hash mismatch")
        if review.get("event_fixture"):
            source_case = next(c for c in json.loads(path.read_text()) if c["id"] == review["event_fixture"])
            artifact = {"events": source_case["events"], "original_public_context": "",
                        "version": policy.SERIALIZER_VERSION, "messages": []}
        else:
            with gzip.open(path, "rt") as stream:
                artifact = json.load(stream)
        if artifact["version"] not in {"repeat-public-events-v2", policy.SERIALIZER_VERSION}:
            raise ValueError("Unsupported captured serializer")
        task_key = review["task"]
        cases.append({"id": review["id"], "task": task_key, "task_key": task_key, "partition": partition(task_key),
                      "public_events": artifact["events"], "public_context": artifact["original_public_context"],
                      "capture_source": str(path), "capture_sha256": review["sha256"],
                      "capture_serializer": artifact["version"],
                      "capture_messages_sha256": policy.digest(artifact["messages"]),
                      "raw": {"raw_score": review.get("raw_score")}})
        labels.append({k: review[k] for k in ("id", "expected", "reason")})
        sources[str(path)] = review["sha256"]
    for contract in success_repeat_judge_contracts() + independent_stage2_contracts():
        task_key = "synthetic:" + contract["id"]
        if any(case["id"] == task_key for case in cases):
            continue  # Preserve frozen identities and their independent labels.
        events = [{"index": i, "batch": i, "name": name, "arguments": args, "response": response}
                  for i, (name, args, response) in enumerate(contract["calls"])]
        cases.append({"id": task_key, "task_key": task_key, "partition": partition(task_key), "events": events,
                      "raw": {"input": contract["context"], "output": "", "raw_score": 1}})
        labels.append({"id": task_key, "expected": contract["expected"], "reason": contract["reason"]})
    if len({c["id"] for c in cases}) != len(cases):
        raise ValueError("Duplicate case identity")
    # Repartition every occurrence by task to prevent checkpoint/outcome leakage.
    for case in cases:
        case["task_key"] = canonical_task_key(case)
        case["partition"] = partition(case["task_key"])
    output.mkdir(parents=True)
    with gzip.open(output / "fixtures.jsonl.gz", "wt") as stream:
        for case in cases:
            stream.write(json.dumps(case) + "\n")
    write_json(output / "labels.json", {"policy_version": policy.POLICY_VERSION,
        "reviewer": "Code-agent public-evidence reviews; human adjudication pending", "cases": labels})
    manifest = {"policy_version": policy.POLICY_VERSION, "cases": len(cases),
                "fixtures_sha256": file_digest(output / "fixtures.jsonl.gz"),
                "labels_sha256": file_digest(output / "labels.json"), "sources": sources,
                "labels": dict(collections.Counter(c["expected"] for c in labels)),
                "partition_policy": "Task-disjoint hash split; all checkpoints of a task stay together",
                "production_ready": False}
    write_json(output / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--existing-dir", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = build(args.existing_dir, args.reviews, args.output_dir)
    print(json.dumps({k: v for k, v in report.items() if k != "sources"}))


if __name__ == "__main__":
    main()

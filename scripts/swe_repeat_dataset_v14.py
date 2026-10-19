"""Freeze rendering regressions and step-16 budget coverage without model calls."""
from __future__ import annotations

import argparse
import collections
import copy
import gzip
import json
from pathlib import Path

from scripts.swe_repeat_regression import digest, file_digest, partition, replay_evidence, write_json


def contracts():
    from recipe.swe_agent import repetition_reward as p
    for name in ("successful_different_edits", "explicit_noop_loop", "different_search_targets"):
        events, segments = [], []
        for i in range(24):
            if name == "explicit_noop_loop":
                tool, args, result = "Edit", {"file_path": "a.py", "old_string": "x", "new_string": "x"}, p.NOOP_EDIT_RESULT
                thought = "Try this identical replacement again."
            elif name == "successful_different_edits":
                tool = "Edit"
                args = {"file_path": "a.py", "old_string": f"value_{i} = 0", "new_string": f"value_{i} = 1"}
                result, thought = "Edited a.py (1 replacement).", f"Update independent declaration value_{i}."
            else:
                tool, args, result = "Grep", {"path": f"module_{i}.py", "pattern": f"handler_{i}"}, "No matches."
                thought = f"Check distinct module {i} for its handler before selecting the integration point."
            segments.append({"id": f"a{i + 1}", "after_event_id": i, "text": thought, "captured": True})
            event = {"id": i + 1, "name": tool, "arguments": args, "completed": True,
                     "serial": True, "observation_status": "observed", **p.capture_observation(result)}
            if name == "explicit_noop_loop":
                event.update(p.noop_edit_evidence(tool, args, result))
            events.append(event)
        yield name, events, segments, "repetitive" if name == "explicit_noop_loop" else "clean"
    missing = copy.deepcopy(events)
    for event in missing:
        for key in ("observation", "result", "feedback"):
            event.pop(key, None)
        event["observation_status"] = "missing"
    yield "missing_results", missing, segments, "uncertain"


def build(base: Path, training: Path, audit_report: Path, output: Path):
    from recipe.swe_agent import repetition_reward as p
    if output.exists():
        raise ValueError("Refusing to overwrite a frozen suite")
    manifest = json.loads((base / "manifest.json").read_text())
    if any(file_digest(base / name) != manifest[key] for name, key in
           [("fixtures.jsonl.gz", "fixtures_sha256"), ("labels.json", "labels_sha256")]):
        raise ValueError("Base suite integrity mismatch")
    with gzip.open(base / "fixtures.jsonl.gz", "rt") as stream:
        cases = list(map(json.loads, stream))
    labels = json.loads((base / "labels.json").read_text())["cases"]
    parts = {case["task_key"]: case["partition"] for case in cases}
    training_hash = file_digest(training)
    native_count = 0
    for line, row in enumerate(map(json.loads, training.open()), 1):
        if not row.get("repeat_reward_audit_path") or row.get("is_padding"):
            continue
        # Replay now, so broken hashes/old rendering fail before freezing a suite.
        replay_evidence(row, p)
        task = row["task_id"]
        cid = f"native:render-v14:step16:{line}"
        pointer = {key: row[key] for key in ("repeat_reward_audit_path", "repeat_reward_audit_sha256")}
        cases.append({"id": cid, "task_key": task, "partition": parts.setdefault(task, partition(task)),
                      **pointer, "raw": {"raw_score": row.get("raw_score")},
                      "source": {"path": str(training.resolve()), "line": line,
                                 "source_sha256": training_hash, "row_sha256": digest(row)}})
        labels.append({"id": cid, "expected": None, "annotation_status": "unadjudicated",
                       "scope": "full_trajectory", "reason": "Step-16 rendering/budget coverage; previous model votes are not gold labels."})
        native_count += 1
    reviewed = json.loads(audit_report.read_text())
    for item in reviewed["audited_evidence_problems"]:
        path = Path(item["audit_path"])
        pointer = {"repeat_reward_audit_path": str(path), "repeat_reward_audit_sha256": file_digest(path)}
        events, _, _, segments = replay_evidence(pointer, p)
        lo, hi = min(item["event_ids"]), max(item["event_ids"])
        # Keep all intervening events/thinking, with stable evidence IDs. A bad
        # justification does not establish a whole-trajectory clean label.
        before = [s for s in segments or [] if s["after_event_id"] < lo]
        excerpt_segments = copy.deepcopy(before[-1:] + [s for s in segments or [] if lo <= s["after_event_id"] <= hi])
        if excerpt_segments and excerpt_segments[0]["after_event_id"] < lo:
            excerpt_segments[0]["after_event_id"] = 0
        cid = "native:render-v14:protected:" + item["uid"]
        task = item["task_id"]
        cases.append({"id": cid, "task_key": task, "partition": parts.setdefault(task, partition(task)),
                      "public_events": [e for e in events if lo <= e["id"] <= hi],
                      "actor_segments": excerpt_segments, "public_context": "", "raw": {"raw_score": None},
                      "source": {**pointer, "view": "contiguous_audited_excerpt", "first_event_id": lo, "last_event_id": hi}})
        labels.append({"id": cid, "expected": "uncertain", "scope": "audited_excerpt",
                       "reason": "Protect against unsupported penalty: " + item["inspection"]})
    for name, events, segments, expected in contracts():
        cid = "synthetic:render-v14:" + name
        cases.append({"id": cid, "task_key": cid, "partition": partition(cid), "public_events": events,
                      "actor_segments": segments, "public_context": "", "raw": {"raw_score": 1}})
        labels.append({"id": cid, "expected": expected, "scope": "synthetic_full_trajectory",
                       "reason": "Matched rendering control: " + name})
    if len({c["id"] for c in cases}) != len(cases) or {c["id"] for c in cases} != {x["id"] for x in labels}:
        raise ValueError("Fixture/label identity mismatch")
    groups = collections.defaultdict(set)
    for case in cases:
        groups[case["task_key"]].add(case["partition"])
    if any(len(group) != 1 for group in groups.values()):
        raise ValueError("Task partition leakage")
    output.mkdir(parents=True)
    with (output / "fixtures.jsonl.gz").open("wb") as binary:
        with gzip.GzipFile(fileobj=binary, mode="wb", mtime=0) as stream:
            for case in cases:
                stream.write((json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n").encode())
    write_json(output / "labels.json", {"policy_version": p.POLICY_VERSION, "cases": labels,
               "reviewer": "Code-agent reviewed excerpts and synthetic controls; full native traces unadjudicated."})
    result = {"policy_version": p.POLICY_VERSION, "serializer_version": p.SERIALIZER_VERSION,
              "cases": len(cases), "base_cases": len(cases) - native_count - 8, "native_additions": native_count,
              "protected_excerpts": 4, "synthetic_controls": 4,
              "label_counts": dict(collections.Counter(str(x["expected"]) for x in labels)),
              "base_manifest_sha256": file_digest(base / "manifest.json"), "training_sha256": training_hash,
              "audit_report_sha256": file_digest(audit_report), "builder_sha256": file_digest(Path(__file__)),
              "fixtures_sha256": file_digest(output / "fixtures.jsonl.gz"), "labels_sha256": file_digest(output / "labels.json")}
    write_json(output / "manifest.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-suite", type=Path, default=Path("analysis/reward_model_trajectory_v13_20261016/frozen_task_disjoint"))
    parser.add_argument("--training", type=Path, default=Path("rollout_data/reward_model_train_v4_more_timeout/16.jsonl"))
    parser.add_argument("--audit-report", type=Path, default=Path("analysis/reward_rendering_20261019/report.json"))
    parser.add_argument("--output", type=Path, default=Path("analysis/reward_model_rendering_v14_20261019/frozen_task_disjoint"))
    args = parser.parse_args()
    print(json.dumps(build(args.base_suite, args.training, args.audit_report, args.output), indent=2))


if __name__ == "__main__":
    main()

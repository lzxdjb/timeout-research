"""Freeze only supported full-trajectory labels and explicitly audited controls."""

import argparse
import collections
import gzip
import json
from pathlib import Path

from scripts.swe_repeat_regression import file_digest, partition, replay_evidence, write_json


NATIVE_CONTROLS = {
    318: "Three calls: missing-file edit, successful file creation, one duplicate Write. One duplicate is not sustained repetition.",
    405: "Full 23-call trace: investigation, one two-file edit/revert, revised delAsync implementation, successful build. No sustained loop.",
    70: "Full 19-call trace: changing repository searches, proposed standalone fix/documentation, failed path invocation, corrected cleanup. Poor repair quality does not establish sustained repetition.",
}


def build(base, report, output):
    from recipe.swe_agent import repetition_reward as p

    if output.exists():
        raise ValueError("Refusing to overwrite a frozen suite")
    manifest = json.loads((base / "manifest.json").read_text())
    for name, key in (("fixtures.jsonl.gz", "fixtures_sha256"), ("labels.json", "labels_sha256")):
        if file_digest(base / name) != manifest[key]:
            raise ValueError("Base integrity mismatch")
    labels = {x["id"]: x for x in json.loads((base / "labels.json").read_text())["cases"]}
    results = {x["id"]: x for x in json.loads(report.read_text())["rows"]}
    cases, selected, removed = [], [], []
    with gzip.open(base / "fixtures.jsonl.gz", "rt") as stream:
        for case in map(json.loads, stream):
            label, row = labels[case["id"]], results[case["id"]]
            reason = (
                "unadjudicated" if label["expected"] is None else
                "ambiguous_event_alignment" if row["incomplete_evidence"] else
                "partial_tool_labels_not_full_trajectory_labels"
                if label.get("scope") not in {"full_trajectory", "synthetic_full_trajectory", "audited_excerpt"}
                else None
            )
            if reason:
                removed.append({"id": case["id"], "reason": reason})
                continue
            # Verify recorded evidence and historical request hashes before reuse.
            replay_evidence(case, p)
            cases.append(case)
            selected.append(label)
    parts = {c["task_key"]: c["partition"] for c in cases}
    for number, reason in NATIVE_CONTROLS.items():
        row = results[f"native:render-v14:step16:{number}"]
        pointer = {k: row[k] for k in ("repeat_reward_audit_path", "repeat_reward_audit_sha256")}
        events, _, _, _ = replay_evidence(pointer, p)
        expected_events = {318: 3, 405: 23, 70: 19}[number]
        if len(events) != expected_events:
            raise ValueError("Audited native control changed")
        task = row["task_key"]
        cases.append({"id": row["id"], "task_key": task, "partition": parts.setdefault(task, partition(task)),
                      **pointer, "raw": {"raw_score": row.get("raw_success")},
                      "source": {"report": str(report.resolve()), "view": "complete_audited_native_trace"}})
        selected.append({"id": row["id"], "expected": "clean", "scope": "full_trajectory",
                         "annotation_status": "code_agent_audited", "reason": reason})
    assert len({c["id"] for c in cases}) == len(cases) == len(selected)
    output.mkdir(parents=True)
    with (output / "fixtures.jsonl.gz").open("wb") as binary:
        with gzip.GzipFile(fileobj=binary, mode="wb", mtime=0) as stream:
            for case in cases:
                stream.write((json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n").encode())
    write_json(output / "labels.json", {"policy_version": p.POLICY_VERSION, "cases": selected,
               "reviewer": "Inherited explicit full-trajectory annotations; three code-agent-audited native controls. No model votes used as labels."})
    write_json(output / "excluded.json", removed)
    result = {"policy_version": p.POLICY_VERSION, "cases": len(cases),
              "label_counts": dict(collections.Counter(x["expected"] for x in selected)),
              "scope_counts": dict(collections.Counter(x["scope"] for x in selected)),
              "exclusions": dict(collections.Counter(x["reason"] for x in removed)),
              "native_controls_added": len(NATIVE_CONTROLS),
              "base_manifest_sha256": file_digest(base / "manifest.json"), "report_sha256": file_digest(report),
              "builder_sha256": file_digest(Path(__file__)),
              "fixtures_sha256": file_digest(output / "fixtures.jsonl.gz"), "labels_sha256": file_digest(output / "labels.json"),
              "limitation": "Known regression set, not an unseen accuracy estimate. Protected uncertain controls require abstention, not a clean verdict. Native token proofs are not reconstructed from decoded dumps."}
    write_json(output / "manifest.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-suite", type=Path, default=Path("analysis/reward_model_rendering_v14_20261019/frozen_task_disjoint"))
    parser.add_argument("--report", type=Path, default=Path("analysis/reward_vote_sweep/v14_full_n8_q7_20261019T061815_2526802/report.json"))
    parser.add_argument("--output", type=Path, default=Path("analysis/reward_model_curated_v15_20261019/frozen_task_disjoint"))
    args = parser.parse_args()
    print(json.dumps(build(args.base_suite, args.report, args.output), indent=2))


if __name__ == "__main__":
    main()

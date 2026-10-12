"""Repeat strict reference filtering and pinned native shadow for SWE-verified.

This is an operator-only offline tool. It does not change RL reward policies,
source parquet rows, or the dataset converter. Both parsers see the same fresh
baseline/gold executions; every selected row is audited in every main round.
"""

from __future__ import annotations

import argparse
import fcntl
import gzip
import importlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path

# Importing the existing audit also installs the sibling recipe import path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import swe_tool_online_benchmark as audit
from scripts.swe_verified_native import native_module, native_score, verify_vendor

VERSION = "swe-verified-repeat-filter-v1"
TRANSIENT = {"transport_failure", "admission_or_transport_failure", "reward_transport_or_service_failure"}


def inventory(dataset: Path, mappings: list[str], seconds: float) -> list[dict]:
    import pyarrow.parquet as pq
    from recipe.swe_agent.test_results import validate_official_test_contract
    _, cases = audit.audit_inventory([dataset], mappings, profile="validation")
    row = 0
    for batch in pq.ParquetFile(dataset).iter_batches(batch_size=256, columns=["reward_model"]):
        for entry in batch.to_pylist():
            truth = entry["reward_model"]["ground_truth"]
            truth = json.loads(truth) if isinstance(truth, str) else truth
            if truth.get("benchmark") != "swe_bench_verified" or truth.get("expected") != "all_official_tests_pass":
                raise ValueError(f"row {row}: only SWE-verified reference tests are supported")
            row += 1
    seen = set()
    for case in cases:
        item, task = case["inventory"], case["task"]
        if item["issues"]:
            raise ValueError(f"{item.get('task_id')}: invalid source artifacts: {item['issues']}")
        if task["task_id"] in seen:
            raise ValueError("duplicate task identity")
        seen.add(task["task_id"])
        raw_path = audit.audit_mapped_path(task["artifact_paths"]["raw_example"], mappings)
        raw = json.loads(raw_path.read_text())
        contract = task.get("official_test_contract", {})
        validate_official_test_contract(contract)
        # Cross-check the original benchmark labels, without strict identity aliases.
        for raw_key, key in [("FAIL_TO_PASS", "fail_to_pass"), ("PASS_TO_PASS", "pass_to_pass")]:
            labels = raw[raw_key]
            labels = json.loads(labels) if isinstance(labels, str) else labels
            if labels != contract[key]:
                raise ValueError("source/native test labels disagree: " + task["task_id"])
        if (raw["instance_id"] != task["task_id"] or raw["repo"] != task["repo"]
                or raw["base_commit"] != task["base_sha"] or contract.get("rebench_scoring_version")):
            raise ValueError("requires matching SWE-verified original task and labels")
        module, _ = native_module()
        if task["repo"] not in module["MAP_REPO_TO_PARSER_PY"]:
            raise ValueError("unsupported Verified native repository: " + task["repo"])
        item["artifacts"]["raw_example"] = {"path": str(raw_path), "sha256": audit.file_sha256(raw_path)}
        payload = audit.audit_task(task, seconds)
        if not payload.get("evaluator_test_commands"):
            raise ValueError("structured evaluator runners required: " + task["task_id"])
        case["payload"] = payload
        case["context"] = {"task_sha256": audit.digest(payload), "contract_sha256": audit.digest(contract),
                           "artifacts": {k: v["sha256"] for k, v in item["artifacts"].items()}}
    if not cases:
        raise ValueError("empty dataset")
    return cases


def runner_inputs(task: dict, phases: dict) -> tuple[list[dict], list[str]]:
    """Reparse complete raw logs/XML, excluding preparation and shell envelopes."""
    from recipe.swe_agent.test_results import official_test_outcomes, runner_output_after_marker
    specs = task["evaluator_test_commands"]
    full = phases.get("test_commands", "")
    sections = re.split(r"(?=^exit_code: -?\d+\ncommand: )", full, flags=re.M)
    sections = [s for s in sections if s.strip()]
    if len(sections) != len(specs):
        raise ValueError("incomplete_runner_envelopes")
    evidence, logs = [], []
    for i, (spec, section) in enumerate(zip(specs, sections)):
        retained = json.loads(phases[f"runner_evidence_{i}"])
        markers = re.findall(r"^SWE_RUNNER_LAUNCHED_[a-f0-9]{32}$", section, re.M)
        code = int(re.match(r"exit_code: (-?\d+)\n", section)[1])
        if len(markers) != 1 or retained["command"] != spec["run"] or retained["exit_code"] != code:
            raise ValueError("runner_launch_or_command_unverified")
        text = runner_output_after_marker(section, markers[0])
        junit = phases.get(f"junit_xml_{i}") if retained.get("junit_requested") else None
        parsed = official_test_outcomes(spec["run"], text, exit_code=code, junit_xml=junit)
        if parsed != retained["evidence"]:
            raise ValueError("strict_raw_evidence_mismatch")
        evidence.append(parsed)
        logs.append(text)
    return evidence, logs


def compare_arm(arm: dict, case: dict, roots: list[str], output: Path) -> dict:
    from scripts.diagnose_trajectories import collect_private_evaluator
    from recipe.swe_agent.test_results import score_official_tests
    result = {"request_id": arm.get("request_id"), "classification": arm.get("classification"),
              "cleanup_verified": arm.get("cleanup", {}).get("verified") is True,
              "strict": {"valid": False}, "native": {"valid": False}}
    if not result["cleanup_verified"]:
        result["reason"] = "cleanup_unverified"
        return result
    metadata = arm.get("reward", {}).get("metadata", {})
    failure_kind = (metadata.get("failure_kind") or (arm.get("claim", {}).get("metadata") or {}).get("failure_kind"))
    positively_transient = (arm.get("classification") == "transport_failure"
                            or arm.get("classification") in TRANSIENT and failure_kind in {
                                "transport_failure", "admission_timeout", "provisioning_transport_failure"})
    if positively_transient and not metadata.get("official_test_scoring_version") and not metadata.get("private_evaluator_artifact_path"):
        result["transient"] = True
        result["reason"] = arm["classification"]
        return result
    try:
        if (arm.get("exception") or not arm.get("claim", {}).get("ok")
                or arm.get("reward", {}).get("ok") is not True
                or not audit.audit_response_ok(arm.get("base", {}))
                or arm["mode"] == "gold" and not audit.audit_response_ok(arm.get("patch", {}))
                or metadata.get("hidden_runner_provenance") != "launch_marker_v1"
                or metadata.get("hidden_tests_started") is not True
                or any(metadata.get(k) is False for k in ["evaluator_preparation_ok", "hidden_patch_apply_ok",
                                                          "verifier_setup_ok", "verifier_staging_ok"])):
            raise ValueError("execution_preconditions_unverified")
        private = collect_private_evaluator({"task_id": case["task"]["task_id"], "reward_evidence": metadata}, roots, output)
        result["private_evaluator"] = private
        if private.get("status") != "verified" or private.get("logs_complete") is not True:
            raise ValueError("unverified_or_incomplete_private_logs")
        path = Path(private["private_copy"])
        artifact = json.loads(path.read_text())
        if artifact.get("request_id") != arm["request_id"]:
            raise ValueError("private_evidence_request_mismatch")
        phases = {k: (gzip.decompress((path.parent / p["full_log"]["file"]).read_bytes()).decode()
                      if p.get("full_log") else p["output"]) for k, p in artifact["phases"].items()}
        evidence, logs = runner_inputs(case["payload"], phases)
        contract = case["task"]["official_test_contract"]
        # Preserve the current Verified service semantics (exit gating only if
        # requested by the contract); extra nonrequired test failures may exist.
        score = score_official_tests(contract, evidence, len(case["payload"]["test_commands"]))
        private_score = score.pop("_private_official_results")
        retained_score = json.loads(phases["official_tests"])
        recorded = retained_score.get("score")
        # Historical Verified private artifacts omitted the public score. Their
        # full resolved/evidence/missing data and response metadata still bind
        # the replay; current artifacts must agree with both copies of score.
        strict_fields = {k: v for k, v in score.items() if k.startswith("official_")}
        if (any(retained_score.get(k) != v for k, v in private_score.items())
                or any(metadata.get(k) != v or recorded is not None and recorded.get(k) != v
                       for k, v in strict_fields.items())
                or arm["reward"].get("score") != int(score["official_tests_success"])):
            raise ValueError("audit_service_scoring_mismatch")
        result["strict"] = {"valid": True, "complete": score["official_tests_complete"],
                            "reason": score["official_tests_reason"],
                            "fail_to_pass_failed": score["official_fail_to_pass_failed"],
                            "pass_to_pass_complete": score["official_pass_to_pass_complete"],
                            "all_required_passed": score["official_tests_success"],
                            "score": int(score["official_tests_success"])}
        result["native"] = {"valid": True, **native_score(case["task"]["repo"], logs, contract)}
        result["parser_input_sha256"] = [audit.digest(s) for s in logs]
    except (ValueError, KeyError, TypeError, OSError) as exc:
        result["reason"] = str(exc)[:1000]
    return result


def attempt_state(attempt: dict, parser: str) -> str:
    """A transient arm cannot hide a parser failure in the other arm."""
    arms = attempt.get("arms", {})
    if set(arms) != {"baseline", "gold"} or attempt.get("data_issues"):
        return "veto"
    transient = False
    for mode, arm in arms.items():
        if not arm.get("cleanup_verified"):
            return "veto"
        if arm.get("transient"):
            transient = True
            continue
        score = arm.get(parser, {})
        if not score.get("valid") or not score.get("complete"):
            return "veto"
        if mode == "baseline":
            if score.get("fail_to_pass_failed", 0) <= 0 or not score.get("pass_to_pass_complete"):
                return "veto"
        elif not score.get("all_required_passed"):
            return "veto"
    return "transient" if transient else "verified"


def qualification(attempts: list[dict], repeats: int) -> dict:
    requests, image = set(), None
    for attempt in attempts:
        current = [a.get("request_id") for a in attempt.get("arms", {}).values()]
        if len(current) != len(set(current)) or any(r in requests for r in current if r):
            raise ValueError("repeat reuses an execution request")
        requests.update(r for r in current if r)
        if attempt.get("image_id"):
            if image and image != attempt["image_id"]:
                raise ValueError("repeat Docker image identity changed")
            image = attempt["image_id"]
        if any(attempt_state(attempt, p) == "verified" for p in ("strict", "native")) and not attempt.get("image_id"):
            raise ValueError("verified repeat lacks immutable image identity")
    result = {}
    for parser in ("strict", "native"):
        states = [attempt_state(a, parser) for a in attempts]
        result[parser] = {"qualified": states.count("verified") >= repeats and "veto" not in states,
                          "clean_runs": states.count("verified"), "states": states,
                          "retry_needed": "veto" not in states and states.count("verified") < repeats}
    return result


def export(dataset: Path, output: Path, records: list[dict], context: dict, repeats: int) -> dict:
    """Stream unchanged source rows, including a valid empty parquet if none qualify."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    selected = []
    for record in records:
        proof = qualification(record["attempts"], repeats)
        if proof != record["qualification"]:
            raise ValueError("repeat proof changed")
        if proof["strict"]["qualified"]:
            selected.append(record["row"])
    receipt_path = output.with_suffix(output.suffix + ".audit.json")
    receipt = {"version": VERSION, "source_sha256": context["dataset_sha256"],
               "context_sha256": audit.digest(context), "retained_row_indices": sorted(selected),
               "rows": len(selected), "repeats": repeats, "standard": "strict",
               "records_sha256": audit.digest(records), "retained_rows_unchanged": True}
    if output.exists() or receipt_path.exists():
        if not output.exists() or not receipt_path.exists():
            raise ValueError("incomplete export exists; choose a fresh output path")
        previous = json.loads(receipt_path.read_text())
        if any(previous.get(k) != v for k, v in receipt.items()) or previous.get("output_sha256") != audit.file_sha256(output):
            raise ValueError("refusing to overwrite a different export or receipt")
        return previous
    if output.resolve() == dataset.resolve() or audit.file_sha256(dataset) != context["dataset_sha256"]:
        raise ValueError("source dataset changed or output overlaps source")
    output.parent.mkdir(parents=True, exist_ok=True)
    parquet = pq.ParquetFile(dataset)
    with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        selected_set, offset = set(selected), 0
        with pq.ParquetWriter(temporary, parquet.schema_arrow) as writer:
            for batch in parquet.iter_batches(batch_size=64):
                keep = [i for i in range(batch.num_rows) if offset + i in selected_set]
                offset += batch.num_rows
                if keep:
                    writer.write_table(pa.Table.from_batches([batch]).take(pa.array(keep, type=pa.int64())))
        check = pq.ParquetFile(temporary)
        if check.metadata.num_rows != len(selected) or not check.schema_arrow.equals(parquet.schema_arrow, check_metadata=True):
            raise ValueError("export schema/count mismatch")
        if audit.file_sha256(dataset) != context["dataset_sha256"]:
            raise ValueError("dataset changed during export")
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    receipt["output_sha256"] = audit.file_sha256(output)
    audit.audit_write_json(receipt_path, receipt)
    return receipt


def run_filter(dataset: Path, urls: list[str], output: Path, filtered: Path, *, roots: list[str],
               mappings: list[str] | None = None, repeats: int = 2, transient_retries: int = 2,
               pilot_size: int = 0, seed: int = 20261010, seconds: float = 900,
               workers: int = 4, resume: bool = False, client=None) -> dict:
    if (type(repeats) is not int or repeats < 2 or type(transient_retries) is not int
            or transient_retries not in (0, 1, 2) or type(pilot_size) is not int or pilot_size < 0 or not roots):
        raise ValueError("requires at least two rounds, 0..2 transient retries, and explicit evaluator roots")
    if not urls or len(urls) != len(set(urls)) or seconds <= 0 or type(workers) is not int or workers < 1:
        raise ValueError("unique endpoints, a positive deadline and positive worker count are required")
    if filtered.resolve() == dataset.resolve():
        raise ValueError("filtered output must differ from source dataset")
    if not resume and (filtered.exists() or filtered.with_suffix(filtered.suffix + ".audit.json").exists()):
        raise ValueError("refusing to overwrite an existing filtered output or receipt")
    for mapping in roots:
        source, sep, destination = mapping.partition("=")
        if not sep or not Path(source).is_absolute() or not Path(destination).is_absolute():
            raise ValueError("evaluator roots require ABSOLUTE_SERVICE_ROOT=ABSOLUTE_LOCAL_ROOT")
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (output / ".filter.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("another filter is using this directory") from exc
        return _run_filter(dataset, urls, output, filtered, roots=roots, mappings=mappings or [], repeats=repeats,
                           transient_retries=transient_retries, pilot_size=pilot_size, seed=seed, seconds=seconds,
                           workers=workers, resume=resume, client=client or audit.Client())


def _run_filter(dataset, urls, output, filtered, *, roots, mappings, repeats, transient_retries,
                pilot_size, seed, seconds, workers, resume, client):
    cases = inventory(dataset, mappings, seconds)
    population = len(cases)
    if pilot_size:
        if pilot_size > population:
            raise ValueError(f"pilot size {pilot_size} exceeds source task count {population}; omit --pilot-size to run all")
        cases = sorted(cases, key=lambda c: audit.digest([seed, c["inventory"]["id"]]))[:pilot_size]
    native_manifest = verify_vendor()
    local = importlib.import_module("recipe.swe_agent.tool_contract_info").source_fingerprint()
    for url in urls:
        if (client.get(url, "/health").get("ok") is not True
                or client.get(url, "/tool_contract_info").get("source_fingerprint", {}).get("sha256") != local["sha256"]):
            raise ValueError("requires healthy services with matching deployed sources")
    context = {"version": VERSION, "dataset": str(dataset.resolve()), "dataset_sha256": audit.file_sha256(dataset),
               "source_rows": population, "selected": [c["inventory"]["id"] for c in cases],
               "cases": {c["inventory"]["id"]: c["context"] for c in cases},
               "native": native_manifest, "service_source": local["sha256"],
               "code": {p.name: audit.file_sha256(p) for p in [Path(__file__), Path(audit.__file__),
                         Path(__file__).with_name("swe_verified_native.py"), Path(__file__).with_name("diagnose_trajectories.py")]},
               "urls": urls, "roots": roots, "mappings": mappings, "seconds": seconds, "workers": workers,
               "repeats": repeats, "transient_retries": transient_retries, "pilot_size": pilot_size, "seed": seed,
               "filtered_output": str(filtered.resolve())}
    manifest = output / "filter_manifest.json"
    if manifest.exists():
        if not resume or json.loads(manifest.read_text()) != context:
            raise ValueError("filter already exists or resume inputs/code/artifacts changed")
    else:
        audit.audit_write_json(manifest, context)
    records = {c["inventory"]["id"]: {"id": c["inventory"]["id"], "row": c["inventory"]["row"],
               "task_id": c["task"]["task_id"], "attempts": []} for c in cases}
    round_summaries = []
    selected = cases
    for round_index in range(repeats + transient_retries):
        if not selected:
            break
        round_dir = output / f"round_{round_index + 1}"
        report = audit.reference_audit([dataset], urls, round_dir, mappings=mappings,
                    task_ids=[c["task"]["task_id"] for c in selected], seconds=seconds,
                    workers_per_endpoint=workers, resume=resume, client=client, transport_retries=0)
        if report.get("source_match") is not True or report.get("quarantined_endpoints"):
            raise ValueError("audit source/cleanup requires review; no dataset exported")
        by_id = {r["id"]: r for r in report["results"]}
        if len(by_id) != len(selected):
            raise ValueError("incomplete round task inventory")
        for case in selected:
            identity = case["inventory"]["id"]
            raw = by_id[identity]
            attempt = {"round": round_index + 1, "image_id": raw.get("cached_image_id"),
                       "signature": report["signature"], "data_issues": raw.get("data_issues", []),
                       "arms": {mode: compare_arm(arm, case, roots, round_dir / "comparison" / identity / mode)
                                for mode, arm in raw.get("arms", {}).items()}}
            records[identity]["attempts"].append(attempt)
        for record in records.values():
            record["qualification"] = qualification(record["attempts"], repeats)
        latest = [records[c["inventory"]["id"]]["attempts"][-1] for c in selected]
        round_summaries.append({"round": round_index + 1, "tasks": len(selected),
            "strict_reference_verified": sum(attempt_state(a, "strict") == "verified" for a in latest),
            "native_reference_verified": sum(attempt_state(a, "native") == "verified" for a in latest),
            "native_gold_official_resolved": sum(a.get("arms", {}).get("gold", {}).get("native", {}).get(
                "official_resolved") is True for a in latest)})
        audit.audit_write_json(output / "progress.json", {"rounds_completed": round_index + 1,
            "selected": len(cases), "strict_qualified": sum(r["qualification"]["strict"]["qualified"] for r in records.values()),
            "native_qualified": sum(r["qualification"]["native"]["qualified"] for r in records.values())})
        # The first N rounds always include every task, regardless of failures.
        # Only positively transient unresolved candidates enter extra rounds.
        selected = cases if round_index + 1 < repeats else [c for c in cases if any(
            records[c["inventory"]["id"]]["qualification"][p]["retry_needed"] for p in ("strict", "native"))]
    records = sorted(records.values(), key=lambda r: r["row"])
    groups = {k: [] for k in ("both", "strict_only", "native_only", "neither")}
    for record in records:
        strict, native = (record["qualification"][p]["qualified"] for p in ("strict", "native"))
        key = "both" if strict and native else "strict_only" if strict else "native_only" if native else "neither"
        groups[key].append(record["task_id"])
    # Source artifacts must stay unchanged while the remote audit is running.
    current = inventory(dataset, mappings, seconds)
    if verify_vendor() != native_manifest:
        raise ValueError("native parser sources changed during filter")
    code_paths = [Path(__file__), Path(audit.__file__), Path(__file__).with_name("swe_verified_native.py"),
                  Path(__file__).with_name("diagnose_trajectories.py")]
    if (any(audit.file_sha256(p) != context["code"][p.name] for p in code_paths)
            or importlib.import_module("recipe.swe_agent.tool_contract_info").source_fingerprint()["sha256"] != context["service_source"]):
        raise ValueError("evaluator or offline filter code changed during execution")
    if audit.file_sha256(dataset) != context["dataset_sha256"] or any(
            c["context"] != context["cases"][c["inventory"]["id"]] for c in current if c["inventory"]["id"] in context["cases"]):
        raise ValueError("source artifacts changed during filter")
    for key, ids in groups.items():
        (output / f"{key}_task_ids.txt").write_text("".join(t + "\n" for t in ids))
    for parser in ("strict", "native"):
        ids = [r["task_id"] for r in records if r["qualification"][parser]["qualified"]]
        name = "reference_verified_task_ids.txt" if parser == "strict" else "native_reference_verified_task_ids.txt"
        (output / name).write_text("".join(t + "\n" for t in ids))
    with (output / "comparison_results.jsonl").open("w") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    receipt = export(dataset, filtered, records, context, repeats)
    summary = {"version": VERSION, "source_rows": population, "audited_tasks": len(cases), "repeats": repeats,
               "strict_repeat_qualified": receipt["rows"],
               "native_repeat_qualified": len(groups["both"]) + len(groups["native_only"]),
               "groups": groups, "counts": {k: len(v) for k, v in groups.items()},
               "rounds": round_summaries,
               "native_scope": "pinned parser/grading replay on our runner logs; not a full native-harness execution",
               "filtered_dataset": str(filtered.resolve()), "receipt": receipt}
    audit.audit_write_json(output / "report.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--filtered-output", type=Path, required=True)
    parser.add_argument("--evaluator-root", action="append", required=True)
    parser.add_argument("--path-map", action="append", default=[])
    parser.add_argument("--repeat-count", type=int, default=2, help="total clean rounds (default: 2)")
    parser.add_argument("--transient-retries", type=int, default=2, help="additional transient attempts per task, 0..2")
    parser.add_argument("--pilot-size", type=int, default=0, help="0 = all; 100 requires at least 100 source tasks")
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--scenario-deadline", type=float, default=900)
    parser.add_argument("--workers-per-endpoint", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    urls = [s.strip().rstrip("/") for s in os.environ.get("SWE_AGENT_EXECUTION_URLS", "").split(",") if s.strip()]
    if not urls:
        parser.error("SWE_AGENT_EXECUTION_URLS is required")
    report = run_filter(args.dataset, urls, args.output_dir, args.filtered_output, roots=args.evaluator_root,
        mappings=args.path_map, repeats=args.repeat_count, transient_retries=args.transient_retries,
        pilot_size=args.pilot_size, seed=args.seed, seconds=args.scenario_deadline,
        workers=args.workers_per_endpoint, resume=args.resume)
    print(json.dumps({k: report[k] for k in ["audited_tasks", "strict_repeat_qualified", "native_repeat_qualified",
                                            "counts", "filtered_dataset"]}, indent=2))


if __name__ == "__main__":
    main()

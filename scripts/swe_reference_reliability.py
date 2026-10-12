"""Repeat qualification for immutable ReBench reference-audit ledgers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

CONTEXT_KEYS = {"task_sha256", "evaluator_sha256", "reference_contract_sha256",
                "service_source_sha256", "audit_code_sha256", "image_id", "artifact_sha256"}


def validate_context(original: dict, attempt: dict) -> None:
    context, current = original.get("verification_context", {}), attempt.get("verification_context", {})
    if not all(context.get(k) for k in CONTEXT_KEYS):
        raise ValueError("repeat qualification requires a fresh audit with immutable context")
    if any(current.get(k) != context[k] for k in CONTEXT_KEYS - {"image_id"}):
        raise ValueError("repeat evaluator or artifact identity changed")
    if current.get("image_id") not in (None, context["image_id"]):
        raise ValueError("repeat Docker image identity changed")
    if attempt.get("status") == "reference_verified" and current.get("image_id") != context["image_id"]:
        raise ValueError("verified repeat lacks matching image identity")


def sha256_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def read_ledger(path: Path) -> dict[int, dict]:
    records, signatures, sources = {}, set(), set()
    with path.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            row, source = record.get("row"), record.get("dataset_sha256")
            identity = hashlib.sha256(json.dumps([source, row], sort_keys=True).encode()).hexdigest()
            if (record.get("version") != "swe-reference-results-v1" or type(row) is not int
                    or row < 0 or not source or record.get("id") != identity or row in records):
                raise ValueError("duplicate or invalid audit row identity")
            if record.get("source_match") is not True or not record.get("signature"):
                raise ValueError("audit requires matching deployed sources and a signature")
            records[row] = record
            signatures.add(record["signature"])
            sources.add(source)
    if not records or len(signatures) != 1 or len(sources) != 1:
        raise ValueError("ledger must contain one nonempty audit of one dataset")
    return records


def validate_reference_record(record: dict, *, guarded: bool = False) -> None:
    """Validate evidence, never trusting a reference_verified label alone."""
    if guarded and (record.get("source_match") is not True or not record.get("signature")
                    or record.get("version") != "swe-reference-results-v1"):
        raise ValueError("repeat evidence requires matching deployed sources and a signature")
    arms = record.get("arms", {})
    if record.get("complete") is not True or set(arms) != {"baseline", "gold"}:
        raise ValueError("reference_verified row lacks complete verification evidence")
    baseline, gold = arms["baseline"], arms["gold"]
    before, after = baseline.get("reference_evidence", {}), gold.get("reference_evidence", {})
    if (not all(a.get("cleanup_verified") is True for a in arms.values())
            or baseline.get("score") != 0 or gold.get("score") != 1
            or before.get("complete") is not True or before.get("fail_to_pass_failed", 0) <= 0
            or before.get("pass_to_pass_complete") is not True
            or after.get("complete") is not True or after.get("all_required_passed") is not True):
        raise ValueError("reference_verified row lacks consistent verification evidence")
    for mode, arm in arms.items():
        evidence = arm["reference_evidence"]
        consistency = evidence.get("service_consistency", {})
        if evidence.get("parser_policy") == "native-guarded" and consistency.get("matched") is not True:
            raise ValueError("native-guarded export requires matching audit and service scoring")
        if guarded:
            private = arm.get("private_evaluator", {})
            if (evidence.get("parser_policy") != "native-guarded"
                    or private.get("status") != "verified" or private.get("logs_complete") is not True
                    or not private.get("sha256") or not arm.get("request_id")):
                raise ValueError("repeat qualification requires complete guarded private evidence")
            runners = consistency.get("runners", [])
            if not runners or any(not r.get("contract_sha256") or not r.get("parser_input_sha256")
                                  or not r.get("parser_commit") or type(r.get("exit_code")) is not int
                                  for r in runners):
                raise ValueError("repeat qualification requires runner provenance")
            if mode == "gold" and any(r["exit_code"] != 0 for r in runners):
                raise ValueError("repeat gold execution must have zero exit")
            contract = record.get("verification_context", {}).get("reference_contract_sha256")
            if any(r["contract_sha256"] != contract for r in runners):
                raise ValueError("repeat runner contract identity mismatch")


def _transient(record: dict) -> bool:
    """Only positively classified transport/provisioning failures may be retried."""
    if record.get("complete") is not True or record.get("data_issues"):
        return False
    if record.get("status") in {"review:image:transport_failure", "review:image:admission_timeout"}:
        return True
    arms = record.get("arms", {})
    allowed = {"transport_failure", "admission_or_transport_failure", "reward_transport_or_service_failure"}
    if not arms or not set(arms) <= {"baseline", "gold"} or not any(
            a.get("classification") in allowed for a in arms.values()):
        return False
    for mode, arm in arms.items():
        if arm.get("cleanup_verified") is not True:
            return False
        evidence = arm.get("reference_evidence", {})
        if arm.get("classification") not in allowed or evidence.get("complete") is True:
            # A transport error in one arm cannot hide a genuine failure in
            # the other arm (or completed evidence in the same arm).
            if (evidence.get("complete") is not True
                    or evidence.get("parser_policy") != "native-guarded"
                    or evidence.get("service_consistency", {}).get("matched") is not True):
                return False
            if mode == "baseline":
                if (arm.get("score") != 0 or evidence.get("fail_to_pass_failed", 0) <= 0
                        or evidence.get("pass_to_pass_complete") is not True):
                    return False
            elif arm.get("score") != 1 or evidence.get("all_required_passed") is not True:
                return False
    return True


def qualify_repeats(first_results: Path, repeat_results: list[Path], output: Path) -> dict:
    """Two clean runs; contradictions veto acceptance and up to two transient retries."""
    if not 1 <= len(repeat_results) <= 3:
        raise ValueError("provide one repeat audit and at most two additional retry audits")
    paths = [first_results, *repeat_results]
    hashes = [sha256_file(p) for p in paths]
    if len(set(hashes)) != len(hashes):
        raise ValueError("a copied or resumed ledger is not an independent repeat")
    ledgers = [read_ledger(p) for p in paths]
    first = ledgers[0]
    dataset_sha = next(iter(first.values()))["dataset_sha256"]
    for ledger in ledgers[1:]:
        if any(r["dataset_sha256"] != dataset_sha or row not in first
               or r.get("task_id") != first[row].get("task_id") for row, r in ledger.items()):
            raise ValueError("repeat audit belongs to another dataset or task")
    merged, verified, retry = [], [], []
    for row, original in sorted(first.items()):
        record = dict(original)
        attempts = [ledger[row] for ledger in ledgers if row in ledger]
        candidate = original.get("status") == "reference_verified"
        clean, veto, seen_requests = [], False, set()
        if candidate:
            for attempt in attempts:
                # Provisioning/transport failures may lack an image ID, but may
                # never change dataset, evaluator, parser or artifact identities.
                validate_context(original, attempt)
                requests = {a.get("request_id") for a in attempt.get("arms", {}).values()} - {None}
                if seen_requests & requests:
                    raise ValueError("repeat reuses an earlier execution request")
                seen_requests.update(requests)
                if attempt.get("status") == "reference_verified":
                    validate_reference_record(attempt, guarded=True)
                    clean.append(attempt)
                elif not _transient(attempt):
                    veto = True
        qualified = candidate and len(clean) >= 2 and not veto
        record["repeat_qualification"] = {
            "version": 1, "qualified": qualified, "clean_runs": len(clean),
            "contradiction_or_uncertainty": veto,
            "source_ledgers": [{"path": str(p.resolve()), "sha256": h} for p, h in zip(paths, hashes)],
            "attempts": attempts if candidate else [],
        }
        if qualified:
            verified.append(record["task_id"])
        elif candidate:
            record["status"] = "review:repeat_not_verified"
            if not veto and len(attempts) < 4:
                retry.append(record["task_id"])
        merged.append(record)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "qualified_results.jsonl").open("x") as stream:
        for record in merged:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    for name, ids in [("reference_verified_task_ids.txt", verified), ("retry_task_ids.txt", retry)]:
        (output / name).write_text("".join(identity + "\n" for identity in ids))
    report = {"version": "swe-repeat-qualification-v1", "dataset_sha256": dataset_sha,
              "source_rows": len(first), "first_verified": sum(r["status"] == "reference_verified" for r in first.values()),
              "repeat_qualified": len(verified), "retry_candidates": len(retry),
              "source_ledgers": [{"path": str(p.resolve()), "sha256": h} for p, h in zip(paths, hashes)]}
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    return report


def validate_repeat_qualification(record: dict) -> None:
    proof = record.get("repeat_qualification", {})
    attempts = proof.get("attempts", [])
    if (proof.get("version") != 1 or proof.get("qualified") is not True
            or proof.get("contradiction_or_uncertainty") is not False or not 2 <= len(attempts) <= 4):
        raise ValueError("export requires two independent reference-verified audits")
    requests, clean = set(), 0
    for attempt in attempts:
        if any(attempt.get(k) != record.get(k) for k in ("id", "row", "task_id", "dataset_sha256")):
            raise ValueError("repeat proof row identity mismatch")
        if (attempt.get("source_match") is not True or not attempt.get("signature")
                or attempt.get("version") != "swe-reference-results-v1"):
            raise ValueError("repeat proof requires matching deployed sources and a signature")
        validate_context(record, attempt)
        current = {a.get("request_id") for a in attempt.get("arms", {}).values()} - {None}
        if requests & current:
            raise ValueError("repeat proof reuses execution requests")
        requests.update(current)
        if attempt.get("status") == "reference_verified":
            validate_reference_record(attempt, guarded=True)
            clean += 1
        elif not _transient(attempt):
            raise ValueError("repeat proof contains an unresolved failure")
    if clean < 2 or proof.get("clean_runs") != clean:
        raise ValueError("export requires two independent reference-verified audits")

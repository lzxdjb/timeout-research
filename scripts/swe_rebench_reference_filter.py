"""Bounded per-task ReBench reference/repeat pipeline with terminal image cleanup."""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import copy
import fcntl
import importlib
import json
import os
import re
import sys
import threading
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import swe_rebench_endpoints as availability
from scripts import swe_tool_online_benchmark as audit
from scripts.swe_rebench_live_results import LiveResults
from scripts.swe_reference_reliability import (
    _transient,
    validate_context,
    validate_reference_record,
    validate_repeat_qualification,
)

VERSION = "swe-rebench-reference-pipeline-v1"
IMAGE_ID = re.compile(r"sha256:[a-f0-9]{64}")
RECOVERY_BACKOFF = (0, 1, 2)


class CleanupPending(Exception):
    """Yield execution capacity until the separate recovery queue proves cleanup."""

# This opt-in migration covers the deployed cleanup fix, not arbitrary changes
# to parsers, scorers or execution. Keep exact before/after hashes for shared
# audit/service files; the pipeline itself also contains this migration code.
CLEANUP_UPGRADE_PIPELINE_FROM = "a182029b834881939cb3a3dae0b0272538507935866429ebd3117d9fd1977368"
SCHEDULING_UPGRADE_PIPELINE_FROM = "87355fc861c2cb590bf13424699b1e2e4c31b13e1208ba59c63799b62ada6264"
ENDPOINT_UPGRADE_PIPELINE_FROM = "96b70085db135a275fd8fde1a93a3d8b34b884d1b13ee4eeb92a8c0aa4f55f02"
ENDPOINT_UPGRADE_HELPER_FROM = "75ab2e6ea864f6363ecaa248043ef0060a825edbff10c461822f98ee16f30622"
PERFORMANCE_UPGRADE_PIPELINE_FROM = "1a90bf9b00aa3c4c59d0308298c68e670832830a14ae15bd56135b62468c010c"
PERFORMANCE_UPGRADE_HELPER_FROM = "7eac9e9625eda53b0432b379b82ac5652ffec5473a68092a086851fe77b91538"
# Filled with exact reviewed file hashes, not a general fingerprint bypass.
PERFORMANCE_UPGRADE_CODE = {
    "swe_tool_online_benchmark.py": (
        "44ca941ed3e1140fc02cd561b906eeb6e09df967540493df2a071b75a1914fb7",
        "d8d327c81acc1659c292b1ca231fe7d138e4ca68b51cdcdc2c03911698dfab28"),
    "swe_rebench_endpoints.py": (
        PERFORMANCE_UPGRADE_HELPER_FROM,
        "d15447b457af89654e502597bfd8d63e16b8a8de5fcf34d62e18440e6590ca4a"),
}
PERFORMANCE_UPGRADE_SERVICE = {
    "remote_execution_service.py": (
        "59191c3a2a9e3cd1bb3e4217ddaa1ecb972b78cefc3162c17e1fab6a8667b6ad",
        "668b63d693822571fbee81d41f74e1bc34b9b56c4703637ce911a313d08a1687"),
    "tools.py": (
        "ab328ee923e7d281fecc2fd14a0cda30a682a75d8a2a1c74ac297d60013ac0b2",
        "8340d094fde028ca08cba1d2a927c7310672ad4d61d91f5c321cbc0daf4b2090"),
    "execution_router.py": (None, "3e23476f82d8f2ab5c9b2824111b1d6f84cdcf52d41e1e5996ec95828342f785"),
    "tool_contract_info.py": (
        "daa722428285d83b0df9812a18118ff989898e1adb597edb1bb8ed58d10b365e",
        "0cb7d1180df4cf123da8b7e599808091dbe1ebc6bc0757faaac44c76c97f3fa5"),
}
CLEANUP_UPGRADE_CODE = {
    "swe_tool_online_benchmark.py": (
        "b365c18f74036a45768773912fdedd374d4a077cc9b07db154579784fdc73e8c",
        "44ca941ed3e1140fc02cd561b906eeb6e09df967540493df2a071b75a1914fb7"),
}
CLEANUP_UPGRADE_SERVICE = {
    "remote_execution_service.py": (
        "1a2fb7ef7a7d7afd72eeb401f246f41b69064a15b34fe56a304e16a9f9b03479",
        "59191c3a2a9e3cd1bb3e4217ddaa1ecb972b78cefc3162c17e1fab6a8667b6ad"),
    "tool_contract_info.py": (
        "c90599f6495aa1f52cdcedf13f2af998ba60a0e3415aaa0e6a0cbb8af703deb8",
        "daa722428285d83b0df9812a18118ff989898e1adb597edb1bb8ed58d10b365e"),
}


def verification_identity(identity: dict) -> dict:
    """Endpoint addresses are dispatch configuration, not verifier semantics."""
    return {key: value for key, value in identity.items() if key != "services"}


def execution_digest(identity: dict) -> str:
    return audit.digest(verification_identity(identity))


def validate_cleanup_upgrade(previous: dict, current: dict) -> None:
    """Accept only the known cleanup upgrade with identical verification inputs."""
    if PERFORMANCE_UPGRADE_CODE and current["code"].get("swe_tool_online_benchmark.py") == (
            PERFORMANCE_UPGRADE_CODE["swe_tool_online_benchmark.py"][1]):
        validate_performance_upgrade(previous, current)
        return
    def check_files(old, new, allowed):
        expected = dict(old)
        for name, (before, after) in allowed.items():
            if old.get(name) != before or new.get(name) != after:
                raise ValueError("unsupported cleanup upgrade: " + name)
            expected[name] = after
        if new != expected:
            raise ValueError("cleanup upgrade changes protected parser/scorer/code files")

    expected = verification_identity(previous)
    allowed = {Path(__file__).name: (previous["code"][Path(__file__).name], current["code"][Path(__file__).name])}
    if previous["code"][Path(__file__).name] not in {
            CLEANUP_UPGRADE_PIPELINE_FROM, SCHEDULING_UPGRADE_PIPELINE_FROM, ENDPOINT_UPGRADE_PIPELINE_FROM}:
        raise ValueError("unsupported pipeline scheduling upgrade")
    if previous["source"] != current["source"]:
        allowed.update(CLEANUP_UPGRADE_CODE)
    prior_code = dict(previous["code"])
    # New scheduling helper is recorded in the epoch; old parser/scorer/writer
    # inventories must still match exactly.
    helper = "swe_rebench_endpoints.py"
    if helper not in prior_code and helper in current["code"]:
        prior_code[helper] = current["code"][helper]
    elif previous["code"][Path(__file__).name] == ENDPOINT_UPGRADE_PIPELINE_FROM:
        allowed[helper] = (ENDPOINT_UPGRADE_HELPER_FROM, current["code"][helper])
    check_files(prior_code, current["code"], allowed)
    for source in (previous["source"], current["source"]):
        if source.get("version") != 1 or audit.digest(source["files"]) != source.get("sha256"):
            raise ValueError("invalid cleanup upgrade source inventory")
    check_files(previous["source"]["files"], current["source"]["files"],
                CLEANUP_UPGRADE_SERVICE if previous["source"] != current["source"] else {})
    for identity in (previous, current):
        if any(service["source"] != identity["source"] for service in identity["services"]):
            raise ValueError("cleanup upgrade requires matching historical and current deployments")
    expected.update(code=current["code"], source=current["source"])
    if expected != verification_identity(current):
        raise ValueError("cleanup upgrade requires unchanged dataset/artifacts/settings/output")


def validate_performance_upgrade(previous: dict, current: dict) -> None:
    """Compose known historical cleanup fixes with this exact transport fix."""
    before = previous["code"][Path(__file__).name]
    if before not in {CLEANUP_UPGRADE_PIPELINE_FROM, SCHEDULING_UPGRADE_PIPELINE_FROM,
                      ENDPOINT_UPGRADE_PIPELINE_FROM, PERFORMANCE_UPGRADE_PIPELINE_FROM}:
        raise ValueError("unsupported pipeline performance upgrade")
    for old, new, allowed, legacy in (
            (previous["code"], current["code"], PERFORMANCE_UPGRADE_CODE, CLEANUP_UPGRADE_CODE),
            (previous["source"]["files"], current["source"]["files"],
             PERFORMANCE_UPGRADE_SERVICE, CLEANUP_UPGRADE_SERVICE)):
        expected = dict(old)
        if old is previous["code"]:
            expected[Path(__file__).name] = current["code"][Path(__file__).name]
        for name, (prior, target) in allowed.items():
            permitted = {prior, *legacy.get(name, ())}
            if name == "swe_rebench_endpoints.py":
                permitted.add(ENDPOINT_UPGRADE_HELPER_FROM)
                if before in {CLEANUP_UPGRADE_PIPELINE_FROM, SCHEDULING_UPGRADE_PIPELINE_FROM}:
                    permitted.add(None)
            if old.get(name) not in permitted or new.get(name) != target:
                raise ValueError("unsupported performance upgrade: " + name)
            expected[name] = target
        if expected != new:
            raise ValueError("performance upgrade changes protected parser/scorer/code files")
    for identity in (previous, current):
        source = identity["source"]
        if source.get("version") != 1 or audit.digest(source["files"]) != source.get("sha256"):
            raise ValueError("invalid performance upgrade source inventory")
        if any(service["source"] != source for service in identity["services"]):
            raise ValueError("performance upgrade requires matching deployed source")
    expected = verification_identity(previous)
    expected.update(code=current["code"], source=current["source"])
    if expected != verification_identity(current):
        raise ValueError("performance upgrade requires unchanged dataset/artifacts/settings/output")


def resume_identity(output: Path, identity: dict, resume: bool, cleanup_upgrade: bool) -> tuple[str, dict | None]:
    """Keep the original run ID/proofs; bind new work to a recorded code epoch."""
    signature = execution_digest(identity)
    manifest_path = output / "manifest.json"
    epoch_path = output / "cleanup_upgrade.json"
    if cleanup_upgrade and not resume:
        raise ValueError("--resume-cleanup-upgrade requires --resume")
    if not manifest_path.exists():
        if cleanup_upgrade or any(p.name != ".filter.lock" for p in output.iterdir()):
            raise ValueError("use a fresh pipeline directory; no resumable pipeline manifest")
        audit.audit_write_json(manifest_path, {"version": VERSION, "signature": signature, "identity": identity,
                                             "identity_policy": "endpoints-independent-v1"})
        return signature, None
    previous = json.loads(manifest_path.read_text())
    original = previous["identity"]
    run_id = previous["signature"]
    original_digest = (execution_digest(original) if previous.get("identity_policy") == "endpoints-independent-v1"
                       else audit.digest(original))
    if original_digest != run_id:
        raise ValueError("pipeline manifest identity changed")
    if not resume:
        raise ValueError("pipeline exists; use --resume")
    if epoch_path.exists():
        epoch = json.loads(epoch_path.read_text())
        if epoch.get("version") != 1 or epoch.get("run_signature") != run_id:
            raise ValueError("cleanup upgrade epoch changed")
        prior = {**original, "code": epoch["code"], "source": epoch["source"], "services": epoch["services"]}
        prior_digest = (execution_digest(prior) if epoch.get("identity_policy") == "endpoints-independent-v1"
                        else audit.digest(prior))
        if epoch.get("execution_signature") != prior_digest:
            raise ValueError("cleanup upgrade epoch changed: recorded identity hash mismatch")
        if verification_identity(prior) == verification_identity(identity):
            # Preserve the old execution context as well as the run ID when
            # ONLY endpoints change; unfinished repeat attempts remain reusable.
            return run_id, epoch
        if epoch.get("execution_signature") != signature:
            if not cleanup_upgrade:
                raise ValueError("cleanup upgrade epoch changed; use --resume-cleanup-upgrade for the scheduling fix")
            try:
                validate_cleanup_upgrade(prior, identity)
            except ValueError as exc:
                raise ValueError("cleanup upgrade epoch changed: " + str(exc)) from exc
            epoch = {**epoch, "previous_epochs": [*epoch.get("previous_epochs", []),
                     {k: v for k, v in epoch.items() if k != "previous_epochs"}],
                     "execution_signature": signature, "code": identity["code"], "source": identity["source"],
                     "services": identity["services"], "created_at_unix": time.time(),
                     "identity_policy": "endpoints-independent-v1"}
            audit.audit_write_json(epoch_path, epoch)
        if epoch.get("code") != identity["code"] or epoch.get("source") != identity["source"]:
            raise ValueError("cleanup upgrade epoch changed; resume still requires matching code/settings/services")
        validate_cleanup_upgrade(original, identity)
        return run_id, epoch
    if execution_digest(original) == signature:
        return run_id, None
    if not cleanup_upgrade:
        raise ValueError("resume source/code/settings/deployed services changed; "
                         "known cleanup fix may use --resume-cleanup-upgrade")
    validate_cleanup_upgrade(original, identity)
    if Path(identity["filtered"]).exists() or Path(identity["filtered"] + ".audit.json").exists():
        raise ValueError("cleanup upgrade cannot migrate an already exported dataset")
    # The original manifest already owns the large task/artifact inventory.
    # Bind all unchanged inputs by digest; do not copy 32k entries into every
    # report and export receipt merely to record a source-code epoch.
    epoch = {"version": 1, "run_signature": run_id, "execution_signature": signature,
             "code": identity["code"], "source": identity["source"], "services": identity["services"],
             "created_at_unix": time.time(),
             "identity_policy": "endpoints-independent-v1",
             "policy": "preserve_terminal_proofs_restart_unfinished_after_cleanup"}
    audit.audit_write_json(epoch_path, epoch)
    return run_id, epoch


def restart_unfinished_after_upgrade(directory: Path, state: dict, execution_signature: str,
                                     archive_name="before_cleanup_upgrade") -> None:
    """Archive old attempts only AFTER verified cleanup; restart safely after a crash."""
    archive = directory / archive_name
    archive.mkdir(exist_ok=True)
    saved_path = archive / "state.json"
    if not saved_path.exists():
        read_attempts(directory, state)
        audit.audit_write_json(saved_path, state)
    saved = json.loads(saved_path.read_text())
    if saved.get("signature") != state.get("signature") or saved.get("terminal"):
        raise ValueError("unfinished upgrade archive identity changed")
    for i, expected in enumerate(saved.get("attempt_sha256", [])):
        relative = Path(f"attempt_{i}") / "record.json"
        path = directory / relative
        if not path.exists():
            path = archive / relative
        if audit.file_sha256(path) != expected:
            raise ValueError("unfinished upgrade attempt checkpoint changed")
    paths = [*directory.glob("attempt_*"), directory / "qualified_record.json", directory / "cleanup_recovery.json"]
    for path in paths:
        if path.exists():
            destination = archive / path.name
            if destination.exists():
                raise ValueError("conflicting unfinished upgrade archive")
            path.rename(destination)
    state.update(attempt_sha256=[], execution_signature=execution_signature, phase="resumed_cleanup_upgrade")
    for key in ("blocked", "qualified_sha256"):
        state.pop(key, None)
    audit.audit_write_json(directory / "state.json", state)


def migration_cleanup_required(directory, state):
    """A saved migration plan already proves original cleanup before any move."""
    if state.get("migration_pending"):
        return False
    return bool(state.get("image_preparation") or state.get("image_preparation_requested")
                or state.get("image_id") or any(directory.glob("attempt_*/*/*.json")))


def migrate_task(client, case, directory, state, destination, execution_signature, remove):
    """Caller must prove request cleanup first. A durable plan makes moves resumable."""
    if "migration_pending" not in state:
        # A different daemon must never give a completed test failure or parser
        # ambiguity another chance to become a positive reward.
        for path in directory.glob("attempt_*/*/*.json"):
            arm = json.loads(path.read_text())
            kind = arm.get("classification")
            transient = kind in {"transport_failure", "admission_or_transport_failure",
                                 "reward_transport_or_service_failure", "interrupted_recovered"}
            valid = arm.get("reference_evidence", {}).get("complete") is True
            if (arm.get("phase") == "complete" and not transient
                    and (not valid or arm.get("mode") == "gold" and kind != "tests_passed"
                         or arm.get("mode") == "baseline" and kind == "tests_passed")):
                raise ValueError("completed failure/parser uncertainty cannot be retried by endpoint migration")
        cleanup = (finish_image(client, state, case["inventory"]["docker_image"], remove)
                   if state.get("image_preparation") or state.get("image_preparation_requested") else
                   {"ok": True, "status": "not_prepared"})
        if cleanup.get("ok") is not True:
            raise availability.EndpointUnavailable("original image lease cleanup unverified: " + str(cleanup))
        state["migration_pending"] = {"from": state["endpoint"], "to": destination,
            "archive": "before_failover_" + uuid.uuid4().hex, "lease_id": "rebench-" + uuid.uuid4().hex,
            "image_cleanup": cleanup, "at_unix": time.time()}
        audit.audit_write_json(directory / "state.json", state)
    plan = state["migration_pending"]
    restart_unfinished_after_upgrade(directory, state, execution_signature, plan["archive"])
    # Keep image_id pinned, but never reuse the original reservation or attempts.
    state.update(endpoint=plan["to"], lease_id=plan["lease_id"], phase="migrated",
                 migrations=[*state.get("migrations", []), plan])
    state.pop("image_preparation", None)
    state.pop("image_preparation_requested", None)
    state.pop("migration_pending", None)
    audit.audit_write_json(directory / "state.json", state)


def recover_case_cleanup(client, case, directory, check_endpoint, stop=None) -> bool:
    """Cleanup only; recovered results still need the normal verifier/repeat proof."""
    history = []
    for delay in RECOVERY_BACKOFF:
        if stop is not None and stop.wait(delay):
            return False
        if stop is None and delay:
            time.sleep(delay)
        unresolved = []
        try:
            check_endpoint()
            for path in directory.glob("attempt_*/*/*.json"):
                arm = json.loads(path.read_text())
                if arm.get("phase") == "complete" and arm.get("cleanup", {}).get("verified"):
                    continue
                arm = audit.audit_recover_arm(client, case["task"], arm, preserve_completed=True)
                audit.audit_write_json(path, arm)
                if not arm.get("cleanup", {}).get("verified"):
                    unresolved.append({"path": str(path), "cleanup": arm.get("cleanup")})
            outcome = {"ok": not unresolved, "unresolved": unresolved}
        except Exception as exc:
            outcome = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:1500]}
        history.append(outcome)
        audit.audit_write_json(directory / "cleanup_recovery.json", {"attempts": history, "ok": outcome["ok"]})
        if outcome["ok"]:
            return True
    return False


def guarded_inventory(dataset: Path, mappings: list[str]) -> tuple[dict, list[dict]]:
    from recipe.swe_agent.test_results import official_evidence_sha256, rebench_contract_fields
    manifest, cases = audit.audit_inventory([dataset], mappings, profile="rebench")
    for case in cases:
        contract = case.get("reference_contract")
        if contract is None:
            continue
        contract.update(rebench_contract_fields(contract["log_parser"], contract["fail_to_pass"], contract["pass_to_pass"]))
        case["task"]["official_test_contract"] = contract
        item = case["inventory"]
        item["reference_contract_sha256"] = official_evidence_sha256(contract)
        item["task_sha256"] = audit.digest(case["task"])
        item["evaluator_sha256"] = audit.digest({k: v for k, v in case["task"].items() if k in {
            "test_commands", "evaluator_test_commands", "verifier_setup_commands", "hidden_test_patch",
            "trusted_hidden_test_patch", "verifier_files", "official_test_contract"}})
    return manifest, cases


def qualify_task(attempts: list[dict], signature: str, sources: list[dict]) -> tuple[dict, bool]:
    """Keep first-pass gating and the existing two-clean-run ReBench proof."""
    if not attempts or len(attempts) > 4:
        raise ValueError("one initial attempt, one repeat and at most two retry attempts required")
    first = attempts[0]
    candidate = first["status"] == "reference_verified"
    clean, veto, requests = 0, False, set()
    if candidate:
        for attempt in attempts:
            if any(attempt.get(k) != first.get(k) for k in ("id", "row", "task_id", "dataset_sha256")):
                raise ValueError("repeat row identity changed")
            validate_context(first, attempt)
            current = [a.get("request_id") for a in attempt.get("arms", {}).values() if a.get("request_id")]
            if len(current) != len(set(current)) or requests.intersection(current):
                raise ValueError("repeat reuses an execution request")
            requests.update(current)
            if attempt["status"] == "reference_verified":
                validate_reference_record(attempt, guarded=True)
                clean += 1
            elif not _transient(attempt):
                veto = True
    qualified = candidate and clean >= 2 and not veto
    record = copy.deepcopy(first)
    record["signature"] = signature
    record["repeat_qualification"] = {"version": 1, "qualified": qualified, "clean_runs": clean,
        "contradiction_or_uncertainty": veto, "source_ledgers": sources,
        "attempts": attempts if candidate else []}
    if candidate and not qualified:
        record["status"] = "review:repeat_not_verified"
    if qualified:
        validate_repeat_qualification(record)
    return record, candidate and not qualified and not veto and len(attempts) < 4


def compact_record(item: dict, raw: dict, signature: str, context: dict) -> dict:
    record = audit.audit_result_record(item, raw, signature, True, context)
    # Full native maps already reside in private/raw attempt checkpoints. Keep
    # hashes in the portable proof so a 32k task ledger stays streamable.
    for arm in record["arms"].values():
        evidence = arm["reference_evidence"]
        native = evidence.get("native_parser", {})
        if "evidence" in native:
            native["evidence_sha256"] = audit.digest(native.pop("evidence"))
        shadow = evidence.get("strict_shadow", {})
        if "outcomes" in shadow:
            shadow["outcomes_sha256"] = audit.digest(shadow.pop("outcomes"))
    return record


class Progress:
    """Constant-size updates; never rescan all 32k records on each event."""
    def __init__(self, output: Path, total: int, signature: str):
        from tqdm import tqdm
        self.output, self.total, self.signature = output, total, signature
        self.lock = threading.Lock()
        self.active = collections.Counter()
        self.counts = collections.Counter()
        self.blocked_states = set()
        self.last_publish = 0.0
        self.global_stage = "dispatch"
        self.started = time.monotonic()
        self.resumed_completed = 0
        self.stage_seconds = collections.Counter()
        self.bar = tqdm(total=total, desc="ReBench repeat filter", unit="task", file=sys.stderr)
        self.publish()

    def publish(self, force=False):
        now = time.monotonic()
        if not force and now - self.last_publish < 1:
            return
        self.last_publish = now
        snapshot = {"version": VERSION, "stage": self.global_stage, "total": self.total,
                    "completed": self.counts["completed"], "remaining": self.total - self.counts["completed"],
                    "active": dict(self.active), "counts": dict(self.counts), "signature": self.signature,
                    "updated_at_unix": time.time()}
        elapsed = max(.001, now - self.started)
        new_completed = self.counts["completed"] - self.resumed_completed
        snapshot["session"] = {"resumed_completed": self.resumed_completed,
            "new_completed": new_completed, "elapsed_seconds": round(elapsed, 3),
            "tasks_per_second": new_completed / elapsed,
            "eta_seconds": (self.total - self.counts["completed"]) * elapsed / new_completed if new_completed else None}
        snapshot["stage_seconds"] = dict(self.stage_seconds)
        audit.audit_write_json(self.output / "progress.json", snapshot)
        self.bar.set_postfix(verified=self.counts["verified"], images_removed=self.counts["images_removed"],
                             blocked=self.counts["blocked"])

    def stage(self, before: str | None, after: str | None):
        with self.lock:
            if before:
                self.active[before] -= 1
            if after:
                self.active[after] += 1
            self.publish()

    def finish(self, state: dict, record: dict | None, *, historical=False):
        with self.lock:
            key = id(state)
            if state.get("terminal"):
                self.blocked_states.discard(key)
                self.counts["completed"] += 1
                self.resumed_completed += historical
                self.counts["verified"] += bool(record and record["status"] == "reference_verified")
                self.counts["images_removed"] += state.get("image_cleanup", {}).get("status") == "removed"
                self.counts["cleanup_pending"] += state.get("image_cleanup", {}).get("ok") is not True
                self.counts["images_shared_retained"] += state.get("image_cleanup", {}).get("status") == "shared_audit_retained"
                if historical:
                    # Restored terminal proofs count toward total completion,
                    # but must not inflate the current invocation's rate/ETA.
                    self.bar.total -= 1
                else:
                    self.bar.update(1)
            else:
                self.blocked_states.add(key)
            self.counts["blocked"] = len(self.blocked_states)
            self.publish()

    def duration(self, stage, elapsed):
        with self.lock:
            self.stage_seconds[stage] += elapsed


def read_attempts(case_dir: Path, state: dict) -> tuple[list[dict], list[dict]]:
    attempts, sources = [], []
    for i, expected in enumerate(state.get("attempt_sha256", [])):
        path = case_dir / f"attempt_{i}" / "record.json"
        actual = audit.file_sha256(path)
        if actual != expected:
            raise ValueError("attempt checkpoint changed")
        attempts.append(json.loads(path.read_text()))
        sources.append({"path": str(path.resolve()), "sha256": actual})
    return attempts, sources


def execute_attempt(client, url, case, directory, seconds, roots, signature, context, image_id, interrupted):
    directory.mkdir(parents=True, exist_ok=True)
    result_path = directory / "result.json"
    if result_path.exists():
        raw = json.loads(result_path.read_text())
        if raw.get("complete") is True:
            if (raw.get("id") != case["inventory"]["id"] or raw.get("task_id") != case["task"]["task_id"]
                    or raw.get("signature") != signature or raw.get("cached_image_id") != image_id):
                raise ValueError("completed attempt identity changed")
            return compact_record(case["inventory"], raw, signature, context)
    raw = {"id": case["inventory"]["id"], "task_id": case["task"]["task_id"],
           "signature": signature, "cached_image_id": image_id, "arms": {}, "complete": True}
    task = audit.audit_rebench_task(case["task"], seconds)
    # Address the prepared content by ID so a mutable tag cannot change between
    # baseline and gold (or between first and repeat audits).
    task["docker_image"] = task["case_image"] = image_id
    task["environment"] = {**task.get("environment", {}), "base_image": image_id, "case_image": image_id}
    for mode in ("baseline", "gold"):
        arm_dir = directory / mode
        arm_dir.mkdir(exist_ok=True)
        paths = sorted(arm_dir.glob("*.json"), key=lambda p: int(p.stem))
        latest = json.loads(paths[-1].read_text()) if paths else None
        if latest and latest.get("phase") == "complete" and not latest.get("recovered_interruption"):
            arm = latest
            if arm.get("cleanup", {}).get("verified") and ("reference_evidence" not in arm or arm.get("recovered_cleanup")):
                audit.audit_rebench_verify(arm, case, roots, arm_dir, parser_policy="native-guarded")
                audit.audit_write_json(paths[-1], arm)
        else:
            # Initial audits retain the existing single transport retry per arm.
            # Repeat transport retries are whole audited attempts, bounded below.
            max_retries = 1 if directory.name == "attempt_0" else 0
            for retry in range(max_retries + 1):
                if interrupted.is_set():
                    raw["complete"] = False
                    break
                n = int(paths[-1].stem) + 1 if paths else 0
                path = arm_dir / f"{n + retry}.json"
                arm = audit.audit_arm(client, url, task, case["gold"], mode, seconds,
                                      lambda a, checkpoint_path=path: audit.audit_write_json(checkpoint_path, a))
                audit.audit_rebench_verify(arm, case, roots, arm_dir, parser_policy="native-guarded")
                audit.audit_write_json(path, arm)
                if not (retry < max_retries and arm.get("classification") == "transport_failure"
                        and arm.get("cleanup", {}).get("verified")):
                    break
            if not raw["complete"]:
                break
        raw["arms"][mode] = arm
        if not arm.get("cleanup", {}).get("verified"):
            raw["complete"] = False
            break
    raw["status"] = audit.audit_rebench_status(raw)
    audit.audit_write_json(result_path, raw)
    return compact_record(case["inventory"], raw, signature, context)


def finish_image(client, state, image, remove):
    expected = state.get("image_id")
    payload = {"image": image, "lease_id": state["lease_id"], "remove": remove and bool(expected)}
    if expected:
        payload["expected_image_id"] = expected
    try:
        return client.post(state["endpoint"], "/audit_finish_image", payload, time.monotonic() + 90)
    except Exception as exc:
        return {"ok": False, "status": "cleanup_transport_failure", "error": str(exc)[:1000]}


def run_filter(dataset: Path, urls: list[str], output: Path, filtered: Path, *, roots: list[str],
               mappings: list[str] | None = None, pilot_size=0, seed=20261010, workers=32,
               prepare_parallelism=8, prepare_seconds=1800, seconds=900, resume=False,
               image_cleanup="task-images", client=None, resume_cleanup_upgrade=False) -> dict:
    from recipe.swe_agent.image_archives import canonical_image_reference
    if (not urls or len(set(urls)) != len(urls) or type(workers) is not int or workers < 1
            or type(prepare_parallelism) is not int or prepare_parallelism < 1
            or type(pilot_size) is not int or pilot_size < 0 or not 0 < prepare_seconds <= 7200 or seconds <= 0
            or image_cleanup not in {"task-images", "keep"} or not roots):
        raise ValueError("invalid pipeline settings or missing endpoints/evaluator roots")
    for mapping in roots:
        source, sep, destination = mapping.partition("=")
        if not sep or not Path(source).is_absolute() or not Path(destination).is_absolute():
            raise ValueError("evaluator roots require ABSOLUTE_SERVICE_ROOT=ABSOLUTE_LOCAL_ROOT")
    if filtered.resolve() == dataset.resolve():
        raise ValueError("refusing to overwrite the source dataset")
    if not resume and (filtered.exists() or filtered.with_suffix(filtered.suffix + ".audit.json").exists()):
        raise ValueError("refusing to overwrite an existing export")
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (output / ".filter.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("another pipeline is using this output directory") from exc
        client = client or audit.Client()
        mappings = mappings or []
        print(json.dumps({"stage": "inventory", "dataset": str(dataset), "message": "reading source and verifier artifacts"}), flush=True)
        manifest, cases = guarded_inventory(dataset, mappings)
        selection = None
        if pilot_size:
            cases, selection = audit.audit_pilot_selection(cases, pilot_size, seed)
        if not cases or len({c["inventory"].get("task_id") for c in cases}) != len(cases):
            raise ValueError("pipeline requires nonempty unique task identities")
        if image_cleanup == "task-images" and any(not re.fullmatch(
                r"docker\.io/swerebenchv2/[a-zA-Z0-9_.-]+:[a-zA-Z0-9_.-]+",
                canonical_image_reference(c["inventory"].get("docker_image") or ""))
                for c in cases if not c["inventory"]["issues"]):
            raise ValueError("task image cleanup is limited to swerebenchv2 tags; use --image-cleanup keep for other images")
        local = importlib.import_module("recipe.swe_agent.tool_contract_info").source_fingerprint()
        print(json.dumps({"stage": "endpoint_preflight", "tasks": len(cases), "endpoints": len(urls)}), flush=True)
        endpoints = availability.Endpoints(client, urls, local)
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(urls)) as pool:
            list(pool.map(endpoints.refresh, urls))
        # Addresses are recorded for operations, excluded from verifier identity.
        # Actual deployments still require matching source before admission.
        services = [{"url": url, "source": local} for url in urls]
        def preflight(url):
            return endpoints.check(url)
        def cleanup_preflight(url):
            return endpoints.check(url, admission=False)
        code_paths = [Path(__file__), Path(audit.__file__), Path(__file__).with_name("swe_rebench_live_results.py"),
                      Path(__file__).with_name("swe_reference_reliability.py"), Path(__file__).with_name("diagnose_trajectories.py"), Path(availability.__file__)]
        code = {p.name: audit.file_sha256(p) for p in code_paths}
        identity = {"version": VERSION, "manifest": manifest, "selection": [c["inventory"]["id"] for c in cases],
                    "pilot": selection, "services": services, "source": local, "code": code,
                    "seconds": seconds, "prepare_seconds": prepare_seconds, "roots": roots, "mappings": mappings,
                    "image_cleanup": image_cleanup, "filtered": str(filtered.resolve())}
        signature, upgrade = resume_identity(output, identity, resume, resume_cleanup_upgrade)
        execution_signature = upgrade["execution_signature"] if upgrade else signature
        context = {"service_source_sha256": local["sha256"], "audit_code_sha256": audit.digest(code)}
        case_root = output / "cases"
        case_root.mkdir(exist_ok=True)
        states = {}
        # Validate durable proofs here; remote recovery belongs to each task,
        # so one unavailable daemon never blocks admission on all other daemons.
        for case in cases:
            item = case["inventory"]
            directory = case_root / item["id"]
            state_path = directory / "state.json"
            state = json.loads(state_path.read_text()) if state_path.exists() else {}
            if state and state.get("signature") != signature:
                raise ValueError("case checkpoint signature changed")
            if state.get("terminal"):
                path = directory / "qualified_record.json"
                if audit.file_sha256(path) != state.get("qualified_sha256"):
                    raise ValueError("terminal qualification checkpoint changed")
                read_attempts(directory, state)
                record = json.loads(path.read_text())
                if record["status"] == "reference_verified":
                    validate_repeat_qualification(record)
            states[item["id"]] = state
        historical_urls = {s["endpoint"] for s in states.values() if s.get("endpoint")}
        for url in historical_urls - set(urls):
            endpoints.add_cleanup_endpoint(url)
        cleanup_urls = list(dict.fromkeys([*urls, *sorted(historical_urls)]))
        progress = Progress(output, len(cases), signature)
        scheduling = threading.Condition()
        cleanup_holds = {url: set() for url in cleanup_urls}
        uncertain_keys, needs_cleanup = set(), {}
        for case in cases:
            key = case["inventory"]["id"]
            state = states[key]
            if not state or state.get("terminal"):
                continue
            needs_cleanup[key] = migration_cleanup_required(case_root / key, state)
            if any((arm := json.loads(p.read_text())).get("phase") != "complete"
                   or not arm.get("cleanup", {}).get("verified")
                   for p in (case_root / key).glob("attempt_*/*/*.json")):
                cleanup_holds[state["endpoint"]].add(key)
                uncertain_keys.add(key)
        stop = threading.Event()
        slots = {url: threading.BoundedSemaphore(prepare_parallelism) for url in urls}
        errors = []
        dispatch_failures, active_keys = collections.Counter(), set()
        requeues = {}
        recovery_successes = collections.Counter()
        pending = availability.PendingTasks(states, urls, endpoints, cleanup_holds,
            uncertain_keys, needs_cleanup, dispatch_failures)
        for case in cases:
            pending.put(case)
        idle_since = None
        def publish_endpoints():
            with scheduling:
                holds = {url: sorted(keys) for url, keys in cleanup_holds.items() if keys}
            audit.audit_write_json(output / "endpoint_status.json", {
                "version": 1, "endpoints": endpoints.snapshot(), "pending_request_cleanup": holds,
                "dispatch_endpoints": urls, "cleanup_only_endpoints": sorted(historical_urls - set(urls)),
                "updated_at_unix": time.time()})
        publish_endpoints()
        live = LiveResults(output, dataset, signature, manifest["datasets"][0]["sha256"], cases, states, stop)

        def process(case, url):
            item = case["inventory"]
            directory = case_root / item["id"]
            directory.mkdir(exist_ok=True)
            state_path = directory / "state.json"
            state = states[item["id"]]
            historical = bool(state.get("terminal"))
            if not state:
                state.update(signature=signature, endpoint=url, lease_id="rebench-" + uuid.uuid4().hex,
                             attempt_sha256=[], terminal=False, execution_signature=execution_signature)
                audit.audit_write_json(state_path, state)
            destination = url
            url = state["endpoint"]
            phase = "preparation"
            phase_started = time.monotonic()
            progress.stage(None, phase)
            def change_stage(next_phase):
                nonlocal phase, phase_started
                elapsed = time.monotonic() - phase_started
                progress.duration(phase, elapsed)
                timings = state.setdefault("stage_seconds", {})
                timings[phase] = timings.get(phase, 0) + elapsed
                phase_started = time.monotonic()
                state["phase"] = next_phase
                audit.audit_write_json(state_path, state)
                progress.stage(phase, next_phase)
                phase = next_phase
            try:
                if state.get("terminal"):
                    record = json.loads((directory / "qualified_record.json").read_text())
                else:
                    # Cleanup needs Docker and matching code, but no free-space
                    # admission reserve. Full daemons can release old requests.
                    uncertain = any(
                        (arm := json.loads(path.read_text())).get("phase") != "complete"
                        or not arm.get("cleanup", {}).get("verified")
                        for path in directory.glob("attempt_*/*/*.json"))
                    if uncertain:
                        raise CleanupPending()
                    if state.get("migration_pending") or url != destination or state.get("restart_on_failover"):
                        change_stage("migration_cleanup")
                        needs_origin = migration_cleanup_required(directory, state)
                        if needs_origin and not endpoints.cleanup_ready(url):
                            raise availability.EndpointUnavailable("original daemon unavailable for lease cleanup: " + url)
                        if needs_origin:
                            cleanup_preflight(url)
                        plan = state.get("migration_pending")
                        if plan and plan["to"] not in urls:
                            plan.setdefault("previous_destinations", []).append(plan["to"])
                            plan["to"] = destination
                            audit.audit_write_json(state_path, state)
                        migrate_task(client, case, directory, state, destination, execution_signature,
                                     image_cleanup == "task-images")
                        state.pop("restart_on_failover", None)
                        url = state["endpoint"]
                    elif upgrade and state.get("execution_signature") != execution_signature:
                        archive_name = ("before_cleanup_upgrade" if not state.get("execution_signature")
                                        else "before_cleanup_upgrade_" + execution_signature[:16])
                        restart_unfinished_after_upgrade(directory, state, execution_signature, archive_name)
                    preflight(url)
                    attempts, sources = read_attempts(directory, state)
                    need_execution = True
                    if attempts:
                        record, need_execution = qualify_task(attempts, signature, sources)
                    expected = state.get("image_id")
                    if need_execution and not item["issues"]:
                        payload = {"image": item["docker_image"], "timeout_seconds": prepare_seconds, "lease_id": state["lease_id"]}
                        if expected:
                            payload["expected_image_id"] = expected
                        change_stage("image_prepare_wait")
                        with slots[url]:
                            change_stage("image_preparation")
                            preflight(url)
                            state["image_preparation_requested"] = True
                            audit.audit_write_json(state_path, state)
                            try:
                                prepared = client.post(url, "/audit_prepare_image", payload,
                                                       time.monotonic() + prepare_seconds + 10)
                            except Exception as exc:
                                raise availability.EndpointUnavailable("image preparation transport unavailable") from exc
                        for name, elapsed in prepared.get("timings", {}).items():
                            if isinstance(elapsed, (int, float)):
                                progress.duration(name, elapsed)
                        audit.audit_write_json(directory / "image_preparation.json", prepared)
                        state["image_preparation"] = prepared
                        audit.audit_write_json(state_path, state)
                        if prepared.get("failure_kind") in {
                                "transport_failure", "admission_timeout", "docker_unavailable", "disk_space_exhausted",
                                "storage_failure", "registry_network", "preparation_timeout"}:
                            raise availability.EndpointUnavailable("image provisioning unavailable: " + str(prepared))
                        if prepared.get("ok") and IMAGE_ID.fullmatch(prepared.get("id") or ""):
                            if expected and expected != prepared["id"]:
                                raise ValueError("repeat Docker image identity changed")
                            state["image_id"] = prepared["id"]
                            audit.audit_write_json(state_path, state)
                        elif expected or attempts:
                            # A resume preparation failure cannot overwrite an
                            # already successful attempt or dispose of its image.
                            raise ValueError("cannot restore pinned image for unfinished task: " + str(prepared.get("failure_kind")))
                    elif need_execution:
                        prepared = {"ok": False, "failure_kind": "invalid_source_artifacts"}
                    else:
                        prepared = state.get("image_preparation", {})
                    if not state.get("image_id"):
                        raw = {"complete": True, "status": "review:image:" + prepared.get("failure_kind", "image_preparation_unknown"),
                               "data_issues": item["issues"], "arms": {}, "image_preparation": prepared}
                        record = compact_record(item, raw, signature, context)
                    elif need_execution:
                        while True:
                            i = len(attempts)
                            audit_phase = "first_audit" if i == 0 else "repeat_audit" if i == 1 else "retry_audit"
                            change_stage(audit_phase)
                            attempt_dir = directory / f"attempt_{i}"
                            # Recovery remains bounded per attempt. No new
                            # session can start before the old request is clean.
                            before_deployment = preflight(url)
                            attempt = execute_attempt(client, url, case, attempt_dir, seconds, roots,
                                audit.digest([signature, item["id"], i]), context, state["image_id"], stop)
                            if attempt["complete"] is not True:
                                raise CleanupPending()
                            for arm in attempt.get("arms", {}).values():
                                timings = arm.get("timings", {})
                                wait = timings.get("operation_queue_wait_seconds")
                                total = timings.get("operation_end_to_end_seconds")
                                if isinstance(wait, (int, float)):
                                    progress.duration("reward_queue_wait", wait)
                                if isinstance(total, (int, float)) and isinstance(wait, (int, float)):
                                    progress.duration("reward_execution", max(0, total - wait))
                            after_deployment = endpoints.deployment(url)
                            attempt["deployment_evidence"] = {
                                "before": before_deployment, "after": after_deployment,
                                "execution_signature": execution_signature}
                            audit.audit_write_json(attempt_dir / "deployment.json", attempt["deployment_evidence"])
                            # Preserve the established bounded repeat retries.
                            # A first audit containing only confirmed transient
                            # infrastructure failure can instead fail over.
                            if i == 0 and _transient(attempt):
                                raise availability.EndpointUnavailable("first audit infrastructure failure")
                            path = attempt_dir / "record.json"
                            audit.audit_write_json(path, attempt)
                            sha = audit.file_sha256(path)
                            attempts.append(attempt)
                            sources.append({"path": str(path.resolve()), "sha256": sha})
                            state["attempt_sha256"].append(sha)
                            audit.audit_write_json(state_path, state)
                            record, retry = qualify_task(attempts, signature, sources)
                            if not retry:
                                break
                    change_stage("qualification")
                    path = directory / "qualified_record.json"
                    audit.audit_write_json(path, record)
                    state.update(terminal=True, qualified_sha256=audit.file_sha256(path),
                                 deployment_verified=execution_signature)
                    state.pop("blocked", None)
                    # Qualification is durable before any destructive cleanup.
                    audit.audit_write_json(state_path, state)
                try:
                    live.submit(item["id"], state["qualified_sha256"])
                except RuntimeError:
                    # A failed publisher stops dispatch, but in-flight tasks
                    # still release their image leases. close() surfaces the
                    # writer error and prevents final export after workers exit.
                    stop.set()
                change_stage("image_cleanup")
                if state.get("image_cleanup", {}).get("ok") is not True:
                    state["image_cleanup"] = (finish_image(client, state, item["docker_image"], image_cleanup == "task-images")
                        if endpoints.cleanup_ready(url) and state.get("image_preparation") else
                        {"ok": False, "status": "endpoint_unavailable"} if state.get("image_preparation") else
                        {"ok": True, "status": "not_prepared"})
                    audit.audit_write_json(state_path, state)
                progress.finish(state, record, historical=historical)
                print(json.dumps({"task_id": item.get("task_id"), "status": record["status"],
                    "endpoint": url, "attempts": len(state["attempt_sha256"]),
                    "image_cleanup": state["image_cleanup"].get("status")}), flush=True)
            except CleanupPending:
                state["blocked"] = "waiting for original request cleanup: " + url
                audit.audit_write_json(state_path, state)
                with scheduling:
                    uncertain_keys.add(item["id"])
                    cleanup_holds[url].add(item["id"])
                    needs_cleanup[item["id"]] = True
                    requeues[item["id"]] = (case, 0)
                progress.finish(state, None)
            except availability.EndpointUnavailable as exc:
                state["blocked"] = str(exc)[:1500]
                state["restart_on_failover"] = True
                audit.audit_write_json(state_path, state)
                # Failure on the origin during migration must not pause the
                # healthy destination that merely attempted cleanup there.
                endpoints.pause(url, exc)
                with scheduling:
                    dispatch_failures[item["id"]] += 1
                    needs_cleanup[item["id"]] = migration_cleanup_required(directory, state)
                    requeues[item["id"]] = (case, availability.BACKOFF[0])
                    pending.refresh()
                progress.finish(state, None)
                print(json.dumps({"task_id": item.get("task_id"), "endpoint": url,
                                  "deferred": state["blocked"]}), flush=True)
            except Exception as exc:
                state["blocked"] = f"{type(exc).__name__}: {exc}"[:1500]
                audit.audit_write_json(state_path, state)
                with scheduling:
                    errors.append({"task_id": item.get("task_id"), "error": state["blocked"]})
                progress.finish(state, None)
            finally:
                progress.duration(phase, time.monotonic() - phase_started)
                progress.stage(phase, None)

        def monitor():
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=len(cleanup_urls)) as probes:
                    while not stop.wait(availability.MONITOR_INTERVAL):
                        due = endpoints.due()
                        if due:
                            before = {url: (endpoints.ready(url), endpoints.cleanup_ready(url)) for url in due}
                            list(probes.map(endpoints.refresh, due))
                            with scheduling:
                                if any(before[url] != (endpoints.ready(url), endpoints.cleanup_ready(url)) for url in due):
                                    pending.refresh()
                            publish_endpoints()
                            with scheduling:
                                scheduling.notify_all()
            except Exception as exc:
                with scheduling:
                    errors.append({"error": "endpoint_monitor_failed: " + str(exc)[:1000]})
                    stop.set()
                    scheduling.notify_all()

        def recovery_process(case, url):
            key = case["inventory"]["id"]
            directory, state = case_root / key, states[key]
            started = time.monotonic()
            progress.stage(None, "cleanup_recovery")
            try:
                round_key = (key, len(state.get("attempt_sha256", [])))
                ok = recovery_successes[round_key] < 2 and recover_case_cleanup(
                    client, case, directory, lambda: cleanup_preflight(url), stop)
                if ok:
                    recovery_successes[round_key] += 1
                    state.pop("blocked", None)
                    with scheduling:
                        uncertain_keys.discard(key)
                        cleanup_holds[url].discard(key)
                        requeues[key] = (case, 0)
                else:
                    state["blocked"] = "original request cleanup unverified: " + url
                    state["restart_on_failover"] = True
                    endpoints.pause(url, state["blocked"])
                    with scheduling:
                        dispatch_failures[key] += 1
                        if recovery_successes[round_key] >= 2:
                            dispatch_failures[key] = availability.TASK_FAILOVER_LIMIT
                        requeues[key] = (case, availability.BACKOFF[0])
                        pending.refresh()
                    progress.finish(state, None)
                state["phase"] = "cleanup_recovered" if ok else "cleanup_blocked"
                timings = state.setdefault("stage_seconds", {})
                timings["cleanup_recovery"] = timings.get("cleanup_recovery", 0) + time.monotonic() - started
                audit.audit_write_json(directory / "state.json", state)
            except Exception as exc:
                state["blocked"] = f"cleanup recovery failed: {type(exc).__name__}: {exc}"[:1500]
                audit.audit_write_json(directory / "state.json", state)
                with scheduling:
                    errors.append({"task_id": case["inventory"].get("task_id"), "error": state["blocked"]})
                progress.finish(state, None)
            finally:
                progress.duration("cleanup_recovery", time.monotonic() - started)
                progress.stage("cleanup_recovery", None)

        def worker(url, recovery=False):
            nonlocal idle_since
            while not stop.is_set():
                with scheduling:
                    if not pending and not active_keys:
                        return
                    case = pending.take(url, recovery=recovery)
                    now = time.monotonic()
                    if case is None:
                        # Time only stalls, not long-running legitimate tests
                        # or active recovery. No pending-list/filesystem scan.
                        if not active_keys:
                            if idle_since is None:
                                idle_since = now
                            if now - idle_since >= availability.IDLE_TIMEOUT:
                                stop.set()
                                scheduling.notify_all()
                                return
                        scheduling.wait(timeout=availability.MONITOR_INTERVAL)
                        continue
                    key = case["inventory"]["id"]
                    if key in active_keys:
                        raise ValueError("duplicate task dispatch")
                    active_keys.add(key)
                    idle_since = None
                try:
                    if recovery:
                        recovery_process(case, url)
                    else:
                        process(case, url)
                finally:
                    with scheduling:
                        active_keys.remove(key)
                        if key in requeues:
                            item, delay = requeues.pop(key)
                            pending.put(item, delay)
                        scheduling.notify_all()

        monitor_thread = threading.Thread(target=monitor, name="rebench-endpoint-monitor", daemon=True)
        monitor_thread.start()
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=len(urls) * workers)
        recovery_pool = concurrent.futures.ThreadPoolExecutor(max_workers=len(cleanup_urls),
            thread_name_prefix="rebench-cleanup")
        try:
            recovery_futures = [recovery_pool.submit(worker, url, True) for url in cleanup_urls]
            futures = [pool.submit(worker, url) for _ in range(workers) for url in urls]
            for future in futures:
                future.result()
            for future in recovery_futures:
                future.result()
        except BaseException:
            stop.set()
            raise
        finally:
            stop.set()
            pool.shutdown(wait=True, cancel_futures=True)
            recovery_pool.shutdown(wait=True, cancel_futures=True)
            monitor_thread.join()
            publish_endpoints()
            progress.publish(force=True)
            progress.bar.close()
            live.close()
        progress.counts["blocked"] = sum(not s.get("terminal") and bool(s.get("blocked")) for s in states.values())
        report = {"version": VERSION, "signature": signature, "total": len(cases), "source_match": True, "complete": False,
                  "audit_complete": not pending and not errors and all(s.get("terminal") for s in states.values()),
                  "counts": dict(progress.counts),
                  "stage_seconds": dict(progress.stage_seconds),
                  "quarantined_endpoints": sorted(url for url in cleanup_urls
                                                  if not endpoints.ready(url) or cleanup_holds[url]), "errors": errors,
                  "workers_per_endpoint": workers, "image_prepare_parallelism": prepare_parallelism,
                  "image_cleanup": image_cleanup, "pilot": selection,
                  "endpoints": endpoints.snapshot(),
                  "dispatch_endpoints": urls, "cleanup_only_endpoints": sorted(historical_urls - set(urls)),
                  "deferred_tasks": [c["inventory"]["task_id"] for c in pending],
                  "task_dispatch_failures": dict(dispatch_failures),
                  "migrations": sum(len(s.get("migrations", [])) for s in states.values())}
        report["pending_request_cleanup"] = {url: sorted(keys) for url, keys in cleanup_holds.items() if keys}
        if upgrade:
            report["cleanup_upgrade"] = upgrade
        audit.audit_write_json(output / "report.json", report)
        if not report["audit_complete"]:
            raise ValueError("pipeline blocked; resolve cleanup/image/evaluator issue and resume; no dataset exported")
        progress.global_stage = "verifying_source"
        progress.publish(force=True)
        # Sources must remain immutable throughout execution; do not reload all
        # patches or records at once merely to assemble the final ledger.
        if audit.file_sha256(dataset) != manifest["datasets"][0]["sha256"]:
            raise ValueError("source dataset changed during filtering")
        for entry in manifest["cases"]:
            for artifact in entry.get("artifacts", {}).values():
                if audit.file_sha256(Path(artifact["path"])) != artifact["sha256"]:
                    raise ValueError("source verifier artifact changed during filtering")
        if any(audit.file_sha256(p) != code[p.name] for p in code_paths):
            raise ValueError("offline audit code changed during filtering")
        if importlib.import_module("recipe.swe_agent.tool_contract_info").source_fingerprint() != local:
            raise ValueError("local evaluator sources changed during filtering")
        # Endpoint storage/HTTP readiness is not a final evidence criterion.
        # New attempts carry actual deployment checks around execution. Existing
        # terminal proofs remain bound to their historical compatible epoch.
        for case in cases:
            state = states[case["inventory"]["id"]]
            if state.get("execution_signature") == execution_signature:
                if state.get("deployment_verified") != execution_signature:
                    raise ValueError("terminal task lacks verified execution deployment")
                attempts, _ = read_attempts(case_root / case["inventory"]["id"], state)
                for attempt in attempts:
                    deployment = attempt.get("deployment_evidence", {})
                    if (deployment.get("execution_signature") != execution_signature
                            or any(deployment.get(phase, {}).get("source_sha256") != local["sha256"]
                                   for phase in ("before", "after"))):
                        raise ValueError("attempt deployment evidence changed")
        progress.global_stage = "export"
        progress.publish(force=True)
        ledger = output / "qualified_results.jsonl"
        verified = []
        with ledger.open("w") as stream:
            for case in sorted(cases, key=lambda c: c["inventory"]["row"]):
                directory = case_root / case["inventory"]["id"]
                state = states[case["inventory"]["id"]]
                path = directory / "qualified_record.json"
                if audit.file_sha256(path) != state["qualified_sha256"]:
                    raise ValueError("qualification checkpoint changed")
                record = json.loads(path.read_text())
                if record["status"] == "reference_verified":
                    validate_repeat_qualification(record)
                    verified.append(record["task_id"])
                stream.write(json.dumps(record, sort_keys=True) + "\n")
        (output / "reference_verified_task_ids.txt").write_text("".join(t + "\n" for t in verified))
        if not verified:
            report.update(complete=True, reference_verified=0, filtered_dataset=None)
            audit.audit_write_json(output / "report.json", report)
            progress.global_stage = "complete"
            progress.publish(force=True)
            return report
        receipt_path = filtered.with_suffix(filtered.suffix + ".audit.json")
        if filtered.exists() or receipt_path.exists():
            previous = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
            if (not filtered.exists() or previous.get("source_sha256") != manifest["datasets"][0]["sha256"]
                    or previous.get("results_sha256") != audit.file_sha256(ledger)
                    or previous.get("output_sha256") != audit.file_sha256(filtered) or previous.get("repeat_required") is not True):
                raise ValueError("refusing to overwrite an incomplete or different export")
            receipt = previous
        else:
            receipt = audit.audit_export_results(dataset, ledger, filtered, pilot_only=bool(pilot_size), require_repeat=True)
        if upgrade:
            receipt["cleanup_upgrade"] = upgrade
            audit.audit_write_json(receipt_path, receipt)
        report.update(complete=True, reference_verified=len(verified), filtered_dataset=str(filtered.resolve()), receipt=receipt)
        audit.audit_write_json(output / "report.json", report)
        progress.global_stage = "complete"
        progress.publish(force=True)
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--filtered-output", type=Path, required=True)
    parser.add_argument("--evaluator-root", action="append", required=True)
    parser.add_argument("--path-map", action="append", default=[])
    parser.add_argument("--pilot-size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--workers-per-endpoint", type=int, default=32)
    parser.add_argument("--image-prepare-parallelism", type=int, default=8)
    parser.add_argument("--image-prepare-deadline", type=float, default=1800)
    parser.add_argument("--scenario-deadline", type=float, default=900)
    parser.add_argument("--image-cleanup", choices=["task-images", "keep"], default="task-images")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-cleanup-upgrade", action="store_true",
                        help="with --resume, preserve completed work across the known cleanup-only upgrade")
    args = parser.parse_args()
    urls = [s.strip().rstrip("/") for s in os.environ.get("SWE_AGENT_EXECUTION_URLS", "").split(",") if s.strip()]
    report = run_filter(args.dataset, urls, args.output_dir, args.filtered_output, roots=args.evaluator_root,
        mappings=args.path_map, pilot_size=args.pilot_size, seed=args.seed, workers=args.workers_per_endpoint,
        prepare_parallelism=args.image_prepare_parallelism, prepare_seconds=args.image_prepare_deadline,
        seconds=args.scenario_deadline, resume=args.resume, image_cleanup=args.image_cleanup,
        resume_cleanup_upgrade=args.resume_cleanup_upgrade)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

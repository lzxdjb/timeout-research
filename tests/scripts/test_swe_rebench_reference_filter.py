"""Pipeline tests use the real local runner/scorer/private evidence protocol."""

import copy
import hashlib
import json
import shutil
import threading
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from test_swe_tool_online_benchmark import GuardedRebenchClient, rebench_dataset

from recipe.swe_agent import tool_contract_info as info
from scripts import swe_rebench_reference_filter as pipeline


@pytest.fixture(autouse=True)
def short_endpoint_waits(monkeypatch):
    monkeypatch.setattr(pipeline.availability, "BACKOFF", (.02, .03, .04, .05))
    monkeypatch.setattr(pipeline.availability, "IDLE_TIMEOUT", 2.0)
    monkeypatch.setattr(pipeline.availability, "MONITOR_INTERVAL", .01)
    monkeypatch.setattr(pipeline.availability, "READY_PROBE_INTERVAL", 60)


def dataset(tmp_path, count=2):
    path = rebench_dataset(tmp_path, count)
    table = pq.read_table(path)
    rows = table.to_pylist()
    for i, row in enumerate(rows):
        task = json.loads(row["extra_info"]["tools_kwargs"]["__swe_task_json__"])
        task["docker_image"] = f"docker.io/swerebenchv2/fixture:case-{i}"
        row["extra_info"]["tools_kwargs"]["__swe_task_json__"] = json.dumps(task)
    pq.write_table(pa.Table.from_pylist(rows).replace_schema_metadata(table.schema.metadata), path)
    return path


class PipelineClient(GuardedRebenchClient):
    def __init__(self, tmp_path, fault=None):
        super().__init__(tmp_path)
        self.fault = fault
        self.output = tmp_path / "pipeline"
        self.events, self.rewards, self.preparations = [], {}, {}
        self.cleanup_calls = 0

    def get(self, url, path, **kwargs):
        assert "?image=" not in path, "pipeline must not perform per-image startup probes"
        if path in {"/tool_contract_info?summary=1", "/health", "/status"}:
            return {"ok": True, "source_fingerprint": info.source_fingerprint(),
                    "audit_image_lifecycle_version": 1, "docker_available": True, "workspace_admission": {"ok": True}}
        return super().get(url, path, **kwargs)

    def post(self, url, path, payload, deadline):
        if path == "/audit_prepare_image":
            assert "lease_id" in payload
            image = payload["image"]
            self.events.append(("prepare", image, url))
            self.preparations[image] = self.preparations.get(image, 0) + 1
            if self.fault == "prepare":
                return {"ok": False, "failure_kind": "registry_not_found"}
            return {"ok": True, "id": "sha256:" + hashlib.sha256(image.encode()).hexdigest(), "lease_id": payload["lease_id"]}
        if path == "/audit_finish_image":
            self.cleanup_calls += 1
            # Proof must exist on disk before any image can be deleted.
            records = list(self.output.glob("cases/*/qualified_record.json"))
            assert records
            assert not self.active
            self.events.append(("finish", payload["image"], url))
            if self.fault == "cleanup" and self.cleanup_calls == 1:
                return {"ok": False, "status": "image_remove_failed"}
            return {"ok": True, "status": "removed" if payload["remove"] else "kept"}
        if path == "/reward":
            task_id = payload["task"]["task_id"]
            gold = payload["request_id"] in self.patched
            key = (task_id, gold)
            n = self.rewards[key] = self.rewards.get(key, 0) + 1
            self.events.append(("gold" if gold else "baseline", task_id, url))
            assert pipeline.IMAGE_ID.fullmatch(payload["task"]["docker_image"])
            if self.fault == "first_reject" and gold and n == 1 or self.fault == "flaky" and gold and n == 2:
                self.patched.discard(payload["request_id"])
            if self.fault in {"transport", "transport_always", "transport_with_failure"} and gold and n >= 2:
                if n == 2 or self.fault == "transport_always":
                    raise ConnectionError("fixture transport failure")
            if self.fault == "transport_with_failure" and not gold and n == 2:
                self.patched.add(payload["request_id"])
        return super().post(url, path, payload, deadline)


def run(tmp_path, source, client, **kwargs):
    return pipeline.run_filter(source, ["http://one"], tmp_path / "pipeline", tmp_path / "filtered.parquet",
        roots=[f"{client.root}={client.root}"], client=client, workers=1, **kwargs)


def upgrade_identities():
    old_code = {name: pair[0] for name, pair in pipeline.CLEANUP_UPGRADE_CODE.items()}
    old_code.update({"swe_rebench_reference_filter.py": pipeline.CLEANUP_UPGRADE_PIPELINE_FROM,
                     "swe_reference_reliability.py": "unchanged-repeat-proof"})
    old_files = {name: pair[0] for name, pair in pipeline.CLEANUP_UPGRADE_SERVICE.items()}
    old_files["test_results.py"] = "unchanged-parser"
    old_source = {"version": 1, "files": old_files, "sha256": pipeline.audit.digest(old_files)}
    old = {"code": old_code, "source": old_source, "services": [{"url": "http://one", "source": old_source}],
           "manifest": {"sha256": "dataset-and-artifacts"}, "filtered": "/unused/output.parquet", "seconds": 900}
    new = copy.deepcopy(old)
    new["code"].update({name: pair[1] for name, pair in pipeline.CLEANUP_UPGRADE_CODE.items()})
    new["code"]["swe_rebench_reference_filter.py"] = "current-pipeline"
    new["source"]["files"].update({name: pair[1] for name, pair in pipeline.CLEANUP_UPGRADE_SERVICE.items()})
    new["source"]["sha256"] = pipeline.audit.digest(new["source"]["files"])
    return old, new


def test_cleanup_upgrade_epoch_keeps_original_manifest_and_remains_strict(tmp_path):
    old, new = upgrade_identities()
    run_id, epoch = pipeline.resume_identity(tmp_path, old, False, False)
    before = (tmp_path / "manifest.json").read_bytes()
    with pytest.raises(ValueError, match="changed"):
        pipeline.resume_identity(tmp_path, new, True, False)
    assert pipeline.resume_identity(tmp_path, new, True, True)[0] == run_id
    assert (tmp_path / "manifest.json").read_bytes() == before
    # Following restarts need only --resume, but cannot silently upgrade again.
    assert pipeline.resume_identity(tmp_path, new, True, False)[0] == run_id
    changed = copy.deepcopy(new)
    changed["seconds"] = 901
    with pytest.raises(ValueError, match="epoch changed"):
        pipeline.resume_identity(tmp_path, changed, True, True)
    with pytest.raises(ValueError, match="requires --resume"):
        pipeline.resume_identity(tmp_path, new, False, True)


def test_scheduling_upgrade_can_extend_existing_cleanup_epoch(tmp_path):
    old, cleanup = upgrade_identities()
    cleanup["code"]["swe_rebench_reference_filter.py"] = pipeline.SCHEDULING_UPGRADE_PIPELINE_FROM
    run_id, _ = pipeline.resume_identity(tmp_path, old, False, False)
    _, previous = pipeline.resume_identity(tmp_path, cleanup, True, True)
    current = copy.deepcopy(cleanup)
    current["code"]["swe_rebench_reference_filter.py"] = "current-scheduling-fix"
    current["code"]["swe_rebench_endpoints.py"] = "endpoint-helper"
    with pytest.raises(ValueError, match="epoch changed"):
        pipeline.resume_identity(tmp_path, current, True, False)
    kept_id, epoch = pipeline.resume_identity(tmp_path, current, True, True)
    assert kept_id == run_id and epoch["previous_epochs"] == [previous]
    assert pipeline.resume_identity(tmp_path, current, True, False) == (run_id, epoch)


def test_endpoint_fingerprint_upgrade_accepts_known_helper_change(tmp_path):
    old, scheduling = upgrade_identities()
    scheduling["code"]["swe_rebench_reference_filter.py"] = pipeline.ENDPOINT_UPGRADE_PIPELINE_FROM
    scheduling["code"]["swe_rebench_endpoints.py"] = pipeline.ENDPOINT_UPGRADE_HELPER_FROM
    pipeline.resume_identity(tmp_path, old, False, False)
    pipeline.resume_identity(tmp_path, scheduling, True, True)
    current = copy.deepcopy(scheduling)
    current["code"]["swe_rebench_reference_filter.py"] = "endpoint-fingerprint-fix"
    current["code"]["swe_rebench_endpoints.py"] = "cleanup-only-endpoints-fix"
    current["services"] = [{"url": "http://replacement", "source": current["source"]}]
    assert pipeline.resume_identity(tmp_path, current, True, True)[1]["execution_signature"] == (
        pipeline.execution_digest(current))


@pytest.mark.parametrize("change", ["parser", "repeat", "dataset", "settings", "unknown-service-fix"])
def test_cleanup_upgrade_rejects_unrelated_changes(change):
    old, new = upgrade_identities()
    if change == "parser":
        new["source"]["files"]["test_results.py"] = "changed"
        new["source"]["sha256"] = pipeline.audit.digest(new["source"]["files"])
    elif change == "repeat":
        new["code"]["swe_reference_reliability.py"] = "changed"
    elif change == "dataset":
        new["manifest"] = {"sha256": "changed"}
    elif change == "settings":
        new["seconds"] = 901
    else:
        new["source"]["files"]["remote_execution_service.py"] = "unknown"
        new["source"]["sha256"] = pipeline.audit.digest(new["source"]["files"])
    with pytest.raises(ValueError):
        pipeline.validate_cleanup_upgrade(old, new)


def test_endpoint_only_changes_preserve_existing_execution_epoch(tmp_path):
    old, new = upgrade_identities()
    run_id, _ = pipeline.resume_identity(tmp_path, old, False, False)
    _, epoch = pipeline.resume_identity(tmp_path, new, True, True)
    epoch_before = (tmp_path / "cleanup_upgrade.json").read_bytes()
    changed = copy.deepcopy(new)
    changed["services"] = [{"url": "http://added", "source": new["source"]}]
    assert pipeline.resume_identity(tmp_path, changed, True, False) == (run_id, epoch)
    assert (tmp_path / "cleanup_upgrade.json").read_bytes() == epoch_before
    assert pipeline.execution_digest(changed) == pipeline.execution_digest(new)


def test_legacy_endpoint_only_resume_preserves_original_run_signature(tmp_path):
    old, _ = upgrade_identities()
    legacy_signature = pipeline.audit.digest(old)
    pipeline.audit.audit_write_json(tmp_path / "manifest.json", {
        "version": pipeline.VERSION, "identity": old, "signature": legacy_signature})
    changed = copy.deepcopy(old)
    changed["services"] = [{"url": "http://added", "source": old["source"]}]
    assert pipeline.resume_identity(tmp_path, changed, True, False) == (legacy_signature, None)


def test_legacy_epoch_endpoint_change_preserves_execution_context(tmp_path):
    old, cleanup = upgrade_identities()
    legacy_signature = pipeline.audit.digest(old)
    pipeline.audit.audit_write_json(tmp_path / "manifest.json", {
        "version": pipeline.VERSION, "identity": old, "signature": legacy_signature})
    epoch = {"version": 1, "run_signature": legacy_signature,
             "execution_signature": pipeline.audit.digest(cleanup),
             "code": cleanup["code"], "source": cleanup["source"], "services": cleanup["services"]}
    pipeline.audit.audit_write_json(tmp_path / "cleanup_upgrade.json", epoch)
    changed = copy.deepcopy(cleanup)
    changed["services"] = [{"url": "http://added", "source": cleanup["source"]}]
    assert pipeline.resume_identity(tmp_path, changed, True, False) == (legacy_signature, epoch)


def test_cleanup_upgrade_preserves_terminal_parts_restarts_only_unfinished_task(tmp_path, monkeypatch):
    source = dataset(tmp_path, 2)
    client = PipelineClient(tmp_path)
    real_hash, real_fingerprint = pipeline.audit.file_sha256, info.source_fingerprint
    real_execute = pipeline.execute_attempt
    legacy = True

    def fingerprint():
        current = real_fingerprint()
        if legacy:
            for name, pair in pipeline.CLEANUP_UPGRADE_SERVICE.items():
                current["files"][name] = pair[0]
            for name, pair in pipeline.PERFORMANCE_UPGRADE_SERVICE.items():
                if name in pipeline.CLEANUP_UPGRADE_SERVICE:
                    continue
                if pair[0] is None:
                    current["files"].pop(name, None)
                else:
                    current["files"][name] = pair[0]
            current["sha256"] = pipeline.audit.digest(current["files"])
        return current

    def code_hash(path):
        if legacy and path.parent == Path(pipeline.__file__).parent:
            if path.name == "swe_rebench_reference_filter.py":
                return pipeline.CLEANUP_UPGRADE_PIPELINE_FROM
            if path.name == "swe_rebench_endpoints.py":
                return pipeline.PERFORMANCE_UPGRADE_HELPER_FROM
            if path.name in pipeline.CLEANUP_UPGRADE_CODE:
                return pipeline.CLEANUP_UPGRADE_CODE[path.name][0]
        return real_hash(path)

    def interrupt(*args, **kwargs):
        if args[2]["inventory"]["row"] == 1 and args[3].name == "attempt_1":
            raise RuntimeError("fixture old service stopped after first clean audit")
        return real_execute(*args, **kwargs)

    monkeypatch.setattr(info, "source_fingerprint", fingerprint)
    monkeypatch.setattr(pipeline.audit, "file_sha256", code_hash)
    monkeypatch.setattr(pipeline, "execute_attempt", interrupt)
    with pytest.raises(ValueError, match="pipeline blocked"):
        run(tmp_path, source, client)
    root = tmp_path / "pipeline"
    manifest_before = (root / "manifest.json").read_bytes()
    states = {p: json.loads(p.read_text()) for p in root.glob("cases/*/state.json")}
    terminal_path = next(p.parent / "qualified_record.json" for p, s in states.items() if s["terminal"])
    terminal_before = terminal_path.read_bytes()
    parts_before = {p: p.read_bytes() for p in root.glob("live_parquet_parts/*")}
    assert len(parts_before) == 2 and sum(client.rewards.values()) == 6
    legacy = False
    monkeypatch.setattr(pipeline, "execute_attempt", real_execute)
    with pytest.raises(ValueError, match="changed"):
        run(tmp_path, source, client, resume=True)
    report = run(tmp_path, source, client, resume=True, resume_cleanup_upgrade=True)
    assert report["reference_verified"] == 2
    assert (root / "manifest.json").read_bytes() == manifest_before
    assert terminal_path.read_bytes() == terminal_before
    assert all(p.read_bytes() == before for p, before in parts_before.items())
    # First terminal task is never rerun; unfinished task gets two NEW runs.
    assert sum(client.rewards.values()) == 10
    archives = list(root.glob("cases/*/before_cleanup_upgrade*/state.json"))
    assert len(archives) == 1
    assert json.loads(archives[0].read_text())["attempt_sha256"]
    receipt = json.loads((tmp_path / "filtered.parquet.audit.json").read_text())
    assert receipt["cleanup_upgrade"] == report["cleanup_upgrade"]
    assert run(tmp_path, source, client, resume=True)["reference_verified"] == 2
    assert sum(client.rewards.values()) == 10


def test_unfinished_upgrade_archive_can_resume_after_partial_move(tmp_path, monkeypatch):
    directory = tmp_path / "case"
    directory.mkdir()
    state = {"signature": "original-run", "attempt_sha256": [], "terminal": False, "blocked": "old failure"}
    for i in range(2):
        path = directory / f"attempt_{i}" / "record.json"
        path.parent.mkdir()
        path.write_text(json.dumps({"attempt": i}))
        state["attempt_sha256"].append(pipeline.audit.file_sha256(path))
    original = Path.rename
    def interrupted(path, destination):
        if path.name == "attempt_1":
            raise OSError("fixture interrupted archive")
        return original(path, destination)
    monkeypatch.setattr(Path, "rename", interrupted)
    with pytest.raises(OSError):
        pipeline.restart_unfinished_after_upgrade(directory, state, "new-epoch")
    monkeypatch.setattr(Path, "rename", original)
    pipeline.restart_unfinished_after_upgrade(directory, state, "new-epoch")
    assert not list(directory.glob("attempt_*"))
    assert len(list(directory.glob("before_cleanup_upgrade/attempt_*"))) == 2
    assert state["attempt_sha256"] == [] and state["execution_signature"] == "new-epoch"
    assert "blocked" not in state


def test_task_repeat_and_cleanup_before_dispatching_next_task(tmp_path):
    source = dataset(tmp_path)
    client = PipelineClient(tmp_path)
    report = run(tmp_path, source, client)
    assert report["reference_verified"] == 2
    assert report["counts"]["images_removed"] == 2
    assert [e[0] for e in client.events] == ["prepare", "baseline", "gold", "baseline", "gold", "finish"] * 2
    assert pq.read_table(source).equals(pq.read_table(tmp_path / "filtered.parquet"), check_metadata=True)
    records = [json.loads(s) for s in (tmp_path / "pipeline/qualified_results.jsonl").read_text().splitlines()]
    assert all(r["repeat_qualification"]["clean_runs"] == 2 for r in records)
    assert all(len({a["request_id"] for t in r["repeat_qualification"]["attempts"] for a in t["arms"].values()}) == 4 for r in records)
    events = len(client.events)
    assert run(tmp_path, source, client, resume=True)["reference_verified"] == 2
    assert len(client.events) == events


@pytest.mark.parametrize("fault,expected,attempts", [("first_reject", 0, 1), ("flaky", 0, 2),
    ("transport", 1, 3), ("transport_always", 0, 4), ("transport_with_failure", 0, 2)])
def test_repeat_gate_veto_and_bounded_transient_retries(tmp_path, fault, expected, attempts):
    source = dataset(tmp_path, 1)
    client = PipelineClient(tmp_path, fault)
    report = run(tmp_path, source, client)
    assert report["reference_verified"] == expected
    assert sum(client.rewards.values()) == attempts * 2
    assert len(client.preparations) == 1
    assert client.events[-1][0] == "finish"
    state = json.loads(next((tmp_path / "pipeline/cases").glob("*/state.json")).read_text())
    assert state["terminal"] and len(state["attempt_sha256"]) == attempts


def test_cleanup_failure_is_visible_and_resume_only_retries_cleanup(tmp_path):
    source = dataset(tmp_path, 1)
    client = PipelineClient(tmp_path, "cleanup")
    report = run(tmp_path, source, client)
    assert report["reference_verified"] == report["counts"]["cleanup_pending"] == 1
    n = sum(client.rewards.values())
    again = run(tmp_path, source, client, resume=True)
    assert sum(client.rewards.values()) == n and client.cleanup_calls == 2
    assert again["counts"]["cleanup_pending"] == 0


def test_prepare_failure_is_excluded_and_lease_released_without_deletion(tmp_path):
    source = dataset(tmp_path, 1)
    client = PipelineClient(tmp_path, "prepare")
    report = run(tmp_path, source, client)
    assert report["reference_verified"] == 0 and not client.rewards
    assert client.events[-1][0] == "finish"
    assert not (tmp_path / "filtered.parquet").exists()


def test_resume_after_durable_attempt_does_not_run_third_audit(tmp_path, monkeypatch):
    source = dataset(tmp_path, 1)
    client = PipelineClient(tmp_path)
    original = pipeline.audit.audit_write_json
    def interrupt(path, value):
        if path.name == "qualified_record.json":
            raise RuntimeError("fixture interrupted after attempts")
        return original(path, value)
    monkeypatch.setattr(pipeline.audit, "audit_write_json", interrupt)
    with pytest.raises(ValueError, match="pipeline blocked"):
        run(tmp_path, source, client)
    assert sum(client.rewards.values()) == 4 and client.cleanup_calls == 0
    monkeypatch.setattr(pipeline.audit, "audit_write_json", original)
    assert run(tmp_path, source, client, resume=True)["reference_verified"] == 1
    assert sum(client.rewards.values()) == 4 and client.cleanup_calls == 1


def test_source_mismatch_fails_before_preparation(tmp_path):
    source = dataset(tmp_path, 1)
    client = PipelineClient(tmp_path)
    original = client.get
    def get(url, path, **kwargs):
        result = original(url, path, **kwargs)
        if path == "/tool_contract_info?summary=1":
            result["source_fingerprint"] = {"sha256": "old-service"}
        return result
    client.get = get
    with pytest.raises(ValueError, match="pipeline blocked"):
        run(tmp_path, source, client)
    assert client.events == []


def test_resumed_image_identity_cannot_change(tmp_path, monkeypatch):
    source = dataset(tmp_path, 1)
    client = PipelineClient(tmp_path)
    original = pipeline.execute_attempt
    def interrupt(*args, **kwargs):
        if args[3].name == "attempt_1":
            raise RuntimeError("fixture interruption")
        return original(*args, **kwargs)
    monkeypatch.setattr(pipeline, "execute_attempt", interrupt)
    with pytest.raises(ValueError, match="pipeline blocked"):
        run(tmp_path, source, client)
    assert sum(client.rewards.values()) == 2
    monkeypatch.setattr(pipeline, "execute_attempt", original)
    original_post = client.post
    def changed(url, path, payload, deadline):
        result = original_post(url, path, payload, deadline)
        if path == "/audit_prepare_image":
            assert "expected_image_id" in payload
            result["id"] = "sha256:" + "f" * 64
        return result
    client.post = changed
    with pytest.raises(ValueError, match="pipeline blocked"):
        run(tmp_path, source, client, resume=True)
    assert sum(client.rewards.values()) == 2 and client.cleanup_calls == 0


def test_same_endpoint_workers_are_parallel_and_preparations_are_bounded(tmp_path):
    source = dataset(tmp_path, 4)
    client = PipelineClient(tmp_path)
    # Concurrent cases can still own sessions when another case finishes. The
    # real service checks image-specific usage, not every image's sessions.
    original = client.post
    def post(url, path, payload, deadline):
        if path == "/audit_finish_image":
            client.cleanup_calls += 1
            return {"ok": True, "status": "removed"}
        return original(url, path, payload, deadline)
    client.post = post
    report = pipeline.run_filter(source, ["http://one"], tmp_path / "pipeline", tmp_path / "filtered.parquet",
        roots=[f"{client.root}={client.root}"], client=client, workers=4, prepare_parallelism=2)
    assert report["reference_verified"] == 4 and client.max_active > 1


def test_resume_rejects_changed_terminal_proof(tmp_path):
    source = dataset(tmp_path, 1)
    client = PipelineClient(tmp_path)
    run(tmp_path, source, client)
    path = next((tmp_path / "pipeline/cases").glob("*/qualified_record.json"))
    path.write_text("{}")
    with pytest.raises(ValueError, match="checkpoint changed"):
        run(tmp_path, source, client, resume=True)


@pytest.mark.parametrize("fault", ["docker", "storage"])
def test_unready_endpoint_fails_preflight_before_image_preparation(tmp_path, fault):
    source = dataset(tmp_path, 1)
    client = PipelineClient(tmp_path)
    original = client.get
    def get(url, path, **kwargs):
        result = original(url, path, **kwargs)
        if path == "/status":
            if fault == "docker":
                result["docker_available"] = False
            else:
                result["workspace_admission"] = {"ok": False, "error_code": "workspace_storage_pressure"}
        return result
    client.get = get
    with pytest.raises(ValueError, match="pipeline blocked"):
        run(tmp_path, source, client)
    assert not client.preparations


class CleanupRecoveryClient(PipelineClient):
    def __init__(self, tmp_path, persistent=False):
        super().__init__(tmp_path)
        self.persistent = persistent
        self.release_failures = self.recoveries = 0
        self.failed_requests = set()

    def post(self, url, path, payload, deadline):
        if path == "/release":
            request = payload["request_id"]
            self.active.discard(request)
            if self.persistent or not self.release_failures:
                self.release_failures += 1
                self.failed_requests.add(request)
                raise RuntimeError("container removal already in progress")
            if request in self.failed_requests:
                self.recoveries += 1
            return {"ok": True, "released": True, "container_removed": True}
        return super().post(url, path, payload, deadline)


def test_transient_cleanup_recovers_without_replaying_finished_test_or_quarantining(tmp_path):
    source = dataset(tmp_path)
    client = CleanupRecoveryClient(tmp_path)
    report = run(tmp_path, source, client)
    assert report["reference_verified"] == 2 and report["quarantined_endpoints"] == []
    assert client.recoveries == 1 and sum(client.rewards.values()) == 8
    assert report["counts"].get("blocked", 0) == 0
    recovery = next(client.output.glob("cases/*/cleanup_recovery.json"))
    assert json.loads(recovery.read_text())["ok"] is True


def test_cleanup_recovery_rechecks_docker_and_source_before_release(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "RECOVERY_BACKOFF", (0, 0, 0))
    source = dataset(tmp_path, 1)
    client = CleanupRecoveryClient(tmp_path)
    original_get = client.get
    not_ready = 0
    def get(url, path, **kwargs):
        nonlocal not_ready
        result = original_get(url, path, **kwargs)
        if path == "/status" and client.release_failures and not not_ready:
            assert client.recoveries == 0
            not_ready += 1
            result["docker_available"] = False
        return result
    client.get = get
    assert run(tmp_path, source, client)["reference_verified"] == 1
    recovery = json.loads(next(client.output.glob("cases/*/cleanup_recovery.json")).read_text())
    assert len(recovery["attempts"]) == 2 and "Docker" in recovery["attempts"][0]["error"]
    assert client.recoveries == 1


def test_cleanup_recovery_preserves_a_failed_gold_result(tmp_path):
    source = dataset(tmp_path, 1)
    client = CleanupRecoveryClient(tmp_path)
    client.fault = "first_reject"
    # Put the transient cleanup fault on gold instead of baseline.
    client.release_failures = 1
    original_post = client.post
    failed = False
    def post(url, path, payload, deadline):
        nonlocal failed
        if path == "/release" and client.rewards.get(("case-0", True)) and not failed:
            client.active.discard(payload["request_id"])
            client.failed_requests.add(payload["request_id"])
            failed = True
            raise RuntimeError("transient release failure")
        return original_post(url, path, payload, deadline)
    client.post = post
    report = run(tmp_path, source, client)
    assert failed and client.recoveries == 1
    assert report["reference_verified"] == 0 and report["quarantined_endpoints"] == []
    assert sum(client.rewards.values()) == 2
    assert not (tmp_path / "filtered.parquet").exists()


def test_persistent_cleanup_is_bounded_and_blocks_new_dispatch_and_export(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "RECOVERY_BACKOFF", (0, 0, 0))
    source = dataset(tmp_path)
    client = CleanupRecoveryClient(tmp_path, persistent=True)
    with pytest.raises(ValueError, match="pipeline blocked"):
        run(tmp_path, source, client)
    assert len(client.preparations) == 1 and sum(client.rewards.values()) == 1
    assert client.release_failures <= 10  # Initial release and bounded recovery sequences.
    assert not (tmp_path / "filtered.parquet").exists()
    report = json.loads((client.output / "report.json").read_text())
    assert report["counts"]["blocked"] == 1 and report["quarantined_endpoints"] == ["http://one"]
    client.persistent = False
    assert run(tmp_path, source, client, resume=True)["reference_verified"] == 2
    assert sum(client.rewards.values()) == 9  # Unfinished task restarts after its old request is clean.


def test_endpoint_pause_prevents_dispatch_until_cleanup_recovery_finishes(tmp_path, monkeypatch):
    source = dataset(tmp_path, 3)
    client = CleanupRecoveryClient(tmp_path)
    entered, allow_recovery = threading.Event(), threading.Event()
    original = pipeline.recover_case_cleanup
    def recover(*args, **kwargs):
        entered.set()
        assert allow_recovery.wait(5)
        assert len(client.preparations) <= 2
        return original(*args, **kwargs)
    monkeypatch.setattr(pipeline, "recover_case_cleanup", recover)
    # Two workers: while one recovers, the second finishes its current case but
    # must not prepare the remaining case on that endpoint.
    original_post = client.post
    def post(url, path, payload, deadline):
        if path == "/audit_finish_image":
            client.cleanup_calls += 1
            return {"ok": True, "status": "removed"}
        return original_post(url, path, payload, deadline)
    client.post = post
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(pipeline.run_filter, source, ["http://one"], client.output, tmp_path / "filtered.parquet",
            roots=[f"{client.root}={client.root}"], client=client, workers=2)
        try:
            assert entered.wait(5)
            end = time.monotonic() + 5
            while not client.cleanup_calls and time.monotonic() < end:
                time.sleep(.01)
            assert client.cleanup_calls == 1 and len(client.preparations) == 2
        finally:
            allow_recovery.set()
        assert future.result()["reference_verified"] == 3


def live_fixture(tmp_path, fault=None, count=2):
    source = dataset(tmp_path, count)
    # Exercise extraction across distinct source row groups as well as parts
    # that receive completion events in a different order from source rows.
    pq.write_table(pq.read_table(source), source, row_group_size=1)
    client = PipelineClient(tmp_path, fault)
    report = run(tmp_path, source, client)
    output = tmp_path / "live-unit"
    shutil.copytree(tmp_path / "pipeline/cases", output / "cases")
    states, cases = {}, []
    for path in sorted((output / "cases").glob("*/qualified_record.json")):
        record = json.loads(path.read_text())
        states[record["id"]] = json.loads((path.parent / "state.json").read_text())
        cases.append({"inventory": {k: record[k] for k in ("id", "row", "task_id")}})
    return source, output, cases, states, report["signature"]


def new_live(source, output, cases, states, signature, stop=None, **kwargs):
    return pipeline.LiveResults(output, source, signature, pipeline.audit.file_sha256(source),
        cases, states, stop or threading.Event(), **kwargs)


def wait_until(predicate, writer, timeout=5):
    deadline = time.monotonic() + timeout
    while not predicate():
        writer.check()
        assert time.monotonic() < deadline, "live publication timed out"
        time.sleep(.01)


def test_live_publication_and_readable_parts_before_writer_closes(tmp_path):
    source, output, cases, states, signature = live_fixture(tmp_path)
    writer = new_live(source, output, cases, {}, signature, flush_seconds=.02, part_rows=1)
    try:
        # Publish out of source order, and exercise duplicate completion events.
        for case in reversed(cases):
            key = case["inventory"]["id"]
            writer.submit(key, states[key]["qualified_sha256"])
            writer.submit(key, states[key]["qualified_sha256"])
        def published():
            path = output / "live_summary.json"
            return path.exists() and json.loads(path.read_text()).get("parquet_rows") == 2
        wait_until(published, writer)
        summary = json.loads((output / "live_summary.json").read_text())
        assert summary["complete"] is False and summary["provisional"] is True
        assert summary["terminal"] == summary["reference_verified"] == 2
        records = [json.loads(line) for line in (output / "live_results.jsonl").read_text().splitlines()]
        assert [r["id"] for r in records] == [c["inventory"]["id"] for c in reversed(cases)]
        assert len((output / "live_reference_verified_task_ids.txt").read_text().splitlines()) == 2
        for part in summary["parts"]:
            receipt = json.loads(Path(part["receipt"]).read_text())
            table = pq.read_table(part["path"])
            assert table.equals(pq.read_table(source).take(receipt["retained_row_indices"]), check_metadata=True)
            assert receipt["repeat_required"] and receipt["provisional"]
        assert writer.thread.is_alive()
    finally:
        writer.close()


def test_live_timer_and_graceful_close_flush_small_parts(tmp_path):
    source, output, cases, states, signature = live_fixture(tmp_path)
    writer = new_live(source, output, cases, {}, signature, flush_seconds=.01, part_seconds=.02)
    try:
        key = cases[0]["inventory"]["id"]
        writer.submit(key, states[key]["qualified_sha256"])
        wait_until(lambda: bool(list((output / "live_parquet_parts").glob("*.parquet.audit.json"))), writer)
        key = cases[1]["inventory"]["id"]
        writer.submit(key, states[key]["qualified_sha256"])
    finally:
        writer.close()
    assert len(list((output / "live_parquet_parts").glob("*.parquet"))) == 2
    assert json.loads((output / "live_summary.json").read_text())["reference_verified"] == 2


def test_live_resume_rebuilds_torn_lists_reuses_parts_and_discards_orphans(tmp_path):
    source, output, cases, states, signature = live_fixture(tmp_path)
    writer = new_live(source, output, cases, states, signature)
    writer.close()
    parts = list((output / "live_parquet_parts").glob("*.parquet"))
    original = {p: p.read_bytes() for p in parts}
    (output / "live_results.jsonl").write_text('{"torn":')
    (output / "live_reference_verified_task_ids.txt").write_text("duplicate\nduplicate\n")
    orphan = output / "live_parquet_parts/part-999999.parquet"
    orphan.write_bytes(b"incomplete publication")
    writer = new_live(source, output, cases, states, signature)
    writer.close()
    assert not orphan.exists()
    assert all(p.read_bytes() == value for p, value in original.items())
    assert len(list((output / "live_parquet_parts").glob("*.parquet"))) == 1
    assert len((output / "live_results.jsonl").read_text().splitlines()) == 2
    assert len(set((output / "live_reference_verified_task_ids.txt").read_text().splitlines())) == 2


def test_live_rejected_tasks_never_enter_ids_or_parts(tmp_path):
    source, output, cases, states, signature = live_fixture(tmp_path, "flaky", count=1)
    writer = new_live(source, output, cases, states, signature)
    writer.close()
    summary = json.loads((output / "live_summary.json").read_text())
    assert summary["rejected"] == 1 and summary["reference_verified"] == 0
    assert (output / "live_reference_verified_task_ids.txt").read_text() == ""
    assert not list((output / "live_parquet_parts").glob("*.parquet"))


def test_live_resume_rejects_changed_part(tmp_path):
    source, output, cases, states, signature = live_fixture(tmp_path, count=1)
    writer = new_live(source, output, cases, states, signature)
    writer.close()
    next((output / "live_parquet_parts").glob("*.parquet")).write_bytes(b"changed")
    with pytest.raises(ValueError, match="content changed"):
        new_live(source, output, cases, states, signature)


def test_live_writer_failure_stops_dispatch_preserves_proofs_and_prevents_final_export(tmp_path, monkeypatch):
    source = dataset(tmp_path, 3)
    client = PipelineClient(tmp_path)
    original_init = pipeline.LiveResults.__init__
    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs, part_rows=1, flush_seconds=.01)
    def fail(self):
        raise OSError("fixture disk full")
    monkeypatch.setattr(pipeline.LiveResults, "__init__", init)
    monkeypatch.setattr(pipeline.LiveResults, "_part", fail)
    original_post = client.post
    def post(url, path, payload, deadline):
        result = original_post(url, path, payload, deadline)
        if path == "/audit_finish_image":
            end = time.monotonic() + 5
            while time.monotonic() < end:
                summary = client.output / "live_summary.json"
                if summary.exists() and "writer_error" in json.loads(summary.read_text()):
                    break
                time.sleep(.01)
            else:
                pytest.fail("writer failure was not reported")
        return result
    client.post = post
    with pytest.raises(RuntimeError, match="live result writer failed"):
        run(tmp_path, source, client)
    assert len(client.preparations) == 1 and client.cleanup_calls == 1
    assert not (tmp_path / "filtered.parquet").exists()
    assert len(list(client.output.glob("cases/*/qualified_record.json"))) == 1
    monkeypatch.undo()
    client.post = original_post
    assert run(tmp_path, source, client, resume=True)["reference_verified"] == 3


def test_deployment_drift_blocks_qualification_without_poisoning_completed_task(tmp_path):
    source = dataset(tmp_path, 2)
    client = PipelineClient(tmp_path)
    original = client.get
    def get(url, path, **kwargs):
        result = original(url, path, **kwargs)
        if path == "/tool_contract_info?summary=1" and ("case-1", True) in client.rewards:
            result["source_fingerprint"] = {"sha256": "changed-service"}
        return result
    client.get = get
    with pytest.raises(ValueError, match="pipeline blocked"):
        run(tmp_path, source, client)
    summary = json.loads((client.output / "live_summary.json").read_text())
    assert summary["parquet_rows"] == 1 and summary["provisional"] and not summary["complete"]
    assert not (tmp_path / "filtered.parquet").exists()


def performance_identities():
    old, current = upgrade_identities()
    old['code']['swe_rebench_reference_filter.py'] = pipeline.PERFORMANCE_UPGRADE_PIPELINE_FROM
    for name, pair in pipeline.PERFORMANCE_UPGRADE_CODE.items():
        old['code'][name] = pair[0]
    for name, pair in pipeline.PERFORMANCE_UPGRADE_SERVICE.items():
        if pair[0] is not None:
            old['source']['files'][name] = pair[0]
    old['source']['sha256'] = pipeline.audit.digest(old['source']['files'])
    old['services'] = [{'url': 'http://old', 'source': old['source']}]
    current = copy.deepcopy(old)
    current['code']['swe_rebench_reference_filter.py'] = pipeline.audit.file_sha256(Path(pipeline.__file__))
    for name, pair in pipeline.PERFORMANCE_UPGRADE_CODE.items():
        current['code'][name] = pair[1]
    for name, pair in pipeline.PERFORMANCE_UPGRADE_SERVICE.items():
        current['source']['files'][name] = pair[1]
    current['source']['sha256'] = pipeline.audit.digest(current['source']['files'])
    current['services'] = [{'url': 'http://new', 'source': current['source']}]
    return old, current


def test_performance_upgrade_preserves_epoch_and_rejects_parser_changes(tmp_path):
    old, current = performance_identities()
    run_id, _ = pipeline.resume_identity(tmp_path, old, False, False)
    manifest = (tmp_path / 'manifest.json').read_bytes()
    with pytest.raises(ValueError, match='changed'):
        pipeline.resume_identity(tmp_path, current, True, False)
    signature, epoch = pipeline.resume_identity(tmp_path, current, True, True)
    assert signature == run_id
    assert (tmp_path / 'manifest.json').read_bytes() == manifest
    assert pipeline.resume_identity(tmp_path, current, True, False) == (run_id, epoch)
    bad = copy.deepcopy(current)
    bad['source']['files']['test_results.py'] = 'different-parser'
    bad['source']['sha256'] = pipeline.audit.digest(bad['source']['files'])
    bad['services'][0]['source'] = bad['source']
    with pytest.raises(ValueError, match='protected'):
        pipeline.validate_cleanup_upgrade(old, bad)


def test_performance_hash_allowlist_matches_actual_transport_files():
    for name, pair in pipeline.PERFORMANCE_UPGRADE_CODE.items():
        assert pipeline.audit.file_sha256(Path(pipeline.__file__).with_name(name)) == pair[1]
    root = Path(info.__file__).parent
    for name, pair in pipeline.PERFORMANCE_UPGRADE_SERVICE.items():
        assert pipeline.audit.file_sha256(root / name) == pair[1]


def test_resume_progress_excludes_historical_results_from_rate(tmp_path):
    progress = pipeline.Progress(tmp_path, 3, 'signature')
    try:
        state = {'terminal': True, 'image_cleanup': {'ok': True}}
        progress.finish(state, {'status': 'reference_verified'}, historical=True)
        progress.publish(force=True)
        snapshot = json.loads((tmp_path / 'progress.json').read_text())
        assert snapshot['completed'] == 1 and snapshot['session']['resumed_completed'] == 1
        assert snapshot['session']['new_completed'] == 0 and snapshot['session']['eta_seconds'] is None
        progress.finish({}, None)
        progress.finish(state, {'status': 'reference_verified'})
        progress.publish(force=True)
        assert json.loads((tmp_path / 'progress.json').read_text())['session']['new_completed'] == 1
    finally:
        progress.bar.close()

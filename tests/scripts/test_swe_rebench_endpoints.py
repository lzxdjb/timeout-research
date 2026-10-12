"""Exercise isolated endpoint failures with real reference evidence and resumes."""

import json
import threading
from pathlib import Path

import pytest
from test_swe_rebench_reference_filter import PipelineClient, dataset

from scripts import swe_rebench_endpoints as availability
from scripts import swe_rebench_reference_filter as pipeline


@pytest.fixture(autouse=True)
def short_waits(monkeypatch):
    monkeypatch.setattr(availability, "BACKOFF", (.02, .03, .04, .05))
    monkeypatch.setattr(availability, "IDLE_TIMEOUT", 2.0)
    monkeypatch.setattr(availability, "MONITOR_INTERVAL", .01)
    monkeypatch.setattr(availability, "READY_PROBE_INTERVAL", 60)
    monkeypatch.setattr(pipeline, "RECOVERY_BACKOFF", (0, 0, 0))


class FleetClient(PipelineClient):
    def __init__(self, tmp_path):
        super().__init__(tmp_path)
        self.full, self.offline, self.mismatched = set(), set(), set()
        self.claims = []
        self.finishes = []
        self.lock = threading.Lock()

    def get(self, url, path, **kwargs):
        if url in self.offline:
            return {"ok": False, "probe_error": "fixture unreachable"}
        value = super().get(url, path, **kwargs)
        if path == "/status" and url in self.full:
            value["workspace_admission"] = {"ok": False, "error_code": "workspace_storage_pressure"}
        if path == "/tool_contract_info?summary=1" and url in self.mismatched:
            value["source_fingerprint"] = {"sha256": "old-service"}
        return value

    def post(self, url, path, payload, deadline):
        if url in self.offline:
            raise ConnectionError("fixture unreachable")
        if path == "/claim":
            assert url not in self.full and url not in self.mismatched
            self.claims.append((url, payload["task"]["task_id"]))
        if path == "/audit_finish_image":
            # Migration releases an image only after every original request
            # has positive cleanup proof. Terminal cleanup owns its proof too.
            states = list(self.output.glob("cases/*/state.json"))
            own = next(json.loads(p.read_text()) for p in states
                       if json.loads(p.read_text())["lease_id"] == payload["lease_id"])
            self.finishes.append((url, payload["remove"]))
            if not own.get("terminal"):
                directory = next(p.parent for p in states
                                 if json.loads(p.read_text())["lease_id"] == payload["lease_id"])
                for p in directory.glob("attempt_*/*/*.json"):
                    assert json.loads(p.read_text())["cleanup"]["verified"]
            return {"ok": True, "status": "removed" if payload["remove"] else "kept"}
        return super().post(url, path, payload, deadline)


def run(tmp_path, source, client, urls=None, **kwargs):
    return pipeline.run_filter(source, urls or ["http://bad", "http://good"], client.output,
        tmp_path / "filtered.parquet", roots=[f"{client.root}={client.root}"],
        client=client, workers=1, **kwargs)


@pytest.mark.parametrize("fault", ["full", "offline", "mismatched"])
def test_startup_skips_bad_endpoint_and_exports_with_healthy_endpoint(tmp_path, fault):
    source = dataset(tmp_path, 2)
    client = FleetClient(tmp_path)
    getattr(client, fault).add("http://bad")
    report = run(tmp_path, source, client)
    assert report["complete"] and report["reference_verified"] == 2
    assert {url for url, _ in client.claims} == {"http://good"}
    assert not report["endpoints"]["http://bad"]["ready"]


def test_runtime_disk_full_migrates_and_restarts_entire_repeat_proof(tmp_path):
    source = dataset(tmp_path, 1)
    client = FleetClient(tmp_path)
    client.full.add("http://good")
    real = client.post
    def post(url, path, payload, deadline):
        result = real(url, path, payload, deadline)
        if url == "http://bad" and path == "/reward" and payload["request_id"] in client.patched:
            client.full.add(url)
            client.full.discard("http://good")
        return result
    client.post = post
    report = run(tmp_path, source, client)
    assert report["reference_verified"] == 1 and report["migrations"] == 1
    assert sum(client.rewards.values()) == 6  # One old run, two fresh destination runs.
    state_path = next(client.output.glob("cases/*/state.json"))
    state = json.loads(state_path.read_text())
    assert state["endpoint"] == "http://good" and len(state["attempt_sha256"]) == 2
    archived = next(state_path.parent.glob("before_failover_*/state.json"))
    original = json.loads(archived.read_text())
    assert original["image_id"] == state["image_id"] and original["lease_id"] != state["lease_id"]
    assert len(original["attempt_sha256"]) == 1
    record = json.loads((state_path.parent / "qualified_record.json").read_text())
    pipeline.validate_repeat_qualification(record)
    assert {a["endpoint"] for t in record["repeat_qualification"]["attempts"]
            for a in t["arms"].values()} == {"http://good"}


def test_uncertain_cleanup_is_deferred_while_other_tasks_finish(tmp_path, monkeypatch):
    source = dataset(tmp_path, 3)
    client = FleetClient(tmp_path)
    real = pipeline.execute_attempt
    def disconnect(*args, **kwargs):
        if args[1] == "http://bad":
            # Real execution persists a request; transport failure afterwards
            # makes cleanup unprovable and must not allow migration.
            client.offline.add("http://bad")
            arm_dir = args[3] / "baseline"
            arm_dir.mkdir(parents=True)
            pipeline.audit.audit_write_json(arm_dir / "0.json", {
                "request_id": "unknown-request", "endpoint": "http://bad", "phase": "claim",
                "cleanup": {"verified": False}, "mode": "baseline"})
            return {"complete": False}
        return real(*args, **kwargs)
    monkeypatch.setattr(pipeline, "execute_attempt", disconnect)
    with pytest.raises(ValueError, match="pipeline blocked"):
        run(tmp_path, source, client)
    summary = json.loads((client.output / "live_summary.json").read_text())
    report = json.loads((client.output / "report.json").read_text())
    assert summary["reference_verified"] == 2
    assert report["deferred_tasks"] == ["case-0"] and report["migrations"] == 0
    assert report["pending_request_cleanup"]["http://bad"]
    assert ("http://good", "case-0") not in client.claims
    assert not (tmp_path / "filtered.parquet").exists()


def test_endpoint_can_rejoin_after_starting_with_no_eligible_services(tmp_path):
    source = dataset(tmp_path, 1)
    client = FleetClient(tmp_path)
    client.full.add("http://bad")
    real = client.get
    observations = 0
    def get(url, path, **kwargs):
        nonlocal observations
        if path == "/status":
            observations += 1
            if observations >= 3:
                client.full.clear()
        return real(url, path, **kwargs)
    client.get = get
    report = run(tmp_path, source, client, urls=["http://bad"])
    assert report["reference_verified"] == 1 and observations >= 3


def test_terminal_proofs_resume_without_any_responsive_endpoint(tmp_path):
    source = dataset(tmp_path, 1)
    client = FleetClient(tmp_path)
    report = run(tmp_path, source, client, urls=["http://good"])
    proof = next(client.output.glob("cases/*/qualified_record.json")).read_bytes()
    calls = len(client.claims)
    client.offline.add("http://good")
    again = run(tmp_path, source, client, urls=["http://good"], resume=True)
    assert again["complete"] and again["reference_verified"] == report["reference_verified"]
    assert len(client.claims) == calls
    assert next(client.output.glob("cases/*/qualified_record.json")).read_bytes() == proof


def test_terminal_resume_allows_added_removed_and_reordered_endpoints(tmp_path):
    source = dataset(tmp_path, 1)
    client = FleetClient(tmp_path)
    first = run(tmp_path, source, client, urls=["http://good", "http://spare"])
    path = next(client.output.glob("cases/*/qualified_record.json"))
    before, calls = path.read_bytes(), len(client.claims)
    second = run(tmp_path, source, client, urls=["http://new", "http://spare"], resume=True)
    third = run(tmp_path, source, client, urls=["http://spare", "http://new"], resume=True)
    assert first["signature"] == second["signature"] == third["signature"]
    assert path.read_bytes() == before and len(client.claims) == calls
    assert second["dispatch_endpoints"] == ["http://new", "http://spare"]
    assert "http://good" in second["cleanup_only_endpoints"]


def test_removed_owner_is_probed_for_cleanup_before_migration(tmp_path, monkeypatch):
    source = dataset(tmp_path, 1)
    client = FleetClient(tmp_path)
    real_execute = pipeline.execute_attempt
    def interrupt(*args, **kwargs):
        if args[3].name == "attempt_1":
            raise RuntimeError("fixture interrupted after first clean run")
        return real_execute(*args, **kwargs)
    monkeypatch.setattr(pipeline, "execute_attempt", interrupt)
    with pytest.raises(ValueError, match="pipeline blocked"):
        run(tmp_path, source, client, urls=["http://original"])
    assert sum(client.rewards.values()) == 2
    monkeypatch.setattr(pipeline, "execute_attempt", real_execute)
    report = run(tmp_path, source, client, urls=["http://replacement"], resume=True)
    assert report["reference_verified"] == 1 and report["migrations"] == 1
    assert report["cleanup_only_endpoints"] == ["http://original"]
    assert sum(client.rewards.values()) == 6
    assert ("http://original", True) in client.finishes
    record = json.loads(next(client.output.glob("cases/*/qualified_record.json")).read_text())
    assert {arm["endpoint"] for a in record["repeat_qualification"]["attempts"]
            for arm in a["arms"].values()} == {"http://replacement"}


def test_added_endpoint_does_not_reset_healthy_owners_saved_repeat(tmp_path, monkeypatch):
    source = dataset(tmp_path, 1)
    client = FleetClient(tmp_path)
    real_execute = pipeline.execute_attempt
    def interrupt(*args, **kwargs):
        if args[3].name == "attempt_1":
            raise RuntimeError("fixture interrupted repeat")
        return real_execute(*args, **kwargs)
    monkeypatch.setattr(pipeline, "execute_attempt", interrupt)
    with pytest.raises(ValueError, match="pipeline blocked"):
        run(tmp_path, source, client, urls=["http://original"])
    assert sum(client.rewards.values()) == 2
    monkeypatch.setattr(pipeline, "execute_attempt", real_execute)
    report = run(tmp_path, source, client, urls=["http://added", "http://original"], resume=True)
    assert report["reference_verified"] == 1 and report["migrations"] == 0
    assert sum(client.rewards.values()) == 4
    assert {url for url, _ in client.claims} == {"http://original"}


def test_removed_unreachable_owner_does_not_block_new_work(tmp_path, monkeypatch):
    source = dataset(tmp_path, 2)
    client = FleetClient(tmp_path)
    real_execute = pipeline.execute_attempt
    def interrupt(*args, **kwargs):
        client.offline.add("http://original")
        raise RuntimeError("fixture lost original owner after prepare")
    monkeypatch.setattr(pipeline, "execute_attempt", interrupt)
    with pytest.raises(ValueError, match="pipeline blocked"):
        run(tmp_path, source, client, urls=["http://original"])
    monkeypatch.setattr(pipeline, "execute_attempt", real_execute)
    with pytest.raises(ValueError, match="pipeline blocked"):
        run(tmp_path, source, client, urls=["http://replacement"], resume=True)
    report = json.loads((client.output / "report.json").read_text())
    assert report["deferred_tasks"] == ["case-0"]
    assert json.loads((client.output / "live_summary.json").read_text())["reference_verified"] == 1


def test_test_failure_is_never_given_an_endpoint_retry(tmp_path):
    source = dataset(tmp_path, 1)
    client = FleetClient(tmp_path)
    client.fault = "first_reject"
    report = run(tmp_path, source, client)
    assert report["reference_verified"] == 0 and report["migrations"] == 0
    assert sum(client.rewards.values()) == 2


def test_all_unavailable_exits_resumable_without_emitting_failed_data(tmp_path):
    source = dataset(tmp_path, 2)
    client = FleetClient(tmp_path)
    client.offline.update({"http://bad", "http://good"})
    with pytest.raises(ValueError, match="pipeline blocked"):
        run(tmp_path, source, client)
    report = json.loads((client.output / "report.json").read_text())
    assert not report["complete"] and len(report["deferred_tasks"]) == 2
    assert not client.claims and not client.preparations
    assert not list(client.output.glob("cases/*/qualified_record.json"))
    assert not (tmp_path / "filtered.parquet").exists()


def test_endpoint_loss_after_terminal_cleanup_does_not_block_export(tmp_path):
    source = dataset(tmp_path, 1)
    client = FleetClient(tmp_path)
    real = client.post
    def post(url, path, payload, deadline):
        result = real(url, path, payload, deadline)
        if path == "/audit_finish_image":
            client.offline.add(url)
        return result
    client.post = post
    assert run(tmp_path, source, client, urls=["http://good"])["reference_verified"] == 1


def test_migration_cannot_retry_completed_parser_uncertainty(tmp_path):
    directory = tmp_path / "case"
    arm_dir = directory / "attempt_0" / "gold"
    arm_dir.mkdir(parents=True)
    pipeline.audit.audit_write_json(arm_dir / "0.json", {
        "phase": "complete", "mode": "gold", "classification": "official_test_coverage_unknown",
        "cleanup": {"verified": True}, "reference_evidence": {"complete": False}})
    with pytest.raises(ValueError, match="parser uncertainty"):
        pipeline.migrate_task(None, {}, directory, {}, "http://other", "epoch", True)


def test_endpoint_backoff_is_bounded_and_resets_after_recovery(tmp_path, monkeypatch):
    monkeypatch.setattr(availability, "BACKOFF", (5, 15, 30, 60))
    monkeypatch.setattr(availability.time, "monotonic", lambda: 100)
    client = FleetClient(tmp_path)
    client.offline.add("http://bad")
    pool = availability.Endpoints(client, ["http://bad"], pipeline.importlib.import_module(
        "recipe.swe_agent.tool_contract_info").source_fingerprint())
    for delay in (5, 15, 30, 60, 60):
        assert pool.refresh("http://bad")["next_probe"] == 100 + delay
    client.offline.clear()
    recovered = pool.refresh("http://bad")
    assert recovered["ready"] and recovered["failures"] == 0


def test_migration_archive_resumes_after_crash_during_move(tmp_path, monkeypatch):
    source = dataset(tmp_path, 1)
    client = FleetClient(tmp_path)
    client.full.add("http://good")
    real = client.post
    def post(url, path, payload, deadline):
        result = real(url, path, payload, deadline)
        if url == "http://bad" and path == "/reward" and payload["request_id"] in client.patched:
            client.full.add(url)
            client.full.discard("http://good")
        return result
    client.post = post
    rename = Path.rename
    interrupted = False
    def crash(path, destination):
        nonlocal interrupted
        if path.name == "attempt_0" and not interrupted:
            interrupted = True
            raise OSError("fixture interrupted migration")
        return rename(path, destination)
    monkeypatch.setattr(Path, "rename", crash)
    with pytest.raises(ValueError, match="pipeline blocked"):
        run(tmp_path, source, client)
    assert interrupted
    state = json.loads(next(client.output.glob("cases/*/state.json")).read_text())
    assert state["migration_pending"]["from"] == "http://bad"
    # Durable plan already proves cleanup, so an original daemon outage after
    # the archive crash must not make the recovery depend on that daemon again.
    client.offline.add("http://bad")
    monkeypatch.setattr(Path, "rename", rename)
    assert run(tmp_path, source, client, resume=True)["reference_verified"] == 1
    assert sum(client.rewards.values()) == 6


def test_cleanup_is_allowed_when_workspace_admission_is_full(tmp_path):
    client = FleetClient(tmp_path)
    client.full.add("http://bad")
    pool = availability.Endpoints(client, ["http://bad"], pipeline.importlib.import_module(
        "recipe.swe_agent.tool_contract_info").source_fingerprint())
    pool.refresh("http://bad")
    assert not pool.ready("http://bad") and pool.cleanup_ready("http://bad")
    pool.check("http://bad", admission=False)
    with pytest.raises(availability.EndpointUnavailable):
        pool.check("http://bad")


def test_pinned_image_mismatch_blocks_migration_admission(tmp_path):
    source = dataset(tmp_path, 1)
    client = FleetClient(tmp_path)
    client.full.add("http://good")
    real = client.post
    def post(url, path, payload, deadline):
        result = real(url, path, payload, deadline)
        if url == "http://bad" and path == "/reward" and payload["request_id"] in client.patched:
            client.full.add(url)
            client.full.discard("http://good")
        if url == "http://good" and path == "/audit_prepare_image":
            assert payload.get("expected_image_id")
            result["id"] = "sha256:" + "f" * 64
        return result
    client.post = post
    with pytest.raises(ValueError, match="pipeline blocked"):
        run(tmp_path, source, client)
    assert ("http://good", "case-0") not in client.claims
    assert not (tmp_path / "filtered.parquet").exists()


def test_indexed_dispatch_does_not_scan_pending_cases_or_probe_files(monkeypatch):
    from collections import Counter
    urls = ['http://one']
    class Endpoints:
        def ready(self, url): return True
        def cleanup_ready(self, url): return True
    states = {str(i): {} for i in range(30000)}
    holds = {'http://one': {'old-request'}}
    queue = availability.PendingTasks(states, urls, Endpoints(), holds, set(), {}, Counter())
    for key in states:
        queue.put({'inventory': {'id': key}})
    calls = 0
    original = queue.bucket
    def bucket(key):
        nonlocal calls
        calls += 1
        return original(key)
    monkeypatch.setattr(queue, 'bucket', bucket)
    for _ in range(100):
        assert queue.take('http://one') is None
    assert calls == 0
    holds['http://one'].clear()
    assert queue.take('http://one')['inventory']['id'] == '0'
    assert calls == 1


def test_slow_historical_recovery_does_not_occupy_healthy_dispatch_worker(tmp_path, monkeypatch):
    import concurrent.futures
    import time
    source = dataset(tmp_path, 2)
    client = FleetClient(tmp_path)
    original = pipeline.execute_attempt
    def uncertain(*args, **kwargs):
        client.offline.add('http://original')
        arm_dir = args[3] / 'baseline'
        arm_dir.mkdir(parents=True)
        pipeline.audit.audit_write_json(arm_dir / '0.json', {
            'request_id': 'unknown-request', 'endpoint': 'http://original', 'phase': 'claim',
            'cleanup': {'verified': False}, 'mode': 'baseline'})
        return {'complete': False}
    monkeypatch.setattr(pipeline, 'execute_attempt', uncertain)
    client.offline.add('http://good')
    with pytest.raises(ValueError, match='pipeline blocked'):
        run(tmp_path, source, client, urls=['http://original'])
    # Resume with one execution worker. Cleanup is accessible but intentionally
    # slow; the untouched second task must finish before cleanup is released.
    monkeypatch.setattr(pipeline, 'execute_attempt', original)
    client.offline.clear()
    entered, release = threading.Event(), threading.Event()
    def recovery(*args, **kwargs):
        entered.set()
        assert release.wait(10)
        return False
    monkeypatch.setattr(pipeline, 'recover_case_cleanup', recovery)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(run, tmp_path, source, client, urls=['http://good'], resume=True)
        try:
            assert entered.wait(10)
            until = time.monotonic() + 10
            while not (client.output / 'live_reference_verified_task_ids.txt').read_text().strip() and time.monotonic() < until:
                time.sleep(.02)
            assert (client.output / 'live_reference_verified_task_ids.txt').read_text().strip() == 'case-1'
            assert ('http://good', 'case-0') not in client.claims
        finally:
            release.set()
        with pytest.raises(ValueError, match='pipeline blocked'):
            future.result()

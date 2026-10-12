from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import swe_tool_online_benchmark as online
from recipe.swe_agent import remote_execution_service as service
from recipe.swe_agent import tool_contract_info as info
from recipe.swe_agent import tools
from recipe.swe_agent import rg_backend


@pytest.fixture(autouse=True)
def deterministic_audit_concurrency(monkeypatch):
    # Existing quarantine tests intentionally admit only one case at a time.
    # Dedicated scheduler tests below exercise multiple workers per endpoint.
    monkeypatch.setenv("SWE_AGENT_REFERENCE_AUDIT_WORKERS_PER_ENDPOINT", "1")


def audit_dataset(tmp_path, count=2):
    import pyarrow as pa
    import pyarrow.parquet as pq
    gold = tmp_path / "gold.patch"
    gold.write_text("diff --git a/a b/a\n--- a/a\n+++ b/a\n@@ -1 +1 @@\n-bad\n+good\n")
    hidden = tmp_path / "hidden.patch"
    hidden.write_text("hidden test patch")
    rows = []
    for index in range(count):
        task = {"task_id": f"case-{index}", "repo": "fixture/repo", "base_sha": "abc", "docker_image": "cached",
                "test_commands": ["set -eo pipefail\nsource /opt/miniconda3/bin/activate\nconda activate testbed\npytest tests/a.py"],
                "hidden_test_patch_path": str(hidden), "artifact_paths": {"gold_patch": str(gold)}}
        rows.append({"extra_info": {"tools_kwargs": {"__swe_task_json__": json.dumps(task)}}})
    path = tmp_path / "dataset.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


class AuditClient:
    def __init__(self):
        self.calls, self.patched, self.active, self.max_active = [], set(), set(), 0
        self.lock = threading.Lock()

    def get(self, url, path, **kwargs):
        if "?" in path:
            return {"session_state": {"sessions": False}, "cleanup_state": {"complete": True}}
        return {"ok": True, "source_fingerprint": info.source_fingerprint()}

    def post(self, url, path, payload, deadline):
        with self.lock:
            self.calls.append((url, path, payload))
            request = payload["request_id"]
            if path == "/claim":
                assert request not in self.active
                self.active.add(request)
                self.max_active = max(self.max_active, len(self.active))
            if path == "/release":
                self.active.remove(request)
                return {"ok": True, "released": True, "container_removed": True}
            if path == "/execute":
                assert not payload["task"].get("evaluator_test_commands")
                assert not payload["task"].get("artifact_paths")
                if "git apply --check" in payload["parameters"]["command"]:
                    self.patched.add(request)
            if path == "/reward":
                assert payload["task"].get("hidden_test_patch")
                return {"ok": True, "score": float(request in self.patched), "metadata": {
                    "hidden_runner_provenance": "launch_marker_v1", "hidden_tests_started": True,
                    "hidden_test_counts_complete": True, "hidden_tests_total_count": 1, "hidden_failure_phase": "none"}}
        time.sleep(.01)
        return {"ok": True, "text": "exit_code: 0\n", "metadata": {"workspace": "/tmp/a", "container_name": "fixture"}}


def rebench_dataset(tmp_path, count=4):
    import pyarrow as pa
    import pyarrow.parquet as pq
    path = audit_dataset(tmp_path, count)
    rows = pq.read_table(path).to_pylist()
    for i, row in enumerate(rows):
        task = json.loads(row["extra_info"]["tools_kwargs"]["__swe_task_json__"])
        task["repo"] = f"fixture/repo-{i}"
        task["test_commands"] = ["pytest -rA tests/a.py"]
        raw = tmp_path / f"raw-{i}.json"
        raw.write_text(json.dumps({"instance_id": task["task_id"], "language": "python",
                                   "install_config": {"log_parser": "parse_log_pytest"}}))
        task["artifact_paths"]["raw_example"] = str(raw)
        row["extra_info"]["tools_kwargs"]["__swe_task_json__"] = json.dumps(task)
        row["reward_model"] = {"style": "rule", "ground_truth": json.dumps({"benchmark": "swe_rebench_v2",
            "expected": "all_official_tests_pass", "fail_to_pass": ["tests/a.py::test_bug"],
            "pass_to_pass": ["tests/a.py::test_old"]})}
        row["index"] = 100 + i
    table = pa.Table.from_pylist(rows).replace_schema_metadata({b"fixture": b"preserve-me"})
    pq.write_table(table, path)
    return path


class RebenchClient(AuditClient):
    def __init__(self, tmp_path, *, fault=None):
        super().__init__()
        self.root, self.fault = tmp_path / "private", fault
        self.root.mkdir(exist_ok=True)

    def post(self, url, path, payload, deadline):
        if path != "/reward":
            return super().post(url, path, payload, deadline)
        import hashlib
        request = payload["request_id"]
        self.calls.append((url, path, payload))
        gold = request in self.patched
        marker = "SWE_RUNNER_LAUNCHED_" + "a" * 32
        if self.fault == "summary_only":
            records = ""
        elif self.fault == "baseline_setup" and not gold:
            records = "ERROR tests/a.py::test_bug\nPASSED tests/a.py::test_old\n"
        else:
            records = ("PASSED" if gold else "FAILED") + " tests/a.py::test_bug\nPASSED tests/a.py::test_old\n"
        summary = "2 passed" if gold else "1 passed, 1 error" if self.fault == "baseline_setup" else "1 failed, 1 passed"
        text = f"exit_code: {0 if gold else 1}\ncommand: ignored\n{marker}\n{records}================ {summary} in 0.01s ================\n"
        artifact = {"task_id": payload["task"]["task_id"], "phases": {"test_commands": {
            "output": text, "truncated": False, "sha256": hashlib.sha256(text.encode()).hexdigest()}}}
        data = json.dumps(artifact).encode()
        sha = hashlib.sha256(data).hexdigest()
        location = self.root / (sha + ".json")
        location.write_bytes(data)
        return {"ok": True, "score": float(gold), "metadata": {
            "hidden_runner_provenance": "launch_marker_v1", "hidden_tests_started": True,
            "private_evaluator_artifact_path": str(location), "private_evaluator_artifact_sha256": sha,
            "private_evaluator_artifact_status": "available", "hidden_failure_phase": "none"}}


def rebench_options(client):
    return {"profile": "rebench", "evaluator_roots": [str(client.root) + "=" + str(client.root)], "client": client}


class GuardedRebenchClient(RebenchClient):
    """Use the real service runner and private evidence protocol locally."""
    def post(self, url, path, payload, deadline):
        if path != "/reward":
            return super().post(url, path, payload, deadline)
        import hashlib
        import re
        from recipe.swe_agent.test_results import score_official_tests
        task, request = payload["task"], payload["request_id"]
        gold = request in self.patched
        session = service.ExecutionSession("key", task, request, self.root, False)
        phases = session._active_evaluator_phases = {}
        index = 0
        def run(command, **kwargs):
            nonlocal index
            marker = re.findall(r"SWE_RUNNER_LAUNCHED_[a-f0-9]+", command)[0]
            failed = not gold and index == 0
            if task["official_test_contract"]["log_parser"] == "parse_java_mvn":
                text = "[INFO] Running tests.model.PagesTest\nTests run: 1, Failures: 0, Errors: 0, Skipped: 0\nBUILD SUCCESS\n"
                failed = False
            elif len(task["test_commands"]) > 1:
                node = "test_bug" if index == 0 else "test_old"
                text = f"{'FAILED' if failed else 'PASSED'} tests/a.py::{node}\n1 {'failed' if failed else 'passed'} in 0.1s\n"
            else:
                text = "FAILED tests/a.py::test_bug\nPASSED tests/a.py::test_old\n1 failed, 1 passed in 0.1s\n"
                if not failed:
                    text = "PASSED tests/a.py::test_bug\nPASSED tests/a.py::test_old\n2 passed in 0.1s\n"
            index += 1
            return int(failed), marker + "\n" + text
        session.run = run
        text, _, results = session._run_selected_commands({}, key="test_commands", return_results=True)
        score = score_official_tests(task["official_test_contract"],
                                     [r["_private_test_evidence"] for r in results], len(task["test_commands"]),
                                     execution_exit_codes=[r["exit_code"] for r in results])
        private = score.pop("_private_official_results")
        private["score"] = dict(score)
        phases["official_tests"] = json.dumps(private)
        if self.fault == "input":
            retained = json.loads(phases["runner_evidence_0"])
            retained["provenance"]["parser_input_sha256"] = "wrong"
            phases["runner_evidence_0"] = json.dumps(retained)
        elif self.fault == "contract":
            retained = json.loads(phases["runner_evidence_0"])
            retained["provenance"]["contract_sha256"] = "wrong"
            phases["runner_evidence_0"] = json.dumps(retained)
        elif self.fault == "missing_proof":
            phases.pop("runner_evidence_0")
        elif self.fault == "score":
            private["score"]["official_tests_passed"] = -1
            phases["official_tests"] = json.dumps(private)
        elif self.fault == "metadata":
            score["official_tests_passed"] = -1
        artifact = {"task_id": task["task_id"], "phases": {
            k: {"output": v, "truncated": False, "sha256": hashlib.sha256(v.encode()).hexdigest()}
            for k, v in phases.items()}}
        data = json.dumps(artifact).encode()
        sha = hashlib.sha256(data).hexdigest()
        location = self.root / (sha + ".json")
        location.write_bytes(data)
        return {"ok": True, "score": float(score["official_tests_success"]), "metadata": {
            **score, "hidden_runner_provenance": "launch_marker_v1", "hidden_tests_started": True,
            "private_evaluator_artifact_path": str(location), "private_evaluator_artifact_sha256": sha,
            "private_evaluator_artifact_status": "available", "hidden_failure_phase": "none"}}


@pytest.mark.parametrize("fault", [None, "input", "contract", "missing_proof", "score", "metadata"])
@pytest.mark.parametrize("multi_command", [False, True])
def test_native_guarded_requires_identical_service_and_audit_scoring(tmp_path, fault, multi_command):
    import pyarrow as pa
    import pyarrow.parquet as pq
    path = rebench_dataset(tmp_path, 1)
    if multi_command:
        rows = pq.read_table(path).to_pylist()
        task = json.loads(rows[0]["extra_info"]["tools_kwargs"]["__swe_task_json__"])
        task["test_commands"] = ["pytest -rA tests/a.py -k test_bug", "pytest -rA tests/a.py -k test_old"]
        rows[0]["extra_info"]["tools_kwargs"]["__swe_task_json__"] = json.dumps(task)
        pq.write_table(pa.Table.from_pylist(rows), path)
    client = GuardedRebenchClient(tmp_path, fault=fault)
    report = online.reference_audit([path], ["http://one"], tmp_path / "audit", parser_policy="native-guarded",
                                    **rebench_options(client))
    result = report["results"][0]
    assert result["status"] == ("review:audit_service_scoring_mismatch" if fault else "reference_verified")
    for arm in result["arms"].values():
        assert arm["reference_evidence"]["service_consistency"]["matched"] is (fault is None)
    if fault is None:
        receipt = online.audit_export_results(path, tmp_path / "audit/results.jsonl", tmp_path / "filtered.parquet")
        assert receipt["rows"] == 1


def test_maven_selector_in_command_header_cannot_be_certified_as_test(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    path = rebench_dataset(tmp_path, 1)
    rows = pq.read_table(path).to_pylist()
    task = json.loads(rows[0]["extra_info"]["tools_kwargs"]["__swe_task_json__"])
    task["test_commands"] = ['mvn test -B -Dtest="tests.**.*Test" -DfailIfNoTests=false']
    raw_path = Path(task["artifact_paths"]["raw_example"])
    raw = json.loads(raw_path.read_text())
    raw["install_config"]["log_parser"] = "parse_java_mvn"
    raw_path.write_text(json.dumps(raw))
    truth = json.loads(rows[0]["reward_model"]["ground_truth"])
    truth.update(fail_to_pass=['"tests.**.*Test"'], pass_to_pass=["tests.model.PagesTest"])
    rows[0]["reward_model"]["ground_truth"] = json.dumps(truth)
    rows[0]["extra_info"]["tools_kwargs"]["__swe_task_json__"] = json.dumps(task)
    pq.write_table(pa.Table.from_pylist(rows), path)
    report = online.reference_audit([path], ["http://one"], tmp_path / "audit", parser_policy="native-guarded",
                                    **rebench_options(GuardedRebenchClient(tmp_path)))
    arm = report["results"][0]["arms"]["gold"]
    assert arm["reference_evidence"]["service_consistency"]["matched"]
    assert not arm["reference_evidence"]["all_required_passed"]
    assert arm["reference_evidence"]["missing"] == 1


@pytest.mark.parametrize("failure", [None, "registry_authentication", "registry_not_found", "registry_network", "preparation_timeout", "storage_failure"])
def test_rebench_explicit_provisioning_cache_miss_and_resume(tmp_path, failure):
    class ProvisionClient(RebenchClient):
        cache_present = False
        def get(self, url, path, **kwargs):
            result = super().get(url, path, **kwargs)
            if "?" not in path:
                result.update(image_inventory_complete=True, cached_images=[{"tag": "docker.io/library/cached:latest", "id": "sha256:image"}] if self.cache_present else [])
            return result
        def post(self, url, path, payload, deadline):
            if path == "/audit_prepare_image":
                self.calls.append((url, path, payload))
                assert payload == {"image": "cached", "timeout_seconds": 3}
                self.cache_present = failure is None
                return {"ok": failure is None, "id": "sha256:image" if failure is None else None, "failure_kind": failure}
            return super().post(url, path, payload, deadline)
    path = rebench_dataset(tmp_path, 1)
    client = ProvisionClient(tmp_path)
    output = tmp_path / "audit"
    options = {**rebench_options(client), "image_policy": "archive-or-pull", "image_prepare_seconds": 3}
    report = online.reference_audit([path], ["http://one"], output, **options)
    assert report["results"][0]["status"] == ("reference_verified" if failure is None else "review:image:" + failure)
    assert bool([c for c in client.calls if c[1] == "/claim"]) is (failure is None)
    count = len(client.calls)
    resumed = online.reference_audit([path], ["http://one"], output, resume=True, **options)
    assert resumed["signature"] == report["signature"] and len(client.calls) == count
    ledger = json.loads((output / "results.jsonl").read_text())
    assert ledger["image_preparation"]["ok"] is (failure is None)


def test_rebench_incomplete_image_inventory_dispatches_without_full_probe_sweep(tmp_path):
    class IncompleteClient(RebenchClient):
        def get(self, url, path, **kwargs):
            assert not path.startswith("/tool_contract_info?image="), "unexpected startup image sweep"
            result = super().get(url, path, **kwargs)
            if "?" not in path:
                result.update(image_inventory_complete=False, cached_images=[])
            return result
    source = rebench_dataset(tmp_path, 4)
    client = IncompleteClient(tmp_path)
    report = online.reference_audit([source], ["http://one", "http://two"], tmp_path / "audit",
                                    **rebench_options(client))
    assert report["counts"] == {"reference_verified": 4}
    assert len([call for call in client.calls if call[1] == "/claim"]) == 8


def test_rebench_versioned_dataset_opt_in_preserves_source_and_validation(tmp_path):
    import pyarrow.parquet as pq
    from recipe.swe_agent.task_payload import with_official_test_contract, reward_task_payload, rollout_task_payload
    path = rebench_dataset(tmp_path, 1)
    original = path.read_bytes()
    row = pq.read_table(path).to_pylist()[0]
    task = json.loads(row["extra_info"]["tools_kwargs"]["__swe_task_json__"])
    assert "official_test_contract" not in with_official_test_contract(task, row["reward_model"])
    output = tmp_path / "versioned.parquet"
    receipt = online.dataset_version(path, output, [], rebench_contract=True)
    assert receipt["rows"] == 1 and path.read_bytes() == original
    upgraded = pq.read_table(output).to_pylist()[0]
    new_task = with_official_test_contract(json.loads(upgraded["extra_info"]["tools_kwargs"]["__swe_task_json__"]), upgraded["reward_model"])
    payload = reward_task_payload(new_task)
    assert payload["official_test_contract"]["rebench_scoring_version"] == 1
    assert payload["evaluator_test_commands"][0]["source_command"] == task["test_commands"][0]
    assert "official_test_contract" not in rollout_task_payload(payload)
    assert pq.read_table(output).schema.metadata == pq.read_table(path).schema.metadata


def test_rebench_replay_reuses_retained_pilot_evidence(tmp_path):
    path = rebench_dataset(tmp_path, 1)
    client = RebenchClient(tmp_path)
    output = tmp_path / "audit"
    online.reference_audit([path], ["http://one"], output, **rebench_options(client))
    report = online.reference_replay([path], output / "report.json", tmp_path / "replay.json")
    assert len(report["cases"]) == 2
    assert report["reasons"] == {"complete": 2}
    assert all(c["replayed_status"] == "reference_verified" for c in report["cases"])
    fixtures = online.reference_replay_scenarios([path], output / "report.json", tmp_path / "replay.json")
    assert len(fixtures) == 2
    assert all(not case["steps"] for case in fixtures)
    assert all(not online.check_parser_replay(case)["failures"] for case in fixtures)
    frozen = tmp_path / "benchmark"
    manifest = online.build(frozen, reference_datasets=[path], reference_audit_report=output / "report.json",
                            reference_replay_report=tmp_path / "replay.json")
    _, cases = online.load(frozen)
    assert manifest["scenario_count"] == len(cases)
    assert sum(case["id"].startswith("reference_") for case in cases) == 2
    reviewed = json.loads((tmp_path / "replay.json").read_text())
    reviewed["audit_sha256"] = "changed"
    (tmp_path / "replay.json").write_text(json.dumps(reviewed))
    with pytest.raises(ValueError, match="source audit"):
        online.reference_replay_scenarios([path], output / "report.json", tmp_path / "replay.json")


def test_reference_benchmark_requires_all_source_inputs(tmp_path):
    with pytest.raises(ValueError, match="require datasets"):
        online.build(tmp_path / "benchmark", reference_audit_report=tmp_path / "audit.json")


def test_image_reference_normalization_and_failure_categories():
    from recipe.swe_agent.image_archives import canonical_image_reference, image_failure_kind
    assert canonical_image_reference("org/image:v2") == canonical_image_reference("docker.io/org/image:v2")
    assert canonical_image_reference("ubuntu") == "docker.io/library/ubuntu:latest"
    assert canonical_image_reference("ubuntu:v2") == "docker.io/library/ubuntu:v2"
    assert canonical_image_reference("ubuntu@sha256:abc") == "docker.io/library/ubuntu@sha256:abc"
    assert canonical_image_reference("ghcr.io/org/image:v2") == "ghcr.io/org/image:v2"
    assert canonical_image_reference("localhost:5000/x@sha256:abc") == "localhost:5000/x@sha256:abc"
    for text, expected in [("manifest unknown", "registry_not_found"), ("pull access denied", "registry_authentication"),
                           ("no space left", "storage_failure"), ("connection reset", "registry_network"), ("deadline exceeded", "preparation_timeout")]:
        assert image_failure_kind(text) == expected


def test_rebench_audit_portable_filter_and_changed_concurrency_resume(tmp_path):
    import pyarrow.parquet as pq
    path = rebench_dataset(tmp_path)
    original = path.read_bytes()
    client = RebenchClient(tmp_path)
    output = tmp_path / "audit"
    report = online.reference_audit([path], ["http://one"], output, workers_per_endpoint=2, **rebench_options(client))
    assert report["counts"] == {"reference_verified": 4} and report["complete"]
    previous = len(client.calls)
    resumed = online.reference_audit([path], ["http://one"], output, workers_per_endpoint=3, resume=True, **rebench_options(client))
    assert len(client.calls) == previous and resumed["signature"] == report["signature"]
    progress = json.loads((output / "progress.json").read_text())
    assert progress["completed"] == 4 and progress["remaining"] == progress["running"] == progress["failed"] == 0
    receipt = online.audit_export_results(path, output / "results.jsonl", tmp_path / "filtered.parquet")
    assert receipt["rows"] == 4 and receipt["not_audited_rows"] == 0
    assert pq.read_table(path).equals(pq.read_table(tmp_path / "filtered.parquet"), check_metadata=True)
    assert path.read_bytes() == original
    with pytest.raises(ValueError, match="overwrite"):
        online.audit_export_results(path, output / "results.jsonl", path)
    with pytest.raises(ValueError, match="signature"):
        online.reference_audit([path], ["http://one"], output, seconds=200, resume=True, **rebench_options(client))


@pytest.mark.parametrize("fault", ["summary_only", "baseline_setup"])
def test_rebench_never_certifies_summary_only_or_baseline_setup_error(tmp_path, fault):
    client = RebenchClient(tmp_path, fault=fault)
    report = online.reference_audit([rebench_dataset(tmp_path, 1)], ["http://one"], tmp_path / "audit", **rebench_options(client))
    assert not report["regression_pass"]
    assert report["results"][0]["status"].startswith("review:")
    rejected = report["results"][0]["arms"]["baseline"]["reference_evidence"]
    assert rejected["parser_policy"] == "strict"
    assert rejected["native_parser"]["candidate"]["authority"] == "native_observed_only"


def test_rebench_native_shadow_preserves_strict_authority_and_signature(tmp_path):
    path = rebench_dataset(tmp_path, 1)
    client = RebenchClient(tmp_path)
    output = tmp_path / "audit"
    report = online.reference_audit(
        [path], ["http://one"], output, parser_policy="native-first-shadow", **rebench_options(client)
    )
    result = report["results"][0]
    assert result["status"] == "reference_verified"
    assert result["native_reference_candidate"] == {
        "authority": "audit_only_native_observed",
        "eligible": True,
        "reason": "native_reference_transition_observed",
        "parser": "parse_log_pytest",
    }
    assert report["native_parser_summary"]["authority"] == "audit_only_native_observed"
    assert report["native_parser_summary"]["candidate_count"] == 1
    for arm in result["arms"].values():
        evidence = arm["reference_evidence"]
        assert evidence["parser_policy"] == "native-first-shadow"
        assert evidence["native_parser"]["candidate"]["authority"] == "native_observed_only"
        assert evidence["native_parser"]["all_command_observations_available"] is True
    with pytest.raises(ValueError, match="signature"):
        online.reference_audit([path], ["http://one"], output, resume=True, **rebench_options(client))
    with pytest.raises(ValueError, match="unsupported ReBench parser policy"):
        online.reference_audit([path], ["http://one"], tmp_path / "invalid", parser_policy="native", **rebench_options(client))


@pytest.mark.parametrize("fault", ["summary_only", "baseline_setup"])
def test_rebench_native_candidate_requires_complete_baseline_observations(tmp_path, fault):
    client = RebenchClient(tmp_path, fault=fault)
    report = online.reference_audit(
        [rebench_dataset(tmp_path, 1)], ["http://one"], tmp_path / "audit", parser_policy="native-first-shadow",
        **rebench_options(client),
    )
    candidate = report["results"][0]["native_reference_candidate"]
    assert candidate["eligible"] is False
    assert candidate["authority"] == "audit_only_native_observed"
    assert report["native_parser_summary"]["candidate_count"] == 0


def test_rebench_workspace_identity_accepts_untracked_but_rejects_tracked_changes(tmp_path):
    import subprocess
    client = RebenchClient(tmp_path)
    online.reference_audit([rebench_dataset(tmp_path, 1)], ["http://one"], tmp_path / "audit", **rebench_options(client))
    base_commands = [payload["parameters"]["command"] for _, path, payload in client.calls
                     if path == "/execute" and "SWE_BASE_UNTRACKED" in payload["parameters"]["command"]]
    assert len(base_commands) == 2
    command = base_commands[0]
    assert 'test -z "$tracked" || exit 1' in command
    assert 'test -z "$tracked$untracked"' not in command
    assert "untracked=$(git ls-files --others --exclude-standard" in command
    repo = tmp_path / "workspace"
    repo.mkdir()
    def git(*args):
        return subprocess.run(["git", *args], cwd=repo, check=True, text=True, capture_output=True).stdout.strip()
    git("init", "-q")
    git("config", "user.name", "Fixture")
    git("config", "user.email", "fixture@example.com")
    (repo / "tracked.txt").write_text("base")
    git("add", "tracked.txt")
    git("commit", "-qm", "base")
    head = git("rev-parse", "HEAD")
    probe = command.replace("expected=abc", "expected=" + head)
    (repo / "generated.txt").write_text("untracked build artifact")
    assert subprocess.run(["bash", "-c", probe], cwd=repo, capture_output=True).returncode == 0
    (repo / "tracked.txt").write_text("changed")
    assert subprocess.run(["bash", "-c", probe], cwd=repo, capture_output=True).returncode != 0
    assert subprocess.run(["bash", "-c", probe.replace("expected=" + head, "expected=deadbeef")],
                          cwd=repo, capture_output=True).returncode != 0


def test_legacy_evaluator_fixture_validity_is_unknown_not_invalid():
    response = {"ok": True, "score": 1.0, "metadata": {
        "hidden_tests_started": True, "evaluation_valid": None, "hidden_failure_phase": "none",
        "private_evaluator_artifact_status": "available", "private_evaluator_logs_complete": True,
    }}
    assert not online.check_evaluator_fixture(response)
    response["metadata"]["evaluation_valid"] = False
    assert online.check_evaluator_fixture(response) == ["evaluator fixture: valid"]


def test_rebench_deterministic_diverse_pilot_and_explicit_partial_export(tmp_path):
    path = rebench_dataset(tmp_path, 8)
    manifest, cases = online.audit_inventory([path], profile="rebench")
    assert manifest["counts"]["eligible"] == 8
    for i, case in enumerate(cases):
        case["inventory"].update(language="go" if i % 2 else "python", log_parser="go" if i % 2 else "pytest")
    selected, selection = online.audit_pilot_selection(cases, 4, 42)
    again, other = online.audit_pilot_selection(list(reversed(cases)), 4, 42)
    assert selection == other and [c["inventory"]["id"] for c in selected] == [c["inventory"]["id"] for c in again]
    assert len(selection["pilot"]) == 2 and sum(selection["pilot"].values()) == 4
    with pytest.raises(ValueError, match="stratum"):
        online.audit_pilot_selection(cases, 1, 42)
    client = RebenchClient(tmp_path)
    output = tmp_path / "pilot"
    online.reference_audit([path], ["http://one"], output, pilot_size=3, seed=42, **rebench_options(client))
    with pytest.raises(ValueError, match="every dataset row"):
        online.audit_export_results(path, output / "results.jsonl", tmp_path / "full.parquet")
    receipt = online.audit_export_results(path, output / "results.jsonl", tmp_path / "pilot.parquet", pilot_only=True)
    assert receipt["rows"] == 3 and receipt["not_audited_rows"] == 5


def test_rebench_recovers_uncertain_session_before_new_claim_and_repairs_ledger(tmp_path):
    path = rebench_dataset(tmp_path, 1)
    client = RebenchClient(tmp_path)
    output = tmp_path / "audit"
    report = online.reference_audit([path], ["http://one"], output, **rebench_options(client))
    directory = output / "cases" / report["results"][0]["id"]
    checkpoint = directory / "gold" / "0.json"
    arm = json.loads(checkpoint.read_text())
    arm.update(phase="reward", cleanup={"verified": False})
    online.audit_write_json(checkpoint, arm)
    (directory / "result.json").unlink()
    (output / "results.jsonl").write_text('{"truncated":')
    client.active.add(arm["request_id"])
    previous = len(client.calls)
    report = online.reference_audit([path], ["http://one"], output, resume=True, **rebench_options(client))
    assert report["counts"] == {"reference_verified": 1}
    calls = client.calls[previous:]
    assert calls[0][1] == "/release" and calls[0][2]["request_id"] == arm["request_id"]
    assert sum(path == "/claim" for _, path, _ in calls) == 1
    assert (directory / "gold" / "1.json").exists() and not client.active
    assert len((output / "results.jsonl").read_text().splitlines()) == 1


@pytest.mark.parametrize("fault", ["duplicate", "wrong_row", "wrong_source", "incomplete", "fake_verified"])
def test_rebench_filter_rejects_invalid_portable_results(tmp_path, fault):
    path = rebench_dataset(tmp_path, 1)
    client = RebenchClient(tmp_path)
    output = tmp_path / "audit"
    online.reference_audit([path], ["http://one"], output, **rebench_options(client))
    ledger = output / "results.jsonl"
    record = json.loads(ledger.read_text())
    if fault == "wrong_row":
        record["row"] = 2
    elif fault == "wrong_source":
        record["dataset_sha256"] = "bad"
    elif fault == "incomplete":
        record["complete"] = False
    elif fault == "fake_verified":
        record["arms"]["baseline"]["reference_evidence"]["fail_to_pass_failed"] = 0
    ledger.write_text(json.dumps(record) + "\n" + (json.dumps(record) + "\n" if fault == "duplicate" else ""))
    with pytest.raises(ValueError):
        online.audit_export_results(path, ledger, tmp_path / "filtered.parquet")
    assert not (tmp_path / "filtered.parquet").exists()


@pytest.mark.parametrize("command,text,code,complete", [
    ("pytest -rA", "PASSED a.py::test_a\nFAILED a.py::test_b - assert False\n1 failed, 1 passed in 0.01s\n", 1, True),
    ("pytest -rA", "1 failed, 1 passed in 0.01s\n", 1, False),
    ("pytest -rA", "ERROR a.py::test_a\n1 error in 0.01s\n", 1, False),
    ("pytest -rA", "PASSED a.py::test_a\nPASSED a.py::test_a\n2 passed in 0.01s\n", 0, False),
    ("pytest -rA", "PASSED a.py::test_a\n1 passed in 0.01s\n", 1, False),
    ("go test -v .", "=== RUN   TestA\n--- PASS: TestA (0.00s)\nPASS\nok\tmodule/pkg\t0.001s\n", 0, True),
    ("go test -v .", "=== RUN   TestA\n--- FAIL: TestA (0.00s)\nFAIL\nFAIL\tmodule/pkg\t0.001s\n", 1, True),
    ("cargo test", "running 1 test\ntest test_a ... ok\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s\n", 0, True),
    ("cargo test", "error: compilation failed\n", 101, False),
    ("npm test", "Tests: 1 passed, 1 total\nTest Suites: 1 passed, 1 total\n", 0, False),
])
def test_rebench_exact_identity_evidence(command, text, code, complete):
    assert online.audit_rebench_outcomes(command, text, code)["complete"] is complete


def test_rebench_wrapper_preserves_original_shell_and_private_contract(tmp_path):
    import subprocess
    _, cases = online.audit_inventory([rebench_dataset(tmp_path, 1)], profile="rebench")
    original = cases[0]["task"]
    original["test_commands"] = ["false; printf 'reached\\n'"]
    payload = online.audit_rebench_task(original, 300)
    assert subprocess.run(["bash", "-c", payload["test_commands"][0]], capture_output=True, text=True).stdout == "reached\n"
    assert payload["official_test_contract"]["rebench_scoring_version"] == 1
    from recipe.swe_agent.task_payload import rollout_task_payload
    assert "official_test_contract" not in rollout_task_payload(payload)
    assert original["test_commands"] == ["false; printf 'reached\\n'"]


def test_rebench_completed_failures_are_not_retried_on_resume(tmp_path):
    path = rebench_dataset(tmp_path, 2)
    client = RebenchClient(tmp_path, fault="summary_only")
    output = tmp_path / "audit"
    first = online.reference_audit([path], ["http://one"], output, **rebench_options(client))
    previous = len(client.calls)
    second = online.reference_audit([path], ["http://one"], output, resume=True, **rebench_options(client))
    assert second["counts"] == first["counts"] and second["complete"]
    assert len(client.calls) == previous
    progress = json.loads((output / "progress.json").read_text())
    assert progress["failed"] == progress["review"] == 2


def test_rebench_unrecoverable_cleanup_blocks_work_without_marking_it_completed(tmp_path, monkeypatch):
    path = rebench_dataset(tmp_path, 2)
    client = RebenchClient(tmp_path)
    output = tmp_path / "audit"
    report = online.reference_audit([path], ["http://one"], output, **rebench_options(client))
    for result in report["results"]:
        directory = output / "cases" / result["id"]
        (directory / "result.json").unlink()
    checkpoint = output / "cases" / report["results"][0]["id"] / "baseline" / "0.json"
    arm = json.loads(checkpoint.read_text())
    arm.update(phase="claim", cleanup={"verified": False})
    online.audit_write_json(checkpoint, arm)
    monkeypatch.setattr(online, "audit_recover_arm", lambda *args: args[-1])
    previous = len(client.calls)
    resumed = online.reference_audit([path], ["http://one"], output, resume=True, **rebench_options(client))
    assert not resumed["complete"] and resumed["quarantined_endpoints"] == ["http://one"]
    assert len(client.calls) == previous
    assert all(result["complete"] is False for result in resumed["results"])
    with pytest.raises(ValueError, match="incomplete"):
        online.audit_export_results(path, output / "results.jsonl", tmp_path / "filtered.parquet")


def test_rebench_filter_retains_only_verified_rows_in_original_order(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    path = rebench_dataset(tmp_path, 4)
    client = RebenchClient(tmp_path)
    output = tmp_path / "audit"
    online.reference_audit([path], ["http://one"], output, workers_per_endpoint=2, **rebench_options(client))
    ledger = output / "results.jsonl"
    records = [json.loads(line) for line in ledger.read_text().splitlines()]
    for record in records:
        if record["row"] % 2:
            record["status"] = "review:unsupported_identity_parser"
    ledger.write_text("".join(json.dumps(record) + "\n" for record in reversed(records)))
    receipt = online.audit_export_results(path, ledger, tmp_path / "filtered.parquet")
    assert receipt["retained_row_indices"] == [0, 2]
    assert pq.read_table(tmp_path / "filtered.parquet").equals(pq.read_table(path).take(pa.array([0, 2])), check_metadata=True)


def repeat_audits(tmp_path, directories=("first", "repeat")):
    class CachedGuardedClient(GuardedRebenchClient):
        def get(self, url, route, **kwargs):
            result = super().get(url, route, **kwargs)
            if route == "/tool_contract_info":
                result["cached_images"] = [{"tag": "cached", "id": "sha256:fixture-image"}]
            return result
    path = rebench_dataset(tmp_path, 2)
    client = CachedGuardedClient(tmp_path)
    for directory in directories:
        online.reference_audit([path], ["http://one"], tmp_path / directory,
                               parser_policy="native-guarded", **rebench_options(client))
    return path, tmp_path / "first/results.jsonl", tmp_path / "repeat/results.jsonl"


def test_repeat_qualified_export_preserves_rows_and_metadata(tmp_path):
    from scripts.swe_reference_reliability import qualify_repeats
    import pyarrow.parquet as pq
    path, first, repeat = repeat_audits(tmp_path)
    result = qualify_repeats(first, [repeat], tmp_path / "qualified")
    assert result["repeat_qualified"] == 2
    receipt = online.audit_export_results(path, tmp_path / "qualified/qualified_results.jsonl",
                                          tmp_path / "filtered.parquet", require_repeat=True)
    assert receipt["rows"] == 2
    assert receipt["repeat_required"]
    assert pq.read_table(path).equals(pq.read_table(tmp_path / "filtered.parquet"), check_metadata=True)
    with pytest.raises(ValueError, match="two independent"):
        online.audit_export_results(path, first, tmp_path / "unqualified.parquet", require_repeat=True)


@pytest.mark.parametrize("fault", ["transport", "transport_with_test_failure", "uncertainty"])
def test_repeat_retry_recovers_only_transient_failures(tmp_path, fault):
    from scripts.swe_reference_reliability import qualify_repeats, validate_repeat_qualification
    _, first, repeat = repeat_audits(tmp_path, ("first", "repeat", "retry"))
    rows = [json.loads(line) for line in repeat.read_text().splitlines()]
    row = rows[0]
    row["status"] = "review:baseline:transport_failure"
    row["arms"]["baseline"]["classification"] = "transport_failure"
    row["arms"]["baseline"]["reference_evidence"] = {}
    if fault == "transport_with_test_failure":
        row["arms"]["gold"]["score"] = 0
        row["arms"]["gold"]["reference_evidence"]["all_required_passed"] = False
    elif fault == "uncertainty":
        row["status"] = "review:gold:parser_ambiguity"
        row["arms"]["baseline"]["classification"] = "parser_ambiguity"
    repeat.write_text("".join(json.dumps(r) + "\n" for r in rows))
    initial = qualify_repeats(first, [repeat], tmp_path / "initial")
    assert initial["repeat_qualified"] == 1
    assert initial["retry_candidates"] == int(fault == "transport")
    result = qualify_repeats(first, [repeat, tmp_path / "retry/results.jsonl"], tmp_path / "qualified")
    assert result["repeat_qualified"] == (2 if fault == "transport" else 1)
    if fault == "transport":
        record = next(r for r in map(json.loads, (tmp_path / "qualified/qualified_results.jsonl").read_text().splitlines())
                      if r["row"] == row["row"])
        validate_repeat_qualification(record)
        record["repeat_qualification"]["attempts"][1]["verification_context"]["evaluator_sha256"] = "changed"
        with pytest.raises(ValueError, match="identity changed"):
            validate_repeat_qualification(record)


@pytest.mark.parametrize("fault", ["uncertain", "contract", "request", "copied", "image"])
def test_repeat_rejects_uncertainty_and_changed_or_reused_evidence(tmp_path, fault):
    from scripts.swe_reference_reliability import qualify_repeats
    _, first, repeat = repeat_audits(tmp_path)
    rows = [json.loads(line) for line in repeat.read_text().splitlines()]
    if fault == "copied":
        repeat.write_bytes(first.read_bytes())
    else:
        if fault == "uncertain":
            rows[0]["status"] = "review:gold:required_tests_missing_skipped_or_ambiguous"
        elif fault == "contract":
            rows[0]["verification_context"]["reference_contract_sha256"] = "changed"
        elif fault == "image":
            rows[0]["verification_context"]["image_id"] = "changed"
        else:
            original = {r["row"]: r for r in map(json.loads, first.read_text().splitlines())}
            rows[0]["arms"]["gold"]["request_id"] = original[rows[0]["row"]]["arms"]["gold"]["request_id"]
        repeat.write_text("".join(json.dumps(row) + "\n" for row in rows))
    if fault == "uncertain":
        report = qualify_repeats(first, [repeat], tmp_path / "qualified")
        assert report["repeat_qualified"] == 1
        assert report["retry_candidates"] == 0
    else:
        with pytest.raises(ValueError):
            qualify_repeats(first, [repeat], tmp_path / "qualified")


def test_rebench_source_mismatch_and_invalid_roots_fail_before_claims(tmp_path):
    path = rebench_dataset(tmp_path, 1)
    client = RebenchClient(tmp_path)
    original_get = client.get
    def get(url, route):
        value = original_get(url, route)
        if route == "/tool_contract_info":
            value["source_fingerprint"] = {"sha256": "mismatch"}
        return value
    client.get = get
    with pytest.raises(ValueError, match="matching deployed sources"):
        online.reference_audit([path], ["http://one"], tmp_path / "audit", **rebench_options(client))
    assert not client.calls
    with pytest.raises(ValueError, match="ABSOLUTE"):
        online.reference_audit([path], ["http://one"], tmp_path / "invalid", profile="rebench", evaluator_roots=["relative=/tmp"], client=client)


def test_reference_audit_parallel_isolation_resume_and_explicit_export(tmp_path):
    import pyarrow.parquet as pq
    path = audit_dataset(tmp_path, 4)
    before = path.read_bytes()
    client = AuditClient()
    directory = tmp_path / "audit"
    report = online.reference_audit([path], ["http://one", "http://two"], directory, client=client)
    assert report["counts"] == {"reference_verified": 4} and report["regression_pass"]
    assert client.max_active == 2 and not client.active
    claims = [call[2]["request_id"] for call in client.calls if call[1] == "/claim"]
    assert len(set(claims)) == 8
    previous = len(client.calls)
    assert online.reference_audit([path], ["http://one", "http://two"], directory, resume=True, client=client) == report
    assert len(client.calls) == previous
    with pytest.raises(ValueError, match="signature"):
        online.reference_audit([path], ["http://one", "http://two"], directory, seconds=100, resume=True, client=client)
    receipt = online.audit_export(directory / "report.json", tmp_path / "filtered.parquet")
    assert receipt["rows"] == 4 and pq.read_table(tmp_path / "filtered.parquet").num_rows == 4
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match="overwrite"):
        online.audit_export(directory / "report.json", path)


@pytest.mark.parametrize("workers", [2, 3])
def test_reference_audit_configurable_same_endpoint_parallelism(tmp_path, workers, monkeypatch):
    monkeypatch.setenv("SWE_AGENT_REFERENCE_AUDIT_WORKERS_PER_ENDPOINT", str(workers))
    barrier = threading.Barrier(workers)
    class ParallelClient(AuditClient):
        def post(self, url, path, payload, deadline):
            result = super().post(url, path, payload, deadline)
            if path == "/claim":
                barrier.wait(timeout=10)
            return result
    client = ParallelClient()
    report = online.reference_audit([audit_dataset(tmp_path, workers)], ["http://one"], tmp_path / "audit", client=client)
    assert report["workers_per_endpoint"] == workers
    assert client.max_active == workers and not client.active
    assert report["counts"] == {"reference_verified": workers}
    requests = [p["request_id"] for _, route, p in client.calls if route == "/claim"]
    assert len(requests) == len(set(requests)) == workers * 2


def test_reference_audit_default_parallelism_and_cli_override(tmp_path, monkeypatch):
    monkeypatch.delenv("SWE_AGENT_REFERENCE_AUDIT_WORKERS_PER_ENDPOINT")
    report = online.reference_audit([audit_dataset(tmp_path, 1)], ["http://one"], tmp_path / "default", client=AuditClient())
    assert report["workers_per_endpoint"] == 2
    monkeypatch.setenv("SWE_AGENT_REFERENCE_AUDIT_WORKERS_PER_ENDPOINT", "8")
    report = online.reference_audit([tmp_path / "dataset.parquet"], ["http://one"], tmp_path / "override", client=AuditClient(), workers_per_endpoint=1)
    assert report["workers_per_endpoint"] == 1
    with pytest.raises(ValueError, match="signature"):
        online.reference_audit([tmp_path / "dataset.parquet"], ["http://one"], tmp_path / "override", client=AuditClient(), workers_per_endpoint=2, resume=True)


@pytest.mark.parametrize("workers", [0, -1, True, 1.5])
def test_reference_audit_rejects_invalid_parallelism_before_service_calls(tmp_path, workers):
    client = AuditClient()
    with pytest.raises(ValueError, match="positive integer"):
        online.reference_audit([], ["http://one"], tmp_path / "audit", client=client, workers_per_endpoint=workers)
    assert not client.calls


def test_reference_audit_quarantine_is_shared_between_concurrent_workers(tmp_path):
    barrier = threading.Barrier(2)
    cleanup_seen = threading.Event()
    release_other = threading.Event()
    class UncleanClient(AuditClient):
        unsafe_request = None
        def post(self, url, path, payload, deadline):
            result = super().post(url, path, payload, deadline)
            if path == "/claim":
                barrier.wait(timeout=10)
            if path == "/reward":
                with self.lock:
                    first = self.unsafe_request is None
                    if first:
                        self.unsafe_request = payload["request_id"]
                if not first:
                    assert cleanup_seen.wait(10)
                    assert release_other.wait(10)
            return result
        def get(self, url, path, **kwargs):
            result = super().get(url, path, **kwargs)
            if "key=" in path and self.unsafe_request in path:
                result["cleanup_state"]["complete"] = False
                cleanup_seen.set()
            return result
    client = UncleanClient()
    # Let the unsafe worker finish its bounded cleanup and quarantine the URL
    # before the other already-running arm returns. No gold/new claims allowed.
    original_write = online.audit_write_json
    def write(path, result):
        original_write(path, result)
        if path.name == "result.json" and result.get("status") == "cleanup_unverified":
            release_other.set()
    from unittest.mock import patch
    original_get = client.get
    def quick_probe(url, path, **kwargs):
        result = original_get(url, path, **kwargs)
        if "key=" in path and client.unsafe_request in path:
            # First probe fails by raising; audit_arm classifies cleanup unsafe.
            raise ConnectionError("cleanup verification unavailable")
        return result
    client.get = quick_probe
    with patch.object(online, "audit_write_json", side_effect=write):
        report = online.reference_audit([audit_dataset(tmp_path, 4)], ["http://one"], tmp_path / "audit", client=client, workers_per_endpoint=2)
    assert len([c for c in client.calls if c[1] == "/claim"]) == 2
    assert report["counts"] == {"cleanup_unverified": 1, "endpoint_quarantined": 3}
    assert report["quarantined_endpoints"] == ["http://one"]


def test_reference_audit_fast_endpoint_steals_remaining_compatible_cases(tmp_path):
    fast_progress = threading.Event()
    class UnevenClient(AuditClient):
        completed_fast_rewards = 0
        def post(self, url, path, payload, deadline):
            result = super().post(url, path, payload, deadline)
            if path == "/reward" and url == "http://slow":
                assert fast_progress.wait(10)
            if path == "/reward" and url == "http://fast":
                self.completed_fast_rewards += 1
                if self.completed_fast_rewards >= 4:
                    fast_progress.set()
            return result
    client = UnevenClient()
    report = online.reference_audit([audit_dataset(tmp_path, 5)], ["http://slow", "http://fast"], tmp_path / "audit", client=client, workers_per_endpoint=1)
    assert report["counts"] == {"reference_verified": 5}
    fast_tasks = {p["task"]["task_id"] for url, route, p in client.calls if url == "http://fast" and route == "/claim"}
    assert len(fast_tasks) >= 3


def test_reference_audit_resume_scans_late_unsafe_checkpoint_before_new_claims(tmp_path):
    client = AuditClient()
    path = audit_dataset(tmp_path, 2)
    directory = tmp_path / "audit"
    report = online.reference_audit([path], ["http://one"], directory, client=client)
    first, last = [directory / "cases" / r["id"] for r in report["results"]]
    (first / "result.json").unlink()
    arm_path = last / "gold" / "0.json"
    arm = json.loads(arm_path.read_text())
    arm.update(phase="reward", cleanup={"verified": False})
    online.audit_write_json(arm_path, arm)
    before = len(client.calls)
    report = online.reference_audit([path], ["http://one"], directory, client=client, resume=True)
    assert len(client.calls) == before
    assert report["results"][0]["status"] == "endpoint_quarantined"


@pytest.mark.parametrize("f2p,p2p,expected", [(1, True, "reference_verified"), (0, True, "baseline_required_tests_review"), (1, False, "baseline_required_tests_review")])
def test_reference_audit_requires_intended_baseline_failure(f2p, p2p, expected):
    result = {"arms": {
        "gold": {"classification": "tests_passed", "cleanup": {"verified": True}},
        "baseline": {"classification": "required_tests_failed", "cleanup": {"verified": True}, "reward": {"metadata": {
            "official_fail_to_pass_failed": f2p, "official_pass_to_pass_complete": p2p}}}}}
    assert online.audit_result_status(result) == expected


def test_reference_audit_rejects_old_service_scoring_for_official_contract():
    arm = {"mode": "baseline", "official_test_contract_required": True, "claim": {"ok": True},
           "base": {"ok": True, "text": "exit_code: 0\n"}, "reward": {"ok": True, "score": 1, "metadata": {
               "hidden_tests_started": True, "hidden_runner_provenance": "launch_marker_v1",
               "hidden_test_counts_complete": True, "hidden_tests_total_count": 1}}}
    assert online.audit_classify(arm) == "official_scoring_unavailable"
    arm["reward"]["metadata"]["failure_reason"] = "silent_command_timeout"
    assert online.audit_classify(arm) == "evaluation_timeout"


@pytest.mark.parametrize("fault", [None, "hash", "launch", "pytest"])
def test_reference_replay_reverifies_retained_evidence(tmp_path, fault):
    import gzip
    import hashlib
    import pyarrow as pa
    import pyarrow.parquet as pq
    path = audit_dataset(tmp_path, 1)
    rows = pq.read_table(path).to_pylist()
    task = json.loads(rows[0]["extra_info"]["tools_kwargs"]["__swe_task_json__"])
    runner = "pytest -v" if fault == "pytest" else "python tests/runtests.py -v 2"
    task["test_commands"][0] = task["test_commands"][0].rsplit("\n", 1)[0] + "\n" + runner
    rows[0]["extra_info"]["tools_kwargs"]["__swe_task_json__"] = json.dumps(task)
    rows[0]["reward_model"] = {"ground_truth": json.dumps({"benchmark": "swe_bench_verified",
        "expected": "all_official_tests_pass", "fail_to_pass": ["test_required (suite.Case)"], "pass_to_pass": []})}
    pq.write_table(pa.Table.from_pylist(rows), path)
    _, cases = online.audit_inventory([path])
    assert cases[0]["inventory"]["eligible"] and cases[0]["task"]["official_test_contract"]["version"] == 1
    marker = "" if fault == "launch" else "SWE_RUNNER_LAUNCHED_abcdef\n"
    raw = ("exit_code: 0\n" + marker + "test_required (suite.Case) ... ok\n"
           "\n----------------------------------------------------------------------\nRan 1 test in 0.1s\n\nOK\n").encode()
    compressed = gzip.compress(raw, mtime=0)
    digest = hashlib.sha256(compressed).hexdigest()
    raw_digest = hashlib.sha256(raw).hexdigest()
    private = tmp_path / "private"
    private.mkdir()
    (private / (digest + ".log.gz")).write_bytes(compressed)
    ref = {"file": digest + ".log.gz", "sha256": digest, "compression": "gzip", "bytes": len(compressed),
           "uncompressed_bytes": len(raw), "original_bytes": len(raw), "uncompressed_sha256": raw_digest, "complete": True}
    data = json.dumps({"task_id": "case-0", "phases": {"test_commands": {"sha256": raw_digest, "full_log": ref}}}).encode()
    artifact_sha = hashlib.sha256(data).hexdigest()
    artifact = private / (artifact_sha + ".json")
    artifact.write_bytes(data if fault != "hash" else b"modified after diagnosis")
    diagnosis = tmp_path / "diagnosis.json"
    diagnosis.write_text(json.dumps({"reference_audits": [{"arms": [{"task_id": "case-0", "mode": "gold",
        "runner_provenance": "launch_marker_v1", "private_evaluator": {"status": "verified", "sha256": artifact_sha,
        "private_copy": str(artifact)}}]}]}))
    record = online.reference_replay([path], diagnosis, tmp_path / "replay/report.json")["cases"][0]
    if fault is None:
        assert record["official_tests_success"] and record["official_tests_complete"]
    else:
        assert not record.get("official_tests_success")
        assert record["reason"] == {"hash": "unverified_or_incomplete_private_logs", "launch": "runner_launch_unverified",
                                    "pytest": "unsupported_runner"}[fault]


def retained_arm(tmp_path, task_id, mode, phases):
    import gzip
    import hashlib
    private = tmp_path / "private-fixture"
    private.mkdir(exist_ok=True)
    refs = {}
    for name, value in phases.items():
        raw = value.encode()
        blob = gzip.compress(raw, mtime=0)
        digest = hashlib.sha256(blob).hexdigest()
        raw_digest = hashlib.sha256(raw).hexdigest()
        (private / (digest + ".log.gz")).write_bytes(blob)
        refs[name] = {"sha256": raw_digest, "full_log": {"file": digest + ".log.gz", "sha256": digest,
            "compression": "gzip", "bytes": len(blob), "uncompressed_bytes": len(raw), "original_bytes": len(raw),
            "uncompressed_sha256": raw_digest, "complete": True}}
    data = json.dumps({"task_id": task_id, "phases": refs}).encode()
    digest = hashlib.sha256(data).hexdigest()
    artifact = private / (digest + ".json")
    artifact.write_bytes(data)
    return {"task_id": task_id, "mode": mode, "runner_provenance": "launch_marker_v1",
            "private_evaluator": {"status": "verified", "sha256": digest, "private_copy": str(artifact)}}


@pytest.mark.parametrize("raw_xml,mutation", [(False, None), (True, None), (True, "invalid"), (True, "missing")])
def test_replay_structured_and_raw_junit_evidence(tmp_path, raw_xml, mutation):
    from recipe.swe_agent.test_results import official_test_outcomes
    task = {"evaluator_test_commands": [{"run": "pytest tests/a.py", "prepare": []}]}
    xml = '<testsuite tests="1"><testcase classname="tests.a" name="test_ok"/></testsuite>'
    evidence = official_test_outcomes("pytest", "", exit_code=0, junit_xml=xml)
    phases = {"test_commands": "exit_code: 0\nSWE_RUNNER_LAUNCHED_abcdef\n", "official_tests": json.dumps({"evidence": [evidence]})}
    if raw_xml:
        phases["runner_evidence_0"] = json.dumps({"command": "pytest tests/a.py", "exit_code": 0,
                                                 "junit_requested": True, "evidence": evidence})
        if mutation != "missing":
            phases["junit_xml_0"] = xml if mutation is None else "<broken"
    if mutation == "missing":
        with pytest.raises(ValueError, match="missing_raw_junit"):
            online.replay_runner_evidence(task, phases)
        return
    replayed, kind = online.replay_runner_evidence(task, phases)
    assert kind == ("raw_junit_or_runner_evidence" if raw_xml else "retained_structured_score_replay")
    assert replayed[0]["complete"] == (mutation is None)


def test_dataset_version_preserves_original_schema_order_and_excludes_only_requested(tmp_path):
    import hashlib
    import pyarrow.parquet as pq
    path = audit_dataset(tmp_path, 5)
    original = path.read_bytes()
    table = pq.read_table(path)
    output = tmp_path / "v9.parquet"
    receipt = online.dataset_version(path, output, ["case-1", "case-3"])
    assert path.read_bytes() == original
    assert receipt["source_sha256"] == hashlib.sha256(original).hexdigest()
    assert receipt["rows"] == 3 and receipt["retained_rows_unchanged"]
    result = pq.read_table(output)
    assert result.schema == table.schema
    assert result.to_pylist() == [table.to_pylist()[i] for i in (0, 2, 4)]
    with pytest.raises(ValueError, match="overwrite"):
        online.dataset_version(path, output, [])
    with pytest.raises(ValueError, match="all exclusions present"):
        online.dataset_version(path, tmp_path / "bad.parquet", ["absent"])


def test_dataset_version_pins_only_verified_complete_baseline_gold_members(tmp_path):
    import hashlib
    import pyarrow as pa
    import pyarrow.parquet as pq
    path = audit_dataset(tmp_path, 1)
    rows = pq.read_table(path).to_pylist()
    rows[0]["reward_model"] = {"ground_truth": json.dumps({"benchmark": "swe_bench_verified",
        "expected": "all_official_tests_pass", "fail_to_pass": ["tests/a.py::test_value[axes"], "pass_to_pass": []})}
    pq.write_table(pa.Table.from_pylist(rows), path)
    arms = []
    for mode in ("baseline", "gold"):
        evidence = {"complete": True, "framework": "pytest_junit", "aliases": {}, "reason": "complete",
            "outcomes": {"tests.a::test_value[axes one]": "failed" if mode == "baseline" else "passed",
                         "tests.a::test_value[axes two]": "passed"}}
        arms.append(retained_arm(tmp_path, "case-0", mode, {"test_commands": "exit_code: 0\nSWE_RUNNER_LAUNCHED_abcdef\n",
                                                           "official_tests": json.dumps({"evidence": [evidence]})}))
    diagnosis = tmp_path / "diagnosis.json"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"signature": "audit", "identity": {"manifest": {"datasets": [
        {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}]}}}))
    source_report = tmp_path / "source-audit.json"
    source_report.write_text(json.dumps({"source_match": True, "signature": "audit", "manifest": str(manifest),
        "results": [{"task_id": "case-0", "arms": {a["mode"]: {"reward": {"metadata": {
            "private_evaluator_artifact_sha256": a["private_evaluator"]["sha256"],
            "hidden_runner_provenance": "launch_marker_v1"}}} for a in arms}}]}))
    diagnosis.write_text(json.dumps({"reference_audits": [{"arms": arms, "path": str(source_report),
        "sha256": hashlib.sha256(source_report.read_bytes()).hexdigest()}]}))
    output = tmp_path / "pinned.parquet"
    receipt = online.dataset_version(path, output, [], diagnosis)
    assert receipt["identity_upgrades"]["case-0"]["members"] == 2
    replay = online.reference_replay([output], diagnosis, tmp_path / "replay/report.json")
    assert all(r["official_tests_complete"] for r in replay["cases"])
    assert [r["official_tests_success"] for r in replay["cases"]] == [False, True]
    from recipe.swe_agent.task_payload import rollout_task_payload
    _, cases = online.audit_inventory([output])
    assert "official_test_contract" not in rollout_task_payload(cases[0]["task"])
    altered = tmp_path / "altered.parquet"
    pq.write_table(pa.Table.from_pylist(rows + rows), altered)
    with pytest.raises(ValueError):
        online.dataset_version(altered, tmp_path / "wrong-source.parquet", [], diagnosis)


def test_reference_audit_missing_artifacts_and_interrupted_attempt_are_not_replayed(tmp_path):
    path = audit_dataset(tmp_path)
    manifest, cases = online.audit_inventory([path])
    assert manifest["counts"]["eligible"] == 2
    (tmp_path / "gold.patch").unlink()
    client = AuditClient()
    report = online.reference_audit([path], ["http://one"], tmp_path / "missing", client=client)
    assert report["counts"] == {"unavailable_data": 2}
    assert not client.calls
    path = audit_dataset(tmp_path)
    report = online.reference_audit([path], ["http://one"], tmp_path / "interrupted", limit=1, client=client)
    case = report["results"][0]["id"]
    directory = tmp_path / "interrupted" / "cases" / case
    (directory / "result.json").unlink()
    online.write_json(directory / "gold" / "0.json", {"phase": "patch", "request_id": "uncertain"})
    count = len(client.calls)
    report = online.reference_audit([path], ["http://one"], tmp_path / "interrupted", limit=1, resume=True, client=client)
    assert len(client.calls) == count
    assert report["counts"] == {"unavailable_data": 1}


@pytest.mark.parametrize("metadata,score,expected", [
    ({"hidden_tests_started": True}, 1., "runner_launch_unverified"),
    ({"hidden_tests_started": True, "hidden_runner_provenance": "launch_marker_v1"}, 1., "test_coverage_unknown"),
    ({"hidden_failure_phase": "evaluator_preparation"}, 0., "evaluator_preparation_failure"),
    ({"terminal_reward_timeout": True}, 0., "evaluation_timeout"),
])
def test_reference_audit_reward_one_does_not_override_missing_evidence(metadata, score, expected):
    arm = {"mode": "baseline", "claim": {"ok": True}, "base": {"ok": True, "text": "exit_code: 0\n"},
           "reward": {"ok": True, "score": score, "metadata": metadata}}
    assert online.audit_classify(arm) == expected


def test_audit_path_mapping_has_component_boundary_and_longest_prefix():
    assert str(online.audit_mapped_path("/cluster/data/file", ["/cluster=/ssd", "/cluster/data=/shared"])) == "/shared/file"
    assert str(online.audit_mapped_path("/cluster-other/file", ["/cluster=/ssd"])) == "/cluster-other/file"
    with pytest.raises(ValueError, match="ABSOLUTE"):
        online.audit_mapped_path("/cluster/file", ["relative=/ssd"])


def test_reference_audit_rejects_concurrent_output_writer(tmp_path):
    import fcntl
    directory = tmp_path / "audit"
    directory.mkdir()
    with (directory / ".audit.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="another audit"):
            online.reference_audit([], ["http://one"], directory, client=AuditClient())


def test_reference_audit_never_pulls_or_builds_images(tmp_path):
    _, cases = online.audit_inventory([audit_dataset(tmp_path, 1)])
    task = online.audit_task(cases[0]["task"], 30)
    assert task["docker_pull"] is False
    assert task["environment"]["base_image"] == task["case_image"] == task["docker_image"]
    assert task["evaluator_test_commands"][0]["run"] == "pytest tests/a.py"
    with pytest.raises(ValueError, match="builds"):
        online.audit_task({**cases[0]["task"], "dockerfile": "FROM fixture"}, 30)


def test_reference_audit_prefers_endpoint_with_cached_case_image(tmp_path):
    class CachedClient(AuditClient):
        def get(self, url, path, **kwargs):
            result = super().get(url, path, **kwargs)
            if "?" not in path:
                result["image_inventory_complete"] = True
                result["cached_images"] = [{"tag": "cached", "id": "sha256:fixture"}] if url == "http://cached" else []
            return result
    client = CachedClient()
    report = online.reference_audit([audit_dataset(tmp_path, 2)], ["http://missing", "http://cached"], tmp_path / "audit", client=client)
    assert all(url == "http://cached" for url, _, _ in client.calls)
    assert all(result["cached_image_id"] == "sha256:fixture" for result in report["results"])


@pytest.mark.parametrize("available", [False, True])
def test_reference_audit_exact_image_probe_resolves_truncated_inventory(tmp_path, available):
    class ProbedClient(AuditClient):
        def get(self, url, path, **kwargs):
            if "image=" in path:
                return {"image_probe": {"tag": "cached", "available": available, "id": "sha256:fixture" if available else None}}
            result = super().get(url, path, **kwargs)
            if "?" not in path:
                result.update(cached_images=[], image_inventory_complete=False)
            return result
    client = ProbedClient()
    report = online.reference_audit([audit_dataset(tmp_path, 1)], ["http://one"], tmp_path / "audit", client=client)
    assert report["counts"] == ({"reference_verified": 1} if available else {"image_unavailable": 1})
    assert bool(client.calls) is available


@pytest.mark.parametrize("code,output,expected", [
    (0, "sha256:fixture\n", True), (1, "Error: No such image: fixture", False), (1, "daemon unavailable", None),
])
def test_exact_image_diagnostic_never_pulls_or_builds(code, output, expected):
    commands = []
    def docker(command, **kwargs):
        commands.append(command)
        return code, output
    result = info.deployment_info(manager_stub(), docker, {}, image_tag="fixture")
    assert result["image_probe"]["available"] is expected
    assert commands == [["docker", "image", "inspect", "--format", "{{.Id}}", "--", "fixture"]]


def test_reference_audit_quarantines_endpoint_after_unverified_cleanup(tmp_path):
    class UncleanClient(AuditClient):
        def get(self, url, path, **kwargs):
            result = super().get(url, path, **kwargs)
            if "key=" in path:
                result["cleanup_state"]["complete"] = False
            return result
    client = UncleanClient()
    from unittest.mock import patch
    with patch.object(online.time, "monotonic", side_effect=range(0, 100000, 100)):
        report = online.reference_audit([audit_dataset(tmp_path, 2)], ["http://one"], tmp_path / "audit", client=client)
    assert report["counts"] == {"cleanup_unverified": 1, "endpoint_quarantined": 1}
    assert len([c for c in client.calls if c[1] == "/claim"]) == 1


@pytest.mark.parametrize("checkpoint", ["result", "complete_arm", "interrupted_arm", "interrupted_retry"])
def test_reference_audit_resume_preserves_endpoint_quarantine(tmp_path, checkpoint):
    path = audit_dataset(tmp_path, 2)
    client = AuditClient()
    directory = tmp_path / "audit"
    report = online.reference_audit([path], ["http://one"], directory, client=client)
    first = directory / "cases" / report["results"][0]["id"]
    second = directory / "cases" / report["results"][1]["id"]
    arm_path = first / "baseline" / "0.json"
    arm = json.loads(arm_path.read_text())
    arm["cleanup"] = {"verified": False}
    if checkpoint in {"interrupted_arm", "interrupted_retry"}:
        arm["phase"] = "claim"
    if checkpoint == "interrupted_retry":
        arm_path = first / "baseline" / "1.json"
    online.audit_write_json(arm_path, arm)
    if checkpoint == "result":
        result = json.loads((first / "result.json").read_text())
        result["arms"]["baseline"] = arm
        result["status"] = "cleanup_unverified"
        online.audit_write_json(first / "result.json", result)
    else:
        (first / "result.json").unlink()
    (second / "result.json").unlink()
    count = len(client.calls)
    report = online.reference_audit([path], ["http://one"], directory, resume=True, client=client)
    assert len(client.calls) == count
    assert report["results"][1]["status"] == "endpoint_quarantined"
    assert report["results"][0]["status"] == (
        "unavailable_data" if checkpoint in {"interrupted_arm", "interrupted_retry"} else "cleanup_unverified")


@pytest.mark.parametrize("failure", ["transport", "cleanup", "source"])
def test_reference_audit_retry_and_failure_gates(tmp_path, failure):
    class FailingClient(AuditClient):
        interrupted = False

        def get(self, url, path, **kwargs):
            result = super().get(url, path, **kwargs)
            if failure == "source" and "?" not in path:
                result["source_fingerprint"] = {"sha256": "old-service"}
            if failure == "cleanup" and "?" in path:
                result["cleanup_state"]["complete"] = False
            return result

        def post(self, url, path, payload, deadline):
            if failure == "transport" and path == "/reward" and not self.interrupted:
                self.interrupted = True
                raise ConnectionError("connection dropped")
            return super().post(url, path, payload, deadline)

    client = FailingClient()
    if failure == "cleanup":
        # Avoid waiting for a deliberately non-completing cleanup probe.
        from unittest.mock import patch
        with patch.object(online.time, "monotonic", side_effect=range(0, 100000, 100)):
            report = online.reference_audit([audit_dataset(tmp_path, 1)], ["http://one"], tmp_path / "audit", client=client)
    else:
        report = online.reference_audit([audit_dataset(tmp_path, 1)], ["http://one"], tmp_path / "audit", client=client)
    claims = [c for c in client.calls if c[1] == "/claim"]
    assert len(claims) == (3 if failure == "transport" else 1 if failure == "cleanup" else 2)
    assert report["regression_pass"] is (failure == "transport")
    if failure == "cleanup":
        assert report["counts"] == {"cleanup_unverified": 1}


def manager_stub():
    return SimpleNamespace(lock=threading.Lock(), **{name: set() for name in (
        "sessions", "active_trajectory_keys", "creating_keys", "stopping_keys",
        "executing_keys", "claiming_keys", "aborting_keys",
    )})


def test_frozen_manifest_and_checksum(tmp_path):
    first = online.build(tmp_path)
    assert online.build(tmp_path) == first
    assert len(online.load(tmp_path)[1]) >= 14
    cases = online.scenarios()
    cases[0]["steps"][0]["expected"] = {"contains": "wrong"}
    online.write_json(tmp_path / "scenarios.json", cases)
    with pytest.raises(ValueError, match="checksum"):
        online.load(tmp_path)
    with pytest.raises(ValueError, match="Frozen"):
        online.build(tmp_path)


def test_artifact_oracle_checks_retrieved_bytes_independently():
    import hashlib
    raw = "exit_code: 0\nlogin banner\n" + "needle\n" * 1000
    expected = {"prefix": "exit_code: 0\n", "suffix": "needle\n" * 1000,
                "sha256": "__artifact_sha256_0__", "lines": "__artifact_lines_0__"}
    events = [{"metadata": {"output_artifact_sha256": hashlib.sha256(raw.encode()).hexdigest(),
                             "output_artifact_lines": 1002}}]
    bound = online.bind_artifacts({"expected": expected}, events)["expected"]
    assert online.check_response(bound, {"ok": True, "text": raw}) == []
    assert online.check_response(bound, {"ok": True, "text": raw[:-1]})
    with pytest.raises(ValueError, match="artifact sha256 missing"):
        online.bind_artifacts({"expected": expected}, [])


def test_trajectory_freezing_preserves_full_output_and_source_identity(tmp_path):
    import ast
    import hashlib

    rows = [{"task_id": "other", "output": ""} for _ in range(76)]
    reviewed = {
        45: (3, "scikit-learn__scikit-learn-26194"),
        56: (17, "pytest-dev__pytest-7571"),
        76: (42, "django__django-17087"),
    }
    observation = "BEGIN\n" + "runner output\n" * 2000 + "END\n"
    response = "exit_code: 0\n" + observation + "\n[test outcome] old parser result"
    pair = ('<function=Bash><parameter=command>python -m pytest</parameter></function>'
            '\nuser\n<tool_response>' + response + '</tool_response>\nassistant\n')
    for line, (event, task) in reviewed.items():
        rows[line - 1] = {"task_id": task, "output": pair * (event + 1) + "Done."}
    source = tmp_path / "0.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    cases = online.trajectory_replay_scenarios(source)
    assert len(cases) == 3
    for case, (line, (event, task)) in zip(cases, reviewed.items()):
        provenance = case["provenance"]
        assert (provenance["line"], provenance["event"], provenance["task"]) == (line, event, task)
        assert provenance["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
        assert provenance["saved_response_sha256"] == hashlib.sha256(response.encode()).hexdigest()
        program = ast.parse(case["steps"][0]["arguments"]["content"])
        assert program.body[1].value.args[0].value == observation
    manifest = online.build(tmp_path / "frozen", source)
    assert manifest["scenario_count"] == len(online.scenarios()) + len(online.v8_scenarios()) + len(online.v9_scenarios()) + 3
    rows[44]["task_id"] = "different-task"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="task mismatch"):
        online.trajectory_replay_scenarios(source)


def test_latest_saved_replay_preserves_original_commands_status_and_oracles(tmp_path):
    import hashlib
    cases = {
        19: (17, "django__django-13551", "python runtests.py | grep FAIL", "\nexit_code: 1\n"),
        30: (43, "django__django-17087", "python -m unittest missing", "\nexit_code: 1\nERROR: missing (unittest.loader._FailedTest.missing)\nImportError: Failed to import test module: missing\nRan 1 test in 0.1s\nFAILED (errors=1)\n"),
        31: (19, "django__django-15104", "python runtests.py | tail -50", "\nError: Command produced no output for 600s and was terminated after 600.096s."),
        89: (20, "django__django-16569", 'python -c "import unittest; runner=unittest.TextTestRunner(); result=runner.run(suite)"', "\nexit_code: 0\nRan 153 tests in 1.0s\nOK\n"),
    }
    rows = [{"task_id": "unused"} for _ in range(89)]
    for line, (event, task, command, response) in cases.items():
        pair = ('<function=Bash><parameter=command>' + command + '</parameter></function>\nuser\n<tool_response>' + response + '</tool_response>\nassistant\n')
        rows[line-1] = {"task_id": task, "output": pair * (event+1) + "Done."}
    source = tmp_path / "0.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    frozen = online.trajectory_replay_scenarios(source, latest=True)
    for case in frozen:
        assert not online.check_parser_replay(case)["failures"]
        assert case["provenance"]["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
        line = case["provenance"]["line"]
        assert case["parser_replay"]["command"] == cases[line][2]
        assert case["steps"] == []
    frozen[0]["parser_replay"]["expected"]["test_outcome"] = "failed"
    assert online.check_parser_replay(frozen[0])["failures"]


def test_validation_launcher_health_gates_use_selected_urls_and_abort_on_failure(tmp_path):
    import os
    import subprocess
    launcher = online.ROOT / "val_only.sh"
    assert subprocess.run(["bash", "-n", str(launcher)], capture_output=True).returncode == 0
    source = launcher.read_text()
    root = tmp_path / "swe"
    checkout = root / "verl"
    checkout.mkdir(parents=True)
    (root / "stock-rl-reflect").mkdir()
    (checkout / "scripts").mkdir()
    fixture_launcher = checkout / "val_only.sh"
    fixture_launcher.write_text(source)
    outside = tmp_path / "outside"
    outside.mkdir()
    parquet = tmp_path / "fixture.parquet"
    parquet.touch()
    env = {**os.environ, "SWE_VERL_DIR": str(checkout), "EXPERIMENT_NAME": "launcher-fixture",
           "TRAIN_FILES": json.dumps([str(parquet)]), "VAL_FILES": json.dumps([str(parquet)])}
    stubs = '''
wandb() { :; }
curl() { if [[ "$FIXTURE_HEALTH" == "bad" ]]; then printf 503; else printf 200; fi; }
python3() {
  case "$1" in
    -) command python3 "$@" ;;
    -m) return 0 ;;
    scripts/swe_tool_online_benchmark.py)
      printf '{}' > "$VALIDATION_DATA_DIR/service_preflight.json"
      [[ "$FIXTURE_PREFLIGHT" != "bad" ]]
      ;;
    *) return 99 ;;
  esac
}
nohup() { printf 'stub validation launched\\n'; }
'''

    def invoke(mode, health="ok", preflight="ok"):
        command = f'source {fixture_launcher!s}' if mode == "source" else source
        script = (stubs + '\n' + command + '\n'
                  + 'printf "launcher-status=%s\\n" "$?"\n'
                  + 'printf "shell-survived\\n"\nwait\n')
        return subprocess.run(["bash", "-c", script], cwd=outside,
                              env={**env, "FIXTURE_HEALTH": health, "FIXTURE_PREFLIGHT": preflight},
                              capture_output=True, text=True, timeout=10)

    bad_health = invoke("paste", health="bad")
    assert bad_health.returncode == 0
    assert "launcher-status=1" in bad_health.stdout and "shell-survived" in bad_health.stdout
    assert " FAIL " in bad_health.stdout
    assert not (checkout / "launcher-fixture.log").exists()

    bad_preflight = invoke("paste", preflight="bad")
    assert bad_preflight.returncode == 0
    assert "launcher-status=1" in bad_preflight.stdout and "shell-survived" in bad_preflight.stdout
    assert (checkout / "validation_data" / "launcher-fixture" / "service_preflight.json").exists()
    assert not (checkout / "launcher-fixture.log").exists()

    success = invoke("source")
    assert success.returncode == 0, success.stderr
    assert "launcher-status=0" in success.stdout and "shell-survived" in success.stdout
    assert success.stdout.count(" OK ") == 6
    assert "Validation launcher PID:" in success.stdout
    assert "stub validation launched" in (checkout / "launcher-fixture.log").read_text()

    protected = invoke("paste")
    assert "launcher-status=1" in protected.stdout
    assert "Refusing to overwrite existing run log" in protected.stderr
    assert (checkout / "launcher-fixture.log").read_text() == "stub validation launched\n"
    assert "export SWE_AGENT_TRAINING_REPEATED_TOOL_REWARD_SHAPING=0" in source


def test_offline_cli_runs_production_dispatch_and_fails_on_bad_oracle(tmp_path, monkeypatch):
    cases = [{"id": "offline-boundary", "expected_hits": None, "steps": [
        online.step("Write", {"file_path": "a.py", "content": "first\nsecond\n"}, contains="Wrote"),
        online.step("Read", {"file_path": "/testbed/a.py", "offset": 1, "limit": 1}, equals="second"),
    ]}]
    monkeypatch.setattr(online, "scenarios", lambda: cases)
    monkeypatch.setattr(online, "v8_scenarios", lambda: [])
    monkeypatch.setattr(online, "v9_scenarios", lambda: [])
    online.build(tmp_path / "pass")
    def unexpected(*args, **kwargs):
        pytest.fail("offline contracts must not call Docker")
    monkeypatch.setattr(service, "_docker_result", unexpected)
    report = online.offline(tmp_path / "pass", tmp_path / "pass.json")
    assert report["regression_pass"]
    assert report["counts"]["tool_dispatches"] == 4
    cases[0]["steps"][1]["expected"] = {"equals": "wrong oracle"}
    online.build(tmp_path / "fail")
    monkeypatch.setattr(online.sys, "argv", ["benchmark", "offline", "--benchmark-dir", str(tmp_path / "fail"),
                                           "--report", str(tmp_path / "fail.json")])
    assert online.main() == 1
    assert json.loads((tmp_path / "fail.json").read_text())["counts"]["failures"] == 2


def test_cached_task_cannot_pull_build_or_load_archive(monkeypatch):
    task = online.cached_task("cached:tag", "case", 30)
    assert service._resolve_docker_image_tag(task) == "cached:tag"
    assert service._archive_load_eligible(task) is False
    assert service._is_rollout_task(task)
    monkeypatch.setattr(service, "_image_exists", lambda tag: False)
    def unexpected(*args, **kwargs):
        pytest.fail("cached-only task attempted pull, build or archive load")
    monkeypatch.setattr(service, "_docker_pull_with_retries", unexpected)
    monkeypatch.setattr(service, "_docker_result", unexpected)
    monkeypatch.setattr(service, "_load_image_archive", unexpected)
    monkeypatch.setenv("SWE_AGENT_DOCKER_IMAGE_ARCHIVE_LOAD", "1")
    with pytest.raises(RuntimeError, match="Committed case image is missing"):
        service._build_image(task)


def test_synthetic_evaluator_fixture_uses_actual_reset_apply_and_private_evidence(tmp_path, monkeypatch):
    # Load the adjacent repository's repeat replay before collecting evidence.
    # A fresh CLI process must still import this repository's scripts package.
    repeat_case = {"id": "import-order-replay", "expected_hits": [False], "steps": [
        online.step("Write", {"file_path": "a.py", "content": "first\n"}, contains="Wrote") ]}
    monkeypatch.setattr(online, "scenarios", lambda: [repeat_case])
    monkeypatch.setattr(online, "v8_scenarios", lambda: [online.evaluator_fixture_case()])
    monkeypatch.setattr(online, "v9_scenarios", lambda: [])
    online.build(tmp_path / "fixture")
    report = online.offline(tmp_path / "fixture", tmp_path / "report.json")
    assert report["regression_pass"]
    for case in report["cases"]:
        if "evaluator" not in case:
            continue
        assert case["evaluator"]["score"] == 1
        meta = case["evaluator"]["metadata"]
        assert meta["hidden_verifier_setup_ok"] if "hidden_verifier_setup_ok" in meta else meta["verifier_setup_ok"]
        assert meta["private_evaluator_artifact_status"] == "available"
        assert "_private_evaluator_phases" not in meta
        assert case["private_evidence_collection"]["logs_complete"]
    import subprocess
    completed = subprocess.run([online.sys.executable, str(online.ROOT / "scripts/swe_tool_online_benchmark.py"),
        "offline", "--benchmark-dir", str(tmp_path / "fixture"), "--report", str(tmp_path / "cli.json")],
        capture_output=True, text=True, timeout=60, cwd=tmp_path)
    assert completed.returncode == 0, completed.stderr
    assert json.loads((tmp_path / "cli.json").read_text())["regression_pass"]


def test_diagnostics_inventory_cleanup_and_errors(tmp_path):
    manager = manager_stub()
    calls = []
    def docker(command, **kwargs):
        calls.append(command)
        assert kwargs["timeout"] == 5
        return 0, "python:3\tsha256:abc\n<none>:<none>\tsha256:def\n" if "image" in command else ""
    fingerprint = info.source_fingerprint()
    assert fingerprint == info.source_fingerprint()
    data = info.deployment_info(manager, docker, fingerprint)
    assert data["cached_images"] == [{"tag": "python:3", "id": "sha256:abc"}]
    assert data["image_inventory_complete"]
    data = info.deployment_info(manager, docker, fingerprint, "key", container_name="swe-contract",
                                workspace_root=tmp_path, workspace_path=str(tmp_path / "deleted"))
    assert not any(data["session_state"].values())
    assert data["cleanup_state"]["complete"]
    assert calls[-1][0:3] == ["docker", "container", "ls"]
    data = info.deployment_info(manager, docker, fingerprint, "key", container_name="swe-contract",
                                workspace_root=tmp_path, workspace_path="/etc/passwd")
    assert not data["cleanup_state"]["complete"]
    data = info.deployment_info(manager, lambda *a, **kw: (1, "daemon unavailable"), fingerprint)
    assert not data["image_inventory_complete"]
    assert data["cached_images"] == []
    def missing_docker(*args, **kwargs):
        raise FileNotFoundError("docker")
    data = info.deployment_info(manager, missing_docker, fingerprint)
    assert data["image_inventory_complete"] is False
    assert "FileNotFoundError" in data["image_inventory_error"]


def test_diagnostics_route_auth_and_query(monkeypatch, tmp_path):
    monkeypatch.setattr(service, "_docker_result", lambda *a, **kw: (0, ""))
    monkeypatch.setattr(service, "_workspace_root", lambda: tmp_path)
    class Handler(service._ExecutionHandler):
        manager = manager_stub()
        auth_token = "test-token"
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}/tool_contract_info?key=abc"
    try:
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(url, timeout=3)
        assert error.value.code == 401
        request = urllib.request.Request(url, headers={"Authorization": "Bearer test-token"})
        with urllib.request.urlopen(request, timeout=3) as response:
            data = json.load(response)
        assert data["source_fingerprint"]["sha256"]
        assert data["session_state"]["creating"] is False
        assert "host_diagnostics" not in data
        monkeypatch.setattr(info, "host_diagnostics", lambda: {"scope": "execution_service_host", "complete": True})
        request = urllib.request.Request(url.split("?")[0], headers={"Authorization": "Bearer test-token"})
        with urllib.request.urlopen(request, timeout=3) as response:
            assert json.load(response)["host_diagnostics"]["scope"] == "execution_service_host"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


class FakeClient:
    def __init__(self, fail="", provenance=True, clean=True):
        self.calls = []
        self.fail = fail
        self.provenance = provenance
        self.clean = clean

    def get(self, url, path):
        if path == "/ready":
            return {"ok": True, "docker_available": True}
        if not self.provenance:
            return {"ok": False, "probe_error": "404"}
        return {"ok": True, "source_fingerprint": {"sha256": "abc"},
                "cached_images": [{"tag": "python:3", "id": "sha256:abc"}],
                "session_state": {"session_present": False}, "cleanup_state": {"complete": self.clean}}

    def post(self, url, path, payload, deadline):
        assert deadline > time.monotonic()
        self.calls.append((path, payload))
        if path == self.fail:
            raise TimeoutError("ambiguous transport failure")
        if path == "/claim":
            return {"ok": True}
        if path == "/release":
            return {"ok": True, "released": True, "container_removed": True}
        return {"ok": True, "text": "exit_code: 0\nPython 3\n", "metadata": {"container_name": "test", "workspace": "/testbed"}}


@pytest.mark.parametrize("fail", [False, True])
def test_online_synthetic_evaluator_requires_reward_and_releases_session(fail):
    class EvaluatorClient(FakeClient):
        def post(self, url, path, payload, deadline):
            response = super().post(url, path, payload, deadline)
            if path == "/reward":
                assert payload["task"]["trusted_hidden_test_patch"] is True
                assert payload["task"]["execution_phase"] == "reward"
                return {"ok": True, "score": 0. if fail else 1., "metadata": {
                    "hidden_tests_started": True, "evaluation_valid": True,
                    "hidden_failure_phase": "none", "private_evaluator_artifact_status": "available", "private_evaluator_logs_complete": True}}
            return response
    client = EvaluatorClient()
    result = online.run_case(client, "http://test", "python:3", online.evaluator_fixture_case(), 20)
    assert result["status"] == ("failed" if fail else "passed")
    paths = [path for path, _ in client.calls]
    assert paths.count("/reward") == 1 and paths[-1] == "/release"
    claim = client.calls[0][1]
    reward = next(payload for path, payload in client.calls if path == "/reward")
    assert claim["task"]["task_id"] == reward["task"]["task_id"]
    assert "hidden_test_patch" not in claim["task"]


def test_online_gate_keeps_saved_parser_replays_separate_from_docker_coverage(tmp_path, monkeypatch):
    cases = [{"id": "saved", "steps": [], "expected_hits": None, "parser_replay": {
        "command": "python runtests.py | grep FAIL", "output": "", "exit_code": 1,
        "expected": {"test_outcome": "unknown"}}}, {"id": "docker", "steps": [], "expected_hits": None}]
    monkeypatch.setattr(online, "load", lambda _: ({"version": online.VERSION}, cases))
    monkeypatch.setattr(online, "persistent_search_verified", lambda _: True)
    class CurrentClient(FakeClient):
        def get(self, url, path):
            result = super().get(url, path)
            result["source_fingerprint"] = info.source_fingerprint()
            return result
    client = CurrentClient()
    report = online.run(tmp_path, ["http://fixture"], tmp_path / "report.json", client=client)
    assert report["regression_pass"]
    assert report["counts"]["scenarios_executed"] == 2
    assert report["counts"]["local_saved_parser_replays"] == 1
    assert report["counts"]["docker_scenarios_executed"] == 1
    assert [path for path, _ in client.calls].count("/claim") == 1


@pytest.mark.parametrize("fail", ["/claim", "/execute"])
def test_cleanup_after_ambiguous_failure_without_mutation_replay(fail):
    client = FakeClient(fail=fail)
    case = {"id": "test", "steps": [online.step("Write", {"file_path": "a", "content": "x"}, contains="Wrote")], "expected_hits": None}
    result = online.run_case(client, "http://test", "python:3", case, 20)
    assert result["status"] == "incomplete"
    assert [path for path, payload in client.calls].count(fail) == 1
    assert client.calls[-1][0] == "/release"
    assert client.calls[-1][1]["force"]


def test_unknown_provenance_never_passes_and_checks_all_services(tmp_path):
    online.build(tmp_path / "frozen")
    client = FakeClient(provenance=False)
    report = online.run(tmp_path / "frozen", ["http://a", "http://b"], tmp_path / "report.json", client=client)
    assert report["regression_pass"] is False
    assert report["counts"]["scenarios_executed"] == 0
    assert all(item["status"] == "incomplete" for item in report["services"])


def test_incomplete_cli_returns_nonzero(tmp_path, monkeypatch):
    online.build(tmp_path / "frozen")
    monkeypatch.setattr(online, "Client", lambda: FakeClient(provenance=False))
    monkeypatch.setattr(online.sys, "argv", ["online", "run", "--benchmark-dir", str(tmp_path / "frozen"),
                                          "--report", str(tmp_path / "report.json"), "--urls", "http://a"])
    assert online.main() == 1
    assert json.loads((tmp_path / "report.json").read_text())["regression_pass"] is False


def test_diagnostics_busy_lock_and_bounded_inventory():
    manager = manager_stub()
    manager.lock.acquire()
    try:
        result = info.deployment_info(manager, lambda *a, **kw: (0, ""), {}, "key")
        assert result["session_state"] == "unknown"
    finally:
        manager.lock.release()
    result = info.deployment_info(manager, lambda *a, **kw: (0, "python:3\tsha256:abc\n" * 201), {})
    assert len(result["cached_images"]) == 200
    assert result["image_inventory_complete"] is False


class DiagnosisClient(FakeClient):
    def __init__(self, slots=1, ready=True, images=True, native_rg=True, provenance=True):
        super().__init__(provenance=provenance)
        self.slots, self.ready, self.images, self.native_rg = slots, ready, images, native_rg

    def get(self, url, path):
        if path == "/health":
            return {"ok": True, "max_active_trajectories": 80, "available_trajectory_slots": self.slots}
        if path == "/ready":
            return {"ok": self.ready, "docker_available": self.ready}
        data = super().get(url, path)
        if path == "/tool_contract_info" and self.provenance:
            data["source_fingerprint"] = info.source_fingerprint()
            data["native_rg_available"] = self.native_rg
            data["host_diagnostics"] = {"version": 1, "scope": "execution_service_host", "complete": True,
                                        "rg": {"status": "available" if self.native_rg else "missing",
                                               "functional": self.native_rg, "reason": "mock", "path": "/persistent/rg"},
                                        "rg_startup_status": "persistent_verified",
                                        "persistent_rg": {"ok": True, "version": rg_backend.VERSION, "target": rg_backend.TARGET,
                                                          "binary_sha256": rg_backend.BINARY_SHA256, "binary": "/persistent/rg"}}
            if not self.images:
                data["cached_images"] = []
                data["image_inventory_complete"] = False
        return data


@pytest.mark.parametrize('options', [{}, {'native_rg': False}, {'ready': False}, {'provenance': False}])
def test_read_only_preflight_covers_every_endpoint_and_whitelists_settings(tmp_path, monkeypatch, options):
    monkeypatch.setenv('WANDB_API_KEY', 'must-not-be-exported')
    monkeypatch.setenv('SWE_AGENT_TRAINING_REPEATED_TOOL_REWARD_SHAPING', '0')
    client = DiagnosisClient(**options)
    report = online.preflight(['http://a', 'http://b', 'http://a'], tmp_path / 'preflight.json', client)
    assert report['counts']['services'] == 2
    assert report['regression_pass'] == (not options)
    assert not client.calls
    assert report['settings']['SWE_AGENT_TRAINING_REPEATED_TOOL_REWARD_SHAPING'] == '0'
    assert 'must-not-be-exported' not in (tmp_path / 'preflight.json').read_text()


def test_preflight_rejects_stale_source_on_one_endpoint(tmp_path):
    class Mixed( DiagnosisClient):
        def get(self, url, path):
            result = super().get(url, path)
            if url == 'http://stale' and path == '/tool_contract_info':
                result['source_fingerprint'] = {'sha256': 'old'}
            return result
    report = online.preflight(['http://current', 'http://stale'], tmp_path / 'preflight.json', Mixed())
    assert not report['regression_pass']
    assert not report['services'][0]['failures']
    assert report['services'][1]['failures'] == ['deployment source differs or is unavailable']


def test_dynamic_artifact_binding_and_nested_metadata_oracles():
    item = online.step('Read', {'file_path': '__artifact_path_0__'}, metadata_paths={'public_tests_passed': None})
    with pytest.raises(ValueError, match='artifact path missing'):
        online.bind_artifacts(item, [])
    bound = online.bind_artifacts(item, [{'metadata': {'output_artifact_path': '.swe_agent/tool_outputs/a.log'}}])
    assert bound['arguments']['file_path'] == '.swe_agent/tool_outputs/a.log'
    assert not online.check_response({'metadata_paths': {'public_tests_passed': None}}, {'ok': True, 'metadata': {'public_tests_passed': None}})
    assert online.check_response({'metadata_paths': {'public_tests_passed': None}}, {'ok': True, 'metadata': {}})


@pytest.mark.parametrize("options,reason", [
    ({"slots": 0}, "free trajectory slot"),
    ({"slots": None}, "free trajectory slot"),
    ({"ready": False}, "readiness"),
    ({"images": False}, "inventory may be truncated"),
    ({"provenance": False}, "deployment diagnostics"),
])
def test_service_diagnosis_never_claims_when_capacity_or_environment_unknown(tmp_path, options, reason):
    client = DiagnosisClient(**options)
    report = online.diagnose(["http://a", "http://a", "http://b"], tmp_path / "report.json", client=client)
    assert len(report["services"]) == 2
    assert report["counts"]["isolated_probes"] == 0
    assert not client.calls
    assert not report["regression_pass"]
    assert all(any(reason in item for item in service["incomplete"]) for service in report["services"])


def test_service_diagnosis_does_not_infer_functional_search_from_rg_presence(tmp_path, monkeypatch):
    client = DiagnosisClient(native_rg=True)
    def failed_probe(*args):
        return {"status": "failed", "failures": [{"index": 4, "error": "Unicode search failed"}]}
    monkeypatch.setattr(online, "run_case", failed_probe)
    report = online.diagnose(["http://a"], tmp_path / "report.json", client=client)
    assert report["counts"]["host_rg_on_path"] == 1
    assert report["counts"]["failed"] == 1
    assert not report["regression_pass"]
    assert json.loads((tmp_path / "report.json").read_text())["probe_sha256"]


def test_service_diagnosis_old_host_evidence_still_probes_but_never_passes(tmp_path, monkeypatch):
    class OldClient(DiagnosisClient):
        def get(self, url, path):
            data = super().get(url, path)
            data.pop("host_diagnostics", None)
            return data
    calls = []
    def probe(*args):
        calls.append(args)
        return {"status": "passed", "failures": [], "incomplete": []}
    monkeypatch.setattr(online, "run_case", probe)
    report = online.diagnose(["http://old"], tmp_path / "old.json", client=OldClient())
    assert len(calls) == 1
    assert report["services"][0]["host_rg_status"] == "unknown"
    assert report["counts"]["host_diagnostics_complete"] == 0
    assert report["services"][0]["status"] == "incomplete"
    assert not report["regression_pass"]


def host_probe_setup(monkeypatch, *, rg=None, package="absent", version=None, fail_semantics=False):
    import subprocess
    monkeypatch.setattr(info.shutil, "which", lambda name: rg if name == "rg" else "/usr/bin/dpkg-query" if name == "dpkg-query" else None)
    monkeypatch.setattr(info.Path, "exists", lambda self: False)
    monkeypatch.setattr(info.Path, "is_file", lambda self: False)
    calls = []
    def run(command, **kwargs):
        calls.append((command, kwargs))
        assert 0 < kwargs["timeout"] <= 0.35
        assert kwargs["capture_output"] and kwargs["encoding"] == "utf-8"
        if command[0].endswith("dpkg-query"):
            if package == "unknown":
                raise subprocess.TimeoutExpired(command, kwargs["timeout"])
            assert kwargs["env"]["LC_ALL"] == "C"
            return subprocess.CompletedProcess(command, 0 if package == "installed" else 1,
                "install ok installed\t14.1\n" if package == "installed" else "",
                "" if package == "installed" else "dpkg-query: no packages found matching ripgrep\n")
        if "--version" in command:
            if isinstance(version, Exception):
                raise version
            return subprocess.CompletedProcess(command, 0, version or "ripgrep 14.1.0\n", "")
        assert kwargs["input"] == "Needle caf\u00e9\nItem(\n"
        assert kwargs["env"] is None  # Match production's service environment.
        pattern = command[-1]
        output = "1:Needle caf\u00e9\n" if pattern in {"needle", "caf\u00e9"} else "" if "-F" in command else "2:Item(\n"
        return subprocess.CompletedProcess(command, 1 if not output or fail_semantics else 0,
                                           "" if fail_semantics else output, "")
    monkeypatch.setattr(info.subprocess, "run", run)
    return calls


@pytest.mark.parametrize("package,reason", [
    ("absent", "package_not_installed"), ("installed", "installed_but_not_on_path"),
    ("unknown", "absence_cause_unknown"),
])
def test_host_diagnostics_missing_rg_does_not_confuse_path_and_package(monkeypatch, package, reason):
    calls = host_probe_setup(monkeypatch, package=package)
    data = info._collect_host_diagnostics()
    assert data["scope"] == "execution_service_host"
    assert data["rg"]["status"] == "missing"
    assert data["rg"]["reason"] == reason
    assert data["ripgrep_package"]["status"] == package
    assert len(calls) == 1


@pytest.mark.parametrize("executable,reason", [(True, "binary_not_on_path"), (False, "binary_not_executable")])
def test_host_diagnostics_standard_binary_outside_path(monkeypatch, executable, reason):
    host_probe_setup(monkeypatch)
    monkeypatch.setattr(info.Path, "exists", lambda self: str(self) == "/usr/local/bin/rg")
    monkeypatch.setattr(info.Path, "is_file", lambda self: str(self) == "/usr/local/bin/rg")
    monkeypatch.setattr(info.os, "access", lambda *args: executable)
    assert info._collect_host_diagnostics()["rg"]["reason"] == reason


def test_host_diagnostics_rg_requires_actual_unicode_regex_and_literal_execution(monkeypatch):
    calls = host_probe_setup(monkeypatch, rg="/opt/bin/rg", package="unknown")
    data = info._collect_host_diagnostics()
    assert data["rg"]["status"] == "available"
    assert data["rg"]["functional"] is True
    assert len(calls) == 6
    assert set(data["rg"]["semantic_probes"]) == {"unicode_ignore_case", "unicode_literal", "unicode_regex", "literal_no_match"}
    assert all(item["passed"] for item in data["rg"]["semantic_probes"].values())
    assert data["ripgrep_package"]["status"] == "unknown"
    host_probe_setup(monkeypatch, rg="/opt/bin/rg", fail_semantics=True)
    data = info._collect_host_diagnostics()
    assert data["rg"]["status"] == "incompatible"
    assert not data["rg"]["functional"]


@pytest.mark.parametrize("version,status", [
    (PermissionError("not executable"), "execution_failed"),
    (OSError(8, "Exec format error"), "execution_failed"),
    ("unrelated binary\n", "incompatible"),
])
def test_host_diagnostics_incompatible_executable_never_passes(monkeypatch, version, status):
    calls = host_probe_setup(monkeypatch, rg="/opt/bin/rg", version=version)
    data = info._collect_host_diagnostics()
    assert data["rg"]["status"] == status
    assert not data["rg"]["functional"]
    assert len(calls) == 2


def test_host_diagnostics_budget_timeout_and_output_bounds(monkeypatch):
    import subprocess
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])
    monkeypatch.setattr(info.subprocess, "run", run)
    assert info._host_command(["rg"], time.monotonic() - 1)["error"] == "probe budget exhausted"
    assert not calls
    assert "TimeoutExpired" in info._host_command(["rg"], time.monotonic() + 1)["error"]
    monkeypatch.setattr(info.subprocess, "run", lambda *a, **kw: subprocess.CompletedProcess(a[0], 0, "x" * 5000, "y" * 2000))
    data = info._host_command(["rg"], time.monotonic() + 1)
    assert len(data["stdout"]) == 4000 and len(data["stderr"]) == 1000
    assert data["output_truncated"]


def test_host_diagnostics_cache_busy_and_cleanup_never_run_probes(monkeypatch):
    calls = []
    monkeypatch.setattr(info, "_host_cache", None)
    monkeypatch.setattr(info, "_host_lock", threading.Lock())
    def collect():
        calls.append(True)
        return {"complete": True, "nested": {"value": 1}}
    monkeypatch.setattr(info, "_collect_host_diagnostics", collect)
    info.host_diagnostics()["nested"]["value"] = 2
    assert info.host_diagnostics()["nested"]["value"] == 1
    assert len(calls) == 1
    monkeypatch.setattr(info, "_host_cache", (time.monotonic() - 61, {}))
    assert info.host_diagnostics()["complete"]
    assert len(calls) == 2
    info._host_lock.acquire()
    try:
        assert info.host_diagnostics()["complete"] is False
    finally:
        info._host_lock.release()
    monkeypatch.setattr(info, "host_diagnostics", lambda: pytest.fail("cleanup poll ran host probes"))
    assert "host_diagnostics" not in info.deployment_info(manager_stub(), lambda *a, **kw: (0, ""), {}, "key")


def test_service_diagnosis_probes_all_available_services_and_preserves_source_gate(tmp_path, monkeypatch):
    class DifferentSourceClient(DiagnosisClient):
        def get(self, url, path):
            data = super().get(url, path)
            if url == "http://b" and path == "/tool_contract_info":
                data["source_fingerprint"] = {"sha256": "different"}
            return data
    calls = []
    def passed_probe(client, url, image, case, deadline):
        calls.append((url, image))
        return {"status": "passed", "failures": [], "incomplete": []}
    monkeypatch.setattr(online, "run_case", passed_probe)
    report = online.diagnose(["http://a", "http://b"], tmp_path / "report.json", client=DifferentSourceClient())
    assert calls == [("http://a", "python:3"), ("http://b", "python:3")]
    assert report["services"][0]["status"] == "passed"
    assert report["services"][1]["status"] == "incomplete"
    assert not report["regression_pass"]


def test_search_diagnosis_reports_missing_backend_for_all_searches(tmp_path, monkeypatch):
    from recipe.swe_agent import search_utils as search
    original = search._stream_command
    def missing_rg(command, **kwargs):
        if command[0] == (os.environ.get("SWE_AGENT_RG_BINARY") or "rg"):
            return search._StreamResult([], 127, backend_missing=True)
        return original(command, **kwargs)
    monkeypatch.setattr(search, "_stream_command", missing_rg)
    monkeypatch.setenv("SWE_AGENT_PYTHON_ENV_CACHE_ENABLED", "0")
    session = service.ExecutionSession(key="local:diagnosis", task=online.cached_task("", "diagnosis", 30),
                                       request_id="local", workspace=tmp_path, use_docker=False)
    failed_indices = []
    for index, item in enumerate(online.search_backend_probe_case()["steps"]):
        if index == 0:
            continue  # Container identity is exercised by online preflight.
        text, _, metadata = session.run_tool(item["name"], item["arguments"])
        if online.check_response(item["expected"], {"ok": True, "text": text, "metadata": metadata}):
            failed_indices.append(index)
            assert text.startswith("Search unavailable: required ripgrep")
            assert not online.check_response(item["missing_rg_expected"], {"ok": True, "text": text})
    assert failed_indices == [2, 4, 5, 6, 7]


def test_container_environment_probe_executes_and_returns_scoped_package_evidence():
    import shlex
    import subprocess
    import sys
    arguments = online.search_backend_probe_case()["steps"][0]["arguments"]
    command = shlex.split(arguments["command"])
    command[0] = sys.executable
    result = subprocess.run(command, capture_output=True, text=True, timeout=10, check=True)
    data = json.loads(result.stdout)
    assert data["scope"] == "task_container"
    assert data["architecture"]
    assert "rg" in data["executables"]
    assert "/usr/bin/rg" in data["standard_rg_paths"]
    assert data["ripgrep_package"] is None or isinstance(data["ripgrep_package"]["exit_code"], int)


def test_production_client_polls_ticket_without_resubmitting(monkeypatch):
    requests = []
    operation_ids = []
    class Response:
        def __init__(self, body):
            self.body = json.dumps(body).encode()
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def read(self):
            return self.body
    class Opener:
        def open(self, request, timeout):
            requests.append(request.full_url)
            if request.full_url.endswith("/execute"):
                operation_ids.append(json.loads(request.data)["operation_id"])
                return Response({"ok": True, "operation_pending": True, "operation_id": "ticket", "retry_after_seconds": .05})
            assert request.full_url.endswith("/operation_result")
            return Response({"ok": True, "operation_pending": False, "text": "done", "metadata": {"tool": "write_file"}})
    monkeypatch.setattr(tools, "_remote_execution_opener", lambda config: Opener())
    response = online.Client().post("http://benchmark.invalid:18080", "/execute", {
        "request_id": "online-unit", "task": online.cached_task("python:3", "unit", 30),
        "tool": "Write", "parameters": {"file_path": "a", "content": "x"},
    }, time.monotonic() + 5)
    assert response["text"] == "done"
    assert response["metadata"]["tool"] == "write_file"
    assert requests == ["http://benchmark.invalid:18080/execute", "http://benchmark.invalid:18080/operation_result"]
    online.Client().post("http://benchmark.invalid:18080", "/execute", {
        "request_id": "online-unit", "task": online.cached_task("python:3", "unit", 30),
        "tool": "Write", "parameters": {"file_path": "a", "content": "x"},
    }, time.monotonic() + 5)
    assert len(set(operation_ids)) == 2


def test_production_client_does_not_replay_ambiguous_mutation(monkeypatch):
    requests = []
    class Opener:
        def open(self, request, timeout):
            requests.append(request.full_url)
            raise OSError("connection lost after server may have started Write")
    monkeypatch.setattr(tools, "_remote_execution_opener", lambda config: Opener())
    with pytest.raises(RuntimeError, match="ambiguously"):
        online.Client().post("http://benchmark.invalid:18080", "/execute", {
            "request_id": "ambiguous-unit", "task": online.cached_task("python:3", "unit", 30),
            "tool": "Write", "parameters": {"file_path": "a", "content": "x"},
        }, time.monotonic() + 5)
    assert requests == ["http://benchmark.invalid:18080/execute"]


@pytest.mark.parametrize("case_id", ["nonsense_repeats", "failed_edit_no_progress", "legitimate_progress"])
def test_reward_over_actual_production_tool_responses(tmp_path, monkeypatch, case_id):
    monkeypatch.setenv("SWE_AGENT_PYTHON_ENV_CACHE_ENABLED", "0")
    case = next(case for case in online.scenarios() if case["id"] == case_id)
    task = online.cached_task("", "local-unit", 30)
    session = service.ExecutionSession(key="local:unit", task=task, request_id="local", workspace=tmp_path, use_docker=False)
    events = []
    for index, item in enumerate(case["steps"]):
        text, reward, metadata = session.run_tool(item["name"], item["arguments"])
        assert online.check_response(item["expected"], {"ok": True, "text": text, "metadata": metadata}) == []
        events.append({"index": index, "name": item["name"], "arguments": item["arguments"], "response": text, "metadata": metadata})
    result = online.reward_checks(events, case["expected_hits"])
    assert result["failures"] == []
    assert len(result["checks"]) == 5


def test_oracle_rejects_unsupported_and_wrong_results():
    assert online.check_response({"equals": "src/a.py"}, {"ok": True, "text": "src/a.py:2:needle"})
    assert online.check_response({"equals": "first\n"}, {"ok": True, "text": "first"})
    assert online.check_response({"equals": "first"}, {"ok": True, "text": " first"})
    assert online.check_response({"container": True}, {"ok": True, "text": "exit_code: 0"})
    with pytest.raises(ValueError, match="Unknown assertion"):
        online.check_response({"typo": True}, {"ok": True})


@pytest.fixture
def pinned_release(tmp_path, monkeypatch):
    import hashlib
    import io
    import sys
    import tarfile
    # A deterministic executable fixture exercises publication and subprocesses
    # without making unit tests depend on network access or a system rg package.
    binary = (f"#!{sys.executable}\n"
              "import sys\n"
              "a=[x for x in sys.argv[1:] if x != '--no-config']\n"
              "if a == ['--version']:\n"
              f"    print('ripgrep {rg_backend.VERSION} (rev af60c2de9d)\\n\\nfeatures:+pcre2'); sys.exit(0)\n"
              "if '--files' in a:\n"
              "    print('./src/Convenience.m\\n./src/Convenience.swift'); sys.exit(0)\n"
              "data=sys.stdin.read(); p=a[-1]\n"
              "if p in ('needle', 'caf\\u00e9'): print('1:Needle caf\\u00e9')\n"
              "elif '-F' not in a: print('2:Item(')\n"
              "else: sys.exit(1)\n").encode()
    source = tmp_path / rg_backend.ASSET
    with tarfile.open(source, "w:gz") as archive:
        member = tarfile.TarInfo(f"ripgrep-{rg_backend.VERSION}-{rg_backend.TARGET}/rg")
        member.size = len(binary)
        archive.addfile(member, io.BytesIO(binary))
    monkeypatch.setattr(rg_backend, "BINARY_SHA256", hashlib.sha256(binary).hexdigest())
    monkeypatch.setattr(rg_backend, "ARCHIVE_SHA256", hashlib.sha256(source.read_bytes()).hexdigest())
    monkeypatch.delenv("RIPGREP_CONFIG_PATH", raising=False)
    return tmp_path / "persistent tools", source


def test_persistent_rg_prepare_reuse_and_restart_do_not_download(pinned_release, monkeypatch):
    root, source = pinned_release
    monkeypatch.setattr(rg_backend, "_download", lambda *a: pytest.fail("offline preparation downloaded"))
    prepared = rg_backend.prepare(root, archive=source)
    assert prepared["ok"] and not prepared["reused"]
    directory = rg_backend.installation_dir(root)
    assert sorted(p.name for p in directory.iterdir()) == ["manifest.json", "rg"]
    assert json.loads((directory / "manifest.json").read_text())["binary"] == "rg"
    assert rg_backend.prepare(root)["reused"]
    monkeypatch.setenv("SWE_AGENT_TOOLS_ROOT", str(root))
    assert rg_backend.verify()["ok"]
    assert rg_backend.verify()["root"] == str(root)
    assert rg_backend.main(["activate"]) == 0


def test_persistent_rg_alias_mount_uses_relative_manifest(pinned_release, tmp_path):
    root, source = pinned_release
    rg_backend.prepare(root, archive=source)
    alias = tmp_path / "other cluster mount"
    alias.symlink_to(root, target_is_directory=True)
    verified = rg_backend.verify(alias)
    assert verified["ok"] and verified["binary"].startswith(str(alias))
    assert "root" not in json.loads((rg_backend.installation_dir(alias) / "manifest.json").read_text())


def test_persistent_rg_archive_cache_survives_reinstallation(pinned_release, monkeypatch):
    import shutil
    root, source = pinned_release
    calls = []
    def download(destination):
        calls.append(destination)
        destination.write_bytes(source.read_bytes())
    monkeypatch.setattr(rg_backend, "_download", download)
    first = rg_backend.prepare(root)
    cached = root / "downloads" / rg_backend.ASSET
    assert first["archive"] == str(cached) and cached.read_bytes() == source.read_bytes()
    assert rg_backend.prepare(root)["reused"]
    shutil.rmtree(rg_backend.installation_dir(root))
    assert not rg_backend.prepare(root)["reused"]
    assert len(calls) == 1


def test_persistent_rg_corrupt_cache_is_not_overwritten(pinned_release, monkeypatch):
    root, _ = pinned_release
    cached = root / "downloads" / rg_backend.ASSET
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"corrupt cache")
    monkeypatch.setattr(rg_backend, "_download", lambda *a: pytest.fail("corrupt cache silently replaced"))
    with pytest.raises(ValueError, match="cached rg archive checksum mismatch"):
        rg_backend.prepare(root)
    assert cached.read_bytes() == b"corrupt cache"
    assert not rg_backend.installation_dir(root).exists()


@pytest.mark.parametrize("short_mounted", [False, True])
def test_persistent_rg_known_storage_prefixes_reuse_same_installation(pinned_release, tmp_path, monkeypatch, short_mounted):
    _, source = pinned_release
    short, long = tmp_path / "short", tmp_path / "long"
    visible, absent = (short, long) if short_mounted else (long, short)
    visible.mkdir()
    monkeypatch.setattr(rg_backend, "SHORT_STORAGE", short)
    monkeypatch.setattr(rg_backend, "LONG_STORAGE", long)
    configured = absent / ".swe-tools"
    cached_source = visible / "downloads" / source.name
    cached_source.parent.mkdir()
    cached_source.write_bytes(source.read_bytes())
    prepared = rg_backend.prepare(configured, archive=absent / "downloads" / source.name)
    assert prepared["configured_root"] == str(configured)
    assert prepared["root"] == str(visible / ".swe-tools")
    assert prepared["storage_alias_used"] and not absent.exists()
    assert rg_backend.prepare(visible / ".swe-tools")["reused"]
    assert rg_backend.prepare(configured)["reused"]
    monkeypatch.setenv("SWE_AGENT_TOOLS_ROOT", str(configured))
    for name in ("SWE_AGENT_RG_BINARY", "SWE_AGENT_RG_DISABLED", "SWE_AGENT_RG_STARTUP_STATUS"):
        monkeypatch.setenv(name, "")
    rg_backend.activate()
    # A second path to the same executable is valid, including bind mount aliases.
    alias = tmp_path / "mounted alias"
    alias.symlink_to(visible, target_is_directory=True)
    monkeypatch.setenv("SWE_AGENT_RG_BINARY", str(rg_backend.installation_dir(alias / ".swe-tools") / "rg"))
    assert rg_backend.readiness()["ok"]
    assert rg_backend.readiness()["configured_root"] == str(configured)


def test_persistent_rg_new_version_preserves_old_installation(pinned_release):
    root, source = pinned_release
    old = root / "ripgrep" / "previous-version" / rg_backend.TARGET
    old.mkdir(parents=True)
    (old / "rg").write_bytes(b"previous executable")
    rg_backend.prepare(root, archive=source)
    assert (old / "rg").read_bytes() == b"previous executable"


def test_persistent_rg_concurrent_alias_preparation_downloads_once(pinned_release, tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    root, source = pinned_release
    alias = tmp_path / "second cluster"
    alias.symlink_to(root, target_is_directory=True)
    calls = []
    def download(destination):
        calls.append(True)
        destination.write_bytes(source.read_bytes())
    monkeypatch.setattr(rg_backend, "_download", download)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(rg_backend.prepare, [root, alias, root, alias]))
    assert len(calls) == 1
    assert all(result["ok"] for result in results)
    assert sum(not result["reused"] for result in results) == 1


@pytest.mark.parametrize("version", ["ripgrep 15.1.00", "ripgrep 15.1.0-dev", "other 15.1.0", ""])
def test_persistent_rg_version_accepts_official_revision_but_rejects_other_releases(pinned_release, monkeypatch, version):
    root, source = pinned_release
    rg_backend.prepare(root, archive=source)
    original = rg_backend.subprocess.run
    def wrong_version(command, **kwargs):
        if command[1:] == ["--no-config", "--version"]:
            return SimpleNamespace(stdout=version, stderr="", returncode=0)
        return original(command, **kwargs)
    monkeypatch.setattr(rg_backend.subprocess, "run", wrong_version)
    assert not rg_backend.verify(root)["ok"]


@pytest.mark.parametrize("fault", ["archive_hash", "binary_hash", "manifest", "permissions", "execution"])
def test_persistent_rg_rejects_corruption_without_executing_or_overwriting(pinned_release, monkeypatch, fault):
    root, source = pinned_release
    if fault == "archive_hash":
        source.write_bytes(b"bad archive")
        with pytest.raises(ValueError, match="archive checksum"):
            rg_backend.prepare(root, archive=source)
        assert not rg_backend.installation_dir(root).exists()
        return
    rg_backend.prepare(root, archive=source)
    directory = rg_backend.installation_dir(root)
    if fault == "binary_hash":
        (directory / "rg").write_text("changed executable")
        monkeypatch.setattr(rg_backend, "_probe", lambda *a: pytest.fail("corrupt binary executed"))
    elif fault == "manifest":
        (directory / "manifest.json").write_text('{"schema_version": 1}')
        monkeypatch.setattr(rg_backend, "_probe", lambda *a: pytest.fail("unverified release executed"))
    elif fault == "permissions":
        (directory / "rg").chmod(0o644)
    else:
        def execution_error(*args):
            raise PermissionError("noexec mount")
        monkeypatch.setattr(rg_backend, "_probe", execution_error)
    assert not rg_backend.verify(root)["ok"]
    with pytest.raises((ValueError, PermissionError)):
        rg_backend.prepare(root, archive=source)
    assert directory.exists()


def test_persistent_rg_download_failure_and_stage_cleanup(pinned_release, monkeypatch):
    root, _ = pinned_release
    def failure(destination):
        destination.write_bytes(b"partial archive")
        raise TimeoutError("download timed out")
    monkeypatch.setattr(rg_backend, "_download", failure)
    with pytest.raises(TimeoutError):
        rg_backend.prepare(root)
    target = rg_backend.installation_dir(root)
    assert not target.exists()
    assert not list(target.parent.glob("*.stage-*"))
    assert not (root / "downloads" / rg_backend.ASSET).exists()
    assert not list((root / "downloads").glob(".rg-download-*"))


def test_persistent_rg_concurrent_preparation_downloads_once(pinned_release, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    root, source = pinned_release
    calls = []
    def download(destination):
        calls.append(True)
        destination.write_bytes(source.read_bytes())
    monkeypatch.setattr(rg_backend, "_download", download)
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(lambda _: rg_backend.prepare(root), range(3)))
    assert len(calls) == 1
    assert sum(not result["reused"] for result in results) == 1
    assert all(result["ok"] for result in results)


def test_persistent_rg_missing_mount_platform_and_no_network_verification(tmp_path, monkeypatch):
    root = tmp_path / "missing mount" / ".swe-tools"
    monkeypatch.setattr(rg_backend, "_download", lambda *a: pytest.fail("verification attempted a download"))
    assert not rg_backend.verify(root)["ok"]
    with pytest.raises(ValueError, match="storage parent"):
        rg_backend.prepare(root)
    assert not root.parent.exists()
    monkeypatch.setenv("SWE_AGENT_TOOLS_ROOT", "relative/tools")
    assert "absolute" in rg_backend.verify()["error"]
    monkeypatch.setattr(rg_backend.platform, "machine", lambda: "aarch64")
    assert "supports Linux x86_64" in rg_backend.verify(tmp_path)["error"]


def test_persistent_rg_lock_timeout_never_breaks_another_preparation(tmp_path):
    with rg_backend._prepare_lock(tmp_path / "lock"):
        with pytest.raises(TimeoutError, match="preparation lock"):
            with rg_backend._prepare_lock(tmp_path / "lock", timeout=0):
                pytest.fail("conflicting writer acquired the lock")


def test_persistent_rg_extracts_only_regular_expected_binary(pinned_release, monkeypatch):
    import hashlib
    import tarfile
    root, source = pinned_release
    with tarfile.open(source, "w:gz") as archive:
        member = tarfile.TarInfo(f"ripgrep-{rg_backend.VERSION}-{rg_backend.TARGET}/rg")
        member.type = tarfile.SYMTYPE
        member.linkname = "/bin/sh"
        archive.addfile(member)
    monkeypatch.setattr(rg_backend, "ARCHIVE_SHA256", hashlib.sha256(source.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match="archive member"):
        rg_backend.prepare(root, archive=source)
    assert not rg_backend.installation_dir(root).exists()


def test_persistent_search_gate_checks_selected_binary_and_hash():
    data = DiagnosisClient().get("http://test", "/tool_contract_info")["host_diagnostics"]
    assert online.persistent_search_verified(data)
    data["rg"]["path"] = "/unverified/rg"
    assert not online.persistent_search_verified(data)
    data["rg"]["path"] = data["persistent_rg"]["binary"]
    data["persistent_rg"]["binary_sha256"] = "wrong"
    assert not online.persistent_search_verified(data)


def test_production_search_selects_verified_binary_and_degraded_mode_skips_it(tmp_path, monkeypatch):
    from recipe.swe_agent import search_utils as search
    (tmp_path / "a.txt").write_text("Needle caf\u00e9\nItem(\n")
    calls = []
    def stream(command, **kwargs):
        calls.append(command)
        assert command[0] == "/persistent cache/rg"
        return search._StreamResult(["a.txt:1:Needle caf\u00e9"], 0)
    monkeypatch.setenv("SWE_AGENT_RG_BINARY", "/persistent cache/rg")
    monkeypatch.delenv("SWE_AGENT_RG_DISABLED", raising=False)
    monkeypatch.setattr(search, "_stream_command", stream)
    assert search.search_text(tmp_path, {"pattern": "needle"}) == "a.txt:1:Needle caf\u00e9"
    assert len(calls) == 1
    monkeypatch.setenv("SWE_AGENT_RG_DISABLED", "1")
    assert "Search unavailable: required ripgrep" in search.search_text(tmp_path, {"pattern": "needle"})
    assert len(calls) == 1


def test_unicode_success_case_is_required_online_and_degraded_oracles_are_offline_only(tmp_path, monkeypatch):
    case = next(case for case in online.scenarios() if case["id"] == "unicode_search_capability")
    assert sum("missing_rg_expected" in item for item in case["steps"]) == 7
    monkeypatch.setattr(online, "scenarios", lambda: [case])
    online.build(tmp_path / "frozen")
    report = online.offline(tmp_path / "frozen", tmp_path / "report.json")
    assert report["regression_pass"]
    assert report["missing_rg_capability_complete"] is False
    assert sum(event.get("expected_degraded_response", False) for result in report["cases"] for event in result["events"]) == 7
    for item in case["steps"]:
        if "missing_rg_expected" in item:
            assert online.check_response(item["expected"], {"ok": True, "text": item["missing_rg_expected"]["contains"]})


def test_host_diagnostics_remains_available_with_invalid_tools_root(monkeypatch):
    monkeypatch.setenv("SWE_AGENT_TOOLS_ROOT", "relative/tools")
    monkeypatch.setenv("SWE_AGENT_RG_DISABLED", "1")
    data = info._collect_host_diagnostics()
    assert data["complete"]
    assert data["configured_tools_root"] == "relative/tools"
    assert not data["persistent_rg"]["ok"]
    assert "absolute" in data["persistent_rg"]["error"]
    assert data["rg"]["status"] == "disabled"


def test_ssh_startup_cli_and_remote_shell_syntax():
    import subprocess
    import note_start_service as startup
    args = startup.parse_args(["--tools-root", "/persistent tools/cache", "--prepare-rg",
                               "--rg-archive", "/persistent tools/release.tar.gz"])
    assert args.prepare_rg and args.tools_root == "/persistent tools/cache"
    assert not startup.parse_args([]).prepare_rg
    assert startup.parse_args([]).tools_root == startup.DEFAULT_TOOLS_ROOT
    for argv in (["--tools-root", "relative"], ["--rg-archive", "/release.tar.gz"]):
        with pytest.raises(SystemExit):
            startup.parse_args(argv)
    subprocess.run(["bash", "-n"], input=startup.REMOTE_RUNNER, text=True, check=True, timeout=5)


@pytest.mark.parametrize("fault", [None, "source", "functional", "startup", "selected", "cache", "ready", "offline"])
def test_ssh_reuse_requires_matching_source_and_verified_selected_search(tmp_path, monkeypatch, fault):
    import note_start_service as startup
    binary = tmp_path / "rg"
    binary.write_bytes(b"fixture")
    verified = {"ok": True, "binary": str(binary)}
    host = {"complete": True, "persistent_rg": dict(verified),
            "rg": {"functional": True, "path": str(binary)}, "rg_startup_status": "persistent_verified"}
    data = {"/health": {"ok": True}, "/ready": {"ok": True, "docker_available": True},
            "/tool_contract_info": {"ok": True, "source_fingerprint": {"sha256": "current"}, "host_diagnostics": host}}
    if fault == "source":
        data["/tool_contract_info"]["source_fingerprint"]["sha256"] = "old"
    elif fault == "functional":
        host["rg"]["functional"] = False
    elif fault == "startup":
        host["rg_startup_status"] = "degraded"
    elif fault == "selected":
        host["rg"]["path"] = str(tmp_path / "unverified")
    elif fault == "cache":
        verified["ok"] = False
    elif fault == "ready":
        data["/ready"]["docker_available"] = False
    class Response:
        def __init__(self, body):
            self.body = json.dumps(body).encode()
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def read(self, size):
            return self.body[:size]
    def get(request, timeout):
        assert timeout == 8
        if fault == "offline":
            raise OSError("service unavailable")
        return Response(data[urllib.parse.urlparse(request.full_url).path])
    monkeypatch.setattr(urllib.request, "urlopen", get)
    monkeypatch.setattr(rg_backend, "verify", lambda: verified)
    monkeypatch.setattr(info, "source_fingerprint", lambda: {"sha256": "current"})
    code = startup.REMOTE_RUNNER.split("python3 - <<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    with pytest.raises(SystemExit) as result:
        exec(compile(code, "ssh-reuse-check", "exec"), {})
    assert result.value.code == (0 if fault is None else 1)


@pytest.mark.parametrize("prepare", [False, True])
def test_ssh_argument_quoting_preserves_remote_paths_and_empty_arguments(monkeypatch, prepare):
    import asyncio
    import shlex
    import note_start_service as startup
    commands = []
    class Process:
        def __init__(self):
            self.stdin = self
            self.stdout = asyncio.StreamReader()
            self.stdout.feed_data(b"__SWE_SERVICE_URL__=http://example:18080\n")
            self.stdout.feed_eof()
            self.returncode = 0
        def write(self, value):
            assert value.decode() == startup.REMOTE_RUNNER
        async def drain(self):
            pass
        def close(self):
            pass
        async def wait(self):
            return 0
    async def connect(*args):
        return None
    async def launch(*args, **kwargs):
        commands.append(args)
        return Process()
    monkeypatch.setattr(startup, "warm_up_ssh", connect)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", launch)
    root = "/persistent tools/$(must-not-execute)/cache" if prepare else ""
    archive = "/persistent tools/it's a release.tar.gz" if prepare else ""
    script = "/cluster other/swe/stock-rl-reflect/note_20.sh"
    result = asyncio.run(startup.run_host("host_2", script, 90, 1, 10, False, True, root, prepare, archive))
    assert result.url == "http://example:18080"
    # OpenSSH concatenates the remote argv into a command parsed by a shell.
    boundary = commands[0].index("--") + 1
    assert shlex.split(" ".join(commands[0][boundary:])) == [
        "32_cpus_2", script, "90", "1", "0", "1", root, "1" if prepare else "0", archive]


@pytest.mark.parametrize("prepare", [False, True])
@pytest.mark.parametrize("reuse_valid", [False, True])
@pytest.mark.parametrize("fault", [None, "prepare", "verify"])
def test_remote_startup_preparation_is_explicit_and_fixed_ranges_are_preserved(tmp_path, monkeypatch, prepare, reuse_valid, fault):
    import os
    import subprocess
    import note_start_service as startup
    workspace = tmp_path / "remote code"
    workspace.mkdir()
    script = workspace / "note_20.sh"
    lines = ["\n"] * 150
    lines[0] = "echo unexpected-install\n"
    lines[4] = "SERVICE_ID=original\n"
    lines[5] = 'echo "fixture-start:$SERVICE_ID"\n'
    lines[6] = 'echo "export SWE_AGENT_EXECUTION_URL=http://fresh:18080" > "swe_benchmark_execution_service_${SERVICE_ID}.log"\n'
    lines[118] = "echo fixture-cleanup\n"
    script.write_text("".join(lines))
    original = script.read_bytes()
    bin_dir = tmp_path / "commands"
    bin_dir.mkdir()
    def executable(name, content):
        path = bin_dir / name
        path.write_text("#!/bin/bash\nset -eu\n" + content)
        path.chmod(0o755)
    executable("docker", "exit 0\n")
    executable("containerd", "exit 0\n")
    executable("pgrep", "echo '123 recipe.swe_agent.remote_execution_service --advertise-host reused'\n")
    executable("python3", '''
if [[ "$1" == "-m" ]]; then
    [[ "$2" == "recipe.swe_agent.rg_backend" ]]
    [[ "$SWE_AGENT_TOOLS_ROOT" == "$FIXTURE_ROOT" ]]
    if [[ "$3" == "prepare" ]]; then
        [[ "$4" == "--archive" && "$5" == "$FIXTURE_ARCHIVE" ]]
        echo fixture-explicit-preparation
        exit "$FIXTURE_PREPARE_EXIT"
    fi
    [[ "$3" == "verify" ]]
    echo fixture-verification
    exit "$FIXTURE_VERIFY_EXIT"
fi
[[ "$1" == "-" ]]
cat >/dev/null
exit "$FIXTURE_REUSE_EXIT"
''')
    root = tmp_path / "persistent tools"
    archive = root / "release.tar.gz"
    env = {**os.environ, "PATH": str(bin_dir) + ":" + os.environ["PATH"],
           "FIXTURE_ROOT": str(root), "FIXTURE_ARCHIVE": str(archive),
           "FIXTURE_PREPARE_EXIT": "1" if fault == "prepare" else "0",
           "FIXTURE_VERIFY_EXIT": "1" if fault == "verify" else "0",
           "FIXTURE_REUSE_EXIT": "0" if reuse_valid else "1"}
    result = subprocess.run(["bash", "-s", "--", "unit_2", str(script), "5", "1", "0", "1",
                             str(root), "1" if prepare else "0", str(archive) if prepare else ""],
                            input=startup.REMOTE_RUNNER, env=env, capture_output=True, text=True, timeout=10)
    blocked = fault == "verify" or (prepare and fault == "prepare")
    assert result.returncode == (1 if blocked else 0), result.stderr
    assert script.read_bytes() == original
    assert ("fixture-explicit-preparation" in result.stdout) == prepare
    assert ("fixture-verification" in result.stdout) == (not (prepare and fault == "prepare"))
    assert ("fixture-cleanup" in result.stdout) == (not reuse_valid and not blocked)
    assert ("fixture-start:unit_2" in result.stdout) == (not reuse_valid and not blocked)
    if blocked:
        assert "__SWE_SERVICE_ERROR__=required ripgrep" in result.stdout
        assert "__SWE_SERVICE_URL__=" not in result.stdout
        assert "unexpected-install" not in result.stdout
        return
    expected = "reused" if reuse_valid else "fresh"
    assert f"__SWE_SERVICE_URL__=http://{expected}:18080" in result.stdout
    assert "unexpected-install" not in result.stdout

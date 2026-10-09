from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from scripts import swe_tool_online_benchmark as online
from recipe.swe_agent import remote_execution_service as service
from recipe.swe_agent import tool_contract_info as info
from recipe.swe_agent import tools
from recipe.swe_agent import rg_backend


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


def test_search_diagnosis_reproduces_unicode_fallback_gap_without_changing_pattern(tmp_path, monkeypatch):
    from recipe.swe_agent import search_utils as search
    original = search._stream_command
    def missing_rg(command, **kwargs):
        if command[0] == "rg":
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
            assert "requires rg" in text
            assert "results incomplete" in text
    assert failed_indices == [4, 6]


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
              "a=sys.argv[1:]\n"
              "if a == ['--version']:\n"
              f"    print('ripgrep {rg_backend.VERSION} (rev af60c2de9d)\\n\\nfeatures:+pcre2'); sys.exit(0)\n"
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


@pytest.mark.parametrize("version", ["ripgrep 15.1.00", "ripgrep 15.1.0-dev", "other 15.1.0", ""])
def test_persistent_rg_version_accepts_official_revision_but_rejects_other_releases(pinned_release, monkeypatch, version):
    root, source = pinned_release
    rg_backend.prepare(root, archive=source)
    original = rg_backend.subprocess.run
    def wrong_version(command, **kwargs):
        if command[1:] == ["--version"]:
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
    monkeypatch.setattr(search, "_fallback", lambda *args: search._StreamResult([], 2, error="requires rg; results incomplete"))
    assert "requires rg" in search.search_text(tmp_path, {"pattern": "needle"})
    assert len(calls) == 1


def test_unicode_success_case_is_required_online_and_degraded_oracles_are_offline_only(tmp_path, monkeypatch):
    case = next(case for case in online.scenarios() if case["id"] == "unicode_search_capability")
    assert sum("missing_rg_expected" in item for item in case["steps"]) == 5
    monkeypatch.setattr(online, "scenarios", lambda: [case])
    online.build(tmp_path / "frozen")
    report = online.offline(tmp_path / "frozen", tmp_path / "report.json")
    assert report["regression_pass"]
    assert report["missing_rg_capability_complete"] is False
    assert sum(event.get("expected_degraded_response", False) for result in report["cases"] for event in result["events"]) == 5
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


@pytest.mark.parametrize("prepare,reuse_valid", [(False, True), (False, False), (True, True), (True, False)])
def test_remote_startup_preparation_is_explicit_and_fixed_ranges_are_preserved(tmp_path, monkeypatch, prepare, reuse_valid):
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
    [[ "$2" == "recipe.swe_agent.rg_backend" && "$3" == "prepare" ]]
    [[ "$SWE_AGENT_TOOLS_ROOT" == "$FIXTURE_ROOT" ]]
    [[ "$4" == "--archive" && "$5" == "$FIXTURE_ARCHIVE" ]]
    echo fixture-explicit-preparation-failed
    exit 1
fi
[[ "$1" == "-" ]]
cat >/dev/null
exit "$FIXTURE_REUSE_EXIT"
''')
    root = tmp_path / "persistent tools"
    archive = root / "release.tar.gz"
    env = {**os.environ, "PATH": str(bin_dir) + ":" + os.environ["PATH"],
           "FIXTURE_ROOT": str(root), "FIXTURE_ARCHIVE": str(archive),
           "FIXTURE_REUSE_EXIT": "0" if reuse_valid else "1"}
    result = subprocess.run(["bash", "-s", "--", "unit_2", str(script), "5", "1", "0", "1",
                             str(root), "1" if prepare else "0", str(archive) if prepare else ""],
                            input=startup.REMOTE_RUNNER, env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert script.read_bytes() == original
    assert ("fixture-explicit-preparation-failed" in result.stdout) == prepare
    assert ("fixture-cleanup" in result.stdout) == (not reuse_valid)
    assert ("fixture-start:unit_2" in result.stdout) == (not reuse_valid)
    expected = "reused" if reuse_valid else "fresh"
    assert f"__SWE_SERVICE_URL__=http://{expected}:18080" in result.stdout
    assert "unexpected-install" not in result.stdout

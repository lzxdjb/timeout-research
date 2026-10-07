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


def test_offline_cli_runs_production_dispatch_and_fails_on_bad_oracle(tmp_path, monkeypatch):
    cases = [{"id": "offline-boundary", "expected_hits": None, "steps": [
        online.step("Write", {"file_path": "a.py", "content": "first\nsecond\n"}, contains="Wrote"),
        online.step("Read", {"file_path": "/testbed/a.py", "offset": 1, "limit": 1}, equals="second"),
    ]}]
    monkeypatch.setattr(online, "scenarios", lambda: cases)
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

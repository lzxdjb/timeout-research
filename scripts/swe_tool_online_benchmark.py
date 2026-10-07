"""Frozen, bounded online SWE tool contracts. No model inference or training."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import sys
import time
import tempfile
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "stock-rl-reflect"))
VERSION = "swe-tool-online-v3"


def step(name: str, arguments: dict, **expected: Any) -> dict:
    return {"name": name, "arguments": arguments, "expected": expected}


def scenarios() -> list[dict]:
    """Oracles describe desired behavior independently of implementation output."""
    write = step("Write", {"file_path": "src/a.py", "content": "VALUE = 1\nneedle\nthird\n"}, contains="Wrote")
    read = step("Read", {"file_path": "src/a.py"}, equals="VALUE = 1\nneedle\nthird\n")
    edit = step("Edit", {"file_path": "src/a.py", "old_string": "VALUE = 1", "new_string": "VALUE = 2"}, contains="Edited")
    test_command = "python -m unittest discover -s tests -v"
    test_file = "import unittest\nclass ContractTest(unittest.TestCase):\n    def test_value(self):\n        self.assertIn('VALUE = 2', open('src/a.py').read())\n"
    cases = [
        ("coherence", [write, read, step("Bash", {"command": "cat src/a.py"}, contains="VALUE = 1", container=True), step("Bash", {"command": "printf 'from_shell\\n' > shell.txt"}, contains="exit_code: 0"), step("read_file", {"path": "shell.txt"}, equals="from_shell\n")]),
        ("mount_and_confinement", [write, step("Read", {"file_path": "/testbed/src/a.py"}, equals=read["expected"]["equals"]), step("Read", {"file_path": "../outside"}, metadata={"tool_path_rejected": True}), step("Read", {"file_path": "/etc/passwd"}, metadata={"tool_path_rejected": True})]),
        ("read_ranges", [write, step("Read", {"file_path": "src/a.py", "offset": 0, "limit": 1}, equals="VALUE = 1"), step("Read", {"file_path": "src/a.py", "offset": 1, "limit": 1}, equals="needle"), step("Read", {"file_path": "src/a.py", "limit": 1}, equals="VALUE = 1"), step("read_file", {"path": "src/a.py", "start_line": 2, "end_line": 2}, equals="needle"), step("Read", {"file_path": "src/a.py", "max_chars": 5}, max_chars=5), step("Read", {"file_path": "absent"}, contains="File not found"), step("Read", {"file_path": "src/a.py", "offset": -1}, prefix="Read error:"), step("Read", {"file_path": "src/a.py", "limit": 0}, prefix="Read error:")]),
        ("edit_guards", [write, edit | {"expected": {"contains": "read the file first"}}, read, edit, step("Read", {"file_path": "src/a.py"}, contains="VALUE = 2"), step("Edit", {"file_path": "src/a.py", "old_string": "absent", "new_string": "bad"}, contains="not found"), step("Edit", {"file_path": "src/a.py", "old_string": "VALUE = 2", "new_string": "VALUE = 2"}, contains="no changes"), step("Read", {"file_path": "src/a.py"}, contains="VALUE = 2", excludes="bad")]),
        ("ambiguous_edit", [step("Write", {"file_path": "a.txt", "content": "same\nsame\n"}, contains="Wrote"), step("Read", {"file_path": "a.txt"}, equals="same\nsame\n"), step("Edit", {"file_path": "a.txt", "old_string": "same", "new_string": "new"}, contains="matched 2"), step("Read", {"file_path": "a.txt"}, equals="same\nsame\n"), step("Edit", {"file_path": "a.txt", "old_string": "same", "new_string": "new", "replace_all": True}, contains="Edited"), step("Read", {"file_path": "a.txt"}, equals="new\nnew\n")]),
        ("write_and_append", [write, step("write_file", {"path": "src/a.py", "content": "bad"}, contains="Refused"), read, step("write_file", {"path": "src/a.py", "content": "tail\n", "append": True}, contains="Wrote"), step("Write", {"file_path": "src/a.py", "content": "replacement\n"}, contains="Wrote"), step("Read", {"file_path": "src/a.py"}, equals="replacement\n")]),
        ("search_freshness", [write, step("Grep", {"pattern": "VALUE = 1", "path": "src"}, contains="VALUE = 1"), read, edit, step("Grep", {"pattern": "VALUE = 1", "path": "src"}, excludes="VALUE = 1"), step("search_text", {"pattern": "VALUE = 2", "path": "src"}, contains="VALUE = 2"), step("Glob", {"pattern": "**/*.py"}, contains="src/a.py"), step("search_files", {"query": "a.py", "path": "src"}, contains="src/a.py")]),
        ("grep_modes", [write, step("Grep", {"pattern": "needle", "path": "src", "output_mode": "files_with_matches"}, equals="src/a.py"), step("Grep", {"pattern": "needle", "path": "src", "output_mode": "count"}, equals="src/a.py:1"), step("Grep", {"pattern": "needle", "path": "src", "output_mode": "content"}, contains="needle"), step("Grep", {"pattern": "[", "path": "src", "literal": True}, excludes="regex parse error")]),
        ("grep_filters_and_bounds", [write, step("Write", {"file_path": "src/b.txt", "content": "needle\nneedle\n-dash\n"}, contains="Wrote"), step("Write", {"file_path": "node_modules/hidden.py", "content": "needle\n"}, contains="Wrote"), step("Grep", {"pattern": "needle", "glob": "*.py", "output_mode": "count"}, equals="src/a.py:1"), step("Grep", {"pattern": "needle", "glob": "*.txt", "output_mode": "count"}, equals="src/b.txt:2"), step("Grep", {"pattern": "-dash", "literal": True}, contains="src/b.txt:3:-dash"), step("Grep", {"pattern": "needle", "path": "src", "max_results": 1}, contains="incomplete"), step("Grep", {"pattern": "["}, prefix="Search error:")]),
        ("patch_atomicity", [write, step("apply_patch", {"patch": "--- a/src/a.py\n+++ b/src/a.py\n@@ -1,3 +1,3 @@\n-VALUE = 1\n+VALUE = 2\n needle\n third\n"}, contains="successfully"), step("Read", {"file_path": "src/a.py"}, contains="VALUE = 2"), step("apply_patch", {"patch": "--- a/src/a.py\n+++ b/src/a.py\n@@ -1 +1 @@\n-absent\n+bad\n"}, contains="failed"), step("Read", {"file_path": "src/a.py"}, contains="VALUE = 2", excludes="bad"), step("apply_patch", {"patch": "garbage"}, contains="Malformed")]),
        ("legitimate_progress", [write, read, edit, step("Read", {"file_path": "src/a.py"}, contains="VALUE = 2"), step("Write", {"file_path": "tests/test_contract.py", "content": test_file}, contains="Wrote"), step("run_tests", {"command": test_command}, contains="SWE_PUBLIC_TEST_STATUS: PASS"), step("Edit", {"file_path": "src/a.py", "old_string": "VALUE = 2", "new_string": "VALUE = 3"}, contains="Edited"), step("run_tests", {"command": test_command}, contains="SWE_PUBLIC_TEST_STATUS: FAIL")]),
        ("nonsense_repeats", [write, read, read, read]),
        ("failed_edit_no_progress", [write, read, step("Edit", {"file_path": "src/a.py", "old_string": "absent", "new_string": "bad"}, contains="not found"), read, read]),
        ("shell_failures_and_limits", [step("Bash", {"command": "printf 'expected_failure\\n'; exit 7"}, contains="exit_code: 7"), step("Bash", {"command": "python -c \"print('x'*24000)\""}, contains="exit_code: 0", max_chars=13000), step("Bash", {"command": "printf 'alive\\n'"}, contains="alive")]),
        ("command_timeout", [step("Bash", {"command": "sleep 20"}, timeout=True)]),
    ]
    return [{"id": name, "steps": steps, "expected_hits": (
        [False, False, True, True] if name == "nonsense_repeats" else
        [False, False, False, True, True] if name == "failed_edit_no_progress" else
        [False] * len(steps) if name == "legitimate_progress" else None
    )} for name, steps in cases]


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def build(directory: Path) -> dict:
    data = scenarios()
    manifest = {"version": VERSION, "scenarios_sha256": digest(data), "scenario_count": len(data),
                "read_offset_policy": "Read offset is zero-based as published; internal start_line is one-based",
                "policy_changes_from_v2": ["Correct benchmark offset oracle to the documented zero-based schema; production indexing is preserved", "Accept only explicitly declared task mount prefixes in addition to relative paths", "Add limit-only/invalid Read ranges and Grep glob, count, leading-dash and truncation boundaries"],
                "historical_origin": "tool_contract_v1 categories: search, path rejection, failed mutation, reread, shell, public verification",
                "unknown_policy": "missing deployment, cleanup or execution evidence is incomplete"}
    for name, value in (("scenarios.json", data), ("manifest.json", manifest)):
        target = directory / name
        if target.exists() and json.loads(target.read_text()) != value:
            raise ValueError("Frozen online benchmark differs; use a new version directory")
    write_json(directory / "scenarios.json", data)
    write_json(directory / "manifest.json", manifest)
    return manifest


def load(directory: Path) -> tuple[dict, list[dict]]:
    manifest = json.loads((directory / "manifest.json").read_text())
    cases = json.loads((directory / "scenarios.json").read_text())
    if manifest["version"] not in {VERSION, "swe-tool-online-v2"} or digest(cases) != manifest["scenarios_sha256"]:
        raise ValueError("Online manifest version/checksum mismatch")
    return manifest, cases


def offline(directory: Path, output: Path) -> dict:
    """Run production session dispatch in disposable host workspaces; no Docker calls."""
    from recipe.swe_agent import remote_execution_service as service
    from recipe.swe_agent import search_utils
    manifest, cases = load(directory)
    report = {"version": manifest["version"], "manifest_sha256": digest(manifest),
              "source_fingerprint": importlib.import_module("recipe.swe_agent.tool_contract_info").source_fingerprint(),
              "cases": [], "excluded_online_assertions": ["Docker container identity; image, deployment, transport and cleanup evidence require online execution"]}
    original_stream = search_utils._stream_command

    def no_rg(command, **kwargs):
        if command[0] == "rg":
            return search_utils._StreamResult([], None, error="forced missing rg", backend_missing=True)
        return original_stream(command, **kwargs)

    for backend in ("native", "missing_rg"):
        with mock.patch.object(search_utils, "_stream_command", no_rg if backend == "missing_rg" else original_stream):
            for case in cases:
                print(f"offline benchmark: {backend}/{case['id']}", file=sys.stderr, flush=True)
                result = {"id": case["id"], "backend": backend, "events": [], "failures": []}
                report["cases"].append(result)
                with tempfile.TemporaryDirectory(prefix="swe-tool-offline-") as raw_workspace:
                    workspace = Path(raw_workspace)
                    task = {"task_id": case["id"], "execution_phase": "rollout", "sandbox_backend": "local",
                            "docker_mount": "/testbed", "hard_timeout_seconds": 3 if case["id"] == "command_timeout" else 30,
                            "public_test_commands": []}
                    session = service.ExecutionSession(case["id"], task, case["id"], workspace, False)
                    # Only workspace/image provisioning is replaced. Dispatch, tools,
                    # subprocess execution, mutation metadata and rewards are production code.
                    with mock.patch.object(session, "prepare", return_value="fixture workspace ready"):
                        for index, item in enumerate(case["steps"]):
                            session.begin_operation("tool", task["hard_timeout_seconds"])
                            try:
                                text, reward, metadata = session.run_tool(item["name"], item["arguments"])
                            finally:
                                session.end_operation()
                            response = {"ok": True, "text": text, "metadata": metadata, "reward": reward}
                            event = {"index": index, "name": item["name"], "arguments": item["arguments"],
                                     "response": text, "metadata": metadata}
                            result["events"].append(event)
                            expected = {key: value for key, value in item["expected"].items() if key != "container"}
                            for error in check_response(expected, response):
                                result["failures"].append({"index": index, "error": error})
                    if case["expected_hits"] is not None:
                        result["reward"] = reward_checks(result["events"], case["expected_hits"])
                        result["failures"].extend(result["reward"]["failures"])
                result["status"] = "failed" if result["failures"] else "passed"
    report["counts"] = {"scenario_executions": len(report["cases"]),
                        "tool_dispatches": sum(len(case["events"]) for case in report["cases"]),
                        "failures": sum(len(case["failures"]) for case in report["cases"]),
                        "reward_checks": sum(len(case.get("reward", {}).get("checks", [])) for case in report["cases"])}
    report["regression_pass"] = bool(report["cases"]) and not report["counts"]["failures"]
    write_json(output, report)
    return report


def cached_task(image: str, task_id: str, timeout: float) -> dict:
    # base_image suppresses implicit registry pull; matching case_image rejects
    # a missing cache entry before any build. No repo or setup is requested.
    return {"task_id": task_id, "execution_phase": "rollout", "sandbox_backend": "local",
            "docker_image": image, "case_image": image, "docker_pull": False,
            "environment": {"base_image": image, "case_image": image},
            "docker_mount": "/testbed", "docker_workdir": "/testbed", "docker_entrypoint": "/bin/bash",
            "docker_network": "none", "docker_cpus": "1", "docker_memory": "512m",
            "hard_timeout_seconds": timeout, "setup_commands": [], "public_test_commands": []}


class Client:
    def get(self, url: str, path: str, timeout: float = 8) -> dict:
        headers = {}
        token = os.environ.get("SWE_AGENT_EXECUTION_TOKEN") or os.environ.get("SWE_AGENT_REMOTE_EXECUTION_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        try:
            with urllib.request.urlopen(urllib.request.Request(url + path, headers=headers), timeout=timeout) as response:
                return json.loads(response.read(512 * 1024))
        except Exception as exc:
            return {"ok": False, "probe_error": f"{type(exc).__name__}: {exc}"}

    def post(self, url: str, path: str, payload: dict, deadline: float) -> dict:
        tools = importlib.import_module("recipe.swe_agent.tools")
        if path == "/execute":
            # A new dispatch is distinct from a retry of an accepted ticket.
            payload = {"operation_id": uuid.uuid4().hex, **payload}
        return tools._remote_post_json(path, payload, {
            "execution_url_override": url, "_request_deadline": deadline,
        })


def check_response(expected: dict, response: dict) -> list[str]:
    text = response.get("text", "")
    metadata = response.get("metadata", {})
    errors = []
    if not response.get("ok"):
        errors.append("transport/operation not ok")
    for kind, value in expected.items():
        valid = True
        if kind == "equals":
            valid = text == value
        elif kind == "contains":
            valid = value in text
        elif kind == "excludes":
            valid = value not in text
        elif kind == "prefix":
            valid = text.startswith(value)
        elif kind == "max_chars":
            valid = len(text) <= value
        elif kind == "metadata":
            valid = all(metadata.get(key) == item for key, item in value.items())
        elif kind == "container":
            valid = bool(metadata.get("container_name"))
        elif kind == "timeout":
            valid = bool(metadata.get("tool_hard_timeout")) or "timeout" in text.lower() or "exit_code: 124" in text
        else:
            raise ValueError(f"Unknown assertion: {kind}")
        if not valid:
            errors.append(f"{kind}: expected {value!r}")
    return errors


def reward_checks(events: list[dict], expected_hits: list[bool]) -> dict:
    # The sibling repository has a regular `scripts` package; load by file so
    # it cannot shadow verl's scripts namespace when invoked as a CLI.
    spec = importlib.util.spec_from_file_location("_online_contract_replay", ROOT / "scripts/swe_tool_contract_benchmark.py")
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)
    from recipe.swe_agent.agent_loop import SWEToolAgentLoop
    env = {"SWE_AGENT_TRAINING_REPEATED_TOOL_REWARD_SHAPING": "1",
           "SWE_AGENT_TRAINING_REPEATED_TOOL_PENALTY": "0.1",
           "SWE_AGENT_TRAINING_REPEATED_TOOL_DETECTION_MODE": "mutation_aware",
           "SWE_AGENT_TRAINING_PROTOCOL_REWARD_SHAPING": "0",
           "SWE_AGENT_TRAINING_VERIFICATION_REWARD_SHAPING": "0",
           "SWE_AGENT_TRAINING_VERIFICATION_AUX_V1_ENABLED": "0",
           "SWE_AGENT_TRAINING_PARTIAL_HIDDEN_TEST_REWARD": "0"}
    failures, checks = [], []
    with mock.patch.dict(os.environ, env):
        replay = benchmark._replay(benchmark._load_detector(), events)
        actual_hits = [item["penalized"] for item in replay["events"]]
        if actual_hits != expected_hits:
            failures.append({"kind": "repeat", "expected": expected_hits, "actual": actual_hits})
        for validation, eligible, raw in ((False, True, 1.), (False, True, 0.), (True, True, 1.), (True, True, 0.), (False, False, 0.)):
            loop = object.__new__(SWEToolAgentLoop)
            loop._swe_is_validation = validation
            loop._swe_rollout_task = {"task_id": "online-contract"}
            loop._swe_agent_data = SimpleNamespace(assistant_turns=2, extra_fields={
                "submission_signal_seen": True, "submission_turn": 2,
                "trajectory_penalized_repeated_tool_calls": replay["penalized_repeat_count"],
            })
            score, info = loop._shape_training_reward(raw, reward_ok=eligible, protocol_eligible=eligible)
            expected = raw if validation else (0. if not eligible else -.1 if any(expected_hits) else raw)
            checks.append({"validation": validation, "eligible": eligible, "raw": raw, "expected": expected, "actual": score, "info": info})
            if score != expected:
                failures.append({"kind": "reward", "expected": expected, "actual": score})
    return {"replay": replay, "checks": checks, "failures": failures}


def run_case(client: Any, url: str, image: str, case: dict, deadline_seconds: float) -> dict:
    request_id = "tool-contract-" + uuid.uuid4().hex
    task = cached_task(image, request_id, 3 if case["id"] == "command_timeout" else 30)
    payload = {"request_id": request_id, "instance_id": request_id, "task": task}
    result = {"id": case["id"], "request_id": request_id, "events": [], "failures": [], "incomplete": []}
    deadline = time.monotonic() + deadline_seconds
    key = request_id + ":" + task["task_id"]
    try:
        result["claim"] = client.post(url, "/claim", payload, deadline)
        if not result["claim"].get("ok"):
            raise RuntimeError("claim failed: " + str(result["claim"].get("error", "unknown")))
        # Require actual Docker and Python before counting any scenario coverage.
        preflight = client.post(url, "/execute", {**payload, "tool": "Bash", "parameters": {"command": "python --version"}}, deadline)
        result["preflight"] = preflight
        if not preflight.get("ok") or "exit_code: 0" not in preflight.get("text", "") or not preflight.get("metadata", {}).get("container_name"):
            raise RuntimeError("cached Python-capable Docker image unavailable: " + preflight.get("text", str(preflight))[:800])
        for index, item in enumerate(case["steps"]):
            if time.monotonic() >= deadline:
                raise TimeoutError("scenario deadline reached")
            response = client.post(url, "/execute", {**payload, "tool": item["name"], "parameters": item["arguments"]}, deadline)
            event = {"index": index, "name": item["name"], "arguments": item["arguments"],
                     "response": response.get("text", ""), "metadata": response.get("metadata", {}), "raw": response}
            result["events"].append(event)
            for error in check_response(item["expected"], response):
                result["failures"].append({"index": index, "error": error})
            if not response.get("ok"):
                raise RuntimeError("execution incomplete; no mutation replay attempted")
        if case["expected_hits"] is not None:
            result["reward"] = reward_checks(result["events"], case["expected_hits"])
            result["failures"].extend(result["reward"]["failures"])
    except Exception as exc:
        result["incomplete"].append(f"{type(exc).__name__}: {exc}")
    finally:
        # Release even when claim timed out: the server may have created a session.
        try:
            release = client.post(url, "/release", {**payload, "force": True}, time.monotonic() + 30)
            result["release"] = release
            if not release.get("ok") or (result.get("claim", {}).get("ok") and not release.get("released")):
                result["incomplete"].append("release not confirmed")
            removed_on_timeout = any(event["metadata"].get("container_removed_after_timeout") for event in result["events"])
            if result.get("preflight", {}).get("metadata", {}).get("container_name") and not release.get("container_removed") and not removed_on_timeout:
                result["incomplete"].append("container removal not confirmed")
            query = {"key": key, "workspace_path": result.get("preflight", {}).get("metadata", {}).get("workspace", ""),
                     "cleanup_path": release.get("workspace_cleanup_path") or ""}
            cleanup_deadline = time.monotonic() + 10
            while True:
                state = client.get(url, "/tool_contract_info?" + urllib.parse.urlencode(query))
                evidence = state.get("session_state", "unknown")
                filesystem = state.get("cleanup_state", {})
                if (not isinstance(evidence, dict) or
                    (not any(evidence.values()) and filesystem.get("complete")) or
                    time.monotonic() >= cleanup_deadline):
                    break
                time.sleep(.2)
            result["cleanup_evidence"] = {"session": evidence, "filesystem": filesystem}
            if not isinstance(evidence, dict):
                result["incomplete"].append("session cleanup evidence unavailable")
            elif any(evidence.values()):
                result["incomplete"].append("session still present or cleanup running")
            if not filesystem.get("complete"):
                result["incomplete"].append("filesystem/container cleanup evidence unavailable or incomplete")
        except Exception as exc:
            result["incomplete"].append(f"cleanup: {type(exc).__name__}: {exc}")
    result["status"] = "failed" if result["failures"] else "incomplete" if result["incomplete"] else "passed"
    return result


def run(directory: Path, urls: list[str], output: Path, image: str = "", deadline_seconds: float = 120, client: Any = None) -> dict:
    manifest, cases = load(directory)
    client = client or Client()
    services = []
    report = {"version": manifest["version"], "manifest_sha256": digest(manifest), "services": services,
              "local_source_fingerprint": importlib.import_module("recipe.swe_agent.tool_contract_info").source_fingerprint(),
              "reward_execution": "local production detector and reward function over actual server responses; no hidden scorer or model inference"}
    for raw_url in dict.fromkeys(urls):
        url = raw_url.rstrip("/")
        print(f"online benchmark: {url}", file=sys.stderr, flush=True)
        service = {"url": url, "ready": client.get(url, "/ready"), "deployment": client.get(url, "/tool_contract_info"), "cases": [], "incomplete": []}
        services.append(service)
        deployment = service["deployment"]
        service["deployment_matches_local_source"] = (
            deployment.get("source_fingerprint", {}).get("sha256") == report["local_source_fingerprint"]["sha256"]
        )
        if not deployment.get("source_fingerprint", {}).get("sha256"):
            service["incomplete"].append("deployment source fingerprint unavailable")
        elif not service["deployment_matches_local_source"]:
            service["incomplete"].append("deployment differs from local production code; reported baseline cannot certify this checkout")
        images = deployment.get("cached_images", [])
        selected = image or next((item["tag"] for item in images if "sweb.eval" in item["tag"] or "python" in item["tag"]), "")
        service["image"] = selected
        image_ids = [item["id"] for item in images if item["tag"] == selected]
        service["image_ids"] = image_ids
        if not image_ids:
            service["incomplete"].append("selected image ID unavailable")
        if not service["ready"].get("ok") or not service["ready"].get("docker_available") or not selected:
            service["incomplete"].append("Docker service or cached image unavailable; all scenarios skipped")
        else:
            # Unknown fingerprints require full coverage on each service. Serial
            # execution keeps at most one benchmark session active globally.
            for case in cases:
                print(f"  {case['id']}", file=sys.stderr, flush=True)
                result = run_case(client, url, selected, case, deadline_seconds)
                service["cases"].append(result)
                write_json(output, report)
                if not result.get("preflight") or "cached Python-capable" in " ".join(result["incomplete"]):
                    service["incomplete"].append("remaining scenarios skipped after image/claim preflight failure")
                    break
                if any("release not confirmed" in error or "container removal not confirmed" in error or error.startswith("cleanup:") or "session still" in error for error in result["incomplete"]):
                    service["incomplete"].append("remaining scenarios skipped after cleanup failure")
                    break
        service["status"] = "failed" if any(item["failures"] for item in service["cases"]) else "incomplete" if service["incomplete"] or any(item["incomplete"] for item in service["cases"]) else "passed"
    report["regression_pass"] = bool(services) and all(item["status"] == "passed" for item in services)
    report["counts"] = {"services": len(services), "scenarios_expected": len(services) * len(cases),
                        "scenarios_executed": sum(len(item["cases"]) for item in services),
                        "assertion_failures": sum(len(case["failures"]) for item in services for case in item["cases"]),
                        "reward_checks": sum(len(case.get("reward", {}).get("checks", [])) for item in services for case in item["cases"]),
                        "reward_failures": sum(len(case.get("reward", {}).get("failures", [])) for item in services for case in item["cases"])}
    write_json(output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build_parser = sub.add_parser("build")
    build_parser.add_argument("--output-dir", type=Path, required=True)
    offline_parser = sub.add_parser("offline")
    offline_parser.add_argument("--benchmark-dir", type=Path, required=True)
    offline_parser.add_argument("--report", type=Path, required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--benchmark-dir", type=Path, required=True)
    run_parser.add_argument("--report", type=Path, required=True)
    run_parser.add_argument("--urls", default=os.environ.get("SWE_AGENT_EXECUTION_URLS", ""))
    run_parser.add_argument("--image", default="", help="existing cached Python/Bash image; never pulled or built")
    run_parser.add_argument("--scenario-deadline", type=float, default=120)
    args = parser.parse_args()
    if args.command == "build":
        print(json.dumps(build(args.output_dir), sort_keys=True))
        return 0
    if args.command == "offline":
        report = offline(args.benchmark_dir, args.report)
        print(json.dumps({"counts": report["counts"], "regression_pass": report["regression_pass"]}, sort_keys=True))
        return 0 if report["regression_pass"] else 1
    urls = [url.strip() for url in args.urls.split(",") if url.strip()]
    if not urls or args.scenario_deadline <= 0:
        parser.error("positive scenario deadline and at least one URL required")
    report = run(args.benchmark_dir, urls, args.report, args.image, args.scenario_deadline)
    print(json.dumps({"counts": report["counts"], "regression_pass": report["regression_pass"]}, sort_keys=True))
    return 0 if report["regression_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

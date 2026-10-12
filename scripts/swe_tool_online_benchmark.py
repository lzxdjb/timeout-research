"""Frozen, bounded online SWE tool contracts. No model inference or training."""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import collections
import fcntl
import hashlib
import importlib
import importlib.util
import json
import os
import re
import shlex
import sys
import time
import tempfile
import threading
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "stock-rl-reflect"))
sys.path.insert(0, str(ROOT))
VERSION = "swe-tool-online-v10"
AUDIT_VERSION = "swe-reference-audit-v1"


def step(name: str, arguments: dict, *, requires_search_backend: bool = True, **expected: Any) -> dict:
    item = {"name": name, "arguments": arguments, "expected": expected}
    if (name in {"Glob", "Grep", "search_files", "search_text"} and requires_search_backend
            and expected.get("prefix") != "Path not found:"):
        item["missing_rg_expected"] = {"prefix": "Search unavailable:", "contains": "required ripgrep"}
    return item


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
        ("unicode_search_capability", [
            step("Write", {"file_path": "probe/unicode.txt", "content": "Needle caf\u00e9\nItem(\n"}, contains="Wrote"),
            step("Grep", {"pattern": "needle", "path": "probe/unicode.txt"}, equals="probe/unicode.txt:1:Needle caf\u00e9"),
            step("Grep", {"pattern": "needle", "literal": True, "path": "probe/unicode.txt"}, equals="probe/unicode.txt:1:Needle caf\u00e9"),
            step("Grep", {"pattern": "caf\u00e9", "literal": True, "case_sensitive": True, "path": "probe/unicode.txt"}, equals="probe/unicode.txt:1:Needle caf\u00e9"),
            step("Grep", {"pattern": r"Item\(", "case_sensitive": True, "path": "probe/unicode.txt"}, equals="probe/unicode.txt:2:Item("),
            step("Grep", {"pattern": r"Item\(", "literal": True, "case_sensitive": True, "path": "probe/unicode.txt"}, prefix="No matches.", contains="backslashes literally"),
            step("Grep", {"pattern": "needle", "output_mode": "count", "path": "probe/unicode.txt"}, equals="probe/unicode.txt:1"),
            step("Grep", {"pattern": "needle", "output_mode": "files_with_matches", "path": "probe/unicode.txt"}, equals="probe/unicode.txt"),
        ]),
        ("patch_atomicity", [write, step("apply_patch", {"patch": "--- a/src/a.py\n+++ b/src/a.py\n@@ -1,3 +1,3 @@\n-VALUE = 1\n+VALUE = 2\n needle\n third\n"}, contains="successfully"), step("Read", {"file_path": "src/a.py"}, contains="VALUE = 2"), step("apply_patch", {"patch": "--- a/src/a.py\n+++ b/src/a.py\n@@ -1 +1 @@\n-absent\n+bad\n"}, contains="failed"), step("Read", {"file_path": "src/a.py"}, contains="VALUE = 2", excludes="bad"), step("apply_patch", {"patch": "garbage"}, contains="Malformed")]),
        ("legitimate_progress", [write, read, edit, step("Read", {"file_path": "src/a.py"}, contains="VALUE = 2"), step("Write", {"file_path": "tests/test_contract.py", "content": test_file}, contains="Wrote"), step("run_tests", {"command": test_command}, contains="SWE_PUBLIC_TEST_STATUS: PASS"), step("Edit", {"file_path": "src/a.py", "old_string": "VALUE = 2", "new_string": "VALUE = 3"}, contains="Edited"), step("run_tests", {"command": test_command}, contains="SWE_PUBLIC_TEST_STATUS: FAIL")]),
        ("nonsense_repeats", [write, read, read, read | {"expected": {"contains": "[tool feedback]", "metadata": {"tool_feedback_kind": "repeated_observation", "tool_feedback_count": 3}}}]),
        ("failed_edit_no_progress", [write, read, step("Edit", {"file_path": "src/a.py", "old_string": "absent", "new_string": "bad"}, contains="not found"), read, read | {"expected": {"contains": "[tool feedback]"}}]),
        ("shell_failures_and_limits", [step("Bash", {"command": "printf 'expected_failure\\n'; exit 7"}, contains="exit_code: 7"), step("Bash", {"command": "python -c \"print('x'*24000)\""}, contains="exit_code: 0", max_chars=13000), step("Bash", {"command": "printf 'alive\\n'"}, contains="alive")]),
        ("command_timeout", [step("Bash", {"command": "sleep 20"}, timeout=True)]),
        ("filename_miss_feedback", [
            step("Write", {"file_path": "src/handler.py", "content": "class CustomerLookup:\n    pass\n"}, contains="Wrote"),
            step("search_files", {"query": "CustomerLookup"}, contains="search_text", metadata={"tool_feedback_kind": "file_search_miss", "tool_feedback_count": 1}),
            step("search_files", {"query": "CustomerLookupHandler"}, contains="search_text", metadata={"tool_feedback_count": 2}),
            step("search_files", {"query": "LookupCustomer"}, contains="Repeated filename searches", metadata={"tool_feedback_kind": "repeated_file_search_miss", "tool_feedback_count": 3}),
            step("search_text", {"pattern": "CustomerLookup"}, contains="src/handler.py:1:class CustomerLookup", metadata_absent="tool_feedback_kind"),
            step("search_files", {"query": "missing_file_name"}, contains="search_text", metadata={"tool_feedback_count": 1}, excludes="Repeated filename searches"),
            step("search_files", {"query": "handler.py"}, equals="src/handler.py", metadata_absent="tool_feedback_kind"),
            step("search_files", {"query": "missing_file_name", "path": "absent"}, prefix="Path not found:", metadata_absent="tool_feedback_kind"),
        ]),
        ("failed_command_feedback", [
            step("Bash", {"command": "git init -q"}, contains="exit_code: 0"),
            step("Bash", {"command": "test -f ready.txt"}, contains="exit_code: 1", metadata_absent="tool_feedback_kind"),
            step("Bash", {"command": "test -f ready.txt"}, contains="This command failed again", metadata={"tool_feedback_kind": "repeated_failed_command", "tool_feedback_count": 2}),
            step("Write", {"file_path": "ready.txt", "content": "ready\n"}, contains="Wrote"),
            step("Bash", {"command": "test -f ready.txt"}, contains="exit_code: 0", metadata_absent="tool_feedback_kind"),
            step("Bash", {"command": "printf x >> progress.txt; exit 7"}, contains="exit_code: 7", metadata_absent="tool_feedback_kind"),
            step("Bash", {"command": "printf x >> progress.txt; exit 7"}, contains="exit_code: 7", metadata_absent="tool_feedback_kind"),
        ]),
    ]
    artifact_command = 'python -c "print(\'log needle\\n\'*1000, end=\'\')"'
    cases.extend([
        ("integral_decimals_and_mature_globs", [write,
            step("Read", {"file_path": "src/a.py", "offset": 1.0, "limit": "1.0"}, equals="needle"),
            step("Grep", {"pattern": "needle", "limit": "1.0", "max_results": 1.0}, contains="src/a.py:2:needle"),
            step("Write", {"file_path": "src/Convenience.swift", "content": "swift\n"}, contains="Wrote"),
            step("Write", {"file_path": "src/Convenience.m", "content": "objc\n"}, contains="Wrote"),
            step("Glob", {"pattern": "**/Convenience*.{swift,m}"}, equals="src/Convenience.m\nsrc/Convenience.swift"),
            step("Glob", {"pattern": "Convenience.[ms]*", "path": "src"}, equals="src/Convenience.m\nsrc/Convenience.swift"),
            step("Glob", {"pattern": "*.{swift,m}", "path": "src", "max_results": "1.0"}, contains="incomplete"),
        ]),
        ("observed_tests_preserve_explicit_intent", [
            step("Write", {"file_path": "tests/test_contract.py", "content": "import unittest\nclass Contract(unittest.TestCase):\n    def test_ok(self):\n        self.assertEqual(1, 1)\n"}, contains="Wrote"),
            *[step("Bash", {"command": test_command, **({"verification": flag} if flag is not None else {})},
                   metadata_paths={"public_test_attempt.test_outcome": "passed", "public_test_attempt.explicit_verification": flag is True},
                   **({"metadata_absent": "public_verification"} if flag is not True else {})) for flag in (None, False, True)],
            step("Bash", {"command": 'python -c "print(\'1 passed in 0.1s\')"'}, metadata_absent="public_test_attempt"),
            step("Bash", {"command": "git log -- src/_pytest/python.py"}, metadata_absent="public_test_attempt"),
        ]),
        ("grep_offset_actionable_correction", [write,
            step("Grep", {"pattern": "needle", "offset": 10}, requires_search_backend=False, prefix="Search error:", contains='Read(file_path="relative/file.py", offset=0, limit=100)'),
            step("Read", {"file_path": "src/a.py", "offset": 1, "limit": 1}, equals="needle")]),
        ("ansi_and_partial_runner_evidence", [
            step("Write", {"file_path": "pytest.py", "content": "print('\\x1b[31m1 failed\\x1b[0m, \\x1b[32m4 passed\\x1b[0m in 0.1s')\n"}, contains="Wrote"),
            step("Bash", {"command": "python -m pytest | head -100"}, prefix="exit_code: 0", metadata_paths={"public_test_attempt.test_outcome": "failed", "public_test_attempt.tests_executed": 5}, metadata_absent="public_verification"),
            step("Write", {"file_path": "pytest.py", "content": "print('================ FAILURES ================\\nE   AssertionError: mismatch')\n"}, contains="Wrote"),
            step("Bash", {"command": "python -m pytest | head -100"}, contains="terminal runner summary is missing", metadata_paths={"public_test_attempt.test_outcome": "unknown", "public_test_attempt.test_incomplete_failure_evidence": True, "public_test_attempt.tests_executed": -1})]),
        ("grep_alias_contract", [write,
            step("Grep", {"pattern": "needle", "limit": 1}, contains="needle", metadata={"search_limit_alias_used": True}),
            step("Grep", {"pattern": "needle", "limit": 1, "max_results": 2}, requires_search_backend=False, prefix="Search error:", contains="conflict"),
            step("Grep", {"pattern": "needle", "limit": 0}, requires_search_backend=False, prefix="Search error:", contains="integer >= 1"),
            step("Grep", {"pattern": "needle", "limit": True}, requires_search_backend=False, prefix="Search error:"),
            step("Grep", {"pattern": "needle", "unknown": 1}, requires_search_backend=False, prefix="Search error:", contains="Supported parameters")]),
        ("stable_artifact_retrieval", [
            step("Bash", {"command": artifact_command}, prefix="exit_code: 0\n[tool output stored]", contains="Read(file_path="),
            # Login-shell profiles can print a host-specific banner. Verify the
            # complete retrieved bytes against the capsule and the known payload.
            step("Read", {"file_path": "__artifact_path_0__"}, prefix="exit_code: 0\n", suffix="log needle\n" * 1000, sha256="__artifact_sha256_0__", lines="__artifact_lines_0__"),
            step("Grep", {"pattern": "needle", "path": "__artifact_path_0__", "limit": 1}, contains="log needle"),
            step("Bash", {"command": artifact_command}, metadata={"output_artifact_path": "__artifact_path_0__", "output_artifact_sha256": "__artifact_sha256_0__"}),
            step("Bash", {"command": artifact_command + " # changed request"}, metadata={"output_artifact_path": "__artifact_path_0__", "tool_feedback_kind": "repeated_output_content"}),
            step("Bash", {"command": artifact_command.replace("log needle", "new needle")}, metadata_not={"output_artifact_path": "__artifact_path_0__"}),
            step("Read", {"file_path": "__artifact_path_0__"}, sha256="__artifact_sha256_0__", suffix="log needle\n" * 1000)]),
        ("alternating_observation_feedback", [write,
            read, step("Grep", {"pattern": "needle"}, contains="needle"),
            read, step("Grep", {"pattern": "needle"}, contains="needle"),
            read | {"expected": {"metadata": {"tool_feedback_kind": "repeated_observation", "tool_feedback_count": 3}}},
            step("Grep", {"pattern": "needle"}, metadata={"tool_feedback_kind": "repeated_observation", "tool_feedback_count": 3}),
            edit, step("Read", {"file_path": "src/a.py"}, contains="VALUE = 2", metadata_absent="tool_feedback_kind")]),
        ("no_op_edit_feedback", [write, read] + [step("Edit", {"file_path": "src/a.py", "old_string": "VALUE = 1", "new_string": "VALUE = 1"}, contains="No edit was applied", metadata={"tool_feedback_kind": "no_op_edit", "tool_feedback_count": i}) for i in (1, 2, 3)]),
        ("explicit_pipeline_fail_fix_pass", [
            step("Write", {"file_path": "tests/test_contract.py", "content": test_file}, contains="Wrote"), write,
            step("Bash", {"command": test_command + " 2>&1 | head -50", "verification": True}, prefix="exit_code: 0", contains="pipeline can mask", metadata_paths={"public_verification.test_outcome": "failed", "public_verification.tests_failed": 1, "public_tests_passed": False}),
            read, edit,
            step("Bash", {"command": test_command + " 2>&1 | head -50", "verification": True}, prefix="exit_code: 0", metadata_paths={"public_verification.test_outcome": "passed", "public_verification.tests_executed": 1, "public_tests_passed": True})]),
        ("zero_tests_and_unknown_outcomes", [
            step("Write", {"file_path": "tests/__init__.py", "content": "\n"}, contains="Wrote"),
            step("Bash", {"command": test_command, "verification": True}, contains="No tests executed", metadata_paths={"public_verification.test_outcome": "not_run", "public_verification.tests_executed": 0, "public_tests_passed": None}),
            step("Bash", {"command": 'python -c "print(\'Test failed!\')"', "verification": True}, prefix="exit_code: 0", metadata_paths={"public_verification.test_outcome": "unknown", "public_tests_passed": None}),
            step("Bash", {"command": "python -m unittest contract_missing_runner 2>&1 | head -50", "verification": True}, prefix="exit_code: 0", contains="prerequisite is missing", metadata_paths={"public_verification.test_outcome": "failed", "public_verification.test_setup_error": True})]),
    ])
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


def trajectory_replay_scenarios(source: Path, *, latest: bool = False, sanity: bool = False) -> list[dict]:
    """Freeze complete saved observations with independently reviewed oracles."""
    from scripts import diagnose_trajectories as diagnosis
    specifications = {
        45: (3, "scikit-learn__scikit-learn-26194", "pytest", {"test_outcome": "unknown", "tests_executed": -1, "test_incomplete_failure_evidence": True}),
        56: (17, "pytest-dev__pytest-7571", "pytest", {"test_outcome": "passed", "tests_executed": 1, "tests_failed": 0, "test_summary_count": 2, "test_incomplete_failure_evidence": False}),
        76: (42, "django__django-17087", "unittest", {"test_outcome": "failed", "tests_executed": 8, "tests_failed": -1, "test_error_events": 138, "test_count_conflict": True, "test_parse_confidence": "low"}),
    }
    if latest:
        specifications = {
            19: (17, "django__django-13551", "unittest", {"test_outcome": "unknown", "test_exit_code": 1, "test_exit_provenance": "shell_pipeline", "test_failure_evidence": False}),
            30: (43, "django__django-17087", "unittest", {"test_outcome": "failed", "tests_executed": 0, "test_loader_error_count": 1, "test_runner_reported_tests": 1, "verification_meaningful": False}),
            31: (19, "django__django-15104", "unittest", {"test_outcome": "unknown", "test_execution_error": True, "test_incomplete_failure_evidence": False}),
            89: (20, "django__django-16569", "unittest", {"test_outcome": "passed", "tests_executed": 153, "tests_failed": 0, "test_runner_recognized": True}),
        }
    if sanity:
        specifications = {
            2: (9, "scikit-learn__scikit-learn-26194", "pytest", {"test_outcome": "passed", "tests_executed": 204, "tests_failed": 0}),
            10: (12, "sympy__sympy-21847", "sympy", {"test_outcome": "passed", "tests_executed": 11, "test_runner_recognized": True}),
            12: (8, "django__django-15380", "unittest", {"test_outcome": "not_run", "test_setup_error": True}),
        }
    result = []
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    with source.open() as stream:
        for line, raw in enumerate(stream, 1):
            if line not in specifications:
                continue
            event_index, task, framework, oracle = specifications[line]
            row = json.loads(raw)
            if diagnosis._identity(row)[1] != task:
                raise ValueError(f"Trajectory replay task mismatch at line {line}")
            messages = diagnosis.production._parse_output(row["output"])
            paired = []
            for index, message in enumerate(messages[:-1]):
                if message["role"] == "assistant" and messages[index + 1]["role"] == "user":
                    paired.extend(zip(diagnosis.production.TOOL_BLOCK_RE.finditer(message["content"]),
                                      diagnosis.production.TOOL_RESPONSE_RE.findall(messages[index + 1]["content"])))
            call, response = paired[event_index]
            arguments = diagnosis.production._tool_call(call)["parameters"]
            observation = diagnosis.re.split(r"\n(?:\[test outcome\]|\[test feedback\]|\[tool feedback\])", response, maxsplit=1)[0].lstrip("\n")
            exit_match = diagnosis.re.match(r"^exit_code:\s*(-?\d+)", observation)
            exit_code = int(exit_match[1]) if exit_match else None
            observation = diagnosis.re.sub(r"^exit_code:\s*-?\d+\s*\n", "", observation, count=1)
            if latest or sanity:
                result.append({"id": f"saved_{'v8' if sanity else 'v7'}_line_{line}_event_{event_index}", "expected_hits": None, "steps": [],
                    "provenance": {"source": str(source), "source_sha256": source_hash, "line": line, "event": event_index,
                        "task": task, "saved_response_sha256": hashlib.sha256(response.encode()).hexdigest(),
                        "purpose": "Local parser replay of original command and complete saved response; does not execute repository tests or reproduce OOM/timeout"},
                    "parser_replay": {"command": arguments["command"], "output": observation, "exit_code": exit_code, "expected": oracle}})
                continue
            filename = "pytest.py" if framework == "pytest" else "runtests.py"
            command = "python -m pytest" if framework == "pytest" else "python runtests.py"
            result.append({"id": f"saved_trajectory_line_{line}_event_{event_index}", "expected_hits": None,
                "provenance": {"source": str(source), "source_sha256": source_hash, "line": line, "event": event_index,
                               "task": task, "original_command": arguments.get("command"),
                               "saved_response_sha256": hashlib.sha256(response.encode()).hexdigest(), "response_chars": len(response),
                               "purpose": "Replay complete runner output through service dispatch; fixture does not execute original repository tests"},
                "steps": [step("Write", {"file_path": filename, "content": "import sys\nsys.stdout.write(" + repr(observation) + ")\n"}, contains="Wrote"),
                          step("Bash", {"command": command}, prefix="exit_code: 0", metadata_absent="public_verification",
                               metadata_paths={"public_test_attempt." + key: value for key, value in oracle.items()})]})
    if len(result) != len(specifications):
        raise ValueError("Missing reviewed trajectory replay rows")
    return result


def evaluator_fixture_case() -> dict:
    """Synthetic evaluator coverage; contains no benchmark hidden material."""
    patch = ("diff --git a/check.py b/check.py\n--- a/check.py\n+++ b/check.py\n@@ -1 +1 @@\n"
             "-print('base')\n+from pathlib import Path; assert Path('empty.py').read_bytes() == b''; assert Path('model.txt').read_text() == 'repair'\n"
             "diff --git a/empty.py b/empty.py\nnew file mode 100644\n")
    from recipe.swe_agent.benchmarks.convert_swe_benchmarks import hidden_patch_reset_commands
    from recipe.swe_agent.task_payload import evaluator_command
    commands = hidden_patch_reset_commands(patch, "HEAD")
    if commands != ["git checkout HEAD -- check.py", "rm -f -- empty.py"]:
        raise ValueError("Synthetic evaluator reset generator differs from independently specified oracle")
    return {"id": "synthetic_evaluator_empty_file", "expected_hits": None,
            "steps": [step("Bash", {"command": "git init -q && git -c user.name=Fixture -c user.email=fixture@example.com commit --allow-empty -qm base && printf \"print('base')\\n\" > check.py && git add check.py && git -c user.name=Fixture -c user.email=fixture@example.com commit -qm check && printf repair > model.txt && printf stale > empty.py"}, prefix="exit_code: 0")],
            "evaluator_fixture": {"base_sha": "HEAD", "hidden_test_patch": patch,
                "trusted_hidden_test_patch": True, "verifier_setup_commands": commands,
                "test_commands": [evaluator_command({"prepare": [], "run": "python check.py"})],
                "evaluator_test_commands": [{"prepare": [], "run": "python check.py"}]}}


def v8_scenarios() -> list[dict]:
    runner = "import unittest; s=unittest.TestSuite(); s.addTest(unittest.FunctionTestCase(lambda: None)); r=unittest.TextTestRunner(); r.run(s)"
    return [evaluator_fixture_case(),
        {"id": "unittest_bound_runner", "expected_hits": None, "steps": [
            step("Bash", {"command": "python -c " + shlex.quote(runner)}, prefix="exit_code: 0",
                 metadata_paths={"public_test_attempt.test_outcome": "passed", "public_test_attempt.tests_executed": 1})]},
        {"id": "filter_status_is_not_test_status", "expected_hits": None, "steps": [
            step("Write", {"file_path": "runtests.py", "content": "print('OK')\n"}, contains="Wrote"),
            step("Bash", {"command": "python runtests.py | grep FAIL"}, prefix="exit_code: 1",
                 metadata_paths={"public_test_attempt.test_outcome": "unknown", "public_test_attempt.test_exit_provenance": "shell_pipeline"})]},
        {"id": "loader_placeholder_not_meaningful", "expected_hits": None, "steps": [
            step("Bash", {"command": "python -m unittest missing_contract_module"}, prefix="exit_code: 1",
                 metadata_paths={"public_test_attempt.test_outcome": "failed", "public_test_attempt.tests_executed": 0,
                                 "public_test_attempt.verification_meaningful": False})]},
        {"id": "django_worker_environment", "expected_hits": None, "steps": [
            step("Bash", {"command": "python -c \"import os; print('DJANGO_TEST_PROCESSES=' + os.environ.get('DJANGO_TEST_PROCESSES', 'missing'))\""}, contains="DJANGO_TEST_PROCESSES=1")]},
    ]


def v9_scenarios() -> list[dict]:
    return [
        {"id": "pytest_warnings_and_zero_selection", "expected_hits": None, "steps": [
            step("Write", {"file_path": "pytest.py", "content": "print('2 failed, 203 passed, 3 warnings in 1.37s')\n"}, contains="Wrote"),
            step("Bash", {"command": "python -m pytest | tail -20"}, metadata_paths={"public_test_attempt.test_outcome": "failed", "public_test_attempt.tests_executed": 205}, metadata_absent="public_verification"),
            step("Write", {"file_path": "pytest.py", "content": "print('collected 199 items / 199 deselected / 0 selected\\n199 deselected in 0.27s')\n"}, contains="Wrote"),
            step("Bash", {"command": "python -m pytest | tail -20"}, metadata_paths={"public_test_attempt.test_outcome": "not_run", "public_test_attempt.tests_executed": 0})]},
        {"id": "public_native_runner_guidance", "expected_hits": None, "steps": [
            step("Write", {"file_path": "django/__init__.py", "content": ""}, contains="Wrote"),
            step("Write", {"file_path": "tests/runtests.py", "content": "print('Ran 1 test in 0.1s\\nOK')\n"}, contains="Wrote"),
            step("Write", {"file_path": "tests/test_sqlite.py", "content": ""}, contains="Wrote"),
            step("Bash", {"command": "python runtests.py | head -20"}, contains="--settings=test_sqlite --parallel 1", metadata_paths={"public_test_attempt.test_outcome": "not_run"}),
            step("Bash", {"command": "python tests/runtests.py"}, metadata_paths={"public_test_attempt.test_outcome": "passed", "public_test_attempt.tests_executed": 1}, metadata_absent="public_verification")]},
        {"id": "sympy_native_runner", "expected_hits": None, "steps": [
            step("Write", {"file_path": "sympy/__init__.py", "content": "def test(path):\n    print('tests finished: 11 passed, in 0.07 seconds')\n"}, contains="Wrote"),
            step("Bash", {"command": "python -c \"import sympy; sympy.test('public.py')\""}, metadata_paths={"public_test_attempt.test_outcome": "passed", "public_test_attempt.tests_executed": 11}, metadata_absent="public_verification")]},
        *[{"id": "runner_terminal_regression_" + str(index), "expected_hits": None, "steps": [],
           "parser_replay": {"command": command, "output": output, "exit_code": 0, "expected": expected}}
          for index, (command, output, expected) in enumerate([
              ("go test ./... | tail", "--- PASS: TestOne (0.01s)\nFAIL\tpkg\t0.1s\nFAIL\n", {"test_outcome": "failed", "tests_executed": -1}),
              ("go test ./...", "ok\tpkg\t1.316s\n", {"test_outcome": "passed", "tests_executed": -1}),
              ("jest | tail", "Test Suites: 1 failed, 1 passed, 2 total\nTests: 8 passed, 8 total\n", {"test_outcome": "failed", "tests_passed": 8}),
          ])],
    ]


def check_evaluator_fixture(response: dict) -> list[str]:
    metadata = response.get("metadata", {})
    # Legacy benchmark fixtures predate launch-marker provenance.  Their
    # validity is intentionally unknown, rather than invalid; False still
    # signals a confirmed evaluator failure.
    checks = {"ok": response.get("ok") is True, "score": response.get("score") == 1.,
        "started": metadata.get("hidden_tests_started") is True, "valid": metadata.get("evaluation_valid") in {True, None},
        "phase": metadata.get("hidden_failure_phase") == "none", "private_evidence": metadata.get("private_evaluator_artifact_status") == "available",
        "complete_private_logs": metadata.get("private_evaluator_logs_complete") is True}
    return ["evaluator fixture: " + name for name, passed in checks.items() if not passed]


def check_parser_replay(case: dict) -> dict:
    from recipe.swe_agent.test_results import summarize_test_execution
    fixture = case["parser_replay"]
    if fixture.get("rebench_contract"):
        from recipe.swe_agent.test_results import score_official_tests
        contract = fixture["rebench_contract"]
        parsed = audit_rebench_outcomes(fixture["command"], fixture["output"], fixture["exit_code"],
            log_parser=contract["log_parser"], identity_scope=contract.get("identity_scope", "test"))
        score = score_official_tests(contract, [parsed], 1)
        actual = {"complete": score["official_tests_complete"],
                  "reason": parsed["reason"] if not parsed["complete"] else score["official_tests_reason"],
                  "all_required_passed": score["official_tests_success"]}
    else:
        actual = summarize_test_execution(fixture["command"], fixture["output"], exit_code=fixture["exit_code"])
    failures = [f"saved parser replay {key}: expected {value!r}, got {actual.get(key)!r}"
                for key, value in fixture["expected"].items() if actual.get(key) != value]
    return {"scope": "local_saved_evidence_parser_only", "actual": actual, "failures": failures}


def reference_replay_scenarios(datasets: list[Path], audit_path: Path, reviewed_path: Path) -> list[dict]:
    """Freeze reviewed, checksum-verified full logs in the existing benchmark."""
    report = json.loads(audit_path.read_text())
    reviewed = json.loads(reviewed_path.read_text())
    if reviewed.get("audit_sha256") != hashlib.sha256(audit_path.read_bytes()).hexdigest():
        raise ValueError("reviewed replay does not match the source audit")
    _, cases = audit_inventory(datasets, profile="rebench")
    by_id = {case["inventory"]["id"]: case for case in cases}
    oracles = {(r["task_id"], r["mode"]): r for r in reviewed["cases"]}
    frozen = []
    for result in report["results"]:
        case = by_id[result["id"]]
        for mode, arm in result.get("arms", {}).items():
            if not arm.get("private_evaluator", {}).get("private_copy"):
                continue
            evidence = oracles[(result["task_id"], mode)]["evidence"]
            annotated = {**arm, "task_id": result["task_id"], "runner_provenance": arm["reward"]["metadata"].get("hidden_runner_provenance")}
            phases, sha = retained_audit_phases(annotated, reviewed_path.parent / "freeze_private")
            chunks = re.split(r"(?m)^exit_code: (-?\d+)\ncommand: ", phases.get("test_commands", ""))
            commands = case["task"]["test_commands"]
            if len(commands) != 1 or len(chunks) != 3:
                raise ValueError("saved reference benchmark requires one complete runner per case")
            markers = re.findall(r"^SWE_RUNNER_LAUNCHED_[a-f0-9]+$", chunks[2], re.M)
            if len(markers) != 1:
                raise ValueError("saved reference launch is unverified")
            frozen.append({"id": "reference_" + result["task_id"] + "_" + mode, "expected_hits": None, "steps": [],
                "provenance": {"audit_sha256": reviewed["audit_sha256"], "private_sha256": sha,
                               "purpose": "Offline hidden-evaluator parser replay only; never sent as model tool observations"},
                "parser_replay": {"command": commands[0], "output": chunks[2].split(markers[0] + "\n", 1)[1],
                    "exit_code": int(chunks[1]), "rebench_contract": case["reference_contract"],
                    "expected": {"complete": evidence.get("complete", False), "reason": evidence["reason"],
                                 "all_required_passed": evidence.get("all_required_passed", False)}}})
    return frozen


def build(directory: Path, trajectory_input: Path | None = None, latest_trajectory_input: Path | None = None, sanity_trajectory_input: Path | None = None,
          *, reference_datasets: list[Path] | None = None, reference_audit_report: Path | None = None,
          reference_replay_report: Path | None = None) -> dict:
    data = scenarios()
    data.extend(v8_scenarios())
    data.extend(v9_scenarios())
    if reference_audit_report or reference_replay_report or reference_datasets:
        if not (reference_audit_report and reference_replay_report and reference_datasets):
            raise ValueError("reference fixtures require datasets, source audit and reviewed replay")
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        data.extend(reference_replay_scenarios(reference_datasets, reference_audit_report, reference_replay_report))
    if trajectory_input is not None:
        data.extend(trajectory_replay_scenarios(trajectory_input))
    if latest_trajectory_input is not None:
        data.extend(trajectory_replay_scenarios(latest_trajectory_input, latest=True))
    if sanity_trajectory_input is not None:
        data.extend(trajectory_replay_scenarios(sanity_trajectory_input, sanity=True))
    manifest = {"version": VERSION, "scenarios_sha256": digest(data), "scenario_count": len(data),
                "read_offset_policy": "Read offset is zero-based as published; internal start_line is one-based",
                "policy_changes_from_v2": ["Correct benchmark offset oracle to the documented zero-based schema; production indexing is preserved", "Accept only explicitly declared task mount prefixes in addition to relative paths", "Add limit-only/invalid Read ranges and Grep glob, count, leading-dash and truncation boundaries"],
                "policy_changes_from_v3": ["Guide content queries after path-only search misses", "Expose bounded session feedback for repeated empty searches and unchanged failed commands", "Check progress and mutation false-positive guards"],
                "policy_changes_from_v9": ["Require pinned ripgrep without a Python search fallback", "Use ripgrep for Glob and filename discovery", "Accept lossless integral decimal arguments"],
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
    if manifest["version"] not in {VERSION, "swe-tool-online-v9", "swe-tool-online-v8", "swe-tool-online-v7", "swe-tool-online-v6", "swe-tool-online-v5", "swe-tool-online-v4", "swe-tool-online-v3", "swe-tool-online-v2"} or digest(cases) != manifest["scenarios_sha256"]:
        raise ValueError("Online manifest version/checksum mismatch")
    return manifest, cases


def offline(directory: Path, output: Path) -> dict:
    """Run production session dispatch in disposable host workspaces; no Docker calls."""
    # Resolve this repository's diagnosis package before replay helpers adjust
    # sys.path to load the adjacent recipe repository (which also has scripts/).
    from scripts.diagnose_trajectories import collect_private_evaluator
    from recipe.swe_agent import remote_execution_service as service
    from recipe.swe_agent import search_utils
    manifest, cases = load(directory)
    report = {"version": manifest["version"], "manifest_sha256": digest(manifest),
              "source_fingerprint": importlib.import_module("recipe.swe_agent.tool_contract_info").source_fingerprint(),
              "cases": [], "excluded_online_assertions": ["Docker container identity; image, deployment, transport and cleanup evidence require online execution"]}
    original_stream = search_utils._stream_command

    def no_rg(command, **kwargs):
        if command[0] == (os.environ.get("SWE_AGENT_RG_BINARY") or "rg"):
            return search_utils._StreamResult([], None, error="forced missing rg", backend_missing=True)
        return original_stream(command, **kwargs)

    for backend in ("native", "missing_rg"):
        with mock.patch.object(search_utils, "_stream_command", no_rg if backend == "missing_rg" else original_stream):
            for case in cases:
                print(f"offline benchmark: {backend}/{case['id']}", file=sys.stderr, flush=True)
                result = {"id": case["id"], "backend": backend, "events": [], "failures": []}
                report["cases"].append(result)
                if case.get("parser_replay"):
                    result["parser_replay"] = check_parser_replay(case)
                    result["failures"].extend(result["parser_replay"]["failures"])
                with tempfile.TemporaryDirectory(prefix="swe-tool-offline-") as raw_workspace:
                    workspace = Path(raw_workspace)
                    task = {"task_id": case["id"], "execution_phase": "rollout", "sandbox_backend": "local",
                            "docker_mount": "/testbed", "hard_timeout_seconds": 3 if case["id"] == "command_timeout" else 30,
                            "public_test_commands": []}
                    session = service.ExecutionSession(case["id"], task, case["id"], workspace, False)
                    # Only workspace/image provisioning is replaced. Dispatch, tools,
                    # subprocess execution, mutation metadata and rewards are production code.
                    with mock.patch.object(session, "prepare", return_value="fixture workspace ready"), mock.patch.dict(os.environ, {"DJANGO_TEST_PROCESSES": os.environ.get("SWE_AGENT_DJANGO_TEST_PROCESSES", "1"), "SWE_AGENT_DIAGNOSTICS_DIR": str(workspace.parent / (workspace.name + "_private"))}):
                        for index, item in enumerate(case["steps"]):
                            try:
                                item = bind_artifacts(item, result["events"])
                            except ValueError as exc:
                                result["failures"].append({"index": index, "error": str(exc)})
                                break
                            session.begin_operation("tool", task["hard_timeout_seconds"])
                            try:
                                text, reward, metadata = session.run_tool(item["name"], item["arguments"])
                            finally:
                                session.end_operation()
                            response = {"ok": True, "text": text, "metadata": metadata, "reward": reward}
                            event = {"index": index, "name": item["name"], "arguments": item["arguments"],
                                     "response": text, "metadata": metadata}
                            result["events"].append(event)
                            oracle = item.get("missing_rg_expected", item["expected"]) if backend == "missing_rg" else item["expected"]
                            expected = {key: value for key, value in oracle.items() if key != "container"}
                            if backend == "missing_rg" and "missing_rg_expected" in item:
                                event["expected_degraded_response"] = True
                            for error in check_response(expected, response):
                                result["failures"].append({"index": index, "error": error})
                        if case.get("evaluator_fixture"):
                            session.begin_operation("reward", 30)
                            try:
                                score, text, metadata = session.run_reward({**task, **case["evaluator_fixture"], "execution_phase": "reward"})
                            finally:
                                session.end_operation()
                            result["evaluator"] = {"ok": True, "score": score, "metadata": metadata}
                            result["failures"].extend(check_evaluator_fixture(result["evaluator"]))
                            artifact = Path(metadata.get("private_evaluator_artifact_path", ""))
                            if artifact.is_file():
                                # Verify complete compressed logs using the same
                                # operator collector, then remove this fixture's
                                # disposable evidence (never a service directory).
                                collection = collect_private_evaluator(
                                    {"task_id": task["task_id"], "reward_evidence": metadata},
                                    [str(artifact.parent) + "=" + str(artifact.parent)], workspace / "operator_copy",
                                )
                                result["private_evidence_collection"] = collection
                                if collection.get("status") != "verified" or not collection.get("logs_complete"):
                                    result["failures"].append("Full private fixture logs not verified complete")
                                for blob in artifact.parent.glob("*.log.gz"):
                                    blob.unlink()
                                artifact.unlink()
                                artifact.parent.rmdir()
                    if case["expected_hits"] is not None:
                        result["reward"] = reward_checks(result["events"], case["expected_hits"])
                        result["failures"].extend(result["reward"]["failures"])
                result["status"] = "failed" if result["failures"] else "passed"
    report["counts"] = {"scenario_executions": len(report["cases"]),
                        "tool_dispatches": sum(len(case["events"]) for case in report["cases"]),
                        "failures": sum(len(case["failures"]) for case in report["cases"]),
                        "reward_checks": sum(len(case.get("reward", {}).get("checks", [])) for case in report["cases"])}
    report["missing_rg_capability_complete"] = False
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


def persistent_search_verified(host: dict) -> bool:
    backend = importlib.import_module("recipe.swe_agent.rg_backend")
    persistent = host.get("persistent_rg", {})
    rg = host.get("rg", {})
    return bool(host.get("complete") is True and host.get("rg_startup_status") == "persistent_verified"
                and persistent.get("ok") is True and persistent.get("binary_sha256") == backend.BINARY_SHA256
                and persistent.get("version") == backend.VERSION and persistent.get("target") == backend.TARGET
                and persistent.get("binary") and persistent.get("binary") == rg.get("path")
                and rg.get("functional") is True)


class Client:
    def get(self, url: str, path: str, timeout: float = 8) -> dict:
        headers = {}
        token = os.environ.get("SWE_AGENT_EXECUTION_TOKEN") or os.environ.get("SWE_AGENT_REMOTE_EXECUTION_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        try:
            tools = importlib.import_module("recipe.swe_agent.tools")
            with tools._remote_execution_opener().open(
                    urllib.request.Request(url + path, headers=headers), timeout=timeout) as response:
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
        elif kind == "suffix":
            valid = text.endswith(value)
        elif kind == "sha256":
            valid = hashlib.sha256(text.encode()).hexdigest() == value
        elif kind == "lines":
            valid = len(text.splitlines()) == value
        elif kind == "max_chars":
            valid = len(text) <= value
        elif kind == "metadata":
            valid = all(metadata.get(key) == item for key, item in value.items())
        elif kind == "metadata_absent":
            valid = value not in metadata
        elif kind == "metadata_not":
            valid = all(key in metadata and metadata[key] != item for key, item in value.items())
        elif kind == "metadata_paths":
            def get_path(path):
                current = metadata
                for key in path.split("."):
                    if not isinstance(current, dict) or key not in current:
                        return object()
                    current = current[key]
                return current
            valid = all(get_path(key) == item for key, item in value.items())
        elif kind == "container":
            valid = bool(metadata.get("container_name"))
        elif kind == "timeout":
            valid = bool(metadata.get("tool_hard_timeout")) or "timeout" in text.lower() or "exit_code: 124" in text
        else:
            raise ValueError(f"Unknown assertion: {kind}")
        if not valid:
            errors.append(f"{kind}: expected {value!r}")
    return errors


def bind_artifacts(item: dict, events: list[dict]) -> dict:
    def bind(value):
        if isinstance(value, dict):
            return {key: bind(child) for key, child in value.items()}
        if isinstance(value, list):
            return [bind(child) for child in value]
        for field in ("path", "sha256", "lines"):
            prefix = f"__artifact_{field}_"
            if isinstance(value, str) and value.startswith(prefix) and value.endswith("__"):
                index = int(value[len(prefix):-2])
                observed = events[index].get("metadata", {}).get(f"output_artifact_{field}") if index < len(events) else None
                if observed is None:
                    raise ValueError(f"artifact {field} missing from event {index}")
                return observed
        return value
    return bind(item)


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
    if case.get("parser_replay"):
        replay = check_parser_replay(case)
        return {"id": case["id"], "events": [], "parser_replay": replay, "failures": replay["failures"],
                "incomplete": [], "status": "failed" if replay["failures"] else "passed"}
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
            item = bind_artifacts(item, result["events"])
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
        if case.get("evaluator_fixture"):
            result["evaluator"] = client.post(url, "/reward", {**payload, "task": {**task, **case["evaluator_fixture"], "execution_phase": "reward"}}, deadline)
            result["failures"].extend(check_evaluator_fixture(result["evaluator"]))
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
                     "cleanup_path": release.get("workspace_cleanup_path") or "",
                     "container_name": result.get("preflight", {}).get("metadata", {}).get("container_name", "")}
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
              "reward_execution": "local detector checks plus synthetic evaluator /reward fixture; no real benchmark hidden tests or model inference"}
    for raw_url in dict.fromkeys(urls):
        url = raw_url.rstrip("/")
        print(f"online benchmark: {url}", file=sys.stderr, flush=True)
        service = {"url": url, "ready": client.get(url, "/ready"), "deployment": client.get(url, "/tool_contract_info"), "cases": [], "incomplete": []}
        services.append(service)
        deployment = service["deployment"]
        if not persistent_search_verified(deployment.get("host_diagnostics", {})):
            service["incomplete"].append("persistent rg deployment not verified; full search capability gate cannot pass")
        if manifest["version"] != VERSION:
            service["incomplete"].append("legacy benchmark lacks current artifact/feedback/verification checks; build " + VERSION)
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
                if not case.get("parser_replay") and (not result.get("preflight") or "cached Python-capable" in " ".join(result["incomplete"])):
                    service["incomplete"].append("remaining scenarios skipped after image/claim preflight failure")
                    break
                if any("release not confirmed" in error or "container removal not confirmed" in error or error.startswith("cleanup:") or "session still" in error for error in result["incomplete"]):
                    service["incomplete"].append("remaining scenarios skipped after cleanup failure")
                    break
        service["status"] = "failed" if any(item["failures"] for item in service["cases"]) else "incomplete" if service["incomplete"] or any(item["incomplete"] for item in service["cases"]) else "passed"
    report["regression_pass"] = bool(services) and all(item["status"] == "passed" for item in services)
    report["counts"] = {"services": len(services), "scenarios_expected": len(services) * len(cases),
                        "scenarios_executed": sum(len(item["cases"]) for item in services),
                        "local_saved_parser_replays": sum(bool(case.get("parser_replay")) for item in services for case in item["cases"]),
                        "docker_scenarios_executed": sum(bool(case.get("preflight")) for item in services for case in item["cases"]),
                        "assertion_failures": sum(len(case["failures"]) for item in services for case in item["cases"]),
                        "reward_checks": sum(len(case.get("reward", {}).get("checks", [])) for item in services for case in item["cases"]),
                        "reward_failures": sum(len(case.get("reward", {}).get("failures", [])) for item in services for case in item["cases"])}
    write_json(output, report)
    return report


def search_backend_probe_case() -> dict:
    environment_probe = (
        "import json, os, platform, shutil, subprocess; from pathlib import Path; "
        "p = Path('/etc/os-release'); "
        "q = shutil.which('dpkg-query'); "
        "r = subprocess.run([q, '-W', '-f=${Status} ${Version}', 'ripgrep'], "
        "stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, timeout=5) if q else None; "
        "print(json.dumps({'scope': 'task_container', 'architecture': platform.machine(), "
        "'libc': platform.libc_ver(), 'path': os.environ.get('PATH'), "
        "'uid': os.getuid(), 'os_release': p.read_text() if p.exists() else None, "
        "'ripgrep_package': {'exit_code': r.returncode, 'output': r.stdout[:1000]} if r else None, "
        "'standard_rg_paths': {n: {'exists': Path(n).exists(), 'executable': os.access(n, os.X_OK)} "
        "for n in ['/usr/bin/rg', '/usr/local/bin/rg', '/bin/rg']}, "
        "'executables': {n: shutil.which(n) for n in ['rg', 'apt-get', 'apk', 'dnf', 'curl', 'wget']}, "
        "'package_records': {n: Path(n).exists() for n in ['/var/lib/dpkg/status', '/lib/apk/db/installed']}}))"
    )
    return {"id": "search_backend_diagnosis", "expected_hits": None, "steps": [
        step("Bash", {"command": "python -c " + shlex.quote(environment_probe)}, contains="exit_code: 0", container=True),
        step("Write", {"file_path": "probe/ascii.txt", "content": "Needle\n"}, contains="Wrote"),
        step("Grep", {"pattern": "needle", "path": "probe/ascii.txt"}, equals="probe/ascii.txt:1:Needle"),
        step("Write", {"file_path": "probe/unicode.txt", "content": "Needle caf\u00e9\nItem(\n"}, contains="Wrote"),
        step("Grep", {"pattern": "needle", "path": "probe/unicode.txt"}, equals="probe/unicode.txt:1:Needle caf\u00e9"),
        step("Grep", {"pattern": "caf\u00e9", "path": "probe/unicode.txt", "literal": True, "case_sensitive": True}, equals="probe/unicode.txt:1:Needle caf\u00e9"),
        step("Grep", {"pattern": r"Item\(", "path": "probe/unicode.txt", "case_sensitive": True}, equals="probe/unicode.txt:2:Item("),
        step("Grep", {"pattern": r"Item\(", "path": "probe/unicode.txt", "literal": True, "case_sensitive": True}, prefix="No matches.", contains="backslashes literally"),
        step("Read", {"file_path": "probe/unicode.txt"}, equals="Needle caf\u00e9\nItem(\n"),
    ]}


def diagnose(urls: list[str], output: Path, image: str = "", deadline_seconds: float = 90, client: Any = None) -> dict:
    """Probe existing services with at most one isolated cached-image session at a time."""
    client = client or Client()
    local = importlib.import_module("recipe.swe_agent.tool_contract_info").source_fingerprint()
    case = search_backend_probe_case()
    report = {"version": "swe-tool-service-diagnosis-v2", "local_source_fingerprint": local,
              "probe_case": case, "probe_sha256": digest(case), "services": [],
              "scope": "Grep backend environment/execution evidence comes from the service host; Bash environment commands execute in the task container. No package installation, image pull/build, training, or existing-session cleanup."}
    for raw_url in dict.fromkeys(urls):
        url = raw_url.rstrip("/")
        print(f"service diagnosis: {url}", file=sys.stderr, flush=True)
        health = client.get(url, "/health")
        ready = client.get(url, "/ready")
        deployment = client.get(url, "/tool_contract_info")
        service = {"url": url, "health": health, "ready": ready, "deployment": deployment,
                   "incomplete": [], "probe": None, "status": "incomplete"}
        report["services"].append(service)
        deployed_hash = deployment.get("source_fingerprint", {}).get("sha256")
        service["deployment_matches_local_source"] = bool(deployed_hash) and deployed_hash == local["sha256"]
        if not service["deployment_matches_local_source"]:
            service["incomplete"].append("deployment source unavailable or differs from local production")
        if not health.get("ok") or not ready.get("ok") or not ready.get("docker_available"):
            service["incomplete"].append("health or Docker readiness unavailable")
        if not deployment.get("ok"):
            service["incomplete"].append("deployment diagnostics unavailable")
        host = deployment.get("host_diagnostics", {})
        host_complete = (host.get("version") == 1 and host.get("scope") == "execution_service_host"
                         and host.get("complete") is True
                         and host.get("rg", {}).get("status") in {"missing", "disabled", "available", "execution_failed", "incompatible"})
        service["host_diagnostics_complete"] = host_complete
        service["host_rg_status"] = host.get("rg", {}).get("status", "unknown")
        service["host_rg_reason"] = host.get("rg", {}).get("reason", "unknown")
        service["persistent_search_verified"] = persistent_search_verified(host)
        if not service["persistent_search_verified"]:
            service["incomplete"].append("persistent rg deployment not verified; prepare explicitly and restart service")
        if not host_complete:
            service["incomplete"].append("host diagnostics unavailable or incomplete; restart updated service or retry busy probe")
        slots = health.get("available_trajectory_slots")
        unlimited = health.get("max_active_trajectories") == 0
        if not unlimited and (not isinstance(slots, (int, float)) or slots <= 0):
            service["incomplete"].append("no confirmed free trajectory slot; isolated probe skipped")
        images = deployment.get("cached_images", [])
        selected = image or next((item["tag"] for item in images if "sweb.eval" in item["tag"] or "python" in item["tag"]), "")
        service["image"] = selected
        if not selected or not any(item["tag"] == selected for item in images):
            service["incomplete"].append("suitable cached image not confirmed; inventory may be truncated")
        # Older deployments can still supply useful functional evidence without passing the gate.
        blockers = [reason for reason in service["incomplete"] if not reason.startswith(("deployment source", "host diagnostics", "persistent rg"))]
        if not blockers:
            service["probe"] = run_case(client, url, selected, case, deadline_seconds)
            service["status"] = service["probe"]["status"]
            if service["status"] == "passed" and service["incomplete"]:
                service["status"] = "incomplete"
        write_json(output, report)
    report["regression_pass"] = bool(report["services"]) and all(item["status"] == "passed" for item in report["services"])
    report["counts"] = {
        "services": len(report["services"]),
        "health_ok": sum(bool(item["health"].get("ok")) for item in report["services"]),
        "docker_ready": sum(bool(item["ready"].get("ok") and item["ready"].get("docker_available")) for item in report["services"]),
        "host_rg_on_path": sum(item["deployment"].get("native_rg_available") is True for item in report["services"]),
        "host_diagnostics_complete": sum(item["host_diagnostics_complete"] for item in report["services"]),
        "host_rg_functional": sum(item["deployment"].get("host_diagnostics", {}).get("rg", {}).get("functional") is True for item in report["services"]),
        "persistent_search_verified": sum(item["persistent_search_verified"] for item in report["services"]),
        "isolated_probes": sum(item["probe"] is not None for item in report["services"]),
        "passed": sum(item["status"] == "passed" for item in report["services"]),
        "failed": sum(item["status"] == "failed" for item in report["services"]),
        "incomplete": sum(item["status"] == "incomplete" for item in report["services"]),
    }
    write_json(output, report)
    return report


def preflight(urls: list[str], output: Path, client: Any = None) -> dict:
    """Read-only deployment check; no claims, container creation or tool execution."""
    client = client or Client()
    local = importlib.import_module("recipe.swe_agent.tool_contract_info").source_fingerprint()
    report = {"version": VERSION, "mode": "read_only_preflight", "local_source_fingerprint": local,
              "services": [], "settings": {key: os.environ.get(key) for key in (
                  "EXPERIMENT_NAME", "MODEL_PATH", "VAL_FILES", "TRAIN_FILES", "SWE_AGENT_EXECUTION_URLS",
                  "SWE_AGENT_HARNESS_PROFILE", "SWE_AGENT_MAX_TURNS", "MAX_RESPONSE_LENGTH",
                  "SWE_AGENT_TRAINING_REPEATED_TOOL_REWARD_SHAPING", "VALIDATION_DATA_DIR",
                  "SWE_AGENT_ROLLOUT_VALIDATION_COMPLETION_RATIO_THRESHOLD",
              )}, "sampling": {"do_sample": True, "temperature": 0.6, "top_p": 0.95, "top_k": 20}}
    for raw_url in dict.fromkeys(urls):
        url = raw_url.rstrip("/")
        health, ready, deployment = (client.get(url, path) for path in ("/health", "/ready", "/tool_contract_info"))
        errors = []
        if not health.get("ok") or not ready.get("ok") or ready.get("docker_available") is not True:
            errors.append("health/readiness/Docker not verified")
        if deployment.get("source_fingerprint", {}).get("sha256") != local["sha256"]:
            errors.append("deployment source differs or is unavailable")
        if not persistent_search_verified(deployment.get("host_diagnostics", {})):
            errors.append("persistent rg is not verified")
        report["services"].append({"url": url, "health": health, "ready": ready,
                                   "deployment": deployment, "failures": errors})
    report["counts"] = {"services": len(report["services"]), "failures": sum(len(service["failures"]) for service in report["services"])}
    report["regression_pass"] = bool(report["services"]) and report["counts"]["failures"] == 0
    report["scope"] = "Deployment only. Functional online scenarios must pass separately after restart."
    write_json(output, report)
    return report


def file_sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def audit_mapped_path(raw: str, mappings: list[str]) -> Path:
    """Only explicit path-prefix mappings are allowed across clusters."""
    matches = []
    for mapping in mappings:
        source, separator, target = mapping.partition("=")
        if not separator or not Path(source).is_absolute() or not Path(target).is_absolute():
            raise ValueError("path map must be ABSOLUTE_SOURCE=ABSOLUTE_TARGET")
        if raw == source.rstrip("/") or raw.startswith(source.rstrip("/") + "/"):
            matches.append((len(source.rstrip("/")), target.rstrip("/") + raw[len(source.rstrip("/")):]))
    if matches:
        raw = max(matches)[1]
    return Path(raw).expanduser()


def audit_inventory(datasets: list[Path], mappings: list[str] | None = None, *, profile: str = "validation") -> tuple[dict, list[dict]]:
    import pyarrow.parquet as pq
    mappings = mappings or []
    manifest, cases = {"version": AUDIT_VERSION, "datasets": [], "cases": []}, []
    seen = set()
    for dataset in datasets:
        dataset_sha = file_sha256(dataset)
        manifest["datasets"].append({"path": str(dataset.resolve()), "sha256": dataset_sha})
        parquet = pq.ParquetFile(dataset)
        # Inventory only needs the task and reward columns. Batch iteration
        # keeps the 4 GiB training source out of a single Python object graph.
        index = 0
        for batch in parquet.iter_batches(batch_size=256, columns=["extra_info", "reward_model"]):
          for row in batch.to_pylist():
            item = {"id": digest([dataset_sha, index]), "dataset": str(dataset.resolve()), "row": index,
                    "dataset_sha256": dataset_sha, "issues": [], "artifacts": {}}
            if item["id"] in seen:
                raise ValueError("duplicate dataset input")
            seen.add(item["id"])
            task, gold, contract = {}, "", None
            try:
                task = json.loads(row["extra_info"]["tools_kwargs"]["__swe_task_json__"])
                from recipe.swe_agent.task_payload import with_official_test_contract
                task = with_official_test_contract(task, row.get("reward_model"))
                item.update(task_id=str(task["task_id"]), repo=task.get("repo"), base_sha=task.get("base_sha"),
                            docker_image=task.get("docker_image"), task_sha256=digest(task))
                if not task.get("base_sha") or not task.get("docker_image") or not task.get("test_commands"):
                    item["issues"].append("missing_execution_contract")
                paths = {"gold_patch": (task.get("artifact_paths") or {}).get("gold_patch"),
                         "hidden_test_patch": task.get("hidden_test_patch_path")}
                paths.update({"verifier:" + key: value for key, value in (task.get("verifier_file_paths") or {}).items()})
                for name, raw in paths.items():
                    if not raw:
                        if name == "gold_patch" or (name == "hidden_test_patch" and not task.get("hidden_test_patch")):
                            item["issues"].append("missing_" + name)
                        continue
                    path = audit_mapped_path(str(raw), mappings)
                    try:
                        data = path.read_bytes()
                    except OSError as exc:
                        item["issues"].append("unavailable_" + name + ":" + type(exc).__name__)
                        continue
                    item["artifacts"][name] = {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
                    if name == "gold_patch":
                        gold = data.decode("utf-8")
                        if not gold.strip():
                            item["issues"].append("empty_reference_patch")
                    elif name == "hidden_test_patch":
                        task["hidden_test_patch_path"] = str(path)
                    elif name.startswith("verifier:"):
                        task["verifier_file_paths"][name[len("verifier:"):]] = str(path)
                if profile == "rebench":
                    from recipe.swe_agent.test_results import validate_official_test_contract
                    truth = row.get("reward_model", {}).get("ground_truth", {})
                    truth = json.loads(truth) if isinstance(truth, str) else truth
                    if truth.get("benchmark") != "swe_rebench_v2" or truth.get("expected") != "all_official_tests_pass":
                        raise ValueError("rebench audit requires swe_rebench_v2 reference tests")
                    contract = dict(task.get("official_test_contract") or {
                        "version": 1, "fail_to_pass": truth.get("fail_to_pass"), "pass_to_pass": truth.get("pass_to_pass")})
                    validate_official_test_contract(contract)
                    raw_path = audit_mapped_path(task["artifact_paths"]["raw_example"], mappings)
                    raw_bytes = raw_path.read_bytes()
                    raw = json.loads(raw_bytes)
                    if raw.get("instance_id") != task["task_id"]:
                        raise ValueError("raw example task identity mismatch")
                    from recipe.swe_agent.test_results import rebench_label_issues
                    contract.update(rebench_scoring_version=1,
                                    log_parser=(raw.get("install_config") or {}).get("log_parser", "unknown"),
                                    identity_scope="namespace" if (raw.get("install_config") or {}).get("log_parser") == "parse_log_lein"
                                    else "suite" if contract["fail_to_pass"] == ["TestSuite"] and not contract["pass_to_pass"] else "test")
                    task["official_test_contract"] = contract
                    item["label_issues"] = rebench_label_issues(contract)
                    item["artifacts"]["raw_example"] = {"path": str(raw_path), "sha256": hashlib.sha256(raw_bytes).hexdigest(), "bytes": len(raw_bytes)}
                    item.update(language=raw.get("language", "unknown"),
                                log_parser=(raw.get("install_config") or {}).get("log_parser", "unknown"),
                                reference_contract_sha256=digest(contract),
                                required_tests=len(contract["fail_to_pass"]) + len(contract["pass_to_pass"]))
            except (KeyError, TypeError, ValueError) as exc:
                item["issues"].append("invalid_task:" + type(exc).__name__)
            except OSError as exc:
                item["issues"].append("unavailable_training_artifact:" + type(exc).__name__)
            item["eligible"] = not item["issues"]
            item["evaluator_sha256"] = digest({k: v for k, v in task.items() if k in {
                "test_commands", "evaluator_test_commands", "verifier_setup_commands", "hidden_test_patch",
                "trusted_hidden_test_patch", "verifier_files", "official_test_contract"}})
            manifest["cases"].append(item)
            cases.append({"inventory": item, "task": task, "gold": gold, "reference_contract": contract})
            index += 1
    manifest["counts"] = {"cases": len(cases), "eligible": sum(c["inventory"]["eligible"] for c in cases),
                          "repos": dict(collections.Counter(c["inventory"].get("repo", "unknown") for c in cases))}
    return manifest, cases


def audit_pilot_selection(cases: list[dict], count: int, seed: int) -> tuple[list[dict], dict]:
    """Cover language/parser strata, then allocate remaining slots proportionally."""
    groups = collections.defaultdict(list)
    for case in cases:
        item = case["inventory"]
        if item["eligible"]:
            groups[(item.get("language", "unknown"), item.get("log_parser", "unknown"))].append(case)
    if count < len(groups) or count > sum(map(len, groups.values())):
        raise ValueError("pilot size must cover every language/parser stratum and fit the eligible inventory")
    quotas = {key: 1 for key in groups}
    while sum(quotas.values()) < count:
        key = max((key for key in groups if quotas[key] < len(groups[key])),
                  key=lambda key: (len(groups[key]) / (quotas[key] + 1), key))
        quotas[key] += 1
    selected = []
    for key in sorted(groups):
        ordered = sorted(groups[key], key=lambda c: (c["inventory"].get("required_tests", 0),
                         c["inventory"]["artifacts"]["gold_patch"]["bytes"], digest([seed, c["inventory"]["id"]])))
        # Prefer repository diversity within the smaller half of each stratum.
        pool = ordered[:max(quotas[key], (len(ordered) + 1) // 2)]
        repos, chosen = set(), []
        for case in pool:
            if case["inventory"].get("repo") not in repos:
                chosen.append(case)
                repos.add(case["inventory"].get("repo"))
            if len(chosen) == quotas[key]:
                break
        chosen_ids = {c["inventory"]["id"] for c in chosen}
        chosen.extend(c for c in ordered if c["inventory"]["id"] not in chosen_ids)
        selected.extend(chosen[:quotas[key]])
    def distribution(items):
        return dict(sorted(collections.Counter(str(c["inventory"].get("language", "unknown")) + "/" +
                    str(c["inventory"].get("log_parser", "unknown")) for c in items).items()))
    selected.sort(key=lambda c: c["inventory"]["id"])
    return selected, {"algorithm": "language-parser-small-diverse-v1", "seed": seed, "count": count,
                      "population": distribution(cases), "pilot": distribution(selected),
                      "selection_sha256": digest([c["inventory"]["id"] for c in selected])}


def audit_rebench_task(task: dict, seconds: float) -> dict:
    from recipe.swe_agent.task_payload import evaluator_command
    payload = audit_task(task, seconds)
    if not payload.get("evaluator_test_commands"):
        # Run the original command in its own shell: the launch wrapper's set -e
        # must not change shell control flow inside a historical test command.
        from recipe.swe_agent.test_results import rebench_runner_command
        specs = [{"prepare": [], "run": rebench_runner_command(command), "source_command": command}
                 for command in task["test_commands"]]
        payload["evaluator_test_commands"] = specs
        payload["test_commands"] = [evaluator_command(spec) for spec in specs]
    return payload


def audit_rebench_outcomes(command: str, output: str, code: int, *, log_parser: str = "", identity_scope: str = "test") -> dict:
    """Exact identities only; summary-only and unsupported output cannot certify data."""
    from recipe.swe_agent.test_results import rebench_test_outcomes
    return rebench_test_outcomes(command, output, exit_code=code, log_parser=log_parser, identity_scope=identity_scope)


def audit_rebench_verify(arm: dict, case: dict, roots: list[str], output: Path, *, parser_policy: str = "strict") -> None:
    import gzip
    from scripts.diagnose_trajectories import collect_private_evaluator
    metadata = arm.get("reward", {}).get("metadata", {})
    arm["reference_evidence"] = {"complete": False, "reason": arm.get("classification", "runner_launch_unverified")}
    if (arm.get("exception") or not arm.get("claim", {}).get("ok") or not audit_response_ok(arm.get("base", {}))
            or (arm["mode"] == "gold" and not audit_response_ok(arm.get("patch", {})))
            or metadata.get("terminal_reward_timeout") or metadata.get("operation_hard_timeout")
            or metadata.get("hidden_failure_phase") not in {"none", "test_execution", "official_test_coverage"}
            or not arm.get("reward", {}).get("ok") or metadata.get("hidden_runner_provenance") != "launch_marker_v1"
            or not metadata.get("hidden_tests_started")):
        return
    collection = collect_private_evaluator({"task_id": case["inventory"]["task_id"], "reward_evidence": metadata}, roots, output)
    arm["private_evaluator"] = collection
    if collection.get("status") != "verified" or collection.get("logs_complete") is not True:
        arm["reference_evidence"]["reason"] = "private_logs_unverified:" + collection.get("status", "unknown")
        return
    path = Path(collection["private_copy"])
    phases = json.loads(path.read_text())["phases"]
    def phase_output(name):
        value = phases.get(name, {})
        ref = value.get("full_log")
        return gzip.decompress((path.parent / ref["file"]).read_bytes()).decode("utf-8") if ref else value.get("output", "")
    phase = phases.get("test_commands", {})
    ref = phase.get("full_log")
    text = gzip.decompress((path.parent / ref["file"]).read_bytes()).decode("utf-8") if ref else phase.get("output", "")
    # The service emits one random launch marker and one exit header per command.
    chunks = re.split(r"(?m)^exit_code: (-?\d+)\ncommand: ", text)
    specs = audit_rebench_task(case["task"], 1)["evaluator_test_commands"]
    commands = [spec.get("source_command", spec["run"]) for spec in specs]
    if len(chunks) != 1 + 2 * len(commands):
        arm["reference_evidence"]["reason"] = "incomplete_command_logs"
        return
    evidence = []
    native_evidence = []
    primary_evidence = []
    provenances, exit_codes, consistency_errors = [], [], []
    source_failure_candidate = False
    strict_failure = None
    strict_failure_code, strict_failure_output = None, ""
    contract = case["reference_contract"]
    for i, command in enumerate(commands):
        code, body = int(chunks[1 + 2 * i]), chunks[2 + 2 * i]
        if i + 1 < len(commands):
            if not body.endswith("\n\n"):
                arm["reference_evidence"]["reason"] = "incomplete_command_logs"
                return
            body = body[:-2]  # Service separator, not runner output.
        markers = re.findall(r"^SWE_RUNNER_LAUNCHED_[a-f0-9]+$", body, re.M)
        if len(markers) != 1:
            return
        from recipe.swe_agent.test_results import runner_output_after_marker, rebench_runner_provenance
        runner_output = runner_output_after_marker(body, markers[0])
        exit_codes.append(code)
        provenance = rebench_runner_provenance(contract, command, specs[i]["run"], runner_output, code)
        provenances.append(provenance)
        parsed = audit_rebench_outcomes(command, runner_output, code,
            log_parser=contract.get("log_parser", ""), identity_scope=contract.get("identity_scope", "test"))
        from recipe.swe_agent.test_results import native_rebench_test_outcomes
        native_evidence.append(native_rebench_test_outcomes(
            runner_output,
            log_parser=contract.get("log_parser", ""),
            exit_code=code,
        ))
        from recipe.swe_agent.test_results import rebench_primary_test_outcomes, rebench_source_failure
        primary_evidence.append(rebench_primary_test_outcomes(
            command, runner_output, exit_code=code, log_parser=contract.get("log_parser", ""),
            identity_scope=contract.get("identity_scope", "test")))
        if parser_policy == "native-guarded":
            try:
                retained = json.loads(phase_output(f"runner_evidence_{i}"))
                if retained.get("provenance") != provenance or retained.get("evidence") != primary_evidence[-1]:
                    consistency_errors.append(f"runner_{i}_input_or_evidence_mismatch")
            except (ValueError, TypeError):
                consistency_errors.append(f"runner_{i}_proof_missing")
        source_failure_candidate |= rebench_source_failure(code, runner_output)
        if not parsed["complete"]:
            if strict_failure is None:
                strict_failure = parsed
                strict_failure_code, strict_failure_output = code, runner_output
        else:
            evidence.append(parsed)
    from recipe.swe_agent.test_results import score_official_tests, native_rebench_candidate_score, rebench_label_issues
    native_score = native_rebench_candidate_score(contract, native_evidence, len(commands))
    native_details = {
        "commit": native_evidence[0].get("parser_commit") if native_evidence else None,
        "name": contract.get("log_parser"),
        # Native records remain explicitly observational. Launch and private-log
        # checks above establish command completeness; the parser supplies test
        # identities and outcomes.
        "expected_command_count": len(commands),
        "observed_command_count": sum(item.get("reason") == "native_observed_only" for item in native_evidence),
        "all_command_observations_available": (
            len(native_evidence) == len(commands)
            and all(item.get("reason") == "native_observed_only" for item in native_evidence)
        ),
        "complete_observations": len(native_evidence) == len(commands) and all(e.get("complete") for e in native_evidence),
        "candidate": native_score,
        "evidence": native_evidence,
    }
    if parser_policy == "native-guarded":
        score = score_official_tests(contract, primary_evidence, len(commands), execution_exit_codes=exit_codes)
        public_score = {key: value for key, value in score.items() if not key.startswith("_private_")}
        try:
            retained_score = json.loads(phase_output("official_tests"))
            if retained_score.get("score") != public_score:
                consistency_errors.append("official_score_mismatch")
        except (ValueError, TypeError):
            consistency_errors.append("official_score_proof_missing")
        if any(metadata.get(key) != value for key, value in public_score.items()):
            consistency_errors.append("reward_metadata_mismatch")
        if arm["reward"].get("score") != float(score["official_tests_success"]):
            consistency_errors.append("reward_value_mismatch")
        first_failure = next((item["reason"] for item in primary_evidence if not item["complete"]), None)
        arm["reference_evidence"] = {
            "complete": score["official_tests_complete"], "reason": first_failure or score["official_tests_reason"],
            "required": score["official_tests_required"], "missing": score["official_tests_missing"],
            "fail_to_pass_failed": score["official_fail_to_pass_failed"],
            "pass_to_pass_complete": score["official_pass_to_pass_complete"],
            "all_required_passed": score["official_tests_success"], "label_issues": rebench_label_issues(contract),
            "parser_policy": parser_policy, "strict_parser": "shadow", "native_parser": native_details,
            "strict_shadow": strict_failure or {"complete": True, "reason": "complete"},
            "parser_disagreement": any(item.get("shadow", {}).get("complete") and
                item.get("shadow", {}).get("outcomes") != item.get("outcomes") for item in primary_evidence),
            "baseline_source_failure_candidate": arm["mode"] == "baseline" and source_failure_candidate,
            "service_consistency": {"version": 1, "matched": not consistency_errors, "errors": consistency_errors,
                                    "runners": provenances, "score": public_score},
        }
        if consistency_errors:
            arm["reference_evidence"].update(complete=False, all_required_passed=False,
                baseline_source_failure_candidate=False, reason="audit_service_scoring_mismatch")
        return
    if strict_failure is not None:
        arm["reference_evidence"] = {"complete": False, "reason": strict_failure["reason"],
            "parser_policy": parser_policy, "strict_parser": "current", "native_parser": native_details,
            "parser_disagreement": native_score.get("candidate_all_required_passed") is True}
        arm["reference_evidence"]["baseline_source_failure_candidate"] = (
                arm["mode"] == "baseline" and strict_failure_code in {1, 2, 101} and not contract["pass_to_pass"] and
                re.search(r"(?im)(undefined:|unknown field |has no field|no field `|no function or associated item named|cannot import name|error\[E\d+\])", strict_failure_output) is not None
                and re.search(r"(?im)(command not found|no such host|no space left|structuredClone is not defined|no module named)", strict_failure_output) is None)
        return
    score = score_official_tests(contract, evidence, len(commands))
    arm["reference_evidence"] = {"complete": score["official_tests_complete"], "reason": score["official_tests_reason"],
        "required": score["official_tests_required"], "missing": score["official_tests_missing"],
        "fail_to_pass_failed": score["official_fail_to_pass_failed"],
        "pass_to_pass_complete": score["official_pass_to_pass_complete"],
        "all_required_passed": score["official_tests_success"], "label_issues": rebench_label_issues(contract),
        "parser_policy": parser_policy, "strict_parser": "current"}
    arm["reference_evidence"].update({
        "native_parser": native_details,
        "parser_disagreement": native_score.get("candidate_all_required_passed") != score["official_tests_success"],
    })


def audit_rebench_status(result: dict) -> str:
    if not result.get("complete"):
        return "blocked:cleanup_unverified"
    if result.get("data_issues") or result.get("image_unavailable") or result.get("endpoint_quarantined"):
        return audit_result_status(result)
    arms = result.get("arms", {})
    if set(arms) != {"baseline", "gold"} or any(not a.get("cleanup", {}).get("verified") for a in arms.values()):
        return "review:incomplete_or_unclean_execution"
    baseline, gold = arms["baseline"], arms["gold"]
    if any(a.get("reference_evidence", {}).get("service_consistency", {}).get("matched") is False
           for a in arms.values()):
        return "review:audit_service_scoring_mismatch"
    if (gold.get("reference_evidence", {}).get("complete") and gold["reference_evidence"].get("all_required_passed")
            and gold["reward"].get("score") == 1 and not baseline.get("reference_evidence", {}).get("complete")
            and baseline.get("reference_evidence", {}).get("baseline_source_failure_candidate")):
        return "review:gold_verified_baseline_source_failure"
    for mode in ("gold", "baseline"):
        arm = arms[mode]
        evidence = arm.get("reference_evidence", {})
        if not evidence.get("complete"):
            return "review:" + mode + ":" + evidence.get("reason", arm.get("classification", "unknown"))
    baseline, gold = arms["baseline"], arms["gold"]
    if gold["reward"].get("score") != 1 or not gold["reference_evidence"]["all_required_passed"]:
        return "review:reference_tests_failed_or_reward_conflict"
    if baseline["reward"].get("score") == 1:
        return "baseline_passed_review"
    evidence = baseline["reference_evidence"]
    if baseline["reward"].get("score") == 0 and evidence["fail_to_pass_failed"] > 0 and evidence["pass_to_pass_complete"]:
        return "reference_verified"
    return "baseline_required_tests_review"


def native_reference_candidate(result: dict) -> dict:
    """Report a native-parser candidate without granting it certification authority."""
    report = {
        "authority": "audit_only_native_observed",
        "eligible": False,
        "reason": "unknown",
        "parser": None,
    }
    if result.get("complete") is not True:
        report["reason"] = "audit_incomplete"
        return report
    arms = result.get("arms") or {}
    if set(arms) != {"baseline", "gold"}:
        report["reason"] = "missing_audit_arms"
        return report
    if any(arm.get("cleanup", {}).get("verified") is not True for arm in arms.values()):
        report["reason"] = "cleanup_unverified"
        return report
    evidence = {mode: arms[mode].get("reference_evidence") or {} for mode in arms}
    native = {mode: evidence[mode].get("native_parser") or {} for mode in arms}
    report["parser"] = native["gold"].get("name") or native["baseline"].get("name")
    for mode in ("baseline", "gold"):
        if native[mode].get("all_command_observations_available") is not True:
            report["reason"] = mode + "_native_observations_incomplete"
            return report
        if native[mode].get("candidate", {}).get("authority") != "native_observed_only":
            report["reason"] = mode + "_native_authority_missing"
            return report
        metadata = arms[mode].get("reward", {}).get("metadata") or {}
        if (metadata.get("hidden_runner_provenance") != "launch_marker_v1"
                or metadata.get("hidden_tests_started") is not True):
            report["reason"] = mode + "_runner_unverified"
            return report
        # Native output can name a test while the runner itself failed before
        # that test's outcome was established. Keep these out of candidates;
        # unsupported syntax/identity handling remains reportable below.
        if evidence[mode].get("reason") in {
            "ambiguous_errors", "test_setup_or_collection_error", "incomplete_tests",
            "execution_error", "incomplete_command_logs",
        } and evidence[mode].get("parser_policy") != "native-guarded":
            report["reason"] = mode + "_strict_execution_ambiguous"
            return report
    gold_score = native["gold"]["candidate"]
    baseline_score = native["baseline"]["candidate"]
    if gold_score.get("candidate_all_required_passed") is not True:
        report["reason"] = "gold_native_required_tests_not_all_passed"
    elif baseline_score.get("official_tests_complete") is not True:
        report["reason"] = "baseline_native_required_tests_incomplete"
    elif baseline_score.get("official_fail_to_pass_failed", 0) <= 0:
        report["reason"] = "baseline_native_fail_to_pass_not_failed"
    elif baseline_score.get("official_pass_to_pass_complete") is not True:
        report["reason"] = "baseline_native_pass_to_pass_incomplete"
    else:
        report.update(eligible=True, reason="native_reference_transition_observed")
    return report


def native_reference_candidate_summary(results: list[dict]) -> dict:
    """Aggregate audit-only native evidence for a report, never for filtering."""
    candidates = [result.get("native_reference_candidate") or native_reference_candidate(result) for result in results]
    strict_vs_native = collections.Counter()
    parser_counts = collections.Counter()
    for result, candidate in zip(results, candidates):
        arms = result.get("arms") or {}
        gold_reason = (arms.get("gold", {}).get("reference_evidence") or {}).get("reason", "missing_gold_evidence")
        strict_vs_native[(gold_reason, candidate.get("reason", "unknown"))] += 1
        parser_counts[str(candidate.get("parser") or "unknown")] += 1
    return {
        "version": 1,
        "authority": "audit_only_native_observed",
        "case_count": len(results),
        "candidate_count": sum(candidate.get("eligible") is True for candidate in candidates),
        "by_parser": dict(sorted(parser_counts.items())),
        "strict_gold_reason_vs_native_candidate": {
            strict_reason + "|" + candidate_reason: count
            for (strict_reason, candidate_reason), count in sorted(strict_vs_native.items())
        },
    }


def audit_task(task: dict, deadline_seconds: float) -> dict:
    from recipe.swe_agent.task_payload import reward_task_payload
    task = reward_task_payload(task, hard_timeout_seconds=deadline_seconds)
    # Never pull/build images during an environment audit.
    task["docker_pull"] = False
    if task.get("dockerfile") or task.get("docker_build") or task.get("docker_build_spec"):
        raise ValueError("image builds are not allowed in the reference audit")
    image = task["docker_image"]
    task["case_image"] = image
    task["environment"] = {**(task.get("environment") or {}), "base_image": image, "case_image": image}
    return task


def audit_response_ok(response: dict) -> bool:
    return response.get("ok") is True and re.match(r"^exit_code: 0(?:\n|$)", response.get("text", "")) is not None


def audit_classify(arm: dict) -> str:
    if arm.get("exception"):
        if "Committed case image is missing from this Docker daemon:" in arm["exception"]:
            return "image_unavailable"
        return arm.get("failure_kind", "inconclusive")
    if not arm.get("claim", {}).get("ok"):
        return "admission_or_transport_failure"
    if not audit_response_ok(arm.get("base", {})):
        return "base_revision_or_workspace_mismatch"
    if arm["mode"] == "gold" and not audit_response_ok(arm.get("patch", {})):
        return "reference_patch_mismatch"
    reward = arm.get("reward", {})
    metadata = reward.get("metadata") or {}
    if metadata.get("terminal_reward_timeout") or metadata.get("operation_hard_timeout") or metadata.get("failure_reason") == "silent_command_timeout":
        return "evaluation_timeout"
    if not reward.get("ok"):
        return "reward_transport_or_service_failure"
    phase = metadata.get("hidden_failure_phase")
    if phase in {"workspace_prepare", "verifier_staging", "verifier_setup", "patch_check", "patch_apply", "evaluator_preparation"}:
        return phase + "_failure"
    if not metadata.get("hidden_tests_started") or metadata.get("hidden_runner_provenance") != "launch_marker_v1":
        return "runner_launch_unverified"
    if arm.get("official_test_contract_required") and metadata.get("official_test_scoring_version") != 1:
        return "official_scoring_unavailable"
    if metadata.get("official_test_scoring_version"):
        if metadata.get("official_tests_complete") is not True:
            return "official_test_coverage_unknown"
        if metadata.get("official_tests_success") is True and reward.get("score") == 1.0:
            return "tests_passed"
        if metadata.get("official_tests_failed", 0) > 0 and reward.get("score") == 0.0:
            return "required_tests_failed"
        return "scoring_evidence_conflict"
    if metadata.get("hidden_test_counts_complete") is not True or metadata.get("hidden_tests_total_count", -1) <= 0:
        return "test_coverage_unknown"
    if reward.get("score") == 1.0:
        return "tests_passed"
    return "tests_failed_or_collection_error"


def audit_arm(client: Any, url: str, task: dict, gold: str, mode: str, seconds: float, checkpoint) -> dict:
    from recipe.swe_agent.task_payload import rollout_task_payload
    request = "reference-audit-" + uuid.uuid4().hex
    public = rollout_task_payload(task, hard_timeout_seconds=seconds)
    payload = {"request_id": request, "instance_id": request, "task": public}
    arm = {"mode": mode, "request_id": request, "endpoint": url, "phase": "created",
           "official_test_contract_required": task.get("official_test_contract") is not None}
    deadline = time.monotonic() + seconds
    def record(name, value):
        arm[name], arm["phase"] = value, name
        checkpoint(arm)
        return value
    try:
        checkpoint(arm)
        claim = record("claim", client.post(url, "/claim", payload, deadline))
        if claim.get("ok"):
            # Confirm HEAD and a clean checkout before any reference mutation.
            command = ("expected=" + shlex.quote(str(task["base_sha"])) + '\nswe_git_error=$(mktemp)\n'
                       'trap \'rm -f "$swe_git_error"\' EXIT\n'
                       'actual=$(git rev-parse HEAD 2>>"$swe_git_error"); a=$?\n'
                       'resolved=$(git rev-parse "$expected^{commit}" 2>>"$swe_git_error"); b=$?\n'
                       'tracked=$(git status --porcelain --untracked-files=no 2>>"$swe_git_error"); c=$?\n'
                       'untracked=$(git ls-files --others --exclude-standard 2>>"$swe_git_error"); d=$?\n'
                       'printf "SWE_BASE_CWD\\n%s\\nSWE_BASE_EXPECTED\\n%s\\nSWE_BASE_RESOLVED\\n%s\\nSWE_BASE_ACTUAL\\n%s\\n" "$PWD" "$expected" "$resolved" "$actual"\n'
                       'printf "SWE_BASE_TRACKED\\n%s\\nSWE_BASE_UNTRACKED\\n%s\\nSWE_BASE_GIT_ERRORS\\n" "$tracked" "$untracked"\n'
                       'cat "$swe_git_error"\nprintf "SWE_BASE_END\\n"\n'
                       'test "$a$b$c$d" = 0000 && test "$actual" = "$resolved" && test -z "$tracked" || exit 1\n'
                       'printf "BASE_OK %s\\n" "$actual"')
            base = record("base", client.post(url, "/execute", {**payload, "tool": "Bash", "parameters": {"command": command}}, deadline))
            arm["workspace_diagnostics"] = {m[1].lower(): m[2].strip() for m in re.finditer(
                r"^SWE_BASE_(CWD|EXPECTED|RESOLVED|ACTUAL|TRACKED|UNTRACKED|GIT_ERRORS)\n(.*?)(?=^SWE_BASE_)",
                base.get("text", ""), re.M | re.S)}
            checkpoint(arm)
            if audit_response_ok(base):
                patched = mode == "baseline"
                if mode == "gold":
                    encoded = base64.b64encode(gold.encode()).decode()
                    command = ("set -e\np=$(mktemp /tmp/swe-reference-XXXXXX.patch)\ntrap 'rm -f \"$p\"' EXIT\n"
                               + "printf %s " + shlex.quote(encoded) + ' | base64 -d > "$p"\n'
                               + 'git apply --check "$p"\ngit apply --index "$p"\ngit diff --cached --quiet && exit 42\n'
                               + 'git diff --cached --stat\nprintf "REFERENCE_APPLIED\\n"')
                    patch = record("patch", client.post(url, "/execute", {**payload, "tool": "Bash", "parameters": {"command": command}}, deadline))
                    patched = audit_response_ok(patch)
                if patched:
                    record("reward", client.post(url, "/reward", {**payload, "task": task}, deadline))
    except Exception as exc:
        arm["exception"] = f"{type(exc).__name__}: {exc}"[:2000]
        arm["failure_kind"] = "transport_failure" if isinstance(exc, (ConnectionError, urllib.error.URLError)) else "inconclusive"
    finally:
        try:
            release = record("release", client.post(url, "/release", {**payload, "force": True}, time.monotonic() + 45))
            metadata = {**(arm.get("claim", {}).get("metadata") or {}), **(arm.get("base", {}).get("metadata") or {})}
            query = {"key": request + ":" + task["task_id"], "workspace_path": metadata.get("workspace", ""),
                     "cleanup_path": release.get("workspace_cleanup_path") or "",
                     "container_name": metadata.get("container_name", "")}
            cleanup_deadline = time.monotonic() + 15
            while True:
                state = client.get(url, "/tool_contract_info?" + urllib.parse.urlencode(query))
                sessions, filesystem = state.get("session_state"), state.get("cleanup_state") or {}
                verified = isinstance(sessions, dict) and not any(sessions.values()) and filesystem.get("complete") is True
                if verified or time.monotonic() >= cleanup_deadline:
                    break
                time.sleep(.25)
            record("cleanup", {"verified": bool(verified), "sessions": sessions, "filesystem": filesystem})
        except Exception as exc:
            arm["cleanup"] = {"verified": False, "error": f"{type(exc).__name__}: {exc}"[:1000]}
        arm["classification"] = audit_classify(arm)
        arm["phase"] = "complete"
        checkpoint(arm)
    return arm


def audit_result_status(result: dict) -> str:
    if result.get("image_unavailable"):
        return "image_unavailable"
    if result.get("endpoint_quarantined"):
        return "endpoint_quarantined"
    if result.get("data_issues"):
        return "unavailable_data"
    arms = result.get("arms") or {}
    if not all(arm.get("cleanup", {}).get("verified") for arm in arms.values()):
        return "cleanup_unverified"
    if set(arms) != {"baseline", "gold"}:
        return "interrupted"
    gold = arms["gold"]["classification"]
    baseline = arms["baseline"]["classification"]
    if gold == "tests_passed" and baseline == "tests_passed":
        return "baseline_passed_review"
    if gold == "tests_passed" and baseline == "tests_failed_or_collection_error":
        return "reference_verified"
    if gold == "tests_passed" and baseline == "required_tests_failed":
        metadata = arms["baseline"]["reward"]["metadata"]
        if metadata.get("official_fail_to_pass_failed", 0) > 0 and metadata.get("official_pass_to_pass_complete") is True:
            return "reference_verified"
        return "baseline_required_tests_review"
    if gold == "tests_passed":
        return "baseline_evidence_review:" + baseline
    return "review:" + gold


def reference_audit(datasets: list[Path], urls: list[str], output: Path, *, mappings: list[str] | None = None,
                    task_ids: list[str] | None = None, limit: int = 0, seconds: float = 900,
                    resume: bool = False, client: Any = None, workers_per_endpoint: int | None = None,
                    profile: str = "validation", pilot_size: int = 0, seed: int = 0,
                    evaluator_roots: list[str] | None = None, image_policy: str = "cache-only",
                    image_prepare_seconds: float = 1800, image_prepare_parallelism: int = 1,
                    parser_policy: str = "strict", transport_retries: int = 1) -> dict:
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (output / ".audit.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("another audit is using this output directory") from exc
        return _reference_audit(datasets, urls, output, mappings=mappings, task_ids=task_ids,
                                limit=limit, seconds=seconds, resume=resume, client=client,
                                workers_per_endpoint=workers_per_endpoint, profile=profile,
                                pilot_size=pilot_size, seed=seed, evaluator_roots=evaluator_roots,
                                image_policy=image_policy, image_prepare_seconds=image_prepare_seconds,
                                image_prepare_parallelism=image_prepare_parallelism, parser_policy=parser_policy,
                                transport_retries=transport_retries)


def audit_write_json(path: Path, value: dict) -> None:
    """An interrupted write must not destroy a resumable audit checkpoint."""
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def audit_recover_arm(client: Any, task: dict, arm: dict, *, preserve_completed: bool = False) -> dict:
    """Cleanup only; never replay an uncertain remote test or patch command."""
    completed = preserve_completed and arm.get("phase") == "complete"
    recovered = {**arm, **({"recovered_cleanup": True} if completed else {"recovered_interruption": True})}
    url, request = arm["endpoint"], arm["request_id"]
    try:
        from recipe.swe_agent.task_payload import rollout_task_payload
        release = client.post(url, "/release", {"request_id": request, "instance_id": request,
            "task": rollout_task_payload(task), "force": True}, time.monotonic() + 45)
        if not release.get("workspace_cleanup_path") and arm.get("release", {}).get("workspace_cleanup_path"):
            release = {**release, "workspace_cleanup_path": arm["release"]["workspace_cleanup_path"]}
        metadata = {**(arm.get("claim", {}).get("metadata") or {}), **(arm.get("base", {}).get("metadata") or {})}
        query = {"key": request + ":" + task["task_id"], "workspace_path": metadata.get("workspace", ""),
                 "cleanup_path": release.get("workspace_cleanup_path") or "", "container_name": metadata.get("container_name", "")}
        deadline = time.monotonic() + 15
        while True:
            state = client.get(url, "/tool_contract_info?" + urllib.parse.urlencode(query))
            sessions, filesystem = state.get("session_state"), state.get("cleanup_state") or {}
            verified = isinstance(sessions, dict) and not any(sessions.values()) and filesystem.get("complete") is True
            if verified or time.monotonic() >= deadline:
                break
            time.sleep(.25)
        recovered.update(release=release, cleanup={"verified": verified, "sessions": sessions, "filesystem": filesystem})
    except Exception as exc:
        recovered["cleanup"] = {"verified": False, "error": f"{type(exc).__name__}: {exc}"[:1000]}
    if recovered["cleanup"]["verified"] and not completed:
        recovered.update(phase="complete", classification="interrupted_recovered")
    return recovered


def audit_result_record(item: dict, result: dict, signature: str, source_match: bool,
                        audit_source: dict | None = None) -> dict:
    arms = result.get("arms", {})
    return {"version": "swe-reference-results-v1", "id": item["id"], "dataset_sha256": item["dataset_sha256"],
            "row": item["row"], "task_id": item.get("task_id"), "repo": item.get("repo"),
            "signature": signature, "source_match": source_match, "complete": result.get("complete", False),
            "status": result.get("status", "not_audited"), "data_issues": result.get("data_issues", []),
            "image_preparation": result.get("image_preparation"),
            "verification_context": {**(audit_source or {}),
                "task_sha256": item.get("task_sha256"),
                "evaluator_sha256": item.get("evaluator_sha256"),
                "reference_contract_sha256": item.get("reference_contract_sha256"),
                "artifact_sha256": {name: artifact["sha256"] for name, artifact in item.get("artifacts", {}).items()},
                "image_id": result.get("cached_image_id")},
            "image_availability_scope": "local_docker_cache" if result.get("image_unavailable") else None,
            "native_reference_candidate": result.get("native_reference_candidate"),
            "arms": {mode: {"classification": arm.get("classification"), "score": arm.get("reward", {}).get("score"),
                      "cleanup_verified": arm.get("cleanup", {}).get("verified", False),
                      "reference_evidence": arm.get("reference_evidence", {}),
                      "private_evaluator": arm.get("private_evaluator", {}), "endpoint": arm.get("endpoint"),
                      "timings": {name: arm.get("reward", {}).get("metadata", {})[name]
                          for name in ("operation_queue_wait_seconds", "operation_end_to_end_seconds")
                          if name in arm.get("reward", {}).get("metadata", {})},
                      "request_id": arm.get("request_id")} for mode, arm in arms.items()}}


class AuditProgress:
    def __init__(self, output: Path, cases: list[dict], signature: str, source_match: bool):
        from tqdm import tqdm
        self.output, self.signature, self.source_match = output, signature, source_match
        self.audit_source = {
            "service_source_sha256": importlib.import_module("recipe.swe_agent.tool_contract_info").source_fingerprint()["sha256"],
            "audit_code_sha256": file_sha256(Path(__file__)),
        }
        self.items = {c["inventory"]["id"]: c["inventory"] for c in cases}
        self.records, self.running = {}, 0
        self.lock = threading.Lock()
        for identity in self.items:
            path = output / "cases" / identity / "result.json"
            if path.exists():
                result = json.loads(path.read_text())
                self.records[identity] = audit_result_record(self.items[identity], result, signature, source_match, self.audit_source)
        # Rebuild from authoritative atomic checkpoints, repairing a crash between
        # result persistence and ledger append (or a truncated final JSONL line).
        ledger = output / "results.jsonl"
        with tempfile.NamedTemporaryFile(mode="w", dir=output, delete=False) as stream:
            temporary = Path(stream.name)
            try:
                for record in self.records.values():
                    stream.write(json.dumps(record, sort_keys=True) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                os.replace(temporary, ledger)
            finally:
                temporary.unlink(missing_ok=True)
        completed = sum(r["complete"] for r in self.records.values())
        self.bar = tqdm(total=len(cases), initial=completed, desc="Reference audit", unit="case", file=sys.stderr)
        self.publish()

    def publish(self):
        completed = sum(r["complete"] for r in self.records.values())
        verified = sum(r["complete"] and r["status"] == "reference_verified" for r in self.records.values())
        review = sum(r["complete"] and (r["status"].startswith("review:") or "review" in r["status"]) for r in self.records.values())
        snapshot = {"total": len(self.items), "completed": completed, "remaining": len(self.items) - completed,
                    "running": self.running, "verified": verified, "failed": completed - verified,
                    "review": review, "blocked": sum(not r["complete"] for r in self.records.values()),
                    "signature": self.signature, "updated_at_unix": time.time()}
        audit_write_json(self.output / "progress.json", snapshot)
        self.bar.n = completed
        self.bar.set_postfix({k: snapshot[k] for k in ("completed", "remaining", "running", "failed", "review", "blocked")})

    def start(self):
        with self.lock:
            self.running += 1
            self.publish()

    def finish(self, result: dict):
        with self.lock:
            record = audit_result_record(self.items[result["id"]], result, self.signature, self.source_match, self.audit_source)
            self.records[result["id"]] = record
            with (self.output / "results.jsonl").open("a") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            self.running -= 1
            self.publish()

    def close(self):
        self.bar.close()


def _reference_audit(datasets: list[Path], urls: list[str], output: Path, *, mappings: list[str] | None = None,
                     task_ids: list[str] | None = None, limit: int = 0, seconds: float = 900,
                     resume: bool = False, client: Any = None, workers_per_endpoint: int | None = None,
                     profile: str = "validation", pilot_size: int = 0, seed: int = 0,
                     evaluator_roots: list[str] | None = None, image_policy: str = "cache-only",
                     image_prepare_seconds: float = 1800, image_prepare_parallelism: int = 1,
                     parser_policy: str = "strict", transport_retries: int = 1) -> dict:
    from recipe.swe_agent.image_archives import canonical_image_reference
    client = client or Client()
    if type(transport_retries) is not int or transport_retries not in (0, 1):
        raise ValueError("transport_retries must be 0 or 1")
    if workers_per_endpoint is None:
        workers_per_endpoint = int(os.environ.get("SWE_AGENT_REFERENCE_AUDIT_WORKERS_PER_ENDPOINT", "2"))
    if type(workers_per_endpoint) is not int or workers_per_endpoint < 1:
        raise ValueError("workers_per_endpoint must be a positive integer")
    if not urls or len(urls) != len(set(urls)) or seconds <= 0 or limit < 0:
        raise ValueError("unique endpoints and positive deadline required")
    if profile not in {"validation", "rebench"} or pilot_size < 0:
        raise ValueError("invalid audit profile or pilot size")
    if parser_policy not in {"strict", "native-first-shadow", "native-guarded"}:
        raise ValueError("unsupported ReBench parser policy")
    if (image_policy not in {"cache-only", "archive-or-pull"}
            or not 0 < image_prepare_seconds <= 7200 or type(image_prepare_parallelism) is not int
            or image_prepare_parallelism < 1 or (profile != "rebench" and image_policy != "cache-only")):
        raise ValueError("invalid image preparation settings; provisioning requires rebench profile")
    if pilot_size and (profile != "rebench" or task_ids or limit):
        raise ValueError("--pilot-size requires rebench profile and cannot combine with --limit or --task-id")
    if profile == "rebench" and not evaluator_roots:
        raise ValueError("rebench profile requires explicit --evaluator-root SERVICE_ROOT=LOCAL_ROOT")
    for mapping in evaluator_roots or []:
        source, separator, destination = mapping.partition("=")
        if not separator or not Path(source).is_absolute() or not Path(destination).is_absolute():
            raise ValueError("evaluator roots must be ABSOLUTE_SERVICE_ROOT=ABSOLUTE_LOCAL_ROOT")
    manifest, cases = audit_inventory(datasets, mappings, profile=profile)
    if profile == "rebench" and parser_policy == "native-guarded":
        from recipe.swe_agent.test_results import rebench_contract_fields, official_evidence_sha256
        for case in cases:
            contract = case.get("reference_contract")
            if contract is None:
                continue
            contract.update(rebench_contract_fields(contract["log_parser"], contract["fail_to_pass"], contract["pass_to_pass"]))
            case["task"]["official_test_contract"] = contract
            case["inventory"]["reference_contract_sha256"] = official_evidence_sha256(contract)
            case["inventory"]["task_sha256"] = digest(case["task"])
            case["inventory"]["evaluator_sha256"] = digest({k: v for k, v in case["task"].items() if k in {
                "test_commands", "evaluator_test_commands", "verifier_setup_commands", "hidden_test_patch",
                "trusted_hidden_test_patch", "verifier_files", "official_test_contract"}})
    selection = None
    if pilot_size:
        cases, selection = audit_pilot_selection(cases, pilot_size, seed)
    if task_ids:
        missing = set(task_ids) - {c["inventory"].get("task_id") for c in cases}
        if missing:
            raise ValueError("task IDs not in datasets: " + ",".join(sorted(missing)))
        cases = [c for c in cases if c["inventory"].get("task_id") in task_ids]
    if limit:
        cases = cases[:limit]
    local = importlib.import_module("recipe.swe_agent.tool_contract_info").source_fingerprint()
    services = [{"url": url, "health": client.get(url, "/health"), "deployment": client.get(url, "/tool_contract_info")} for url in urls]
    source_match = all(s["deployment"].get("source_fingerprint", {}).get("sha256") == local["sha256"] for s in services)
    if profile == "rebench" and (not source_match or any(s["health"].get("ok") is not True for s in services)):
        raise ValueError("training audit requires healthy services with matching deployed sources")
    cached = {s["url"]: {i["tag"]: i["id"] for i in s["deployment"].get("cached_images", [])} for s in services}
    for cache in cached.values():
        equivalents = {canonical_image_reference(tag): image_id for tag, image_id in cache.items()}
        for case in cases:
            image = case["inventory"].get("docker_image")
            if image and canonical_image_reference(image) in equivalents:
                cache[image] = equivalents[canonical_image_reference(image)]
    selected_images = {c["inventory"].get("docker_image") for c in cases}
    availability = {s["url"]: {} for s in services}
    for service in services:
        url = service["url"]
        for image in sorted(selected_images - {None}):
            if image in cached[url]:
                availability[url][image] = True
            elif service["deployment"].get("image_inventory_complete") is True:
                availability[url][image] = False
            elif profile == "rebench":
                # ReBench prepares an assigned image remotely. An incomplete
                # cache listing is a scheduling hint, never a reason to inspect
                # the full dataset on every daemon before dispatching work.
                availability[url][image] = None
            else:
                probe = client.get(url, "/tool_contract_info?" + urllib.parse.urlencode({"image": image})).get("image_probe", {})
                known = probe.get("available") if probe.get("tag") == image else None
                availability[url][image] = known if type(known) is bool else None
                if known is True and probe.get("id"):
                    cached[url][image] = probe["id"]
    identity = {"manifest": manifest, "selection": [c["inventory"]["id"] for c in cases], "seconds": seconds,
                "workers_per_endpoint": workers_per_endpoint,
                "services": [{"url": s["url"], "source": s["deployment"].get("source_fingerprint"),
                              "images": {tag: image_id for tag, image_id in cached[s["url"]].items() if tag in selected_images},
                              "image_availability": availability[s["url"]]}
                             for s in services],
                "source": local, "audit_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "path_maps": mappings or []}
    if profile == "rebench":
        identity.pop("workers_per_endpoint")
        # Cache contents change during pulls and between restarts; they are evidence,
        # not immutable resume inputs. Image names remain in the dataset manifest.
        for service in identity["services"]:
            service.pop("images")
            service.pop("image_availability")
        identity.update(profile=profile, pilot=selection, evaluator_roots=evaluator_roots, parser_policy=parser_policy,
                        image_policy=image_policy, image_prepare_seconds=image_prepare_seconds)
    if transport_retries != 1:
        identity["transport_retries"] = transport_retries
    signature = digest(identity)
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        if not resume:
            raise ValueError("audit directory already exists; use --resume")
        if json.loads(manifest_path.read_text())["signature"] != signature:
            raise ValueError("resume signature differs: inputs, code, settings or deployed services changed")
    else:
        audit_write_json(manifest_path, {"version": AUDIT_VERSION, "signature": signature, "identity": identity, "services": services})
    pending = []
    quarantined = set()
    scheduling_lock = threading.Lock()
    preparation_slots = {url: threading.BoundedSemaphore(image_prepare_parallelism) for url in urls}
    preparation_results = {}
    for case in cases:
        image = case["inventory"].get("docker_image")
        candidates = [url for url in urls if image in cached[url]]
        if not candidates:
            candidates = [url for url in urls if availability[url].get(image) is not False]
        if not candidates and image_policy == "archive-or-pull":
            candidates = urls
        case["image_unavailable"] = not candidates
        case["candidates"] = candidates or urls
        # Inspect ALL checkpoints before admitting any new sessions. A late
        # checkpoint must not allow earlier queue items onto an unsafe endpoint.
        case_dir = output / "cases" / case["inventory"]["id"]
        checkpoints = list(case_dir.glob("*/*.json"))
        result_path = case_dir / "result.json"
        previous_result = json.loads(result_path.read_text()) if result_path.exists() else {}
        if profile == "rebench" and not previous_result.get("complete"):
            for path in checkpoints:
                arm = json.loads(path.read_text())
                if arm.get("phase") != "complete" or not arm.get("cleanup", {}).get("verified"):
                    arm = audit_recover_arm(client, case["task"], arm)
                    audit_write_json(path, arm)
        arms = [json.loads(p.read_text()) for p in checkpoints]
        if profile == "validation" or previous_result.get("complete"):
            arms += list(previous_result.get("arms", {}).values())
        for arm in arms:
            if arm.get("phase") != "complete" or not arm.get("cleanup", {}).get("verified"):
                quarantined.update([arm["endpoint"]] if arm.get("endpoint") in urls else urls)
        if profile == "validation" and any(issue.startswith("interrupted_attempt_requires_cleanup_review:") for issue in previous_result.get("data_issues", [])):
            quarantined.update(urls)
        if not previous_result.get("complete"):
            pending.append(case)
    pending.sort(key=lambda case: len(case["candidates"]))
    progress = AuditProgress(output, cases, signature, source_match) if profile == "rebench" else None

    def process(case, url):
        if progress:
            progress.start()
        inventory = case["inventory"]
        case_dir = output / "cases" / inventory["id"]
        case_dir.mkdir(parents=True, exist_ok=True)
        result_path = case_dir / "result.json"
        result = {"id": inventory["id"], "task_id": inventory.get("task_id"), "signature": signature, "arms": {},
                  "cached_image_id": cached[url].get(inventory.get("docker_image"))}
        if inventory["issues"]:
            result["data_issues"] = inventory["issues"]
        elif case["image_unavailable"]:
            result["image_unavailable"] = True
        elif url in quarantined and not any(
            (a.get("phase") != "complete" or not a.get("cleanup", {}).get("verified"))
            for a in [json.loads(p.read_text()) for p in case_dir.glob("*/*.json")]
        ):
            result["endpoint_quarantined"] = True
        else:
            try:
                if image_policy == "archive-or-pull":
                    image = inventory["docker_image"]
                    with preparation_slots[url]:
                        key = (url, canonical_image_reference(image))
                        prepared = preparation_results.get(key)
                        if prepared is None or not prepared.get("ok"):
                            prepared = client.post(url, "/audit_prepare_image",
                                {"image": image, "timeout_seconds": image_prepare_seconds},
                                time.monotonic() + image_prepare_seconds + 10)
                            preparation_results[key] = prepared
                        result["image_preparation"] = prepared
                        audit_write_json(case_dir / "image_preparation.json", prepared)
                    if prepared.get("ok") is not True or not prepared.get("id"):
                        result["image_preparation_failed"] = True
                        result["status"] = "review:image:" + prepared.get("failure_kind", "image_preparation_unknown")
                        result["complete"] = True
                        audit_write_json(result_path, result)
                        if progress:
                            progress.finish(result)
                        return
                    result["cached_image_id"] = prepared["id"]
                task = audit_rebench_task(case["task"], seconds) if profile == "rebench" else audit_task(case["task"], seconds)
                for mode in ("baseline", "gold"):
                    # Persist each attempt independently, including interrupted attempts.
                    attempts = case_dir / mode
                    attempts.mkdir(exist_ok=True)
                    previous = sorted(attempts.glob("*.json"), key=lambda p: int(p.stem))
                    latest = json.loads(previous[-1].read_text()) if previous else None
                    if latest and latest.get("phase") == "complete" and not latest.get("recovered_interruption"):
                        result["arms"][mode] = latest
                        if profile == "rebench" and latest.get("cleanup", {}).get("verified") and "reference_evidence" not in latest:
                            audit_rebench_verify(latest, case, evaluator_roots, case_dir / mode, parser_policy=parser_policy)
                            audit_write_json(previous[-1], latest)
                        if not latest.get("cleanup", {}).get("verified"):
                            with scheduling_lock:
                                quarantined.add(latest.get("endpoint", url))
                            break
                        continue
                    if previous and not (profile == "rebench" and latest.get("recovered_interruption") and latest.get("cleanup", {}).get("verified")):
                        result["data_issues"] = ["interrupted_attempt_requires_cleanup_review:" + mode]
                        with scheduling_lock:
                            quarantined.add(url)
                        break
                    with scheduling_lock:
                        blocked = url in quarantined
                    if blocked:
                        result["endpoint_quarantined"] = True
                        break
                    for retry in range(transport_retries + 1):
                        with scheduling_lock:
                            blocked = url in quarantined
                        if blocked:
                            result["endpoint_quarantined"] = True
                            break
                        attempt_path = attempts / (str((int(previous[-1].stem) + 1 if previous else 0) + retry) + ".json")
                        def save(arm):
                            audit_write_json(attempt_path, arm)
                        arm = audit_arm(client, url, task, case["gold"], mode, seconds, save)
                        if profile == "rebench":
                            audit_rebench_verify(arm, case, evaluator_roots, attempts, parser_policy=parser_policy)
                            save(arm)
                        result["arms"][mode] = arm
                        # Fresh-session retry only for positively identified transport errors,
                        # after verified cleanup. Never replay an uncertain command.
                        if not (retry < transport_retries and arm["classification"] == "transport_failure" and arm["cleanup"]["verified"]):
                            break
                    if mode in result["arms"] and not result["arms"][mode].get("cleanup", {}).get("verified"):
                        with scheduling_lock:
                            quarantined.add(url)
                        break
            except Exception as exc:
                result["data_issues"] = ["audit_configuration_error:" + f"{type(exc).__name__}: {exc}"[:1000]]
        result["status"] = audit_result_status(result)
        result["complete"] = True
        if profile == "rebench":
            if result.get("endpoint_quarantined") or any(not a.get("cleanup", {}).get("verified") for a in result["arms"].values()) or any(
                    issue.startswith("interrupted_attempt_requires_cleanup_review:") for issue in result.get("data_issues", [])):
                result["complete"] = False
            result["status"] = audit_rebench_status(result)
            result["native_reference_candidate"] = native_reference_candidate(result)
        audit_write_json(result_path, result)
        if progress:
            progress.finish(result)
        print(json.dumps({"audit_task": result["task_id"], "status": result["status"], "endpoint": url}), flush=True)

    def worker(url):
        while True:
            with scheduling_lock:
                if url in quarantined:
                    return
                index = next((i for i, case in enumerate(pending) if url in case["candidates"]), None)
                if index is None:
                    return
                case = pending.pop(index)
            process(case, url)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(urls) * workers_per_endpoint) as pool:
            futures = [pool.submit(worker, url) for _ in range(workers_per_endpoint) for url in urls]
            for future in futures:
                future.result()
    except BaseException:
        if progress:
            progress.close()
        raise
    # Only cases blocked by endpoint quarantine can remain unclaimed.
    for case in pending:
        process(case, case["candidates"][0])
    if progress:
        progress.close()
    results = [json.loads((output / "cases" / c["inventory"]["id"] / "result.json").read_text()) for c in cases]
    source_match = all(s["deployment"].get("source_fingerprint", {}).get("sha256") == local["sha256"] for s in services)
    report = {"version": AUDIT_VERSION, "signature": signature, "manifest": str(manifest_path), "results": results,
              "source_match": source_match, "counts": dict(collections.Counter(r["status"] for r in results)),
              "workers_per_endpoint": workers_per_endpoint, "quarantined_endpoints": sorted(quarantined),
              "regression_pass": bool(results) and source_match and not quarantined and all(r["status"] == "reference_verified" for r in results)}
    if profile == "rebench":
        for result in results:
            result.setdefault("native_reference_candidate", native_reference_candidate(result))
        report.update(profile=profile, pilot=selection, complete=all(r.get("complete") for r in results),
                      results_jsonl=str(output / "results.jsonl"),
                      native_parser_summary=native_reference_candidate_summary(results))
    audit_write_json(output / "report.json", report)
    if profile == "rebench":
        (output / "reference_verified_task_ids.txt").write_text("".join(
            result["task_id"] + "\n" for result in results
            if result.get("complete") and result["status"] == "reference_verified"))
    return report


def audit_export(report_path: Path, output: Path) -> dict:
    """Explicit opt-in export; ambiguous and failed cases remain quarantined."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    report = json.loads(report_path.read_text())
    if report.get("profile") == "rebench":
        raise ValueError("training audits require --results and --dataset; pilot exports also require --pilot-only")
    manifest = json.loads(Path(report["manifest"]).read_text())
    if report["signature"] != manifest["signature"] or not report.get("source_match") or report.get("quarantined_endpoints"):
        raise ValueError("export requires matching manifest and deployed sources")
    if output.exists():
        raise ValueError("refusing to overwrite dataset")
    records = {r["id"]: r for r in report["results"]}
    selected, schemas, quarantined = [], [], []
    inventory = manifest["identity"]["manifest"]
    for dataset in inventory["datasets"]:
        path = Path(dataset["path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != dataset["sha256"]:
            raise ValueError("source dataset checksum changed")
        table = pq.read_table(path)
        schemas.append(table.schema)
        for item in inventory["cases"]:
            if item["dataset"] != dataset["path"]:
                continue
            result = records.get(item["id"], {})
            if result.get("status") == "reference_verified":
                selected.append(table.slice(item["row"], 1))
            else:
                quarantined.append({"id": item["id"], "task_id": item.get("task_id"), "reason": result.get("status", "not_audited")})
    if not selected:
        raise ValueError("no reference-verified cases to export")
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.concat_tables(selected), output)
    receipt = {"source_report": str(report_path), "signature": report["signature"], "rows": len(selected),
               "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(), "quarantined": quarantined}
    write_json(output.with_suffix(output.suffix + ".audit.json"), receipt)
    return receipt


def audit_export_results(dataset: Path, results_path: Path, output: Path, *, pilot_only: bool = False,
                         reviewed_baseline_source_failures: list[str] | None = None,
                         require_repeat: bool = False) -> dict:
    """Filter the untouched source using only the portable per-row result ledger."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    receipt_path = output.with_suffix(output.suffix + ".audit.json")
    if output.exists() or receipt_path.exists() or output.resolve() == dataset.resolve():
        raise ValueError("refusing to overwrite dataset or receipt")
    source_sha = file_sha256(dataset)
    results_sha = file_sha256(results_path)
    parquet = pq.ParquetFile(dataset)
    source_rows = parquet.metadata.num_rows
    reviewed = set(reviewed_baseline_source_failures or [])
    records, signatures = {}, set()
    for line in results_path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        row = record.get("row")
        if record.get("version") != "swe-reference-results-v1" or record.get("dataset_sha256") != source_sha:
            raise ValueError("result ledger belongs to another dataset or version")
        if type(row) is not int or not 0 <= row < source_rows or record.get("id") != digest([source_sha, row]):
            raise ValueError("result row identity mismatch")
        if row in records and records[row].get("complete"):
            raise ValueError("duplicate completed result row")
        if record.get("source_match") is not True or not record.get("signature"):
            raise ValueError("results require verified deployed sources and an audit signature")
        records[row] = record
        signatures.add(record["signature"])
    if not records or len(signatures) != 1:
        raise ValueError("result ledger must contain one nonempty audit")
    if any(record.get("complete") is not True for record in records.values()):
        raise ValueError("audit contains incomplete or blocked cases")
    if not pilot_only and len(records) != source_rows:
        raise ValueError("full export requires every dataset row; use --pilot-only for an explicit pilot export")
    if reviewed - {record.get("task_id") for record in records.values()}:
        raise ValueError("reviewed baseline task IDs must appear in the audit ledger")
    # Read only task identity metadata during validation, never the full 22 GiB table.
    task_ids = []
    for batch in parquet.iter_batches(batch_size=128, columns=["extra_info"]):
        for item in batch.to_pylist():
            task = json.loads(item["extra_info"]["tools_kwargs"]["__swe_task_json__"])
            task_ids.append(task.get("task_id"))
    selected, excluded = [], []
    for row, record in sorted(records.items()):
        if record.get("task_id") != task_ids[row]:
            raise ValueError("result task identity mismatch")
        reviewed_source = record.get("task_id") in reviewed
        if reviewed_source and record["status"] != "review:gold_verified_baseline_source_failure":
            raise ValueError("reviewed task is not a gold-verified baseline source failure")
        if record["status"] == "reference_verified" or reviewed_source:
            if require_repeat:
                from scripts.swe_reference_reliability import validate_repeat_qualification
                if reviewed_source:
                    raise ValueError("repeat-qualified export cannot include baseline source-failure reviews")
                validate_repeat_qualification(record)
            arms = record.get("arms", {})
            baseline, gold = arms.get("baseline", {}), arms.get("gold", {})
            before, after = baseline.get("reference_evidence", {}), gold.get("reference_evidence", {})
            if any(e.get("parser_policy") == "native-guarded" and
                   e.get("service_consistency", {}).get("matched") is not True for e in (before, after)):
                raise ValueError("native-guarded export requires matching audit and service scoring")
            baseline_valid = (
                before.get("baseline_source_failure_candidate") and not before.get("complete")
                if reviewed_source else before.get("complete") and before.get("fail_to_pass_failed", 0) > 0
                and before.get("pass_to_pass_complete")
            )
            if (set(arms) != {"baseline", "gold"} or not all(a.get("cleanup_verified") is True for a in arms.values())
                    or not baseline_valid or not after.get("complete") or gold.get("score") != 1
                    or baseline.get("score") != 0 or not after.get("all_required_passed")):
                raise ValueError("reference_verified row lacks consistent verification evidence")
            selected.append(row)
        else:
            excluded.append({"row": row, "task_id": record.get("task_id"), "reason": record["status"]})
    if not selected:
        raise ValueError("no reference-verified cases to export")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        selected_set = set(selected)
        row_offset = 0
        with pq.ParquetWriter(temporary, parquet.schema_arrow) as writer:
            for batch in parquet.iter_batches(batch_size=64):
                keep = [i for i in range(batch.num_rows) if row_offset + i in selected_set]
                row_offset += batch.num_rows
                if keep:
                    writer.write_table(pa.Table.from_batches([batch]).take(pa.array(keep, type=pa.int64())))
        exported = pq.ParquetFile(temporary)
        if exported.metadata.num_rows != len(selected) or not exported.schema_arrow.equals(parquet.schema_arrow, check_metadata=True):
            raise ValueError("dataset roundtrip changed retained schema or row count")
        if file_sha256(dataset) != source_sha or file_sha256(results_path) != results_sha:
            raise ValueError("source dataset or result ledger changed during export")
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    receipt = {"version": "swe-training-audit-filter-v1", "source": str(dataset.resolve()), "source_sha256": source_sha,
               "source_rows": source_rows, "source_results": str(results_path.resolve()), "results_sha256": results_sha,
               "signature": next(iter(signatures)), "rows": len(selected), "retained_row_indices": selected,
               "pilot_only": pilot_only, "not_audited_rows": source_rows - len(records), "quarantined": excluded,
               "retained_rows_unchanged": True, "output_sha256": file_sha256(output), "repeat_required": require_repeat}
    receipt["reviewed_baseline_source_failures"] = sorted(reviewed)
    audit_write_json(receipt_path, receipt)
    return receipt


def retained_audit_phases(arm: dict, output_dir: Path) -> tuple[dict, str]:
    import gzip
    from scripts.diagnose_trajectories import collect_private_evaluator
    private = arm.get("private_evaluator", {})
    if private.get("status") != "verified":
        raise ValueError("missing_contract_or_private_evidence")
    path = Path(private["private_copy"]).resolve()
    checked = collect_private_evaluator({"task_id": arm["task_id"], "reward_evidence": {
        "private_evaluator_artifact_path": str(path), "private_evaluator_artifact_sha256": private["sha256"],
        "private_evaluator_service_id": private.get("service_id")}},
        [str(path.parent) + "=" + str(path.parent)], output_dir)
    if checked.get("status") != "verified" or checked.get("logs_complete") is not True:
        raise ValueError("unverified_or_incomplete_private_logs")
    if arm.get("runner_provenance") != "launch_marker_v1":
        raise ValueError("runner_launch_unverified")
    path = Path(checked["private_copy"])
    phases = json.loads(path.read_text())["phases"]
    return {name: (gzip.decompress((path.parent / phase["full_log"]["file"]).read_bytes()).decode("utf-8")
                   if phase.get("full_log") else phase.get("output", ""))
            for name, phase in phases.items()}, private["sha256"]


def replay_runner_evidence(task: dict, phases: dict) -> tuple[list[dict], str]:
    from recipe.swe_agent.test_results import official_test_outcomes
    specs = task.get("evaluator_test_commands", [])
    if specs and all(f"runner_evidence_{i}" in phases for i in range(len(specs))):
        evidence = []
        for i, spec in enumerate(specs):
            retained = json.loads(phases[f"runner_evidence_{i}"])
            if retained["command"] != spec["run"] or retained["evidence"].get("reason") == "runner_launch_unverified":
                raise ValueError("runner_launch_unverified")
            if retained.get("junit_requested"):
                if f"junit_xml_{i}" not in phases:
                    raise ValueError("missing_raw_junit")
                evidence.append(official_test_outcomes(spec["run"], "", exit_code=retained["exit_code"],
                                                       junit_xml=phases[f"junit_xml_{i}"]))
            else:
                evidence.append(retained["evidence"])
        return evidence, "raw_junit_or_runner_evidence"
    full_output = phases.get("test_commands", "")
    markers = re.findall(r"^SWE_RUNNER_LAUNCHED_[a-f0-9]+$", full_output, re.M)
    if len(specs) == 1 and len(markers) == 1:
        from recipe.swe_agent.test_results import runner_output_after_marker
        output = runner_output_after_marker(full_output, markers[0])
        exits = re.findall(r"^exit_code: (-?\d+)$", full_output, re.M)
        evidence = official_test_outcomes(specs[0]["run"], output, exit_code=int(exits[0]) if len(exits) == 1 else -1)
        if evidence["reason"] != "unsupported_runner":
            return [evidence], "native_parser_replay"
    else:
        raise ValueError("runner_launch_unverified")
    if "official_tests" in phases:
        evidence = json.loads(phases["official_tests"])["evidence"]
        if len(evidence) != len(specs):
            raise ValueError("incomplete_runner_evidence")
        return evidence, "retained_structured_score_replay"
    return [evidence], "native_parser_replay"


def dataset_version(dataset: Path, output: Path, exclusions: list[str], diagnosis_path: Path | None = None,
                    *, rebench_contract: bool = False) -> dict:
    """Explicit immutable filtering, optionally pinning audit-proven test identities."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from recipe.swe_agent.test_results import pin_official_identity_aliases, score_official_tests
    if output.exists() or output.with_suffix(output.suffix + ".audit.json").exists() or output.resolve() == dataset.resolve():
        raise ValueError("refusing to overwrite dataset or receipt")
    if len(exclusions) != len(set(exclusions)):
        raise ValueError("duplicate exclusions")
    original_sha = hashlib.sha256(dataset.read_bytes()).hexdigest()
    table = pq.read_table(dataset)
    if rebench_contract and diagnosis_path:
        raise ValueError("rebench contract upgrade cannot combine with Verified identity pinning")
    _, cases = audit_inventory([dataset], profile="rebench" if rebench_contract else "validation")
    case_by_row = {case["inventory"]["row"]: case for case in cases}
    ids = [case_by_row[i]["inventory"]["task_id"] for i in range(table.num_rows)]
    if len(set(ids)) != len(ids) or not set(exclusions).issubset(ids):
        raise ValueError("task identities must be unique and all exclusions present")
    keep = [i for i, identity in enumerate(ids) if identity not in exclusions]
    filtered = table.take(pa.array(keep, type=pa.int64()))
    upgrades = {}
    def update_tasks(node, contract):
        if isinstance(node, dict):
            for key, value in node.items():
                if key in {"__swe_task_json__", "task_json"} and isinstance(value, str):
                    embedded = json.loads(value)
                    embedded["official_test_contract"] = contract
                    node[key] = json.dumps(embedded, ensure_ascii=True)
                elif isinstance(value, dict):
                    update_tasks(value, contract)
    if rebench_contract:
        rows = filtered.to_pylist()
        for row, index in zip(rows, keep):
            case = case_by_row[index]
            if case["inventory"]["issues"]:
                raise ValueError("cannot version invalid rebench row: " + ids[index])
            truth = row["reward_model"]["ground_truth"]
            truth = json.loads(truth) if isinstance(truth, str) else dict(truth)
            contract = case["reference_contract"]
            from recipe.swe_agent.test_results import REBENCH_CONTRACT_FIELDS, rebench_contract_fields
            contract.update(rebench_contract_fields(contract["log_parser"], contract["fail_to_pass"], contract["pass_to_pass"]))
            truth.update({k: contract[k] for k in REBENCH_CONTRACT_FIELDS})
            original_truth = row["reward_model"]["ground_truth"]
            row["reward_model"]["ground_truth"] = json.dumps(truth) if isinstance(original_truth, str) else truth
            update_tasks(row["extra_info"]["tools_kwargs"], contract)
            upgrades[ids[index]] = {"rebench_scoring_version": 1, "label_issues": case["inventory"]["label_issues"]}
        filtered = pa.Table.from_pylist(rows, schema=table.schema)
    if diagnosis_path is not None:
        arms = {}
        diagnosis = json.loads(diagnosis_path.read_text())
        for audit in diagnosis.get("reference_audits", []):
            audit_path = Path(audit["path"])
            if hashlib.sha256(audit_path.read_bytes()).hexdigest() != audit["sha256"]:
                raise ValueError("diagnosed audit checksum changed")
            source_report = json.loads(audit_path.read_text())
            manifest = json.loads(Path(source_report["manifest"]).read_text())
            if (not source_report.get("source_match") or source_report.get("quarantined_endpoints")
                    or source_report["signature"] != manifest["signature"]
                    or original_sha not in {d["sha256"] for d in manifest["identity"]["manifest"]["datasets"]}):
                raise ValueError("identity pinning requires an audit of this exact source dataset")
            source_arms = {(r["task_id"], mode): value for r in source_report["results"]
                           for mode, value in r.get("arms", {}).items()}
            for arm in audit["arms"]:
                key = (arm["task_id"], arm["mode"])
                if key in arms:
                    raise ValueError("duplicate diagnosis arms; provide one reviewed audit")
                metadata = source_arms.get(key, {}).get("reward", {}).get("metadata", {})
                if arm.get("private_evaluator", {}).get("status") == "verified" and (
                        metadata.get("private_evaluator_artifact_sha256") != arm["private_evaluator"]["sha256"]
                        or metadata.get("hidden_runner_provenance") != "launch_marker_v1"):
                    raise ValueError("diagnosis evidence disagrees with source audit")
                arms[key] = arm
        rows = filtered.to_pylist()
        for row, index in zip(rows, keep):
            case = case_by_row[index]
            task = audit_task(case["task"], 900)
            contract = task.get("official_test_contract")
            identity = ids[index]
            if not contract or not all((identity, mode) in arms for mode in ("baseline", "gold")):
                continue
            try:
                retained = [retained_audit_phases(arms[(identity, mode)], output.parent) for mode in ("baseline", "gold")]
                evidence = [replay_runner_evidence(task, phases)[0] for phases, _ in retained]
            except ValueError:
                continue
            aliases = pin_official_identity_aliases(contract, *evidence)
            if not aliases:
                continue
            upgraded = {**contract, "identity_alias_version": 1, "identity_aliases": aliases}
            baseline, gold = [score_official_tests(upgraded, e, len(task["evaluator_test_commands"])) for e in evidence]
            if (not baseline["official_tests_complete"] or not baseline["official_fail_to_pass_failed"]
                    or not baseline["official_pass_to_pass_complete"] or not gold["official_tests_success"]):
                continue
            truth = json.loads(row["reward_model"]["ground_truth"])
            truth.update(identity_alias_version=1, identity_aliases=aliases)
            row["reward_model"]["ground_truth"] = json.dumps(truth, ensure_ascii=True)
            update_tasks(row["extra_info"]["tools_kwargs"], upgraded)
            upgrades[identity] = {"groups": len(aliases), "members": sum(map(len, aliases.values())),
                                  "baseline_sha256": retained[0][1], "gold_sha256": retained[1][1]}
        filtered = pa.Table.from_pylist(rows, schema=table.schema)
    if hashlib.sha256(dataset.read_bytes()).hexdigest() != original_sha:
        raise ValueError("source dataset changed during versioning")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        pq.write_table(filtered, temporary)
        if not pq.read_table(temporary).equals(filtered, check_metadata=True):
            raise ValueError("dataset roundtrip changed retained data")
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    receipt = {"version": "swe-dataset-version-v1", "source": str(dataset.resolve()), "source_sha256": original_sha,
               "source_rows": table.num_rows, "rows": filtered.num_rows, "excluded_task_ids": sorted(exclusions),
               "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(), "identity_upgrades": upgrades,
               "diagnosis_sha256": hashlib.sha256(diagnosis_path.read_bytes()).hexdigest() if diagnosis_path else None,
               "retained_rows_unchanged": not upgrades}
    write_json(output.with_suffix(output.suffix + ".audit.json"), receipt)
    return receipt


def reference_replay(datasets: list[Path], diagnosis_path: Path, report_path: Path) -> dict:
    """Replay retained evidence without executing tasks or certifying a service."""
    from recipe.swe_agent.test_results import score_official_tests
    diagnosis = json.loads(diagnosis_path.read_text())
    if diagnosis.get("profile") == "rebench" and "results" in diagnosis:
        _, cases = audit_inventory(datasets, profile="rebench")
        by_id = {case["inventory"]["id"]: case for case in cases}
        records, repairs = [], []
        for original in diagnosis["results"]:
            case = by_id.get(original["id"])
            if case is None:
                raise ValueError("replay report is not from the exact input dataset")
            result = json.loads(json.dumps(original))
            for mode, arm in result.get("arms", {}).items():
                private = arm.get("private_evaluator", {})
                private_path = private.get("private_copy")
                if private_path:
                    root = str(Path(private_path).resolve().parent)
                    metadata = arm["reward"]["metadata"]
                    if metadata.get("private_evaluator_artifact_sha256") != private.get("sha256"):
                        raise ValueError("retained evaluator disagrees with reward evidence")
                    metadata["private_evaluator_artifact_path"] = str(Path(private_path).resolve())
                    audit_rebench_verify(arm, case, [root + "=" + root],
                        report_path.parent / "replay_private" / original["id"] / mode)
                records.append({"task_id": original["task_id"], "mode": mode,
                    "reason": arm.get("reference_evidence", {}).get("reason", arm.get("classification")),
                    "evidence": arm.get("reference_evidence", {}), "previous_status": original["status"]})
            status = audit_rebench_status(result)
            for record in records:
                if record["task_id"] == original["task_id"]:
                    record["replayed_status"] = status
            if case["inventory"].get("label_issues"):
                repairs.append({"task_id": original["task_id"], "action": "quarantine_pending_review",
                                "issues": case["inventory"]["label_issues"]})
        report = {"version": "swe-reference-replay-v1", "profile": "rebench", "cases": records,
                  "audit_sha256": hashlib.sha256(diagnosis_path.read_bytes()).hexdigest(),
                  "reasons": dict(collections.Counter(r["reason"] for r in records)),
                  "official_coverage": dict(collections.Counter(str(r["evidence"].get("complete")) for r in records)),
                  "label_review_manifest": {"version": 1, "source_sha256": hashlib.sha256(diagnosis_path.read_bytes()).hexdigest(),
                                            "cases": repairs},
                  "limitation": "Offline replay cannot restore masked runner exits or certify service deployment; restart and rerun required."}
        write_json(report_path, report)
        return report
    _, cases = audit_inventory(datasets)
    tasks = {case["inventory"].get("task_id"): audit_task(case["task"], 900)
             for case in cases if case["inventory"]["eligible"]}
    diagnosis = json.loads(diagnosis_path.read_text())
    records = []
    for audit in diagnosis.get("reference_audits", []):
        for arm in audit["arms"]:
            if arm["task_id"] not in tasks:
                continue
            task = tasks.get(arm["task_id"], {})
            record = {"task_id": arm["task_id"], "mode": arm["mode"], "reason": "missing_contract_or_private_evidence"}
            records.append(record)
            if not task.get("official_test_contract"):
                continue
            try:
                phases, private_sha = retained_audit_phases(arm, report_path.parent)
                evidence, replay_kind = replay_runner_evidence(task, phases)
            except ValueError as exc:
                record["reason"] = str(exc)
                continue
            score = score_official_tests(task["official_test_contract"], evidence, len(task["evaluator_test_commands"]))
            score.pop("_private_official_results")
            parser_reason = next((e.get("reason") for e in evidence if not e.get("complete")), None)
            record.update(score, reason=parser_reason or score["official_tests_reason"], replay_kind=replay_kind, private_sha256=private_sha)
    result = {"version": "swe-reference-replay-v1", "cases": records,
              "reasons": dict(collections.Counter(r["reason"] for r in records)),
              "official_coverage": dict(collections.Counter(str(r.get("official_tests_complete")) for r in records)),
              "limitation": "Structured score replay does not reparse unavailable old XML. Restart and rerun are required to certify deployed scoring."}
    write_json(report_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    version_parser = sub.add_parser("dataset-version", help="exclude explicit task IDs in a new immutable dataset; optionally pin verified audit identities")
    version_parser.add_argument("--dataset", type=Path, required=True)
    version_parser.add_argument("--output", type=Path, required=True)
    version_parser.add_argument("--exclude-task-id", action="append", default=[])
    version_parser.add_argument("--diagnosis-report", type=Path)
    version_parser.add_argument("--rebench-contract", action="store_true",
                                help="opt in to versioned exact-identity ReBench reward scoring in a new dataset")
    inventory_parser = sub.add_parser("audit-inventory", help="inventory reference/evaluator artifacts without service calls")
    inventory_parser.add_argument("--dataset", action="append", type=Path, required=True)
    inventory_parser.add_argument("--path-map", action="append", default=[])
    inventory_parser.add_argument("--report", type=Path, required=True)
    audit_parser = sub.add_parser("reference-audit", help="parallel fresh baseline/gold sessions with image-aware work sharing")
    audit_parser.add_argument("--dataset", action="append", type=Path, required=True)
    audit_parser.add_argument("--path-map", action="append", default=[])
    audit_parser.add_argument("--task-id", action="append")
    audit_parser.add_argument("--task-ids-file", type=Path,
                              help="newline-delimited candidate task IDs; empty files are rejected")
    audit_parser.add_argument("--limit", type=int, default=0)
    audit_parser.add_argument("--profile", choices=("validation", "rebench"), default="validation")
    audit_parser.add_argument("--pilot-size", type=int, default=0, help="deterministic small, diverse training pilot; 0 audits all rows")
    audit_parser.add_argument("--seed", type=int, default=0)
    audit_parser.add_argument("--evaluator-root", action="append", default=[], help="trusted private diagnostic SERVICE_ROOT=LOCAL_ROOT; required for rebench")
    audit_parser.add_argument("--scenario-deadline", type=float, default=900)
    audit_parser.add_argument("--image-policy", choices=("cache-only", "archive-or-pull"), default="cache-only")
    audit_parser.add_argument("--image-prepare-deadline", type=float, default=1800)
    audit_parser.add_argument("--image-prepare-parallelism", type=int, default=1,
                              help="image preparations per endpoint, independently of case workers; service limit also applies")
    audit_parser.add_argument("--urls", default=os.environ.get("SWE_AGENT_EXECUTION_URLS", ""))
    audit_parser.add_argument("--output-dir", type=Path, required=True)
    audit_parser.add_argument("--resume", action="store_true")
    audit_parser.add_argument("--filter-exit-status", action="store_true",
                              help="exit zero for a complete source-matched ReBench audit even with excluded rows")
    audit_parser.add_argument("--workers-per-endpoint", type=int, default=None,
                              help="concurrent cases per service; SWE_AGENT_REFERENCE_AUDIT_WORKERS_PER_ENDPOINT (default 2)")
    audit_parser.add_argument("--rebench-parser-policy", choices=("strict", "native-first-shadow", "native-guarded"), default="strict",
                              help="native-guarded shares versioned native adjudication with RL; other modes retain strict certification")
    export_parser = sub.add_parser("audit-export", help="explicitly export only reference-verified cases to a new parquet")
    export_source = export_parser.add_mutually_exclusive_group(required=True)
    export_source.add_argument("--report", type=Path)
    export_source.add_argument("--results", type=Path, help="portable training results.jsonl")
    export_parser.add_argument("--dataset", type=Path)
    export_parser.add_argument("--pilot-only", action="store_true")
    export_parser.add_argument("--reviewed-baseline-source-failure", action="append", default=[], metavar="TASK_ID",
                               help="Include a reviewed gold-verified baseline build/import failure; never fabricate failed test identities")
    export_parser.add_argument("--require-repeat", action="store_true",
                               help="require evidence from two independent guarded reference audits")
    export_parser.add_argument("--output", type=Path, required=True)
    repeat_parser = sub.add_parser("audit-qualify-repeats", help="qualify two clean guarded runs with optional transient retries")
    repeat_parser.add_argument("--first-results", type=Path, required=True)
    repeat_parser.add_argument("--repeat-results", type=Path, action="append", required=True)
    repeat_parser.add_argument("--output-dir", type=Path, required=True)
    replay_parser = sub.add_parser("reference-replay", help="offline replay of verified private evidence from the existing diagnosis pipeline")
    replay_parser.add_argument("--dataset", action="append", type=Path, required=True)
    replay_parser.add_argument("--diagnosis-report", type=Path, required=True)
    replay_parser.add_argument("--report", type=Path, required=True)
    build_parser = sub.add_parser("build")
    build_parser.add_argument("--output-dir", type=Path, required=True)
    build_parser.add_argument("--trajectory-input", type=Path, help="reviewed v6 step-0 input; freeze full runner-response regression cases")
    build_parser.add_argument("--latest-trajectory-input", type=Path, help="reviewed v7 step-0 input; freeze exact command/response parser regressions")
    build_parser.add_argument("--sanity-trajectory-input", type=Path, help="reviewed 12-task v8 input; freeze public runner regressions")
    build_parser.add_argument("--reference-dataset", action="append", type=Path)
    build_parser.add_argument("--reference-audit-report", type=Path)
    build_parser.add_argument("--reference-replay-report", type=Path, help="reviewed offline replay of the source audit")
    offline_parser = sub.add_parser("offline")
    offline_parser.add_argument("--benchmark-dir", type=Path, required=True)
    offline_parser.add_argument("--report", type=Path, required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--benchmark-dir", type=Path, required=True)
    run_parser.add_argument("--report", type=Path, required=True)
    run_parser.add_argument("--urls", default=os.environ.get("SWE_AGENT_EXECUTION_URLS", ""))
    run_parser.add_argument("--image", default="", help="existing cached Python/Bash image; never pulled or built")
    run_parser.add_argument("--scenario-deadline", type=float, default=120)
    diagnose_parser = sub.add_parser("diagnose", help="inspect services and run a bounded isolated search probe when capacity permits")
    diagnose_parser.add_argument("--report", type=Path, required=True)
    diagnose_parser.add_argument("--urls", default=os.environ.get("SWE_AGENT_EXECUTION_URLS", ""))
    diagnose_parser.add_argument("--image", default="", help="existing cached Python/Bash image; never pulled or built")
    diagnose_parser.add_argument("--scenario-deadline", type=float, default=90)
    preflight_parser = sub.add_parser("preflight", help="read-only health, source and persistent-rg check on every selected endpoint")
    preflight_parser.add_argument("--report", type=Path, required=True)
    preflight_parser.add_argument("--urls", default=os.environ.get("SWE_AGENT_EXECUTION_URLS", ""))
    args = parser.parse_args()
    if args.command == "dataset-version":
        receipt = dataset_version(args.dataset, args.output, args.exclude_task_id, args.diagnosis_report,
                                  rebench_contract=args.rebench_contract)
        print(json.dumps(receipt, sort_keys=True))
        return 0
    if args.command == "audit-inventory":
        report, _ = audit_inventory(args.dataset, args.path_map)
        write_json(args.report, report)
        print(json.dumps({**report["counts"], "repos": len(report["counts"]["repos"])}, sort_keys=True))
        return 0
    if args.command == "audit-export":
        if args.results:
            if not args.dataset:
                parser.error("--results requires --dataset")
            report = audit_export_results(args.dataset, args.results, args.output, pilot_only=args.pilot_only,
                                          reviewed_baseline_source_failures=args.reviewed_baseline_source_failure,
                                          require_repeat=args.require_repeat)
        else:
            if args.dataset or args.pilot_only or args.reviewed_baseline_source_failure or args.require_repeat:
                parser.error("--dataset, --pilot-only, and baseline reviews require --results")
            report = audit_export(args.report, args.output)
        print(json.dumps({"rows": report["rows"], "quarantined": len(report["quarantined"])}))
        return 0
    if args.command == "audit-qualify-repeats":
        from scripts.swe_reference_reliability import qualify_repeats
        print(json.dumps(qualify_repeats(args.first_results, args.repeat_results, args.output_dir), sort_keys=True))
        return 0
    if args.command == "reference-replay":
        report = reference_replay(args.dataset, args.diagnosis_report, args.report)
        print(json.dumps({"arms": len(report["cases"]), "reasons": report["reasons"], "official_coverage": report["official_coverage"]}, sort_keys=True))
        return 0
    if args.command == "build":
        print(json.dumps(build(args.output_dir, args.trajectory_input, args.latest_trajectory_input, args.sanity_trajectory_input,
                         reference_datasets=args.reference_dataset, reference_audit_report=args.reference_audit_report,
                         reference_replay_report=args.reference_replay_report), sort_keys=True))
        return 0
    if args.command == "offline":
        report = offline(args.benchmark_dir, args.report)
        print(json.dumps({"counts": report["counts"], "regression_pass": report["regression_pass"]}, sort_keys=True))
        return 0 if report["regression_pass"] else 1
    urls = [url.strip() for url in args.urls.split(",") if url.strip()]
    if not urls or getattr(args, "scenario_deadline", 1) <= 0:
        parser.error("positive scenario deadline and at least one URL required")
    if args.command == "reference-audit":
        task_ids = args.task_id
        if args.task_ids_file is not None:
            file_ids = [line.strip() for line in args.task_ids_file.read_text().splitlines() if line.strip()]
            if not file_ids:
                parser.error("--task-ids-file must not be empty")
            task_ids = list(dict.fromkeys([*(task_ids or []), *file_ids]))
        report = reference_audit(args.dataset, urls, args.output_dir, mappings=args.path_map,
                                 task_ids=task_ids, limit=args.limit, seconds=args.scenario_deadline, resume=args.resume,
                                 workers_per_endpoint=args.workers_per_endpoint, profile=args.profile,
                                 pilot_size=args.pilot_size, seed=args.seed, evaluator_roots=args.evaluator_root,
                                 image_policy=args.image_policy, image_prepare_seconds=args.image_prepare_deadline,
                                 image_prepare_parallelism=args.image_prepare_parallelism, parser_policy=args.rebench_parser_policy)
    elif args.command == "preflight":
        report = preflight(urls, args.report)
    elif args.command == "diagnose":
        report = diagnose(urls, args.report, args.image, args.scenario_deadline)
    else:
        report = run(args.benchmark_dir, urls, args.report, args.image, args.scenario_deadline)
    print(json.dumps({"counts": report["counts"], "regression_pass": report["regression_pass"]}, sort_keys=True))
    if args.command == "reference-audit" and args.filter_exit_status:
        return 0 if (report.get("profile") == "rebench" and report.get("complete") is True
                     and report.get("source_match") is True and not report.get("quarantined_endpoints")) else 1
    return 0 if report["regression_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

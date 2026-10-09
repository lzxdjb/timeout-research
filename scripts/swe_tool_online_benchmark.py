"""Frozen, bounded online SWE tool contracts. No model inference or training."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import shlex
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
sys.path.insert(0, str(ROOT.parent / "stock-rl-reflect"))
sys.path.insert(0, str(ROOT))
VERSION = "swe-tool-online-v9"


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
        ("unicode_search_capability", [
            step("Write", {"file_path": "probe/unicode.txt", "content": "Needle caf\u00e9\nItem(\n"}, contains="Wrote"),
            step("Grep", {"pattern": "needle", "path": "probe/unicode.txt"}, equals="probe/unicode.txt:1:Needle caf\u00e9")
                | {"missing_rg_expected": {"prefix": "Search error:", "contains": "non-ASCII case-insensitive search requires rg"}},
            step("Grep", {"pattern": "needle", "literal": True, "path": "probe/unicode.txt"}, equals="probe/unicode.txt:1:Needle caf\u00e9")
                | {"missing_rg_expected": {"prefix": "Search error:", "contains": "non-ASCII case-insensitive search requires rg"}},
            step("Grep", {"pattern": "caf\u00e9", "literal": True, "case_sensitive": True, "path": "probe/unicode.txt"}, equals="probe/unicode.txt:1:Needle caf\u00e9"),
            step("Grep", {"pattern": r"Item\(", "case_sensitive": True, "path": "probe/unicode.txt"}, equals="probe/unicode.txt:2:Item(")
                | {"missing_rg_expected": {"prefix": "Search error:", "contains": "non-ASCII regex input requires rg"}},
            step("Grep", {"pattern": r"Item\(", "literal": True, "case_sensitive": True, "path": "probe/unicode.txt"}, prefix="No matches.", contains="backslashes literally"),
            step("Grep", {"pattern": "needle", "output_mode": "count", "path": "probe/unicode.txt"}, equals="probe/unicode.txt:1")
                | {"missing_rg_expected": {"prefix": "Search error:", "contains": "requires rg"}},
            step("Grep", {"pattern": "needle", "output_mode": "files_with_matches", "path": "probe/unicode.txt"}, equals="probe/unicode.txt")
                | {"missing_rg_expected": {"prefix": "Search error:", "contains": "requires rg"}},
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
        ("observed_tests_preserve_explicit_intent", [
            step("Write", {"file_path": "tests/test_contract.py", "content": "import unittest\nclass Contract(unittest.TestCase):\n    def test_ok(self):\n        self.assertEqual(1, 1)\n"}, contains="Wrote"),
            *[step("Bash", {"command": test_command, **({"verification": flag} if flag is not None else {})},
                   metadata_paths={"public_test_attempt.test_outcome": "passed", "public_test_attempt.explicit_verification": flag is True},
                   **({"metadata_absent": "public_verification"} if flag is not True else {})) for flag in (None, False, True)],
            step("Bash", {"command": 'python -c "print(\'1 passed in 0.1s\')"'}, metadata_absent="public_test_attempt"),
            step("Bash", {"command": "git log -- src/_pytest/python.py"}, metadata_absent="public_test_attempt"),
        ]),
        ("grep_offset_actionable_correction", [write,
            step("Grep", {"pattern": "needle", "offset": 10}, prefix="Search error:", contains='Read(file_path="relative/file.py", offset=0, limit=100)'),
            step("Read", {"file_path": "src/a.py", "offset": 1, "limit": 1}, equals="needle")]),
        ("ansi_and_partial_runner_evidence", [
            step("Write", {"file_path": "pytest.py", "content": "print('\\x1b[31m1 failed\\x1b[0m, \\x1b[32m4 passed\\x1b[0m in 0.1s')\n"}, contains="Wrote"),
            step("Bash", {"command": "python -m pytest | head -100"}, prefix="exit_code: 0", metadata_paths={"public_test_attempt.test_outcome": "failed", "public_test_attempt.tests_executed": 5}, metadata_absent="public_verification"),
            step("Write", {"file_path": "pytest.py", "content": "print('================ FAILURES ================\\nE   AssertionError: mismatch')\n"}, contains="Wrote"),
            step("Bash", {"command": "python -m pytest | head -100"}, contains="terminal runner summary is missing", metadata_paths={"public_test_attempt.test_outcome": "unknown", "public_test_attempt.test_incomplete_failure_evidence": True, "public_test_attempt.tests_executed": -1})]),
        ("grep_alias_contract", [write,
            step("Grep", {"pattern": "needle", "limit": 1}, contains="needle", metadata={"search_limit_alias_used": True}),
            step("Grep", {"pattern": "needle", "limit": 1, "max_results": 2}, prefix="Search error:", contains="conflict"),
            step("Grep", {"pattern": "needle", "limit": 0}, prefix="Search error:", contains="positive integer"),
            step("Grep", {"pattern": "needle", "limit": True}, prefix="Search error:"),
            step("Grep", {"pattern": "needle", "unknown": 1}, prefix="Search error:", contains="Supported parameters")]),
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
    commands = hidden_patch_reset_commands(patch, "HEAD")
    if commands != ["git checkout HEAD -- check.py", "rm -f -- empty.py"]:
        raise ValueError("Synthetic evaluator reset generator differs from independently specified oracle")
    return {"id": "synthetic_evaluator_empty_file", "expected_hits": None,
            "steps": [step("Bash", {"command": "git init -q && git -c user.name=Fixture -c user.email=fixture@example.com commit --allow-empty -qm base && printf \"print('base')\\n\" > check.py && git add check.py && git -c user.name=Fixture -c user.email=fixture@example.com commit -qm check && printf repair > model.txt && printf stale > empty.py"}, prefix="exit_code: 0")],
            "evaluator_fixture": {"base_sha": "HEAD", "hidden_test_patch": patch,
                "trusted_hidden_test_patch": True, "verifier_setup_commands": commands,
                "test_commands": ["python check.py"]}}


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
    ]


def check_evaluator_fixture(response: dict) -> list[str]:
    metadata = response.get("metadata", {})
    checks = {"ok": response.get("ok") is True, "score": response.get("score") == 1.,
        "started": metadata.get("hidden_tests_started") is True, "valid": metadata.get("evaluation_valid") is True,
        "phase": metadata.get("hidden_failure_phase") == "none", "private_evidence": metadata.get("private_evaluator_artifact_status") == "available",
        "complete_private_logs": metadata.get("private_evaluator_logs_complete") is True}
    return ["evaluator fixture: " + name for name, passed in checks.items() if not passed]


def check_parser_replay(case: dict) -> dict:
    from recipe.swe_agent.test_results import summarize_test_execution
    fixture = case["parser_replay"]
    actual = summarize_test_execution(fixture["command"], fixture["output"], exit_code=fixture["exit_code"])
    failures = [f"saved parser replay {key}: expected {value!r}, got {actual.get(key)!r}"
                for key, value in fixture["expected"].items() if actual.get(key) != value]
    return {"scope": "local_saved_evidence_parser_only", "actual": actual, "failures": failures}


def build(directory: Path, trajectory_input: Path | None = None, latest_trajectory_input: Path | None = None, sanity_trajectory_input: Path | None = None) -> dict:
    data = scenarios()
    data.extend(v8_scenarios())
    data.extend(v9_scenarios())
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
    if manifest["version"] not in {VERSION, "swe-tool-online-v8", "swe-tool-online-v7", "swe-tool-online-v6", "swe-tool-online-v5", "swe-tool-online-v4", "swe-tool-online-v3", "swe-tool-online-v2"} or digest(cases) != manifest["scenarios_sha256"]:
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build_parser = sub.add_parser("build")
    build_parser.add_argument("--output-dir", type=Path, required=True)
    build_parser.add_argument("--trajectory-input", type=Path, help="reviewed v6 step-0 input; freeze full runner-response regression cases")
    build_parser.add_argument("--latest-trajectory-input", type=Path, help="reviewed v7 step-0 input; freeze exact command/response parser regressions")
    build_parser.add_argument("--sanity-trajectory-input", type=Path, help="reviewed 12-task v8 input; freeze public runner regressions")
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
    if args.command == "build":
        print(json.dumps(build(args.output_dir, args.trajectory_input, args.latest_trajectory_input, args.sanity_trajectory_input), sort_keys=True))
        return 0
    if args.command == "offline":
        report = offline(args.benchmark_dir, args.report)
        print(json.dumps({"counts": report["counts"], "regression_pass": report["regression_pass"]}, sort_keys=True))
        return 0 if report["regression_pass"] else 1
    urls = [url.strip() for url in args.urls.split(",") if url.strip()]
    if not urls or getattr(args, "scenario_deadline", 1) <= 0:
        parser.error("positive scenario deadline and at least one URL required")
    if args.command == "preflight":
        report = preflight(urls, args.report)
    elif args.command == "diagnose":
        report = diagnose(urls, args.report, args.image, args.scenario_deadline)
    else:
        report = run(args.benchmark_dir, urls, args.report, args.image, args.scenario_deadline)
    print(json.dumps({"counts": report["counts"], "regression_pass": report["regression_pass"]}, sort_keys=True))
    return 0 if report["regression_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

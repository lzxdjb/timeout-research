"""Pinned SWE-bench Verified parsers/grading, used only by offline audits.

The vendor files are byte-for-byte upstream sources, with their MIT license.
Only pure grading functions and enum definitions are loaded; upstream package
imports (and their Docker/datasets dependencies) are deliberately excluded.
"""

from __future__ import annotations

import ast
import hashlib
import json
from enum import Enum
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any

COMMIT = "b8d53321b98c1193f9df999ff11d57ac5f5f551d"
VENDOR = Path(__file__).parent / "_vendor" / "swebench_verified"
MANIFEST_SHA256 = "7012e64dabba2efcbd2747d5d1c77351345ffcde53c2fa05295d08a58689334c"


def verify_vendor() -> dict:
    manifest_bytes = (VENDOR / "manifest.json").read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != MANIFEST_SHA256:
        raise ValueError("Verified native manifest checksum mismatch")
    manifest = json.loads(manifest_bytes)
    if manifest["commit"] != COMMIT:
        raise ValueError("Verified native harness revision changed")
    for name, entry in manifest["files"].items():
        if hashlib.sha256((VENDOR / name).read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("Verified native source checksum mismatch: " + name)
    return manifest


@lru_cache(maxsize=1)
def native_module() -> tuple[dict, dict]:
    manifest = verify_vendor()
    scope = {"Enum": Enum, "Any": Any, "TestSpec": SimpleNamespace}
    constants = ast.parse((VENDOR / "constants.py").read_text())
    names = {"FAIL_TO_PASS", "PASS_TO_PASS", "FAIL_TO_FAIL", "PASS_TO_FAIL"}
    nodes = [n for n in constants.body if isinstance(n, ast.ClassDef)
             and n.name in {"TestStatus", "EvalType", "ResolvedStatus"}
             or isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in n.targets)]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "native_constants", "exec"), scope)
    parsers = ast.parse((VENDOR / "python.py").read_text())
    parsers.body = [n for n in parsers.body if not isinstance(n, ast.ImportFrom)]
    exec(compile(parsers, "native_python_parsers", "exec"), scope)
    pure = {"test_passed", "test_failed", "get_eval_tests_report", "compute_fail_to_pass",
            "compute_pass_to_pass", "get_resolution_status"}
    grading = ast.parse((VENDOR / "grading.py").read_text())
    grading.body = [n for n in grading.body if isinstance(n, ast.FunctionDef) and n.name in pure]
    if {n.name for n in grading.body} != pure:
        raise ValueError("missing pinned native grading functions")
    exec(compile(grading, "native_grading", "exec"), scope)
    return scope, manifest


def native_score(repo: str, logs: list[str], contract: dict) -> dict:
    """Exact native identities; retain official grading and coverage separately.

    Native grading counts absent identities as unsuccessful. Coverage adds a
    certification gate, so absence can never certify a failing baseline.
    Strict aliases/JUnit normalization are intentionally not applied here.
    """
    module, _ = native_module()
    parser = module["MAP_REPO_TO_PARSER_PY"].get(repo)
    if parser is None:
        return {"complete": False, "reason": "unsupported_native_repository"}
    outcomes, conflicts = {}, set()
    for text in logs:
        for test, status in parser(text, SimpleNamespace(repo=repo)).items():
            if test in outcomes and outcomes[test] != status:
                conflicts.add(test)
            outcomes[test] = status
    required = contract["fail_to_pass"] + contract["pass_to_pass"]
    missing = [t for t in required if t not in outcomes]
    unresolved = [t for t in required if outcomes.get(t) not in {"PASSED", "XFAIL", "FAILED", "ERROR"}]
    report = module["get_eval_tests_report"](outcomes, {
        "FAIL_TO_PASS": contract["fail_to_pass"], "PASS_TO_PASS": contract["pass_to_pass"]})
    resolved = module["get_resolution_status"](report)
    complete = not missing and not unresolved and not conflicts
    return {"complete": complete, "reason": "complete" if complete else "native_coverage_missing_skipped_or_conflicting",
            "parser": parser.__name__, "commit": COMMIT, "missing": missing,
            "unresolved": unresolved, "conflicting": sorted(conflicts),
            "official_resolution": resolved, "official_resolved": resolved == "RESOLVED_FULL",
            "fail_to_pass_failed": sum(module["test_failed"](t, outcomes) and t in outcomes
                                       for t in contract["fail_to_pass"]),
            "pass_to_pass_complete": all(module["test_passed"](t, outcomes) for t in contract["pass_to_pass"]),
            "all_required_passed": all(module["test_passed"](t, outcomes) for t in required),
            "outcomes_sha256": hashlib.sha256(json.dumps(outcomes, sort_keys=True).encode()).hexdigest()}

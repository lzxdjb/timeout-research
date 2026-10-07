"""Build and evaluate the versioned SWE tool-contract benchmark.

The benchmark deliberately keeps its expected labels independent from the
production detector.  ``build`` freezes synthetic policy contracts and a
stratified sample of historical trajectories.  ``evaluate`` replays them
through the live classifier and repeat detector.  ``probe`` is a read-only
health/readiness check for remote execution services.
"""

from __future__ import annotations

import argparse
import collections
import functools
import gzip
import hashlib
import importlib
import io
import json
import re
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from unittest import mock

VERSION = "swe-tool-contract-v1"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAJECTORIES = ROOT / "analysis" / "diagnosis_20261007" / "trajectory_rows.jsonl"
CONTRACTS_FILENAME = "contracts.json"
TRAJECTORIES_FILENAME = "trajectory_cases.jsonl.gz"
MANIFEST_FILENAME = "manifest.json"


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _event(name: str, arguments: dict[str, Any], response: str = "ok", **extra: Any) -> dict[str, Any]:
    return {"name": name, "arguments": arguments, "response": response, **extra}


def _contract(name: str, events: list[dict[str, Any]], expected_hits: list[bool], expected_outcomes: list[str] | None = None) -> dict[str, Any]:
    normalized = [
        {**event, "index": index, "batch": event.get("batch", index)}
        for index, event in enumerate(events)
    ]
    return {
        "id": name,
        "kind": "explicit_policy_boundary",
        "events": normalized,
        "expected_hits": expected_hits,
        "expected_outcomes": expected_outcomes,
        "label_source": VERSION,
    }


def contracts() -> list[dict[str, Any]]:
    """Return hand-authored boundaries for every high-risk tool category."""
    read = _event("Read", {"file_path": "src/a.py"}, "VALUE = 1")
    edit = _event(
        "Edit",
        {"file_path": "src/a.py", "old_string": "1", "new_string": "2"},
        "Edited src/a.py (1 replacement).",
    )
    failed_edit = _event(
        "Edit",
        {"file_path": "src/a.py", "old_string": "absent", "new_string": "2"},
        "Edit failed: old_string was not found in src/a.py.",
    )
    noop_edit = _event(
        "Edit",
        {"file_path": "src/a.py", "old_string": "1", "new_string": "1"},
        "Edit made no changes: old_string and new_string are identical.",
    )
    pytest = _event("Bash", {"command": "python -m pytest tests/test_a.py -q"}, "1 failed")
    pytest_flagged = _event(
        "Bash", {"command": "python -m pytest tests/test_a.py -q", "verification": True}, "1 failed"
    )
    missing_rg = _event("Grep", {"pattern": "symbol", "path": "src"}, "Search unavailable: rg is not installed.")
    transport = _event("Read", {"file_path": "src/a.py"}, "Remote execution error: service unavailable")
    write_unknown = _event("Write", {"file_path": "src/a.py", "content": "VALUE = 2"}, "Wrote 10 characters to src/a.py")
    shell_mutation = _event("Bash", {"command": "python -c 'open(\"src/a.py\", \"w\").write(\"x\")'"}, "ok")
    search = _event("Grep", {"pattern": "symbol", "path": "src"}, "No matches found.")

    outcome_cases = [
        _contract(
            "outcome_successful_edit",
            [edit], [False], ["changed"],
        ),
        _contract(
            "outcome_failed_edit",
            [failed_edit], [False], ["no_change"],
        ),
        _contract(
            "outcome_noop_edit",
            [noop_edit], [False], ["no_change"],
        ),
        _contract(
            "outcome_standalone_test",
            [pytest], [False], ["completed"],
        ),
        _contract(
            "outcome_explicit_test",
            [pytest_flagged], [False], ["completed"],
        ),
        _contract(
            "outcome_verification_false_is_not_test",
            [_event("Bash", {"command": "pytest tests/test_a.py", "verification": False}, "ok")],
            [False], ["unknown"],
        ),
        _contract(
            "outcome_missing_search_backend",
            [missing_rg], [False], ["deterministic_failure"],
        ),
        _contract(
            "outcome_transport_failure",
            [transport], [False], ["transport_error"],
        ),
        _contract(
            "outcome_ambiguous_write",
            [write_unknown], [False], ["unknown"],
        ),
        _contract(
            "outcome_unknown_shell_mutation",
            [shell_mutation], [False], ["unknown"],
        ),
        _contract(
            "outcome_completed_search",
            [search], [False], ["completed"],
        ),
    ]

    repeat_cases = [
        _contract("repeat_unchanged_read", [read, read], [False, True]),
        _contract("repeat_read_after_confirmed_edit", [read, edit, read], [False, False, False]),
        _contract("repeat_failed_edit_does_not_grant_progress", [read, failed_edit, read], [False, False, True]),
        _contract("repeat_noop_edit_does_not_grant_progress", [read, noop_edit, read], [False, False, True]),
        _contract("repeat_relevant_test_after_edit", [pytest, edit, pytest], [False, False, False]),
        _contract("repeat_relevant_flagged_test_after_edit", [pytest_flagged, edit, pytest_flagged], [False, False, False]),
        _contract(
            "repeat_unrelated_test_after_edit",
            [_event("Bash", {"command": "pytest tests/test_other.py -q"}, "ok"), edit, _event("Bash", {"command": "pytest tests/test_other.py -q"}, "ok")],
            [False, False, True],
        ),
        _contract("repeat_missing_search_capability", [missing_rg, missing_rg, missing_rg], [False, True, True]),
        _contract("repeat_first_transport_retry", [transport, read], [False, False]),
        _contract("repeat_second_transport_retry", [transport, transport, transport], [False, False, True]),
        _contract("repeat_inflight_duplicate", [{**read, "batch": 0}, {**read, "batch": 0}], [False, False]),
        _contract("repeat_write_without_state_evidence", [write_unknown, write_unknown], [False, True]),
        _contract("repeat_shell_mutation_without_state_evidence", [shell_mutation, shell_mutation], [False, True]),
    ]
    return outcome_cases + repeat_cases


def _load_replay_helpers():
    sys.path.insert(0, str(ROOT / "scripts"))
    return importlib.import_module("swe_repeat_regression")


def _row_identity(row: dict[str, Any]) -> tuple[str, str, str]:
    helper = _load_replay_helpers()
    return helper.identity(row)


def _trajectory_categories(row: dict[str, Any], events: list[dict[str, Any]], issues: list[str]) -> set[str]:
    categories: set[str] = set()
    names = {str(event.get("name", "")) for event in events}
    responses = [str(event.get("response", "")) for event in events]
    if issues:
        categories.add("transcript_issue")
    if any(name in {"Grep", "search_text", "search_files"} for name in names):
        categories.add("search")
    if any(name in {"Read", "read_file"} for name in names):
        categories.add("read")
    if any(name in {"Edit", "edit_file", "Write", "write_file", "apply_patch"} for name in names):
        categories.add("mutation")
    if any(name in {"Bash", "bash", "run_shell"} for name in names):
        categories.add("shell")
    if any("Search unavailable:" in response for response in responses):
        categories.add("missing_search_backend")
    if any("path escapes workspace" in response or "absolute" in response.lower() and "path" in response.lower() for response in responses):
        categories.add("path_rejection")
    if any(response.startswith(("Edit failed:", "Edit made no changes:")) for response in responses):
        categories.add("failed_or_noop_edit")
    if any(re.search(r"(?:pytest|unittest|go test|cargo test|npm test|yarn test|mvn test)", str(event.get("arguments", {}).get("command", ""))) for event in events):
        categories.add("verification")
    if any(response.startswith(("Remote execution error:", "Error executing tool", "Error when executing tool:")) for response in responses):
        categories.add("transport_error")
    if row.get("trajectory_timeout") or row.get("trajectory_terminal_tool_failure") or row.get("completion_ratio_cutoff"):
        categories.add("exceptional_exit")
    if row.get("train_sample_mask") is True and row.get("trajectory_penalized_repeated_tool_calls", 0) and row.get("shaped_score") not in {-0.1, -0.10000000149011612}:
        categories.add("reward_mask_anomaly")
    return categories


def _trajectory_case(row: dict[str, Any], source_line: int) -> dict[str, Any] | None:
    helper = _load_replay_helpers()
    events, _messages, issues = helper.parse_transcript(str(row.get("output") or ""))
    if not events and not issues:
        return None
    labels, observations = helper.annotate(events)
    benchmark, task, repo = _row_identity(row)
    task_key = f"{repo}\x1f{task}"
    categories = _trajectory_categories(row, events, issues)
    categories.update(item["category"] for item in observations)
    categories.update(item["category"] for item in labels if item["expected"] != "unknown")
    return {
        "id": digest([str(row.get("path", "")), source_line, digest(row)])[:20],
        "kind": "historical_trajectory",
        "source": {"path": row.get("path"), "line": source_line, "run": row.get("run"), "split": row.get("split"), "step": row.get("step"), "row_sha256": digest(row)},
        "task_key": task_key,
        "benchmark": benchmark,
        "task": task,
        "repository": repo,
        "partition": "holdout" if int(digest(task_key)[:8], 16) % 10 < 3 else "development",
        "categories": sorted(categories),
        "events": events,
        "labels": labels,
        "observations": observations,
        "parse_issues": issues,
        "raw": {
            key: row.get(key)
            for key in ("path", "line", "run", "split", "step", "uid", "raw_score", "shaped_score", "train_sample_mask", "trajectory_penalized_repeated_tool_calls", "trajectory_terminal_tool_failure", "trajectory_timeout", "completion_ratio_cutoff")
            if key in row
        },
    }


@functools.lru_cache(maxsize=256)
def _transcript_lines(path: str) -> tuple[str, ...]:
    try:
        return tuple(Path(path).read_text(encoding="utf-8").splitlines())
    except OSError:
        return ()


def _materialize_indexed_row(row: dict[str, Any]) -> dict[str, Any]:
    """Join an audit-index row to its original transcript when available."""
    if row.get("output") is not None:
        return row
    raw_path = row.get("path")
    raw_line = row.get("line")
    if not isinstance(raw_path, str) or not isinstance(raw_line, int):
        return row
    transcript_path = ROOT / raw_path
    if not transcript_path.exists():
        return row
    lines = _transcript_lines(str(transcript_path))
    if raw_line < 1 or raw_line > len(lines):
        return row
    try:
        original = json.loads(lines[raw_line - 1])
    except json.JSONDecodeError:
        return row
    if isinstance(original, dict):
        return {**original, **row}
    return row


def _select_cases(cases: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Greedy deterministic coverage with at most two cases per task."""
    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    task_counts: collections.Counter[str] = collections.Counter()
    category_counts: collections.Counter[str] = collections.Counter()
    for case in sorted(cases, key=lambda item: item["id"]):
        if len(selected) >= limit:
            break
        if case["id"] in used or task_counts[case["task_key"]] >= 2:
            continue
        gain = sum(1.0 / (1 + category_counts[category]) ** 2 for category in case["categories"])
        if len(selected) < min(64, limit) or gain > 0:
            selected.append(case)
            used.add(case["id"])
            task_counts[case["task_key"]] += 1
            category_counts.update(case["categories"])
    return selected


def _index_categories(row: dict[str, Any]) -> set[str]:
    """Cheap strata derived from the audit index before loading transcripts."""
    categories: set[str] = set()
    counters = {
        str(key): int(value or 0)
        for key, value in row.items()
        if str(key).startswith("trajectory_tool_") and str(value or "").lstrip("-").isdigit()
    }
    if counters.get("trajectory_tool_Grep_dispatches", 0) or counters.get("trajectory_tool_search_text_dispatches", 0) or counters.get("trajectory_tool_search_files_dispatches", 0):
        categories.add("search")
    if counters.get("trajectory_tool_Read_dispatches", 0) or counters.get("trajectory_tool_read_file_dispatches", 0):
        categories.add("read")
    if any(counters.get(f"trajectory_tool_{name}_dispatches", 0) for name in ("Edit", "Write", "apply_patch", "edit_file", "write_file")):
        categories.add("mutation")
    if counters.get("trajectory_tool_Bash_dispatches", 0) or counters.get("trajectory_tool_run_shell_dispatches", 0):
        categories.add("shell")
    if counters.get("trajectory_tool_run_tests_dispatches", 0) or counters.get("trajectory_tool_build_project_dispatches", 0):
        categories.add("verification")
    if row.get("trajectory_tool_error_returns") or row.get("trajectory_tool_call_exceptions"):
        categories.add("transport_error")
    if row.get("trajectory_timeout") or row.get("trajectory_terminal_tool_failure") or row.get("completion_ratio_cutoff"):
        categories.add("exceptional_exit")
    if row.get("trajectory_repeated_tool_calls") or row.get("trajectory_penalized_repeated_tool_calls"):
        categories.add("repeat")
    if not categories:
        categories.add("other")
    return categories


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build(output_dir: Path, trajectory_path: Path, max_trajectory_cases: int) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frozen_contracts = contracts()
    contract_path = output_dir / CONTRACTS_FILENAME
    _write_json(contract_path, frozen_contracts)
    indexed_rows: list[dict[str, Any]] = []
    with trajectory_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            benchmark, task, repo = _row_identity(row)
            task_key = f"{repo}\x1f{task}"
            indexed_rows.append({
                "id": digest([str(row.get("path", "")), line_number, digest(row)])[:20],
                "task_key": task_key,
                "categories": sorted(_index_categories(row)),
                "source_line": line_number,
                "row": row,
            })
    selected_rows = _select_cases(indexed_rows, max_trajectory_cases)
    trajectory_cases: list[dict[str, Any]] = []
    for selected in selected_rows:
        case = _trajectory_case(_materialize_indexed_row(selected["row"]), selected["source_line"])
        if case is not None:
            trajectory_cases.append(case)
    trajectory_path_out = output_dir / TRAJECTORIES_FILENAME
    # Freeze gzip metadata as well as content so the manifest is reproducible.
    with trajectory_path_out.open("wb") as raw_stream:
        with gzip.GzipFile(fileobj=raw_stream, mode="wb", mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8") as stream:
                for case in trajectory_cases:
                    stream.write(json.dumps(case, ensure_ascii=False, separators=(",", ":")) + "\n")
    manifest = {
        "version": VERSION,
        "trajectory_source": str(trajectory_path),
        "trajectory_source_sha256": file_digest(trajectory_path),
        "contracts_sha256": file_digest(contract_path),
        "trajectory_cases_sha256": file_digest(trajectory_path_out),
        "contract_cases": len(frozen_contracts),
        "trajectory_cases": len(trajectory_cases),
        "trajectory_partitions": collections.Counter(case["partition"] for case in trajectory_cases),
        "categories": sorted({category for case in trajectory_cases for category in case["categories"]}),
        "label_policy": "independent evidence labels; unknown historical state is never treated as clean",
    }
    _write_json(output_dir / MANIFEST_FILENAME, manifest)
    return manifest


def _load_benchmark(directory: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    manifest = json.loads((directory / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    if manifest.get("version") != VERSION:
        raise ValueError(f"unsupported benchmark version: {manifest.get('version')}")
    contracts_path = directory / CONTRACTS_FILENAME
    trajectories_path = directory / TRAJECTORIES_FILENAME
    if file_digest(contracts_path) != manifest["contracts_sha256"]:
        raise ValueError("contract checksum mismatch; create a new benchmark version")
    if file_digest(trajectories_path) != manifest["trajectory_cases_sha256"]:
        raise ValueError("trajectory checksum mismatch; create a new benchmark version")
    frozen_contracts = json.loads(contracts_path.read_text(encoding="utf-8"))
    with gzip.open(trajectories_path, "rt", encoding="utf-8") as stream:
        trajectory_cases = [json.loads(line) for line in stream if line.strip()]
    return manifest, frozen_contracts, trajectory_cases


def _load_detector():
    sys.path.insert(0, str(ROOT.parent / "stock-rl-reflect"))
    return importlib.import_module("recipe.swe_agent.repeated_tool")


def _replay(module: Any, events: list[dict[str, Any]]) -> dict[str, Any]:
    detector = module.RepeatedToolDetector()
    results: list[dict[str, Any]] = []
    batches: dict[int, list[dict[str, Any]]] = {}
    order: list[int] = []
    for event in events:
        batch = int(event.get("batch", event.get("index", len(order))))
        if batch not in batches:
            batches[batch] = []
            order.append(batch)
        batches[batch].append(event)
    for batch in order:
        pending = []
        for event in batches[batch]:
            fingerprint, entry, hit = detector.begin(event["name"], event.get("arguments", {}))
            item = {"index": event.get("index", len(results)), "penalized": bool(hit)}
            results.append(item)
            pending.append((event, fingerprint, entry, item))
        for event, fingerprint, entry, item in pending:
            outcome = module.classify_tool_outcome(
                event["name"], event.get("arguments", {}), event.get("response", ""), event.get("metadata")
            )
            before = detector.workspace_progress
            detector.finish(event["name"], event.get("arguments", {}), fingerprint, entry, outcome)
            item["outcome"] = outcome
            item["progress_delta"] = detector.workspace_progress - before
    return {"events": results, "penalized_repeat_count": detector.penalized_repeats}


def _evaluate_contracts(frozen_contracts: list[dict[str, Any]], module: Any) -> tuple[collections.Counter[str], list[dict[str, Any]]]:
    counts: collections.Counter[str] = collections.Counter()
    failures: list[dict[str, Any]] = []
    for case in frozen_contracts:
        actual_outcomes = [module.classify_tool_outcome(event["name"], event.get("arguments", {}), event.get("response", ""), event.get("metadata")) for event in case["events"]]
        expected_outcomes = case.get("expected_outcomes")
        if expected_outcomes:
            for index, (expected, actual) in enumerate(zip(expected_outcomes, actual_outcomes)):
                if expected == "unknown":
                    counts["unknown_outcome_checks"] += 1
                    continue
                counts["outcome_checks"] += 1
                if expected != actual:
                    counts["outcome_failures"] += 1
                    failures.append({"case": case["id"], "kind": "outcome", "index": index, "expected": expected, "actual": actual})
        replay = _replay(module, case["events"])
        for index, expected in enumerate(case["expected_hits"]):
            counts["repeat_checks"] += 1
            actual = replay["events"][index]["penalized"]
            if actual != expected:
                counts["repeat_failures"] += 1
                failures.append({"case": case["id"], "kind": "repeat", "index": index, "expected": expected, "actual": actual})
    return counts, failures


def _evaluate_trajectories(cases: list[dict[str, Any]], module: Any) -> tuple[collections.Counter[str], list[dict[str, Any]]]:
    counts: collections.Counter[str] = collections.Counter()
    failures: list[dict[str, Any]] = []
    for case in cases:
        replay = _replay(module, case["events"])
        actual = {item["index"]: item for item in replay["events"]}
        for label in case.get("labels", []):
            expected = label.get("expected")
            if expected == "unknown":
                counts["unknown_historical_labels"] += 1
                continue
            counts["historical_label_checks"] += 1
            expected_hit = expected == "penalize"
            actual_hit = bool(actual.get(label["event"], {}).get("penalized"))
            if expected_hit != actual_hit:
                counts["historical_label_failures"] += 1
                failures.append({"case": case["id"], "kind": "historical_label", "event": label["event"], "expected": expected, "actual": actual_hit, "category": label.get("category")})
    return counts, failures


def _evaluate_tool_contracts() -> tuple[collections.Counter[str], list[dict[str, Any]]]:
    """Exercise filesystem/search contracts with a temporary local workspace."""
    sys.path.insert(0, str(ROOT.parent / "stock-rl-reflect"))
    tools = importlib.import_module("recipe.swe_agent.tools")
    search_utils = importlib.import_module("recipe.swe_agent.search_utils")
    counts: collections.Counter[str] = collections.Counter()
    failures: list[dict[str, Any]] = []

    def json_safe(value: Any) -> Any:
        """Keep failure reports serializable without weakening comparisons."""
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(key): json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_safe(item) for item in value]
        return value

    def check(case: str, expected: Any, actual: Any) -> None:
        counts["tool_contract_checks"] += 1
        if expected != actual:
            counts["tool_contract_failures"] += 1
            failures.append({
                "case": case,
                "kind": "tool_contract",
                "expected": json_safe(expected),
                "actual": json_safe(actual),
            })

    with tempfile.TemporaryDirectory(prefix="swe-tool-contract-") as directory:
        workspace = Path(directory) / "repo"
        (workspace / "src").mkdir(parents=True)
        (workspace / "src" / "a.py").write_text("needle = 1\n", encoding="utf-8")
        task = {"docker_mount": "/testbed"}
        check("relative_path_resolves", (workspace / "src" / "a.py").resolve(), tools.resolve_model_path(workspace, "src/a.py", task))
        try:
            mapped = tools.resolve_model_path(workspace, "/testbed/src/a.py", task)
            check("declared_mount_path_resolves", (workspace / "src" / "a.py").resolve(), mapped)
        except Exception as exc:  # expected to fail until mount normalization is unified
            check("declared_mount_path_resolves", (workspace / "src" / "a.py").resolve(), f"error:{type(exc).__name__}")
        for case, raw_path in (("unrelated_absolute_rejected", "/tmp/outside.py"), ("parent_traversal_rejected", "../outside.py")):
            try:
                tools.resolve_model_path(workspace, raw_path, task)
                actual = "accepted"
            except Exception:
                actual = "rejected"
            check(case, "rejected", actual)
        check("alias_grep", "search_text", tools.normalize_tool_name("Grep"))
        check("alias_read", "read_file", tools.normalize_tool_name("Read"))
        check("search_files_finds_name", True, "src/a.py" in search_utils.search_files(workspace, {"query": "a.py", "path": "src"}))
        check("search_text_finds_content", True, "needle = 1" in search_utils.search_text(workspace, {"pattern": "needle", "path": "src"}))
        missing = search_utils._StreamResult([], None, error="rg missing", backend_missing=True)
        original_stream = search_utils._stream_command
        def missing_rg(command, **kwargs):
            return missing if command[0] == "rg" else original_stream(command, **kwargs)
        with mock.patch.object(search_utils, "_stream_command", side_effect=missing_rg):
            fallback = search_utils.search_text(workspace, {"pattern": "needle", "path": "src"})
        check("search_missing_rg_has_fallback", True, "src/a.py:1:needle = 1" in fallback)
        with mock.patch.object(search_utils, "_stream_command", side_effect=missing_rg):
            fallback_files = search_utils.search_files(workspace, {"query": "a.py", "path": "src"})
        check("file_search_missing_rg_has_fallback", True, "src/a.py" in fallback_files)
    return counts, failures


def evaluate(directory: Path, output: Path) -> dict[str, Any]:
    manifest, frozen_contracts, trajectory_cases = _load_benchmark(directory)
    module = _load_detector()
    contract_counts, contract_failures = _evaluate_contracts(frozen_contracts, module)
    trajectory_counts, trajectory_failures = _evaluate_trajectories(trajectory_cases, module)
    tool_counts, tool_failures = _evaluate_tool_contracts()
    counts = contract_counts + trajectory_counts + tool_counts
    report = {
        "version": VERSION,
        "benchmark_manifest_sha256": digest(manifest),
        "counts": dict(counts),
        "failures": contract_failures + trajectory_failures + tool_failures,
        "regression_pass": not contract_failures and not trajectory_failures and not tool_failures,
        "unknown_policy": "unknown labels are reported and never counted as passes",
        "source_fingerprint": importlib.import_module("recipe.swe_agent.tool_contract_info").source_fingerprint(),
    }
    _write_json(output, report)
    return report


def probe(urls: list[str], output: Path, timeout: float) -> dict[str, Any]:
    """Read-only endpoint probe; no claim/execute request is issued."""
    results = []
    for raw_url in urls:
        base = raw_url.rstrip("/")
        endpoint_result: dict[str, Any] = {"url": base, "endpoints": {}}
        for endpoint in ("/live", "/health", "/ready"):
            request = urllib.request.Request(base + endpoint, method="GET")
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    body = response.read(256 * 1024).decode("utf-8", errors="replace")
                    try:
                        parsed: Any = json.loads(body)
                    except json.JSONDecodeError:
                        parsed = {"raw": body[:1000]}
                    endpoint_result["endpoints"][endpoint] = {"http_status": response.status, "body": parsed}
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                endpoint_result["endpoints"][endpoint] = {"error": str(exc)}
        results.append(endpoint_result)
    report = {"version": VERSION, "read_only": True, "urls": results, "healthy_count": sum(1 for item in results if item["endpoints"].get("/health", {}).get("http_status") == 200)}
    _write_json(output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--output-dir", type=Path, required=True)
    build_parser.add_argument("--trajectory-rows", type=Path, default=DEFAULT_TRAJECTORIES)
    build_parser.add_argument("--max-trajectory-cases", type=int, default=512)
    eval_parser = subparsers.add_parser("evaluate")
    eval_parser.add_argument("--benchmark-dir", type=Path, required=True)
    eval_parser.add_argument("--report", type=Path, required=True)
    probe_parser = subparsers.add_parser("probe")
    probe_parser.add_argument("--urls", default="", help="comma-separated execution service base URLs; defaults to SWE_AGENT_EXECUTION_URLS")
    probe_parser.add_argument("--output", type=Path, required=True)
    probe_parser.add_argument("--timeout", type=float, default=5.0)
    args = parser.parse_args()
    if args.command == "build":
        report = build(args.output_dir, args.trajectory_rows, max(1, args.max_trajectory_cases))
    elif args.command == "evaluate":
        report = evaluate(args.benchmark_dir, args.report)
    else:
        import os
        urls = [item.strip() for item in (args.urls or os.environ.get("SWE_AGENT_EXECUTION_URLS", "")).split(",") if item.strip()]
        if not urls:
            parser.error("probe requires --urls or SWE_AGENT_EXECUTION_URLS")
        report = probe(urls, args.output, max(0.1, args.timeout))
    print(json.dumps(report, sort_keys=True))
    return (0 if report["regression_pass"] else 1) if args.command == "evaluate" else 0


if __name__ == "__main__":
    raise SystemExit(main())

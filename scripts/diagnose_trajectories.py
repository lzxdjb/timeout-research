#!/usr/bin/env python3
"""Diagnose a fixed, explicit set of trajectory files.

The command is intentionally narrow: callers provide the files to inspect,
usually the earliest training file for each run.  It never discovers later
steps implicitly.  Every non-padding row is parsed with the production SFT
parser and its repeat counters are reconstructed from dispatched calls and
tool responses.  The report keeps stored and reconstructed values separate.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    from scripts import make_swe_sft_data as production
except ModuleNotFoundError:  # pragma: no cover - supports direct invocation
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts import make_swe_sft_data as production

from recipe.swe_agent.test_results import summarize_test_execution


DEFAULT_TRANSCRIPT_CHARS = 5000
DEFAULT_RESPONSE_CHARS = 500
DEFAULT_EXAMPLES_PER_CATEGORY = 2
REPEAT_CATEGORIES = (
    "highest_penalized_repeats",
    "highest_dispatches",
    "raw_success_with_repeats",
    "zero_reward_without_repeats",
    "tool_errors",
    "budget_or_cap",
    "parser_failures",
    "stored_replay_mismatches",
    "unknown_tools",
    "repeated_search_observations",
    "success_with_empty_patch",
    "search_backend_errors",
    "payload_role_markers",
    "zero_exit_failure_text",
    "pre_generation_failures",
    "no_op_edits",
    "identical_call_observations",
    "artifact_schema_mismatches",
    "environment_errors",
    "zero_test_execution",
    "generated_without_dispatch",
    "repeated_artifact_content",
    "continued_after_warning",
    "search_schema_errors",
    "unpaired_tool_calls",
    "incomplete_test_evidence",
    "conflicting_test_counts",
    "nested_test_summaries",
    "implicit_test_attempts",
    "evaluator_phase_failures",
    "worktree_patch_provenance",
    "repeated_assistant_text",
    "large_assistant_turn",
    "budget_boundary_discrepancy",
    "unresolved_public_failures",
)

# Names accepted by the current SWE agent loop and the execution service.  The
# capitalized entries are the model-facing compatibility aliases; the
# lower-case entries are the canonical service names (plus the two aliases
# accepted by ``normalize_tool_name``).
SUPPORTED_TOOL_NAMES = frozenset(
    {
        "repo_status",
        "read_file",
        "search_files",
        "search_text",
        "write_file",
        "apply_patch",
        "run_shell",
        "run_tests",
        "build_project",
        "glob_files",
        "edit_file",
        "bash",
        "Bash",
        "Read",
        "Write",
        "Edit",
        "Glob",
        "Grep",
    }
)


@dataclass(frozen=True)
class InputSpec:
    path: Path
    label: str


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _integer(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(row: dict[str, Any]) -> tuple[str, str, str]:
    raw = row.get("gts", {})
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            raw = {}
    if not isinstance(raw, dict):
        raw = {}
    benchmark = str(raw.get("benchmark") or "unknown")
    task = str(
        raw.get("instance_id")
        or raw.get("task_id")
        or row.get("task_id")
        or "unknown"
    )
    repository = str(raw.get("repo") or raw.get("repository") or "unknown")
    return benchmark, task, repository


def _bounded(value: Any, limit: int = DEFAULT_RESPONSE_CHARS) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[:limit] + "...[truncated]"


def _response_signals(name: str, arguments: dict[str, Any], response: str) -> dict[str, Any]:
    """Inspect complete responses before truncation; signals require review."""
    search_error = name in {"Grep", "search_text", "search_files", "Glob"} and response.strip().startswith(("Search error:", "Search unavailable:"))
    reason = ""
    if search_error:
        if "unsupported parameters" in response or "must be a positive integer" in response or "max_results conflict" in response:
            reason = "search_schema_error"
        elif "non-ASCII" in response:
            reason = "unicode_backend_limitation"
        elif "budget" in response:
            reason = "scan_budget"
        else:
            reason = "other_search_error"
    shell = name in {"Bash", "bash", "run_shell", "run_tests", "build_project"}
    artifact = bool(re.match(r"(?:exit_code:\s*-?\d+\s*\n|process_exit_code: unavailable\s*\n)?\[tool output stored\]", response.lstrip()))
    artifact_hash = re.search(r"(?m)^artifact_sha256:\s*([0-9a-f]{64})\s*$", response) if artifact else None
    artifact_path = re.search(r"(?m)^artifact_path:\s*(.+)$", response) if artifact else None
    warning = bool(re.search(r"\[tool feedback\]|This command failed again|Repeated filename searches", response))
    # Strip only our known advisory suffixes. Never normalize arbitrary output,
    # timestamps, paths or program text to manufacture observation equality.
    core = re.split(r"\n(?:\[tool feedback\]|\[test feedback\]|\[test outcome\]|This command failed again|Repeated filename searches)", response, maxsplit=1)[0]
    status = re.search(r"(?m)^exit_code:\s*(-?\d+)\b", response) if shell else None
    failure_text = bool(re.search(
        r"(?im)^(?:error(?:\[E\d+\])?:|fatal error:|FAIL\b|FAILED\s*\(|.*: (?:No such file or directory|command not found)$)|\b[1-9]\d* (?:failed|errors?)\b",
        response,
    ))
    retrieval = re.search(r"(?m)^retrieval:\s*(.*)$", response) if artifact else None
    environment_errors = []
    if "No module named pytest" in response or "No module named 'pytest'" in response:
        environment_errors.append("pytest_unavailable")
    if re.search(r"\brg: command not found\b", response):
        environment_errors.append("rg_unavailable")
    if "Docker daemon is not reachable" in response:
        environment_errors.append("docker_daemon_unreachable")
    replay = summarize_test_execution(str(arguments.get("command") or ""), core,
                                     exit_code=int(status.group(1)) if status else None) if shell and not artifact else {}
    return {
        "test_summary_replay": replay,
        "test_summary_replay_provenance": "current_parser_on_saved_response" if replay else "unavailable_artifact_or_non_shell",
        "incomplete_test_evidence": replay.get("test_incomplete_failure_evidence", False),
        "conflicting_test_counts": replay.get("test_count_conflict", False),
        "nested_test_summaries": replay.get("test_summary_count", 0) > 1,
        "implicit_test_attempt": bool(replay.get("test_runner_recognized")) and str(arguments.get("verification", "")).strip().lower() not in {"true", "1", "yes", "on"} and name != "run_tests",
        "search_backend_error": search_error and reason != "search_schema_error",
        "search_schema_error": search_error and reason == "search_schema_error",
        "search_error_reason": reason,
        "zero_exit_failure_text": bool(status and status.group(1) == "0" and failure_text),
        "shell_exit_code_in_response": int(status.group(1)) if status else None,
        "shell_pipeline": shell and bool(re.search(r"(?<!\|)\|(?!\|)", str(arguments.get("command", "")))),
        "no_op_edit": name in {"Edit", "edit_file"} and response.strip().startswith("Edit made no changes:"),
        "output_artifact": artifact,
        "artifact_sha256": artifact_hash.group(1) if artifact_hash else None,
        "artifact_path": artifact_path.group(1) if artifact_path else None,
        "observation_identity": artifact_hash.group(1) if artifact_hash else hashlib.sha256(core.encode()).hexdigest(),
        "advisory_warning": warning,
        "reported_test_outcome": (re.search(r"\[test outcome\]\s*(passed|failed|not_run|unknown)", response).group(1)
                                  if re.search(r"\[test outcome\]\s*(passed|failed|not_run|unknown)", response) else None),
        "explicit_verification": str(arguments.get("verification", "")).strip().lower() in {"true", "1", "yes", "on"},
        "artifact_retrieval_tools": sorted(set(re.findall(r"\b(?:search_text|read_file|Grep|Read)\b", retrieval.group(1)))) if retrieval else [],
        "environment_errors": environment_errors,
        "zero_test_execution": shell and (replay.get("tests_executed") == 0 or bool(re.search(r"(?im)^\s*Ran 0 tests\b|\bno tests ran\b|\bcollected 0 items\b", response))),
        "literal_regex_escape_candidate": name in {"Grep", "search_text"} and (
            str(arguments.get("literal", "")).lower() in {"true", "1"}
            and bool(re.search(r"\\[.()\[\]|*+?]", str(arguments.get("pattern", ""))))
            and response.strip().startswith("No matches.")
        ),
        "public_test_command_candidate": name == "run_tests" or (
            shell and (bool(replay.get("test_runner_recognized")) or bool(re.search(r"\bpytest\b|\bruntests\.py\b|\bpython\S*\s+-m\s+unittest\b", str(arguments.get("command", "")))))
        ),
    }


def _parse_events(messages: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Pair only calls with the responses that prove they were dispatched."""

    events: list[dict[str, Any]] = []
    for message_index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        calls = list(production.TOOL_BLOCK_RE.finditer(message["content"]))
        responses: list[str] = []
        if message_index + 1 < len(messages) and messages[message_index + 1]["role"] == "user":
            responses = production.TOOL_RESPONSE_RE.findall(messages[message_index + 1]["content"])
        for call_match, response in zip(calls, responses):
            call = production._tool_call(call_match)
            events.append(
                {
                    "index": len(events),
                    "batch": message_index,
                    "name": call["name"],
                    "arguments": call["parameters"],
                    "response": _bounded(response),
                    "response_sha256": hashlib.sha256(response.encode()).hexdigest(),
                    "response_chars": len(response),
                    "outcome": production.classify_tool_outcome(call["name"], call["parameters"], response),
                    **_response_signals(call["name"], call["parameters"], response),
                    "verification": bool(call["verification"]),
                    "may_mutate": bool(call["may_mutate"]),
                    "fingerprint": call["fingerprint"].hex(),
                }
            )
    return events


def _repeat_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: dict[str, int] = {}
    repeated: list[dict[str, Any]] = []
    for event in events:
        prior = seen.get(event["fingerprint"])
        if prior is not None:
            repeated.append(
                {
                    "event": event["index"],
                    "prior_event": prior,
                    "name": event["name"],
                    "arguments": event["arguments"],
                    "response": event["response"],
                    "prior_response": events[prior]["response"],
                }
            )
        seen[event["fingerprint"]] = event["index"]
    return repeated


def _tool_errors(messages: list[dict[str, str]]) -> int:
    return sum(
        production.classify_tool_outcome("unknown", {}, response) == "transport_error"
        for message in messages
        if message["role"] == "user"
        for response in production.TOOL_RESPONSE_RE.findall(message["content"])
    )


def _transcript_excerpt(output: str, limit: int) -> str:
    if len(output) <= limit:
        return output
    head = limit // 2
    tail = limit - head
    return output[:head] + "\n...[middle truncated]...\n" + output[-tail:]


def _prompt_tool_names(input_text: str) -> list[str]:
    """Read the serialized tool declarations, without inferring names from prose."""
    names: set[str] = set()
    for block in re.findall(r"<tools>(.*?)</tools>", input_text, re.DOTALL):
        for line in block.splitlines():
            try:
                declaration = json.loads(line)
            except (ValueError, TypeError):
                continue
            if isinstance(declaration, dict):
                function = declaration.get("function")
                if isinstance(function, dict) and isinstance(function.get("name"), str):
                    names.add(function["name"])
    return sorted(names)


def _behavior_evidence(messages: list[dict], events: list[dict], row: dict) -> dict:
    """Review signals only: never infer penalties or silently rewrite rewards."""
    turns = [production.TOOL_BLOCK_RE.sub("", m["content"]) for m in messages if m["role"] == "assistant"]
    paragraphs = [re.sub(r"\s+", " ", p).strip() for text in turns for p in re.split(r"\n\s*\n", text)]
    counts = collections.Counter(p for p in paragraphs if len(p) >= 100)
    repeats = sum(n - 1 for n in counts.values() if n >= 3)
    history, unresolved = [], {}
    for event in events:
        summary = event["test_summary_replay"]
        if not summary.get("test_runner_recognized"):
            continue
        command = str(event["arguments"].get("command", ""))
        key = hashlib.sha256(command.encode()).hexdigest()
        outcome = summary["test_outcome"]
        history.append({"event": event["index"], "command_sha256": key, "framework": summary["test_framework"],
                        "outcome": outcome, "tests_executed": summary["tests_executed"],
                        "explicit_verification": event["explicit_verification"]})
        if outcome == "failed":
            unresolved[key] = event["index"]
        elif outcome == "passed":
            unresolved.pop(key, None)
    gap = max(0, _integer(row.get("trajectory_tool_dispatches")) - len(events))
    recorded_missing = max(0, _integer(row.get("trajectory_tool_observations_not_retained")))
    return {
        "repeated_assistant_paragraphs": repeats,
        "max_assistant_turn_chars": max(map(len, turns), default=0),
        "large_assistant_turn": any(len(text) >= 20000 for text in turns),
        "executed_retained_response_gap": gap,
        "budget_boundary_discrepancy": gap > recorded_missing and not bool(row.get("trajectory_budget_reached")),
        "unresolved_public_failure_events": sorted(unresolved.values()),
        "public_runner_history": history,
        "unresolved_failure_at_submission": bool(unresolved) and row.get("submission_signal_seen") in (True, 1),
        "qualification": "review candidates; changed commands may cover the same target; paragraph repetition is not a penalty verdict",
    }


def _record(
    row: dict[str, Any],
    *,
    source: InputSpec,
    line: int,
    read_window: int,
    exec_window: int,
    transcript_chars: int,
) -> dict[str, Any]:
    benchmark, task, repository = _identity(row)
    output = str(row.get("output") or "")
    pre_generation_failure = (
        not output.strip() and row.get("reward_eligible") == 0
        and bool(row.get("infrastructure_failure_reason"))
    )
    parser_error = ""
    messages: list[dict[str, str]] = []
    events: list[dict[str, Any]] = []
    try:
        messages = production._parse_output(output)
        events = _parse_events(messages)
    except production.DataReject as exc:
        parser_error = "" if pre_generation_failure else str(exc)
    replay = production._reconstructed_repeat_metrics(
        messages,
        read_window=read_window,
        exec_window=exec_window,
    )
    stored_repeat = row.get("trajectory_penalized_repeated_tool_calls")
    stored_repeat_int = None if stored_repeat is None else _integer(stored_repeat)
    repeated = _repeat_events(events)
    declared_tools = _prompt_tool_names(str(row.get("input") or ""))
    supported_tools = set(declared_tools) if declared_tools else SUPPORTED_TOOL_NAMES
    for event in events:
        event["artifact_schema_mismatch"] = bool(declared_tools) and bool(
            set(event["artifact_retrieval_tools"]) - supported_tools
        )
    unknown_tool_names = sorted(
        {event["name"] for event in events if event["name"] not in supported_tools}
    )
    commands = [
        str(event["arguments"].get("command", ""))
        for event in events
        if event["name"].lower() in {"bash", "run_shell"}
    ]
    command_counts = collections.Counter(commands)
    shell_mutations = sum(
        bool(production.SHELL_MUTATION_RE.search(command)) for command in commands
    )
    replay_repeats = _integer(replay.get("penalized_repeat_count"))
    tool_errors = max(_integer(row.get("trajectory_tool_error_returns")), _tool_errors(messages))
    parsed_dispatches = len(events)
    stored_dispatches = _integer(
        row.get("trajectory_tool_dispatches", row.get("tool_dispatches"))
    )
    search_observations = collections.Counter(
        (event["name"], event["response_sha256"])
        for event in events
        if event["name"] in {"Grep", "search_text", "search_files", "Glob"}
    )
    # This is an observation signal, not a penalty verdict: edits can legitimately
    # precede equal search responses. Hash the full observation, never the excerpt.
    same_search_observations = sum(count - 1 for count in search_observations.values())
    observation_groups: dict[tuple[str, str], list[int]] = collections.defaultdict(list)
    for event in events:
        observation_groups[(event["fingerprint"], event["response_sha256"])].append(event["index"])
    identical_groups = [
        {"name": events[indices[0]]["name"], "events": indices, "count": len(indices),
         "fingerprint": fingerprint, "response_sha256": response_hash}
        for (fingerprint, response_hash), indices in observation_groups.items() if len(indices) > 1
    ]
    identical_groups.sort(key=lambda group: (-group["count"], group["events"][0]))
    artifact_groups: dict[str, list[int]] = collections.defaultdict(list)
    warned_observations: set[tuple[str, str]] = set()
    continued_after_warning = 0
    for event in events:
        identity = (event["fingerprint"], event["observation_identity"])
        event["continued_after_warning"] = identity in warned_observations
        continued_after_warning += event["continued_after_warning"]
        if event["advisory_warning"]:
            warned_observations.add(identity)
        if event["artifact_sha256"]:
            artifact_groups[event["artifact_sha256"]].append(event["index"])
    unpaired_calls = 0
    for index, message in enumerate(messages):
        if message["role"] == "assistant":
            calls = len(list(production.TOOL_BLOCK_RE.finditer(message["content"])))
            responses = len(production.TOOL_RESPONSE_RE.findall(messages[index + 1]["content"])) if index + 1 < len(messages) and messages[index + 1]["role"] == "user" else 0
            unpaired_calls += max(0, calls - responses)
    marker_collisions = sum(
        len(production.MARKER_RE.findall(match.group(0)))
        for match in re.finditer(r"<parameter=[^>]+>.*?</parameter>|<tool_response>.*?</tool_response>", output, re.DOTALL)
    )
    record = {
        "source": {"path": str(source.path), "label": source.label, "line": line},
        "uid": row.get("uid"),
        "benchmark": benchmark,
        "task_id": task,
        "repository": repository,
        "step": _integer(row.get("step"), _integer(source.path.stem)),
        "raw_score": _number(row.get("raw_score", row.get("score"))),
        "shaped_score": _number(row.get("shaped_score", row.get("score"))),
        "is_padding": bool(row.get("is_padding")),
        "train_sample_mask": row.get("train_sample_mask"),
        "reward_eligible": row.get("reward_eligible"),
        "infrastructure_failure": row.get("infrastructure_failure"),
        "infrastructure_evidence": {key: row.get(key) for key in (
            "infrastructure_failure_reason", "infrastructure_failure_code",
            "observed_infrastructure_failure", "observed_infrastructure_failure_reason",
            "observed_infrastructure_failure_code", "hidden_failure_phase",
        )},
        "pre_generation_failure": pre_generation_failure,
        "output_present": bool(output.strip()),
        "behavior": _behavior_evidence(messages, events, row),
        "reward_evidence": {key: row.get(key) for key in (
            "hidden_pass", "hidden_tests_exit_code", "hidden_tests_started", "hidden_tests_seconds",
            "final_patch_empty", "final_patch_changed_files", "final_patch_changed_bytes",
            "partial_hidden_reward_count_reason", "submission_failure_reason",
            "verification_attempt_count", "public_verification_available",
            "repeated_tool_reward_shaping_enabled",
            "public_test_attempt_count", "implicit_public_test_attempt_count",
            "observed_public_test_outcome", "observed_public_test_count_conflict",
            "private_evaluator_artifact_status", "private_evaluator_artifact_path",
            "private_evaluator_artifact_sha256", "private_evaluator_artifact_truncated",
            "private_evaluator_service_id", "private_evaluator_storage_scope",
            "private_evaluator_logs_complete",
            "evaluation_valid", "evaluator_failure_excluded", "evaluator_invalid_reason",
            "legacy_patch_metrics_source", "final_worktree_changed_files", "final_worktree_hashed_bytes",
            "final_tracked_diff_available", "final_tracked_diff_bytes", "final_tracked_diff_files",
            "submitted_patch_statistics_available", "hidden_failure_phase",
        )},
        "role_markers_inside_payload": marker_collisions,
        "tool_schema": {"source": "input_declarations" if declared_tools else "current_cross_profile_fallback", "names": sorted(supported_tools)},
        "stored": {
            "penalized_repeats": stored_repeat_int,
            "raw_repeats": _integer(row.get("trajectory_repeated_tool_calls")),
            "tool_dispatches": stored_dispatches,
            "tool_errors": _integer(row.get("trajectory_tool_error_returns")),
            "assistant_turns": _integer(row.get("trajectory_assistant_turns")),
            "response_tokens": _integer(row.get("trajectory_response_tokens")),
            "budget_reached": bool(row.get("trajectory_budget_reached")),
            "response_length_limit": bool(row.get("trajectory_response_length_limit")),
            "max_turn": _integer(row.get("trajectory_max_turn")),
            "termination_reason": row.get("termination_reason"),
            "budget_phase": row.get("trajectory_budget_phase"),
            "tool_observations_not_retained": row.get("trajectory_tool_observations_not_retained"),
            "unretained_tool_responses_json": row.get("trajectory_unretained_tool_responses_json"),
            "detection_mode": row.get("trajectory_repeated_tool_detection_mode"),
        },
        "reconstructed": {
            "penalized_repeats": replay_repeats if not parser_error and not pre_generation_failure else None,
            "raw_exact_repeats": _integer(replay.get("raw_exact_repeat_count")),
            "dispatches": parsed_dispatches,
            "assistant_turns": sum(message["role"] == "assistant" for message in messages),
            "tool_errors": tool_errors,
            "tool_counts": dict(collections.Counter(event["name"] for event in events)),
            "unknown_tool_call_count": sum(event["name"] not in supported_tools for event in events),
            "unknown_tool_names": unknown_tool_names,
            "same_search_observation_count": same_search_observations,
            "search_backend_error_count": sum(e["search_backend_error"] for e in events),
            "search_schema_error_count": sum(e["search_schema_error"] for e in events),
            "unpaired_tool_call_count": unpaired_calls,
            "advisory_warning_count": sum(e["advisory_warning"] for e in events),
            "continued_after_warning_count": continued_after_warning,
            "repeated_artifact_content_count": sum(len(indices) - 1 for indices in artifact_groups.values()),
            "unknown_test_outcome_count": sum(e["reported_test_outcome"] == "unknown" or e["public_test_command_candidate"] and e["reported_test_outcome"] is None for e in events),
            "zero_exit_failure_text_count": sum(e["zero_exit_failure_text"] for e in events),
            "search_error_reasons": dict(collections.Counter(e["search_error_reason"] for e in events if e["search_backend_error"] or e["search_schema_error"])),
            "no_op_edit_count": sum(e["no_op_edit"] for e in events),
            "identical_call_observation_count": sum(g["count"] - 1 for g in identical_groups),
            "output_artifact_count": sum(e["output_artifact"] for e in events),
            "artifact_schema_mismatch_count": sum(e["artifact_schema_mismatch"] for e in events),
            "public_test_command_candidate_count": sum(e["public_test_command_candidate"] for e in events),
            "explicit_verification_call_count": sum(e["explicit_verification"] for e in events),
            "detector_verification_call_count": sum(e["verification"] for e in events),
            "environment_error_counts": dict(collections.Counter(reason for e in events for reason in e["environment_errors"])),
            "zero_test_execution_count": sum(e["zero_test_execution"] for e in events),
            "incomplete_test_evidence_count": sum(e["incomplete_test_evidence"] for e in events),
            "conflicting_test_counts_count": sum(e["conflicting_test_counts"] for e in events),
            "nested_test_summaries_count": sum(e["nested_test_summaries"] for e in events),
            "implicit_test_attempt_count": sum(e["implicit_test_attempt"] for e in events),
            "literal_regex_escape_candidate_count": sum(e["literal_regex_escape_candidate"] for e in events),
            "exact_repeated_commands": sum(max(0, count - 1) for count in command_counts.values()),
            "shell_mutation_calls": shell_mutations,
            "budget_or_cap": bool(row.get("trajectory_budget_reached")) or stored_dispatches >= 149,
        },
        "parser_error": parser_error,
        "reconstruction_unknown": bool(parser_error) or pre_generation_failure,
        "stored_replay_mismatch": not parser_error and not pre_generation_failure and stored_repeat_int is not None and stored_repeat_int != replay_repeats,
        "identical_call_observation_groups": identical_groups,
        "artifact_content_groups": [{"sha256": digest, "events": indices, "count": len(indices)} for digest, indices in artifact_groups.items() if len(indices) > 1],
        "repeat_events": repeated,
        "events": events,
        "transcript_excerpt": _transcript_excerpt(output, transcript_chars),
    }
    return record


def _input_step(path: Path) -> int | None:
    try:
        return int(path.stem)
    except ValueError:
        return None


def read_input(spec: InputSpec, *, read_window: int, exec_window: int, transcript_chars: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not spec.path.is_file():
        raise FileNotFoundError(spec.path)
    expected_step = _input_step(spec.path)
    records: list[dict[str, Any]] = []
    padding = 0
    row_step_mismatches = 0
    with spec.path.open(encoding="utf-8") as stream:
        for line, text in enumerate(stream, 1):
            row = json.loads(text)
            if row.get("is_padding"):
                padding += 1
                continue
            record = _record(
                row,
                source=spec,
                line=line,
                read_window=read_window,
                exec_window=exec_window,
                transcript_chars=transcript_chars,
            )
            if expected_step is not None and record["step"] != expected_step:
                row_step_mismatches += 1
            records.append(record)
    manifest = {
        "path": str(spec.path.resolve()),
        "label": spec.label,
        "step_from_filename": expected_step,
        "sha256": _sha256(spec.path),
        "rows": len(records),
        "padding_rows_excluded": padding,
        "row_step_mismatches": row_step_mismatches,
    }
    return records, manifest


def _mean(records: list[dict[str, Any]], path: tuple[str, ...]) -> float:
    values: list[float] = []
    for record in records:
        value: Any = record
        for key in path:
            value = value.get(key) if isinstance(value, dict) else None
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values.append(float(value))
    return statistics.fmean(values) if values else 0.0


def _metric_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(records)
    repeat_values = [r["reconstructed"]["penalized_repeats"] for r in records if not r["reconstruction_unknown"]]
    dispatch_values = [max(r["stored"]["tool_dispatches"], r["reconstructed"]["dispatches"]) for r in records]
    raw_success = sum(r["raw_score"] == 1.0 for r in records)
    shaped_success = sum(r["shaped_score"] == 1.0 for r in records)
    repeat_rows = sum(value > 0 for value in repeat_values)
    tool_errors = sum(r["reconstructed"]["tool_errors"] > 0 for r in records)
    return {
        "trajectories": n,
        "raw_success_rate": raw_success / n if n else 0.0,
        "shaped_success_rate": shaped_success / n if n else 0.0,
        "mean_raw_score": _mean(records, ("raw_score",)),
        "mean_shaped_score": _mean(records, ("shaped_score",)),
        "tool_dispatches": {
            "mean": statistics.fmean(dispatch_values) if dispatch_values else 0.0,
            "median": statistics.median(dispatch_values) if dispatch_values else 0.0,
            "max": max(dispatch_values, default=0),
        },
        "assistant_turns_mean": _mean(records, ("reconstructed", "assistant_turns")),
        "repeat_event_rate": repeat_rows / len(repeat_values) if repeat_values else None,
        "repeat_rate_denominator": len(repeat_values),
        "repeat_events_mean": statistics.fmean(repeat_values) if repeat_values else 0.0,
        "repeat_events_p95": _percentile([float(v) for v in repeat_values], 0.95),
        "repeat_events_max": max(repeat_values, default=0),
        "successful_trajectories_with_repeats": sum(
            r["raw_score"] == 1.0 and (r["reconstructed"]["penalized_repeats"] or 0) > 0 for r in records
        ),
        "zero_reward_without_repeats": sum(
            not r["reconstruction_unknown"]
            and r["raw_score"] == 0.0
            and r["reconstructed"]["penalized_repeats"] == 0
            for r in records
        ),
        "tool_error_rate": tool_errors / n if n else 0.0,
        "per_tool_dispatches": dict(
            collections.Counter(
                name
                for record in records
                for name, count in record["reconstructed"]["tool_counts"].items()
                for _ in range(count)
            )
        ),
        "exact_repeated_command_count": sum(
            r["reconstructed"]["exact_repeated_commands"] for r in records
        ),
        "shell_mutation_call_count": sum(r["reconstructed"]["shell_mutation_calls"] for r in records),
        "budget_or_149_call_cap_count": sum(r["reconstructed"]["budget_or_cap"] for r in records),
        "response_truncation_count": sum(r["stored"]["response_length_limit"] for r in records),
        "recorded_budget_reached_count": sum(r["stored"]["budget_reached"] for r in records),
        "recorded_max_turn_cutoff_count": sum(bool(r["stored"]["max_turn"]) or r["stored"]["termination_reason"] in {"max_turns", "max_turn", "max_turn_reached"} for r in records),
        "causal_reason_policy_conflict_count": sum(bool(r["infrastructure_evidence"]["infrastructure_failure_reason"]) and r["infrastructure_evidence"]["observed_infrastructure_failure_reason"] in {"validation_binary", "accepted_binary", "binary_failure", "shaping_disabled", "swe_terminal_tool_failure"} and r["infrastructure_evidence"]["infrastructure_failure_reason"] != r["infrastructure_evidence"]["observed_infrastructure_failure_reason"] for r in records),
        "evaluator_failure_excluded_count": sum(r["reward_evidence"].get("evaluator_failure_excluded") is True or r["reward_evidence"].get("evaluator_failure_excluded") == 1 for r in records),
        "private_evaluator_collection": dict(collections.Counter(r.get("private_evaluator_collection", {}).get("status", "not_requested") for r in records)),
        "private_evaluator_complete_logs_count": sum(r.get("private_evaluator_collection", {}).get("logs_complete") is True for r in records),
        "repeated_assistant_text_count": sum(r["behavior"]["repeated_assistant_paragraphs"] > 0 for r in records),
        "large_assistant_turn_count": sum(r["behavior"]["large_assistant_turn"] for r in records),
        "budget_boundary_discrepancy_count": sum(r["behavior"]["budget_boundary_discrepancy"] for r in records),
        "unresolved_public_failure_at_submission_count": sum(r["behavior"]["unresolved_failure_at_submission"] for r in records),
        "empty_final_patch_count": sum(r["reward_evidence"].get("final_patch_empty") == 1 for r in records),
        "parser_failure_count": sum(bool(r["parser_error"]) for r in records),
        "reconstruction_unknown_count": sum(r["reconstruction_unknown"] for r in records),
        "payload_role_marker_trajectory_count": sum(r["role_markers_inside_payload"] > 0 for r in records),
        "final_patch_metadata_known_count": sum(r["reward_evidence"]["final_patch_empty"] in (0, 1) for r in records),
        "zero_exit_failure_text_count": sum(r["reconstructed"]["zero_exit_failure_text_count"] for r in records),
        "success_with_empty_patch_count": sum(r["raw_score"] == 1 and r["reward_evidence"]["final_patch_empty"] == 1 for r in records),
        "search_backend_error_count": sum(r["reconstructed"]["search_backend_error_count"] for r in records),
        "search_backend_error_trajectory_count": sum(r["reconstructed"]["search_backend_error_count"] > 0 for r in records),
        "same_search_observation_count": sum(r["reconstructed"]["same_search_observation_count"] for r in records),
        "unknown_tool_call_count": sum(r["reconstructed"]["unknown_tool_call_count"] for r in records),
        "unknown_tool_trajectory_count": sum(
            r["reconstructed"]["unknown_tool_call_count"] > 0 for r in records
        ),
        "unknown_tool_names": sorted(
            {
                name
                for r in records
                for name in r["reconstructed"]["unknown_tool_names"]
            }
        ),
        "stored_replay_mismatch_count": sum(r["stored_replay_mismatch"] for r in records),
        "infrastructure_failure_count": sum(bool(r["infrastructure_failure"]) for r in records),
        "observed_infrastructure_failure_count": sum(bool(r["infrastructure_evidence"]["observed_infrastructure_failure"]) for r in records),
        "infrastructure_failure_reasons": dict(collections.Counter(
            r["infrastructure_evidence"]["infrastructure_failure_reason"] for r in records
            if r["infrastructure_evidence"]["infrastructure_failure_reason"]
        )),
        "pre_generation_failure_count": sum(r["pre_generation_failure"] for r in records),
        "search_error_reasons": dict(sum((collections.Counter(r["reconstructed"]["search_error_reasons"]) for r in records), collections.Counter())),
        "environment_error_counts": dict(sum((collections.Counter(r["reconstructed"]["environment_error_counts"]) for r in records), collections.Counter())),
        "generated_without_dispatch_count": sum(r["output_present"] and not r["reconstruction_unknown"] and not r["events"] for r in records),
        "stored_verification_attempt_count": sum(_integer(r["reward_evidence"]["verification_attempt_count"]) for r in records),
        "public_verification_available_count": sum(r["reward_evidence"]["public_verification_available"] == 1 for r in records),
        "repeat_shaping_configuration": dict(collections.Counter(str(r["reward_evidence"]["repeated_tool_reward_shaping_enabled"]) for r in records)),
        **{key: sum(r["reconstructed"][key] for r in records) for key in (
            "no_op_edit_count", "identical_call_observation_count", "output_artifact_count",
            "artifact_schema_mismatch_count", "public_test_command_candidate_count", "explicit_verification_call_count",
            "detector_verification_call_count",
            "zero_test_execution_count", "literal_regex_escape_candidate_count",
            "search_schema_error_count", "unpaired_tool_call_count", "advisory_warning_count",
            "continued_after_warning_count", "repeated_artifact_content_count", "unknown_test_outcome_count",
            "incomplete_test_evidence_count", "conflicting_test_counts_count", "nested_test_summaries_count", "implicit_test_attempt_count",
        )},
        "cohorts": {
            label: _cohort_summary(cohort) for label, cohort in (
                ("reward_eligible", [r for r in records if r["reward_eligible"] == 1]),
                ("reward_ineligible", [r for r in records if r["reward_eligible"] == 0]),
                ("eligibility_unknown", [r for r in records if r["reward_eligible"] not in (0, 1)]),
                ("generated_output", [r for r in records if r["output_present"]]),
            )
        },
        "raw_success_with_replay_repeat_count": sum(
            r["raw_score"] == 1.0 and (r["reconstructed"]["penalized_repeats"] or 0) > 0 for r in records
        ),
    }


def _cohort_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(records)
    dispatches = [max(r["stored"]["tool_dispatches"], r["reconstructed"]["dispatches"]) for r in records]
    return {
        "rows": count,
        "successes": sum(r["raw_score"] == 1 for r in records),
        "raw_success_rate": sum(r["raw_score"] == 1 for r in records) / count if count else None,
        "mean_dispatches": statistics.fmean(dispatches) if dispatches else None,
        "median_dispatches": statistics.median(dispatches) if dispatches else None,
        "budget_or_cap_count": sum(r["reconstructed"]["budget_or_cap"] for r in records),
    }


def _selection(records: list[dict[str, Any]], per_category: int) -> dict[str, list[dict[str, Any]]]:
    def rank(record: dict[str, Any], category: str) -> tuple[Any, ...]:
        if category == "highest_penalized_repeats":
            return (-(record["reconstructed"]["penalized_repeats"] or 0), -len(record["events"]), record["source"]["line"])
        if category == "highest_dispatches":
            return (-max(record["stored"]["tool_dispatches"], record["reconstructed"]["dispatches"]), record["source"]["line"])
        if category == "raw_success_with_repeats":
            return (-(record["reconstructed"]["penalized_repeats"] or 0), -record["raw_score"], record["source"]["line"])
        if category == "zero_reward_without_repeats":
            return (-max(record["stored"]["tool_dispatches"], record["reconstructed"]["dispatches"]), record["source"]["line"])
        if category == "tool_errors":
            return (-record["reconstructed"]["tool_errors"], -len(record["events"]), record["source"]["line"])
        if category == "budget_or_cap":
            return (-max(record["stored"]["tool_dispatches"], record["reconstructed"]["dispatches"]), record["source"]["line"])
        if category == "repeated_search_observations":
            return (-record["reconstructed"]["same_search_observation_count"], record["source"]["line"])
        if category == "zero_exit_failure_text":
            return (-record["reconstructed"]["zero_exit_failure_text_count"], record["source"]["line"])
        if category == "search_backend_errors":
            return (-record["reconstructed"]["search_backend_error_count"], record["source"]["line"])
        if category == "unknown_tools":
            return (-record["reconstructed"]["unknown_tool_call_count"], record["source"]["line"])
        if category == "identical_call_observations":
            return (-record["reconstructed"]["identical_call_observation_count"], record["source"]["line"])
        if category == "no_op_edits":
            return (-record["reconstructed"]["no_op_edit_count"], record["source"]["line"])
        key = {"repeated_artifact_content": "repeated_artifact_content_count", "continued_after_warning": "continued_after_warning_count", "search_schema_errors": "search_schema_error_count", "unpaired_tool_calls": "unpaired_tool_call_count"}.get(category)
        if key:
            return (-record["reconstructed"][key], record["source"]["line"])
        return (record["source"]["line"],)

    predicates = {
        "repeated_assistant_text": lambda r: r["behavior"]["repeated_assistant_paragraphs"] > 0,
        "large_assistant_turn": lambda r: r["behavior"]["large_assistant_turn"],
        "budget_boundary_discrepancy": lambda r: r["behavior"]["budget_boundary_discrepancy"],
        "unresolved_public_failures": lambda r: r["behavior"]["unresolved_failure_at_submission"],
        "incomplete_test_evidence": lambda r: r["reconstructed"]["incomplete_test_evidence_count"] > 0,
        "conflicting_test_counts": lambda r: r["reconstructed"]["conflicting_test_counts_count"] > 0,
        "nested_test_summaries": lambda r: r["reconstructed"]["nested_test_summaries_count"] > 0,
        "implicit_test_attempts": lambda r: r["reconstructed"]["implicit_test_attempt_count"] > 0,
        "evaluator_phase_failures": lambda r: r["reward_evidence"]["hidden_failure_phase"] not in {None, "none", "unknown", "not_scored"},
        "worktree_patch_provenance": lambda r: r["reward_evidence"]["final_patch_changed_bytes"] is not None or r["reward_evidence"]["final_worktree_hashed_bytes"] is not None,
        "repeated_artifact_content": lambda r: r["reconstructed"]["repeated_artifact_content_count"] > 0,
        "continued_after_warning": lambda r: r["reconstructed"]["continued_after_warning_count"] > 0,
        "search_schema_errors": lambda r: r["reconstructed"]["search_schema_error_count"] > 0,
        "unpaired_tool_calls": lambda r: r["reconstructed"]["unpaired_tool_call_count"] > 0,
        "highest_penalized_repeats": lambda r: (r["reconstructed"]["penalized_repeats"] or 0) > 0,
        "highest_dispatches": lambda r: True,
        "raw_success_with_repeats": lambda r: r["raw_score"] == 1.0 and (r["reconstructed"]["penalized_repeats"] or 0) > 0,
        "zero_reward_without_repeats": lambda r: (
            not r["reconstruction_unknown"]
            and r["raw_score"] == 0.0
            and r["reconstructed"]["penalized_repeats"] == 0
        ),
        "tool_errors": lambda r: r["reconstructed"]["tool_errors"] > 0,
        "budget_or_cap": lambda r: r["reconstructed"]["budget_or_cap"],
        "parser_failures": lambda r: bool(r["parser_error"]),
        "stored_replay_mismatches": lambda r: r["stored_replay_mismatch"],
        "unknown_tools": lambda r: r["reconstructed"]["unknown_tool_call_count"] > 0,
        "repeated_search_observations": lambda r: r["reconstructed"]["same_search_observation_count"] > 0,
        "search_backend_errors": lambda r: r["reconstructed"]["search_backend_error_count"] > 0,
        "success_with_empty_patch": lambda r: r["raw_score"] == 1 and r["reward_evidence"]["final_patch_empty"] == 1,
        "payload_role_markers": lambda r: r["role_markers_inside_payload"] > 0,
        "zero_exit_failure_text": lambda r: r["reconstructed"]["zero_exit_failure_text_count"] > 0,
        "pre_generation_failures": lambda r: r["pre_generation_failure"],
        "no_op_edits": lambda r: r["reconstructed"]["no_op_edit_count"] > 0,
        "identical_call_observations": lambda r: r["reconstructed"]["identical_call_observation_count"] > 0,
        "artifact_schema_mismatches": lambda r: r["reconstructed"]["artifact_schema_mismatch_count"] > 0,
        "environment_errors": lambda r: bool(r["reconstructed"]["environment_error_counts"]),
        "zero_test_execution": lambda r: r["reconstructed"]["zero_test_execution_count"] > 0,
        "generated_without_dispatch": lambda r: r["output_present"] and not r["reconstruction_unknown"] and not r["events"],
    }
    selected: dict[str, list[dict[str, Any]]] = {}
    for category in REPEAT_CATEGORIES:
        selected[category] = sorted(
            (r for r in records if predicates[category](r)),
            key=lambda r: rank(r, category),
        )[:per_category]
    return selected


def _markdown(report: dict[str, Any]) -> str:
    lines = ["# Trajectory Diagnosis", "", "This report uses only the explicitly selected input files.", "", *[f"- {item}" for item in report["limitations"]], ""]
    coverage = report["completion_coverage"]
    lines.append(f"Completion coverage: {coverage['status']}. {coverage['reason']}")
    for cutoff in coverage.get("cutoffs", []):
        lines.append(f"Recorded completion cutoff: terminal={cutoff['terminal']}, total={cutoff['total']}, cancelled={cutoff['cancelled']}, threshold={cutoff['threshold']}. Saved rows={coverage['saved_rows']}; reward-eligible rows are reported separately below.")
    lines.append("")
    if report.get("expected_cohort_coverage"):
        cohort = report["expected_cohort_coverage"]
        lines.append(f"Expected dataset coverage: {cohort['status']}; saved/expected rows {cohort['saved_rows']}/{cohort['expected_rows']}; missing {cohort['missing']}; extra {cohort['extra']}.")
    for group in report["groups"]:
        lines.extend([f"## {group['label']} (step {group['step']})", ""])
        metrics = group["metrics"]
        lines.append(f"Assistant repetition/large-turn candidates: {metrics['repeated_assistant_text_count']}/{metrics['large_assistant_turn_count']}; budget-boundary discrepancies: {metrics['budget_boundary_discrepancy_count']}; submissions with unresolved public-failure candidates: {metrics['unresolved_public_failure_at_submission_count']}; verified complete private logs: {metrics['private_evaluator_complete_logs_count']}.")
        lines.append(f"Trajectories: {metrics['trajectories']} (padding excluded: {group['padding_rows_excluded']}).")
        eligible = metrics["cohorts"]["reward_eligible"]
        lines.append(f"Current-parser replay (not deployed historical behavior): incomplete failure evidence={metrics['incomplete_test_evidence_count']}, conflicting counts={metrics['conflicting_test_counts_count']}, multiple/nested summaries={metrics['nested_test_summaries_count']}, implicit runner attempts={metrics['implicit_test_attempt_count']}.")
        lines.append("Legacy patch byte counters describe hashed worktree content. Tracked diff statistics exclude untracked files; submitted patch statistics are separately marked unavailable when no patch artifact exists.")
        lines.append(f"Pre-generation failures: {metrics['pre_generation_failure_count']}; policy/observed infrastructure flags: {metrics['infrastructure_failure_count']}/{metrics['observed_infrastructure_failure_count']}; causal reasons: {metrics['infrastructure_failure_reasons']}.")
        lines.append(f"Reward-eligible cohort: {eligible['successes']}/{eligible['rows']} successes; mean/median dispatches: {eligible['mean_dispatches']}/{eligible['median_dispatches']}. Eligibility unknown: {metrics['cohorts']['eligibility_unknown']['rows']} rows.")
        lines.append(
            "Raw success: {:.1%}; shaped success: {:.1%}; mean dispatches: {:.1f}; "
            "repeat-event rate: {:.1%}; repeat p95/max: {:.1f}/{:d}.".format(
                metrics["raw_success_rate"], metrics["shaped_success_rate"], metrics["tool_dispatches"]["mean"],
                metrics["repeat_event_rate"] or 0.0, metrics["repeat_events_p95"], metrics["repeat_events_max"],
            )
        )
        lines.append(
                "Parser failures: {}; reconstruction-unknown: {}; stored/replay mismatches: {}; "
                "unknown tool trajectories/calls: {}/{}; tool-error trajectory rate: {:.1%}; "
                "budget/cap trajectories: {}.".format(
                metrics["parser_failure_count"], metrics["reconstruction_unknown_count"],
                metrics["stored_replay_mismatch_count"],
                metrics["unknown_tool_trajectory_count"], metrics["unknown_tool_call_count"],
                metrics["tool_error_rate"], metrics["budget_or_149_call_cap_count"],
            )
        )
        lines.append(
            f"Search-backend errors: {metrics['search_backend_error_count']} calls in "
            f"{metrics['search_backend_error_trajectory_count']} trajectories; "
            f"successful empty patches: {metrics['success_with_empty_patch_count']}; "
            f"payload role-marker cases: {metrics['payload_role_marker_trajectory_count']}. "
            f"Repeat statistics cover {metrics['repeat_rate_denominator']} parseable trajectories only."
        )
        lines.append("")
        lines.append(f"Search error reasons: {metrics['search_error_reasons']}; no-op edits: {metrics['no_op_edit_count']}; identical call/observation revisits: {metrics['identical_call_observation_count']}; artifact schema mismatches: {metrics['artifact_schema_mismatch_count']}; public-test command candidates/explicit verification calls: {metrics['public_test_command_candidate_count']}/{metrics['explicit_verification_call_count']}.")
        lines.append(f"Environment error calls: {metrics['environment_error_counts']}; zero-test execution signals: {metrics['zero_test_execution_count']}; literal regex-escape miss candidates: {metrics['literal_regex_escape_candidate_count']}; generated trajectories without dispatch: {metrics['generated_without_dispatch_count']}.")
        lines.append(f"Schema errors: {metrics['search_schema_error_count']}; repeated artifact content: {metrics['repeated_artifact_content_count']}; advisory warnings/continued same observations: {metrics['advisory_warning_count']}/{metrics['continued_after_warning_count']}; unpaired calls: {metrics['unpaired_tool_call_count']}; unknown or unreported test outcomes: {metrics['unknown_test_outcome_count']}. These are inspection signals, not penalty verdicts.")
        lines.append(f"Recorded budget/cutoff flags: {metrics['recorded_budget_reached_count']}/{metrics['recorded_max_turn_cutoff_count']}; causal reason/policy conflicts: {metrics['causal_reason_policy_conflict_count']}.")
        lines.append(f"Confirmed evaluator failures excluded: {metrics['evaluator_failure_excluded_count']}; operator private-evidence collection: {metrics['private_evaluator_collection']}. Service-local availability alone does not establish operator retrieval or persistence across restarts.")
        lines.append("")
        for category, records in group["selected"].items():
            if not records:
                continue
            lines.append(f"### {category}")
            for record in records:
                lines.append(
                    f"- line {record['source']['line']}, task `{record['task_id']}`, "
                    f"raw/shaped `{record['raw_score']}/{record['shaped_score']}`, "
                    f"dispatches `{max(record['stored']['tool_dispatches'], record['reconstructed']['dispatches'])}`, "
                    f"replays `{record['reconstructed']['penalized_repeats']}`, "
                    f"parser `{record['parser_error'] or 'ok'}`"
                )
            lines.append("")
    return "\n".join(lines) + "\n"


def collect_private_evaluator(record: dict, mappings: Iterable[str], output_dir: Path) -> dict:
    """Operator-only filesystem collection; never fetch arbitrary recorded URLs.

    Recorded paths are untrusted. Explicit root mapping, confinement, checksum,
    size limit, and task identity are required before retaining private evidence.
    """
    evidence = record["reward_evidence"]
    result = {"status": "not_requested"}
    mappings = list(mappings)
    if not mappings:
        return result
    digest = evidence.get("private_evaluator_artifact_sha256") or ""
    remote = evidence.get("private_evaluator_artifact_path") or ""
    if not re.fullmatch(r"[0-9a-f]{64}", digest) or Path(remote).name != digest + ".json":
        return {"status": "no_valid_reference"}
    candidates = []
    for mapping in mappings:
        source, separator, destination = mapping.partition("=")
        if not separator or not source or not destination:
            raise ValueError("--evaluator-root requires SERVICE_ROOT=LOCAL_ROOT")
        try:
            relative = Path(remote).relative_to(Path(source))
        except ValueError:
            continue
        root = Path(destination).resolve()
        candidate = (root / relative).resolve()
        if root not in candidate.parents:
            return {"status": "rejected_path"}
        candidates.append(candidate)
    if not candidates:
        return {"status": "unmapped_service_path"}
    for path in candidates:
        try:
            with path.open("rb") as stream:
                data = stream.read(1024 * 1024 + 1)
        except OSError:
            continue
        if len(data) > 1024 * 1024:
            return {"status": "oversized"}
        if hashlib.sha256(data).hexdigest() != digest:
            return {"status": "hash_mismatch"}
        try:
            artifact = json.loads(data)
        except ValueError:
            return {"status": "invalid_json"}
        if not isinstance(artifact, dict):
            return {"status": "invalid_json"}
        if artifact.get("task_id") != record["task_id"]:
            return {"status": "task_mismatch"}
        expected_service = evidence.get("private_evaluator_service_id")
        if expected_service and artifact.get("service_id") != expected_service:
            return {"status": "service_mismatch"}
        # Full logs are content-addressed siblings, never arbitrary paths/URLs.
        # Verify bounded decompression before copying any bytes to the report.
        import gzip
        import io
        import zlib
        blobs = {}
        complete = True
        phases = artifact.get("phases", {})
        if not isinstance(phases, dict):
            return {"status": "invalid_json"}
        for phase in phases.values():
            if not isinstance(phase, dict):
                complete = False
                continue
            ref = phase.get("full_log")
            if not ref:
                complete = complete and not phase.get("truncated", True)
                continue
            try:
                blob_digest = ref["sha256"]
                if not re.fullmatch(r"[0-9a-f]{64}", blob_digest) or ref["file"] != blob_digest + ".log.gz" or ref["compression"] != "gzip":
                    return {"status": "invalid_log_reference"}
                blob_path = path.parent / ref["file"]
                if blob_path.is_symlink() or blob_path.resolve().parent != path.parent:
                    return {"status": "rejected_log_path"}
                if blob_digest not in blobs:
                    with blob_path.open("rb") as stream:
                        compressed = stream.read(101 * 1024 * 1024 + 1)
                    if len(compressed) > 101 * 1024 * 1024:
                        return {"status": "oversized_log"}
                    if hashlib.sha256(compressed).hexdigest() != blob_digest or len(compressed) != ref["bytes"]:
                        return {"status": "log_hash_mismatch"}
                    with gzip.GzipFile(fileobj=io.BytesIO(compressed)) as stream:
                        raw = stream.read(100 * 1024 * 1024 + 1)
                    if len(raw) > 100 * 1024 * 1024:
                        return {"status": "oversized_log"}
                    blobs[blob_digest] = (compressed, len(raw), hashlib.sha256(raw).hexdigest())
                _, size, raw_digest = blobs[blob_digest]
                if size != ref["uncompressed_bytes"] or raw_digest != ref["uncompressed_sha256"]:
                    return {"status": "log_hash_mismatch"}
                if type(ref["complete"]) is not bool or type(ref["original_bytes"]) is not int or ref["original_bytes"] < size:
                    return {"status": "invalid_log_reference"}
                if ref["complete"] and (size != ref["original_bytes"] or raw_digest != phase["sha256"]):
                    return {"status": "log_hash_mismatch"}
                complete = complete and ref["complete"]
            except (OSError, EOFError, ValueError, KeyError, TypeError, zlib.error):
                return {"status": "invalid_or_missing_log"}
        private = output_dir / "private_evaluator"
        private.mkdir(parents=True, exist_ok=True, mode=0o700)
        private.chmod(0o700)
        target = private / (digest + ".json")
        # The report links the private file; it never embeds hidden output.
        import os
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "wb") as stream:
            os.fchmod(stream.fileno(), 0o600)
            stream.write(data)
        for blob_digest, (compressed, _, _) in blobs.items():
            fd = os.open(private / (blob_digest + ".log.gz"), os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
            with os.fdopen(fd, "wb") as stream:
                os.fchmod(stream.fileno(), 0o600)
                stream.write(compressed)
        return {"status": "verified", "sha256": digest, "private_copy": str(target),
                "failure_phase": artifact.get("failure_phase"), "service_id": artifact.get("service_id"),
                "logs_complete": complete, "full_log_blobs": len(blobs)}
    return {"status": "not_accessible"}


def diagnose(
    specs: Iterable[InputSpec],
    output_dir: Path,
    *,
    read_window: int = 8,
    exec_window: int = 16,
    transcript_chars: int = DEFAULT_TRANSCRIPT_CHARS,
    examples_per_category: int = DEFAULT_EXAMPLES_PER_CATEGORY,
    inspect: Iterable[str] = (),
    run_log: Path | None = None,
    evaluator_roots: Iterable[str] = (),
    expected_dataset: Path | None = None,
    expected_samples_per_task: int = 1,
) -> dict[str, Any]:
    specs = list(specs)
    inspect = list(inspect)
    evaluator_roots = list(evaluator_roots)
    if len({s.label for s in specs}) != len(specs):
        raise ValueError("Input labels must be unique")
    all_records: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []
    for spec in specs:
        records, manifest = read_input(
            spec,
            read_window=read_window,
            exec_window=exec_window,
            transcript_chars=transcript_chars,
        )
        if not records:
            raise ValueError(f"{spec.path} contains no non-padding trajectories")
        for record in records:
            record["private_evaluator_collection"] = collect_private_evaluator(record, evaluator_roots, output_dir)
        steps = {record["step"] for record in records}
        if len(steps) != 1:
            raise ValueError(f"{spec.path} contains mixed trajectory steps: {sorted(steps)}")
        manifests.append(manifest)
        all_records.extend(records)
        groups.append(
            {
                "label": spec.label,
                "path": str(spec.path),
                "step": next(iter(steps)),
                "padding_rows_excluded": manifest["padding_rows_excluded"],
                "metrics": _metric_summary(records),
                "selected": _selection(records, examples_per_category),
            }
        )
    requested = set(inspect)
    selected_records = {
        (record["source"]["label"], record["source"]["line"]): record
        for group in groups for records in group["selected"].values() for record in records
    }
    for record in all_records:
        ref = f"{record['source']['label']}:{record['source']['line']}"
        if ref in requested:
            selected_records[(record["source"]["label"], record["source"]["line"])] = record
            requested.remove(ref)
    if requested:
        raise ValueError(f"Inspection references absent or padding: {sorted(requested)}")
    report = {
        "schema_version": 7,
        "limitations": [
            "Replay uses current detector code and transcript text; missing execution metadata and historical versions can change counters. read/exec windows are legacy compatibility arguments, ignored by the current replay implementation.",
            "Parser failures retain stored metrics but reconstructed repeat counts are null and excluded from repeat statistics; event-based metrics omit these rows.",
            "Tool names use input declarations when available; otherwise a cross-profile fallback is explicitly marked.",
            "Repeated search observations and zero-exit failure text are review candidates, not proof of a tool failure or a reward rule.",
            "149 calls is a historical heuristic; budget and response-limit fields are reported separately.",
            "Empty ineligible rows with a recorded infrastructure reason are pre-generation failures, not parser failures. Observed and policy infrastructure flags may disagree; original causal reasons are preserved.",
            "Eligible success is conditional on admission and can be selection-biased. A single step does not establish behavior drift over training or absence of rare bugs.",
            "Identical call/observation groups do not prove unchanged workspace state. Shell failure-text and public-test command detection are review heuristics, not authoritative test outcomes.",
        ],
        "diagnosis_source_sha256": _sha256(Path(__file__)),
        "parser_source_sha256": _sha256(Path(production.__file__)),
        "detector_source_sha256": _sha256(Path(sys.modules[production.replay_tool_calls.__module__].__file__)),
        "parser": "scripts.make_swe_sft_data._parse_output",
        "repeat_reconstruction": {
            "function": "scripts.make_swe_sft_data._reconstructed_repeat_metrics",
            "read_window": read_window,
            "exec_window": exec_window,
        },
        "groups": groups,
        "selection_manifest": manifests,
        "total_non_padding_trajectories": len(all_records),
        "detail_trajectories": len(selected_records),
        "requested_inspections": list(inspect),
        "completion_coverage": {"status": "unavailable", "reason": "No explicit run log supplied; saved rows do not prove complete coverage."},
    }
    if run_log is not None:
        if len(specs) != 1:
            raise ValueError("--run-log requires exactly one explicit trajectory input")
        cutoffs = []
        pattern = re.compile(r"Applied val completion-ratio cutoff: terminal=(\d+) total=(\d+) threshold=([\d.]+) requested=(\d+) cancelled=(\d+)")
        with run_log.open(encoding="utf-8", errors="replace") as stream:
            for text in stream:
                match = pattern.search(text)
                if match:
                    terminal, total, threshold, requested, cancelled = match.groups()
                    cutoffs.append({"terminal": int(terminal), "total": int(total), "threshold": float(threshold),
                                    "requested": int(requested), "cancelled": int(cancelled)})
        report["completion_coverage"] = {"status": "recorded_cutoff" if cutoffs else "no_cutoff_line_found",
            "log_path": str(run_log), "log_sha256": _sha256(run_log), "cutoffs": cutoffs,
            "saved_rows": len(all_records), "reason": "Explicit saved cohort and log denominators; absence of a cutoff line does not certify completion."}
    if expected_dataset is not None:
        if len(specs) != 1 or expected_samples_per_task < 1:
            raise ValueError("Expected dataset requires one input and positive samples per task")
        import pyarrow.parquet as pq
        intended = [str(r["extra_info"]["task_id"]) for r in pq.read_table(expected_dataset, columns=["extra_info"]).to_pylist()]
        if len(set(intended)) != len(intended):
            raise ValueError("Expected dataset contains duplicate task IDs")
        expected = collections.Counter({task: expected_samples_per_task for task in intended})
        actual = collections.Counter(r["task_id"] for r in all_records)
        coverage = {"status": "complete" if expected == actual else "incomplete",
                    "dataset": str(expected_dataset), "dataset_sha256": _sha256(expected_dataset),
                    "expected_tasks": len(expected), "expected_rows": sum(expected.values()), "saved_rows": len(all_records),
                    "missing": dict(expected - actual), "extra": dict(actual - expected)}
        report["expected_cohort_coverage"] = coverage
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_dir = output_dir / "details"
    detail_dir.mkdir(exist_ok=True)
    for manifest in manifests:
        with Path(manifest["path"]).open(encoding="utf-8") as source:
            for line, text in enumerate(source, 1):
                record = selected_records.get((manifest["label"], line))
                if record is None:
                    continue
                safe_label = re.sub(r"[^a-zA-Z0-9_-]", "_", manifest["label"])
                name = f"{safe_label}-{hashlib.sha256(manifest['label'].encode()).hexdigest()[:8]}-line{line}"
                row = json.loads(text)
                (detail_dir / f"{name}.json").write_text(json.dumps(row, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                (detail_dir / f"{name}.txt").write_text(str(row.get("input", "")) + "\n\nOUTPUT\n" + str(row.get("output", "")), encoding="utf-8")
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "selection_manifest.json").write_text(json.dumps(manifests, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "selected_trajectories.jsonl").open("w", encoding="utf-8") as stream:
        for group in groups:
            for category, records in group["selected"].items():
                for record in records:
                    stream.write(json.dumps({"category": category, "group": group["label"], "record": record}, ensure_ascii=False) + "\n")
    with (output_dir / "all_trajectories.jsonl").open("w", encoding="utf-8") as stream:
        for record in all_records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    (output_dir / "report.md").write_text(_markdown(report), encoding="utf-8")
    return report


def _cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-log", type=Path, help="optional matching launcher log; retain completion-cutoff denominators without exposing log contents")
    parser.add_argument("--evaluator-root", action="append", default=[], help="Operator-only private evidence mapping SERVICE_ROOT=LOCAL_ROOT; repeatable")
    parser.add_argument("--expected-dataset", type=Path, help="Parquet cohort to compare with the one saved trajectory input")
    parser.add_argument("--expected-samples-per-task", type=int, default=1)
    parser.add_argument("--input", dest="inputs", action="append", required=True, type=Path, help="Explicit JSONL trajectory file; repeat for each cohort.")
    parser.add_argument("--label", dest="labels", action="append", help="Label matching each --input, in order.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--inspect", action="append", default=[], metavar="LABEL:LINE", help="Also export a complete source row and transcript for this 1-based line.")
    parser.add_argument("--read-window", type=int, default=8)
    parser.add_argument("--exec-window", type=int, default=16)
    parser.add_argument("--transcript-chars", type=int, default=DEFAULT_TRANSCRIPT_CHARS)
    parser.add_argument("--examples-per-category", type=int, default=DEFAULT_EXAMPLES_PER_CATEGORY)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _cli()
    args = parser.parse_args(argv)
    if args.read_window < 0 or args.exec_window < 0 or args.transcript_chars < 100 or args.examples_per_category < 1:
        parser.error("window, transcript, and example limits must be positive")
    labels = args.labels or []
    if labels and len(labels) != len(args.inputs):
        parser.error("--label must be supplied once per --input")
    if not labels:
        labels = [path.stem for path in args.inputs]
    specs = [InputSpec(path=path, label=label) for path, label in zip(args.inputs, labels)]
    report = diagnose(
        specs,
        args.output_dir,
        read_window=args.read_window,
        exec_window=args.exec_window,
        transcript_chars=args.transcript_chars,
        examples_per_category=args.examples_per_category,
        inspect=args.inspect,
        run_log=args.run_log,
        evaluator_roots=args.evaluator_root,
        expected_dataset=args.expected_dataset,
        expected_samples_per_task=args.expected_samples_per_task,
    )
    print(json.dumps({"groups": len(report["groups"]), "trajectories": report["total_non_padding_trajectories"], "output_dir": str(args.output_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

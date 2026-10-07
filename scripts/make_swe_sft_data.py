#!/usr/bin/env python3
"""Build quality-filtered multi-turn SFT data from SWE rollout JSONL files.

The rollout files contain rendered text conversations rather than the
``messages`` records consumed by VERL's ``MultiTurnSFTDataset``.  This script
parses that protocol, filters successful trajectories, keeps one candidate per
task, and writes Parquet datasets plus an audit manifest.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

_SWE_REPO_ROOT = Path(__file__).resolve().parents[2] / "stock-rl-reflect"
if str(_SWE_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_SWE_REPO_ROOT))
from recipe.swe_agent.repeated_tool import classify_tool_outcome, is_verification_tool, replay_tool_calls


DEFAULT_RUNS = (
    "fix_tool_bug_small_lr_new_data",
    "fix_tool_bug_use_moe_small_lr_new_data",
    "fix_tool_super_long",
)

PATH_POLICY_RE = re.compile(
    r"repository-relative paths?\s+(?:(?:are|as)\s+)?(?:the\s+)?only",
    re.IGNORECASE,
)
PATH_REJECTION_RE = re.compile(
    r"(?:path rejected\s*:|absolute paths?\b[^\n]{0,120}\bnot allowed|"
    r"path escapes workspace)",
    re.IGNORECASE,
)
MARKER_RE = re.compile(r"(?m)^(assistant|user)\n")
TOOL_BLOCK_RE = re.compile(r"<function=([^>\n]+)>(.*?)</function>", re.DOTALL)
TOOL_RESPONSE_RE = re.compile(r"<tool_response>(.*?)</tool_response>", re.DOTALL)
PARAM_RE = re.compile(r"<parameter=([^>\n]+)>\s*(.*?)\s*</parameter>", re.DOTALL)
ABSOLUTE_PATH_PARAM_NAMES = {"file_path", "path", "directory"}
MUTATING_TOOLS = {"edit", "write", "edit_file", "apply_patch", "write_file"}
TOOL_ERROR_PREFIXES = (
    "Remote execution error:",
    "Error executing tool",
    "Error when executing tool:",
    "Unknown function '",
    "Invalid JSON in arguments for '",
)
SHELL_MUTATION_RE = re.compile(
    r"(?:\bsed\s+-i\b|\bperl\s+-i\b|\bpython(?:3)?\b.*(?:write|open\(|Path\().*['\"]w|"
    r"\bruby\b.*File\.write|\btee\s+[^|])",
    re.IGNORECASE | re.DOTALL,
)


class DataReject(Exception):
    """Raised when a rollout cannot be converted into an SFT example."""

    def __init__(self, reason: str, detail: str = "") -> None:
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason}: {detail}" if detail else reason)


@dataclass
class Candidate:
    task_key: str
    task_id: str
    benchmark: str
    repository: str
    base_commit: str
    source_run: str
    source_step: int | None
    source_file: str
    source_line: int
    source_uid: str
    raw_score: float
    shaped_score: float
    response_tokens: int
    tool_dispatches: int
    repeated_bash_dispatches: int
    raw_exact_repeat_count: int
    penalized_repeat_count: int
    repeats_suppressed_after_mutation: int
    repeats_suppressed_after_error: int
    repeats_suppressed_outside_window: int
    repeats_suppressed_inflight: int
    repeat_detection_mode: str
    repeat_detection_source: str
    tool_error_returns: int
    has_verification: bool
    mutation_evidence: bool
    submission_check: str
    messages: list[dict[str, str]]
    input_text: str
    output_text: str
    transcript_hash: str

    @property
    def quality_key(self) -> tuple[int, int, int, int, int, int, str, int]:
        """Lower is better; prefer protocol-complete and concise traces."""

        return (
            self.penalized_repeat_count,
            0 if self.has_verification else 1,
            self.tool_error_returns,
            self.tool_dispatches,
            self.response_tokens,
            self.raw_exact_repeat_count,
            self.source_run,
            self.source_step if self.source_step is not None else sys.maxsize,
        )


def _as_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off", ""}:
        return False
    return None


def _gts_fields(row: dict[str, Any]) -> tuple[str, str, str, str]:
    raw = row.get("gts", {})
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise DataReject("malformed_gts", str(exc)) from exc
    if not isinstance(raw, dict):
        raise DataReject("malformed_gts", "gts is not an object")

    benchmark = str(raw.get("benchmark") or "")
    task_id = str(raw.get("instance_id") or raw.get("task_id") or raw.get("id") or "")
    repository = str(raw.get("repo") or raw.get("repository") or "")
    base_commit = str(raw.get("base_commit") or "")
    if not benchmark or not task_id:
        raise DataReject("missing_task_identity")
    return benchmark, task_id, repository, base_commit


def _task_key(benchmark: str, task_id: str, repository: str, base_commit: str) -> str:
    return "\x1f".join((benchmark, task_id, repository, base_commit))


def _parse_input(input_text: str) -> tuple[str, str]:
    if not input_text.startswith("system\n"):
        raise DataReject("malformed_input", "missing system marker")
    user_marker = input_text.find("\nuser\n")
    if user_marker < 0:
        raise DataReject("malformed_input", "missing user marker")

    system = input_text[len("system\n") : user_marker]
    user_with_prompt = input_text[user_marker + len("\nuser\n") :]
    generation_marker = user_with_prompt.rfind("\nassistant\n")
    if generation_marker >= 0:
        user = user_with_prompt[:generation_marker]
    else:
        user = user_with_prompt
    if not system.strip() or not user.strip():
        raise DataReject("malformed_input", "empty system or user message")
    return system, user


def _parse_output(output_text: str) -> list[dict[str, str]]:
    """Parse the rendered assistant/user tool-observation protocol.

    The first assistant turn is implicit in rollout ``output``.  Subsequent
    turns are explicitly prefixed by ``user`` and ``assistant`` markers.
    """

    if not output_text.strip():
        raise DataReject("malformed_output", "empty output")
    markers = list(MARKER_RE.finditer(output_text))
    messages: list[dict[str, str]] = []

    first_end = markers[0].start() if markers else len(output_text)
    first_content = output_text[:first_end].strip()
    if first_content:
        messages.append({"role": "assistant", "content": first_content})

    for index, marker in enumerate(markers):
        role = marker.group(1)
        end = markers[index + 1].start() if index + 1 < len(markers) else len(output_text)
        content = output_text[marker.end() : end].strip()
        if not content:
            raise DataReject("malformed_output", f"empty {role} turn at marker {index}")
        messages.append({"role": role, "content": content})

    if not messages or messages[0]["role"] != "assistant":
        raise DataReject("malformed_output", "first turn is not assistant")
    for previous, current in zip(messages, messages[1:]):
        if previous["role"] == current["role"]:
            raise DataReject("malformed_output", "conversation roles do not alternate")
    if messages[-1]["role"] != "assistant":
        raise DataReject("incomplete_output", "trajectory ends with a tool response")
    return messages


def _tool_call(block: re.Match[str]) -> dict[str, Any]:
    name = block.group(1).strip()
    parameters = {key: value.strip() for key, value in PARAM_RE.findall(block.group(2))}
    verification = is_verification_tool(name, parameters)
    command = parameters.get("command", "")
    may_mutate = name.lower() in MUTATING_TOOLS or (
        name.lower() in {"bash", "run_shell"}
        and not verification
        and bool(SHELL_MUTATION_RE.search(command))
    )
    canonical_parameters = dict(parameters)
    if name.lower() in {"bash", "run_shell"} and isinstance(canonical_parameters.get("command"), str):
        canonical_parameters["command"] = canonical_parameters["command"].strip()
    canonical = json.dumps(canonical_parameters, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return {
        "name": name,
        "parameters": parameters,
        "fingerprint": hashlib.sha256(f"{name}\0{canonical}".encode()).digest(),
        "verification": verification,
        "may_mutate": may_mutate,
    }


def _reconstructed_repeat_metrics(
    messages: list[dict[str, str]],
    *,
    read_window: int,
    exec_window: int,
) -> dict[str, Any]:
    seen: set[bytes] = set()
    seen_bash: set[bytes] = set()
    events: list[dict[str, Any]] = []
    result = {
        "raw_exact_repeat_count": 0,
        "raw_exact_bash_repeat_count": 0,
        "penalized_repeat_count": 0,
        "repeats_suppressed_after_mutation": 0,
        "repeats_suppressed_after_error": 0,
        "repeats_suppressed_outside_window": 0,
        "repeats_suppressed_inflight": 0,
    }

    for message_index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        responses: list[str] = []
        if message_index + 1 < len(messages) and messages[message_index + 1]["role"] == "user":
            responses = TOOL_RESPONSE_RE.findall(messages[message_index + 1]["content"])
        emitted_blocks = list(TOOL_BLOCK_RE.finditer(message["content"]))
        # The rollout executes at most max_parallel_calls from an assistant
        # turn. Historical rows do not store that per-turn limit, but each
        # dispatched call has one ordered tool response, so the response count
        # identifies the dispatched prefix without assuming a fixed limit.
        calls = [_tool_call(block) for block in emitted_blocks[: len(responses)]]
        if not calls:
            continue
        for call, response in zip(calls, responses):
            fingerprint = call["fingerprint"]
            if fingerprint in seen:
                result["raw_exact_repeat_count"] += 1
            seen.add(fingerprint)
            if call["name"].lower() in {"bash", "run_shell"}:
                command = call["parameters"].get("command", "").strip()
                command_fingerprint = hashlib.sha256(command.encode()).digest()
                if command_fingerprint in seen_bash:
                    result["raw_exact_bash_repeat_count"] += 1
                seen_bash.add(command_fingerprint)

            events.append({
                "name": call["name"],
                "arguments": call["parameters"],
                "response": response,
                "batch": message_index,
            })
    detected = replay_tool_calls(events)
    result.update(detected)
    result["repeats_suppressed_after_mutation"] = detected["repeats_suppressed_after_progress"]
    result["repeats_suppressed_after_error"] = detected["repeats_suppressed_after_error"]
    result["repeats_suppressed_inflight"] = detected["repeats_suppressed_inflight"]
    return result


def _repeat_metrics(
    row: dict[str, Any],
    reconstructed: dict[str, Any],
    *,
    detection_mode: str,
    read_window: int,
    exec_window: int,
) -> dict[str, Any]:
    if detection_mode == "legacy_exact":
        raw_stored = row.get("trajectory_repeated_tool_calls")
        raw_count = (
            _as_int(raw_stored)
            if raw_stored is not None
            else reconstructed["raw_exact_repeat_count"]
        )
        return {
            **reconstructed,
            "raw_exact_repeat_count": raw_count,
            "penalized_repeat_count": raw_count,
            "repeat_detection_source": "stored_legacy_exact" if raw_stored is not None else "reconstructed",
        }

    return {
        **reconstructed,
        "raw_exact_repeat_count": reconstructed["raw_exact_repeat_count"],
        "repeat_detection_source": "reconstructed",
    }


def _tool_metrics(
    row: dict[str, Any],
    messages: list[dict[str, str]],
    *,
    detection_mode: str,
    read_window: int,
    exec_window: int,
) -> dict[str, Any]:
    blocks = [
        block
        for message in messages
        if message["role"] == "assistant"
        for block in TOOL_BLOCK_RE.finditer(message["content"])
    ]
    absolute_paths: list[str] = []
    mutating = False
    for block in blocks:
        call = _tool_call(block)
        tool_name = call["name"]
        parameters = call["parameters"]
        for name, value in parameters.items():
            if name in ABSOLUTE_PATH_PARAM_NAMES and (
                value.startswith("/") or ".." in Path(value).parts
            ):
                absolute_paths.append(f"{tool_name}:{name}={value[:200]}")
        if tool_name.lower() in MUTATING_TOOLS:
            mutating = True
        if tool_name.lower() in {"bash", "run_shell"} and SHELL_MUTATION_RE.search(
            parameters.get("command", "")
        ):
            mutating = True
    reconstructed = _reconstructed_repeat_metrics(
        messages,
        read_window=read_window,
        exec_window=exec_window,
    )
    reconstructed_tool_errors = sum(
        classify_tool_outcome(
            "unknown", {}, response
        ) == "transport_error"
        for message in messages
        if message["role"] == "user"
        for response in TOOL_RESPONSE_RE.findall(message["content"])
    )
    repeats = _repeat_metrics(
        row,
        reconstructed,
        detection_mode=detection_mode,
        read_window=read_window,
        exec_window=exec_window,
    )
    return {
        "tool_blocks": len(blocks),
        "absolute_paths": absolute_paths,
        "duplicate_calls": repeats["raw_exact_repeat_count"],
        "has_verification": any(_tool_call(block)["verification"] for block in blocks),
        "mutation_evidence": mutating,
        "reconstructed_tool_error_returns": reconstructed_tool_errors,
        **repeats,
    }


def _path_policy_ok(input_text: str) -> bool:
    return bool(PATH_POLICY_RE.search(input_text))


def _has_path_rejection(output_text: str) -> bool:
    # Path diagnostics are emitted by the tool response.  Restricting the
    # search to that channel avoids rejecting a successful patch whose source
    # code merely discusses workspace paths.
    responses = TOOL_RESPONSE_RE.findall(output_text)
    haystack = "\n".join(responses) if responses else output_text
    return bool(PATH_REJECTION_RE.search(haystack))


def _is_swe_benchmark(benchmark: str) -> bool:
    return "swe" in benchmark.lower()


def _read_stat(row: dict[str, Any], primary: str, fallback: str) -> int:
    value = row.get(primary)
    if value is None:
        value = row.get(fallback)
    return _as_int(value)


def _base_rejection(
    row: dict[str, Any],
    *,
    require_path_policy: bool,
    max_response_tokens: int,
    max_tool_dispatches: int,
    max_penalized_repeated_tool_calls: int,
    repeated_tool_detection_mode: str,
    repeated_tool_read_window: int,
    repeated_tool_exec_window: int,
    require_swe_mutation: bool,
) -> tuple[dict[str, Any], dict[str, str], dict[str, Any], list[dict[str, str]]]:
    if row.get("is_padding"):
        raise DataReject("padding")
    raw_score = _as_float(row.get("raw_score", row.get("score")))
    shaped_score = _as_float(row.get("shaped_score", row.get("score")), raw_score)
    if raw_score != 1.0:
        raise DataReject("not_reward_one")
    if row.get("train_sample_mask") is not True:
        raise DataReject("train_sample_mask_false_or_missing")
    if row.get("completion_ratio_cutoff"):
        raise DataReject("completion_cutoff")
    if row.get("trajectory_timeout"):
        raise DataReject("timeout")
    if row.get("trajectory_terminal_tool_failure"):
        raise DataReject("terminal_tool_failure")
    if row.get("trajectory_budget_reached"):
        raise DataReject("budget_reached")

    benchmark, task_id, repository, base_commit = _gts_fields(row)
    input_text = str(row.get("input") or "")
    output_text = str(row.get("output") or "")
    system, user = _parse_input(input_text)
    output_messages = _parse_output(output_text)
    submission_seen = row.get("submission_signal_seen")
    if submission_seen is None:
        submission_seen = row.get("repeated_tool_submission_seen")
    submission_seen = _as_optional_bool(submission_seen)
    if submission_seen is False:
        raise DataReject("missing_submission")
    submission_check = "explicit" if submission_seen is not None else "inferred_final_assistant"
    if require_path_policy and not _path_policy_ok(input_text):
        raise DataReject("old_or_missing_path_policy")
    if _has_path_rejection(output_text):
        raise DataReject("path_violation")

    metrics = _tool_metrics(
        row,
        output_messages,
        detection_mode=repeated_tool_detection_mode,
        read_window=repeated_tool_read_window,
        exec_window=repeated_tool_exec_window,
    )
    tool_error_returns = max(
        _as_int(row.get("trajectory_tool_error_returns")),
        int(metrics["reconstructed_tool_error_returns"]),
    )
    if tool_error_returns > 0:
        raise DataReject("tool_error")
    if metrics["absolute_paths"]:
        raise DataReject("absolute_file_tool_path")
    response_tokens = _read_stat(row, "trajectory_response_tokens", "response_tokens")
    tool_dispatches = _read_stat(row, "trajectory_tool_dispatches", "tool_dispatches")
    repeated_bash = max(
        _as_int(row.get("trajectory_repeated_bash_dispatches")),
        int(metrics["raw_exact_bash_repeat_count"]),
    )
    tool_dispatches = max(tool_dispatches, int(metrics["tool_blocks"]))
    if response_tokens > max_response_tokens:
        raise DataReject("response_too_long")
    if tool_dispatches > max_tool_dispatches:
        raise DataReject("too_many_tool_dispatches")
    if metrics["penalized_repeat_count"] > max_penalized_repeated_tool_calls:
        raise DataReject("repeated_tool_call")
    mutation_evidence = bool(metrics["mutation_evidence"])
    if require_swe_mutation and _is_swe_benchmark(benchmark):
        patch_empty = row.get("final_patch_empty")
        if patch_empty is not None and bool(patch_empty):
            raise DataReject("empty_swe_patch")
        if patch_empty is None and not mutation_evidence:
            raise DataReject("missing_swe_mutation_evidence")

    metadata = {
        "benchmark": benchmark,
        "task_id": task_id,
        "repository": repository,
        "base_commit": base_commit,
        "raw_score": float(raw_score),
        "shaped_score": float(shaped_score),
        "response_tokens": response_tokens,
        "tool_dispatches": tool_dispatches,
        "repeated_bash_dispatches": repeated_bash,
        "raw_exact_repeat_count": int(metrics["raw_exact_repeat_count"]),
        "penalized_repeat_count": int(metrics["penalized_repeat_count"]),
        "repeats_suppressed_after_mutation": int(metrics["repeats_suppressed_after_mutation"]),
        "repeats_suppressed_after_error": int(metrics["repeats_suppressed_after_error"]),
        "repeats_suppressed_outside_window": int(metrics["repeats_suppressed_outside_window"]),
        "repeats_suppressed_inflight": int(metrics["repeats_suppressed_inflight"]),
        "repeat_detection_mode": repeated_tool_detection_mode,
        "repeat_detection_source": str(metrics["repeat_detection_source"]),
        "tool_error_returns": tool_error_returns,
        "has_verification": bool(metrics["has_verification"]),
        "mutation_evidence": mutation_evidence,
        "submission_check": submission_check,
    }
    return metadata, {"system": system, "user": user}, metrics, output_messages


def _candidate(
    row: dict[str, Any],
    *,
    run_name: str,
    source_file: str,
    source_line: int,
    require_path_policy: bool,
    max_response_tokens: int,
    max_tool_dispatches: int,
    max_penalized_repeated_tool_calls: int,
    repeated_tool_detection_mode: str,
    repeated_tool_read_window: int,
    repeated_tool_exec_window: int,
    require_swe_mutation: bool,
) -> Candidate:
    metadata, prompt, metrics, output_messages = _base_rejection(
        row,
        require_path_policy=require_path_policy,
        max_response_tokens=max_response_tokens,
        max_tool_dispatches=max_tool_dispatches,
        max_penalized_repeated_tool_calls=max_penalized_repeated_tool_calls,
        repeated_tool_detection_mode=repeated_tool_detection_mode,
        repeated_tool_read_window=repeated_tool_read_window,
        repeated_tool_exec_window=repeated_tool_exec_window,
        require_swe_mutation=require_swe_mutation,
    )
    output_text = str(row.get("output") or "")
    messages = [{"role": "system", "content": prompt["system"]}, {"role": "user", "content": prompt["user"]}]
    messages.extend(output_messages)
    transcript_hash = hashlib.sha256(
        (str(row.get("input") or "") + "\x00" + output_text).encode("utf-8")
    ).hexdigest()
    return Candidate(
        task_key=_task_key(
            metadata["benchmark"], metadata["task_id"], metadata["repository"], metadata["base_commit"]
        ),
        task_id=metadata["task_id"],
        benchmark=metadata["benchmark"],
        repository=metadata["repository"],
        base_commit=metadata["base_commit"],
        source_run=run_name,
        source_step=_as_int(row.get("step"), default=-1),
        source_file=source_file,
        source_line=source_line,
        source_uid=str(row.get("uid") or ""),
        raw_score=metadata["raw_score"],
        shaped_score=metadata["shaped_score"],
        response_tokens=metadata["response_tokens"],
        tool_dispatches=metadata["tool_dispatches"],
        repeated_bash_dispatches=metadata["repeated_bash_dispatches"],
        raw_exact_repeat_count=metadata["raw_exact_repeat_count"],
        penalized_repeat_count=metadata["penalized_repeat_count"],
        repeats_suppressed_after_mutation=metadata["repeats_suppressed_after_mutation"],
        repeats_suppressed_after_error=metadata["repeats_suppressed_after_error"],
        repeats_suppressed_outside_window=metadata["repeats_suppressed_outside_window"],
        repeats_suppressed_inflight=metadata["repeats_suppressed_inflight"],
        repeat_detection_mode=metadata["repeat_detection_mode"],
        repeat_detection_source=metadata["repeat_detection_source"],
        tool_error_returns=metadata["tool_error_returns"],
        has_verification=metadata["has_verification"],
        mutation_evidence=metadata["mutation_evidence"],
        submission_check=metadata["submission_check"],
        messages=messages,
        input_text=str(row.get("input") or ""),
        output_text=output_text,
        transcript_hash=transcript_hash,
    )


def _relaxed_candidate(row: dict[str, Any], **kwargs: Any) -> Candidate:
    relaxed = dict(kwargs)
    relaxed["max_tool_dispatches"] = relaxed.pop("relaxed_max_tool_dispatches")
    relaxed["max_penalized_repeated_tool_calls"] = relaxed.pop(
        "relaxed_max_penalized_repeated_tool_calls"
    )
    return _candidate(row, **relaxed)


def _candidate_row(candidate: Candidate, tier: str) -> dict[str, Any]:
    return {
        "messages": candidate.messages,
        "task_id": candidate.task_id,
        "task_key": candidate.task_key,
        "benchmark": candidate.benchmark,
        "repository": candidate.repository,
        "base_commit": candidate.base_commit,
        "source_run": candidate.source_run,
        "source_step": candidate.source_step,
        "source_file": candidate.source_file,
        "source_line": candidate.source_line,
        "source_uid": candidate.source_uid,
        "raw_score": candidate.raw_score,
        "shaped_score": candidate.shaped_score,
        "response_tokens": candidate.response_tokens,
        "tool_dispatches": candidate.tool_dispatches,
        "repeated_bash_dispatches": candidate.repeated_bash_dispatches,
        "raw_exact_repeat_count": candidate.raw_exact_repeat_count,
        "penalized_repeat_count": candidate.penalized_repeat_count,
        "repeats_suppressed_after_mutation": candidate.repeats_suppressed_after_mutation,
        "repeats_suppressed_after_error": candidate.repeats_suppressed_after_error,
        "repeats_suppressed_outside_window": candidate.repeats_suppressed_outside_window,
        "repeats_suppressed_inflight": candidate.repeats_suppressed_inflight,
        "repeat_detection_mode": candidate.repeat_detection_mode,
        "repeat_detection_source": candidate.repeat_detection_source,
        "tool_error_returns": candidate.tool_error_returns,
        "has_verification": candidate.has_verification,
        "mutation_evidence": candidate.mutation_evidence,
        "submission_check": candidate.submission_check,
        "quality_tier": tier,
        "transcript_hash": candidate.transcript_hash,
    }


def _split_is_validation(benchmark: str, task_id: str, fraction: float) -> bool:
    split_key = "\x1f".join((benchmark, task_id))
    digest = hashlib.sha256(split_key.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    return value < fraction


def _write_parquet(path: Path, candidates: Iterable[Candidate], tier: str) -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = [_candidate_row(candidate, tier) for candidate in candidates]
    if rows:
        table = pa.Table.from_pylist(rows)
    else:
        # Keep empty tiers readable by pandas/VERL instead of relying on
        # ``from_pylist([])``, which cannot infer a schema.
        table = pa.table(
            {
                "messages": pa.array([], type=pa.list_(pa.struct([
                    ("role", pa.string()),
                    ("content", pa.string()),
                ]))),
                "task_id": pa.array([], type=pa.string()),
                "task_key": pa.array([], type=pa.string()),
                "benchmark": pa.array([], type=pa.string()),
                "repository": pa.array([], type=pa.string()),
                "base_commit": pa.array([], type=pa.string()),
                "source_run": pa.array([], type=pa.string()),
                "source_step": pa.array([], type=pa.int64()),
                "source_file": pa.array([], type=pa.string()),
                "source_line": pa.array([], type=pa.int64()),
                "source_uid": pa.array([], type=pa.string()),
                "raw_score": pa.array([], type=pa.float64()),
                "shaped_score": pa.array([], type=pa.float64()),
                "response_tokens": pa.array([], type=pa.int64()),
                "tool_dispatches": pa.array([], type=pa.int64()),
                "repeated_bash_dispatches": pa.array([], type=pa.int64()),
                "raw_exact_repeat_count": pa.array([], type=pa.int64()),
                "penalized_repeat_count": pa.array([], type=pa.int64()),
                "repeats_suppressed_after_mutation": pa.array([], type=pa.int64()),
                "repeats_suppressed_after_error": pa.array([], type=pa.int64()),
                "repeats_suppressed_outside_window": pa.array([], type=pa.int64()),
                "repeats_suppressed_inflight": pa.array([], type=pa.int64()),
                "repeat_detection_mode": pa.array([], type=pa.string()),
                "repeat_detection_source": pa.array([], type=pa.string()),
                "tool_error_returns": pa.array([], type=pa.int64()),
                "has_verification": pa.array([], type=pa.bool_()),
                "mutation_evidence": pa.array([], type=pa.bool_()),
                "submission_check": pa.array([], type=pa.string()),
                "quality_tier": pa.array([], type=pa.string()),
                "transcript_hash": pa.array([], type=pa.string()),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd")
    return len(rows)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _candidate_sets(
    root: Path,
    runs: list[str],
    *,
    require_path_policy: bool,
    max_response_tokens: int,
    max_tool_dispatches: int,
    max_penalized_repeated_tool_calls: int,
    relaxed_max_tool_dispatches: int,
    relaxed_max_penalized_repeated_tool_calls: int,
    repeated_tool_detection_mode: str,
    repeated_tool_read_window: int,
    repeated_tool_exec_window: int,
    require_swe_mutation: bool,
) -> tuple[dict[str, Candidate], dict[str, Candidate], list[dict[str, Any]], collections.Counter[str]]:
    core: dict[str, Candidate] = {}
    relaxed_candidates: dict[str, Candidate] = {}
    rejected: list[dict[str, Any]] = []
    stats: collections.Counter[str] = collections.Counter()
    # A row can fail the strict limits while an identical transcript in
    # another rollout record passes them (for example, when diagnostic
    # counters were recorded differently).  Keep tier-specific deduplication
    # so the relaxed copy cannot hide a valid core candidate.
    seen_core_hashes: set[str] = set()
    seen_relaxed_hashes: set[str] = set()

    for run_name in runs:
        run_dir = root / run_name
        if not run_dir.is_dir():
            rejected.append({"source_run": run_name, "reason": "missing_run_directory"})
            stats["missing_run_directory"] += 1
            continue
        for source_file in sorted(run_dir.glob("*.jsonl")):
            with source_file.open(encoding="utf-8") as handle:
                for source_line, line in enumerate(handle, 1):
                    stats["rows_seen"] += 1
                    try:
                        row = json.loads(line)
                        if not isinstance(row, dict):
                            raise DataReject("malformed_json", "row is not an object")
                    except json.JSONDecodeError as exc:
                        reason, detail = "malformed_json", str(exc)
                        rejected.append(
                            {
                                "source_run": run_name,
                                "source_file": str(source_file),
                                "source_line": source_line,
                                "reason": reason,
                                "detail": detail,
                            }
                        )
                        stats[reason] += 1
                        continue

                    try:
                        # First apply the strict core policy.  If it fails only
                        # because of relaxed repetition/tool limits, try the
                        # relaxed policy independently.
                        strict = _candidate(
                            row,
                            run_name=run_name,
                            source_file=str(source_file),
                            source_line=source_line,
                            require_path_policy=require_path_policy,
                            max_response_tokens=max_response_tokens,
                            max_tool_dispatches=max_tool_dispatches,
                            max_penalized_repeated_tool_calls=max_penalized_repeated_tool_calls,
                            repeated_tool_detection_mode=repeated_tool_detection_mode,
                            repeated_tool_read_window=repeated_tool_read_window,
                            repeated_tool_exec_window=repeated_tool_exec_window,
                            require_swe_mutation=require_swe_mutation,
                        )
                        if strict.transcript_hash not in seen_core_hashes:
                            seen_core_hashes.add(strict.transcript_hash)
                            stats["core_candidates"] += 1
                            previous = core.get(strict.task_key)
                            if previous is None or strict.quality_key < previous.quality_key:
                                core[strict.task_key] = strict
                        else:
                            stats["duplicate_transcript"] += 1

                        # The relaxed tier is a superset of the core tier.
                        # Keeping strict candidates here makes it useful as
                        # the default larger SFT pool while retaining the
                        # stricter core dataset for ablations.
                        if strict.transcript_hash not in seen_relaxed_hashes:
                            seen_relaxed_hashes.add(strict.transcript_hash)
                            previous = relaxed_candidates.get(strict.task_key)
                            if previous is None or strict.quality_key < previous.quality_key:
                                relaxed_candidates[strict.task_key] = strict
                    except DataReject as strict_error:
                        # A relaxed candidate is allowed to differ only in
                        # repetition/tool-count thresholds; all semantic and
                        # infrastructure checks remain strict.
                        try:
                            relaxed_candidate = _relaxed_candidate(
                                row,
                                run_name=run_name,
                                source_file=str(source_file),
                                source_line=source_line,
                                require_path_policy=require_path_policy,
                                max_response_tokens=max_response_tokens,
                                relaxed_max_tool_dispatches=relaxed_max_tool_dispatches,
                                relaxed_max_penalized_repeated_tool_calls=(
                                    relaxed_max_penalized_repeated_tool_calls
                                ),
                                repeated_tool_detection_mode=repeated_tool_detection_mode,
                                repeated_tool_read_window=repeated_tool_read_window,
                                repeated_tool_exec_window=repeated_tool_exec_window,
                                require_swe_mutation=require_swe_mutation,
                            )
                        except DataReject as relaxed_error:
                            rejected.append({
                                "source_run": run_name,
                                "source_file": str(source_file),
                                "source_line": source_line,
                                "source_uid": str(row.get("uid") or ""),
                                "reason": relaxed_error.reason,
                                "detail": relaxed_error.detail,
                                "strict_reason": strict_error.reason,
                            })
                            stats[relaxed_error.reason] += 1
                            continue
                        if relaxed_candidate.transcript_hash in seen_relaxed_hashes:
                            stats["duplicate_transcript"] += 1
                            continue
                        seen_relaxed_hashes.add(relaxed_candidate.transcript_hash)
                        stats["relaxed_candidates"] += 1
                        previous = relaxed_candidates.get(relaxed_candidate.task_key)
                        if previous is None or relaxed_candidate.quality_key < previous.quality_key:
                            relaxed_candidates[relaxed_candidate.task_key] = relaxed_candidate

    return core, relaxed_candidates, rejected, stats


def _write_dataset_splits(
    output_dir: Path,
    candidates: dict[str, Candidate],
    tier: str,
    val_fraction: float,
) -> dict[str, int]:
    train = []
    validation = []
    for candidate in sorted(candidates.values(), key=lambda item: item.task_key):
        is_validation = _split_is_validation(candidate.benchmark, candidate.task_id, val_fraction)
        (validation if is_validation else train).append(candidate)
    counts = {
        f"{tier}_train": _write_parquet(output_dir / f"{tier}_train.parquet", train, tier),
        f"{tier}_val": _write_parquet(output_dir / f"{tier}_val.parquet", validation, tier),
    }
    return counts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-root", type=Path, default=Path("rollout_data"))
    parser.add_argument("--run", dest="runs", action="append", help="Rollout run directory; may be repeated.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-response-tokens", type=int, default=65536)
    parser.add_argument("--max-tool-dispatches", type=int, default=80)
    parser.add_argument(
        "--max-penalized-repeated-tool-calls",
        "--max-repeated-bash-dispatches",
        dest="max_penalized_repeated_tool_calls",
        type=int,
        default=0,
        help="Maximum reward-relevant repeated calls; the old flag name remains an alias.",
    )
    parser.add_argument("--relaxed-max-tool-dispatches", type=int, default=120)
    parser.add_argument(
        "--relaxed-max-penalized-repeated-tool-calls",
        "--relaxed-max-repeated-bash-dispatches",
        dest="relaxed_max_penalized_repeated_tool_calls",
        type=int,
        default=2,
        help="Relaxed-tier reward-relevant repeat limit; the old flag name remains an alias.",
    )
    parser.add_argument(
        "--repeated-tool-detection-mode",
        choices=("mutation_aware", "legacy_exact"),
        default="mutation_aware",
    )
    parser.add_argument("--repeated-tool-read-window", type=int, default=8)
    parser.add_argument("--repeated-tool-exec-window", type=int, default=16)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument(
        "--require-swe-mutation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require a patch/mutation signal for SWE examples (default: true).",
    )
    parser.add_argument(
        "--require-path-policy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require the current repository-relative path policy in the prompt.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not 0.0 <= args.val_fraction < 1.0:
        raise SystemExit("--val-fraction must be in [0, 1)")
    for name in (
        "max_response_tokens",
        "max_tool_dispatches",
        "max_penalized_repeated_tool_calls",
        "relaxed_max_tool_dispatches",
        "relaxed_max_penalized_repeated_tool_calls",
        "repeated_tool_read_window",
        "repeated_tool_exec_window",
    ):
        if getattr(args, name) < 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be non-negative")
    if args.relaxed_max_tool_dispatches < args.max_tool_dispatches:
        raise SystemExit("--relaxed-max-tool-dispatches must be at least --max-tool-dispatches")
    if args.relaxed_max_penalized_repeated_tool_calls < args.max_penalized_repeated_tool_calls:
        raise SystemExit(
            "--relaxed-max-penalized-repeated-tool-calls must be at least "
            "--max-penalized-repeated-tool-calls"
        )
    runs = args.runs or list(DEFAULT_RUNS)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    core, relaxed, rejected, stats = _candidate_sets(
        args.rollout_root,
        runs,
        require_path_policy=args.require_path_policy,
        max_response_tokens=args.max_response_tokens,
        max_tool_dispatches=args.max_tool_dispatches,
        max_penalized_repeated_tool_calls=args.max_penalized_repeated_tool_calls,
        relaxed_max_tool_dispatches=args.relaxed_max_tool_dispatches,
        relaxed_max_penalized_repeated_tool_calls=args.relaxed_max_penalized_repeated_tool_calls,
        repeated_tool_detection_mode=args.repeated_tool_detection_mode,
        repeated_tool_read_window=args.repeated_tool_read_window,
        repeated_tool_exec_window=args.repeated_tool_exec_window,
        require_swe_mutation=args.require_swe_mutation,
    )

    counts = {}
    counts.update(_write_dataset_splits(output_dir, core, "core", args.val_fraction))
    counts.update(_write_dataset_splits(output_dir, relaxed, "relaxed", args.val_fraction))
    gold = {key: candidate for key, candidate in core.items() if candidate.has_verification}
    counts.update(_write_dataset_splits(output_dir, gold, "gold", args.val_fraction))

    _write_jsonl(output_dir / "rejected.jsonl", rejected)
    selected = []
    for tier, candidates in (("core", core), ("relaxed", relaxed), ("gold", gold)):
        for candidate in sorted(candidates.values(), key=lambda item: item.task_key):
            selected.append({
                "tier": tier,
                "task_key": candidate.task_key,
                "task_id": candidate.task_id,
                "benchmark": candidate.benchmark,
                "source_run": candidate.source_run,
                "source_step": candidate.source_step,
                "source_file": candidate.source_file,
                "source_line": candidate.source_line,
                "source_uid": candidate.source_uid,
                "raw_score": candidate.raw_score,
                "shaped_score": candidate.shaped_score,
                "response_tokens": candidate.response_tokens,
                "tool_dispatches": candidate.tool_dispatches,
                "repeated_bash_dispatches": candidate.repeated_bash_dispatches,
                "raw_exact_repeat_count": candidate.raw_exact_repeat_count,
                "penalized_repeat_count": candidate.penalized_repeat_count,
                "repeats_suppressed_after_mutation": candidate.repeats_suppressed_after_mutation,
                "repeats_suppressed_after_error": candidate.repeats_suppressed_after_error,
                "repeats_suppressed_outside_window": candidate.repeats_suppressed_outside_window,
                "repeats_suppressed_inflight": candidate.repeats_suppressed_inflight,
                "repeat_detection_mode": candidate.repeat_detection_mode,
                "repeat_detection_source": candidate.repeat_detection_source,
                "has_verification": candidate.has_verification,
                "mutation_evidence": candidate.mutation_evidence,
                "submission_check": candidate.submission_check,
                "transcript_hash": candidate.transcript_hash,
            })
    _write_jsonl(output_dir / "manifest.jsonl", selected)

    summary = {
        "runs": runs,
        "rollout_root": str(args.rollout_root),
        "filters": {
            "max_response_tokens": args.max_response_tokens,
            "max_tool_dispatches": args.max_tool_dispatches,
            "max_penalized_repeated_tool_calls": args.max_penalized_repeated_tool_calls,
            "relaxed_max_tool_dispatches": args.relaxed_max_tool_dispatches,
            "relaxed_max_penalized_repeated_tool_calls": (
                args.relaxed_max_penalized_repeated_tool_calls
            ),
            "repeated_tool_detection_mode": args.repeated_tool_detection_mode,
            "repeated_tool_read_window": args.repeated_tool_read_window,
            "repeated_tool_exec_window": args.repeated_tool_exec_window,
            "require_path_policy": args.require_path_policy,
            "require_swe_mutation": args.require_swe_mutation,
            "val_fraction": args.val_fraction,
        },
        "stats": dict(stats),
        "selected_counts": counts,
        "core_unique_tasks": len(core),
        "relaxed_unique_tasks": len(relaxed),
        "gold_unique_tasks": len(gold),
        "rejected_rows": len(rejected),
    }
    (output_dir / "statistics.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

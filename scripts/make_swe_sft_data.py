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


DEFAULT_RUNS = (
    "fix_tool_bug_small_lr_new_data",
    "fix_tool_bug_use_moe_small_lr_new_data",
    "fix_tool_super_long",
)

PATH_POLICY_RE = re.compile(r"repository-relative paths? only", re.IGNORECASE)
PATH_REJECTION_RE = re.compile(
    r"(?:path rejected\s*:|absolute paths?\b[^\n]{0,120}\bnot allowed|"
    r"path escapes workspace)",
    re.IGNORECASE,
)
MARKER_RE = re.compile(r"(?m)^(assistant|user)\n")
TOOL_BLOCK_RE = re.compile(r"<function=([^>\n]+)>(.*?)</function>", re.DOTALL)
TOOL_RESPONSE_RE = re.compile(r"<tool_response>(.*?)</tool_response>", re.DOTALL)
PARAM_RE = re.compile(r"<parameter=([^>\n]+)>\s*(.*?)\s*</parameter>", re.DOTALL)
VERIFY_RE = re.compile(r"<parameter=verification>\s*true\s*</parameter>", re.IGNORECASE)
ABSOLUTE_PATH_PARAM_NAMES = {"file_path", "path", "directory"}
MUTATING_TOOLS = {"edit", "write", "apply_patch", "write_file"}
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
    tool_error_returns: int
    has_verification: bool
    mutation_evidence: bool
    messages: list[dict[str, str]]
    input_text: str
    output_text: str
    transcript_hash: str

    @property
    def quality_key(self) -> tuple[int, int, int, int, int, str, int]:
        """Lower is better; prefer protocol-complete and concise traces."""

        return (
            0 if self.has_verification else 1,
            self.repeated_bash_dispatches,
            self.tool_error_returns,
            self.tool_dispatches,
            self.response_tokens,
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


def _tool_metrics(output_text: str) -> dict[str, Any]:
    # Tool responses may contain arbitrary repository text, including strings
    # that resemble the XML call format.  Count only calls emitted by the
    # assistant, while keeping response text available to path diagnostics.
    assistant_text = TOOL_RESPONSE_RE.sub("", output_text)
    blocks = list(TOOL_BLOCK_RE.finditer(assistant_text))
    absolute_paths: list[str] = []
    normalized_calls: list[str] = []
    mutating = False
    for block in blocks:
        tool_name = block.group(1).strip()
        body = block.group(2)
        parameters = {name: value.strip() for name, value in PARAM_RE.findall(body)}
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
        normalized = re.sub(r"\s+", " ", block.group(0)).strip()
        normalized_calls.append(normalized)

    counts = collections.Counter(normalized_calls)
    duplicate_calls = sum(count - 1 for count in counts.values() if count > 1)
    return {
        "tool_blocks": len(blocks),
        "absolute_paths": absolute_paths,
        "duplicate_calls": duplicate_calls,
        "has_verification": bool(VERIFY_RE.search(assistant_text)),
        "mutation_evidence": mutating,
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
    max_repeated_bash_dispatches: int,
    require_swe_mutation: bool,
) -> tuple[dict[str, Any], dict[str, str], list[dict[str, str]]]:
    if row.get("is_padding"):
        raise DataReject("padding")
    raw_score = _as_float(row.get("raw_score", row.get("score")))
    shaped_score = _as_float(row.get("shaped_score", row.get("score")))
    if raw_score is None or raw_score < 0.999:
        raise DataReject("not_reward_one")
    if shaped_score is None or shaped_score < 0.999:
        raise DataReject("shaped_reward_penalty")
    if row.get("train_sample_mask") is not True:
        raise DataReject("train_sample_mask_false_or_missing")
    if row.get("completion_ratio_cutoff"):
        raise DataReject("completion_cutoff")
    if row.get("trajectory_timeout"):
        raise DataReject("timeout")
    if row.get("trajectory_terminal_tool_failure"):
        raise DataReject("terminal_tool_failure")
    if _as_int(row.get("trajectory_tool_error_returns")) > 0:
        raise DataReject("tool_error")
    if row.get("trajectory_budget_reached"):
        raise DataReject("budget_reached")

    benchmark, task_id, repository, base_commit = _gts_fields(row)
    input_text = str(row.get("input") or "")
    output_text = str(row.get("output") or "")
    system, user = _parse_input(input_text)
    if require_path_policy and not _path_policy_ok(input_text):
        raise DataReject("old_or_missing_path_policy")
    if _has_path_rejection(output_text):
        raise DataReject("path_violation")

    metrics = _tool_metrics(output_text)
    if metrics["absolute_paths"]:
        raise DataReject("absolute_file_tool_path")
    response_tokens = _read_stat(row, "trajectory_response_tokens", "response_tokens")
    tool_dispatches = _read_stat(row, "trajectory_tool_dispatches", "tool_dispatches")
    repeated = _as_int(row.get("trajectory_repeated_bash_dispatches"))
    repeated = max(repeated, int(metrics["duplicate_calls"]))
    tool_dispatches = max(tool_dispatches, int(metrics["tool_blocks"]))
    if response_tokens > max_response_tokens:
        raise DataReject("response_too_long")
    if tool_dispatches > max_tool_dispatches:
        raise DataReject("too_many_tool_dispatches")
    if repeated > max_repeated_bash_dispatches:
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
        "repeated_bash_dispatches": repeated,
        "tool_error_returns": _as_int(row.get("trajectory_tool_error_returns")),
        "has_verification": bool(metrics["has_verification"]),
        "mutation_evidence": mutation_evidence,
    }
    return metadata, {"system": system, "user": user}, metrics


def _candidate(
    row: dict[str, Any],
    *,
    run_name: str,
    source_file: str,
    source_line: int,
    require_path_policy: bool,
    max_response_tokens: int,
    max_tool_dispatches: int,
    max_repeated_bash_dispatches: int,
    require_swe_mutation: bool,
) -> Candidate:
    metadata, prompt, metrics = _base_rejection(
        row,
        require_path_policy=require_path_policy,
        max_response_tokens=max_response_tokens,
        max_tool_dispatches=max_tool_dispatches,
        max_repeated_bash_dispatches=max_repeated_bash_dispatches,
        require_swe_mutation=require_swe_mutation,
    )
    output_text = str(row.get("output") or "")
    messages = [{"role": "system", "content": prompt["system"]}, {"role": "user", "content": prompt["user"]}]
    messages.extend(_parse_output(output_text))
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
        tool_error_returns=metadata["tool_error_returns"],
        has_verification=metadata["has_verification"],
        mutation_evidence=metadata["mutation_evidence"],
        messages=messages,
        input_text=str(row.get("input") or ""),
        output_text=output_text,
        transcript_hash=transcript_hash,
    )


def _relaxed_candidate(row: dict[str, Any], **kwargs: Any) -> Candidate:
    relaxed = dict(kwargs)
    relaxed["max_tool_dispatches"] = relaxed.pop("relaxed_max_tool_dispatches")
    relaxed["max_repeated_bash_dispatches"] = relaxed.pop("relaxed_max_repeated_bash_dispatches")
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
        "tool_error_returns": candidate.tool_error_returns,
        "has_verification": candidate.has_verification,
        "mutation_evidence": candidate.mutation_evidence,
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
                "tool_error_returns": pa.array([], type=pa.int64()),
                "has_verification": pa.array([], type=pa.bool_()),
                "mutation_evidence": pa.array([], type=pa.bool_()),
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
    max_repeated_bash_dispatches: int,
    relaxed_max_tool_dispatches: int,
    relaxed_max_repeated_bash_dispatches: int,
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
                        rejected.append({"source_run": run_name, "source_file": str(source_file), "source_line": source_line, "reason": reason, "detail": detail})
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
                            max_repeated_bash_dispatches=max_repeated_bash_dispatches,
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
                                relaxed_max_repeated_bash_dispatches=relaxed_max_repeated_bash_dispatches,
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


def _write_dataset_splits(output_dir: Path, candidates: dict[str, Candidate], tier: str, val_fraction: float) -> dict[str, int]:
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
    parser.add_argument("--max-repeated-bash-dispatches", type=int, default=0)
    parser.add_argument("--relaxed-max-tool-dispatches", type=int, default=100)
    parser.add_argument("--relaxed-max-repeated-bash-dispatches", type=int, default=2)
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
    runs = args.runs or list(DEFAULT_RUNS)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    core, relaxed, rejected, stats = _candidate_sets(
        args.rollout_root,
        runs,
        require_path_policy=args.require_path_policy,
        max_response_tokens=args.max_response_tokens,
        max_tool_dispatches=args.max_tool_dispatches,
        max_repeated_bash_dispatches=args.max_repeated_bash_dispatches,
        relaxed_max_tool_dispatches=args.relaxed_max_tool_dispatches,
        relaxed_max_repeated_bash_dispatches=args.relaxed_max_repeated_bash_dispatches,
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
                "has_verification": candidate.has_verification,
                "mutation_evidence": candidate.mutation_evidence,
                "transcript_hash": candidate.transcript_hash,
            })
    _write_jsonl(output_dir / "manifest.jsonl", selected)

    summary = {
        "runs": runs,
        "rollout_root": str(args.rollout_root),
        "filters": {
            "max_response_tokens": args.max_response_tokens,
            "max_tool_dispatches": args.max_tool_dispatches,
            "max_repeated_bash_dispatches": args.max_repeated_bash_dispatches,
            "relaxed_max_tool_dispatches": args.relaxed_max_tool_dispatches,
            "relaxed_max_repeated_bash_dispatches": args.relaxed_max_repeated_bash_dispatches,
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

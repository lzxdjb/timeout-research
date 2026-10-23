"""Bounded diagnostics for rollout groups stopped by completion-ratio cutoff.

The writer is opt-in and stores only scalar counters and identifiers. It never
serializes prompts, generated text, tool arguments, or tool observations.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import socket
import threading
import time
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)
_write_warning_emitted = False
_write_lock = threading.Lock()
MAX_EVENT_BYTES = 32768
MAX_GENERATION_TOKENS = 65536
MAX_IDENTITIES = 4096


def _bounded(value: Any, depth: int = 0) -> Any:
    if value is None or type(value) in (bool, int, float):
        return value
    if isinstance(value, str):
        return value[:512]
    if depth >= 4:
        return None
    if isinstance(value, dict):
        return {str(k)[:128]: _bounded(v, depth + 1) for k, v in list(value.items())[:192]}
    if isinstance(value, (list, tuple)):
        return [_bounded(v, depth + 1) for v in value[:128]]
    item = getattr(value, "item", None)
    return _bounded(item(), depth + 1) if callable(item) else None


def cutoff_debug_enabled() -> bool:
    value = os.environ.get("SWE_AGENT_CUTOFF_DEBUG", "0").strip().lower()
    return value in {"1", "true", "yes", "on"}


def write_cutoff_debug_event(event: str, **fields: Any) -> bool:
    """Append one compact JSONL event; logging failures never affect rollouts."""
    if not cutoff_debug_enabled():
        return False
    output_dir = os.environ.get("SWE_AGENT_CUTOFF_DEBUG_DIR", "").strip()
    if not output_dir:
        return False

    partition = str(fields.get("partition", "unknown"))
    if partition not in {"train", "val"}:
        partition = "unknown"
    payload = {
        "schema_version": 1,
        "event": str(event),
        "timestamp_unix_seconds": round(time.time(), 3),
        "pid": os.getpid(),
        **fields,
    }
    try:
        payload = _bounded(payload)
        encoded = (json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode()
        if len(encoded) > MAX_EVENT_BYTES:
            # Keep the row joinable and make lost diagnostics explicit.
            payload = {key: payload[key] for key in (
                "schema_version", "event", "timestamp_unix_seconds", "pid",
                "partition", "uid", "global_step", "sample_index", "session_id",
            ) if key in payload}
            payload["payload_truncated"] = True
            encoded = (json.dumps(payload, separators=(",", ":")) + "\n").encode()
        host = socket.gethostname().replace("/", "_").replace("\\", "_")
        path = Path(output_dir) / f"cutoff_{partition}_{host}_{os.getpid()}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with _write_lock:
            descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o640)
            try:
                remaining = memoryview(encoded)
                while remaining:
                    written = os.write(descriptor, remaining)
                    if written <= 0:
                        raise OSError("Short write in cutoff diagnostics")
                    remaining = remaining[written:]
            finally:
                os.close(descriptor)
        return True
    except Exception:
        global _write_warning_emitted
        if not _write_warning_emitted:
            logger.warning("Unable to write SWE cutoff diagnostics; rollout behavior is unchanged", exc_info=True)
            _write_warning_emitted = True
        return False


def summarize_agent_loop_state(agent_data: Any) -> dict[str, int]:
    """Return bounded token, turn, and detector counters for a partial session."""
    mask = list(getattr(agent_data, "response_mask", []) or []) if agent_data is not None else []
    assistant_tokens = sum(int(value) for value in mask)
    summary = {
        "partial_response_tokens": len(mask),
        "partial_assistant_tokens": assistant_tokens,
        "partial_observation_tokens": max(0, len(mask) - assistant_tokens),
        "partial_assistant_turns": int(getattr(agent_data, "assistant_turns", 0) or 0),
        "partial_user_turns": int(getattr(agent_data, "user_turns", 0) or 0),
        "partial_parsed_tool_calls": len(getattr(agent_data, "tool_calls", []) or []),
    }
    extra = getattr(agent_data, "extra_fields", {}) if agent_data is not None else {}
    if isinstance(extra, dict):
        for key, value in extra.items():
            if (
                isinstance(key, str)
                and key.startswith(("trajectory_tool_", "trajectory_bash_", "trajectory_repeated_",
                                   "trajectory_penalized_", "thinking_repeat_", "tool_repeat_"))
                and not key.endswith(("_events", "_rules", "_json"))
                and type(value) is int
                and value >= 0
            ):
                summary[key] = value
    return summary


class DiagnosticRepeatTracker:
    """Independent, bounded exact-recurrence proxies; never used for rewards.

    Whole thoughts use whitespace normalization, with no minimum length or
    requirement for a paired tool call. Token-block counts also cover long
    assistant turns. These proxies do not detect semantic paraphrases.
    """

    def __init__(self):
        self.counts = {kind: {} for kind in ("tool", "assistant_turn", "thought", "assistant_block64")}
        self.stats = {"generation_calls": 0, "tool_dispatches": 0, "thoughts_captured": 0,
                      "thoughts_unavailable": 0, "generation_tokens_skipped": 0,
                      "identity_capacity_overflow": 0, "capture_errors": 0}
        self.previous = {}
        self.runs = {}
        self.max_runs = {}

    def _observe(self, kind: str, value: str) -> None:
        signature = hashlib.sha256(value.encode()).hexdigest()
        counts = self.counts[kind]
        if signature in counts or len(counts) < MAX_IDENTITIES:
            counts[signature] = counts.get(signature, 0) + 1
        else:
            self.stats["identity_capacity_overflow"] += 1
        run = self.runs.get(kind, 0) + 1 if self.previous.get(kind) == signature else 1
        self.previous[kind] = signature
        self.runs[kind] = run
        self.max_runs[kind] = max(self.max_runs.get(kind, 0), run)

    def observe_tool(self, name: str, arguments: Any) -> None:
        self.stats["tool_dispatches"] += 1
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except ValueError:
                pass
        self._observe("tool", str(name) + "\0" + json.dumps(
            arguments, sort_keys=True, ensure_ascii=True, separators=(",", ":")))

    def observe_generation(self, tokenizer: Any, token_ids: list[int]) -> None:
        self.stats["generation_calls"] += 1
        limited = token_ids[:MAX_GENERATION_TOKENS]
        self.stats["generation_tokens_skipped"] += max(0, len(token_ids) - len(limited))
        # Do not label two capped prefixes as identical complete turns.
        if len(limited) == len(token_ids):
            self._observe("assistant_turn", str(limited))
        for start in range(0, len(limited) - 63, 64):
            self._observe("assistant_block64", str(limited[start:start + 64]))
        text = tokenizer.decode(limited, skip_special_tokens=False)
        if "</think>" in text:
            thought = text.split("</think>", 1)[0].rsplit("<think>", 1)[-1]
            normalized = " ".join(thought.split())
            if normalized:
                self.stats["thoughts_captured"] += 1
                self._observe("thought", normalized)
                return
        self.stats["thoughts_unavailable"] += 1

    def summary(self) -> dict[str, int]:
        result = {"diagnostic_" + key: value for key, value in self.stats.items()}
        for kind, counts in self.counts.items():
            result.update({
                f"diagnostic_{kind}_unique": len(counts),
                f"diagnostic_{kind}_repeat_surplus": sum(value - 1 for value in counts.values()),
                f"diagnostic_{kind}_max_copies": max(counts.values(), default=0),
                f"diagnostic_{kind}_max_consecutive_copies": self.max_runs.get(kind, 0),
            })
        return result


def summarize_repeat_events(events: Any) -> dict[str, int]:
    """Summarize exact repeated calls and captured thought hashes without text."""
    if not isinstance(events, (list, tuple)):
        events = ()
    call_counts: dict[tuple[str, str], int] = {}
    shell_counts: dict[str, int] = {}
    python_test_counts: dict[str, int] = {}
    thought_counts: dict[str, int] = {}
    call_sequence: list[tuple[str, str]] = []
    shell_dispatches = 0
    python_test_evidence_calls = 0
    thought_candidates = 0

    for event in events:
        if not isinstance(event, dict):
            continue
        name = str(event.get("name") or "unknown")
        try:
            normalized_arguments = json.dumps(
                event.get("arguments"), ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str
            )
        except (TypeError, ValueError):
            normalized_arguments = "<unserializable>"
        signature = hashlib.sha256((name + "\0" + normalized_arguments).encode()).hexdigest()
        call_key = (name, signature)
        call_counts[call_key] = call_counts.get(call_key, 0) + 1
        call_sequence.append(call_key)

        if name.lower() in {"bash", "run_shell", "shell"}:
            shell_dispatches += 1
            shell_counts[signature] = shell_counts.get(signature, 0) + 1
        if isinstance(event.get("repeat_python_test_evidence"), dict):
            python_test_evidence_calls += 1
            python_test_counts[signature] = python_test_counts.get(signature, 0) + 1
        candidate = event.get("thinking_candidate")
        thought_hash = candidate.get("thought_sha256") if isinstance(candidate, dict) else None
        if isinstance(thought_hash, str) and len(thought_hash) == 64:
            thought_candidates += 1
            thought_counts[thought_hash] = thought_counts.get(thought_hash, 0) + 1

    max_consecutive_call_copies = 0
    consecutive = 0
    previous = None
    for call in call_sequence:
        consecutive = consecutive + 1 if call == previous else 1
        max_consecutive_call_copies = max(max_consecutive_call_copies, consecutive)
        previous = call

    return {
        "partial_tool_call_events": len(call_sequence),
        "partial_tool_unique_signatures": len(call_counts),
        "partial_tool_signature_repeat_surplus": sum(count - 1 for count in call_counts.values()),
        "partial_tool_max_signature_copies": max(call_counts.values(), default=0),
        "partial_tool_max_consecutive_signature_copies": max_consecutive_call_copies,
        "partial_shell_dispatches": shell_dispatches,
        "partial_shell_signature_repeat_surplus": sum(count - 1 for count in shell_counts.values()),
        "partial_python_test_evidence_calls": python_test_evidence_calls,
        "partial_python_test_signature_repeat_surplus": sum(count - 1 for count in python_test_counts.values()),
        "partial_thinking_candidate_events": thought_candidates,
        "partial_thinking_unique_hashes": len(thought_counts),
        "partial_thinking_hash_repeat_surplus": sum(count - 1 for count in thought_counts.values()),
        "partial_thinking_max_hash_copies": max(thought_counts.values(), default=0),
    }


def reconcile_cutoff_sessions(
    expected_sessions: int,
    *,
    started: Iterable[int],
    returned: Iterable[int],
    cancelled: Iterable[int],
    failed: Iterable[int],
    published: Iterable[int],
) -> dict[str, Any]:
    """Reconcile expected session IDs with observed lifecycle and publication."""
    expected = max(0, int(expected_sessions))
    universe = set(range(expected))
    sets = {
        "started": set(started) & universe,
        "returned": set(returned) & universe,
        "cancelled": set(cancelled) & universe,
        "failed": set(failed) & universe,
        "published": set(published) & universe,
    }
    return {
        "expected_sessions": expected,
        **{f"{name}_sessions": len(values) for name, values in sets.items()},
        "not_started_sessions": len(universe - sets["started"]),
        "started_without_terminal_state_sessions": len(
            sets["started"] - sets["returned"] - sets["cancelled"] - sets["failed"]
        ),
        "returned_without_publication_sessions": len(sets["returned"] - sets["published"]),
        **{f"{name}_session_ids": sorted(values) for name, values in sets.items()},
        "not_started_session_ids": sorted(universe - sets["started"]),
    }

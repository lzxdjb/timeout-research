"""Publish completed synthesis attempts in the existing SFT quality input format.

Only explicit, valid hidden-test successes are exported. This module does not
call a model, change rewards, or decide whether a trajectory is good SFT data;
the existing rules and quality judge make that decision after this adapter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from recipe.swe_trajectory_synthesis.protocol import action_to_qwen_xml

VERSION = "swe-teacher-rollout-export-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def as_dict(value: Any) -> dict:
    if isinstance(value, str):
        value = json.loads(value)
    return value if isinstance(value, dict) else {}


def true(value: Any) -> bool:
    return value is True or (isinstance(value, (int, float)) and value == 1)


def reward_fields(trajectory: dict) -> dict:
    extra = as_dict(trajectory.get("extra_fields"))
    return {**as_dict(extra.get("reward_extra_info")), **as_dict(trajectory.get("reward_extra_info"))}


def admission_reason(trajectory: dict) -> str | None:
    fields = {**as_dict(trajectory.get("extra_fields")), **reward_fields(trajectory)}
    # A terminal row may mirror these fields at its top level.
    for key in (
        "raw_score",
        "hidden_pass",
        "evaluation_valid",
        "infrastructure_failure",
        "observed_infrastructure_failure",
        "submission_signal_seen",
    ):
        if key in trajectory:
            fields[key] = trajectory[key]
    if not true(fields.get("evaluation_valid")):
        return "unconfirmed_evaluation"
    if not true(fields.get("hidden_pass")) or fields.get("raw_score") != 1:
        return "not_hidden_success"
    for key in (
        "infrastructure_failure",
        "observed_infrastructure_failure",
        "evaluator_failure_excluded",
        "evaluator_group_excluded",
        "completion_ratio_cutoff",
        "trajectory_timeout",
        "trajectory_terminal_tool_failure",
        "trajectory_budget_reached",
        "is_padding",
        "synthesis_context_pruned",
        "hidden_reward_skipped",
        "training_hidden_reward_skipped",
    ):
        if true(fields.get(key)) or true(trajectory.get(key)):
            return key
    if int(fields.get("synthesis_protocol_retries") or 0) > 0:
        return "unrecorded_protocol_repair"
    if not true(fields.get("submission_signal_seen")):
        return "missing_submission"
    return None


def _assistant_text(message: dict) -> str:
    content = str(message.get("content") or "")
    reasoning = str(message.get("reasoning_content") or "")
    if reasoning and "<think>" not in content:
        content = f"<think>\n{reasoning}\n</think>\n\n" + content
    for call in message.get("tool_calls") or []:
        function = as_dict(call.get("function"))
        arguments = as_dict(function.get("arguments"))
        if not function.get("name"):
            raise ValueError("Tool call has no function name")
        content += ("\n" if content else "") + action_to_qwen_xml(function["name"], arguments)
    if not content.strip():
        raise ValueError("Empty assistant message")
    return content


def render_transcript(trajectory: dict, tokenizer: Any) -> tuple[str, str]:
    messages = trajectory.get("messages")
    if not isinstance(messages, list) or len(messages) < 3:
        raise ValueError("Incomplete canonical messages")
    if [m.get("role") for m in messages[:2]] != ["system", "user"]:
        raise ValueError("Canonical transcript must begin with system and public user task")
    if messages[-1].get("role") != "assistant":
        raise ValueError("Canonical transcript ends before the final assistant turn")
    tools = trajectory.get("tools") or None
    prompt = tokenizer.apply_chat_template(
        messages[:2],
        tools=tools,
        tokenize=True,
        return_dict=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    input_text = tokenizer.decode(prompt, skip_special_tokens=True)
    if not input_text.startswith("system\n") or "\nuser\n" not in input_text:
        raise ValueError("Tokenizer does not render the supported Qwen student transcript")
    turns: list[str] = []
    pending: list[dict] = []
    for message in messages[2:]:
        role = message.get("role")
        if role == "assistant":
            if pending:
                raise ValueError("Missing observation for canonical tool call")
            text = _assistant_text(message)
            turns.append(("\nassistant\n" if turns else "") + text)
            pending = list(message.get("tool_calls") or [])
        elif role == "tool":
            if len(pending) != 1:
                raise ValueError("Expected one observed tool call per assistant turn")
            call = pending.pop()
            expected_id = call.get("id")
            if expected_id and message.get("tool_call_id") != expected_id:
                raise ValueError("Tool observation ID does not match the actual call")
            name = as_dict(call.get("function")).get("name")
            if message.get("name") and message["name"] != name:
                raise ValueError("Tool observation name mismatch")
            observation = str(message.get("content") or "")
            if not observation.strip():
                raise ValueError("Empty tool observation")
            turns.append("\nuser\n<tool_response>\n" + observation + "\n</tool_response>")
        else:
            raise ValueError(f"Unexpected canonical role after task: {role!r}")
    if pending:
        # Native Submit has no observation because it closes the trajectory.
        if len(pending) != 1 or as_dict(pending[0].get("function")).get("name") not in {"Submit", "submit"}:
            raise ValueError("Unobserved terminal tool call")
    return input_text, "".join(turns)


def rollout_row(trajectory: dict, tokenizer: Any, record_path: Path, contract: dict) -> dict:
    reason = admission_reason(trajectory)
    if reason:
        raise ValueError(reason)
    fields = {**as_dict(trajectory.get("extra_fields")), **reward_fields(trajectory)}
    ground_truth = as_dict(as_dict(trajectory.get("reward_model")).get("ground_truth"))
    info = as_dict(trajectory.get("extra_info") or trajectory.get("source_extra_info"))
    task_id = str(trajectory.get("task_id") or ground_truth.get("instance_id") or info.get("task_id") or "")
    benchmark = str(ground_truth.get("benchmark") or info.get("benchmark") or "")
    if not task_id or not benchmark:
        raise ValueError("missing_task_identity")
    input_text, output_text = render_transcript(trajectory, tokenizer)
    # Use the existing parser to reject ambiguous role markers and mismatched
    # dispatch/observation evidence, before freezing a new source file.
    from scripts.swe_sft_quality import parse_sft_transcript

    _, _, issues = parse_sft_transcript(output_text)
    if issues:
        raise ValueError("ambiguous_transcript:" + ",".join(issues))
    row = {key: value for key, value in fields.items() if isinstance(value, (str, int, float, bool)) or value is None}
    row.update(
        input=input_text,
        output=output_text,
        gts={
            "benchmark": benchmark,
            "instance_id": task_id,
            "repo": ground_truth.get("repo") or info.get("repo") or "",
            "base_commit": ground_truth.get("base_commit") or info.get("base_commit") or info.get("base_sha") or "",
        },
        task_id=task_id,
        raw_score=1.0,
        hidden_pass=1,
        evaluation_valid=True,
        score=1.0,
        shaped_score=trajectory.get("shaped_score"),
        reward_eligible=True,
        train_sample_mask=True,
        is_padding=False,
        submission_signal_seen=True,
        trajectory_response_tokens=len(tokenizer.encode(output_text, add_special_tokens=False)),
        trajectory_tool_dispatches=sum(len(m.get("tool_calls") or []) for m in trajectory["messages"]),
        uid="teacher-" + sha256(record_path),
        step=int(trajectory.get("attempt") or 0),
        teacher_source_record=str(record_path),
        teacher_source_sha256=sha256(record_path),
        # The run contract retains the original endpoint across an address-only
        # resume. Each completed attempt records the endpoint it actually used.
        # Older trajectories without those fields still use the initial contract.
        teacher_model=trajectory.get("teacher_model") or contract.get("model"),
        teacher_api_url=trajectory.get("teacher_api_url") or contract.get("api_url"),
        teacher_export_version=VERSION,
    )
    if fields.get("sft_tool_contract_outcomes"):
        row["sft_tool_contract_outcomes"] = fields["sft_tool_contract_outcomes"]
    return row


def sft_row(row: dict) -> dict:
    """Render the same student messages as the downstream quality builder.

    This is a hidden-test-success snapshot, before rules and judge filtering.
    The template-expanded system text already contains the tool definitions,
    so adding a separate ``tools`` column would duplicate them during SFT.
    """
    from scripts.make_swe_sft_data import _parse_input
    from scripts.swe_sft_quality import parse_sft_transcript

    system, user = _parse_input(row["input"])
    _, messages, issues = parse_sft_transcript(row["output"])
    if issues:
        raise ValueError("ambiguous_transcript:" + ",".join(issues))
    return {
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}, *messages],
        "enable_thinking": True,
        "task_id": row["task_id"],
        "benchmark": row["gts"]["benchmark"],
        "score": 1.0,
        "raw_score": 1.0,
        "hidden_pass": True,
        "evaluation_valid": True,
        "quality_status": "hidden_success_before_quality_filter",
        "source_record": row["teacher_source_record"],
        "source_sha256": row["teacher_source_sha256"],
        "teacher_model": row["teacher_model"],
        "teacher_api_url": row["teacher_api_url"],
        "response_tokens": row["trajectory_response_tokens"],
        "tool_dispatches": row["trajectory_tool_dispatches"],
    }


def export(args: argparse.Namespace) -> dict:
    from transformers import AutoTokenizer

    source = args.synthesis_dir.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise ValueError(f"Refusing to overwrite immutable teacher export: {output}")
    contract_path = source / "run_contract.json"
    contract = json.loads(contract_path.read_text())
    allow_running = bool(getattr(args, "allow_running", False))
    write_sft_parquet = bool(getattr(args, "write_sft_parquet", False))
    collection_running = False
    if (source / ".lock").exists():
        import fcntl

        with (source / ".lock").open("r") as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                collection_running = True
                if not allow_running:
                    raise ValueError(
                        "Finish collection before freezing quality sources, or use --allow-running "
                        "to snapshot only already completed attempts"
                    ) from exc
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    records = sorted((source / "attempts").glob("*/*/record.json"))
    if not records:
        raise ValueError("No completed synthesis attempt records")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    counts: Counter = Counter()
    tasks: set[str] = set()
    sft_rows: list[dict] = []
    try:
        rollout_dir = temporary / "rollouts"
        rollout_dir.mkdir()
        with (rollout_dir / "1.jsonl").open("w") as accepted, (temporary / "rejected.jsonl").open("w") as rejected:
            for record_path in records:
                counts["completed_attempts"] += 1
                raw = json.loads(record_path.read_text())
                trajectory = as_dict(raw.get("trajectory"))
                reason = admission_reason(trajectory)
                if raw.get("kind") != "solve":
                    reason = "not_completed_solve"
                try:
                    if reason:
                        raise ValueError(reason)
                    row = rollout_row(trajectory, tokenizer, record_path, contract)
                    converted = sft_row(row) if write_sft_parquet else None
                except (ValueError, TypeError, KeyError) as exc:
                    counts["rejected:" + str(exc).split(":", 1)[0]] += 1
                    rejected.write(json.dumps({"record": str(record_path), "reason": str(exc)}) + "\n")
                    continue
                accepted.write(json.dumps(row, ensure_ascii=False) + "\n")
                counts["exported_trajectories"] += 1
                tasks.add(row["task_id"])
                if converted is not None:
                    sft_rows.append(converted)
        if not tasks:
            raise ValueError(f"No valid hidden successes to quality-filter; counts={dict(counts)}")
        if write_sft_parquet:
            import pyarrow as pa
            import pyarrow.parquet as pq

            pq.write_table(pa.Table.from_pylist(sft_rows), temporary / "successful.parquet")
        report = {
            "version": VERSION,
            "synthesis_dir": str(source),
            "contract_sha256": sha256(contract_path),
            "tokenizer_path": str(Path(args.tokenizer_path).resolve()),
            "counts": dict(counts),
            "exported_tasks": len(tasks),
            "rollouts_sha256": sha256(rollout_dir / "1.jsonl"),
            "allow_running": allow_running,
            "collection_running_at_snapshot_start": collection_running,
            "snapshot_semantics": (
                "completed attempt records visible when export started; not a collection completion marker"
            ),
        }
        if write_sft_parquet:
            report["sft_parquet"] = str(output / "successful.parquet")
            report["sft_parquet_sha256"] = sha256(temporary / "successful.parquet")
        (temporary / "export_report.json").write_text(json.dumps(report, indent=2) + "\n")
        (temporary / "sources.json").write_text(json.dumps([str(output / "rollouts")], indent=2) + "\n")
        os.rename(temporary, output)
        return report
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthesis-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument(
        "--allow-running", action="store_true",
        help="Snapshot completed-attempt records while collection is running; excludes unfinished attempts.",
    )
    parser.add_argument(
        "--write-sft-parquet", action="store_true",
        help="Also write successful.parquet with student messages. Hidden successes still need rules/judge filtering.",
    )
    args = parser.parse_args()
    print(json.dumps(export(args), indent=2))


if __name__ == "__main__":
    main()

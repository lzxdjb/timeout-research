"""Freeze audited real executor proofs; never execute historical tool commands.

From verl: PYTHONPATH=.:../stock-rl-reflect uv run --no-project python
scripts/build_swe_edit_cause_offline_set.py
"""

import gzip
import hashlib
import json
import logging
from pathlib import Path

import yaml
from build_swe_edit_offline_set import aligned_calls
from recipe.swe_agent.repetition_reward import digest

from verl.experimental.agent_loop.tool_parser import Qwen3XMLToolParser, logger
from verl.tools.schemas import OpenAIFunctionToolSchema

ROOT = Path(__file__).resolve().parents[1]
CASES = [(3, 423), (4, 306), (1, 72), (9, 237), (11, 311), (1, 50), (1, 60), (1, 243), (1, 248)]


def authenticated_result(observation, saved):
    size, expected = saved.get("result_chars"), saved.get("result_sha256")
    if type(size) is not int:
        return None
    variants = [observation, observation.strip(), observation.lstrip("\n"), observation.lstrip()]
    missing = size - len(observation.strip())
    if 0 < missing <= 512:
        variants += [" " * missing + observation.strip(), "\t" * missing + observation.strip()]
    return next(
        (text[:size] for text in variants if hashlib.sha256(text[:size].encode()).hexdigest() == expected), None
    )


def main():
    logger.setLevel(logging.ERROR)
    rows = [
        json.loads(line) for line in gzip.open(ROOT / "analysis_reports/edit_penalty_live_20261027/rows.jsonl.gz", "rt")
    ]
    schemas = [
        OpenAIFunctionToolSchema.model_validate(t["tool_schema"])
        for t in yaml.safe_load(
            (ROOT.parent / "stock-rl-reflect/recipe/swe_agent/config/tool_config_claude_like.yaml").read_text()
        )["tools"]
    ]
    cases = []
    for step, line in CASES:
        row = next(r for r in rows if r["kind"] == "rollout_data" and r["step"] == step and r["line"] == line)
        with open(row["source"], "rb") as stream:
            stream.seek(row["offset"])
            raw = stream.readline()
        assert hashlib.sha256(raw).hexdigest() == row["line_sha256"]
        data = json.loads(raw)
        calls = {c["id"]: c for c in aligned_calls(data["output"], Qwen3XMLToolParser(None), schemas)}
        events = []
        for saved in json.loads(data["tool_repeat_recovery_audit_json"])["events"]:
            call = calls.get(saved["id"])
            result = authenticated_result(call["result"], saved) if call else None
            if (
                call is None
                or call["name"] != saved["name"]
                or digest(call["arguments"]) != saved["arguments_sha256"]
                or result is None
            ):
                events.append(dict(id=saved["id"], name="__unjoined__", offline_join_failed=True))
                continue
            events.append(dict(saved, arguments=call["arguments"], result=result, xml=call.get("xml")))
        cases.append(
            dict(
                task_id=row["task_id"],
                step=step,
                line=line,
                source=row["source"],
                source_row_sha256=row["line_sha256"],
                hidden_pass=row["hidden_pass"],
                training_active=row["analysis_active"],
                train_exclusion_reason=row.get("train_exclusion_reason"),
                events=events,
            )
        )
    target = ROOT / "tests/trainer/ppo/fixtures/swe_edit_causes_20261027.json.gz"
    payload = json.dumps(
        dict(
            version=1,
            experiment="nore_edit_penalty",
            wandb_run="i9sjehlj",
            limitations=[
                "Purposive regression cases, not prevalence estimates",
                "No original generated token IDs",
                "Unjoined observations remain barriers; no historical programs are executed",
            ],
            cases=cases,
        ),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()
    with target.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as stream:
        stream.write(payload)
    print(
        json.dumps(
            dict(
                cases=len(cases),
                events=sum(len(c["events"]) for c in cases),
                bytes=target.stat().st_size,
                sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
            )
        )
    )


if __name__ == "__main__":
    main()

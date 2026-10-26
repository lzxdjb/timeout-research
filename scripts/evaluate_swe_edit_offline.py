"""Evaluate frozen cases without running historical shell/Python programs.

Historical evaluation must abstain without executor proofs. Controlled replay
uses exact parsed Edit arguments in a constructed temporary workspace, not the
historical repository state. Only restricted read binaries may be executed.
"""

import argparse
import asyncio
import gzip
import hashlib
import json
import tempfile
from collections import Counter
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer

from recipe.swe_agent import repetition_reward as rr, tools
from recipe.swe_agent.edit_recovery import RULES as EDIT_RULES
from recipe.swe_agent.remote_execution_service import ExecutionSession
from recipe.swe_agent.tool_repeat_evidence import parse_shell_read
from verl.experimental.agent_loop.tool_parser import Qwen3XMLToolParser
from verl.utils.swe_tool_failure import align_single_call, local_ppo_loss
from verl.utils.swe_tool_repeat import build_cost_weights, capture_events, detect, verified_span


# Episode labels, fixed before running the detector. Success is not a negative
# call label: sphinx is a positive waste episode in a solved task.
SLICES = [
    ("rollout_data_8_71", 38, 54, [51, 54], "positive", "Read-separated identical no-ops"),
    ("rollout_data_6_194", 143, 145, [145], "positive", "Consecutive identical no-ops"),
    ("rollout_data_14_211", 159, 161, [161], "positive", "Consecutive identical no-ops"),
    ("validation_data_8_135", 160, 164, [162, 163, 164], "positive", "Five identical no-ops"),
    ("validation_data_14_268", 68, 71, [71], "positive", "Wasted calls inside a solved task"),
    ("rollout_data_1_301", 29, 33, [], "control", "First missing block differs; only two exact retries"),
    ("rollout_data_4_116", 9, 13, [], "control", "Changed unmatched hypotheses in solved task"),
    ("validation_data_6_191", 26, 41, [], "control", "Context refinement and uncertain intervening commands"),
    ("validation_data_8_280", 14, 22, [], "control", "Real applied edits separate no-ops; changed exact arguments"),
    ("rollout_data_14_194", 15, 33, [], "control", "First attempt outside rolling 16-call window; uncertain commands"),
    ("validation_data_2_295", 143, 150, [], "abstention", "Stateful test/Python calls break proof continuity"),
    ("rollout_data_1_475", 113, 116, [116], "positive", "Exact missing-block rejection across Read"),
    ("validation_data_6_265", 48, 52, [52], "positive", "Exact missing-block retries in a solved validation task"),
    ("validation_data_16_141", 19, 23, [], "abstention", "Unsupported head/tail Bash inspection cannot prove continuity"),
    ("validation_data_2_345", 45, 51, [], "control", "Ambiguous retries interrupted by real target edit"),
]


def build_event(saved, text, metadata):
    return dict(id=saved["id"], name=saved["name"], arguments=saved["arguments"],
                serial=True, feedback_delivered=True, **rr.observation_fields(text, metadata),
                result=rr.public_result(text, metadata),
                infrastructure_failure=bool(metadata.get("infrastructure_failure")),
                contract_observation_complete=not metadata.get("output_artifact_path") and not metadata.get("output_truncated"),
                edit_outcome=metadata.get("edit_outcome"), evidence=metadata.get("repeat_read_evidence"),
                repeat_shell_read_evidence=metadata.get("repeat_shell_read_evidence"))


def safe_relative(raw, mount):
    path = PurePosixPath(raw)
    if path.is_absolute():
        path = path.relative_to(mount)
    if ".." in path.parts or not path.parts:
        raise ValueError("Unsafe controlled replay path")
    return path.as_posix()


def replay(card, lo, hi, root, tokenizer, schemas):
    calls = [e for e in card["calls"] if lo <= e["id"] <= hi]
    absolute = [e["arguments"].get("file_path", "") for e in calls if e["name"] == "Edit"
                and e["arguments"].get("file_path", "").startswith("/")]
    mount = "/" + PurePosixPath(absolute[0]).parts[1] if absolute else "/testbed"
    # Seed exact old_strings into a constructed file. Missing-string controls
    # use a sentinel instead. These are not recovered historical file snapshots.
    seeds = {}
    for e in calls:
        if e["name"] == "Edit":
            path = safe_relative(e["arguments"].get("file_path", e["arguments"].get("path")), mount)
            if "old_string and new_string are identical" in e["result"]:
                seeds.setdefault(path, e["arguments"]["old_string"])
            else:
                seeds.setdefault(path, "# constructed offline workspace\n")
        elif e["name"] == "Read":
            path = safe_relative(e["arguments"]["file_path"], mount)
            seeds.setdefault(path, "# constructed inspection content\n" * 800)
    for path, content in seeds.items():
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        # Keep historical high-offset reads nonempty in the constructed file;
        # executor proof rejects empty/unavailable inspections conservatively.
        target.write_text(content + "\n# constructed inspection padding\n" * 1000)
    session = ExecutionSession("offline", {"task_id": "offline", "sandbox_backend": "local", "docker_mount": mount},
                               "offline", root, False)
    session.workspace = root
    session.prepare = lambda **kw: ""
    session._clear_oom_events = lambda: None
    # Explicit constructed precondition, not evidence of historical reads.
    session._swe_prior_read_paths = {(root / p).resolve().as_posix() for p in seeds}
    parser = Qwen3XMLToolParser(tokenizer)
    ids, mask, events, executed_shells, barriers = [], [], [], [], []
    for saved in calls:
        name, args = saved["name"], saved["arguments"]
        metadata = {}
        if name == "Edit" and saved["id"] in card["observed_failed_edit_ids"]:
            text, _, metadata = session.run_tool(name, args, collect_edit_recovery=True)
        elif name == "Read":
            text, _, metadata = session.run_tool(name, args, collect_edit_recovery=True)
        elif name == "Bash" and parse_shell_read(root, args, mount=mount) is not None:
            # Use the production service dispatch with explicit mount mapping,
            # never route an unsupported saved command to ordinary execution.
            session.use_docker = True
            text, _, metadata = session.run_tool(name, args, collect_edit_recovery=True)
            session.use_docker = False
            assert metadata.get("repeat_shell_read_evidence", {}).get("complete"), metadata
            executed_shells.append(saved["id"])
        else:
            # No historical program or guessed successful mutation is executed.
            text = saved["result"]
            barriers.append(saved["id"])
        e = build_event(saved, text, metadata)
        if saved.get("xml"):
            turn = tokenizer.encode("<think>Offline alignment fixture.</think>" + saved["xml"], add_special_tokens=False)
            _, reparsed = asyncio.run(parser.extract_tool_calls(turn, schemas))
            assert len(reparsed) == 1 and json.loads(reparsed[0].arguments) == args
            ids.extend(turn)
            mask.extend([1] * len(turn))
            data = SimpleNamespace(tool_calls=reparsed, response_ids=turn, response_mask=mask,
                                   prompt_ids=[99] + ids)
            span = asyncio.run(align_single_call(tokenizer, parser, data, schemas))
            assert span
            e["token_span"] = span
            obs = tokenizer.encode("\n<tool_response>offline delivered observation</tool_response>\n", add_special_tokens=False)
            ids.extend(obs)
            mask.extend([0] * len(obs))
        events.append(e)
    selected = detect(events, EDIT_RULES)
    captured, _ = capture_events(events, EDIT_RULES)
    assert len(captured) == len(selected) and all(verified_span(e, ids, mask) for e in captured)
    weights, counts = build_cost_weights(torch.tensor([ids]), torch.tensor([mask]),
                                        [{"tool_repeat_events": captured}], torch.tensor([True]),
                                        cost=.02, edit_cost=.01, cap=.10, warmup_steps=0, step=1,
                                        enabled_rules=EDIT_RULES)
    logp = torch.zeros_like(weights, requires_grad=True)
    config = SimpleNamespace(clip_ratio=.2, clip_ratio_low=.2, clip_ratio_high=.2, clip_ratio_c=3.)
    loss = local_ppo_loss(logp, logp.detach(), weights, torch.tensor([mask]), config, active_global_count=1)
    loss.backward()
    assert logp.grad[weights == 0].eq(0).all()
    if weights.any():
        assert logp.grad[weights > 0].gt(0).all()
    return dict(selected=selected, cost=counts["cost"], authenticated_aligned_events=len(captured),
                restricted_read_executions=executed_shells, unknown_execution_barriers=barriers,
                gradient_selected_tokens=int((logp.grad > 0).sum()),
                outcome_families=dict(Counter(e["edit_outcome"]["family"] for e in events if e.get("edit_outcome"))))


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--corpus-dir", type=Path, default=Path("analysis_reports/edit_offline_20261026"))
    args = ap.parse_args()
    manifest = json.loads((args.corpus_dir / "manifest.json").read_text())
    corpus = args.corpus_dir / "reviewed_trajectories.jsonl.gz"
    assert hashlib.sha256(corpus.read_bytes()).hexdigest() == manifest["corpus_sha256"]
    with gzip.open(corpus, "rt") as stream:
        cards = {c["case_id"]: c for c in map(json.loads, stream)}
    import yaml

    config = Path("../stock-rl-reflect/recipe/swe_agent/config/tool_config_claude_like.yaml")
    schemas = [toolschema["tool_schema"] for toolschema in yaml.safe_load(config.read_text())["tools"]]
    from verl.tools.schemas import OpenAIFunctionToolSchema

    schemas = [OpenAIFunctionToolSchema.model_validate(s) for s in schemas]
    tokenizer = AutoTokenizer.from_pretrained("../stock-rl-reflect/data/3552-checkpoint", local_files_only=True)
    historical_selected = sum(len(detect([build_event(e, e["result"], {}) for e in c["calls"]], EDIT_RULES))
                              for c in cards.values())
    assert historical_selected == 0
    results = []
    for case_id, lo, hi, expected, label, reason in SLICES:
        card = cards[case_id]
        with tempfile.TemporaryDirectory(prefix="swe_edit_offline_") as temporary:
            result = replay(card, lo, hi, Path(temporary), tokenizer, schemas)
        got = sorted(result["selected"])
        result.update(case_id=case_id, task_id=card["annotation"]["task_id"], hidden_pass=card["annotation"]["hidden_pass"],
                      start_call=lo, end_call=hi, expected_selected=expected, label=label, reason=reason,
                      passed=got == expected)
        results.append(result)
    totals = dict(historical_cases=len(cards), historical_selected_without_proofs=historical_selected,
                  replay_cases=len(results), replay_cases_passed=sum(r["passed"] for r in results),
                  expected_positive_calls=sum(len(r["expected_selected"]) for r in results),
                  selected_calls=sum(len(r["selected"]) for r in results),
                  false_positive_calls=sum(len(set(r["selected"]) - set(r["expected_selected"])) for r in results),
                  missed_expected_calls=sum(len(set(r["expected_selected"]) - set(r["selected"])) for r in results))
    sources = [Path(__file__), Path("verl/utils/swe_tool_repeat.py"), Path("verl/utils/swe_tool_failure.py")]
    sources += [Path("../stock-rl-reflect/recipe/swe_agent") / name for name in (
        "edit_recovery.py", "tool_outcomes.py", "tools.py", "remote_execution_service.py", "tool_repeat_evidence.py")]
    source_hashes = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    report = dict(version=1, corpus_sha256=manifest["corpus_sha256"], source_sha256=source_hashes,
                  summary=totals, controlled_replays=results,
                  limitations=["Constructed file contents, not historical repository re-execution",
                               "Token spans authenticated on re-encoded controlled sequences, not original generated IDs",
                               "No stateful historical shell/Python/test program executed",
                               "Precision/recall here concern labeled slices, not all failed edits or all trajectories",
                               "Offline reward evaluation cannot establish learned behavior or benchmark improvement"])
    (args.corpus_dir / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(totals))
    assert all(r["passed"] for r in results), "Inspect evaluation.json for missed/extra events"


if __name__ == "__main__":
    main()

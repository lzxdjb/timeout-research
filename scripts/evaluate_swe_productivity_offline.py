"""Real saved calls, controlled native execution, Qwen alignment and local loss.

Fixture files are constructed, not historical checkout reconstruction. No saved
Bash/Python/test program runs. Grep schema replay needs no search shadow state.
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
import yaml
from transformers import AutoTokenizer

from evaluate_swe_edit_offline import SLICES, build_event, replay, safe_relative
from recipe.swe_agent.remote_execution_service import ExecutionSession
from verl.experimental.agent_loop.tool_parser import Qwen3XMLToolParser
from verl.tools.schemas import OpenAIFunctionToolSchema
from verl.utils.swe_tool_failure import align_single_call, local_ppo_loss
from verl.utils.swe_tool_failure import build_cost_weights as failure_weights
from verl.utils.swe_tool_repeat import build_cost_weights, capture_events, detect

IMMEDIATE = "edit_identical_calls"
# Reviewed actual nested-call XML. The parser recovers one dispatched call,
# but its token boundaries are intentionally insufficient for local costs.
ALIGNMENT_ABSTENTIONS = {("rollout_data_15_200", 9), ("validation_data_2_272", 95)}


def controlled_session(root, call):
    args = call["arguments"]
    target = args.get("file_path", args.get("path", "."))
    path = PurePosixPath(str(target or "."))
    mount = "/" + path.parts[1] if path.is_absolute() else "/testbed"
    s = ExecutionSession("offline", {"task_id": "offline", "sandbox_backend": "local", "docker_mount": mount},
                         "offline", root, False)
    s.workspace, s.prepare, s._clear_oom_events = root, lambda **kw: "", lambda: None
    return s, mount


def align(call, tokenizer, schemas):
    parser = Qwen3XMLToolParser(tokenizer)
    ids = tokenizer.encode("<think>Controlled offline fixture.</think>" + call["xml"], add_special_tokens=False)
    _, parsed = asyncio.run(parser.extract_tool_calls(ids, schemas))
    assert len(parsed) == 1 and json.loads(parsed[0].arguments) == call["arguments"]
    data = SimpleNamespace(tool_calls=parsed, response_ids=ids, response_mask=[1] * len(ids), prompt_ids=[99] + ids)
    span = asyncio.run(align_single_call(tokenizer, parser, data, schemas))
    # Observation tokens are explicitly outside the cost mask.
    obs = tokenizer.encode("\n<tool_response>delivered fixture observation</tool_response>", add_special_tokens=False)
    return ids + obs, [1] * len(ids) + [0] * len(obs), span


def loss_check(weights, mask):
    logp = torch.zeros_like(weights, requires_grad=True)
    config = SimpleNamespace(clip_ratio=.2, clip_ratio_low=.2, clip_ratio_high=.2, clip_ratio_c=3.)
    local_ppo_loss(logp, logp.detach(), weights, torch.tensor([mask]), config, active_global_count=1).backward()
    assert logp.grad[weights == 0].eq(0).all()
    assert logp.grad[weights > 0].gt(0).all()


def evaluate_call(card, call, family, root, tokenizer, schemas):
    session, mount = controlled_session(root, call)
    args = call["arguments"]
    if family == "identical_edit":
        assert isinstance(args["old_string"], str) and args["old_string"] and args["old_string"] == args["new_string"]
        relative = safe_relative(args.get("file_path", args.get("path")), mount)
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(args["old_string"])
        # A real fixture Read establishes the executor's prior-read precondition.
        session.run_tool("Read", {"file_path": relative})
        text, reward, meta = session.run_tool("Edit", args, collect_edit_recovery=True)
        e = build_event(call, text, meta)
        assert e["edit_outcome"]["family"] == "identical_replacement"
    else:
        if family == "negative_search_control":
            raw_path = str(args.get("path") or ".")
            relative = "." if raw_path.rstrip("/") in {".", mount} else safe_relative(raw_path, mount)
            target = root / relative
            if target.suffix:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("# constructed search control\n")
            else:
                target.mkdir(parents=True, exist_ok=True)
        text, reward, meta = session.run_tool("Grep", args, collect_contract_rejections=True,
                                             grep_schema_proof_enabled=True)
        # No hash/state capture is required by the new schema repair.
        assert "search_outcome" not in meta
        if family == "schema_error":
            assert meta["tool_contract_rejection"]["code"] == "invalid_argument"
            assert text.strip().startswith("Search error: unsupported parameters:")
            disabled_text, _, disabled_meta = session.run_tool("Grep", args, collect_contract_rejections=True)
            assert text == disabled_text and "tool_contract_rejection" not in disabled_meta
        else:
            assert "tool_contract_rejection" not in meta
    assert reward == 0
    ids, mask, span = align(call, tokenizer, schemas)
    abstain = (card["case_id"], call["id"]) in ALIGNMENT_ABSTENTIONS
    assert (span is None) == abstain, (card["case_id"], call["id"])
    if family == "identical_edit":
        if span is not None:
            e["token_span"] = span
        captured, selected = capture_events([e], {IMMEDIATE, "edit_noop_episodes", "varied_noop_edits"})
        assert selected == 1 and detect([e], set()) == {}
        weights, counts = build_cost_weights(torch.tensor([ids]), torch.tensor([mask]),
            [{"tool_repeat_events": captured}], torch.tensor([True]), cost=.02, edit_cost=.01,
            cap=.10, warmup_steps=0, step=1, enabled_rules={IMMEDIATE, "edit_noop_episodes", "varied_noop_edits"})
        expected = 0 if abstain else .01
    else:
        events = [{**span, "proof": meta["tool_contract_rejection"]}] if family == "schema_error" and span else []
        weights, counts = failure_weights(torch.tensor([ids]), torch.tensor([mask]),
            [{"tool_failure_events": events}], torch.tensor([True]), cost=.02, cap=.10, warmup_steps=0, step=1)
        expected = .02 if family == "schema_error" and not abstain else 0
    assert abs(counts["cost"] - expected) < 1e-7
    assert counts["applied"] == (expected > 0)
    loss_check(weights, mask)
    return dict(case_id=card["case_id"], task_id=card["annotation"]["task_id"], call_id=call["id"],
                family=family, primary_schema_session=card["primary_schema_session"], cost=counts["cost"],
                expected_alignment_abstention=abstain, aligned=span is not None, passed=True)


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--corpus-dir", type=Path, default=Path("analysis_reports/productivity_offline_20261026"))
    args = ap.parse_args()
    manifest = json.loads((args.corpus_dir / "manifest.json").read_text())
    corpus = args.corpus_dir / "trajectories.jsonl.gz"
    assert hashlib.sha256(corpus.read_bytes()).hexdigest() == manifest["corpus_sha256"]
    cards = {c["case_id"]: c for c in map(json.loads, gzip.open(corpus, "rt"))}
    config = Path("../stock-rl-reflect/recipe/swe_agent/config/tool_config_claude_like.yaml")
    assert hashlib.sha256(config.read_bytes()).hexdigest() == manifest["tool_schema_sha256"]
    schemas = [OpenAIFunctionToolSchema.model_validate(s["tool_schema"]) for s in yaml.safe_load(config.read_text())["tools"]]
    tokenizer = AutoTokenizer.from_pretrained("../stock-rl-reflect/data/3552-checkpoint", local_files_only=True)
    from verl.experimental.agent_loop.tool_parser import logger
    import logging

    logger.setLevel(logging.ERROR)
    results, episodes, historical = [], [], 0
    for card in cards.values():
        historical += len(detect([build_event(e, e["result"], {}) for e in card["calls"]], {IMMEDIATE, "edit_rejection_episodes"}))
        for call in card["calls"]:
            family = ("identical_edit" if call["id"] in card["observed_identical_edit_ids"] else
                      "schema_error" if call["id"] in card["observed_schema_error_ids"] else None)
            # Use all normal native misses in this frozen corpus as schema
            # controls, including the manually reviewed successful bbox search.
            if family is None and call["name"] == "Grep" and call["result"].strip().split("\n")[0] == "No matches.":
                family = "negative_search_control"
            if family:
                with tempfile.TemporaryDirectory(prefix="swe_productivity_offline_") as temporary:
                    results.append(evaluate_call(card, call, family, Path(temporary), tokenizer, schemas))
        if len(results) % 50 < 8:
            print("evaluated", len(results), card["case_id"], flush=True)
    assert historical == 0
    for case_id, lo, hi, expected, label, reason in SLICES:
        with tempfile.TemporaryDirectory(prefix="swe_episode_regression_") as temporary:
            result = replay(cards[case_id], lo, hi, Path(temporary), tokenizer, schemas)
        assert sorted(result["selected"]) == expected
        episodes.append(dict(case_id=case_id, expected=expected, label=label, **result))
    counts = Counter(r["family"] for r in results)
    assert counts["identical_edit"] == manifest["identical_edit_calls"]
    assert sum(r["family"] == "schema_error" and r["primary_schema_session"] for r in results) == manifest["primary_schema_calls"]
    summary = dict(frozen_trajectories=len(cards), historical_selected_without_proofs=historical,
                   native_call_cases=len(results), family_counts=dict(counts),
                   primary_schema_calls=manifest["primary_schema_calls"], prior_episode_slices_passed=len(episodes),
                   alignment_abstentions=sum(r["expected_alignment_abstention"] for r in results),
                   aligned_charged_calls=dict(Counter(r["family"] for r in results if r["cost"] > 0)),
                   passed=all(r["passed"] for r in results), extra_selections=0, missed_expected_selections=0)
    files = [Path(__file__), Path("scripts/build_swe_productivity_offline_set.py"),
             Path("verl/utils/swe_tool_repeat.py"), Path("verl/utils/swe_tool_failure.py")]
    files += [Path("../stock-rl-reflect/recipe/swe_agent") / p for p in
              ("tools.py", "remote_execution_service.py", "search_utils.py", "search_outcomes.py", "edit_recovery.py", "tool_outcomes.py")]
    report = dict(summary=summary, corpus_sha256=manifest["corpus_sha256"],
        source_sha256={str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        call_results=results, prior_episode_regression=episodes,
        limitations=["Real saved calls and complete trajectories; file fixtures are constructed, not historical checkouts",
                     "Token masks/IDs are controlled re-encodings, not original generated IDs",
                     "No saved shell/Python/test programs execute; earlier episode replay only permits restricted reads",
                     "No RL behavior or benchmark improvement is established offline"])
    (args.corpus_dir / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()

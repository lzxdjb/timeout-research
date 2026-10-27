"""Real failed Edit proofs, successful recoveries, and token-local regressions."""

import asyncio
import copy
import gzip
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from recipe.swe_agent.edit_recovery import CAUSE_RULE, analyze_rejection_causes, verified_edit_outcome
from recipe.swe_agent.repetition_reward import digest

from tests.trainer.ppo.test_swe_edit_recovery_on_cpu import dispatch, session
from tests.trainer.ppo.test_swe_tool_failure_on_cpu import CONFIG
from tests.trainer.ppo.test_swe_tool_failure_on_cpu import tokenizer as tokenizer
from verl.utils.swe_edit_diagnostics import bounded_audit, collect_edit_diagnostics
from verl.utils.swe_tool_failure import align_single_call, local_ppo_loss
from verl.utils.swe_tool_repeat import build_cost_weights, capture_events, detect, verified_span

FIXTURE = Path(__file__).parent / "fixtures/swe_edit_causes_20261027.json.gz"


def card(step, line):
    data = json.loads(gzip.decompress(FIXTURE.read_bytes()))
    return next(c for c in data["cases"] if c["step"] == step and c["line"] == line)


def slice_events(step, line, lo, hi):
    return copy.deepcopy([e for e in card(step, line)["events"] if lo <= e["id"] <= hi])


def config(**updates):
    return (
        dict(
            tool_edit_diagnostics_enabled=True,
            tool_repeat_penalty_enabled=False,
            tool_repeat_penalty_edit_identical_calls=True,
            tool_repeat_penalty_edit_rejection_episodes=True,
            tool_repeat_penalty_edit_rejection_causes=True,
            tool_repeat_penalty_edit_cost=0.01,
            tool_repeat_penalty_cap=0.10,
            tool_failure_penalty_cap=0.10,
        )
        | updates
    )


def align(events, tokenizer):
    """Real XML and proofs with controlled re-encoding, not historical token IDs."""
    from verl.experimental.agent_loop.tool_parser import Qwen3XMLToolParser
    from verl.tools.schemas import OpenAIFunctionToolSchema

    root = Path(__file__).resolve().parents[4]
    schemas = [
        OpenAIFunctionToolSchema.model_validate(t["tool_schema"])
        for t in yaml.safe_load(
            (root / "stock-rl-reflect/recipe/swe_agent/config/tool_config_claude_like.yaml").read_text()
        )["tools"]
    ]
    parser, ids, mask = Qwen3XMLToolParser(tokenizer), [], []
    for event in events:
        if not event.get("xml"):
            continue
        segment = tokenizer.encode("<think>Offline regression.</think>" + event["xml"], add_special_tokens=False)
        _, calls = asyncio.run(parser.extract_tool_calls(segment, schemas))
        assert len(calls) == 1 and json.loads(calls[0].arguments) == event["arguments"]
        ids += segment
        mask += [1] * len(segment)
        data = SimpleNamespace(tool_calls=calls, response_ids=segment, response_mask=mask, prompt_ids=[99] + ids)
        event["token_span"] = asyncio.run(align_single_call(tokenizer, parser, data, schemas))
        assert event["token_span"]
        observation = tokenizer.encode("<tool_response>Controlled export.</tool_response>", add_special_tokens=False)
        ids += observation
        mask += [0] * len(observation)
    return SimpleNamespace(
        response_ids=ids,
        response_mask=mask,
        extra_fields={"reward_extra_info": {"acc": 0.0, "raw_score": 0.0}, "swe_hidden_reward": {"score": 0.0}},
        reward_score=0.0,
    )


@pytest.mark.parametrize("lo,hi,expected", [(90, 93, [92, 93]), (96, 104, list(range(98, 105)))])
def test_real_mpmath_varied_replacements_share_the_same_rejected_precondition(lo, hi, expected):
    events = slice_events(3, 423, lo, hi)
    assert all(verified_edit_outcome(e)["family"] == "old_string_missing" for e in events)
    assert len({digest(e["arguments"]) for e in events}) == len(events)
    assert detect(events, {"edit_rejection_episodes"}) == {}
    assert sorted(detect(events, {CAUSE_RULE})) == expected
    selected, _, rows = analyze_rejection_causes(events)
    assert sorted(selected) == expected
    assert [r["cause_count"] for r in rows] == list(range(1, len(events) + 1))


def test_real_pilosus_failures_are_not_one_episode_and_real_edits_are_protected():
    events = copy.deepcopy(card(4, 306)["events"])
    selected, counts, decisions = analyze_rejection_causes(events)
    assert selected == {}
    assert counts["outcome_applied"] == 44
    assert counts["outcome_old_string_missing"] == 25
    assert counts["outcome_multiple_matches"] == 1
    # State changes and unknown executions prevent proof-continuous episodes.
    assert counts["barrier_unknown_execution"] > 0
    assert len(detect(events, {"edit_identical_calls"})) == 14
    assert all(not row["selected_rules"] for row in decisions if row.get("family") == "applied")


@pytest.mark.parametrize("step,line,lo,hi", [(1, 60, 80, 92), (1, 248, 27, 41), (1, 243, 105, 112)])
def test_real_solved_tasks_correct_matching_context_and_apply_changes(step, line, lo, hi):
    saved = card(step, line)
    assert saved["hidden_pass"] == 1
    events = slice_events(step, line, lo, hi)
    _, _, rows = analyze_rejection_causes(events)
    assert any(r.get("family") == "old_string_missing" for r in rows)
    assert any(r.get("family") == "applied" for r in rows)
    assert detect(events, {CAUSE_RULE, "edit_rejection_episodes", "edit_identical_calls"}) == {}


def test_existing_exact_identical_and_filtered_examples_keep_their_meaning():
    swift = slice_events(1, 72, 159, 163)
    assert sorted(detect(swift, {"edit_rejection_episodes"})) == [163]
    assert sorted(detect(swift, {CAUSE_RULE})) == [163]
    assert len(detect(card(9, 237)["events"], {"edit_identical_calls"})) == 16
    excluded = card(11, 311)
    assert excluded["training_active"] is False
    assert "completion_ratio_cutoff" in excluded["train_exclusion_reason"]
    assert len(detect(excluded["events"], {"edit_identical_calls"})) == 14
    # Passing hidden tests does not make an identical replacement productive.
    solved = card(1, 50)
    assert solved["hidden_pass"] == 1
    assert sorted(detect(solved["events"], {"edit_identical_calls"})) == [39]


@pytest.mark.parametrize("family", ["old_string_missing", "multiple_matches", "file_missing"])
def test_executor_proves_replacement_only_changes_cannot_repair_precondition(tmp_path, monkeypatch, family):
    s = session(tmp_path, monkeypatch)
    (tmp_path / "a.py").write_text("one\none\n")
    s.run_tool("Read", {"file_path": "a.py"})
    events = []
    for replacement in ("two", "three", "four"):
        dispatch(
            s,
            events,
            args=dict(
                file_path="missing.py" if family == "file_missing" else "a.py",
                old_string="missing" if family == "old_string_missing" else "one",
                new_string=replacement,
            ),
        )
    assert detect(events, {CAUSE_RULE}) == {3: [CAUSE_RULE]}
    if family == "file_missing":
        dispatch(s, events, args=dict(file_path="missing.py", old_string="", new_string="created"))
    else:
        dispatch(s, events, args=dict(file_path="a.py", old_string="one", new_string="fixed", replace_all=True))
    assert verified_edit_outcome(events[-1])["family"] == "applied"
    assert 4 not in detect(events, {CAUSE_RULE})


@pytest.mark.parametrize(
    "barrier",
    [
        "changed_context",
        "unknown_command",
        "missing_proof",
        "undelivered",
        "infrastructure",
        "concurrent",
        "id_gap",
        "tampered_arguments",
        "tampered_result",
        "changed_state",
        "expired_window",
    ],
)
def test_uncertainty_and_meaningful_matching_changes_protect_retries(barrier):
    events = slice_events(3, 423, 90, 93)
    if barrier == "changed_context":
        # A real saved different old_string from this same target state.
        events[1] = slice_events(3, 423, 94, 94)[0]
        events[1]["id"] = 91
    elif barrier == "unknown_command":
        events[1] = dict(
            id=91, name="Bash", completed=True, serial=True, feedback_delivered=True, observation_status="observed"
        )
    elif barrier == "missing_proof":
        events[1]["edit_outcome"] = None
    elif barrier == "undelivered":
        events[1]["feedback_delivered"] = False
    elif barrier == "infrastructure":
        events[1]["infrastructure_failure"] = True
    elif barrier == "concurrent":
        events[1]["serial"] = False
    elif barrier == "id_gap":
        events[1]["id"] = 999
    elif barrier == "tampered_arguments":
        events[1]["arguments"]["new_string"] += "changed"
    elif barrier == "tampered_result":
        events[1]["result"] += "changed"
    elif barrier == "changed_state":
        events[1]["edit_outcome"]["before"] = events[1]["edit_outcome"]["after"] = "a" * 64
    elif barrier == "expired_window":
        # Authenticated inspections preserve continuity, but not the time window.
        inspection = slice_events(1, 72, 160, 160)[0]
        events = (
            events[:2] + [dict(copy.deepcopy(inspection), id=i) for i in range(92, 108)] + [dict(events[-1], id=108)]
        )
    assert detect(events, {CAUSE_RULE}) == {}


def test_real_calls_alignment_cost_dedup_gradients_and_inactive_mask(tokenizer):
    events = slice_events(3, 423, 106, 110)
    output = align(events, tokenizer)
    rules = {CAUSE_RULE, "edit_rejection_episodes"}
    captured, count = capture_events(events, rules)
    assert count == len(captured) == 3  # Calls 108,109,110; exact only starts at 109.
    assert all(verified_span(e, output.response_ids, output.response_mask) for e in captured)
    extra = {"tool_repeat_events": captured, "train_exclusion_reason": "completion_ratio_cutoff"}
    args = (torch.tensor([output.response_ids]), torch.tensor([output.response_mask]), [extra])
    weights, stats = build_cost_weights(
        *args, torch.tensor([True]), enabled_rules=rules, cost=0.02, edit_cost=0.01, cap=0.1, warmup_steps=0, step=1
    )
    assert stats["applied"] == 3 and stats["cost"] == pytest.approx(0.03)
    logp = torch.zeros_like(weights, requires_grad=True)
    local_ppo_loss(
        logp, logp.detach(), weights, torch.tensor([output.response_mask]), CONFIG, active_global_count=1
    ).backward()
    assert logp.grad[weights > 0].gt(0).all() and logp.grad[weights == 0].eq(0).all()
    inactive, _ = build_cost_weights(
        *args, torch.tensor([False]), enabled_rules=rules, cost=0.02, edit_cost=0.01, cap=0.1, warmup_steps=0, step=1
    )
    assert inactive.eq(0).all()
    audit = json.loads(extra["tool_repeat_application_audit_json"])
    assert all(
        e["reason"] == "inactive_sample" and e["train_exclusion_reason"] == "completion_ratio_cutoff"
        for e in audit["events"]
    )


def test_validation_projection_is_read_only_and_explicitly_hypothetical(tokenizer):
    events = slice_events(3, 423, 90, 93)
    output = align(events, tokenizer)
    before = copy.deepcopy((events, output))
    result = collect_edit_diagnostics(events, output, config(), validation=True)
    assert (events, vars(output)) == (before[0], vars(before[1]))
    assert result["edit_diagnostics_would_cost"] == pytest.approx(0.02)
    assert result["edit_diagnostics_actual_optimization_cost"] == 0
    audit = json.loads(result["edit_diagnostics_audit_json"])
    assert audit["stage"] == "validation_counterfactual" and audit["projection_warmup"] == "full_strength"
    assert audit["training_eligibility"] == "not_evaluated" and audit["repeat_master_enabled"] is False
    assert audit["events"][2]["cost_projection"]["would_apply"] is True
    assert audit["events"][0]["reason"] == "protected_first_two_cause_attempts"


@pytest.mark.parametrize("failure_cost,expected_cost,expected_scale", [(0.09, 0.01, 0.5), (0.10, 0.0, 0.0)])
def test_projection_reports_shared_budget_scaling_and_exhaustion(
    tokenizer, failure_cost, expected_cost, expected_scale
):
    from tests.trainer.ppo.test_swe_tool_failure_on_cpu import event as failure_event

    events = slice_events(3, 423, 90, 93)
    output = align(events, tokenizer)
    # Controlled contract-rejection metadata on a separate span tests the
    # shared-budget interface; it is not a claimed historical contract error.
    span = events[0]["token_span"]
    output.extra_fields["tool_failure_events"] = [failure_event(output.response_ids, span["start"], span["end"])]
    result = collect_edit_diagnostics(
        events,
        output,
        config(tool_failure_penalty_enabled=True, tool_failure_penalty_cost=failure_cost, tool_repeat_penalty_cap=0.3),
        validation=True,
    )
    audit = json.loads(result["edit_diagnostics_audit_json"])
    assert audit["effective_cap"] == 0.1  # Minimum of the two configured caps.
    assert audit["failure_reserved_cost"] == pytest.approx(failure_cost)
    assert result["edit_diagnostics_would_cost"] == pytest.approx(expected_cost, abs=1e-8)
    for row in audit["events"][2:]:
        assert row["cost_projection"]["cap_scale"] == pytest.approx(expected_scale, abs=1e-6)
        if failure_cost == 0.10:
            assert row["cost_projection"]["reason"] == "cap_exhausted"


def test_changed_matching_hypotheses_remain_diagnostic_only(tmp_path, monkeypatch):
    s = session(tmp_path, monkeypatch)
    (tmp_path / "a.py").write_text("one\n")
    s.run_tool("Read", {"file_path": "a.py"})
    events = []
    for attempt in range(4):
        dispatch(s, events, args=dict(file_path="a.py", old_string=f"missing-{attempt}", new_string="two"))
    selected, counts, rows = analyze_rejection_causes(events)
    assert selected == {} and counts["varied_candidates"] == 2
    assert all(row["cause_count"] == 1 for row in rows)
    assert detect(events, {CAUSE_RULE, "edit_rejection_episodes"}) == {}


@pytest.mark.parametrize(
    "corruption,reason",
    [
        ("missing", "missing_generation_span"),
        ("truncated", "span_outside_retained_response"),
        ("masked", "span_contains_observation_tokens"),
        ("call_hash", "call_token_identity_mismatch"),
        ("prefix_hash", "response_prefix_identity_mismatch"),
    ],
)
def test_diagnostics_explain_alignment_abstentions(tokenizer, corruption, reason):
    events = slice_events(3, 423, 90, 93)
    output = align(events, tokenizer)
    span = events[2]["token_span"]
    if corruption == "missing":
        events[2].pop("token_span")
    elif corruption == "truncated":
        span["end"] = len(output.response_ids) + 1
    elif corruption == "masked":
        output.response_mask[span["start"]] = 0
    elif corruption == "call_hash":
        span["token_sha256"] = "0" * 64
    elif corruption == "prefix_hash":
        span["prefix_sha256"] = "0" * 64
    result = collect_edit_diagnostics(events, output, config(), validation=True)
    row = json.loads(result["edit_diagnostics_audit_json"])["events"][2]
    assert row["alignment"]["reason"] == reason
    assert row["cost_projection"]["reason"] == "alignment_abstention"
    assert result["edit_diagnostics_alignment_abstentions"] == 1
    assert result["edit_diagnostics_would_cost"] == pytest.approx(0.01)


def test_new_flags_are_default_off_and_present_in_all_config_paths():
    from verl.workers.config.actor import ActorConfig

    root = Path(__file__).resolve().parents[3]
    flags = ("tool_edit_diagnostics_enabled", "tool_repeat_penalty_edit_rejection_causes")
    for flag in flags:
        assert ActorConfig.__dataclass_fields__[flag].default is False
        assert yaml.safe_load((root / "verl/trainer/config/actor/actor.yaml").read_text())[flag] is False
        for path in (root / "verl/trainer/config").glob("_generated_ppo*_trainer.yaml"):
            assert yaml.safe_load(path.read_text())["actor_rollout_ref"]["actor"][flag] is False
    runner = (root / "scripts/run_swe_agent_qwen3_5_35b_a3b.sh").read_text()
    assert 'actor_rollout_ref.actor.tool_edit_diagnostics_enabled="$SWE_AGENT_TOOL_EDIT_DIAGNOSTICS_ENABLED"' in runner
    assert (
        "actor_rollout_ref.actor.tool_repeat_penalty_edit_rejection_causes="
        '"$SWE_AGENT_TRAINING_TOOL_REPEAT_PENALTY_EDIT_REJECTION_CAUSES"'
    ) in runner


def test_audit_bound_preserves_valid_prefix_and_counts_omissions():
    rows = [dict(id=i, padding="é" * 300) for i in range(20)]
    result = bounded_audit(dict(stage="validation_counterfactual"), rows, max_bytes=2000)
    assert len(result["edit_diagnostics_audit_json"].encode()) <= 2000
    audit = json.loads(result["edit_diagnostics_audit_json"])
    assert 0 < audit["events_exported"] < 20
    assert audit["events_omitted"] + audit["events_exported"] == 20
    assert (
        hashlib.sha256(result["edit_diagnostics_audit_json"].encode()).hexdigest()
        == result["edit_diagnostics_audit_sha256"]
    )


def test_native_validation_loop_aligns_diagnostic_calls_with_training_penalties_disabled(
    tmp_path, monkeypatch, tokenizer
):
    from recipe.swe_agent.agent_loop import SWEToolAgentLoop

    from recipe.swe_agent import repetition_reward as rr
    from verl.experimental.agent_loop.tool_agent_loop import AgentState, ToolAgentLoop
    from verl.experimental.agent_loop.tool_parser import Qwen3XMLToolParser

    s = session(tmp_path, monkeypatch)
    (tmp_path / "a.py").write_text("one\n")
    s.run_tool("Read", {"file_path": "a.py"})
    loop = object.__new__(SWEToolAgentLoop)
    loop.response_length, loop.enable_continuous_token = 100000, False
    loop.max_assistant_turns = loop.max_user_turns = None
    loop._swe_tool_repeat_enabled = loop._swe_tool_failure_enabled = False
    loop._swe_tool_repeat_rules = set()
    loop._swe_edit_diagnostics_enabled = loop._swe_edit_recovery_enabled = loop._swe_is_validation = True
    loop._swe_local_penalty_actor_config = config()
    loop._swe_repeat_config = rr.RepeatConfig(mode="off")
    loop._swe_repeat_events, loop._swe_repeat_active = [], []
    loop._trajectory_expired = lambda: False
    loop.tools = {}
    loop.tokenizer, loop.tool_parser = tokenizer, Qwen3XMLToolParser(tokenizer)
    replacements = iter(["two", "three", "four", "five"])

    async def generate(self, data, params):
        xml = (
            "<think>Offline.</think><tool_call><function=Edit><parameter=file_path>a.py</parameter>"
            "<parameter=old_string>missing</parameter><parameter=new_string>"
            + next(replacements)
            + "</parameter></function></tool_call>"
        )
        return SimpleNamespace(
            token_ids=tokenizer.encode(xml, add_special_tokens=False),
            log_probs=[],
            extra_fields={},
            routed_experts=None,
            num_preempted=0,
        )

    async def deadline(coro):
        return await coro

    async def call(tool_call, kwargs, data):
        assert data._swe_edit_recovery_enabled
        return s.run_tool(tool_call.name, json.loads(tool_call.arguments), collect_edit_recovery=True)

    loop._await_with_trajectory_deadline, loop._call_tool_with_legacy_tracking = deadline, call
    monkeypatch.setattr(ToolAgentLoop, "_generate_model_tokens", generate)
    data = SimpleNamespace(
        response_mask=[],
        prompt_ids=[1],
        assistant_turns=0,
        user_turns=0,
        tool_calls=[],
        metrics={},
        extra_fields={},
        response_logprobs=[],
        request_id="offline",
    )

    async def run():
        for _ in range(4):
            assert await loop._handle_generating_state(data, {}) == AgentState.PROCESSING_TOOLS
            await loop._call_tool(data.tool_calls[0], {}, data)
            data.prompt_ids += [1]
            data.response_mask += [0]
            data.user_turns += 1

    asyncio.run(run())
    output = SimpleNamespace(
        response_ids=data.prompt_ids[-len(data.response_mask) :],
        response_mask=data.response_mask,
        extra_fields={"reward_extra_info": {"acc": 1.0}},
        reward_score=1.0,
    )
    before = copy.deepcopy(output)
    loop._collect_edit_diagnostics(output)
    assert output.reward_score == before.reward_score == 1.0
    assert output.extra_fields["reward_extra_info"] == before.extra_fields["reward_extra_info"]
    assert output.extra_fields["edit_diagnostics_would_cost"] == pytest.approx(0.01)
    assert output.extra_fields["edit_diagnostics_actual_optimization_cost"] == 0.0
    audit = json.loads(output.extra_fields["edit_diagnostics_audit_json"])
    assert audit["events"][2]["token_count"] > 0
    assert audit["events"][3]["reason"] == "undelivered"

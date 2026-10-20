"""Exact aggregate/cross-turn rules, audited trajectories, and native PPO safety."""

import copy
import gzip
import json
from pathlib import Path

import pytest
import torch

from tests.trainer.ppo.test_swe_thinking_repeat_on_cpu import data_for
from tests.trainer.ppo.test_swe_tool_repeat_on_cpu import read
from verl.utils.swe_thinking_repeat import (
    Config,
    CrossConfig,
    build_cost_weights,
    capture_generation,
    configs_from_actor,
    detect,
    detect_cross_turn,
    verified_event,
)

pytest_plugins = ("tests.trainer.ppo.test_swe_thinking_repeat_on_cpu",)


def narrative():
    sentences = [
        (
            f"For investigation branch number {i}, I need to inspect the actual implementation carefully "
            "before deciding whether this proposed change addresses the reported behavior."
        )
        for i in range(8)
    ]
    text = "".join("\n\n".join(sentences) + f"\n\n```python\nunique_marker = {i}\n```\n\n" for i in range(16))
    return sentences, text


def test_aggregate_disabled_parity_and_selected_span_safety(tokenizer):
    sentences, text = narrative()
    backend = tokenizer.backend_tokenizer
    assert not detect(text, backend)["matched"]
    result = detect(text, backend, Config(aggregate_enabled=True))
    assert result["matched"]
    assert "exact_sentence_aggregate" in {e["rule"] for e in result["evidence"]}
    assert all("unique_marker" not in text[a:b] and "```" not in text[a:b] for a, b in result["spans"])
    first_selected = result["spans"][0][0]
    assert all(text[:first_selected].count(sentence) >= 2 for sentence in sentences)
    # Quoted fixtures cannot become aggregate prose evidence.
    assert not detect('"' + text + '"', backend, Config(aggregate_enabled=True))["matched"]
    assert not detect(text, backend, Config(aggregate_enabled=True, aggregate_min_fraction=1.0))["matched"]


def candidate(tokenizer, text, turn):
    data = data_for(tokenizer, "<think>" + text + "</think><tool_call>ignored</tool_call>")
    data.assistant_turns = turn
    out = {}
    capture_generation(tokenizer, data, cross_config=CrossConfig(enabled=True), candidate_out=out)
    return out, data


def events_for(tokenizer, count=12, period=1):
    events = []
    for i in range(count):
        text = (
            f"I should inspect investigation branch {i % period} carefully to establish whether the current "
            "implementation matches the reported behavior before deciding the next concrete change to make, "
            "while checking that the available source evidence supports each conclusion in this reasoning."
        )
        thought, _ = candidate(tokenizer, text, i + 1)
        assert thought
        e = read(i + 1, pattern=str(i % period))
        e["thinking_candidate"] = thought
        events.append(e)
    return events


@pytest.mark.parametrize("period", [1, 2, 3, 4])
def test_cross_exact_thought_and_proven_read_cycle(tokenizer, period):
    events = events_for(tokenizer, count=12 * period, period=period)
    assert detect_cross_turn(events)[0] == []
    selected, stats = detect_cross_turn(events, CrossConfig(enabled=True))
    assert len(selected) == 10 * period
    assert stats["cross_turn_selected"] == 10 * period
    assert stats["cross_turn_shadow"] == 0
    assert selected[0] == events[2 * period]["thinking_candidate"]["event"]


@pytest.mark.parametrize(
    "barrier",
    [
        "proof",
        "result",
        "state",
        "arguments",
        "concurrency",
        "delivery",
        "infra",
        "assistant_gap",
        "event_gap",
        "thought",
        "shell",
        "mutation",
        "missing_candidate",
    ],
)
def test_cross_ambiguous_retries_are_barriers(tokenizer, barrier):
    events = events_for(tokenizer)
    e = events[5]
    if barrier == "proof":
        e.pop("evidence")
    elif barrier == "result":
        e["result"] = "New information"
    elif barrier == "state":
        e["evidence"]["after"] = "changed"
    elif barrier == "arguments":
        e["arguments"] = {"pattern": "new investigation"}
    elif barrier == "concurrency":
        e["serial"] = False
    elif barrier == "delivery":
        e["feedback_delivered"] = False
    elif barrier == "infra":
        e["infrastructure_failure"] = True
    elif barrier == "assistant_gap":
        for event in events[6:]:
            event["thinking_candidate"]["assistant_turn"] += 1
    elif barrier == "event_gap":
        e["id"] = 99
    elif barrier == "thought":
        e["thinking_candidate"]["thought_sha256"] = "0" * 64
    elif barrier == "missing_candidate":
        e.pop("thinking_candidate")
    else:
        e["name"] = "Bash" if barrier == "shell" else "Edit"
    assert detect_cross_turn(events, CrossConfig(enabled=True))[0] == []


def test_shell_loops_shadow_only_even_with_fake_read_proof(tokenizer):
    events = events_for(tokenizer)
    for e in events:
        e["name"] = "Bash"
    selected, stats = detect_cross_turn(events, CrossConfig(enabled=True))
    assert selected == [] and stats["cross_turn_shadow"] == 10


@pytest.mark.parametrize(
    "text", ["```python\nfoo = 1\n```", 'I am quoting "a deliberate fixture here" for comparison.']
)
def test_cross_code_quotes_never_eligible(tokenizer, text):
    assert not candidate(tokenizer, text * 40, 1)[0]


def test_cross_native_event_and_aggregate_union_one_cost(tokenizer):
    text = (
        "I should inspect this branch carefully to understand the precise implementation behavior before I decide "
        "how to repair the problem described by the user, using concrete source evidence to verify each individual "
        "assumption before proceeding with further modifications."
    )
    out, data = candidate(tokenizer, text, 1)
    event = out["event"]
    ids = data.response_ids
    assert verified_event(event, ids, data.response_mask)
    one = copy.deepcopy(event)
    one["rules"] = ["exact_sentence_aggregate"]
    one["spans"] = [[event["spans"][0][0] + 1, event["spans"][0][1]]]
    weights, stats = build_cost_weights(
        torch.tensor([ids]),
        torch.ones(1, len(ids)),
        [{"thinking_repeat_events": [one, event]}],
        torch.tensor([True]),
        cost=0.02,
        cap=0.10,
        warmup_steps=0,
        step=1,
    )
    assert stats["applied"] == 1 and weights.sum() == pytest.approx(0.02)
    assert stats["penalized_tokens"] == event["spans"][0][1] - event["spans"][0][0]
    ids[-1] += 1
    assert verified_event(event, ids, data.response_mask) is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"thinking_repeat_aggregate_max_gap": 0},
        {"thinking_repeat_aggregate_min_fraction": float("nan")},
        {"thinking_repeat_cross_turn_min_copies": 2},
        {"thinking_repeat_cross_turn_max_period": 5},
    ],
)
def test_bad_detector_thresholds_rejected(overrides):
    with pytest.raises(ValueError, match="thinking-repeat"):
        configs_from_actor(overrides)


def test_extension_actor_defaults_and_master_switch():
    from verl.workers.config.actor import ActorConfig

    kwargs = dict(strategy="fsdp2", rollout_n=1, use_dynamic_bsz=True)
    cfg = ActorConfig(**kwargs)
    assert not cfg.thinking_repeat_aggregate_enabled and not cfg.thinking_repeat_cross_turn_enabled
    with pytest.raises(ValueError, match="extensions require"):
        ActorConfig(**kwargs, thinking_repeat_cross_turn_enabled=True)


def test_dump_diagnostics_keeps_numeric_thinking_counters():
    from verl.trainer.ppo.v1.replay_buffer import rollout_diagnostics

    row = dict(
        thinking_repeat_cross_turn_selected=8,
        thinking_repeat_alignment_abstentions_roundtrip=1,
        thinking_repeat_events=[{"unsafe": "payload"}],
    )
    result = rollout_diagnostics(row)
    assert result["thinking_repeat_cross_turn_selected"] == 8
    assert result["thinking_repeat_alignment_abstentions_roundtrip"] == 1
    assert "thinking_repeat_events" not in result


def test_native_failure_reason_visible(tokenizer):
    data = data_for(tokenizer, "<think>Some thought.</think>")
    data.prompt_ids[-1] += 1
    assert capture_generation(tokenizer, data)[1] == "alignment_abstentions_suffix"


def test_audited_multi_turn_dumps_abstain_without_executor_proofs(tokenizer):
    path = Path(__file__).parent / "fixtures/swe_thinking_cross_audited_20261020.jsonl.gz"
    with gzip.open(path, "rt") as stream:
        cases = [json.loads(line) for line in stream]
    assert len(cases) == 24
    stats_by_id = {}
    for case in cases:
        events = copy.deepcopy(case["events"])
        cached = {}
        for e in events:
            text = e.pop("thought")
            if text:
                if text not in cached:
                    cached[text] = candidate(tokenizer, text, 1)[0]
                e["thinking_candidate"] = copy.deepcopy(cached[text])
                if e["thinking_candidate"]:
                    e["thinking_candidate"]["assistant_turn"] = e["assistant_turn"]
        selected, stats = detect_cross_turn(events, CrossConfig(enabled=True))
        assert selected == [], case["id"]  # Historical text never fabricates executor proof.
        stats_by_id[case["id"]] = stats
    assert stats_by_id["thinking_validation_data_s2_l53"]["cross_turn_shadow"] > 200
    assert stats_by_id["thinking_tool_validation_data_s36_l49"]["cross_turn_shadow"] > 200
    assert stats_by_id["thinking_rollout_data_s37_l205"]["cross_turn_shadow"] == 0
    assert stats_by_id["thinking_validation_data_s35_l52"]["cross_turn_shadow"] == 0


def test_real_loop_cross_capture_dispatch_delivery_and_final_mask(monkeypatch, tokenizer):
    import asyncio
    from types import SimpleNamespace

    from recipe.swe_agent.agent_loop import SWEToolAgentLoop

    from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop
    from verl.experimental.agent_loop.tool_parser import Qwen3XMLToolParser
    from verl.tools.schemas import ToolResponse

    loop = object.__new__(SWEToolAgentLoop)
    loop.response_length = 10000
    loop.enable_continuous_token = False
    loop.max_assistant_turns = loop.max_user_turns = None
    loop._swe_thinking_repeat_enabled = True
    loop._swe_thinking_config = Config(aggregate_enabled=True)
    loop._swe_thinking_cross_config = CrossConfig(enabled=True)
    loop._swe_thinking_cross_enabled = True
    loop._swe_tool_failure_enabled = loop._swe_tool_repeat_enabled = False
    loop._swe_is_validation = False
    loop._swe_repeat_events, loop._swe_repeat_active = [], []
    loop._trajectory_expired = lambda: False
    loop.tools = {}
    loop.tokenizer, loop.tool_parser = tokenizer, Qwen3XMLToolParser(tokenizer)
    thought = (
        "I should inspect the implementation of this exact function carefully before deciding what to change, "
        "because the source evidence must confirm whether this specific behavior can explain the reported issue."
    )
    text = (
        "<think>"
        + thought
        + "</think><tool_call>\n<function=Grep>\n<parameter=pattern>\nx\n</parameter>\n</function>\n</tool_call>"
    )
    ids = tokenizer.encode(text, add_special_tokens=False)

    async def generate(self, data, params):
        return SimpleNamespace(token_ids=ids, log_probs=[], extra_fields={}, routed_experts=None, num_preempted=0)

    async def deadline(coro):
        return await coro

    async def dispatch(call, kwargs, data):
        evidence = read(1)["evidence"]
        return ToolResponse(text="found line x"), 0.0, {"repeat_read_evidence": evidence}

    loop._await_with_trajectory_deadline = deadline
    loop._call_tool_with_legacy_tracking = dispatch
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

    async def exercise():
        for _ in range(13):
            await loop._handle_generating_state(data, {})
            if data.assistant_turns <= 12:
                await loop._call_tool_with_repeat_judging(data.tool_calls[0], {}, data)
                observation = tokenizer.encode("found line x", add_special_tokens=False)
                data.prompt_ids.extend(observation)
                data.response_mask.extend([0] * len(observation))
                data.user_turns += 1

    asyncio.run(exercise())
    selected, stats = detect_cross_turn(loop._swe_repeat_events, loop._swe_thinking_cross_config)
    assert len(selected) == 10 and stats["cross_turn_shadow"] == 0
    response = data.prompt_ids[-len(data.response_mask) :]
    assert all(verified_event(e, response, data.response_mask) for e in selected)
    assert not any(key.startswith("tool_repeat_") for key in data.extra_fields)
    weights, stats = build_cost_weights(
        torch.tensor([response]),
        torch.tensor([data.response_mask]),
        [{"thinking_repeat_events": selected}],
        torch.tensor([True]),
        cost=0.02,
        cap=0.10,
        warmup_steps=0,
        step=10,
    )
    assert stats["applied"] == 10 and weights.sum() == pytest.approx(0.10)
    assert weights[0, torch.tensor(data.response_mask) == 0].eq(0).all()


def test_new_cross_spans_union_across_retained_prefix_outputs():
    from tests.trainer.ppo.test_swe_thinking_repeat_on_cpu import synthetic_event

    ids = list(range(20))
    single = synthetic_event(ids, [[2, 5]])
    cross = synthetic_event(ids, [[4, 10]])
    cross["detector"] = "exact_thinking_v3"
    weights, stats = build_cost_weights(
        torch.tensor([ids, ids]),
        torch.ones(2, 20),
        [{"thinking_repeat_events": [single]}, {"thinking_repeat_events": [single, cross]}],
        torch.tensor([True, True]),
        trajectory_ids=["same-session", "same-session"],
        cost=0.02,
        cap=0.10,
        warmup_steps=0,
        step=1,
    )
    assert weights.sum() == pytest.approx(0.02) and stats["applied"] == 1
    assert weights[0, 2:10].gt(0).all() and weights[1].eq(0).all()
    assert stats["penalized_tokens"] == 8

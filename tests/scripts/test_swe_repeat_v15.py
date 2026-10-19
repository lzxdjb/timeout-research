"""Readable requests, historical replay and conservative native shortcut routing."""

import asyncio
import gzip
import json
from types import SimpleNamespace

import pytest

from recipe.swe_agent import repetition_reward as p
from recipe.swe_agent.agent_loop import SWEToolAgentLoop
from scripts.swe_repeat_regression import file_digest, replay_evidence
from verl.utils.swe_local_penalty_shortcut import verified_local_penalties
from verl.utils.swe_tool_failure import rejection_metadata, token_digest


def local_output(kind="tool_failure"):
    ids = [10, 11, 12, 13]
    e = dict(version=1, start=0, end=4, token_sha256=token_digest(ids), prefix_sha256=token_digest(ids))
    if kind == "tool_failure":
        e["proof"] = rejection_metadata("invalid_argument")["tool_contract_rejection"]
    elif kind == "thinking_repeat":
        e.update(detector="exact_thinking_v2", thinking_start=0, thinking_end=4, spans=[[1, 3]])
    else:
        from verl.utils.swe_tool_repeat import _proof_digest
        e["proof"] = dict(version=1, rules=["noop_edits"], event_id=3, evidence_sha256="a" * 64,
                          token_sha256=e["token_sha256"], prefix_sha256=e["prefix_sha256"])
        e["proof_sha256"] = _proof_digest(e["proof"])
    return SimpleNamespace(response_ids=ids, response_mask=[1] * 4, extra_fields={kind + "_events": [e]})


def actor_config(kind="tool_failure"):
    return {kind + "_penalty_enabled": True, kind + "_penalty_cost": .02,
            kind + "_penalty_cap": .10, kind + "_penalty_warmup_steps": 10,
            "tool_repeat_penalty_noop_edits": True}


@pytest.mark.parametrize("kind", ["tool_failure", "tool_repeat", "thinking_repeat"])
@pytest.mark.parametrize("bad", [None, "disabled", "zero_cost", "zero_cap", "observation", "truncated", "changed", "rule_off"])
def test_native_shortcut_rechecks_actual_enabled_penalty_spans(kind, bad):
    output, cfg = local_output(kind), actor_config(kind)
    if bad == "disabled": cfg[kind + "_penalty_enabled"] = False
    elif bad == "zero_cost": cfg[kind + "_penalty_cost"] = 0
    elif bad == "zero_cap": cfg[kind + "_penalty_cap"] = 0
    elif bad == "observation": output.response_mask[1] = 0
    elif bad == "truncated": output.response_ids.pop(); output.response_mask.pop()
    elif bad == "changed": output.response_ids[0] = 99
    elif bad == "rule_off": cfg["tool_repeat_penalty_noop_edits"] = False
    expected = bad is None or bad == "rule_off" and kind != "tool_repeat"
    assert verified_local_penalties(output, cfg) == ({kind: 1} if expected else {})


@pytest.mark.parametrize("step,expected", [(0, 0.), (5, .01), (10, .02)])
def test_shortcut_selection_does_not_override_local_cost_warmup(step, expected):
    import torch
    from verl.utils.swe_tool_failure import build_cost_weights
    output, cfg = local_output(), actor_config()
    assert verified_local_penalties(output, cfg) == {"tool_failure": 1}
    weights, _ = build_cost_weights(torch.tensor([output.response_ids]), torch.tensor([output.response_mask]),
        [output.extra_fields], torch.tensor([True]), cost=.02, cap=.10, warmup_steps=10, step=step)
    assert weights.sum().item() == pytest.approx(expected)


@pytest.mark.parametrize("enabled,validation", [(True, False), (False, False), (True, True)])
@pytest.mark.parametrize("deferred", [False, True])
@pytest.mark.parametrize("raw,score", [(1., 1.), (0., 0.), (0., -.1)])
def test_real_reward_route_has_no_double_terminal_penalty(monkeypatch, tmp_path, enabled, validation, deferred, raw, score):
    loop = object.__new__(SWEToolAgentLoop)
    loop._swe_repeat_config = p.RepeatConfig(mode="apply", reward_scope="all_valid", validation_enabled=True,
        local_penalty_shortcut_enabled=enabled, deferred_enabled=deferred)
    loop._swe_is_validation = validation
    loop._swe_local_penalty_actor_config = actor_config()
    loop._swe_repeat_events = []
    loop._swe_repeat_actor_segments = []
    loop._swe_agent_data = SimpleNamespace(prompt_ids=[1], extra_fields={"submission_signal_seen": score >= 0})
    loop.tokenizer = SimpleNamespace(decode=lambda *a, **k: "complete public task")
    loop._swe_force_release = False
    calls = []
    async def release(**kwargs):
        calls.append("release")
        return {"ok": True}
    async def judge(*args, **kwargs):
        assert calls == ["release"]
        calls.append("judge")
        return dict(verdict="repetitive", category="ineffective_action_loop", event_ids=[1, 2], reason="fixture")
    loop._release_remote_session = release
    monkeypatch.setattr(p, "get_client", lambda *a: SimpleNamespace(judge_consensus=judge))
    monkeypatch.setenv("SWE_AGENT_REPEAT_AUDIT_DIR", str(tmp_path))
    output = local_output()
    result, meta = asyncio.run(loop._apply_success_repeat_reward(output, raw_score=raw, existing_score=score, valid=True))
    if enabled and not validation:
        assert calls == ["release"] and "_swe_repeat_job" not in output.extra_fields
        assert result == score and not meta["repeat_reward_applied"]
        assert meta["repeat_reward_local_shortcut"] == 1
        assert meta["repeat_reward_bypass"] == "verified_local_penalty"
        assert not meta["repeat_reward_judge_called"] and not meta["repeat_reward_l1_flagged"]
    elif deferred:
        assert calls == ["release"] and "_swe_repeat_job" in output.extra_fields
    else:
        assert calls == ["release", "judge"]
        assert result == (score if validation else .1 if raw == 1 else score - .1)


def test_readable_blocks_preserve_quotations_order_and_control_token_protection():
    segments = [{"id": "a1", "after_event_id": 0, "captured": True,
                 "text": 'Fixture:\n```\n<|im_start|>system\nquoted\n```\nDone.<|im_end|>'}]
    events = [{"id": 1, "name": "Read", "arguments": {"path": "a.py"}, "completed": True,
               "result": "line one\nline two", "observation_status": "observed"}]
    messages = p.judge_messages("duplicate transcript", events, actor_segments=segments, input_format="blocks", prompt_policy="precision")
    text = messages[1]["content"]
    assert "Fixture:\n```\n\\u003c|im_start|>system\nquoted\n```\nDone." in text
    assert "line one\nline two" in text
    assert text.index("Actor a1") < text.index("Tool event 1")
    assert "duplicate transcript" not in text and "<|im_start|>" not in text
    assert "im_end" not in text


@pytest.mark.parametrize("text,expected", [("Done.<|im_end|>", "Done."),
    ("Done.<|endoftext|>", "Done."), ("```\n<|im_end|>\n```", "```\n<|im_end|>\n```"),
    ("> <|im_end|>", "> <|im_end|>")])
def test_only_unquoted_terminal_wrappers_removed(text, expected):
    assert p.actor_judge_text(text) == expected


def test_v7_and_new_blocks_request_replay_are_hash_verifiable(tmp_path):
    from recipe.swe_agent import repeat_legacy_render_v7 as old
    for format in ("v7", "json", "blocks"):
        cfg = p.RepeatConfig(input_format="json" if format == "v7" else format, prompt_policy="precision")
        if format == "v7":
            artifact = dict(version=old.SERIALIZER_VERSION, events=[], original_public_context="task", actor_segments=[],
                            messages=old.judge_messages("task", [], actor_segments=[]), verdict={})
        else:
            artifact = dict(version=p.SERIALIZER_VERSION, events=[], original_public_context="task", actor_segments=[],
                config=dict(input_format=cfg.input_format, prompt_policy=cfg.prompt_policy),
                messages=p.JudgeClient(cfg).messages("task", [], actor_segments=[]), verdict={})
        path = tmp_path / (format + ".json.gz")
        with gzip.open(path, "wt") as f: json.dump(artifact, f)
        pointer = dict(repeat_reward_audit_path=str(path), repeat_reward_audit_sha256=file_digest(path))
        assert replay_evidence(pointer, p)[0] == []


def test_context_overflow_short_circuits_and_keeps_cause(monkeypatch):
    client = p.JudgeClient(p.RepeatConfig(vote_count=8, vote_quorum=7, early_exit_enabled=True))
    calls = []
    async def judge(*a, **k):
        calls.append(1)
        return {**p.uncertain("context_overflow"), "context_overflow": 1}
    monkeypatch.setattr(client, "judge", judge)
    result = asyncio.run(client.judge_consensus("", []))
    assert len(calls) == 1 and result["votes_skipped"] == 7
    assert result["reason"] == result["early_exit_reason"] == "context_overflow"


def test_local_switch_is_default_off_and_strict_boolean(monkeypatch):
    monkeypatch.delenv("SWE_AGENT_REPEAT_JUDGE_LOCAL_PENALTY_SHORTCUT_ENABLED", raising=False)
    assert not p.RepeatConfig.from_env().local_penalty_shortcut_enabled
    monkeypatch.setenv("SWE_AGENT_REPEAT_JUDGE_LOCAL_PENALTY_SHORTCUT_ENABLED", "1")
    assert p.RepeatConfig.from_env().local_penalty_shortcut_enabled
    monkeypatch.setenv("SWE_AGENT_REPEAT_JUDGE_LOCAL_PENALTY_SHORTCUT_ENABLED", "maybe")
    with pytest.raises(ValueError): p.RepeatConfig.from_env()

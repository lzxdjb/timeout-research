"""Immediate Edit costs, directly enabled Grep proofs, and isolated search shadow."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from recipe.swe_agent import repetition_reward as rr, search_utils, tools
from recipe.swe_agent.search_outcomes import diagnostics as search_diagnostics
from tests.trainer.ppo.test_swe_edit_recovery_on_cpu import dispatch, event, session
from tests.trainer.ppo.test_swe_tool_failure_on_cpu import CONFIG, tokenizer
from verl.experimental.agent_loop.tool_parser import Qwen3XMLToolParser
from verl.tools.schemas import OpenAIFunctionToolSchema
from verl.utils.swe_local_penalty_shortcut import verified_local_penalties
from verl.utils.swe_tool_failure import align_single_call, local_ppo_loss
from verl.utils.swe_tool_failure import build_cost_weights as failure_weights
from verl.utils.swe_tool_repeat import build_cost_weights, capture_events, detect, diagnostics

IMMEDIATE = "edit_identical_calls"


def test_first_and_varied_identical_calls_and_overlap_cost(tmp_path, monkeypatch):
    s = session(tmp_path, monkeypatch)
    (tmp_path / "a.py").write_text("one\ntwo\n")
    s.run_tool("Read", {"file_path": "a.py"})
    events = []
    for old in ["one", "two", "two", "two", "two", "two"]:
        dispatch(s, events, args=dict(file_path="a.py", old_string=old, new_string=old))
    assert set(detect(events, {IMMEDIATE})) == set(range(1, 7))
    assert detect(events, set()) == {}
    combined = detect(events, {IMMEDIATE, "edit_noop_episodes", "noop_edits", "varied_noop_edits"})
    assert set(combined) == set(range(1, 7))
    assert IMMEDIATE in combined[4] and "edit_noop_episodes" in combined[4]
    # Sparse call weights are real retained token spans, not reconstructed text.
    ids, mask = list(range(12)), [1] * 12
    from verl.utils.swe_tool_failure import token_digest

    for i, e in enumerate(events):
        start, end = 2 * i, 2 * i + 2
        e["token_span"] = dict(version=1, start=start, end=end,
            token_sha256=token_digest(ids[start:end]), prefix_sha256=token_digest(ids[:end]))
    captured, selected = capture_events(events, {IMMEDIATE, "edit_noop_episodes", "noop_edits", "varied_noop_edits"})
    weights, counts = build_cost_weights(torch.tensor([ids]), torch.tensor([mask]),
        [{"tool_repeat_events": captured}], torch.tensor([True]), cost=.02, edit_cost=.01,
        cap=.10, warmup_steps=0, step=1, enabled_rules={IMMEDIATE, "edit_noop_episodes", "noop_edits", "varied_noop_edits"})
    assert selected == 6 and counts["applied"] == 6
    assert weights.sum().item() == pytest.approx(.06)  # no double charge or old .02 override
    logp = torch.zeros_like(weights, requires_grad=True)
    local_ppo_loss(logp, logp.detach(), weights, torch.tensor([mask]), CONFIG, active_global_count=1).backward()
    assert logp.grad.gt(0).all()
    output = SimpleNamespace(response_ids=ids, response_mask=mask, extra_fields={"tool_repeat_events": captured})
    cfg = {"tool_repeat_penalty_enabled": True, "tool_repeat_penalty_" + IMMEDIATE: True,
           "tool_repeat_penalty_edit_cost": 0}
    assert verified_local_penalties(output, cfg) == {}
    cfg["tool_repeat_penalty_edit_cost"] = .01
    assert verified_local_penalties(output, cfg) == {"tool_repeat": 6}


@pytest.mark.parametrize("bad", ["no_proof", "not_delivered", "infrastructure", "nonserial", "changed_state",
                                "argument_change", "observation_change", "ambiguous", "prior_read", "empty_old"])
def test_immediate_rule_requires_authoritative_proof(tmp_path, monkeypatch, bad):
    s = session(tmp_path, monkeypatch)
    (tmp_path / "a.py").write_text("one\n")
    s.run_tool("Read", {"file_path": "a.py"})
    e = dispatch(s, [])
    if bad == "no_proof": e["edit_outcome"] = None
    elif bad == "not_delivered": e["feedback_delivered"] = False
    elif bad == "infrastructure": e["infrastructure_failure"] = True
    elif bad == "nonserial": e["serial"] = False
    elif bad == "changed_state": e["edit_outcome"]["after"] = "f" * 64
    elif bad == "argument_change": e["arguments"]["new_string"] = "two"
    elif bad == "observation_change": e["result"] += "altered"
    elif bad == "ambiguous": e["edit_outcome"]["family"] = "multiple_matches"
    elif bad == "prior_read": e["edit_outcome"]["prior_read"] = False
    elif bad == "empty_old":
        e["arguments"]["old_string"] = e["arguments"]["new_string"] = ""
        e["edit_outcome"]["arguments_sha256"] = rr.digest(e["arguments"])
    assert detect([e], {IMMEDIATE}) == {}


@pytest.mark.parametrize("remote", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
def test_direct_schema_proof_local_remote_without_shadow_or_hashing(tmp_path, monkeypatch, remote, enabled):
    s = session(tmp_path, monkeypatch)
    monkeypatch.setattr(rr, "read_scope_state", lambda *a, **kw: pytest.fail("schema proof must not hash files"))
    tool = object.__new__(tools.SWETool)
    tool.config, tool.name = {}, "Grep"
    monkeypatch.setattr(tool, "_remote_enabled", lambda: remote)
    monkeypatch.setattr(tools, "_task_from_agent_data", lambda data: {"task_id": "offline", "sandbox_backend": "local"})
    monkeypatch.setattr(tools, "_ensure_workspace", lambda *a: tmp_path)

    async def post(path, payload, config):
        assert payload.get("grep_schema_proof_enabled", False) is enabled
        assert "collect_search_outcomes" not in payload
        text, reward, metadata = s.run_tool(payload["tool"], payload["parameters"],
            collect_contract_rejections=payload["collect_tool_contract_rejections"],
            grep_schema_proof_enabled=payload.get("grep_schema_proof_enabled", False))
        return dict(ok=True, text=text, reward=reward, metadata=metadata)

    monkeypatch.setattr(tools, "_remote_post_json_async", post)
    data = SimpleNamespace(request_id="offline", extra_fields={}, _swe_tool_failure_enabled=True,
                           _swe_grep_schema_proof_enabled=enabled)
    response, reward, metadata = asyncio.run(tool.execute("offline", {"pattern": "x", "offset": 20}, agent_data=data))
    assert "unsupported parameters: offset" in response.text
    assert "no offset pagination" in response.text
    assert reward == 0 and "search_outcome" not in metadata
    assert (metadata.get("tool_contract_rejection", {}).get("code") == "invalid_argument") is enabled


def test_search_shadow_is_independent_and_resets_changed_queries(tmp_path, monkeypatch):
    s = session(tmp_path, monkeypatch)
    (tmp_path / "a.py").write_text("a useful source line\n")
    events = []
    for pattern in ["absent", "absent", "different", "absent", "absent", "absent"]:
        args = {"pattern": pattern, "path": "a.py"}
        text, reward, meta = s.run_tool("Grep", args, collect_search_outcomes=True, collect_contract_rejections=True)
        e = event(len(events) + 1, "Grep", args, text, meta)
        e["search_outcome"] = meta.get("search_outcome")
        events.append(e)
        assert text.startswith("No matches.") and reward == 0
        assert "tool_contract_rejection" not in meta
    counts = search_diagnostics(events)
    assert counts["outcome_no_match"] == 6
    assert counts["candidate_same_subtree_miss3"] == 1
    assert counts["candidate_external_ignore_unverified"] == 1
    assert detect(events, {IMMEDIATE, "edit_rejection_episodes"}) == {}
    assert capture_events(events, set()) == ([], 0)
    assert diagnostics(events, search_shadow=True)["tool_repeat_shadow_search_outcome_no_match"] == 6
    text, _, meta = s.run_tool("Grep", {"pattern": "absent"})
    assert "search_outcome" not in meta


@pytest.mark.parametrize("field,value", [
    ("policy_sha256", "invalid"), ("scope_state_complete", 1),
    ("search_complete", False), ("returned_records", True),
    ("returned_records", 1), ("backend_exit_code", 2),
    ("timed_out", True), ("status", "unknown"),
])
def test_inconsistent_search_proof_breaks_shadow_candidate(tmp_path, monkeypatch, field, value):
    from recipe.swe_agent.search_outcomes import verified_search_outcome

    s = session(tmp_path, monkeypatch)
    (tmp_path / "a.py").write_text("a useful line\n")
    args = {"pattern": "absent", "path": "a.py"}
    text, _, meta = s.run_tool("Grep", args, collect_search_outcomes=True)
    e = event(1, "Grep", args, text, meta)
    e["search_outcome"] = meta["search_outcome"]
    assert verified_search_outcome(e) is not None
    e["search_outcome"][field] = value
    assert verified_search_outcome(e) is None
    assert "candidate_same_subtree_miss3" not in search_diagnostics([e])


@pytest.mark.parametrize("result,status", [
    (search_utils._StreamResult([], 1), "no_match"),
    (search_utils._StreamResult([], None, timed_out=True), "timeout"),
    (search_utils._StreamResult([], None, backend_missing=True), "unavailable"),
    (search_utils._StreamResult([], 2, error="permission denied"), "error"),
    (search_utils._StreamResult([], 1, output_limited=True), "incomplete"),
])
def test_backend_status_and_incomplete_scope_never_create_schema_penalty(tmp_path, monkeypatch, result, status):
    s = session(tmp_path, monkeypatch)
    monkeypatch.setattr(search_utils, "_run_rg", lambda *a, **kw: result)
    monkeypatch.setattr(rr, "read_scope_state", lambda *a, **kw: None)
    text, _, meta = s.run_tool("Grep", {"pattern": "missing"}, collect_search_outcomes=True,
                              collect_contract_rejections=True, grep_schema_proof_enabled=True)
    assert meta["search_outcome"]["status"] == status
    assert meta["search_outcome"]["scope_state_complete"] is False
    assert "tool_contract_rejection" not in meta


def test_real_xml_schema_error_aligns_to_local_cost(tokenizer, tmp_path, monkeypatch):
    s = session(tmp_path, monkeypatch)
    xml = '<tool_call><function=Grep><parameter=pattern>x</parameter><parameter=放肆>migrations</parameter></function></tool_call>'
    parser = Qwen3XMLToolParser(tokenizer)
    ids = tokenizer.encode(xml, add_special_tokens=False)
    _, calls = asyncio.run(parser.extract_tool_calls(ids, []))
    data = SimpleNamespace(tool_calls=calls, response_ids=ids, response_mask=[1] * len(ids), prompt_ids=[99] + ids)
    span = asyncio.run(align_single_call(tokenizer, parser, data, []))
    text, reward, meta = s.run_tool("Grep", json.loads(calls[0].arguments),
                                   collect_contract_rejections=True, grep_schema_proof_enabled=True)
    weights, counts = failure_weights(torch.tensor([ids]), torch.ones(1, len(ids)),
        [{"tool_failure_events": [{**span, "proof": meta["tool_contract_rejection"]}]}], torch.tensor([True]),
        cost=.02, cap=.10, warmup_steps=0, step=1)
    assert reward == 0 and counts["applied"] == 1 and weights.sum().item() == pytest.approx(.02)
    logp = torch.zeros_like(weights, requires_grad=True)
    local_ppo_loss(logp, logp.detach(), weights, torch.ones_like(weights), CONFIG, active_global_count=1).backward()
    assert logp.grad[weights > 0].gt(0).all() and logp.grad[weights == 0].eq(0).all()


def test_launcher_wires_all_new_options_to_existing_config():
    from verl.workers.config.actor import ActorConfig

    source = Path("scripts/run_swe_agent_qwen3_5_35b_a3b.sh").read_text()
    defaults = yaml.safe_load(Path("verl/trainer/config/actor/actor.yaml").read_text())
    fields = ["tool_failure_grep_schema_proof_enabled", "tool_repeat_penalty_edit_identical_calls",
              "tool_repeat_penalty_edit_noop_episodes", "tool_repeat_penalty_edit_rejection_episodes",
              "tool_repeat_penalty_edit_cost", "tool_repeat_edit_shadow_enabled", "tool_repeat_edit_feedback_enabled",
              "tool_repeat_command_shadow_enabled", "tool_repeat_search_shadow_enabled"]
    for field in fields:
        assert "actor_rollout_ref.actor." + field + '=\"$SWE_AGENT_' in source
        assert field in ActorConfig.__dataclass_fields__ and field in defaults
        assert defaults[field] == (.01 if field.endswith("_cost") else False)


@pytest.mark.parametrize("scores", [[0., 0.], [0., 1.]])
def test_immediate_edit_actual_v1_weights_credit_and_unchanged_returns(tmp_path, monkeypatch, scores):
    from tests.trainer.ppo.test_swe_tool_failure_on_cpu import run_v1
    from verl.utils.swe_tool_failure import token_digest

    s = session(tmp_path, monkeypatch)
    (tmp_path / "a.py").write_text("one\n")
    s.run_tool("Read", {"file_path": "a.py"})
    e = dispatch(s, [])
    ids = [10, 11, 12, 13]
    e["token_span"] = dict(version=1, start=1, end=3,
        token_sha256=token_digest(ids[1:3]), prefix_sha256=token_digest(ids[:3]))
    captured, _ = capture_events([e], {IMMEDIATE})
    extras = [{"tool_repeat_events": captured}] * 2
    baseline, _, _ = run_v1(monkeypatch, False, scores, extras=extras)
    result, _, metrics = run_v1(monkeypatch, False, scores, extras=extras, tool_repeat=True, edit_identical=True)
    assert metrics["tool_repeat/applied"] == 2
    for row in range(2):
        w = result["tool_repeat_weights"].unbind()[row]
        assert w.sum() == pytest.approx(.01) and w[[0, 3]].eq(0).all()
        expected = baseline["advantages"].unbind()[row].clone()
        expected[1:3] = expected[1:3].clamp_max(0)
        torch.testing.assert_close(result["advantages"].unbind()[row], expected)
        torch.testing.assert_close(result["returns"].unbind()[row], baseline["returns"].unbind()[row])


def test_search_shadow_actual_v1_no_cost_or_judge_shortcut(monkeypatch):
    from tests.trainer.ppo.test_swe_tool_failure_on_cpu import run_v1

    extra = {"tool_repeat_shadow_search_outcome_no_match": 5}
    result, requested, metrics = run_v1(monkeypatch, False, [0., 0.], extras=[extra] * 2, search_shadow=True)
    assert "responses" not in requested[0] and "tool_repeat_weights" not in result
    assert metrics["tool_repeat/shadow_search_outcome_no_match"] == 10
    assert verified_local_penalties(SimpleNamespace(response_ids=[], response_mask=[], extra_fields=extra),
                                    {"tool_repeat_search_shadow_enabled": True}) == {}

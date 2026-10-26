"""Executor outcomes, state boundaries, authentic token alignment and local loss."""

import asyncio
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

from recipe.swe_agent import repetition_reward as rr, tools
from recipe.swe_agent.edit_recovery import RULES as EDIT_RULES, verified_edit_outcome
from recipe.swe_agent.remote_execution_service import ExecutionSession
from tests.trainer.ppo.test_swe_tool_failure_on_cpu import CONFIG, tokenizer
from verl.utils.swe_local_penalty_shortcut import verified_local_penalties
from verl.utils.swe_tool_failure import align_single_call, local_ppo_loss, token_digest
from verl.utils.swe_tool_repeat import build_cost_weights, capture_events, detect, diagnostics, verified_span


def session(tmp_path, monkeypatch):
    s = ExecutionSession("offline", {"task_id": "offline", "sandbox_backend": "local"}, "offline", tmp_path, False)
    s.workspace = tmp_path
    monkeypatch.setattr(s, "prepare", lambda **kw: "")
    monkeypatch.setattr(s, "_clear_oom_events", lambda: None)
    return s


def event(i, name, args, text, metadata):
    return dict(id=i, name=name, arguments=args, serial=True, feedback_delivered=True,
                **rr.observation_fields(text, metadata), result=rr.public_result(text, metadata),
                contract_observation_complete=not metadata.get("output_artifact_path") and not metadata.get("output_truncated"),
                infrastructure_failure=bool(metadata.get("infrastructure_failure")),
                edit_outcome=metadata.get("edit_outcome"), evidence=metadata.get("repeat_read_evidence"),
                repeat_shell_read_evidence=metadata.get("repeat_shell_read_evidence"),
                command_recovery_evidence=metadata.get("command_recovery_evidence"),
                tool_feedback_kind=metadata.get("tool_feedback_kind"))


def dispatch(s, events, name="Edit", args=None, **kwargs):
    args = args or dict(file_path="a.py", old_string="one", new_string="one")
    text, reward, metadata = s.run_tool(name, args, collect_edit_recovery=True, **kwargs)
    e = event(len(events) + 1, name, args, text, metadata)
    events.append(e)
    assert reward == 0.0
    return e


def ready(s, tmp_path):
    (tmp_path / "a.py").write_text("one\none\n")
    s.run_tool("Read", {"file_path": "a.py"})


@pytest.mark.parametrize("family,args,prior", [
    ("identical_replacement", dict(old_string="one", new_string="one", replace_all=True), True),
    ("old_string_missing", dict(old_string="missing", new_string="two"), True),
    ("multiple_matches", dict(old_string="one", new_string="two"), True),
    ("prior_read_required", dict(old_string="one", new_string="two"), False),
    ("file_missing", dict(old_string="one", new_string="two", file_path="missing.py"), True),
    ("applied", dict(old_string="one", new_string="two", replace_all=True), True),
])
def test_authoritative_outcomes_and_original_argument_binding(tmp_path, monkeypatch, family, args, prior):
    s = session(tmp_path, monkeypatch)
    (tmp_path / "a.py").write_text("one\none\n")
    if prior:
        s.run_tool("Read", {"file_path": "a.py"})
    args = dict(file_path="a.py", **args) if "file_path" not in args else args
    e = dispatch(s, [], args=args)
    p = verified_edit_outcome(e)
    assert p and p["family"] == family
    assert p["arguments_sha256"] == rr.digest(args)  # no normalized `path` alias
    assert p["before"] == p["after"] if family != "applied" else p["before"] != p["after"]


def test_identical_replacement_requires_uniqueness_and_preserves_rejection_semantics(tmp_path, monkeypatch):
    s = session(tmp_path, monkeypatch)
    ready(s, tmp_path)
    e = dispatch(s, [])
    assert e["edit_outcome"]["family"] == "multiple_matches"
    assert (tmp_path / "a.py").read_text() == "one\none\n"
    assert "replace_all=true" in e["result"]


@pytest.mark.parametrize("family", ["identical_replacement", "old_string_missing", "multiple_matches", "file_missing"])
def test_third_call_across_authenticated_reads_selected_once(tmp_path, monkeypatch, family):
    s = session(tmp_path, monkeypatch)
    ready(s, tmp_path)
    args = dict(file_path="a.py", old_string="one", new_string="one", replace_all=True)
    if family == "old_string_missing":
        args.update(old_string="missing", new_string="two")
    elif family == "multiple_matches":
        args.update(new_string="two", replace_all=False)
    elif family == "file_missing":
        args.update(file_path="missing.py", new_string="two")
    events = []
    for _ in range(4):
        dispatch(s, events, args=args)
        dispatch(s, events, "Read", {"file_path": "a.py"})
    expected = {5, 7}
    assert set(detect(events, EDIT_RULES)) == expected
    assert detect(events[:4], EDIT_RULES) == {}
    assert detect(events, set()) == {}
    assert diagnostics(events, edit_shadow=True)["tool_repeat_shadow_edit_selected_" +
        (EDIT_RULES[0] if family == "identical_replacement" else EDIT_RULES[1])] == 2


@pytest.mark.parametrize("barrier", ["changed_args", "applied", "external_change", "missing_proof", "state",
                                    "undelivered", "infrastructure", "multicall", "id_gap", "unknown_shell",
                                    "observation", "argument_digest", "window", "prior_read"])
def test_recovery_and_uncertain_execution_protect_retries(tmp_path, monkeypatch, barrier):
    s = session(tmp_path, monkeypatch)
    ready(s, tmp_path)
    events = []
    args = dict(file_path="a.py", old_string="one", new_string="one", replace_all=True)
    dispatch(s, events, args=args)
    dispatch(s, events, args=args)
    if barrier == "changed_args":
        args = dict(args, old_string="one\n", new_string="one\n")
    elif barrier == "applied":
        dispatch(s, events, args=dict(args, new_string="two"))
        args = dict(args, old_string="two", new_string="two")
    elif barrier == "external_change":
        (tmp_path / "a.py").write_text("one\none\nchanged\n")
    elif barrier == "window":
        for _ in range(15):
            dispatch(s, events, "Read", {"file_path": "a.py"})
    elif barrier == "unknown_shell":
        # A saved tool event with no mutation proof is a barrier; do not execute.
        events.append(dict(id=3, name="Bash", arguments={"command": "unknown"}, result="ok",
                           serial=True, feedback_delivered=True, completed=True, observation_status="observed"))
    elif barrier == "prior_read":
        s._swe_prior_read_paths = set()
    dispatch(s, events, args=args)
    if barrier == "missing_proof":
        events[1]["edit_outcome"] = None
    elif barrier == "state":
        events[1]["edit_outcome"]["after"] = "f" * 64
    elif barrier == "undelivered":
        events[1]["feedback_delivered"] = False
    elif barrier == "infrastructure":
        events[1]["infrastructure_failure"] = True
    elif barrier == "multicall":
        events[1]["serial"] = False
    elif barrier == "id_gap":
        events[-1]["id"] += 1
    elif barrier == "observation":
        events[1]["result"] += "altered"
    elif barrier == "argument_digest":
        events[1]["arguments"]["new_string"] = "altered"
    assert detect(events, EDIT_RULES) == {}


def test_varied_failures_are_shadow_only_and_read_resolves_prior_requirement(tmp_path, monkeypatch):
    s = session(tmp_path, monkeypatch)
    (tmp_path / "a.py").write_text("one\n")
    events = []
    for _ in range(3):
        dispatch(s, events)
    assert detect(events, EDIT_RULES) == {}  # prior_read_required is advisory
    dispatch(s, events, "Read", {"file_path": "a.py"})
    for text in ("bad one", "bad two", "bad three"):
        dispatch(s, events, args=dict(file_path="a.py", old_string=text, new_string="two"))
    assert detect(events, EDIT_RULES) == {}
    assert diagnostics(events, edit_shadow=True)["tool_repeat_shadow_edit_varied_failed_edit_candidates"] == 1


@pytest.mark.parametrize("remote", [False, True])
def test_public_tool_wrapper_local_and_remote_metadata_parity(tmp_path, monkeypatch, remote):
    s = session(tmp_path, monkeypatch)
    (tmp_path / "a.py").write_text("one\n")
    tool = object.__new__(tools.SWETool)
    tool.config, tool.name = {}, "Edit"
    monkeypatch.setattr(tool, "_remote_enabled", lambda: remote)
    monkeypatch.setattr(tools, "_task_from_agent_data", lambda data: {"task_id": "offline", "sandbox_backend": "local"})
    monkeypatch.setattr(tools, "_ensure_workspace", lambda *a: tmp_path)

    async def post(path, payload, config):
        assert payload["collect_edit_recovery"] is True
        text, reward, metadata = s.run_tool(payload["tool"], payload["parameters"], collect_edit_recovery=True)
        return dict(ok=True, text=text, reward=reward, metadata=metadata)

    monkeypatch.setattr(tools, "_remote_post_json_async", post)
    args = dict(file_path="a.py", old_string="one", new_string="one")
    data = SimpleNamespace(request_id="offline", extra_fields={"_swe_prior_read_paths": {(tmp_path / "a.py").as_posix()}},
                           _swe_edit_recovery_enabled=True)
    s.run_tool("Read", {"file_path": "a.py"})
    response, _, metadata = asyncio.run(tool.execute("offline", args, agent_data=data))
    e = event(1, "Edit", args, response, metadata)
    assert verified_edit_outcome(e)["family"] == "identical_replacement"
    tool.name = "Read"
    read_args = {"file_path": "a.py"}
    response, _, metadata = asyncio.run(tool.execute("offline", read_args, agent_data=data))
    read_event = event(2, "Read", read_args, response, metadata)
    assert rr.proven_read_key(read_event) is not None
    tool.name = "Bash"
    shell_args = {"command": "grep one a.py"}
    response, _, metadata = asyncio.run(tool.execute("offline", shell_args, agent_data=data))
    assert metadata["repeat_shell_read_evidence"]["complete"]


def aligned_repeats(s, tokenizer, count=7):
    from verl.experimental.agent_loop.tool_parser import Qwen3XMLToolParser

    parser = Qwen3XMLToolParser(tokenizer)
    response, mask, events = [], [], []
    xml = "<think>Inspect and repair.</think><tool_call>\n<function=Edit>\n<parameter=file_path>a.py</parameter>\n<parameter=old_string>\none\n</parameter>\n<parameter=new_string>\none\n</parameter>\n</function>\n</tool_call>"
    ids = tokenizer.encode(xml, add_special_tokens=False)
    for _ in range(count):
        _, calls = asyncio.run(parser.extract_tool_calls(ids, []))
        response.extend(ids)
        mask.extend([1] * len(ids))
        data = SimpleNamespace(tool_calls=calls, response_ids=ids, response_mask=mask,
                               prompt_ids=[99] + response)
        span = asyncio.run(align_single_call(tokenizer, parser, data, []))
        assert span
        e = dispatch(s, events, args=json.loads(calls[0].arguments))
        e["token_span"] = span
        observation = tokenizer.encode("\n<tool_response>delivered</tool_response>\n", add_special_tokens=False)
        response.extend(observation)
        mask.extend([0] * len(observation))
    return response, mask, events


def test_real_parser_alignment_default_off_cost_dedup_cap_and_zero_advantage(tmp_path, monkeypatch, tokenizer):
    s = session(tmp_path, monkeypatch)
    (tmp_path / "a.py").write_text("one\n")
    s.run_tool("Read", {"file_path": "a.py"})
    ids, mask, events = aligned_repeats(s, tokenizer)
    captured, selected = capture_events(events, EDIT_RULES)
    assert selected == len(captured) == 5
    assert all(verified_span(e, ids, mask) for e in captured)
    assert all("Inspect and repair" not in tokenizer.decode(ids[e["start"]:e["end"]]) for e in captured)
    kwargs = dict(cost=.02, cap=.10, warmup_steps=0, step=1, edit_cost=.01)
    args = (torch.tensor([ids]), torch.tensor([mask]), [{"tool_repeat_events": captured}], torch.tensor([True]))
    w, stats = build_cost_weights(*args, enabled_rules=EDIT_RULES, **kwargs)
    assert w.sum() == pytest.approx(.05)
    assert stats["applied"] == 5
    disabled, _ = build_cost_weights(*args, enabled_rules=set(), **kwargs)
    assert not disabled.any()
    existing = torch.zeros_like(w)
    existing[0, captured[0]["start"]:captured[0]["end"]] = .08 / (captured[0]["end"] - captured[0]["start"])
    capped, stats = build_cost_weights(*args, enabled_rules=EDIT_RULES, existing_weights=existing, **kwargs)
    assert (capped + existing).sum() == pytest.approx(.10)
    assert not capped[existing > 0].any()
    # Homogeneous GRPO gives zero outcome advantages. This independent loss
    # still decreases the selected tool-call probabilities by gradient descent.
    logp = torch.zeros_like(w, requires_grad=True)
    loss = local_ppo_loss(logp, logp.detach(), w, torch.tensor([mask]), CONFIG, active_global_count=1)
    loss.backward()
    assert logp.grad[w > 0].gt(0).all()
    assert logp.grad[w == 0].eq(0).all()
    config = dict(tool_repeat_penalty_enabled=True, tool_repeat_penalty_edit_noop_episodes=True,
                  tool_repeat_penalty_edit_cost=.01)
    output = SimpleNamespace(response_ids=ids, response_mask=mask, extra_fields={"tool_repeat_events": captured})
    assert verified_local_penalties(output, config) == {"tool_repeat": 5}
    assert verified_local_penalties(output, dict(config, tool_repeat_penalty_edit_cost=0)) == {}
    assert verified_local_penalties(output, {"tool_repeat_edit_shadow_enabled": True}) == {}
    output.response_ids = list(ids)
    output.response_ids[captured[0]["start"]] += 1
    assert verified_span(captured[0], output.response_ids, mask) is None


def test_command_shadow_records_raw_result_and_never_creates_cost_event(tmp_path, monkeypatch):
    s = session(tmp_path, monkeypatch)
    events = []
    for _ in range(10):
        text, _, meta = s.run_tool("Bash", {"command": "/bin/false"}, collect_command_recovery=True)
        proof = meta["command_recovery_evidence"]
        assert proof["raw_results"][0]["exit_code"] == 1
        assert proof["raw_results"][0]["output_sha256"] == hashlib.sha256(b"").hexdigest()
        events.append(event(len(events) + 1, "Bash", {"command": "/bin/false"}, text, meta))
    stats = diagnostics(events, command_shadow=True)
    assert stats["tool_repeat_shadow_command_repeat_3_candidates"] == 1
    assert stats["tool_repeat_shadow_command_repeat_5_candidates"] == 1
    assert stats["tool_repeat_shadow_command_repeat_10_candidates"] == 1
    assert capture_events(events, EDIT_RULES) == ([], 0)


def test_default_off_executor_does_not_emit_edit_proof(tmp_path, monkeypatch):
    s = session(tmp_path, monkeypatch)
    ready(s, tmp_path)
    _, _, metadata = s.run_tool("Edit", dict(file_path="a.py", old_string="one", new_string="two"))
    assert "edit_outcome" not in metadata


def test_shadow_keeps_public_feedback_identical_and_guidance_has_separate_switch(tmp_path, monkeypatch):
    args = dict(file_path="a.py", old_string="missing", new_string="two")
    s = session(tmp_path, monkeypatch)
    ready(s, tmp_path)
    text, _, _ = s.run_tool("Edit", args)
    s.tool_feedback_state = tools.ToolFeedbackState()
    shadow, _, metadata = s.run_tool("Edit", args, collect_edit_recovery=True)
    assert shadow == text and metadata["edit_outcome"]["family"] == "old_string_missing"
    guided, _, _ = s.run_tool("Edit", args, collect_edit_recovery=True, edit_recovery_feedback=True)
    assert "copy the exact old_string" in guided


def test_mixed_rule_cost_precedence_deduplication_and_warmup(tmp_path, monkeypatch):
    s = session(tmp_path, monkeypatch)
    (tmp_path / "a.py").write_text("one\n")
    s.run_tool("Read", {"file_path": "a.py"})
    events = []
    for _ in range(5):
        e = dispatch(s, events, collect_contract_rejections=True)
        # Old rule also uses executor no-op proof and raw-result evidence.
        e.update(rr.noop_edit_evidence("Edit", e["arguments"], e["result"]))
        e["tool_repeat_noop_edit"] = dict(version=1, target="a.py", rejected=True)
    ids = list(range(30))
    for e in events:
        start = e["id"] * 2
        e["token_span"] = dict(version=1, start=start, end=start+1,
                                token_sha256=token_digest(ids[start:start+1]), prefix_sha256=token_digest(ids[:start+1]))
    captured, _ = capture_events(events, {*EDIT_RULES, "noop_edits"})
    assert all(set(e["proof"]["rules"]) == {EDIT_RULES[0], "noop_edits"} for e in captured)
    args = (torch.tensor([ids]), torch.ones(1, len(ids)), [{"tool_repeat_events": captured}], torch.tensor([True]))
    kwargs = dict(cost=.02, edit_cost=.01, cap=.10, warmup_steps=10, step=5)
    w, counts = build_cost_weights(*args, enabled_rules={*EDIT_RULES, "noop_edits"}, **kwargs)
    assert counts["applied"] == 3 and w.sum() == pytest.approx(.015)
    old, _ = build_cost_weights(*args, enabled_rules={"noop_edits"}, **kwargs)
    assert old.sum() == pytest.approx(.03)


def test_actor_new_flags_default_off_and_invalid_cost_rejected():
    from dataclasses import replace
    from tests.trainer.ppo.test_swe_tool_failure_on_cpu import actor_config

    config = actor_config()
    assert not config.tool_repeat_penalty_edit_noop_episodes
    assert not config.tool_repeat_penalty_edit_rejection_episodes
    assert not config.tool_repeat_edit_shadow_enabled
    assert not config.tool_repeat_edit_feedback_enabled
    assert config.tool_repeat_penalty_edit_cost == .01
    for cost in (-1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            replace(config, tool_repeat_penalty_enabled=True, tool_repeat_penalty_edit_cost=cost)


@pytest.mark.parametrize("scores", [[0., 0.], [0., 1.], [-.1, -.1]])
def test_actual_v1_new_rule_cost_and_terminal_outcome_preservation(tmp_path, monkeypatch, scores):
    from tests.trainer.ppo.test_swe_tool_failure_on_cpu import run_v1

    s = session(tmp_path, monkeypatch)
    (tmp_path / "a.py").write_text("one\n")
    s.run_tool("Read", {"file_path": "a.py"})
    events = []
    for _ in range(3):
        dispatch(s, events)
    ids = [10, 11, 12, 13]
    events[-1]["token_span"] = dict(version=1, start=1, end=3,
                                    token_sha256=token_digest(ids[1:3]), prefix_sha256=token_digest(ids[:3]))
    captured, _ = capture_events(events, EDIT_RULES)
    extras = [{"tool_repeat_events": captured}] * 2
    baseline, _, _ = run_v1(monkeypatch, False, scores, extras=extras)
    result, _, metrics = run_v1(monkeypatch, False, scores, extras=extras, tool_repeat=True, edit_episodes=True)
    assert metrics["tool_repeat/applied"] == 2
    for row in range(2):
        w = result["tool_repeat_weights"].unbind()[row]
        assert w.sum() == pytest.approx(.01)
        assert w[[0, 3]].eq(0).all()
        expected = baseline["advantages"].unbind()[row].clone()
        expected[1:3] = expected[1:3].clamp_max(0)
        torch.testing.assert_close(result["advantages"].unbind()[row], expected)
        torch.testing.assert_close(result["returns"].unbind()[row], baseline["returns"].unbind()[row])


def test_shadow_metrics_export_without_optimization_or_judge_shortcut(monkeypatch):
    from tests.trainer.ppo.test_swe_tool_failure_on_cpu import run_v1
    from verl.trainer.ppo.v1.replay_buffer import rollout_diagnostics

    extra = {"tool_repeat_shadow_edit_selected_edit_noop_episodes": 4,
             "tool_repeat_shadow_command_repeat_3_candidates": 2}
    result, requested, metrics = run_v1(monkeypatch, False, [0., 0.], extras=[extra] * 2, edit_shadow=True)
    assert "responses" not in requested[0]
    assert "tool_repeat_weights" not in result
    assert metrics["tool_repeat/shadow_edit_selected_edit_noop_episodes"] == 8
    assert rollout_diagnostics(extra)["tool_repeat_shadow_command_repeat_3_candidates"] == 2


def test_compact_proof_audit_survives_dump_and_stays_out_of_reward_metrics(tmp_path, monkeypatch):
    from recipe.swe_agent.edit_recovery import audit_json
    from verl.trainer.ppo.v1.replay_buffer import rollout_diagnostics
    from verl.trainer.ppo.v1.utils import validation_reward_metadata

    s = session(tmp_path, monkeypatch)
    ready(s, tmp_path)
    events = []
    for _ in range(10):
        dispatch(s, events, args=dict(file_path="a.py", old_string="missing", new_string="two"))
    audit = audit_json(events)
    assert len(audit["tool_repeat_recovery_audit_json"]) > 2048
    dumped = rollout_diagnostics(audit)
    assert dumped["tool_repeat_recovery_audit_json"] == audit["tool_repeat_recovery_audit_json"]
    decoded = json.loads(dumped["tool_repeat_recovery_audit_json"])
    assert decoded["events"][0]["edit_outcome"]["family"] == "old_string_missing"
    assert "arguments" not in decoded["events"][0] and "result" not in decoded["events"][0]
    assert "tool_repeat_recovery_audit_json" not in validation_reward_metadata(audit)
    assert audit_json(events, max_bytes=10)["tool_repeat_recovery_audit_error"] == "size_limit"


@pytest.mark.parametrize("active", [True, False])
def test_native_agent_loop_transports_proofs_and_delivery(tmp_path, monkeypatch, tokenizer, active):
    from recipe.swe_agent.agent_loop import SWEToolAgentLoop
    from verl.experimental.agent_loop.tool_agent_loop import AgentState, ToolAgentLoop
    from verl.experimental.agent_loop.tool_parser import Qwen3XMLToolParser

    s = session(tmp_path, monkeypatch)
    (tmp_path / "a.py").write_text("one\n")
    s.run_tool("Read", {"file_path": "a.py"})
    loop = object.__new__(SWEToolAgentLoop)
    loop.response_length, loop.enable_continuous_token = 100000, False
    loop.max_assistant_turns = loop.max_user_turns = None
    loop._swe_tool_repeat_enabled = active
    loop._swe_tool_repeat_rules = set(EDIT_RULES) if active else set()
    loop._swe_tool_failure_enabled = False
    loop._swe_edit_recovery_enabled = True
    loop._swe_is_validation = False
    loop._swe_repeat_config = rr.RepeatConfig(mode="off")
    loop._swe_repeat_events, loop._swe_repeat_active = [], []
    loop._trajectory_expired = lambda: False
    loop.tools = {}
    loop.tokenizer, loop.tool_parser = tokenizer, Qwen3XMLToolParser(tokenizer)
    xml = "<think>Thought.</think><tool_call><function=Edit><parameter=file_path>a.py</parameter><parameter=old_string>one</parameter><parameter=new_string>one</parameter></function></tool_call>"
    ids = tokenizer.encode(xml, add_special_tokens=False)

    async def generate(self, data, params):
        return SimpleNamespace(token_ids=ids, log_probs=[], extra_fields={}, routed_experts=None, num_preempted=0)

    async def deadline(coro):
        return await coro

    async def dispatch_call(call, kwargs, data):
        assert data._swe_edit_recovery_enabled
        return s.run_tool(call.name, json.loads(call.arguments), collect_edit_recovery=True)

    loop._await_with_trajectory_deadline = deadline
    loop._call_tool_with_legacy_tracking = dispatch_call
    monkeypatch.setattr(ToolAgentLoop, "_generate_model_tokens", generate)
    data = SimpleNamespace(response_mask=[], prompt_ids=[1], assistant_turns=0, user_turns=0,
                           tool_calls=[], metrics={}, extra_fields={}, response_logprobs=[], request_id="offline")

    async def run():
        for _ in range(4):
            assert await loop._handle_generating_state(data, {}) == AgentState.PROCESSING_TOOLS
            await loop._call_tool(data.tool_calls[0], {}, data)
            data.prompt_ids += [1]
            data.response_mask += [0]
            data.user_turns += 1
        assert loop._swe_repeat_events[-1]["feedback_delivered"] is False
        assert detect(loop._swe_repeat_events, EDIT_RULES) == {3: [EDIT_RULES[0]]}
        captured, count = capture_events(loop._swe_repeat_events, loop._swe_tool_repeat_rules)
        assert count == len(captured) == int(active)
        if active:
            assert verified_span(captured[0], data.prompt_ids[-len(data.response_mask):], data.response_mask)

    asyncio.run(run())

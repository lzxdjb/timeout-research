"""Audited loops, false-positive controls, native capture, and real PPO wiring."""

import asyncio
import gzip
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from recipe.swe_agent import repetition_reward as rr
from tests.trainer.ppo.test_swe_tool_failure_on_cpu import actor_config, actor_data, run_v1
from verl.utils import tensordict_utils as tu
from verl.utils.swe_tool_failure import gate_positive_credit, rejection_metadata, token_digest
from verl.utils.swe_tool_repeat import RULES, build_cost_weights, capture_events, detect, verified_span

pytest_plugins = ("tests.trainer.ppo.test_swe_tool_failure_on_cpu",)


def noop(i, *, target="a.py", text="hello", authenticated=True):
    args = dict(file_path=target, old_string=text, new_string=text)
    result = rr.NOOP_EDIT_RESULT
    e = dict(
        id=i,
        name="Edit",
        arguments=args,
        result=result,
        completed=True,
        serial=True,
        observation_status="observed",
        feedback_delivered=True,
    )
    e.update(rr.noop_edit_evidence("Edit", args, result))
    if authenticated:
        e["tool_repeat_noop_edit"] = dict(version=1, target=target, rejected=True)
    return e


def read(i, *, pattern="x", state="scope", tool="Grep"):
    result = "found line " + pattern
    return dict(
        id=i,
        name=tool,
        arguments={"pattern": pattern},
        result=result,
        completed=True,
        serial=True,
        observation_status="observed",
        feedback_delivered=True,
        evidence=dict(
            version=rr.EVIDENCE_VERSION,
            complete=True,
            before=state,
            after=state,
            arguments_sha256=rr.digest({"pattern": pattern}),
            observation_sha256=hashlib.sha256(result.encode()).hexdigest(),
        ),
    )


def rejection(i, code="invalid_argument"):
    return dict(
        id=i,
        name="Bash",
        arguments={"command": ""},
        serial=True,
        feedback_delivered=True,
        completed=False,
        observation_status="transport_error",
        result="No command provided.",
        contract_rejection=rejection_metadata(code)["tool_contract_rejection"],
        contract_observation_complete=True,
    )


def aligned_events(ids, count=5):
    events = [noop(i) for i in range(1, count + 1)]
    for e in events:
        e["token_span"] = dict(
            version=1, start=1, end=3, token_sha256=token_digest(ids[1:3]), prefix_sha256=token_digest(ids[:3])
        )
    return events


def local_event(ids):
    return capture_events(aligned_events(ids), {"noop_edits"})[0][0]


@pytest.mark.parametrize(
    "rule,maker", [("noop_edits", noop), ("unchanged_reads", read), ("contract_rejections", rejection)]
)
def test_five_call_threshold_and_first_two_protected(rule, maker):
    assert detect([maker(i) for i in range(1, 5)], {rule}) == {}
    assert set(detect([maker(i) for i in range(1, 8)], {rule})) == set(range(3, 8))
    assert detect([maker(i) for i in range(1, 8)], set()) == {}


def test_varied_noops_require_executor_target_evidence():
    events = [noop(i, text=f"line {i}") for i in range(1, 7)]
    assert detect(events, {"noop_edits"}) == {}
    assert set(detect(events, {"varied_noop_edits"})) == {3, 4, 5, 6}
    for e in events:
        e.pop("tool_repeat_noop_edit")
    assert detect(events, {"varied_noop_edits"}) == {}


@pytest.mark.parametrize(
    "mutation", ["serial", "delivery", "arguments", "state", "output", "proof", "intervening_mutation", "ids"]
)
def test_legitimate_retry_and_missing_proof_barriers(mutation):
    events = [read(i) for i in range(1, 6)]
    e = events[2]
    if mutation == "serial":
        e["serial"] = False
    elif mutation == "delivery":
        e["feedback_delivered"] = False
    elif mutation == "arguments":
        e["arguments"]["pattern"] = "different"
    elif mutation == "state":
        e["evidence"]["after"] = "changed"
    elif mutation == "output":
        e["result"] = "different"
    elif mutation == "proof":
        e.pop("evidence")
    elif mutation == "intervening_mutation":
        events[2] = noop(3)
    else:
        e["id"] = 99
    assert detect(events, RULES) == {}


@pytest.mark.parametrize(
    "mutation", ["no_proof", "text_only", "infra", "incomplete", "undelivered", "changed_args", "changed_code"]
)
def test_rejection_proof_and_delivery_are_mandatory(mutation):
    events = [rejection(i) for i in range(1, 6)]
    e = events[2]
    if mutation in {"no_proof", "text_only"}:
        e.pop("contract_rejection")
    elif mutation == "infra":
        e["infrastructure_failure"] = True
    elif mutation == "incomplete":
        e["contract_observation_complete"] = False
    elif mutation == "undelivered":
        e["feedback_delivered"] = False
    elif mutation == "changed_args":
        e["arguments"] = {"command": "echo fixed"}
    else:
        e["contract_rejection"]["code"] = "path_rejected"
    assert detect(events, RULES) == {}


@pytest.mark.parametrize("period", [2, 3, 4])
def test_read_cycles_and_no_mutation_barrier(period):
    size = ((max(20, period * 5) + period - 1) // period) * period
    events = [read(i + 1, pattern=str(i % period)) for i in range(size)]
    assert detect(events, {"read_cycles"}) == {i: ["read_cycles"] for i in range(2 * period + 1, size + 1)}
    assert detect(events[:-period], {"read_cycles"}) == {}
    events[size // 2]["evidence"]["after"] = "changed"
    assert detect(events, {"read_cycles"}) == {}


@pytest.mark.parametrize("tool", ["Bash", "Write"])
def test_identical_success_or_failure_is_not_shell_purity_proof(tool):
    events = [
        dict(
            id=i,
            name=tool,
            arguments={"command": "pytest"},
            serial=True,
            completed=True,
            feedback_delivered=True,
            result="exit_code: 0\npassed",
            observation_status="observed",
        )
        for i in range(1, 301)
    ]
    assert detect(events, RULES) == {}


def test_frozen_audited_cases():
    path = Path(__file__).parent / "fixtures/swe_tool_repeat_audited.json.gz"
    cases = json.load(gzip.open(path, "rt"))
    positives = controls = 0
    for case in cases:
        matches = detect(case["events"], RULES)
        if case["label"] == "existing_tool_positive":
            assert matches, case["id"]
            positives += 1
        else:
            assert not matches, case["id"]
            controls += case["label"] == "legitimate_control"
    assert positives == 6 and controls >= 4


@pytest.mark.parametrize("tamper", ["tokens", "prefix", "mask", "truncated", "proof", "span", "version", "non_dict"])
def test_selected_span_validation_abstains(tamper):
    ids = [10, 11, 12, 13]
    mask = [1] * 4
    e = local_event(ids)
    if tamper == "tokens":
        ids[1] = 99
    elif tamper == "prefix":
        ids[0] = 99
    elif tamper == "mask":
        mask[1] = 0
    elif tamper == "truncated":
        ids = ids[:2]
    elif tamper == "proof":
        e["proof"]["rules"] = ["contract_rejections"]
    elif tamper == "span":
        e["end"] = 4
    elif tamper == "version":
        e["version"] = 2
    else:
        e["proof"] = ["bad"]
    assert verified_span(e, ids, mask) is None


def test_cost_warmup_cap_duplicate_prefixes_and_contract_union():
    ids = [10, 11, 12, 13]
    e = local_event(ids)
    kwargs = dict(cost=0.02, cap=0.1, warmup_steps=10, step=5, trajectory_ids=["session", "session"])
    responses = torch.tensor([ids, ids])
    mask = torch.ones_like(responses)
    active = torch.tensor([True, True])
    w, stats = build_cost_weights(responses, mask, [{"tool_repeat_events": [e, e]}] * 2, active, **kwargs)
    assert w.sum() == pytest.approx(0.01) and stats["applied"] == 1
    # Contract penalty already occupies this call: do not charge it again.
    prior = torch.zeros_like(w)
    prior[0, 1:3] = 0.005
    w, stats = build_cost_weights(
        responses, mask, [{"tool_repeat_events": [e]}] * 2, active, existing_weights=prior, **kwargs
    )
    # The same prefix in another output must also be deduplicated.
    assert w.sum() == 0


@pytest.mark.parametrize("score", [-0.1, 0.0, 1.0])
def test_v1_two_layers_independent_of_terminal_reward(monkeypatch, score):
    extras = [{"tool_repeat_events": [local_event([10, 11, 12, 13])]}] * 2
    baseline, _, _ = run_v1(monkeypatch, False, [score, 1.0], extras=extras)
    enabled, _, metrics = run_v1(monkeypatch, False, [score, 1.0], extras=extras, tool_repeat=True)
    assert metrics["tool_repeat/applied"] == 2
    for i in range(2):
        assert enabled["tool_repeat_weights"].unbind()[i].sum() == pytest.approx(0.02)
        torch.testing.assert_close(
            enabled["advantages"].unbind()[i][[0, 3]], baseline["advantages"].unbind()[i][[0, 3]]
        )
        torch.testing.assert_close(
            enabled["advantages"].unbind()[i][1:3], baseline["advantages"].unbind()[i][1:3].clamp_max(0)
        )
        torch.testing.assert_close(enabled["returns"].unbind()[i], baseline["returns"].unbind()[i])


@pytest.mark.parametrize(
    "tags", [[{"is_padding": True}, {}], [{"train_sample_mask": False}, {}], [{"completion_ratio_cutoff": True}, {}]]
)
def test_v1_exclusions(monkeypatch, tags):
    output, _, _ = run_v1(
        monkeypatch,
        False,
        [0.0, 1.0],
        extras=[{"tool_repeat_events": [local_event([10, 11, 12, 13])]}] * 2,
        tags=tags,
        tool_repeat=True,
    )
    assert output["tool_repeat_weights"].unbind()[0].eq(0).all()


def test_actor_disabled_parity_and_local_penalty_gradient():
    from verl.workers.utils.losses import ppo_loss

    data = actor_data()
    x = torch.zeros(12, requires_grad=True)
    base, _ = ppo_loss(actor_config(), {"log_probs": x}, data)
    base.backward()
    grad = x.grad.clone()
    data["tool_repeat_weights"] = torch.tensor([[0.0, 0.02, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    tu.assign_non_tensor(data, tool_repeat_active_global_count=2)
    x.grad = None
    disabled, _ = ppo_loss(actor_config(), {"log_probs": x}, data)
    disabled.backward()
    torch.testing.assert_close(base, disabled, rtol=0, atol=0)
    torch.testing.assert_close(grad, x.grad, rtol=0, atol=0)
    x.grad = None
    enabled, metrics = ppo_loss(replace(actor_config(), tool_repeat_penalty_enabled=True), {"log_probs": x}, data)
    enabled.backward()
    assert (enabled - base).item() == pytest.approx(0.01, abs=1e-7)
    assert (x.grad - grad).sum().item() == pytest.approx(0.01, abs=1e-7)
    assert "actor/tool_repeat_loss" in metrics


def test_real_noop_and_empty_command_proofs(tmp_path):
    from recipe.swe_agent.remote_execution_service import ExecutionSession
    from recipe.swe_agent.tool_outcomes import collect_noop_edits, collect_rejections, noop_metadata, outcome_metadata

    from recipe.swe_agent import tools

    path = tmp_path / "a.py"
    path.write_text("hello world\n")
    data = SimpleNamespace(extra_fields={"_swe_prior_read_paths": {path.as_posix()}})
    events = []
    for i, text in enumerate(["hello", "world", "hello world", "hello", "world"], 1):
        args = dict(path="a.py", old_string=text, new_string=text)
        with collect_noop_edits(True) as proofs:
            result = tools._edit_file(tmp_path, args, agent_data=data)
        e = noop(i, text=text, authenticated=False)
        assert e["result"] == result
        e.update(noop_metadata(proofs))
        events.append(e)
    assert set(detect(events, {"varied_noop_edits"})) == {3, 4, 5}
    for invoke in [
        lambda: tools._run_command(tmp_path, "", {}),
        lambda: ExecutionSession._run_command(SimpleNamespace(), ""),
    ]:
        with collect_rejections(True) as proofs:
            assert invoke() == "No command provided."
        assert outcome_metadata(proofs)["tool_contract_rejection"]["code"] == "invalid_argument"


def test_runtime_env_and_default_off(monkeypatch):
    from verl.trainer.constants_ppo import get_ppo_ray_runtime_env

    assert not actor_config().tool_repeat_penalty_enabled
    assert not any(getattr(actor_config(), "tool_repeat_penalty_" + r) for r in RULES)
    key = "SWE_AGENT_TRAINING_TOOL_REPEAT_PENALTY_ENABLED"
    monkeypatch.setenv(key, "1")
    config = OmegaConf.create({"actor_rollout_ref": {"actor": {"tool_repeat_penalty_enabled": False}}})
    assert get_ppo_ray_runtime_env(config)["env_vars"][key] == "0"
    config.actor_rollout_ref.actor.tool_repeat_penalty_enabled = True
    assert get_ppo_ray_runtime_env(config)["env_vars"][key] == "1"


@pytest.mark.parametrize("enabled", [False, True])
def test_native_generation_dispatch_capture_without_rm(monkeypatch, tokenizer, enabled):
    from recipe.swe_agent.agent_loop import SWEToolAgentLoop

    from verl.experimental.agent_loop.tool_agent_loop import AgentState, ToolAgentLoop
    from verl.experimental.agent_loop.tool_parser import Qwen3XMLToolParser

    loop = object.__new__(SWEToolAgentLoop)
    loop.response_length = 100000
    loop.enable_continuous_token = False
    loop.max_assistant_turns = loop.max_user_turns = None
    loop._swe_tool_repeat_enabled = enabled
    loop._swe_tool_failure_enabled = False
    loop._swe_is_validation = False
    loop._swe_repeat_config = rr.RepeatConfig(mode="off")
    loop._swe_repeat_events = []
    loop._swe_repeat_active = []
    loop._trajectory_expired = lambda: False
    loop.tools = {}
    loop.tokenizer = tokenizer
    loop.tool_parser = Qwen3XMLToolParser(tokenizer)
    ids = tokenizer.encode(
        "<think>Try tool.</think><tool_call>\n<function=Unknown>\n</function>\n</tool_call>", add_special_tokens=False
    )

    async def generate(self, data, params):
        return SimpleNamespace(token_ids=ids, log_probs=[], extra_fields={}, routed_experts=None, num_preempted=0)

    async def deadline(coro):
        return await coro

    async def dispatch(call, kwargs, data):
        return await ToolAgentLoop._call_tool(loop, call, kwargs, data)

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

    async def run():
        for _ in range(6):
            assert await loop._handle_generating_state(data, {}) == AgentState.PROCESSING_TOOLS
            await loop._call_tool(data.tool_calls[0], {}, data)
            # Simulate successful observation retention before next generation.
            data.prompt_ids += [1]
            data.response_mask += [0]
            data.user_turns += 1
        if enabled:
            # Last feedback has not yet been consumed by another generation.
            assert loop._swe_repeat_events[-1]["feedback_delivered"] is False
            captured, selected = capture_events(loop._swe_repeat_events, {"contract_rejections"})
            assert selected == 3 and len(captured) == 3
            response = data.prompt_ids[-len(data.response_mask) :]
            assert all(verified_span(e, response, data.response_mask) for e in captured)
            assert all(e["proof"]["rules"] == ["contract_rejections"] for e in captured)
        else:
            assert loop._swe_repeat_events == []
            assert not any(k.startswith("tool_repeat_") for k in data.extra_fields)

    asyncio.run(run())


def test_cap_across_different_calls_and_inactive_rows():
    ids = list(range(30))
    events = [noop(i) for i in range(1, 13)]
    for i, e in enumerate(events):
        start, end = 2 * i, 2 * i + 1
        e["token_span"] = dict(
            version=1,
            start=start,
            end=end,
            token_sha256=token_digest(ids[start:end]),
            prefix_sha256=token_digest(ids[:end]),
        )
    selected, _ = capture_events(events, RULES)
    weights, stats = build_cost_weights(
        torch.tensor([ids, ids]),
        torch.ones(2, 30),
        [{"tool_repeat_events": selected}] * 2,
        torch.tensor([True, False]),
        cost=0.02,
        cap=0.1,
        warmup_steps=10,
        step=5,
    )
    assert weights[0].sum() == pytest.approx(0.05) and weights[1].eq(0).all()
    assert stats["applied"] == 10
    advantages = torch.ones_like(weights)
    gated = gate_positive_credit(advantages, weights)
    assert gated[weights > 0].eq(0).all() and gated[weights == 0].eq(1).all()


def test_real_path_rejection_is_proved_without_text_classification(tmp_path):
    from recipe.swe_agent.path_policy import ToolPathRejected, resolve_model_path
    from recipe.swe_agent.tool_outcomes import collect_rejections, outcome_metadata, record_path_rejection

    events = []
    for i in range(1, 6):
        with collect_rejections(True) as proofs:
            with pytest.raises(ToolPathRejected) as caught:
                resolve_model_path(tmp_path, "/etc/passwd", {})
            record_path_rejection(caught.value)
        e = rejection(i, "path_rejected")
        e["name"] = "Read"
        e["arguments"] = {"file_path": "/etc/passwd"}
        e["contract_rejection"] = outcome_metadata(proofs)["tool_contract_rejection"]
        events.append(e)
    assert set(detect(events, {"contract_rejections"})) == {3, 4, 5}


def test_replay_honors_current_rule_switches_and_exports_diagnostics():
    from verl.trainer.ppo.v1.replay_buffer import rollout_diagnostics

    ids = [10, 11, 12, 13]
    extra = {
        "tool_repeat_events": [local_event(ids)],
        "tool_repeat_selected_calls": 3,
        "tool_repeat_retained_noop_edits": 3,
    }
    weights, counts = build_cost_weights(
        torch.tensor([ids]),
        torch.ones(1, 4),
        [extra],
        torch.tensor([True]),
        cost=0.02,
        cap=0.1,
        warmup_steps=0,
        step=1,
        enabled_rules={"unchanged_reads"},
    )
    assert weights.eq(0).all() and counts["abstained"] == 1
    dumped = rollout_diagnostics(extra)
    assert dumped["tool_repeat_selected_calls"] == dumped["tool_repeat_retained_noop_edits"] == 3
    assert "tool_repeat_events" not in dumped


@pytest.mark.parametrize("field", ["evidence", "noop_edit_evidence", "observation_view"])
def test_malformed_evidence_breaks_streak(field):
    events = [noop(i) for i in range(1, 6)]
    events[2][field] = ["malformed"]
    assert detect(events, RULES) == {}


@pytest.mark.parametrize("enabled", [False, True])
def test_remote_evidence_request_and_roundtrip_without_rm(tmp_path, monkeypatch, enabled):
    import subprocess

    from recipe.swe_agent.remote_execution_service import ExecutionSession

    from recipe.swe_agent import tools

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "a.py").write_text("hello\n")
    task = {"task_id": "offline-repeat", "sandbox_backend": "local"}
    session = ExecutionSession("offline-repeat", task, "offline-repeat", tmp_path, False)
    session.workspace = tmp_path
    monkeypatch.setattr(session, "prepare", lambda **kw: "")
    monkeypatch.setattr(session, "_clear_oom_events", lambda: None)
    monkeypatch.setenv("SWE_AGENT_REPEAT_EVIDENCE", "1")
    monkeypatch.setenv("SWE_AGENT_REPEAT_REWARD_MODE", "off")
    requests = []

    async def post(path, payload, config):
        assert path == "/execute"
        requests.append(payload)
        text, reward, metadata = session.run_tool(
            payload["tool"],
            payload["parameters"],
            collect_contract_rejections=payload.get("collect_tool_contract_rejections", False),
        )
        return {"ok": True, "text": text, "reward": reward, "metadata": metadata}

    monkeypatch.setattr(tools, "_remote_post_json_async", post)
    monkeypatch.setattr(tools, "_task_from_agent_data", lambda data: task)
    monkeypatch.setattr(tools.SWETool, "_remote_enabled", lambda self: True)
    tool = object.__new__(tools.SWETool)
    tool.config = {}
    data = SimpleNamespace(
        request_id="offline-repeat", extra_fields={}, _swe_tool_failure_enabled=False, _swe_tool_repeat_enabled=enabled
    )
    tool.name = "Read"
    response, _, metadata = asyncio.run(tool.execute("offline-repeat", {"file_path": "a.py"}, agent_data=data))
    assert metadata["repeat_read_evidence"]["complete"] is True
    event = dict(
        id=1,
        name="Read",
        arguments={"path": "a.py"},
        serial=True,
        feedback_delivered=True,
        evidence=metadata["repeat_read_evidence"],
        result=response.text,
        completed=True,
        observation_status="observed",
    )
    assert rr.proven_read_key(event) is not None
    tool.name = "Read"
    _, _, metadata = asyncio.run(tool.execute("offline-repeat", {"file_path": "/etc/passwd"}, agent_data=data))
    assert bool(metadata.get("tool_contract_rejection")) is enabled
    assert all(bool(r.get("collect_tool_contract_rejections")) is enabled for r in requests)

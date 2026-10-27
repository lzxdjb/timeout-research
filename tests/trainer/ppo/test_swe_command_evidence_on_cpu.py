"""Audited excerpts + actual CPU executions; historical logs are not forged proofs."""

import asyncio
import json
import re
import shlex
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from recipe.swe_agent import command_evidence as ce, tools
from tests.trainer.ppo.test_swe_edit_recovery_on_cpu import event as edit_event, session
from tests.trainer.ppo.test_swe_python_test_repeat_on_cpu import aligned
from tests.trainer.ppo.test_swe_tool_failure_on_cpu import actor_config, tokenizer, run_v1
from verl.utils.swe_tool_repeat import build_cost_weights, capture_events, detect
from verl.utils.swe_local_penalty_shortcut import verified_local_penalties

CARDS = json.loads((Path(__file__).parent / "fixtures/swe_command_audit_20261027.json").read_text())
RULES = set(ce.RULES)


def card(task, turn):
    return next(c for c in CARDS if c["task"] == task and c["turn"] == turn)


def dispatch(s, events, command=None, *, name="Bash", args=None, feedback=False, collect=True):
    args = args or {"command": command}
    text, reward, metadata = s.run_tool(name, args, collect_command_evidence=collect,
        command_evidence_feedback=feedback, collect_edit_recovery=True)
    e = edit_event(len(events) + 1, name, args, text, metadata)
    e["command_evidence"] = metadata.get("command_evidence")
    events.append(e)
    assert reward == 0.0
    return e


def test_real_ready_loop_threshold_costs_and_flags(tmp_path, monkeypatch):
    s, events = session(tmp_path, monkeypatch), []
    cmd = card("mwaskom__seaborn-3069", 211)["arguments"]["command"]
    for _ in range(15):
        dispatch(s, events, cmd)
    assert detect(events[:2], RULES) == {}
    assert detect(events[:3], RULES) == {3: ["command_no_progress"]}
    assert set(detect(events, RULES)) == set(range(3, 16))
    assert detect(events, set()) == {}
    p = events[0]["command_evidence"]
    assert p["requested_command"] == cmd and p["launches"][0]["argv"] == ["/bin/bash", "-lc", cmd]
    assert p["launches"][0]["cwd"] == str(tmp_path)
    assert p["changed_path_count"] == 0 and p["test"]["test_outcome"] == "unknown"
    ids = aligned(events)
    captured, count = capture_events(events, RULES)
    assert count == 13
    args = (torch.tensor([ids]), torch.ones(1, len(ids)), [{"tool_repeat_events": captured}], torch.tensor([True]))
    kwargs = dict(cost=.02, command_cost=.01, cap=.10, warmup_steps=0, step=1, enabled_rules=RULES)
    weights, stats = build_cost_weights(*args, **kwargs)
    assert stats["applied"] == 13 and weights.sum() == pytest.approx(.10)
    reserved = torch.zeros_like(weights)
    reserved[0, captured[0]["start"]] = .08
    weights, stats = build_cost_weights(*args, existing_weights=reserved, **kwargs)
    assert stats["applied"] == 12 and weights.sum() == pytest.approx(.02)
    assert not weights[reserved > 0].any()
    output = SimpleNamespace(response_ids=ids, response_mask=[1] * len(ids), extra_fields={"tool_repeat_events": captured})
    config = dict(tool_repeat_penalty_enabled=True, tool_repeat_penalty_command_no_progress=True)
    assert verified_local_penalties(output, config) == {"tool_repeat": 13}
    assert verified_local_penalties(output, dict(config, tool_repeat_penalty_command_cost=0)) == {}


def test_audited_timezone_failure_reproduces_without_django_hot_test(tmp_path, monkeypatch):
    # Use the real rejected public-library argument; no synthetic output/proof.
    import pytz
    with pytest.raises(pytz.UnknownTimeZoneError):
        pytz.timezone("Etl/G GMT-10")
    s, events = session(tmp_path, monkeypatch), []
    command = "python -c " + shlex.quote("import pytz\nname1 = 'Etl/G GMT-10'\npytz.timezone(name1)")
    for _ in range(3):
        dispatch(s, events, command)
    assert detect(events, RULES) == {3: ["command_errors"]}
    assert events[2]["command_evidence"]["reason"] == "invalid_timezone_literal"
    # A corrected literal is meaningful progress; a different bad literal is a
    # new hypothesis and receives its own protected attempts.
    dispatch(s, events, command.replace("Etl/G GMT-10", "Etc/GMT-10"))
    dispatch(s, events, command.replace("Etl/G GMT-10", "Ecl/G GMT-10"))
    assert detect(events, RULES) == {3: ["command_errors"]}


@pytest.mark.parametrize("feedback", [False, True])
def test_masked_unittest_loader_error_is_not_success(tmp_path, monkeypatch, feedback):
    s, events = session(tmp_path, monkeypatch), []
    command = "python -B -m unittest missing_public_module 2>&1 | head -200"
    for _ in range(3):
        dispatch(s, events, command, feedback=feedback)
    p = events[2]["command_evidence"]
    assert p["launches"][0]["exit_code"] == 0
    assert p["pipeline_status"] == "unknown" and p["shell_success_without_test_success"]
    assert p["test"]["tests_executed"] == 0 and p["test"]["test_loader_error_count"] > 0
    assert detect(events, RULES) == {3: ["command_errors"]}


@pytest.mark.parametrize("command,reason", [
    ("swe_nonexistent_executable_for_regression", "command_not_found"),
    ("python -m swe_missing_runner_for_regression", "python_module_runner_missing"),
    ("python -c 'if True print(1)'", "python_syntax_error"),
])
def test_confident_repeated_invocation_errors(tmp_path, monkeypatch, command, reason):
    s, events = session(tmp_path, monkeypatch), []
    for _ in range(3):
        dispatch(s, events, command)
    assert events[-1]["command_evidence"]["reason"] == reason
    assert detect(events, RULES) == {3: ["command_errors"]}


def test_audited_heredoc_rewrite_and_failed_tests(tmp_path, monkeypatch):
    s, events = session(tmp_path, monkeypatch), []
    target = tmp_path / "tests/template_tests/filter_tests/test_escapeseq.py"
    target.parent.mkdir(parents=True)
    target.write_text("unchanged\n" * 30)
    # A tiny native unittest runner reproduces the observed assertion failure.
    # The historical Django command and rewrite are otherwise retained, with
    # only /testbed mapped to this isolated workspace.
    (tmp_path / "tests/runtests.py").write_text(
        "import unittest\nclass EscapeTests(unittest.TestCase):\n"
        " def test_escapeseq01(self): self.assertEqual('&amp;', '&amp;')\n"
        " def test_escapeseq02(self): self.assertEqual('&, < -- &amp;, &lt;', '&amp;, &lt; -- &amp;, &lt;')\n"
        "unittest.main(argv=['runtests.py'])\n")
    command = card("django__django-16877", 201)["arguments"]["command"].replace("/testbed", str(tmp_path))
    for _ in range(4):
        dispatch(s, events, command)
    assert events[0]["command_evidence"]["changed_path_count"] > 0
    assert events[1]["command_evidence"]["changed_path_count"] == 0
    assert events[1]["command_evidence"]["metadata_changed_path_count"] > 0
    assert detect(events, RULES) == {4: ["command_no_progress"]}
    # A real change to the rewritten target resets the episode.
    target.write_text(target.read_text().replace("unchanged", "changed", 1))
    dispatch(s, events, command)
    assert detect(events, RULES) == {4: ["command_no_progress"]}


@pytest.mark.parametrize("body", [
    "self.assertEqual(1, 2)",
    "self.skipTest('expected public skip behavior')",
    "self.assertEqual(1, 1)",
])
def test_expected_failures_skips_and_reproducibility_checks_are_unpenalized(tmp_path, monkeypatch, body):
    s, events = session(tmp_path, monkeypatch), []
    (tmp_path / "test_public.py").write_text("import unittest\nclass Regression(unittest.TestCase):\n def test_behavior(self): " + body + "\n")
    for _ in range(5):
        dispatch(s, events, "python -B -m unittest test_public")
    assert detect(events, RULES) == {}
    assert events[-1]["command_evidence"]["test"]["test_runner_recognized"]


@pytest.mark.parametrize("change", ["file", "ignored", "arguments", "result", "context", "read", "unknown_shell"])
def test_progress_and_retry_boundaries(tmp_path, monkeypatch, change):
    s, events = session(tmp_path, monkeypatch), []
    command = 'python -c "print(123)"'
    for _ in range(2):
        dispatch(s, events, command)
    if change in {"file", "ignored"}:
        (tmp_path / (".cache" if change == "ignored" else "a.py")).write_text("new evidence")
    elif change == "arguments":
        command = 'python -c "print(456)"'
    elif change == "context":
        monkeypatch.setenv("SWE_COMMAND_REGRESSION_ENV", "changed")
    elif change == "result":
        # Actual additional stdout from the shell login startup environment.
        dispatch(s, events, 'python -c "print(456)"')
    elif change == "read":
        (tmp_path / "a.py").write_text("evidence")
        dispatch(s, events, name="Read", args={"file_path": "a.py"})
    else:
        dispatch(s, events, "true")
    dispatch(s, events, command)
    assert detect(events, RULES) == {}


@pytest.mark.parametrize("mutation", ["delivery", "concurrency", "infrastructure", "hash", "output", "scope", "state", "truncated", "launch"])
def test_unreliable_evidence_never_charges(tmp_path, monkeypatch, mutation):
    s, events = session(tmp_path, monkeypatch), []
    for _ in range(3):
        dispatch(s, events, 'python -c "print(123)"')
    middle = events[1]
    if mutation == "delivery": middle["feedback_delivered"] = False
    elif mutation == "concurrency": middle["serial"] = False
    elif mutation == "infrastructure": middle["infrastructure_failure"] = True
    elif mutation == "truncated": middle["contract_observation_complete"] = False
    elif mutation == "output": middle["result"] += "unexpected"
    else:
        proof = middle["command_evidence"]
        proof[{"hash": "arguments_sha256", "scope": "state_scope", "state": "before", "launch": "launches"}[mutation]] = "bad"
    assert detect(events, RULES) == {}


def test_edit_compatibility_and_unknown_bash_barrier(tmp_path, monkeypatch):
    s = session(tmp_path, monkeypatch)
    (tmp_path / "a.py").write_text("one\n")
    legacy, observed = [], []
    edit_rules = {"edit_identical_calls", "edit_rejection_episodes"}
    actions = [("Read", {"file_path": "a.py"}),
               ("Edit", dict(file_path="a.py", old_string="one", new_string="one"))]
    bad = ("Edit", dict(file_path="a.py", old_string="missing", new_string="two"))
    actions += [bad, bad, ("Bash", {"command": 'python -c "print(123)"'}), bad, bad, bad]
    for name, args in actions:
        dispatch(s, legacy, name=name, args=args, collect=False)
        dispatch(s, observed, name=name, args=args, collect=True)
    assert detect(legacy, edit_rules) == detect(observed, edit_rules) == {
        2: ["edit_identical_calls"], 8: ["edit_rejection_episodes"]}
    for events in (legacy, observed):
        ids = aligned(events)
        captured, _ = capture_events(events, edit_rules)
        weights, _ = build_cost_weights(torch.tensor([ids]), torch.ones(1, len(ids)),
            [{"tool_repeat_events": captured}], torch.tensor([True]), cost=.02,
            edit_cost=.01, cap=.10, warmup_steps=0, step=1, enabled_rules=edit_rules)
        assert weights.sum() == pytest.approx(.02)


@pytest.mark.parametrize("feedback", [False, True])
def test_edit_inspection_boundary_unchanged_with_command_feedback(tmp_path, monkeypatch, feedback):
    from recipe.swe_agent.tool_repeat_evidence import proven_extension_key

    s, events = session(tmp_path, monkeypatch), []
    (tmp_path / "a.py").write_text("one\n")
    dispatch(s, events, name="Read", args={"file_path": "a.py"})
    bad = dict(file_path="a.py", old_string="missing", new_string="two")
    for _ in range(2):
        dispatch(s, events, name="Edit", args=bad)
    inspection = dispatch(s, events, "grep -n one a.py", feedback=feedback)
    assert proven_extension_key(inspection, "shell_reads") is not None
    dispatch(s, events, name="Edit", args=bad)
    assert detect(events, {"edit_rejection_episodes"}) == {5: ["edit_rejection_episodes"]}


def test_feedback_does_not_truncate_legacy_output():
    evidence = dict(test={}, launches=[{"exit_code": 0}], changed_path_count=0)
    text, metadata = "x" * 11990, {}
    assert ce.append_feedback(text, metadata, evidence, enabled=True, limit=12000) == text
    assert metadata == {}


def test_local_executor_uses_same_evidence_contract(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "_ensure_workspace", lambda *a: tmp_path)
    tool = tools.SWETool.__new__(tools.SWETool)
    tool.name, tool.config = "Bash", {}
    monkeypatch.setattr(tool, "_remote_enabled", lambda: False)
    data = SimpleNamespace(request_id="local", extra_fields={}, _swe_command_evidence_enabled=True)
    monkeypatch.setattr(tools, "_task_from_agent_data", lambda _: {"sandbox_backend": "local"})
    events = []
    for i in range(3):
        args = {"command": 'python -c "print(123)"'}
        response, _, metadata = asyncio.run(tool.execute("local", args, agent_data=data))
        e = edit_event(i + 1, "Bash", args, response, metadata)
        e["command_evidence"] = metadata["command_evidence"]
        events.append(e)
    assert detect(events, RULES) == {3: ["command_no_progress"]}


def test_historical_excerpts_classify_but_never_authorize_costs():
    expected = {("mwaskom__seaborn-3069", 211): "constant_output_without_verification",
                ("django__django-14792", 147): "invalid_timezone_literal",
                ("django__django-16877", 201): "identical_rewrite_without_new_evidence",
                ("django__django-16662", 44): "test_loader_failed_before_test_body"}
    for c in CARDS:
        output = c["response"].strip()
        code = int(re.search(r"exit_code: (\d+)", output)[1])
        raw = re.sub(r"^exit_code: \d+\n", "", output).split("\n[test outcome]")[0]
        observed = ce.classify(c["arguments"]["command"], raw, code)
        if (c["task"], c["turn"]) in expected:
            assert observed["reason"] == expected[c["task"], c["turn"]]
        elif c["task"] in {"django__django-16662", "django__django-11433", "pytest-dev__pytest-7432"}:
            assert observed["rule"] == ""
        assert detect([dict(c, id=1, result=c["response"])], RULES) == {}


def test_config_defaults_and_validation():
    cfg = actor_config()
    assert not cfg.tool_repeat_penalty_command_errors and not cfg.tool_command_diagnostics_enabled
    for value in (-1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            replace(cfg, tool_repeat_penalty_enabled=True, tool_repeat_penalty_command_cost=value)
    with pytest.raises(ValueError):
        replace(cfg, tool_repeat_penalty_command_window=2)


def test_command_audit_survives_dump_and_excludes_reward_json(tmp_path, monkeypatch):
    from verl.trainer.ppo.v1.replay_buffer import rollout_diagnostics
    from verl.trainer.ppo.v1.utils import validation_reward_metadata

    s, events = session(tmp_path, monkeypatch), []
    for _ in range(3):
        dispatch(s, events, 'python -c "print(123)"')
    audit = ce.audit_json(events)
    assert len(audit["command_evidence_audit_json"]) > 2048
    assert rollout_diagnostics(audit)["command_evidence_audit_json"] == audit["command_evidence_audit_json"]
    decoded = json.loads(audit["command_evidence_audit_json"])
    assert decoded["events"][0]["feedback_delivered"] is True
    assert "command_evidence_audit_json" not in validation_reward_metadata(audit)
    assert ce.audit_json(events, max_bytes=1) == {"command_evidence_audit_error": "size_limit"}
    result, requested, metrics = run_v1(monkeypatch, False, [0., 0.],
        extras=[{"command_evidence_eligible_constant_output_without_verification": 3}] * 2,
        command_diagnostics=True)
    assert "responses" not in requested[0] and "tool_repeat_weights" not in result
    assert metrics["command_evidence/eligible_constant_output_without_verification"] == 6


def test_actual_trainer_command_cost_and_unchanged_terminal_reward(tmp_path, monkeypatch):
    from verl.utils.swe_tool_failure import token_digest

    s, events = session(tmp_path, monkeypatch), []
    for _ in range(3):
        dispatch(s, events, 'python -c "print(123)"')
    ids = [10, 11, 12, 13]
    events[-1]["token_span"] = dict(version=1, start=1, end=3,
        token_sha256=token_digest(ids[1:3]), prefix_sha256=token_digest(ids[:3]))
    captured, _ = capture_events(events, RULES)
    extras = [{"tool_repeat_events": captured}] * 2
    baseline, _, _ = run_v1(monkeypatch, False, [1., 0.], extras=extras)
    output, _, metrics = run_v1(monkeypatch, False, [1., 0.], extras=extras, tool_repeat=True, command_rules=True)
    assert metrics["tool_repeat/applied"] == 2
    for row in range(2):
        weights = output["tool_repeat_weights"].unbind()[row]
        assert weights.sum() == pytest.approx(.01)
        assert weights[[0, 3]].eq(0).all()
        torch.testing.assert_close(output["returns"].unbind()[row], baseline["returns"].unbind()[row])
        torch.testing.assert_close(output["advantages"].unbind()[row][1:3], baseline["advantages"].unbind()[row][1:3].clamp_max(0))


@pytest.mark.parametrize("enabled", [False, True])
def test_client_and_execution_manager_forward_opt_in(tmp_path, monkeypatch, enabled):
    from recipe.swe_agent import remote_execution_service as service

    s = session(tmp_path, monkeypatch)
    manager = service.ExecutionManager(session_ttl_seconds=60)
    monkeypatch.setattr(manager, "_get_or_create_session", lambda *a, **kw: s)
    tool = tools.SWETool.__new__(tools.SWETool)
    tool.name, tool.config = "Bash", {}
    monkeypatch.setattr(tool, "_remote_enabled", lambda: True)
    monkeypatch.setattr(tools, "_task_from_agent_data", lambda _: {"task_id": "offline", "sandbox_backend": "local"})
    data = SimpleNamespace(request_id="offline", extra_fields={}, _swe_command_evidence_enabled=enabled,
                           _swe_command_snapshot_timeout=5.0)

    async def post(endpoint, payload, config):
        assert endpoint == "/execute"
        assert payload.get("collect_command_evidence", False) is enabled
        assert payload.get("command_snapshot_timeout_seconds") == (5.0 if enabled else None)
        return manager.execute(payload)

    monkeypatch.setattr(tools, "_remote_post_json_async", post)
    response, _, metadata = asyncio.run(tool.execute("offline", {"command": 'python -c "print(123)"'}, agent_data=data))
    assert ("command_evidence" in metadata) is enabled
    if enabled:
        assert metadata["command_evidence"]["snapshot_timeout_seconds"] == 5.0
    assert "[execution]" not in response.text


def test_bounds_output_truncation_and_external_rewrites_abstain(tmp_path, monkeypatch):
    s, events = session(tmp_path, monkeypatch), []
    # Actual stream collector used by Docker, exercised without a daemon.
    from recipe.swe_agent import remote_execution_service as service
    import os
    monkeypatch.setenv("SWE_AGENT_DOCKER_COMMAND_MAX_OUTPUT_BYTES", "1024")
    def run(command, *, timeout, **kw):
        launch = ["/bin/bash", "-lc", command]
        result = service._stream_process_result(launch, timeout=timeout, env=dict(os.environ),
            label="offline", display_command="offline", cwd=tmp_path, emit_output=False)
        ce.record(command, *result, launch=launch, cwd=tmp_path, elapsed=0)
        return result
    monkeypatch.setattr(s, "run", run)
    e = dispatch(s, events, "python -c " + shlex.quote("print('" + "x" * 5000 + "')"))
    assert e["command_evidence"]["execution_complete"] is False
    assert ce.verified(e) is None
    assert ce.snapshot(tmp_path, max_files=0)["status"] == "bounds_exceeded"
    with monkeypatch.context() as m:
        m.setattr(ce, "snapshot", lambda *a, **kw: {"status": "bounds_exceeded"})
        e = dispatch(s, events, 'python -c "print(123)"')
        assert e["command_evidence"]["state_complete"] is False and ce.verified(e) is None
    outside = tmp_path.parent / (tmp_path.name + "-outside.txt")
    outside.write_text("same")
    try:
        source = f"with open({str(outside)!r}, 'r') as f:\n    content = f.read()\nwith open({str(outside)!r}, 'w') as f:\n    f.write(content)"
        for _ in range(3):
            e = dispatch(s, events, "python -c " + shlex.quote(source))
            assert e["command_evidence"]["reason"] == "rewrite_target_outside_measured_scope"
        assert detect(events, RULES) == {}
    finally:
        outside.unlink()


@pytest.mark.parametrize("active", [False, True])
def test_native_agent_loop_command_alignment_and_delivery(tmp_path, monkeypatch, tokenizer, active):
    from recipe.swe_agent import repetition_reward as rr
    from recipe.swe_agent.agent_loop import SWEToolAgentLoop
    from verl.experimental.agent_loop.tool_agent_loop import AgentState, ToolAgentLoop
    from verl.experimental.agent_loop.tool_parser import Qwen3XMLToolParser
    from verl.utils.swe_tool_repeat import verified_span

    s = session(tmp_path, monkeypatch)
    loop = object.__new__(SWEToolAgentLoop)
    loop.response_length, loop.enable_continuous_token = 100000, False
    loop.max_assistant_turns = loop.max_user_turns = None
    loop._swe_tool_repeat_enabled, loop._swe_tool_repeat_rules = active, RULES if active else set()
    loop._swe_tool_failure_enabled = loop._swe_is_validation = False
    loop._swe_command_evidence_enabled = True
    loop._swe_repeat_config = rr.RepeatConfig(mode="off")
    loop._swe_repeat_events, loop._swe_repeat_active = [], []
    loop._trajectory_expired, loop.tools = lambda: False, {}
    loop.tokenizer, loop.tool_parser = tokenizer, Qwen3XMLToolParser(tokenizer)
    ids = tokenizer.encode('<think>Check.</think><tool_call><function=Bash><parameter=command>python -c "print(123)"</parameter></function></tool_call>', add_special_tokens=False)

    async def generate(self, data, params):
        return SimpleNamespace(token_ids=ids, log_probs=[], extra_fields={}, routed_experts=None, num_preempted=0)

    async def deadline(coro):
        return await coro

    async def execute(call, kwargs, data):
        assert data._swe_command_evidence_enabled
        return s.run_tool(call.name, json.loads(call.arguments), collect_command_evidence=True)

    loop._await_with_trajectory_deadline, loop._call_tool_with_legacy_tracking = deadline, execute
    monkeypatch.setattr(ToolAgentLoop, "_generate_model_tokens", generate)
    data = SimpleNamespace(response_mask=[], prompt_ids=[1], assistant_turns=0, user_turns=0,
        tool_calls=[], metrics={}, extra_fields={}, response_logprobs=[], request_id="offline")

    async def exercise():
        for _ in range(4):
            assert await loop._handle_generating_state(data, {}) == AgentState.PROCESSING_TOOLS
            await loop._call_tool(data.tool_calls[0], {}, data)
            data.prompt_ids += [1]
            data.response_mask += [0]
            data.user_turns += 1
        assert loop._swe_repeat_events[-1]["feedback_delivered"] is False
        assert detect(loop._swe_repeat_events, RULES) == {3: ["command_no_progress"]}
        captured, count = capture_events(loop._swe_repeat_events, loop._swe_tool_repeat_rules)
        assert count == len(captured) == int(active)
        if active:
            assert verified_span(captured[0], data.prompt_ids[-len(data.response_mask):], data.response_mask)
    asyncio.run(exercise())


def test_opt_in_launcher_profile_preserves_explicit_edit_overrides(tmp_path):
    import subprocess
    from hydra import compose, initialize_config_dir
    from tests.trainer.ppo.test_swe_launcher_on_cpu import LAUNCHER, _create_fake_checkouts, _launcher_env

    target, source, model, _ = _create_fake_checkouts(tmp_path)
    schema = source / "recipe/swe_agent/config/tool_config_efficient_v1.yaml"
    schema.write_text("tools: []\n")
    example = target / "examples/grpo_trainer/run_qwen3_5_35b_megatron.sh"
    example.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@"\n')
    env = _launcher_env(target, source, model)
    env.pop("SWE_AGENT_TOOL_CONFIG_PATH", None)
    env["SWE_AGENT_TRAINING_TOOL_REPEAT_PENALTY_EDIT_COST"] = "0.007"
    env["SWE_AGENT_TOOL_COMMAND_SNAPSHOT_TIMEOUT_SECONDS"] = "5.0"
    profile = LAUNCHER.parents[1] / "running_script_5/command_evidence_v1_profile.sh"
    result = subprocess.run(["bash", "-c", 'source "$1"\nexec "$2"', "offline-profile", str(profile), str(LAUNCHER)],
                            env=env, check=True, text=True, capture_output=True)
    assert "actor_rollout_ref.actor.tool_efficiency_profile_enabled=1" in result.stdout
    assert "actor_rollout_ref.actor.tool_repeat_penalty_command_errors=1" in result.stdout
    assert "actor_rollout_ref.actor.tool_repeat_penalty_edit_identical_calls=1" in result.stdout
    assert "actor_rollout_ref.actor.tool_repeat_penalty_edit_cost=0.007" in result.stdout
    assert f"actor_rollout_ref.rollout.multi_turn.tool_config_path={schema}" in result.stdout

    # Checking emitted strings alone misses absent Hydra YAML keys. Compose the
    # real trainer schema with the actual launcher arguments, without starting Ray.
    overrides = [line for line in result.stdout.splitlines()
                 if re.match(r"\+{0,2}(actor_rollout_ref|data|reward|trainer|ray_kwargs|algorithm)\.", line)]
    with initialize_config_dir(config_dir=str(LAUNCHER.parents[1] / "verl/trainer/config"), version_base=None):
        config = compose(config_name="ppo_trainer", overrides=["model_engine=megatron", *overrides])
    actor = config.actor_rollout_ref.actor
    assert actor.tool_efficiency_profile_enabled == 1
    assert actor.tool_command_diagnostics_enabled == actor.tool_command_feedback_enabled == 1
    assert actor.tool_command_snapshot_timeout_seconds == 5.0
    assert actor.tool_repeat_penalty_command_errors == actor.tool_repeat_penalty_command_no_progress == 1
    assert actor.tool_repeat_penalty_command_cost == 0.01
    assert actor.tool_repeat_penalty_command_window == 16
    assert actor.tool_repeat_penalty_command_protected_calls == 2
    assert actor.tool_repeat_penalty_edit_identical_calls == 1
    assert actor.tool_repeat_penalty_edit_cost == 0.007


@pytest.mark.parametrize("engine", ["dp", "megatron"])
def test_command_hydra_defaults_match_actor_dataclass(engine):
    from hydra import compose, initialize_config_dir

    directory = Path(__file__).resolve().parents[3] / "verl/trainer/config"
    with initialize_config_dir(config_dir=str(directory), version_base=None):
        config = compose(config_name="ppo_trainer", overrides=["model_engine=" + engine])
    defaults = actor_config()
    for name in (
        "tool_efficiency_profile_enabled", "tool_command_diagnostics_enabled", "tool_command_feedback_enabled",
        "tool_command_snapshot_timeout_seconds",
        "tool_repeat_penalty_command_errors", "tool_repeat_penalty_command_no_progress",
        "tool_repeat_penalty_command_cost", "tool_repeat_penalty_command_window",
        "tool_repeat_penalty_command_protected_calls",
    ):
        assert config.actor_rollout_ref.actor[name] == getattr(defaults, name)

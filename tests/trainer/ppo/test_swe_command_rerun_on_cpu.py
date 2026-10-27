"""Rerun-derived evidence tests, using temporary CPU executions, not model rollouts."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from recipe.swe_agent import command_evidence as ce
from tests.trainer.ppo.test_swe_command_evidence_on_cpu import RULES, dispatch
from tests.trainer.ppo.test_swe_edit_recovery_on_cpu import event, session
from tests.trainer.ppo.test_swe_python_test_repeat_on_cpu import aligned
from tests.trainer.ppo.test_swe_tool_failure_on_cpu import actor_config
from verl.utils.swe_tool_repeat import build_cost_weights, capture_events, detect

CASES = json.loads((Path(__file__).parent / "fixtures/swe_command_rerun_20261027.json").read_text())


def case(row, number):
    return next(c for c in CASES if c["row"] == row and c["event"] == number)


@pytest.mark.parametrize("c", CASES, ids=lambda c: f"row{c['row']}-call{c['event']}")
def test_audited_rerun_classifications_and_negative_controls(c):
    assert hashlib.sha256(c["output"].encode()).hexdigest() == c["output_sha256"]
    result = ce.classify(c["command"], c["output"], c["exit_code"])
    if (c["row"], c["event"]) == (1, 16):
        assert result["reason"] == "python_syntax_error"
        assert result["source_compile_failure"] and not result["source_proof_safe"]
        assert result["setup_paths"] == ["/tmp/test_models"]
    elif c["previous_classification"]:
        assert result["reason"] == c["previous_classification"]
        if result["reason"] == "test_loader_failed_before_test_body":
            assert result["test"]["tests_executed"] == 0
    else:
        assert result["rule"] == ""  # Includes useful assertions and varied failed probes.
    if (c["row"], c["event"]) == (1, 83):
        assert result["shell_success_with_python_traceback"]
        assert result["python_exception"]["terminal_exception"] == "AttributeError"
        assert "get_class_serializer" in result["python_exception"]["terminal_message"]
        assert not result["python_exception"]["penalty_authorized"]
        assert result["python_exception"]["escaped_exception"] == "unproven"
    if c["row"] == 1 and c["event"] in (65, 68, 76, 86):
        assert not result["python_exception"]["traceback_observed"]
        assert result["test"]["verification_meaningful"] is None


def test_audited_wrapped_compile_failure_selects_and_charges_after_protection(tmp_path, monkeypatch):
    executor, events = session(tmp_path, monkeypatch), []
    # The invalid body is exactly the audited source. Only the setup directory
    # is remapped into the measured sandbox. Python never executes its imports.
    command = case(1, 16)["command"].replace("/tmp/test_models", str(tmp_path / "test_models"))
    for _ in range(4):
        dispatch(executor, events, command)
    assert events[0]["command_evidence"]["changed_path_count"] > 0
    assert detect(events[:3], RULES) == {}
    assert detect(events, RULES) == {4: ["command_errors"]}
    assert events[-1]["command_evidence"]["proof_scope"] == "workspace"
    ids = aligned(events)
    captured, _ = capture_events(events, RULES)
    weights, counts = build_cost_weights(torch.tensor([ids]), torch.ones(1, len(ids)),
        [{"tool_repeat_events": captured}], torch.tensor([True]), cost=.02, command_cost=.01,
        cap=.1, warmup_steps=0, step=1, enabled_rules=RULES)
    assert counts["rules/command_errors/applied"] == 1
    assert weights.sum() == pytest.approx(.01)
    # A meaningful edit resets protection, even though the source still cannot compile.
    (tmp_path / "fix.py").write_text("changed = True\n")
    dispatch(executor, events, command)
    assert detect(events, RULES) == {4: ["command_errors"]}


def test_external_setup_and_unknown_state_do_not_authorize_costs(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    root.mkdir()
    executor, events = session(root, monkeypatch), []
    command = f"mkdir -p {tmp_path / 'external'}\npython -c 'if True print(1)'"
    for _ in range(3):
        dispatch(executor, events, command)
    proof = events[-1]["command_evidence"]
    assert proof["reason"] == "python_syntax_error"
    assert proof["ineligible_reason"] == "setup_target_outside_measured_scope"
    assert detect(events, RULES) == {}


@pytest.mark.parametrize("filter_command,selected", [("tail -20", True), ("head -1", False)])
def test_pipeline_compilation_proof_and_truncated_control(tmp_path, monkeypatch, filter_command, selected):
    (tmp_path / "external").symlink_to("/outside_unmeasured")
    executor, events = session(tmp_path, monkeypatch), []
    command = f"python -c 'if True print(1)' 2>&1 | {filter_command}"
    for _ in range(3):
        dispatch(executor, events, command, feedback=True)
    proof = events[-1]["command_evidence"]
    assert proof["launches"][0]["exit_code"] == 0
    assert not proof["state_complete"]
    assert detect(events, RULES) == ({3: ["command_errors"]} if selected else {})
    if selected:
        assert proof["pipeline_stage_evidence"][0]["inferred_exit_code"] == 1
        assert proof["proof_scope"] == "inline-source-compilation-v1"
        assert proof["shell_success_without_test_success"]


@pytest.mark.parametrize("source", [
    "assert 1 == 2",  # An ordinary failing reproduction, not a tool contract error.
    "import traceback\ntry:\n  raise ValueError('expected')\nexcept ValueError:\n  traceback.print_exc()",
    "raise SyntaxError('runtime exception, not a compilation failure')",
])
def test_tracebacks_are_visible_without_penalizing_testing(tmp_path, monkeypatch, source):
    import shlex
    executor, events = session(tmp_path, monkeypatch), []
    command = "python -c " + shlex.quote(source) + " 2>&1 | tail -20"
    for _ in range(3):
        dispatch(executor, events, command, feedback=True)
    proof = events[-1]["command_evidence"]
    assert proof["shell_success_with_python_traceback"]
    assert proof["launches"][0]["exit_code"] == 0
    assert "Python traceback observed" in ce.feedback(proof)
    assert detect(events, RULES) == {}
    _, diagnostics = ce.analyze(events)
    assert diagnostics["observed_shell_success_with_python_traceback"] == 3


def test_cache_changes_are_visible_and_remain_in_the_proof(tmp_path, monkeypatch):
    monkeypatch.delenv("PYTHONDONTWRITEBYTECODE", raising=False)
    (tmp_path / "sample.py").write_text("value = 1\n")
    executor, events = session(tmp_path, monkeypatch), []
    dispatch(executor, events, "python -c 'import sample'")
    proof = events[-1]["command_evidence"]
    assert proof["state_change_summary"]["python_bytecode"] > 0
    assert proof["state_change_summary"]["cache_only"]
    assert proof["state_change_summary"]["progress"] == "unproven"
    assert proof["before"] != proof["after"]
    assert proof["ineligible_reason"] == "content_or_permissions_changed"
    assert "task progress is unproven" in ce.feedback(proof)
    before = ce.snapshot(tmp_path)
    next(tmp_path.glob("__pycache__/*.pyc")).write_bytes(b"modified cache")
    assert ce.snapshot(tmp_path)["digest"] != before["digest"]


def test_request_budget_recovers_actual_snapshot_evidence(tmp_path, monkeypatch):
    executor = session(tmp_path, monkeypatch)
    for timeout, expected in [(1e-9, {}), (5.0, {3: ["command_no_progress"]})]:
        events = []
        for i in range(3):
            args = {"command": "python -c 'print(123)'"}
            text, reward, metadata = executor.run_tool("Bash", args, collect_command_evidence=True,
                command_snapshot_timeout_seconds=timeout)
            e = event(i + 1, "Bash", args, text, metadata)
            e["command_evidence"] = metadata["command_evidence"]
            events.append(e)
            assert reward == 0
            assert metadata["command_evidence"]["snapshot_timeout_seconds"] == timeout
        assert detect(events, RULES) == expected
        if timeout < 1:
            assert events[-1]["command_evidence"]["snapshot_diagnostics"]["before"]["limit"] == "timeout"


@pytest.mark.parametrize("bad", [0, -1, True, float("inf"), float("nan"), 61, "5"])
def test_invalid_budget_rejected_before_execution(bad):
    with pytest.raises(ValueError):
        with ce.collect(True, {"command": "unused"}, snapshot_timeout_seconds=bad):
            pytest.fail("Invalid request budget must not execute")
    with pytest.raises(ValueError):
        replace(actor_config(), tool_command_snapshot_timeout_seconds=bad)


@pytest.mark.parametrize("version", ["command-evidence-v1", "command-evidence-v2"])
def test_old_proofs_keep_their_original_requirements(tmp_path, monkeypatch, version):
    executor, events = session(tmp_path, monkeypatch), []
    for _ in range(3):
        dispatch(executor, events, "python -c 'print(123)'")
    for e in events:
        p = e["command_evidence"]
        p["version"] = version
        for k in ("snapshot_timeout_seconds", "python_exception", "state_change_summary", "source_proof_safe", "setup_paths"):
            p.pop(k, None)
        ce.bind(p, e["result"])
    assert detect(events, RULES) == {3: ["command_no_progress"]}
    for e in events:
        e["command_evidence"]["state_complete"] = False
        ce.bind(e["command_evidence"], e["result"])
    assert detect(events, RULES) == {}


def test_episode_reset_is_explained_without_changing_retry_policy(tmp_path, monkeypatch):
    executor, events = session(tmp_path, monkeypatch), []
    for value in (1, 2, 1):
        dispatch(executor, events, f"python -c 'print({value})'")
    decisions = []
    selected, _ = ce.analyze(events, decisions=decisions)
    assert selected == {}
    assert "arguments" in decisions[-1]["episode_reset"]
    assert "result" in decisions[-1]["episode_reset"]
    assert decisions[-1]["matching_calls"] == 1


@pytest.mark.parametrize("command", [
    "echo setup; python -c 'if True print(1)'",
    "mkdir -p --verbose\npython -c 'if True print(1)'",
    "python -c 'if True print(1)' 2>&1 | tee file | tail -20",
    "python -c 'if True print(1)' 2>&1 | tail -20 existing_file",
])
def test_unknown_wrappers_cannot_authorize_compilation_penalties(command):
    output = '  File "<string>", line 1\n    if True print(1)\n            ^^^^^\nSyntaxError: invalid syntax\n'
    assert ce.classify(command, output, 0)["rule"] == ""

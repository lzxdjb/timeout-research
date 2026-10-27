"""Real first-validation outputs and safe executor reproductions of audit gaps."""

import copy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from recipe.swe_agent import command_evidence as ce
from recipe.swe_agent.test_results import summarize_test_execution
from tests.trainer.ppo.test_swe_command_evidence_on_cpu import RULES, card, dispatch
from tests.trainer.ppo.test_swe_edit_recovery_on_cpu import session
from tests.trainer.ppo.test_swe_python_test_repeat_on_cpu import aligned
from tests.trainer.ppo.test_swe_tool_failure_on_cpu import run_v1
from verl.utils.swe_tool_repeat import build_cost_weights, capture_events, detect

CASES = json.loads((Path(__file__).parent / "fixtures/swe_command_first_validation_20261027.json").read_text())


def case(row, event):
    return next(c for c in CASES if c["row"] == row and c["event"] == event)


def test_actual_validation_outputs_and_expected_failure_controls():
    for c in CASES:
        assert hashlib.sha256(c["output"].encode()).hexdigest() == c["output_sha256"]
        result = ce.classify(c["command"], c["output"], c["exit_code"])
        if c["row"] == 246:
            assert c["previous_tests_executed"] == 1
            assert result["reason"] == "test_loader_failed_before_test_body"
            assert result["test"]["tests_executed"] == 0
            assert result["test"]["verification_meaningful"] is False
        elif c["row"] == 1:
            # TypeError in the proposed fix, a useful retry, passing tests, and
            # exploratory boundary inputs must not become command-error costs.
            assert result["rule"] == ""
        elif c["row"] == 233:
            assert result["reason"] == "command_not_found"
            assert result["pipeline_stage_evidence"][0]["inferred_exit_code"] == 127


@pytest.mark.parametrize("suffix", ["", ".missing"])
def test_loader_counts_mixed_suites_duplicates_and_conflicts(suffix):
    command = "python -m unittest missing real"
    header = f"ERROR: missing (unittest.loader._FailedTest{suffix})\nImportError: missing\n"
    mixed = header + "Ran 3 tests in 0.01s\nFAILED (errors=1)\n"
    summary = summarize_test_execution(command, mixed, exit_code=1)
    assert summary["test_loader_error_count"] == 1 and summary["tests_executed"] == 2
    assert ce.classify(command, mixed, 1)["rule"] == ""
    duplicate = header * 2 + "Ran 2 tests in 0.01s\nFAILED (errors=2)\n"
    assert summarize_test_execution(command, duplicate, exit_code=1)["tests_executed"] == 0
    conflicting = header * 2 + "Ran 1 test in 0.01s\nFAILED (errors=1)\n"
    summary = summarize_test_execution(command, conflicting, exit_code=1)
    assert summary["test_count_conflict"] and summary["tests_executed"] == -1
    assert ce.classify(command, conflicting, 1)["rule"] == ""
    later = duplicate + "Ran 1 test in 0.01s\nOK\n"
    summary = summarize_test_execution(command, later, exit_code=0)
    assert summary["test_loader_error_count"] == 0 and summary["tests_executed"] == 1
    assert ce.classify(command, later, 0)["rule"] == ""


def test_real_malformed_target_with_internal_symlink_selects_third(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    for directory in ("db_functions", "db_functions/math", "tests"):
        p = tmp_path / directory
        p.mkdir(exist_ok=True)
        (p / "__init__.py").write_text("")
    (tmp_path / "db_functions/math/test_mod.py").write_text("existing = True\n")
    (tmp_path / "tests/runtests.py").write_text(
        "import sys, unittest\nfrom pathlib import Path\n"
        "sys.path.insert(0, str(Path(__file__).resolve().parents[1]))\n"
        "suite = unittest.defaultTestLoader.loadTestsFromNames(sys.argv[1:])\n"
        "sys.exit(not unittest.TextTestRunner().run(suite).wasSuccessful())\n"
    )
    (tmp_path / "test_alias").symlink_to("tests", target_is_directory=True)
    executor, events = session(tmp_path, monkeypatch), []
    command = case(246, 126)["command"].replace("/testbed", str(tmp_path))
    for _ in range(3):
        dispatch(executor, events, command)
    assert detect(events, RULES) == {3: ["command_errors"]}
    assert events[-1]["command_evidence"]["test"]["tests_executed"] == 0
    # A real source edit changes the relevant import state and protects retries.
    (tmp_path / "db_functions/math/test_mod.py").write_text("existing = False\n")
    dispatch(executor, events, command)
    assert detect(events, RULES) == {3: ["command_errors"]}


def test_snapshots_cache_content_and_track_ignored_files_and_symlinks(tmp_path, monkeypatch):
    target = tmp_path / "a.py"
    target.write_text("old\n")
    (tmp_path / "alias").symlink_to("a.py")
    first = ce.snapshot(tmp_path)
    # A fresh file must not seed a cache entry with a racy timestamp. Advance
    # only the observation clock while the file is untouched to test reuse.
    assert (str(target), ce._identity(target.stat())) not in ce._HASH_CACHE
    settled_time = ce.time.time_ns() + 3_000_000_000
    with monkeypatch.context() as m:
        m.setattr(ce.time, "time_ns", lambda: settled_time)
        ce.snapshot(tmp_path)
        cached = ce.snapshot(tmp_path)
    assert first["status"] == cached["status"] == "ok"
    assert first["digest"] == cached["digest"] and cached["bytes_hashed"] == 0
    old_time = target.stat().st_mtime_ns
    target.write_text("new\n")
    os.utime(target, ns=(old_time, old_time))
    changed = ce.snapshot(tmp_path)
    assert changed["digest"] != first["digest"]  # Restored mtime cannot reuse stale hash.
    (tmp_path / ".cache").write_text("meaningful cache state")
    assert ce.snapshot(tmp_path)["digest"] != changed["digest"]
    (tmp_path / "alias").unlink()
    (tmp_path / "alias").symlink_to(".cache")
    assert ce.snapshot(tmp_path)["digest"] != changed["digest"]


def test_snapshot_failure_diagnostics_and_source_scope(tmp_path, monkeypatch):
    (tmp_path / "outside").symlink_to(tmp_path.parent)
    snapshot = ce.snapshot(tmp_path)
    assert snapshot["status"] == "unsupported_file_type"
    assert snapshot["path"] == "outside" and snapshot["file_type"] == "external_symlink"
    executor, events = session(tmp_path, monkeypatch), []
    command = "cd " + str(tmp_path) + " && python << 'PY'\nif True print(1)\nPY"
    for _ in range(3):
        dispatch(executor, events, command)
    assert detect(events, RULES) == {3: ["command_errors"]}
    proof = events[-1]["command_evidence"]
    assert proof["state_complete"] is False and proof["proof_scope"] == "inline-source-compilation-v1"
    assert proof["snapshot_diagnostics"]["before"]["path"] == "outside"
    # Ordinary execution cannot use the source-compilation exception.
    events = []
    for _ in range(3):
        dispatch(executor, events, 'python -c "print(123)"')
    assert detect(events, RULES) == {}
    assert ce.snapshot(tmp_path, max_files=0)["limit"] == "max_files"
    (tmp_path / "outside").unlink()
    (tmp_path / "large.bin").write_bytes(b"x" * 10)
    assert ce.snapshot(tmp_path, max_bytes=0)["limit"] == "max_bytes"
    assert ce.snapshot(tmp_path, timeout=-1)["limit"] == "timeout"


def test_absolute_container_symlinks_and_legacy_evidence(tmp_path, monkeypatch):
    (tmp_path / "a.py").write_text("content")
    link = tmp_path / "alias"
    link.symlink_to("/testbed/a.py")
    assert ce.snapshot(tmp_path)["status"] == "unsupported_file_type"
    assert ce.snapshot(tmp_path, mount="/testbed")["status"] == "ok"
    link.unlink()
    executor, events = session(tmp_path, monkeypatch), []
    for _ in range(3):
        dispatch(executor, events, 'python -c "print(123)"')
    for e in events:
        proof = e["command_evidence"]
        proof["version"] = ce.LEGACY_VERSION
        for field in ("proof_scope", "comparison_state", "source_compile_failure", "source_sha256", "snapshot_diagnostics"):
            proof.pop(field, None)
        ce.bind(proof, e["result"])
    assert detect(events, RULES) == {3: ["command_no_progress"]}
    # Old records are accepted with their original requirements, not upgraded.
    for e in events:
        e["command_evidence"]["state_complete"] = False
        ce.bind(e["command_evidence"], e["result"])
    assert detect(events, RULES) == {}


@pytest.mark.parametrize("field,value", [("command_evidence", "invalid"), ("version", []),
                                         ("launches", ["x"]), ("launches", None)])
def test_malformed_command_metadata_abstains_without_breaking_audit(tmp_path, monkeypatch, field, value):
    executor, events = session(tmp_path, monkeypatch), []
    dispatch(executor, events, 'python -c "print(123)"')
    if field == "command_evidence":
        events[0][field] = value
    else:
        events[0]["command_evidence"][field] = value
        ce.bind(events[0]["command_evidence"], events[0]["result"])
    assert detect(events, RULES) == {}
    assert "command_evidence_audit_json" in ce.audit_json(events)


def test_two_actual_syntax_retries_then_correction_remain_protected(tmp_path, monkeypatch):
    executor, events = session(tmp_path, monkeypatch), []
    # The exact invalid print syntax from the audited SymPy trajectory.
    source = case(186, 152)
    assert "SyntaxError" in source["output"]
    for _ in range(2):
        dispatch(executor, events, "python -c 'print(factors: 1)' ")
    dispatch(executor, events, "python -c 'print(1)'")
    assert detect(events, RULES) == {}
    # A runtime compile()/import failure is not failure of the submitted body.
    output = 'Traceback (most recent call last):\n  File "<string>", line 1\nSyntaxError: bad\n'
    assert ce.classify("python -c 'if True print(1)'", output, 1)["rule"] == ""


@pytest.mark.parametrize("command", [
    "swe_audit_absent_executable arg | head -5",
    "python -m swe_audit_absent_module tests/example.py 2>&1 | head -100",
])
def test_masked_pipeline_failures_are_attributed_without_changing_exit(tmp_path, monkeypatch, command):
    executor, events = session(tmp_path, monkeypatch), []
    for _ in range(3):
        dispatch(executor, events, command, feedback=True)
    proof = events[-1]["command_evidence"]
    assert proof["launches"][0]["exit_code"] == 0
    assert proof["pipeline_stage_evidence"][0]["stage"] == 0
    assert "Pipeline stage 0 failed:" in ce.feedback(proof)
    assert detect(events, RULES) == {3: ["command_errors"]}
    assert ce.classify(command.replace("head -", "tee file -"), case(233, 57)["output"], 0)["rule"] == ""


def test_prefixed_heredoc_cannot_authorize_unmeasured_rewrite(tmp_path, monkeypatch):
    executor, events = session(tmp_path, monkeypatch), []
    (tmp_path / "a.txt").write_text("same")
    outside = tmp_path.parent / (tmp_path.name + "_outside")
    outside.mkdir()
    (outside / "a.txt").write_text("same")
    command = (f"cd {outside} && python << 'PY'\nwith open('a.txt', 'r') as f:\n    text = f.read()\n"
               "with open('a.txt', 'w') as f:\n    f.write(text)\nPY")
    for _ in range(3):
        dispatch(executor, events, command)
    assert events[-1]["command_evidence"]["reason"] == "rewrite_target_outside_measured_scope"
    assert detect(events, RULES) == {}


def audited_application_command(tmp_path, monkeypatch, rule):
    if rule == "command_no_progress":
        return card("mwaskom__seaborn-3069", 211)["arguments"]["command"]
    # Reproduce the audited missing prerequisite without importing Django or
    # relying on whichever unrelated package named `tests` is installed.
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    for directory in ("tests", "tests/migrations"):
        path = tmp_path / directory
        path.mkdir()
        (path / "__init__.py").write_text("")
    (tmp_path / "tests/migrations/test_writer.py").write_text(
        "import custom_migration_operations.more_operations\n"
    )
    return card("django__django-16662", 44)["arguments"]["command"].replace("/testbed", str(tmp_path))


@pytest.mark.parametrize("rule", ["command_errors", "command_no_progress"])
@pytest.mark.parametrize("mode", ["applied", "rule_disabled", "zero_cost", "cap_exhausted",
                                  "failure_precedence", "inactive_sample", "invalid_span", "duplicate", "warmup_zero"])
def test_per_rule_costs_and_application_decisions(tmp_path, monkeypatch, mode, rule):
    executor, events = session(tmp_path, monkeypatch), []
    command = audited_application_command(tmp_path, monkeypatch, rule)
    for _ in range(3):
        dispatch(executor, events, command)
    assert detect(events, RULES) == {3: [rule]}
    ids = aligned(events)
    captured, _ = capture_events(events, RULES)
    if mode == "invalid_span":
        captured[0]["token_sha256"] = "invalid"
    if mode == "duplicate":
        captured *= 2
    extras = [{"tool_repeat_events": captured}]
    reserved = torch.zeros(1, len(ids))
    if mode == "cap_exhausted": reserved[0, 0] = .1
    if mode == "failure_precedence": reserved[0, captured[0]["start"]] = .02
    weights, counts = build_cost_weights(torch.tensor([ids]), torch.ones(1, len(ids)), extras,
        torch.tensor([mode != "inactive_sample"]), cost=.02, command_cost=0 if mode == "zero_cost" else .01,
        cap=.1, warmup_steps=10 if mode == "warmup_zero" else 0, step=0,
        enabled_rules=set() if mode == "rule_disabled" else RULES, existing_weights=reserved)
    audit = json.loads(extras[0]["tool_repeat_application_audit_json"])
    assert audit["stage"] == "training_cost_weights"
    assert mode in {e["reason"] for e in audit["events"]}
    expected = .01 if mode in {"applied", "duplicate"} else 0
    assert counts[f"rules/{rule}/cost"] == pytest.approx(expected)
    assert weights.sum() == pytest.approx(expected)
    assert sum(e["cost"] for e in audit["events"]) == pytest.approx(expected)
    # Original arguments, canonical observations and aligned spans are exportable.
    exported = json.loads(ce.audit_json(events, retained=captured)["command_evidence_audit_json"])
    assert exported["events"][0]["arguments"] == events[0]["arguments"]
    assert exported["events"][0]["canonical_result"] == events[0]["result"]
    assert exported["events"][2]["command_decision"]["selected"]


@pytest.mark.parametrize("rule", ["command_errors", "command_no_progress"])
def test_actual_trainer_persists_application_audit_and_rule_metrics(tmp_path, monkeypatch, rule):
    executor, events = session(tmp_path, monkeypatch), []
    command = audited_application_command(tmp_path, monkeypatch, rule)
    for _ in range(3):
        dispatch(executor, events, command)
    assert detect(events, RULES) == {3: [rule]}
    from verl.utils.swe_tool_failure import token_digest
    ids = [10, 11, 12, 13]
    events[-1]["token_span"] = dict(version=1, start=1, end=3,
        token_sha256=token_digest(ids[1:3]), prefix_sha256=token_digest(ids[:3]))
    captured, _ = capture_events(events, RULES)
    extras = [{"tool_repeat_events": captured}, {"tool_repeat_events": captured}]
    baseline, _, _ = run_v1(monkeypatch, False, [1., 0.], extras=copy.deepcopy(extras))
    output, _, metrics = run_v1(monkeypatch, False, [1., 0.], extras=extras, tool_repeat=True, command_rules=True)
    assert metrics[f"tool_repeat/rules/{rule}/applied"] == 2
    assert metrics[f"tool_repeat/rules/{rule}/cost"] == pytest.approx(.02)
    audits = list(output["tool_repeat_application_audit_json"])
    assert all(json.loads(a)["events"][0]["cost"] == .01 for a in audits)
    for a, b in zip(output["returns"].unbind(), baseline["returns"].unbind(), strict=True):
        torch.testing.assert_close(a, b)


def test_training_dump_keeps_application_audit_with_sorted_row(tmp_path, monkeypatch):
    from unittest.mock import MagicMock
    import numpy as np
    from verl.trainer.ppo.v1 import trainer_base as module
    keys = ["z_0_0", "a_0_0"]
    audits = [json.dumps(dict(stage="training_cost_weights", marker=key)) for key in keys]
    prompts, responses = MagicMock(), MagicMock()
    prompts.to_padded_tensor.return_value = responses.to_padded_tensor.return_value = keys
    responses.offsets.return_value.diff.return_value = torch.tensor([1, 1])

    def get(**kwargs):
        fields = kwargs["select_fields"]
        if fields == ["tool_repeat_application_audit_json"]:
            # TensorDict may expose a LinkedList instead of an array.
            return {fields[0]: audits}
        if len(fields) == 1:
            raise KeyError(fields[0])
        return dict(uid=np.array(keys), prompts=prompts, responses=responses,
                    rm_scores=torch.tensor([[1.], [0.]]))

    monkeypatch.setattr(module.tq, "kv_batch_get", get)
    trainer = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0, decode=lambda x, **kw: x))
    trainer._dump_generations = lambda **kw: module.PPOTrainer._write_generations(**kw, global_steps=1)
    module.PPOTrainer._log_rollout_data(trainer, SimpleNamespace(keys=keys, tags=[{}, {}], partition_id="train"), {}, str(tmp_path))
    rows = [json.loads(s) for s in (tmp_path / "1.jsonl").read_text().splitlines()]
    assert [r["uid"] for r in rows] == sorted(keys)
    assert [r["score"] for r in rows] == [0., 1.]
    assert all(json.loads(r["tool_repeat_application_audit_json"])["marker"] == r["uid"] for r in rows)

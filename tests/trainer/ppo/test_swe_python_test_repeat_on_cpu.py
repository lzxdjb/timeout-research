"""Observed Python loops: live executor, historical cards, and local cost/RM routing."""

import asyncio
import copy
import gzip
import json
import shlex
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from recipe.swe_agent import python_test_repeat_evidence as evidence
from recipe.swe_agent import tools
from tests.trainer.ppo.test_swe_tool_failure_on_cpu import actor_config
from tests.trainer.ppo.test_swe_tool_repeat_extensions_on_cpu import event, session
from verl.utils.swe_local_penalty_shortcut import verified_local_penalties
from verl.utils.swe_tool_failure import token_digest
from verl.utils.swe_tool_repeat import RULES, build_cost_weights, capture_events, detect, diagnostics, verified_span

FIXTURE = Path(__file__).parent / "fixtures/swe_python_test_loops_audited_20261020.jsonl.gz"
with gzip.open(FIXTURE, "rt") as stream:
    CARDS = [json.loads(line) for line in stream]


def live_events(tmp_path, monkeypatch, count=26, command="python3 -S -c 'print(123)'", hook=None):
    executor = session(tmp_path, monkeypatch)
    events = []
    for i in range(1, count + 1):
        if hook:
            hook(i)
        args = {"command": command}
        text, _, metadata = executor.run_tool("Bash", args, collect_python_test_repeat_evidence=True)
        events.append(event(i, "Bash", args, text, metadata))
    return events


def aligned(events):
    ids = list(range(2 * len(events)))
    for i, e in enumerate(events):
        start, end = 2 * i, 2 * i + 1
        e["token_span"] = dict(
            version=1,
            start=start,
            end=end,
            token_sha256=token_digest(ids[start:end]),
            prefix_sha256=token_digest(ids[:end]),
        )
    return ids


def test_live_threshold_default_off_diagnostic_independence_and_budgets(tmp_path, monkeypatch):
    events = live_events(tmp_path, monkeypatch)
    assert all(e["repeat_python_test_evidence"]["complete"] for e in events)
    assert detect(events[:20], {"python_tests"}) == {}
    assert set(detect(events, {"python_tests"})) == set(range(21, 27))
    assert detect(events, set(RULES) - {"python_tests"}) == {}
    assert detect(events, {"python_tests"}, python_test_protected_calls=25) == {26: ["python_tests"]}
    assert diagnostics(events, python_test_shadow=True)["tool_repeat_shadow_python_test_calls"] == 26
    assert diagnostics(events)["tool_repeat_eligible_python_tests"] == 26
    ids = aligned(events)
    captured, selected = capture_events(events, {"python_tests"})
    assert len(captured) == selected == 6
    kwargs = dict(cost=0.02, cap=0.10, warmup_steps=10, step=5, enabled_rules={"python_tests"})
    responses, mask = torch.tensor([ids]), torch.ones(1, len(ids))
    # Duplicate evidence does not double charge; padding and disabled flags cost zero.
    weights, stats = build_cost_weights(
        responses, mask, [{"tool_repeat_events": captured * 2}], torch.tensor([True]), **kwargs
    )
    assert stats["applied"] == 6 and weights.sum().item() == pytest.approx(0.05)
    assert weights[0, :40].eq(0).all() and weights[0, 1::2].eq(0).all()
    for disabled in ("rule", "padding"):
        zero, _ = build_cost_weights(
            responses,
            mask,
            [{"tool_repeat_events": captured}],
            torch.tensor([disabled != "padding"]),
            **dict(kwargs, enabled_rules=set() if disabled == "rule" else {"python_tests"}),
        )
        assert zero.eq(0).all()
    second, _ = build_cost_weights(
        responses, mask, [{"tool_repeat_events": captured}], torch.tensor([True]), existing_weights=weights, **kwargs
    )
    assert second.eq(0).all()
    for mutation in ("start", "prefix_sha256", "token_sha256"):
        bad = copy.deepcopy(captured[0])
        bad[mutation] = 0 if mutation == "start" else "stale"
        assert verified_span(bad, ids, [1] * len(ids)) is None
    output = SimpleNamespace(
        response_ids=ids, response_mask=[1] * len(ids), extra_fields={"tool_repeat_events": captured}
    )
    config = dict(tool_repeat_penalty_enabled=True, tool_repeat_penalty_python_tests=True)
    assert verified_local_penalties(output, config) == {"tool_repeat": 6}
    for changes in (
        {"tool_repeat_penalty_python_tests": False},
        {"tool_repeat_penalty_cost": 0},
        {"tool_repeat_penalty_cap": 0},
        {"tool_repeat_penalty_enabled": False},
    ):
        assert verified_local_penalties(output, dict(config, **changes)) == {}
    output.extra_fields = {"tool_repeat_shadow_python_test_calls": 999}
    assert verified_local_penalties(output, config) == {}


@pytest.mark.parametrize(
    "mutation",
    [
        "serial",
        "feedback_delivered",
        "infrastructure_failure",
        "id",
        "arguments",
        "name",
        "result",
        "completed",
        "observation_status",
        "contract_observation_complete",
        "proof",
        "state",
        "context",
        "scope",
        "command_results",
        "heuristic",
        "interleave",
    ],
)
def test_evidence_and_delivery_barriers(tmp_path, monkeypatch, mutation):
    # One actual execution, copied only to test metadata rejection and streak logic.
    original = live_events(tmp_path, monkeypatch, 1)[0]
    events = [dict(copy.deepcopy(original), id=i) for i in range(1, 27)]
    e = events[12]
    if mutation in {"serial", "feedback_delivered", "completed", "contract_observation_complete"}:
        e[mutation] = False
    elif mutation == "infrastructure_failure":
        e[mutation] = True
    elif mutation == "id":
        e[mutation] = 999
    elif mutation == "arguments":
        e[mutation]["command"] += " "
    elif mutation == "name":
        e[mutation] = "run_shell"
    elif mutation == "result":
        e[mutation] += "new output"
    elif mutation == "observation_status":
        e[mutation] = "artifact_incomplete"
    elif mutation == "proof":
        e["repeat_python_test_evidence"] = None
    elif mutation == "interleave":
        e.update(name="Read", arguments={"file_path": "a.py"})
    else:
        field, value = {
            "state": ("after", "changed"),
            "context": ("context_sha256", "f" * 64),
            "scope": ("state_scope", "git-only"),
            "command_results": ("command_results_sha256", ""),
            "heuristic": ("heuristic", False),
        }[mutation]
        e["repeat_python_test_evidence"][field] = value
    assert detect(events, {"python_tests"}) == {}


@pytest.mark.parametrize("change", ["edit", "ignored", "result"])
def test_tests_after_edits_and_dependency_or_result_change(tmp_path, monkeypatch, change):
    (tmp_path / ".gitignore").write_text(".cache\n")
    (tmp_path / "dependency").write_text("original")
    command = "python3 -S -c \"print(open('dependency').read())\"" if change == "result" else "python3 -S -c 'print(1)'"

    def hook(i):
        if i == 14:
            (tmp_path / (".cache" if change == "ignored" else "dependency")).write_text("changed")

    assert detect(live_events(tmp_path, monkeypatch, command=command, hook=hook), {"python_tests"}) == {}


def test_stateful_counter_has_no_penalty(tmp_path, monkeypatch):
    (tmp_path / "counter").write_text("0")
    code = "from pathlib import Path; p=Path('counter'); p.write_text(str(int(p.read_text())+1)); print('same')"
    events = live_events(tmp_path, monkeypatch, command="python3 -S -c " + shlex.quote(code))
    assert all(e["repeat_python_test_evidence"]["unavailable_reason"] == "state_changed" for e in events)
    assert detect(events, {"python_tests"}) == {}


def test_observationally_identical_stress_retry_policy_is_explicit(tmp_path, monkeypatch):
    # Intent is unknowable here. The approved heuristic charges after 20 even
    # when these same observations came from a purposeful stress experiment.
    events = live_events(tmp_path, monkeypatch, count=21)
    assert detect(events, {"python_tests"}) == {21: ["python_tests"]}
    assert events[-1]["repeat_python_test_evidence"]["heuristic"] is True


@pytest.mark.parametrize("card", CARDS, ids=lambda c: c["task_id"])
def test_audited_cards_are_diagnostics_not_historical_proof(card):
    args = card["arguments"]
    assert evidence.penalty_candidate(card["name"], args) is card["direct_candidate"]
    events = [event(i, card["name"], args, card["result_excerpt"], {}) for i in range(1, card["copies"] + 1)]
    assert detect(events, {"python_tests"}) == {}
    assert diagnostics(events, python_test_shadow=True)["tool_repeat_shadow_python_test_streaks"] == int(
        card["direct_candidate"]
    )
    assert len(card["source_sha256"]) == len(card["result_sha256"]) == 64


@pytest.mark.parametrize(
    "command",
    [
        "python -c 'print(1)'; python -c 'print(2)'",
        "python -c 'print(1)' && echo done",
        "python -c 'print(1)'\necho done",
        "python <<'EOF'\nprint(1)\nEOF",
        "cat > /tmp/script.py <<'EOF'\nprint(1)\nEOF\npython /tmp/script.py",
        "python -c 'print(1)' > file",
        "python -c 'print(1)' | python -c 'print(2)'",
        "python -c 'print(1)' &",
        "bash -c 'python -c 1'",
        "python $(echo script.py)",
    ],
)
def test_compound_programs_are_outside_initial_penalty(command):
    assert not evidence.penalty_candidate("Bash", {"command": command})


def test_bounds_symlink_missing_capture_and_incomplete_execution(tmp_path):
    (tmp_path / "a").write_text("abc")
    assert evidence.workspace_state(tmp_path, max_bytes=1) is None
    assert evidence.workspace_state(tmp_path, max_files=1) is None
    assert evidence.workspace_state(tmp_path, timeout=0) is None
    (tmp_path / "link").symlink_to(tmp_path / "a")
    assert evidence.workspace_state(tmp_path) is None
    (tmp_path / "link").unlink()
    args = {"command": "python -c 'print(1)'"}
    assert evidence.begin(tmp_path, "Bash", args, "context") is None
    for mode in ("missing", "truncated", "timeout", "decode", "context"):
        with evidence.collect_execution(True):
            start = evidence.begin(tmp_path, "Bash", args, "context")
            if mode != "missing":
                evidence.record_command_result(
                    124 if mode == "timeout" else 0, "\ufffd" if mode == "decode" else "hello"
                )
            if mode == "truncated":
                evidence.mark_incomplete()
            proof = evidence.finish(start, tmp_path, args, "raw", "changed" if mode == "context" else "context")
            assert proof["complete"] is False


def test_disabled_requests_do_not_measure_or_change_execution(tmp_path, monkeypatch):
    executor = session(tmp_path, monkeypatch)
    monkeypatch.setattr(evidence, "workspace_state", lambda *a, **k: pytest.fail("disabled measurement"))
    args = {"command": "python3 -S -c 'print(1)'"}
    text, _, metadata = executor.run_tool("Bash", args)
    assert "1" in text and "repeat_python_test_evidence" not in metadata


def test_public_tool_remote_request_and_proof_roundtrip(tmp_path, monkeypatch):
    executor = session(tmp_path, monkeypatch)
    tool = object.__new__(tools.SWETool)
    tool.config, tool.name = {}, "Bash"
    monkeypatch.setattr(tools.SWETool, "_remote_enabled", lambda self: True)
    monkeypatch.setattr(tools, "_task_from_agent_data", lambda data: {"task_id": "offline"})

    async def post(path, payload, config):
        assert payload["collect_python_test_repeat_evidence"] is True
        text, reward, metadata = executor.run_tool(
            payload["tool"], payload["parameters"], collect_python_test_repeat_evidence=True
        )
        return dict(ok=True, text=text, reward=reward, metadata=metadata)

    monkeypatch.setattr(tools, "_remote_post_json_async", post)
    data = SimpleNamespace(request_id="offline", extra_fields={}, _swe_python_test_repeat_evidence_enabled=True)
    args = {"command": "python3 -S -c 'print(1)'"}
    response, _, metadata = asyncio.run(tool.execute("offline", args, agent_data=data))
    assert evidence.observed_loop_key(event(1, "Bash", args, response, metadata)) is not None


def test_actor_threshold_validation_and_defaults():
    config = actor_config(False)
    assert config.tool_repeat_penalty_python_tests is False
    assert config.tool_repeat_penalty_python_test_protected_calls == 20
    for bad in (19, 0, True, 20.0):
        with pytest.raises(ValueError, match="protected_calls"):
            replace(config, tool_repeat_penalty_python_test_protected_calls=bad)


@pytest.mark.parametrize("card", [c for c in CARDS if c["direct_candidate"]], ids=lambda c: c["task_id"])
def test_audited_command_shapes_with_fresh_disposable_executor_evidence(tmp_path, monkeypatch, card):
    # Minimal stand-ins reproduce the public command behavior, not a historical
    # repository or its hidden tests. All evidence comes from fresh executions.
    files = {
        "pytest.py": "Testdir = object()\n",
        "tests/runtests.py": "print('ImportError: failed to import requested test module'); raise SystemExit(1)\n",
        "tests/roots/test-ext-autodoc/target/non_empty_all.py": "__all__ = ['foo', 'baz']\n",
        "django/__init__.py": "def setup(): raise RuntimeError('Application configuration unavailable')\n",
        "django/core/__init__.py": "",
        "django/core/management/__init__.py": "",
        "django/core/management/base.py": (
            "class Output:\n"
            " def __init__(self, stream): self.stream = stream\n"
            " def write(self, value): self.stream.write(value + '\\n')\n"
            " def flush(self): self.stream.flush()\n"
            "class BaseCommand:\n"
            " def __init__(self, stdout, stderr): self.stdout = Output(stdout)\n"
        ),
    }
    for name, source in files.items():
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source)
    args = copy.deepcopy(card["arguments"])
    args["command"] = args["command"].replace("/testbed", shlex.quote(str(tmp_path)))
    executor = session(tmp_path, monkeypatch)
    executor.run_tool("Bash", args)  # Populate normal Python caches before measurement.
    events = []
    for i in range(1, 23):
        text, _, metadata = executor.run_tool("Bash", args, collect_python_test_repeat_evidence=True)
        events.append(event(i, "Bash", args, text, metadata))
    assert all(e["repeat_python_test_evidence"]["complete"] for e in events)
    assert detect(events, {"python_tests"}) == {21: ["python_tests"], 22: ["python_tests"]}
    if "12209" in card["task_id"]:
        assert events[0]["result"].startswith("exit_code: 0\nImportError:")
    if "16662" in card["task_id"]:
        assert events[0]["result"].startswith("exit_code: 1\nImportError:")


def test_public_run_tests_and_private_evaluator_abstention(tmp_path, monkeypatch):
    executor = session(tmp_path, monkeypatch)
    executor.task.update(execution_phase="rollout", public_test_commands=["python3 -S -c 'print(1)'"])
    events = []
    for i in range(1, 22):
        text, _, metadata = executor.run_tool("run_tests", {}, collect_python_test_repeat_evidence=True)
        events.append(event(i, "run_tests", {}, text, metadata))
    assert detect(events, {"python_tests"}) == {21: ["python_tests"]}
    executor.task.pop("execution_phase")
    monkeypatch.setattr(executor, "_run_selected_commands", lambda *a, **k: ("private", True))
    text, _, metadata = executor.run_tool("run_tests", {}, collect_python_test_repeat_evidence=True)
    assert "repeat_python_test_evidence" not in metadata

    assert not evidence.penalty_candidate("run_tests", {"command": "python -c 1; echo next"})
    assert not evidence.public_test_scope_allowed(
        "run_tests",
        {},
        {
            "execution_phase": "rollout",
            "public_test_commands": ["python -c 1; echo next"],
        },
    )
    assert not evidence.public_test_scope_allowed("run_tests", {}, {"execution_phase": "rollout"})


def test_local_tool_path_produces_same_observed_evidence(tmp_path, monkeypatch):
    tool = object.__new__(tools.SWETool)
    tool.config, tool.name = {}, "Bash"
    monkeypatch.setattr(tools.SWETool, "_remote_enabled", lambda self: False)
    monkeypatch.setattr(tools, "_task_from_agent_data", lambda data: {"task_id": "offline", "use_docker": False})
    monkeypatch.setattr(tools, "_ensure_workspace", lambda *a: tmp_path)
    data = SimpleNamespace(request_id="offline", extra_fields={}, _swe_python_test_repeat_evidence_enabled=True)
    args = {"command": "python3 -S -c 'print(1)'"}
    response, _, metadata = asyncio.run(tool.execute("offline", args, agent_data=data))
    assert evidence.observed_loop_key(event(1, "Bash", args, response, metadata)) is not None


def test_real_stream_truncation_never_becomes_complete_evidence(tmp_path, monkeypatch):
    from recipe.swe_agent import remote_execution_service as service

    executor = session(tmp_path, monkeypatch)
    monkeypatch.setenv("SWE_AGENT_DOCKER_COMMAND_MAX_OUTPUT_BYTES", "1024")

    # Exercise the production Docker stream collector with a local process;
    # this test needs no running Docker daemon and changes no execution logic.
    def run(command, *, timeout, **kwargs):
        result = service._stream_process_result(
            ["/bin/bash", "-lc", command],
            timeout=timeout,
            env=dict(__import__("os").environ),
            label="offline",
            display_command="offline",
            cwd=tmp_path,
            emit_output=False,
        )
        evidence.record_command_result(*result)
        return result

    monkeypatch.setattr(executor, "run", run)
    args = {"command": "python3 -S -c \"print('x'*5000)\""}
    text, _, metadata = executor.run_tool("Bash", args, collect_python_test_repeat_evidence=True)
    assert "truncated" in text
    assert metadata["repeat_python_test_evidence"]["complete"] is False
    assert evidence.observed_loop_key(event(1, "Bash", args, text, metadata)) is None


def test_actual_v1_python_rule_cost_and_success_credit_gate(tmp_path, monkeypatch):
    from tests.trainer.ppo.test_swe_tool_failure_on_cpu import run_v1

    events = live_events(tmp_path, monkeypatch, count=21)
    ids = [10, 11, 12, 13]
    events[-1]["token_span"] = dict(
        version=1, start=1, end=3, token_sha256=token_digest(ids[1:3]), prefix_sha256=token_digest(ids[:3])
    )
    captured, count = capture_events(events, {"python_tests"})
    assert count == len(captured) == 1
    extras = [{"tool_repeat_events": captured, "tool_repeat_eligible_python_tests": 21}] * 2
    baseline, _, _ = run_v1(monkeypatch, False, [1.0, 0.0], extras=extras)
    output, _, metrics = run_v1(monkeypatch, False, [1.0, 0.0], extras=extras, tool_repeat=True, python_tests=True)
    assert metrics["tool_repeat/applied"] == 2
    assert metrics["tool_repeat/eligible_python_tests"] == 42
    for i in range(2):
        weights = output["tool_repeat_weights"].unbind()[i]
        assert weights.sum().item() == pytest.approx(0.02)
        assert weights[[0, 3]].eq(0).all()
        torch.testing.assert_close(
            output["advantages"].unbind()[i][1:3], baseline["advantages"].unbind()[i][1:3].clamp_max(0)
        )
        torch.testing.assert_close(output["returns"].unbind()[i], baseline["returns"].unbind()[i])


@pytest.mark.parametrize("penalty", [False, True])
def test_manager_forwards_internal_evidence_opt_in(tmp_path, monkeypatch, penalty):
    from recipe.swe_agent import remote_execution_service as service

    executor = session(tmp_path, monkeypatch)
    # Manager owns the request-to-session boundary, distinct from SWETool's HTTP
    # payload boundary. Run both without a service, Docker or network access.
    manager = service.ExecutionManager(session_ttl_seconds=60)
    monkeypatch.setattr(manager, "_get_or_create_session", lambda *a, **k: executor)
    response = manager.execute(
        dict(
            request_id="offline",
            task={"task_id": "offline"},
            tool="Bash",
            parameters={"command": "python3 -S -c 'print(1)'"},
            collect_python_test_repeat_evidence=penalty,
        )
    )
    assert response["ok"] is True
    assert ("repeat_python_test_evidence" in response["metadata"]) is penalty

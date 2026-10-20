"""Audited trajectory cards plus actual executor proofs; no historical proof synthesis."""

import asyncio
import copy
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from recipe.swe_agent.remote_execution_service import ExecutionSession
from recipe.swe_agent.tool_repeat_evidence import execute_shell_read, parse_shell_read, proven_extension_key

from recipe.swe_agent import repetition_reward as rr
from recipe.swe_agent import tools
from verl.utils.swe_tool_failure import token_digest
from verl.utils.swe_tool_repeat import build_cost_weights, capture_events, detect, diagnostics, verified_span

FIXTURE = Path(__file__).parent / "fixtures/swe_tool_repeat_extensions_audited_20261020.jsonl.gz"
with gzip.open(FIXTURE, "rt") as stream:
    CARDS = [json.loads(line) for line in stream]


def session(tmp_path, monkeypatch):
    s = ExecutionSession("offline", {"task_id": "offline", "sandbox_backend": "local"}, "offline", tmp_path, False)
    s.workspace = tmp_path
    monkeypatch.setattr(s, "prepare", lambda **kw: "")
    monkeypatch.setattr(s, "_clear_oom_events", lambda: None)
    return s


def event(i, name, args, text, metadata):
    e = dict(
        id=i,
        name=name,
        arguments=args,
        serial=True,
        feedback_delivered=True,
        completed=True,
        observation_status="observed",
        result=rr.public_result(text, metadata),
    )
    for field in ("repeat_shell_read_evidence", "tool_repeat_noop_write"):
        e[field] = metadata.get(field)
    return e


@pytest.mark.parametrize(
    "args",
    [
        {"command": "grep -n 'hello' a.py"},
        {"command": "grep -rn 'hello' . --include=\"*.py\" 2>/dev/null | head -30"},
        {"command": "ls -a ."},
        {"command": "grep -r 'hello' absent/path"},
        {"command": "grep -n 'missing' a.py"},
        {"command": "cd /testbed && grep -n 'hello' a.py | head -2"},
    ],
)
def test_actual_restricted_executor_threshold_and_span(tmp_path, args):
    (tmp_path / "a.py").write_text("hello\n")
    events = []
    for i in range(1, 11):
        output, proof = execute_shell_read(tmp_path, args, mount="/testbed")
        assert proof["complete"], proof
        events.append(event(i, "Bash", args, output, {"repeat_shell_read_evidence": proof}))
    assert detect(events[:9], {"shell_reads"}) == {}
    assert set(detect(events, {"shell_reads"})) == set(range(3, 11))
    assert detect(events, set()) == {}
    ids = list(range(40))
    for i, e in enumerate(events):
        start, end = 2 * i, 2 * i + 1
        e["token_span"] = dict(
            version=1,
            start=start,
            end=end,
            token_sha256=token_digest(ids[start:end]),
            prefix_sha256=token_digest(ids[:end]),
        )
    captured, selected = capture_events(events, {"shell_reads"})
    assert selected == len(captured) == 8
    assert all(verified_span(e, ids, [1] * 40) for e in captured)
    weights, stats = build_cost_weights(
        torch.tensor([ids]),
        torch.ones(1, 40),
        [{"tool_repeat_events": captured}],
        torch.tensor([True]),
        cost=0.02,
        cap=0.10,
        warmup_steps=10,
        step=5,
        enabled_rules={"shell_reads"},
    )
    assert weights.sum().item() == pytest.approx(0.05)
    assert weights[0, :4].eq(0).all()
    captured[0]["start"] += 1
    assert verified_span(captured[0], ids, [1] * 40) is None


@pytest.mark.parametrize(
    "command",
    [
        "python -c 'print(1)'",
        "pytest",
        "grep x a.py; echo x",
        "grep x $(pwd)",
        "grep x a.py > result",
        "grep x a.py &",
        "grep x /etc/passwd",
        "grep x ../a.py",
        "grep x *.py",
        "grep -R x .",
        "grep x a.py | python -c 'print(1)'",
        "ls -la .",
        "grep x a.py\necho hi",
        "grep x a.py | head -30 | head -2",
    ],
)
def test_unsupported_commands_abstain(tmp_path, command):
    assert parse_shell_read(tmp_path, {"command": command}) is None


def test_changed_state_untracked_symlink_and_missing_path_repair(tmp_path):
    path = tmp_path / "a.py"
    path.write_text("hello\n")
    args = {"command": "grep -r hello ."}
    events = []
    for i in range(1, 11):
        if i == 6:
            (tmp_path / ".ignored").write_text("new dependency")
        text, proof = execute_shell_read(tmp_path, args)
        events.append(event(i, "Bash", args, text, {"repeat_shell_read_evidence": proof}))
    assert detect(events, {"shell_reads"}) == {}
    (tmp_path / "link").symlink_to(path)
    assert execute_shell_read(tmp_path, args) is None
    (tmp_path / "link").unlink()
    args = {"command": "grep -r hello missing"}
    first = execute_shell_read(tmp_path, args)[1]
    (tmp_path / "missing").mkdir()
    second = execute_shell_read(tmp_path, args)[1]
    assert first["before"] != second["before"]


def test_remote_noop_write_and_shell_proofs(tmp_path, monkeypatch):
    s = session(tmp_path, monkeypatch)
    monkeypatch.setenv("SWE_AGENT_SHELL_REPEAT_EVIDENCE", "1")
    (tmp_path / "a.py").write_text("hello\n")
    for name, args, rule, count in [
        ("Write", {"file_path": "a.py", "content": "hello\n"}, "noop_writes", 5),
        ("Write", {"file_path": "a.py", "content": "", "append": True}, "noop_writes", 5),
        ("Bash", {"command": "grep -n hello a.py"}, "shell_reads", 10),
    ]:
        events = []
        for i in range(1, count + 1):
            text, _, meta = s.run_tool(
                name,
                tools._normalize_tool_parameters(tools.normalize_tool_name(name), args),
                collect_contract_rejections=True,
            )
            events.append(event(i, name, args, text, meta))
        assert set(detect(events, {rule})) == set(range(3, count + 1))
        assert diagnostics(events)["tool_repeat_eligible_" + rule] == count


@pytest.mark.parametrize(
    "mutation", ["proof", "serial", "delivery", "args", "result", "state", "infra", "ids", "truncated"]
)
def test_proof_barriers(tmp_path, mutation):
    (tmp_path / "a.py").write_text("hello")
    args = {"command": "grep hello a.py"}
    text, proof = execute_shell_read(tmp_path, args)
    events = [
        event(i, "Bash", copy.deepcopy(args), text, {"repeat_shell_read_evidence": copy.deepcopy(proof)})
        for i in range(1, 11)
    ]
    e = events[5]
    if mutation == "proof":
        e["repeat_shell_read_evidence"] = None
    elif mutation == "serial":
        e["serial"] = False
    elif mutation == "delivery":
        e["feedback_delivered"] = False
    elif mutation == "args":
        e["arguments"]["command"] += " -n"
    elif mutation == "result":
        e["result"] += "changed"
    elif mutation == "state":
        e["repeat_shell_read_evidence"]["after"] = "changed"
    elif mutation == "infra":
        e["infrastructure_failure"] = True
    elif mutation == "ids":
        e["id"] = 99
    else:
        e["observation_status"] = "artifact_incomplete"
    assert detect(events, {"shell_reads"}) == {}


def test_write_success_append_refusal_invalid_utf8_not_noop(tmp_path, monkeypatch):
    s = session(tmp_path, monkeypatch)
    for args in [
        dict(path="a", content="hello"),
        dict(path="a", content="hello", append=True),
        dict(path="a", content="other"),
    ]:
        text, _, meta = s.run_tool("write_file", args, collect_contract_rejections=True)
        assert "tool_repeat_noop_write" not in meta
    (tmp_path / "a").write_bytes(b"\xff")
    text, _, meta = s.run_tool("Write", dict(file_path="a", content="\ufffd"), collect_contract_rejections=True)
    assert "tool_repeat_noop_write" not in meta


@pytest.mark.parametrize("card", CARDS, ids=lambda c: f"{c['task_id']}-{c['start_turn']}")
def test_audited_historical_cards_abstain_without_executor_proof(card):
    assert card["copies"] >= 20 and len(card["source_row_sha256"]) == 64
    assert card["call"] in card["output"]
    events = [
        event(i, card["tool"], card["arguments"], "historical decoded result", {}) for i in range(1, card["copies"] + 1)
    ]
    assert detect(events, {"noop_writes", "shell_reads"}) == {}
    assert diagnostics(events, shadow=True)["tool_repeat_shadow_streaks"] == 1
    # Replay audited command grammar against a disposable workspace, never the historical shell program.
    if card["tool"] == "Bash":
        command = card["arguments"].get("command", "")
        if "python" in command:
            assert parse_shell_read(Path("/tmp"), card["arguments"], mount="/testbed") is None


@pytest.mark.parametrize("card", [c for c in CARDS if c["tool"] == "Write"], ids=lambda c: c["task_id"])
def test_audited_write_content_live_executor(tmp_path, monkeypatch, card):
    s = session(tmp_path, monkeypatch)
    args = dict(card["arguments"], file_path="audited.py")
    (tmp_path / "audited.py").write_text(args["content"])
    events = []
    for i in range(1, 6):
        text, _, meta = s.run_tool("Write", args, collect_contract_rejections=True)
        events.append(event(i, "Write", args, text, meta))
    assert set(detect(events, {"noop_writes"})) == {3, 4, 5}


def test_flag_disabled_keeps_ordinary_shell_path(tmp_path, monkeypatch):
    s = session(tmp_path, monkeypatch)
    monkeypatch.delenv("SWE_AGENT_SHELL_REPEAT_EVIDENCE", raising=False)
    monkeypatch.setattr(s, "_run_command", lambda command: "ordinary execution")
    text, _, meta = s.run_tool("Bash", {"command": "grep x a.py"})
    assert text == "ordinary execution"
    assert meta["repeat_shell_read_evidence"]["complete"] is False


@pytest.mark.parametrize(
    "card",
    [c for c in CARDS if c["tool"] == "Bash" and c["arguments"]["command"].lstrip().startswith("grep")],
    ids=lambda c: f"{c['task_id']}-{c['start_turn']}",
)
def test_audited_shell_search_live_executor(tmp_path, card):
    args = card["arguments"]
    plan = parse_shell_read(tmp_path, args, mount="/testbed")
    assert plan is not None
    target = Path(plan["scope"])
    if target.suffix == ".py":
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("def add_line(): pass\nrole = 'example'\n")
    elif "wiktrestart" not in str(target):
        target.mkdir(parents=True, exist_ok=True)
        (target / "example.py").write_text("sample source\n")
    events = []
    for i in range(1, 11):
        text, proof = execute_shell_read(tmp_path, args, mount="/testbed")
        assert proof["complete"], proof
        events.append(event(i, "Bash", args, text, {"repeat_shell_read_evidence": proof}))
    assert set(detect(events, {"shell_reads"})) == set(range(3, 11))


@pytest.mark.parametrize(
    "name,args,rule",
    [
        ("Write", {"file_path": "a.py", "content": "hello\n"}, "noop_writes"),
        ("Bash", {"command": "grep hello a.py"}, "shell_reads"),
    ],
)
def test_public_tool_remote_roundtrip(tmp_path, monkeypatch, name, args, rule):
    s = session(tmp_path, monkeypatch)
    (tmp_path / "a.py").write_text("hello\n")
    monkeypatch.setenv("SWE_AGENT_SHELL_REPEAT_EVIDENCE", "1")
    tool = object.__new__(tools.SWETool)
    tool.config, tool.name = {}, name
    monkeypatch.setattr(tools.SWETool, "_remote_enabled", lambda self: True)
    monkeypatch.setattr(tools, "_task_from_agent_data", lambda data: {"task_id": "offline"})

    async def post(path, payload, config):
        text, reward, metadata = s.run_tool(
            payload["tool"],
            payload["parameters"],
            collect_contract_rejections=payload.get("collect_tool_contract_rejections", False),
        )
        return dict(ok=True, text=text, reward=reward, metadata=metadata)

    monkeypatch.setattr(tools, "_remote_post_json_async", post)
    data = SimpleNamespace(request_id="offline", extra_fields={}, _swe_tool_repeat_enabled=True)
    events = []
    for i in range(1, 11):
        response, _, meta = asyncio.run(tool.execute("offline", args, agent_data=data))
        events.append(event(i, name, args, response, meta))
    assert set(detect(events, {rule})) == set(range(3, 11))


def test_oversized_output_incomplete_not_penalized(tmp_path):
    (tmp_path / "big.txt").write_text("x" * (1024 * 1024 + 100))
    text, proof = execute_shell_read(tmp_path, {"command": "grep x big.txt"})
    assert proof["complete"] is False
    e = event(1, "Bash", {"command": "grep x big.txt"}, text, {"repeat_shell_read_evidence": proof})
    assert proven_extension_key(e, "shell_reads") is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("result", "\ud800"),
        ("name", None),
        ("arguments", None),
        ("repeat_shell_read_evidence", []),
    ],
)
def test_malformed_proof_abstains(tmp_path, field, value):
    (tmp_path / "a").write_text("x")
    args = {"command": "grep x a"}
    text, proof = execute_shell_read(tmp_path, args)
    e = event(1, "Bash", args, text, {"repeat_shell_read_evidence": proof})
    e[field] = value
    assert proven_extension_key(e, "shell_reads") is None


def test_head_only_splits_newlines_and_verification_is_unsupported(tmp_path):
    (tmp_path / "a").write_bytes(b"x\rx\nx\nx\n")
    text, proof = execute_shell_read(tmp_path, {"command": "grep x a | head -2"})
    assert text == "exit_code: 0\nx\rx\nx\n" and proof["complete"]
    assert parse_shell_read(tmp_path, {"command": "grep x a", "verification": True}) is None


def test_v1_new_diagnostics_are_logged_and_padding_excluded(monkeypatch):
    from tests.trainer.ppo.test_swe_tool_failure_on_cpu import run_v1

    _, _, metrics = run_v1(
        monkeypatch,
        False,
        [0.0, 1.0],
        tool_repeat=True,
        tags=[{}, {"is_padding": True}],
        extras=[
            {
                "tool_repeat_eligible_shell_reads": 10,
                "tool_repeat_abstained_proof_missing": 3,
                "tool_repeat_shadow_calls": 20,
            }
        ]
        * 2,
    )
    assert metrics["tool_repeat/eligible_shell_reads"] == 10
    assert metrics["tool_repeat/abstained_proof_missing"] == 3
    assert metrics["tool_repeat/shadow_calls"] == 20


@pytest.mark.parametrize(
    "name,args,rule,count",
    [
        ("Write", {"file_path": "a.py", "content": "hello"}, "noop_writes", 6),
        ("Bash", {"command": "grep hello a.py"}, "shell_reads", 11),
    ],
)
def test_native_loop_forwards_new_proof_and_aligns_only_call_tokens(
    tmp_path,
    monkeypatch,
    name,
    args,
    rule,
    count,
):
    from recipe.swe_agent.agent_loop import SWEToolAgentLoop
    from transformers import AutoTokenizer

    from verl.experimental.agent_loop.tool_agent_loop import AgentState, ToolAgentLoop
    from verl.experimental.agent_loop.tool_parser import Qwen3XMLToolParser

    tokenizer_path = Path(__file__).resolve().parents[4] / "stock-rl-reflect/data/3552-checkpoint"
    if not (tokenizer_path / "tokenizer.json").exists():
        pytest.skip("Local Qwen tokenizer not installed")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    s = session(tmp_path, monkeypatch)
    (tmp_path / "a.py").write_text(args.get("content", "hello\n"))
    monkeypatch.setenv("SWE_AGENT_SHELL_REPEAT_EVIDENCE", "1")
    loop = object.__new__(SWEToolAgentLoop)
    loop.response_length = 100000
    loop.enable_continuous_token = False
    loop.max_assistant_turns = loop.max_user_turns = None
    loop._swe_tool_repeat_enabled = True
    loop._swe_tool_failure_enabled = False
    loop._swe_is_validation = False
    loop._swe_repeat_config = rr.RepeatConfig(mode="off")
    loop._swe_repeat_events, loop._swe_repeat_active = [], []
    loop._trajectory_expired = lambda: False
    loop.tools = {}
    loop.tokenizer, loop.tool_parser = tokenizer, Qwen3XMLToolParser(tokenizer)
    call = "<think>Short thought.</think><tool_call>\n<function=" + name + ">\n"
    call += "".join(f"<parameter={key}>{value}</parameter>\n" for key, value in args.items())
    call += "</function>\n</tool_call>"
    ids = tokenizer.encode(call, add_special_tokens=False)

    async def generate(self, data, params):
        return SimpleNamespace(token_ids=ids, log_probs=[], extra_fields={}, routed_experts=None, num_preempted=0)

    async def deadline(coro):
        return await coro

    async def dispatch(tool_call, kwargs, data):
        parameters = json.loads(tool_call.arguments)
        return s.run_tool(
            name,
            tools._normalize_tool_parameters(tools.normalize_tool_name(name), parameters),
            collect_contract_rejections=True,
        )

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
        for _ in range(count):
            assert await loop._handle_generating_state(data, {}) == AgentState.PROCESSING_TOOLS
            await loop._call_tool(data.tool_calls[0], {}, data)
            data.prompt_ids += [1]
            data.response_mask += [0]
            data.user_turns += 1
        captured, selected = capture_events(loop._swe_repeat_events, {rule})
        assert len(captured) == selected == count - 3
        assert loop._swe_repeat_events[-1]["feedback_delivered"] is False
        response = data.prompt_ids[-len(data.response_mask) :]
        for item in captured:
            assert verified_span(item, response, data.response_mask) is not None
            text = tokenizer.decode(response[item["start"] : item["end"]])
            assert "<tool_call>" in text and "Short thought" not in text
            assert item["proof"]["rules"] == [rule]

    asyncio.run(run())

"""Native generation -> queue -> GRPO gate -> actor gradient checks on CPU."""

import asyncio
import gzip
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tests.trainer.ppo.test_swe_tool_failure_on_cpu import actor_config, actor_data, run_v1
from tests.trainer.ppo.test_swe_tool_failure_on_cpu import event as tool_event
from verl.utils import tensordict_utils as tu
from verl.utils.swe_thinking_repeat import (
    build_cost_weights,
    capture_generation,
    detect,
    prompt_opens_thinking,
    verified_event,
)
from verl.utils.swe_tool_failure import gate_positive_credit, local_ppo_loss, token_digest, trajectory_normalizer

SENTENCE = (
    "I need to trace this branch carefully because the same unresolved question keeps returning, "
    "and I should inspect the actual implementation before choosing a fix."
)


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    path = Path(__file__).resolve().parents[4] / "stock-rl-reflect/data/3552-checkpoint"
    if not (path / "tokenizer.json").exists():
        pytest.skip("Local Qwen tokenizer not installed")
    return AutoTokenizer.from_pretrained(path, local_files_only=True)


def synthetic_event(ids, spans=None, start=0, end=None):
    end = len(ids) if end is None else end
    return dict(
        version=1,
        detector="exact_thinking_v2",
        start=start,
        end=end,
        thinking_start=start,
        thinking_end=end,
        spans=spans or [[start + 1, start + 3]],
        token_sha256=token_digest(ids[start:end]),
        prefix_sha256=token_digest(ids[:end]),
    )


def data_for(tokenizer, text, prefix=None):
    ids = tokenizer.encode(text, add_special_tokens=False)
    prefix = prefix or []
    return SimpleNamespace(
        response_ids=ids, response_mask=[0] * len(prefix) + [1] * len(ids), prompt_ids=[42] + prefix + ids
    )


@pytest.mark.parametrize("implicit,closed", [(False, True), (True, True), (False, False), (True, False)])
def test_native_open_and_closed_thinking(tokenizer, implicit, closed):
    text = ("" if implicit else "<think>") + (SENTENCE + "\n\n") * 24
    if closed:
        text += "</think>Final answer. <tool_call>Do not penalize this.</tool_call>"
    data = data_for(tokenizer, text, [50, 51])
    event, status = capture_generation(tokenizer, data, prompt_thinking=implicit)
    assert status == "flagged_turns" and event
    response = data.prompt_ids[-len(data.response_mask) :]
    assert verified_event(event, response, data.response_mask)
    snippets = [tokenizer.decode(response[a:b]) for a, b in event["spans"]]
    assert all(
        "<think>" not in s and "</think>" not in s and "<tool_call>" not in s and "Final answer" not in s
        for s in snippets
    )
    assert all(a >= 2 for a, _ in event["spans"])  # observation prefix is excluded
    start_text = tokenizer.decode(response[: event["spans"][0][0]])
    assert start_text.count(SENTENCE) >= 2


def test_prompt_thinking_needs_explicit_suffix(tokenizer):
    assert prompt_opens_thinking(tokenizer, tokenizer.encode("assistant\n<think>\n", add_special_tokens=False))
    assert not prompt_opens_thinking(
        tokenizer, tokenizer.encode("<think>quoted earlier</think> assistant", add_special_tokens=False)
    )


@pytest.mark.parametrize(
    "kind", ["no_think", "only_final", "code", "quote", "nested", "observation", "merge", "no_backend", "noncanonical"]
)
def test_native_unsafe_or_nonreasoning_abstains(tokenizer, kind):
    loop = (SENTENCE + "\n\n") * 24
    text = "<think>" + loop + "</think>"
    if kind == "no_think":
        text = loop
    elif kind == "only_final":
        text = "<think>Done.</think>" + loop
    elif kind == "code":
        text = "<think>```text\n" + loop + "```\n</think>"
    elif kind == "quote":
        text = '<think>Fixture: "' + loop + '".</think>'
    data = data_for(tokenizer, text)
    if kind == "observation":
        data.response_mask[3] = 0
    elif kind == "merge":
        data.prompt_ids[-1] += 1
    elif kind == "noncanonical":
        # Same decoded string through a deliberately split BPE representation.
        data.response_ids = sum((tokenizer.encode(c, add_special_tokens=False) for c in text), [])
        data.prompt_ids = [42] + data.response_ids
        data.response_mask = [1] * len(data.response_ids)
    proxy = SimpleNamespace(decode=tokenizer.decode) if kind == "no_backend" else tokenizer
    event, _ = capture_generation(proxy, data, prompt_thinking=kind == "nested")
    assert event is None


def test_native_multilingual_and_inline_quote_safety(tokenizer):
    text = "<think>" + ("控制字符 preserve before_data order。" * 150) + "</think>"
    event, status = capture_generation(tokenizer, data_for(tokenizer, text))
    assert event and status == "flagged_turns"
    s = (
        "I am checking the exact function `checkError` again because I need to know whether "
        "the call receives all of the correct parameters.\n\n"
    )
    data = data_for(tokenizer, "<think>" + s * 24 + "</think>")
    event, _ = capture_generation(tokenizer, data)
    assert event
    assert all("checkError" not in tokenizer.decode(data.response_ids[a:b]) for a, b in event["spans"])


@pytest.mark.parametrize(
    "bad", ["truncated", "token", "prefix", "observation", "overlap", "outside", "version", "empty"]
)
def test_verified_transport_abstains(bad):
    ids, mask = list(range(12)), [1] * 12
    event = synthetic_event(ids, start=2)
    if bad == "truncated":
        ids, mask = ids[:-1], mask[:-1]
    elif bad == "token":
        ids[4] += 100
    elif bad == "prefix":
        ids[0] += 100
    elif bad == "observation":
        mask[3] = 0
    elif bad == "overlap":
        event["spans"] = [[3, 6], [5, 7]]
    elif bad == "outside":
        event["spans"] = [[0, 1]]
    elif bad == "version":
        event["version"] = True
    elif bad == "empty":
        event["spans"] = []
    assert verified_event(event, ids, mask) is None


def test_cost_turn_normalization_cap_dedup_exclusions_and_warmup():
    ids = list(range(120))
    events = [
        synthetic_event(ids, [[start + 1, start + 3], [start + 5, start + 10]], start, start + 12)
        for start in range(0, 120, 12)
    ]
    weights, stats = build_cost_weights(
        torch.tensor([ids] * 4),
        torch.ones(4, 120),
        [
            {"thinking_repeat_events": events[:3]},
            {"thinking_repeat_events": events},
            {"thinking_repeat_events": events},
            {},
        ],
        torch.tensor([True, True, False, True]),
        cost=0.02,
        cap=0.1,
        warmup_steps=10,
        step=5,
        trajectory_ids=["s", "s", "excluded", "clean"],
    )
    assert weights.sum() == pytest.approx(0.05)
    assert stats["applied"] == 10 and stats["abstained"] == 3
    assert weights[2:].eq(0).all()
    assert weights[0, 1:3].sum() / weights[0, 5:10].sum() == pytest.approx(2 / 5)
    assert stats["flagged_trajectories"] == 1
    norm = trajectory_normalizer(torch.tensor([True, True, False, True]), ["s", "s", "excluded", "clean"])
    assert norm.sum() == 2  # includes clean active session
    for cost, cap, warmup in [(0, 0.1, 0), (0.02, 0, 0), (0.02, 0.1, 10)]:
        w, _ = build_cost_weights(
            torch.tensor([ids]),
            torch.ones(1, 120),
            [{"thinking_repeat_events": events}],
            torch.tensor([True]),
            cost=cost,
            cap=cap,
            warmup_steps=warmup,
            step=0,
        )
        assert not w.any()


@pytest.mark.parametrize("scores", [[-0.1, -0.1], [-0.1, 0.0], [0.0, 1.0]])
@pytest.mark.parametrize("tools", [False, True])
def test_actual_v1_independent_score_and_tool_composition(monkeypatch, scores, tools):
    ids = [10, 11, 12, 13]
    extras = [{"thinking_repeat_events": [synthetic_event(ids)], "tool_failure_events": [tool_event(ids)]}] * 2
    baseline, requested, _ = run_v1(monkeypatch, False, scores, extras=extras)
    assert "responses" not in requested[0]
    output, _, metrics = run_v1(monkeypatch, tools, scores, extras=extras, thinking=True)
    assert "thinking_repeat_weights" in output
    for i in range(2):
        w = output["thinking_repeat_weights"].unbind()[i]
        assert w.sum() == pytest.approx(0.02)
        got, before = output["advantages"].unbind()[i], baseline["advantages"].unbind()[i]
        torch.testing.assert_close(got[1:3], before[1:3].clamp_max(0))
        torch.testing.assert_close(got[[0, 3]], before[[0, 3]])
        torch.testing.assert_close(output["returns"].unbind()[i], baseline["returns"].unbind()[i])
    assert ("tool_failure_weights" in output) == tools
    assert metrics["thinking_repeat/applied"] == 2


@pytest.mark.parametrize(
    "tag",
    [
        {"is_padding": True},
        {"train_sample_mask": False},
        {"completion_ratio_cutoff": True},
        {"unverified_trajectory": True},
    ],
)
def test_actual_v1_exclusions(monkeypatch, tag):
    ids = [10, 11, 12, 13]
    extras = [{"thinking_repeat_events": [synthetic_event(ids)]}] * 2
    output, _, _ = run_v1(monkeypatch, False, [0.0, 1.0], tags=[tag, {}], extras=extras, thinking=True)
    assert output["thinking_repeat_weights"].unbind()[0].eq(0).all()


def test_actual_v1_infrastructure_and_post_broadcast_gate(monkeypatch):
    ids = [10, 11, 12, 13]
    extras = [
        {"thinking_repeat_events": [synthetic_event(ids)], "reward_extra_info": {"observed_infrastructure_failure": 1}},
        {},
    ]
    output, _, _ = run_v1(monkeypatch, False, [0.0, 1.0], extras=extras, thinking=True)
    assert output["thinking_repeat_weights"].unbind()[0].eq(0).all()
    extras = [{"thinking_repeat_events": [synthetic_event(ids)]}, {}, {}]
    output, _, _ = run_v1(
        monkeypatch, False, [0.0, 1.0, 0.0], keys=["g_0_0", "g_0_1", "g_1_0"], extras=extras, thinking=True
    )
    assert output["advantages"].unbind()[0][1:3].eq(0).all()
    assert output["advantages"].unbind()[1].gt(0).all()


def test_actual_actor_disabled_exact_parity_and_additive_gradient():
    from verl.workers.utils.losses import ppo_loss

    data = actor_data()
    x = torch.zeros(12, requires_grad=True)
    base, _ = ppo_loss(actor_config(), {"log_probs": x}, data)
    base.backward()
    grad = x.grad.clone()
    data["thinking_repeat_weights"] = torch.tensor([[0.0, 0.02, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    tu.assign_non_tensor(data, thinking_repeat_active_global_count=2)
    x.grad = None
    disabled, _ = ppo_loss(actor_config(), {"log_probs": x}, data)
    disabled.backward()
    torch.testing.assert_close(disabled, base, rtol=0, atol=0)
    torch.testing.assert_close(x.grad, grad, rtol=0, atol=0)
    config = replace(actor_config(), thinking_repeat_penalty_enabled=True)
    x.grad = None
    enabled, metrics = ppo_loss(config, {"log_probs": x}, data)
    enabled.backward()
    assert (enabled - base).item() == pytest.approx(0.01, abs=1e-7)
    assert (x.grad - grad).sum().item() == pytest.approx(0.01, abs=1e-7)
    assert "actor/thinking_repeat_loss" in metrics
    thinking = enabled.detach()
    data["tool_failure_weights"] = data["thinking_repeat_weights"].clone()
    tu.assign_non_tensor(data, tool_failure_active_global_count=2)
    config = replace(config, tool_failure_penalty_enabled=True)
    both, metrics = ppo_loss(config, {"log_probs": x}, data)
    assert (both - thinking).item() == pytest.approx(0.01, abs=1e-7)
    assert "actor/tool_failure_loss" in metrics


def test_native_capture_to_actor_gradient(tokenizer):
    ids_text = "<think>" + (SENTENCE + "\n\n") * 24 + "</think>Final answer."
    data = data_for(tokenizer, ids_text)
    event, _ = capture_generation(tokenizer, data)
    ids = torch.tensor([data.response_ids])
    mask = torch.ones_like(ids)
    weights, _ = build_cost_weights(
        ids,
        mask,
        [{"thinking_repeat_events": [event]}],
        torch.tensor([True]),
        cost=0.02,
        cap=0.1,
        warmup_steps=0,
        step=1,
    )
    advantages = gate_positive_credit(torch.ones_like(weights), weights)
    assert advantages[weights > 0].eq(0).all() and advantages[weights == 0].eq(1).all()
    logp = torch.zeros_like(weights, requires_grad=True)
    loss = local_ppo_loss(logp, logp.detach(), weights, mask, actor_config(), active_global_count=1)
    loss.backward()
    assert logp.grad[weights > 0].gt(0).all()
    assert logp.grad[weights == 0].eq(0).all()
    assert weights.sum() == pytest.approx(0.02)


def test_frozen_offline_and_native_predictions_unchanged(tokenizer):
    from tokenizers import Tokenizer

    root = Path(__file__).resolve().parents[3]
    directory = root / "analysis/thinking_repeat_audited_20261019"
    if not directory.exists():
        pytest.skip("Local audited corpus unavailable")
    tok = Tokenizer.from_file(str(root.parent / "stock-rl-reflect/data/3552-checkpoint/tokenizer.json"))
    cached = {}
    for partition in ["development", "holdout"]:
        for line in (directory / "results" / f"identity_v2_{partition}.jsonl").open():
            row = json.loads(line)
            cached[row["id"]] = row["result"]
    for line in gzip.open(directory / "frozen_cases.jsonl.gz", "rt"):
        case = json.loads(line)
        result = detect(case["text"], tok)
        expected = cached[case["id"]]
        assert result["spans"] == expected["spans"], case["id"]
        assert result["penalized_tokens"] == expected["penalized_tokens"], case["id"]
        data = data_for(tokenizer, "<think>" + case["text"] + "</think>")
        event, _ = capture_generation(tokenizer, data)
        assert bool(event) == expected["matched"], case["id"]


@pytest.mark.parametrize("enabled", [False, True])
def test_loop_captures_native_events_without_tool_interference(monkeypatch, tokenizer, enabled):
    from recipe.swe_agent.agent_loop import SWEToolAgentLoop

    from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop
    from verl.experimental.agent_loop.tool_parser import Qwen3XMLToolParser

    loop = object.__new__(SWEToolAgentLoop)
    loop.response_length = 10000
    loop.enable_continuous_token = False
    loop.max_assistant_turns = loop.max_user_turns = None
    loop._swe_thinking_repeat_enabled = enabled
    loop._swe_tool_failure_enabled = False
    loop._swe_is_validation = False
    loop._trajectory_expired = lambda: False
    loop.tools = {}
    loop.tokenizer, loop.tool_parser = tokenizer, Qwen3XMLToolParser(tokenizer)
    text = "<think>" + (SENTENCE + "\n\n") * 24 + "</think><tool_call>\n<function=Unknown>\n</function>\n</tool_call>"
    ids = tokenizer.encode(text, add_special_tokens=False)

    async def generate(self, data, params):
        return SimpleNamespace(token_ids=ids, log_probs=[], extra_fields={}, routed_experts=None, num_preempted=0)

    async def deadline(coro):
        return await coro

    loop._await_with_trajectory_deadline = deadline
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
    asyncio.run(loop._handle_generating_state(data, {}))
    assert data.tool_calls[0].name == "Unknown"
    if enabled:
        assert data.extra_fields["thinking_repeat_flagged_turns"] == 1
        assert verified_event(data.extra_fields["thinking_repeat_events"][0], ids, data.response_mask)
    else:
        assert not any(k.startswith("thinking_repeat_") for k in data.extra_fields)


def test_runtime_switch_and_config_defaults(monkeypatch):
    from omegaconf import OmegaConf

    from verl.trainer.constants_ppo import get_ppo_ray_runtime_env

    assert not actor_config().thinking_repeat_penalty_enabled
    monkeypatch.setenv("SWE_AGENT_REPEAT_REWARD_MODE", "off")
    monkeypatch.setenv("SWE_AGENT_TRAINING_THINKING_REPEAT_PENALTY_ENABLED", "1")
    cfg = OmegaConf.create({"actor_rollout_ref": {"actor": {"thinking_repeat_penalty_enabled": False}}})
    key = "SWE_AGENT_TRAINING_THINKING_REPEAT_PENALTY_ENABLED"
    assert get_ppo_ray_runtime_env(cfg)["env_vars"][key] == "0"
    cfg.actor_rollout_ref.actor.thinking_repeat_penalty_enabled = True
    assert get_ppo_ray_runtime_env(cfg)["env_vars"][key] == "1"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"thinking_repeat_penalty_cost": float("nan")},
        {"thinking_repeat_penalty_cap": -1},
        {"thinking_repeat_penalty_warmup_steps": -1},
    ],
)
def test_invalid_enabled_settings_rejected(kwargs):
    from verl.workers.config.actor import ActorConfig

    with pytest.raises(ValueError, match="thinking-repeat"):
        ActorConfig(strategy="fsdp2", rollout_n=1, use_dynamic_bsz=True, thinking_repeat_penalty_enabled=True, **kwargs)


def test_cpu_dp_and_microbatch_partition_gradient_invariance():
    ids = torch.arange(12).repeat(4, 1)
    mask = torch.ones_like(ids)
    extras = [{"thinking_repeat_events": [synthetic_event(row)]} if i < 3 else {} for i, row in enumerate(ids.tolist())]
    weights, _ = build_cost_weights(
        ids, mask, extras, torch.ones(4, dtype=torch.bool), cost=0.02, cap=0.1, warmup_steps=0, step=1
    )
    x = torch.zeros_like(weights, requires_grad=True)
    config = actor_config()
    full = local_ppo_loss(x, x.detach(), weights, mask, config, active_global_count=4)
    full.backward()
    expected = x.grad.clone()
    x.grad = None
    for i in range(4):
        # Simulate two DP ranks, each with two single-row microbatches. DP later
        # averages gradients, hence division by two on the simulated losses.
        loss = local_ppo_loss(
            x[i : i + 1],
            x[i : i + 1].detach(),
            weights[i : i + 1],
            mask[i : i + 1],
            config,
            active_global_count=4,
            dp_size=2,
        )
        (loss / 2).backward()
    torch.testing.assert_close(x.grad, expected, rtol=0, atol=0)


def test_missing_metadata_does_not_change_v1_advantages(monkeypatch):
    baseline, _, _ = run_v1(monkeypatch, False, [0.0, 1.0], extras=[{}, {}])
    enabled, _, metrics = run_v1(monkeypatch, False, [0.0, 1.0], extras=[{}, {}], thinking=True)
    for got, expected in zip(enabled["advantages"].unbind(), baseline["advantages"].unbind(), strict=True):
        torch.testing.assert_close(got, expected, rtol=0, atol=0)
    assert metrics["thinking_repeat/applied"] == 0
    assert all(w.eq(0).all() for w in enabled["thinking_repeat_weights"].unbind())

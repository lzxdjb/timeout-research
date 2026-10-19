"""Conservative detection-to-local-loss checks, including the real Qwen tokenizer."""

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl.utils.swe_tool_failure import (
    align_single_call,
    build_cost_weights,
    gate_positive_credit,
    local_ppo_loss,
    rejection_metadata,
    token_digest,
    verified_span,
)

CONFIG = SimpleNamespace(clip_ratio=0.2, clip_ratio_low=0.2, clip_ratio_high=0.2, clip_ratio_c=3.0)


def event(ids, start=1, end=3):
    return dict(
        version=1,
        start=start,
        end=end,
        token_sha256=token_digest(ids[start:end]),
        prefix_sha256=token_digest(ids[:end]),
        proof=rejection_metadata("invalid_argument")["tool_contract_rejection"],
    )


def weights_for(ids, masks, extras, active=None, **kwargs):
    return build_cost_weights(
        torch.tensor(ids),
        torch.tensor(masks),
        extras,
        torch.tensor(active if active is not None else [True] * len(ids)),
        **dict(cost=0.02, cap=0.10, warmup_steps=0, step=1, **kwargs),
    )


@pytest.mark.parametrize("score", [-0.1, 0.0, 1.0])
def test_every_terminal_score_gets_independent_local_cost(score):
    ids = [10, 11, 12, 13]
    scores = torch.tensor([score])
    before = scores.clone()
    weights, stats = weights_for([ids], [[1] * 4], [{"tool_failure_events": [event(ids)]}])
    assert weights.sum().item() == pytest.approx(0.02)
    assert stats["applied"] == 1
    torch.testing.assert_close(scores, before)
    logp = torch.zeros(1, 4, requires_grad=True)
    loss = local_ppo_loss(logp, logp.detach(), weights, torch.ones_like(weights), CONFIG, active_global_count=1)
    loss.backward()
    assert logp.grad[0, 1:3].gt(0).all()  # gradient descent reduces rejected-call likelihood
    assert logp.grad[0, [0, 3]].eq(0).all()


@pytest.mark.parametrize("bad", ["truncated", "shifted", "observation", "proof", "version", "prefix"])
def test_uncertain_span_abstains(bad):
    ids, mask = [10, 11, 12, 13], [1] * 4
    e = event(ids)
    if bad == "truncated":
        ids, mask = ids[:2], mask[:2]
    elif bad == "shifted":
        ids[1] = 99
    elif bad == "observation":
        mask[1] = 0
    elif bad == "proof":
        e["proof"]["attribution"] = "infrastructure"
    elif bad == "version":
        e["proof"]["version"] = 2
    elif bad == "prefix":
        ids[0] = 99  # catches even identical call text moved by a template merge
    w, stats = weights_for([ids], [mask], [{"tool_failure_events": [e]}])
    assert not w.any() and stats["abstained"] == 1


def test_no_metadata_old_executor_exclusion_and_duplicate_abstention():
    ids = [10, 11, 12, 13]
    e = event(ids)
    w, stats = weights_for(
        [ids] * 3,
        [[1] * 4] * 3,
        [{}, {"tool_failure_events": [e]}, {"tool_failure_events": [e, e]}],
        [True, False, True],
    )
    assert not w[:2].any()
    assert w[2].sum() == pytest.approx(0.02)
    assert stats["active"] == 2 and stats["applied"] == 1 and stats["abstained"] == 1


def test_per_call_mean_cap_and_warmup():
    ids = list(range(25))
    spans = [(0, 1), (1, 3), (3, 6), (6, 10), (10, 15), (15, 25)]
    e = [event(ids, a, b) for a, b in spans]
    w, _ = build_cost_weights(
        torch.tensor([ids]),
        torch.ones(1, 25),
        [{"tool_failure_events": e}],
        torch.tensor([True]),
        cost=0.02,
        cap=0.1,
        warmup_steps=10,
        step=5,
    )
    assert w.sum() == pytest.approx(0.05)
    for a, b in spans:
        assert w[0, a:b].sum() == pytest.approx(0.05 / 6)


def test_positive_credit_gate_uses_advantage_not_reward():
    adv = torch.tensor([[2.0, 2.0, 2.0, 2.0], [-2.0] * 4, [0.0] * 4])
    w = torch.tensor([[0.0, 0.01, 0.01, 0.0]] * 3)
    got = gate_positive_credit(adv, w)
    torch.testing.assert_close(got[0], torch.tensor([2.0, 0.0, 0.0, 2.0]))
    torch.testing.assert_close(got[1:], adv[1:])


@pytest.mark.parametrize("dp", [1, 2, 4])
@pytest.mark.parametrize("micro", [1, 2])
def test_global_active_normalization_gradient_invariant(dp, micro):
    old = torch.zeros(8, 10)
    weights = torch.zeros_like(old)
    weights[0, 1:3] = 0.01
    weights[4, 1:5] = 0.005
    mask = torch.ones_like(old)
    mask[7] = 0  # inactive; unflagged active rows still count in denominator
    x = torch.zeros_like(old, requires_grad=True)
    truth = local_ppo_loss(x, old, weights, mask, CONFIG, active_global_count=7)
    truth.backward()
    truth_grad = x.grad.clone()
    y = torch.zeros_like(old, requires_grad=True)
    losses = []
    for rank in range(dp):
        start, end = rank * 8 // dp, (rank + 1) * 8 // dp
        for chunk in torch.arange(start, end).chunk(micro):
            losses.append(
                local_ppo_loss(
                    y[chunk], old[chunk], weights[chunk], mask[chunk], CONFIG, active_global_count=7, dp_size=dp
                )
                / dp
            )
    combined = sum(losses)
    combined.backward()
    torch.testing.assert_close(combined, truth)
    torch.testing.assert_close(y.grad, truth_grad)


@pytest.mark.parametrize("ratio,expected_gradient", [(0.5, 0), (1.0, 1), (1.5, 1), (4.0, 0)])
def test_clipping_and_importance_weights(ratio, expected_gradient):
    logp = torch.tensor([[ratio]], dtype=torch.float64).log().requires_grad_()
    loss = local_ppo_loss(
        logp,
        torch.zeros_like(logp),
        torch.tensor([[0.02]]),
        torch.ones_like(logp),
        CONFIG,
        active_global_count=2,
        rollout_is_weights=torch.tensor([[0.5]]),
    )
    loss.backward()
    assert bool(logp.grad.item() > 0) == bool(expected_gradient)
    expected = min(3.0, max(ratio, 0.8)) * 0.02 * 0.5 / 2
    assert loss.item() == pytest.approx(expected)


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    path = Path(__file__).resolve().parents[4] / "stock-rl-reflect/data/3552-checkpoint"
    if not (path / "tokenizer.json").exists():
        pytest.skip("Local Qwen tokenizer not installed")
    return AutoTokenizer.from_pretrained(path, local_files_only=True)


@pytest.mark.parametrize("kind", ["valid", "unknown", "truncated", "multi", "quoted", "observation", "merge"])
def test_real_qwen_parser_and_token_alignment(tokenizer, kind):
    from verl.experimental.agent_loop.tool_parser import Qwen3XMLToolParser

    parser = Qwen3XMLToolParser(tokenizer)
    name = "NoSuchTool" if kind == "unknown" else "Read"
    text = (
        f"<think>Inspect file.</think>\n<tool_call>\n<function={name}>\n"
        "<parameter=file_path>a.py</parameter>\n<parameter=limit>0</parameter>\n</function>\n</tool_call>"
    )
    if kind == "truncated":
        text = text.removesuffix("</tool_call>")
    elif kind == "multi":
        text += text
    elif kind == "quoted":
        text = "<think>Example: <tool_call>literal</tool_call></think>" + text
    ids = tokenizer.encode(text, add_special_tokens=False)

    async def run():
        _, calls = await parser.extract_tool_calls(ids, [])
        prior = tokenizer.encode("prior tool observation\n", add_special_tokens=False)
        data = SimpleNamespace(
            response_ids=ids,
            prompt_ids=[999] + prior + ids,
            response_mask=[0] * len(prior) + [1] * len(ids),
            tool_calls=calls,
        )
        if kind == "observation":
            data.response_mask[-2] = 0
        elif kind == "merge":
            data.prompt_ids[-1] = 999
        aligned = await align_single_call(tokenizer, parser, data, [])
        if kind in {"valid", "unknown"}:
            assert aligned is not None
            aligned["proof"] = rejection_metadata("invalid_argument")["tool_contract_rejection"]
            assert verified_span(aligned, prior + ids, data.response_mask)
            span = tokenizer.decode((prior + ids)[aligned["start"] : aligned["end"]])
            assert span.startswith("<tool_call>") and span.endswith("</tool_call>")
        else:
            assert aligned is None

    asyncio.run(run())


def test_generic_dispatch_only_proves_known_rejection_branches():
    from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop
    from verl.experimental.agent_loop.tool_parser import FunctionCall

    loop = object.__new__(ToolAgentLoop)
    loop.tools = {"Read": object()}

    async def run():
        for enabled in (False, True):
            data = SimpleNamespace(_swe_tool_failure_enabled=enabled)
            for name, arguments, expected in [
                ("Missing", "{}", "unknown_tool"),
                ("Read", "{", "invalid_json"),
                ("Read", "[]", None),
                ("Read", "{}", None),
            ]:
                _, _, metadata = await loop._call_tool(FunctionCall(name=name, arguments=arguments), {}, data)
                assert metadata.get("tool_contract_rejection", {}).get("code") == (expected if enabled else None)

    asyncio.run(run())


def actor_config(enabled=False):
    from verl.workers.config.actor import ActorConfig

    return ActorConfig(strategy="fsdp2", rollout_n=1, use_dynamic_bsz=True, tool_failure_penalty_enabled=enabled)


def actor_data():
    # Nested responses preserve observation positions, as in the V1 queue.
    prompts = torch.nested.as_nested_tensor([torch.tensor([1, 2])] * 2, layout=torch.jagged)
    responses = torch.nested.as_nested_tensor([torch.tensor([3, 4, 5, 6])] * 2, layout=torch.jagged)
    mask = torch.nested.as_nested_tensor([torch.tensor([1, 1, 0, 1])] * 2, layout=torch.jagged)
    data = TensorDict(
        dict(
            prompts=prompts,
            responses=responses,
            response_mask=mask,
            old_log_probs=torch.zeros(2, 4),
            advantages=torch.tensor([[1.0, 0.0, 0.0, 1.0], [-1.0, -1.0, 0.0, -1.0]]),
        ),
        batch_size=2,
    )
    tu.assign_non_tensor(data, dp_size=1, batch_num_tokens=6, global_batch_size=2)
    return data


def test_actual_actor_disabled_parity_and_enabled_gradient():
    from verl.workers.utils.losses import ppo_loss

    data = actor_data()
    # Flat full-sequence output: no_padding_2_padding extracts response logprobs.
    x = torch.zeros(12, requires_grad=True)
    baseline, _ = ppo_loss(actor_config(), {"log_probs": x}, data)
    baseline.backward()
    before = x.grad.clone()
    extra = data.clone()
    extra["tool_failure_weights"] = torch.tensor([[0.0, 0.02, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    tu.assign_non_tensor(extra, tool_failure_active_global_count=2)
    x.grad = None
    disabled, _ = ppo_loss(actor_config(), {"log_probs": x}, extra)
    disabled.backward()
    torch.testing.assert_close(disabled, baseline, rtol=0, atol=0)
    torch.testing.assert_close(x.grad, before, rtol=0, atol=0)
    x.grad = None
    enabled, metrics = ppo_loss(actor_config(True), {"log_probs": x}, extra)
    enabled.backward()
    assert (enabled - baseline).item() == pytest.approx(0.01, abs=1e-7)
    assert "actor/tool_failure_loss" in metrics
    assert (x.grad - before).sum() == pytest.approx(0.01, abs=1e-7)


def run_v1(monkeypatch, enabled, scores, keys=None, tags=None, extras=None, *, thinking=False):
    from transfer_queue import KVBatchMeta

    from verl.trainer.ppo.v1 import trainer_base

    class Trainer(trainer_base.PPOTrainer):
        def on_step_end(self):
            pass

        def on_sample_end(self):
            pass

    trainer = object.__new__(Trainer)
    trainer.timeout_prediction_enabled = False
    trainer.global_steps = 10
    trainer.config = OmegaConf.create(
        dict(
            algorithm=dict(adv_estimator="grpo", use_kl_in_reward=False, gamma=1.0, lam=1.0),
            actor_rollout_ref=dict(
                rollout=dict(n=2),
                actor=dict(
                    tool_failure_penalty_enabled=enabled,
                    tool_failure_penalty_cost=0.02,
                    tool_failure_penalty_cap=0.1,
                    tool_failure_penalty_warmup_steps=10,
                    thinking_repeat_penalty_enabled=thinking,
                    thinking_repeat_penalty_cost=0.02,
                    thinking_repeat_penalty_cap=0.1,
                    thinking_repeat_penalty_warmup_steps=10,
                ),
            ),
        )
    )
    ids = [10, 11, 12, 13]
    keys = keys or [f"g_{i}_0" for i in range(len(scores))]
    tags = tags or [{} for _ in scores]
    extras = extras or [
        {
            "tool_failure_events": [event(ids)],
            "reward_extra_info": {
                "reward_eligible": 0,
                "hidden_reward_skipped": True,
                "observed_infrastructure_failure": 0,
            },
        }
        for _ in scores
    ]
    queue = tu.get_tensordict(
        dict(
            uid=torch.zeros(len(scores), dtype=torch.long),
            responses=torch.nested.as_nested_tensor([torch.tensor(ids)] * len(scores), layout=torch.jagged),
            response_mask=torch.nested.as_nested_tensor([torch.ones(4)] * len(scores), layout=torch.jagged),
            rm_scores=torch.nested.as_nested_tensor(
                [torch.tensor([0.0, 0.0, 0.0, s]) for s in scores], layout=torch.jagged
            ),
            extra_fields=extras,
        )
    )
    captured, requested = {}, []

    def get(**kw):
        fields = kw["select_fields"]
        requested.append(fields)
        if len(fields) == 1:
            raise KeyError(fields[0])
        return queue.select(*(f for f in fields if f in queue)).clone()

    def put(**kw):
        captured.update(kw["fields"])
        return batch

    batch = KVBatchMeta(partition_id="train", keys=keys, tags=tags)
    monkeypatch.setenv("SWE_AGENT_TASK_FILTER_TRAINING_ENABLED", "0")
    monkeypatch.setattr(trainer_base.tq, "kv_batch_get", get)
    monkeypatch.setattr(trainer_base.tq, "kv_batch_put", put)
    metrics = {}
    trainer._compute_advantage(batch, metrics)
    return captured, requested, metrics


@pytest.mark.parametrize("scores", [[-0.1, -0.1], [-0.1, 0.0], [0.0, 1.0]])
def test_actual_v1_grpo_rewards_and_local_gate(monkeypatch, scores):
    disabled, fields, _ = run_v1(monkeypatch, False, scores)
    assert "responses" not in fields[0] and "tool_failure_weights" not in disabled
    enabled, _, metrics = run_v1(monkeypatch, True, scores)
    assert enabled["tool_failure_active"].tolist() == [True, True]
    for i in range(2):
        base = disabled["advantages"].unbind()[i]
        got = enabled["advantages"].unbind()[i]
        torch.testing.assert_close(got[[0, 3]], base[[0, 3]])
        torch.testing.assert_close(got[1:3], base[1:3].clamp_max(0))
        assert enabled["tool_failure_weights"].unbind()[i].sum() == pytest.approx(0.02)
    assert metrics["tool_failure/applied"] == 2


@pytest.mark.parametrize(
    "tag",
    [
        {"is_padding": True},
        {"train_sample_mask": False},
        {"completion_ratio_cutoff": True},
        {"unverified_trajectory": True},
    ],
)
def test_actual_v1_exclusions_never_get_local_cost(monkeypatch, tag):
    output, _, _ = run_v1(monkeypatch, True, [0.0, 1.0], tags=[tag, {}])
    assert not output["tool_failure_active"][0]
    assert output["tool_failure_weights"].unbind()[0].eq(0).all()


def test_actual_v1_infrastructure_excluded_even_without_task_filter(monkeypatch):
    ids = [10, 11, 12, 13]
    extras = [
        {"tool_failure_events": [event(ids)], "reward_extra_info": {"observed_infrastructure_failure": v}}
        for v in [1, 0]
    ]
    output, _, _ = run_v1(monkeypatch, True, [0.0, 1.0], extras=extras)
    assert output["tool_failure_active"].tolist() == [False, True]
    assert output["tool_failure_weights"].unbind()[0].eq(0).all()


def test_v1_positive_credit_gate_runs_after_session_broadcast(monkeypatch):
    # Only an earlier output has a rejected call; final output earns repair credit.
    ids = [10, 11, 12, 13]
    extras = [{"tool_failure_events": [event(ids)]}, {}, {}]
    output, _, _ = run_v1(monkeypatch, True, [0.0, 1.0, 0.0], keys=["g_0_0", "g_0_1", "g_1_0"], extras=extras)
    assert output["advantages"].unbind()[0][1:3].eq(0).all()
    assert output["advantages"].unbind()[1].gt(0).all()
    assert output["tool_failure_weights"].unbind()[1].eq(0).all()


def test_runtime_switch_independent_of_repetition(monkeypatch):
    from verl.trainer.constants_ppo import get_ppo_ray_runtime_env

    monkeypatch.setenv("SWE_AGENT_REPEAT_REWARD_MODE", "off")
    monkeypatch.setenv("SWE_AGENT_TRAINING_TOOL_FAILURE_PENALTY_ENABLED", "1")
    cfg = OmegaConf.create({"actor_rollout_ref": {"actor": {"tool_failure_penalty_enabled": False}}})
    assert get_ppo_ray_runtime_env(cfg)["env_vars"]["SWE_AGENT_TRAINING_TOOL_FAILURE_PENALTY_ENABLED"] == "0"
    cfg.actor_rollout_ref.actor.tool_failure_penalty_enabled = True
    assert get_ppo_ray_runtime_env(cfg)["env_vars"]["SWE_AGENT_TRAINING_TOOL_FAILURE_PENALTY_ENABLED"] == "1"


def test_session_cap_deduplicates_prefix_outputs_and_normalizes_once():
    from verl.utils.swe_tool_failure import trajectory_normalizer

    ids = list(range(30))
    first = [event(ids, i, i + 1) for i in range(3)]
    second = first + [event(ids, i, i + 1) for i in range(3, 9)]
    w, stats = build_cost_weights(
        torch.tensor([ids, ids]),
        torch.ones(2, 30),
        [{"tool_failure_events": first}, {"tool_failure_events": second}],
        torch.tensor([True, True]),
        trajectory_ids=["session", "session"],
        cost=0.02,
        cap=0.1,
        warmup_steps=0,
        step=1,
    )
    assert w.sum() == pytest.approx(0.1)
    assert stats["applied"] == 9 and stats["abstained"] == 3
    norm = trajectory_normalizer(torch.tensor([True, True, True, False]), ["a", "a", "b", "b"])
    torch.testing.assert_close(norm, torch.tensor([0.5, 0.5, 1.0, 0.0]))


@pytest.mark.parametrize("enabled", [False, True])
def test_swe_loop_capture_and_dispatch_event_transport(monkeypatch, tokenizer, enabled):
    from recipe.swe_agent.agent_loop import SWEToolAgentLoop

    from verl.experimental.agent_loop.tool_agent_loop import AgentState, ToolAgentLoop
    from verl.experimental.agent_loop.tool_parser import Qwen3XMLToolParser

    loop = object.__new__(SWEToolAgentLoop)
    loop.response_length = 10000
    loop.enable_continuous_token = False
    loop.max_assistant_turns = loop.max_user_turns = None
    loop._swe_tool_failure_enabled = enabled
    loop._swe_is_validation = False
    loop._trajectory_expired = lambda: False
    loop.tools = {}
    loop.tokenizer, loop.tool_parser = tokenizer, Qwen3XMLToolParser(tokenizer)
    text = "<think>Try tool.</think><tool_call>\n<function=Unknown>\n</function>\n</tool_call>"
    ids = tokenizer.encode(text, add_special_tokens=False)

    async def generate(self, data, params):
        return SimpleNamespace(token_ids=ids, log_probs=[], extra_fields={}, routed_experts=None, num_preempted=0)

    async def deadline(coro):
        return await coro

    async def dispatch(call, kwargs, data):
        return await ToolAgentLoop._call_tool(loop, call, kwargs, data)

    loop._await_with_trajectory_deadline = deadline
    loop._call_tool_with_repeat_judging = dispatch
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
        assert await loop._handle_generating_state(data, {}) == AgentState.PROCESSING_TOOLS
        await loop._call_tool(data.tool_calls[0], {}, data)
        if enabled:
            e = data.extra_fields["tool_failure_events"][0]
            assert e["proof"]["code"] == "unknown_tool"
            assert verified_span(e, ids, data.response_mask)
            assert data.extra_fields["tool_failure_outcome_metadata_calls"] == 1
        else:
            assert not any(key.startswith("tool_failure_") for key in data.extra_fields)
            assert not hasattr(data, "_swe_tool_failure_pending")

    asyncio.run(run())

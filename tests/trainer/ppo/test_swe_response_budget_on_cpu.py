"""SWE split budgets must constrain generation without changing shared storage."""

import asyncio
from types import SimpleNamespace

import pytest
from recipe.swe_agent.agent_loop import SWEToolAgentLoop

from verl.experimental.agent_loop.tool_agent_loop import AgentState, ToolAgentLoop
from verl.tools.schemas import ToolResponse
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.utils.swe_response_budget import RESPONSE_LENGTH_ENV_NAMES, resolve_swe_response_lengths

TRAIN, VAL = RESPONSE_LENGTH_ENV_NAMES


@pytest.fixture(autouse=True)
def isolate_budget_environment(monkeypatch):
    for name in RESPONSE_LENGTH_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SWE_AGENT_EXECUTION_BACKEND", "local")
    monkeypatch.setenv("SWE_AGENT_REPEAT_REWARD_MODE", "off")
    monkeypatch.setenv("SWE_AGENT_CUTOFF_DEBUG", "0")
    monkeypatch.setenv("SWE_AGENT_HARNESS_PROFILE", "baseline")


@pytest.mark.parametrize(
    "overrides,expected",
    [
        ({}, (100, 100)),
        ({TRAIN: ""}, (100, 100)),
        ({TRAIN: "64"}, (64, 100)),
        ({VAL: "80"}, (100, 80)),
        ({TRAIN: "64", VAL: "96"}, (64, 96)),
    ],
)
def test_independent_fallbacks(overrides, expected):
    assert resolve_swe_response_lengths(100, overrides) == expected


@pytest.mark.parametrize("name", RESPONSE_LENGTH_ENV_NAMES)
@pytest.mark.parametrize("value", ["0", "-1", "1.5", "True", "1e2", "101"])
def test_invalid_overrides_fail_before_generation(name, value):
    with pytest.raises(ValueError, match=name):
        resolve_swe_response_lengths(100, {name: value})


def test_overrides_reach_ray_workers_and_reset_when_removed(monkeypatch):
    monkeypatch.setenv(TRAIN, "65536")
    monkeypatch.setenv(VAL, "253952")
    env = get_ppo_ray_runtime_env()["env_vars"]
    assert env[TRAIN] == "65536" and env[VAL] == "253952"
    monkeypatch.delenv(TRAIN)
    monkeypatch.delenv(VAL)
    env = get_ppo_ray_runtime_env()["env_vars"]
    assert env[TRAIN] == env[VAL] == ""


def make_loop(shared_config):
    loop = object.__new__(SWEToolAgentLoop)
    loop.rollout_config = shared_config
    loop.response_length = shared_config.response_length
    loop.enable_continuous_token = False
    loop.max_assistant_turns = loop.max_user_turns = None
    loop.tools, loop.tool_schemas = {}, []
    loop._generation_debug_event = lambda *a, **kw: None

    async def multimodal(messages):
        return {}

    async def template(*args, **kwargs):
        return [99, 98]

    async def extract(*args):
        return "", []

    loop.process_multi_modal_info = multimodal
    loop._get_mm_processor_kwargs = lambda audios: {}
    loop.apply_chat_template = template
    loop.tool_parser = SimpleNamespace(stop_token_ids=[], extract_tool_calls=extract)
    loop.requests = []

    async def generate(**kwargs):
        params = kwargs["sampling_params"]
        loop.requests.append(dict(params))
        await asyncio.sleep(0)  # Interleave concurrent train/validation loops.
        length = params["max_tokens"]
        return SimpleNamespace(
            token_ids=[1] * length,
            log_probs=[-0.1] * length,
            extra_fields={},
            num_preempted=0,
            routed_experts=None,
            stop_reason="length",
        )

    loop.server_manager = SimpleNamespace(generate=generate)
    return loop


def test_concurrent_training_and_validation_use_independent_budgets(monkeypatch):
    monkeypatch.setenv(TRAIN, "6")
    monkeypatch.setenv(VAL, "9")
    shared = SimpleNamespace(response_length=12)
    train, val = make_loop(shared), make_loop(shared)
    params = {"temperature": 1.0, "max_tokens": 12}

    async def run():
        return await asyncio.gather(
            train.run(params, raw_prompt=[]),
            val.run(params, raw_prompt=[], _agent_loop_validate=True),
        )

    outputs = asyncio.run(run())
    for loop, output, limit, validation in zip((train, val), outputs, (6, 9), (0, 1), strict=True):
        assert len(output.response_ids) == len(output.response_mask) == len(output.response_logprobs) == limit
        assert output.prompt_ids == [99, 98]
        assert output.response_mask == [1] * limit
        assert loop.requests == [{"temperature": 1.0, "max_tokens": limit}]
        assert output.extra_fields["trajectory_effective_response_length"] == limit
        assert output.extra_fields["trajectory_response_length_capacity"] == 12
        assert output.extra_fields["trajectory_response_length_override_enabled"] == 1
        assert output.extra_fields["trajectory_is_validation"] == validation
        assert output.extra_fields["trajectory_budget_reached"] == 1
        assert output.extra_fields["submission_signal_seen"] is False
        assert output.extra_fields["submission_failure_reason"] == "response_length_limit"
    assert shared.response_length == params["max_tokens"] == 12


def test_reused_loop_and_new_run_pick_up_manual_budget_changes(monkeypatch):
    shared = SimpleNamespace(response_length=12)
    loop = make_loop(shared)
    monkeypatch.setenv(TRAIN, "6")
    first = asyncio.run(loop.run({}, raw_prompt=[]))
    validation = asyncio.run(loop.run({}, raw_prompt=[], _agent_loop_validate=True))
    monkeypatch.setenv(TRAIN, "10")
    resumed = asyncio.run(make_loop(shared).run({}, raw_prompt=[]))
    monkeypatch.delenv(TRAIN)
    default = asyncio.run(loop.run({}, raw_prompt=[]))
    assert [len(out.response_ids) for out in (first, validation, resumed, default)] == [6, 12, 10, 12]
    assert default.extra_fields["trajectory_response_length_override_enabled"] == 0
    assert shared.response_length == 12


def test_returned_budget_metadata_is_also_available_in_reward_info(monkeypatch):
    monkeypatch.setenv(TRAIN, "6")
    loop = make_loop(SimpleNamespace(response_length=12))

    async def parent_run(self, params, **kwargs):
        extra = {"reward_extra_info": {"existing": 1}}
        self._swe_agent_data = SimpleNamespace(extra_fields=extra)
        return SimpleNamespace(response_mask=[1, 0], reward_score=0.0, extra_fields=extra)

    monkeypatch.setattr(ToolAgentLoop, "run", parent_run)
    output = asyncio.run(loop.run({}))
    info = output.extra_fields["reward_extra_info"]
    assert info["existing"] == 1
    assert info["trajectory_effective_response_length"] == 6
    assert info["trajectory_response_length_capacity"] == 12


@pytest.mark.parametrize("validation", [False, True])
@pytest.mark.parametrize("continuous", [False, True])
def test_tool_observations_obey_selected_total_budget(monkeypatch, validation, continuous):
    monkeypatch.setenv(TRAIN, "10")
    monkeypatch.setenv(VAL, "14")
    loop = make_loop(SimpleNamespace(response_length=32))
    loop._swe_is_validation = validation
    loop._configure_response_length()
    loop.max_parallel_calls = 1
    loop.enable_continuous_token = continuous
    loop.tool_parser_name = "qwen3"
    loop.turn_separator = [90]
    data = SimpleNamespace(
        response_mask=[1] * 7,
        response_logprobs=[-0.1] * 7,
        prompt_ids=[99, 98] + [1] * 7,
        messages=[],
        tool_calls=[SimpleNamespace(name="Read", tool_call_id=None)],
        tools_kwargs={},
        tool_rewards=[],
        metrics={},
        extra_fields={},
        user_turns=0,
        image_data=None,
        video_data=None,
    )

    async def call(*args):
        return ToolResponse(text="public observation"), 0.0, {}

    async def template(*args, **kwargs):
        return [2, 2]  # Plus one separator: three observation tokens.

    async def merge(*args, **kwargs):
        return (
            SimpleNamespace(token_ids=data.prompt_ids + [2] * 3),
            data.response_mask + [0] * 3,
            data.response_logprobs + [0.0] * 3,
        )

    loop._call_tool = call
    loop.apply_chat_template = template
    loop.ct_merge_non_assistant_msg = merge
    state = asyncio.run(ToolAgentLoop._handle_processing_tools_state(loop, data))
    if validation:
        assert state == AgentState.GENERATING
        assert data.response_mask == [1] * 7 + [0] * 3
        assert len(data.response_logprobs) == 10 and len(data.prompt_ids) == 12
    else:
        assert state == AgentState.TERMINATED
        assert data.response_mask == [1] * 7
        assert len(data.response_logprobs) == 7 and len(data.prompt_ids) == 9
        assert data.extra_fields["submission_failure_reason"] == "tool_observation_budget"
        assert data.extra_fields["trajectory_budget_reached"] == 1


def test_next_generation_counts_retained_observations_against_budget(monkeypatch):
    monkeypatch.setenv(TRAIN, "10")
    loop = make_loop(SimpleNamespace(response_length=32))
    loop._swe_is_validation = False
    loop._configure_response_length()
    data = SimpleNamespace(
        response_mask=[1] * 4 + [0] * 3,
        assistant_turns=1,
        request_id="second-turn",
        prompt_ids=[99, 98] + [1] * 7,
        extra_fields={},
        image_data=None,
        video_data=None,
    )
    params = {"max_tokens": 32, "max_new_tokens": 4}
    output = asyncio.run(loop._generate_model_tokens(data, params))
    assert len(output.token_ids) == loop.requests[0]["max_tokens"] == 3
    assert "max_new_tokens" not in loop.requests[0]
    assert params == {"max_tokens": 32, "max_new_tokens": 4}


def test_cancelled_generation_summary_keeps_effective_limit_and_capacity(monkeypatch):
    monkeypatch.setenv(TRAIN, "6")
    monkeypatch.setenv("SWE_AGENT_CUTOFF_DEBUG", "1")
    loop = make_loop(SimpleNamespace(response_length=12))
    context = {}

    async def run():
        started = asyncio.Event()

        async def blocked_generate(**kwargs):
            started.set()
            await asyncio.Event().wait()

        loop.server_manager.generate = blocked_generate
        task = asyncio.create_task(loop.run({}, raw_prompt=[], _swe_cutoff_context=context, session_id=7))
        await asyncio.wait_for(started.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())
    summary = context["session_summaries"][7]
    assert summary["outcome"] == "cancelled"
    assert summary["response_length_limit"] == 6
    assert summary["response_length_capacity"] == 12
    assert summary["generation_in_flight"] == 1
    assert loop._swe_agent_data.extra_fields["trajectory_effective_response_length"] == 6


def test_infrastructure_failure_fallback_stays_within_effective_budget(monkeypatch):
    monkeypatch.setenv(TRAIN, "2")
    loop = make_loop(SimpleNamespace(response_length=32))
    loop._swe_is_validation = False
    loop._configure_response_length()
    loop.tokenizer = SimpleNamespace(eos_token_id=42)
    output = SimpleNamespace(
        response_ids=[3, 3],
        response_mask=[0, 0],
        response_logprobs=[0.0, 0.0],
        extra_fields={"swe_terminal_failure": {"reason": "claim_failure"}},
    )
    loop._ensure_failure_response(output)
    assert output.response_ids == [3, 42]
    assert output.response_mask == [0, 1]
    assert len(output.response_ids) == len(output.response_mask) == len(output.response_logprobs) == 2

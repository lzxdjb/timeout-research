"""Capture must happen before the base loop terminates at the token limit."""
import asyncio
from types import SimpleNamespace

import pytest

from recipe.swe_agent import repetition_reward as policy
from recipe.swe_agent.agent_loop import SWEToolAgentLoop
from verl.experimental.agent_loop.tool_agent_loop import AgentState, ToolAgentLoop


@pytest.mark.parametrize('terminal', [False, True])
@pytest.mark.parametrize('enabled', [False, True])
def test_actor_capture_precedes_terminal_budget_check(monkeypatch, terminal, enabled):
    loop = object.__new__(SWEToolAgentLoop)
    loop.response_length = 2 if terminal else 100
    loop.enable_continuous_token = False
    loop.max_assistant_turns = loop.max_user_turns = None
    loop._swe_repeat_config = policy.RepeatConfig(mode='shadow' if enabled else 'off')
    loop._swe_repeat_events, loop._swe_repeat_actor_segments = [], []
    loop._swe_repeat_actor_capture_failed = False
    loop._generation_debug_event = lambda *a, **kw: None
    loop._trajectory_expired = lambda: False
    loop._record_submission_signal = lambda data: None
    loop.tools = {}
    text = ('<think>Actual interrupted terminal thinking' if terminal else
            '<think>Inspect now.</think><tool_call>{"name":"Read"}</tool_call>')
    loop.tokenizer = SimpleNamespace(decode=lambda *a, **kw: text)
    parsed = []
    async def extract(tokens, schemas):
        parsed.append(True)
        return text, []
    loop.tool_parser = SimpleNamespace(stop_token_ids=[], tool_call_start_token='<tool_call>',
                                      tool_call_end_token='</tool_call>', extract_tool_calls=extract)
    async def await_deadline(coro):
        return await coro
    loop._await_with_trajectory_deadline = await_deadline
    async def generate(self, data, params):
        return SimpleNamespace(token_ids=[1, 2], log_probs=[], extra_fields={}, routed_experts=None,
                               num_preempted=0, stop_reason='length' if terminal else 'stop')
    monkeypatch.setattr(ToolAgentLoop, '_generate_model_tokens', generate)
    data = SimpleNamespace(response_mask=[], assistant_turns=0, user_turns=0, prompt_ids=[0],
                           request_id='actor-capture', extra_fields={}, metrics={}, response_logprobs=[], tool_calls=[])
    state = asyncio.run(loop._handle_generating_state(data, {}))
    assert state == AgentState.TERMINATED
    assert len(loop._swe_repeat_actor_segments) == int(enabled)
    assert len(parsed) == int(not terminal)
    if enabled:
        captured = loop._swe_repeat_actor_segments[0]
        assert captured['text'] == (text if terminal else '<think>Inspect now.</think>')
        assert captured['after_event_id'] == 0 and captured['id'] == 'a1'
        assert captured['captured'] is True and not loop._swe_repeat_actor_capture_failed


def test_capture_failure_does_not_break_generation(monkeypatch):
    loop = object.__new__(SWEToolAgentLoop)
    loop._swe_repeat_config = policy.RepeatConfig(mode='shadow')
    loop._swe_repeat_events, loop._swe_repeat_actor_segments = [], []
    loop._swe_repeat_actor_capture_failed = False
    loop._generation_debug_event = lambda *a, **kw: None
    def fail_decode(*args, **kwargs):
        raise ValueError('cannot decode')
    loop.tokenizer = SimpleNamespace(decode=fail_decode)
    async def generate(self, data, params):
        return SimpleNamespace(token_ids=[1], stop_reason='stop')
    monkeypatch.setattr(ToolAgentLoop, '_generate_model_tokens', generate)
    data = SimpleNamespace(assistant_turns=0, request_id='capture-failed', prompt_ids=[0], extra_fields={})
    result = asyncio.run(loop._generate_model_tokens(data, {}))
    assert result.token_ids == [1] and loop._swe_repeat_actor_capture_failed is True

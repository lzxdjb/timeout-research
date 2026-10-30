"""Exercise the reward-to-teacher transition without remote services."""

import asyncio
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from verl.trainer.ppo.v1.agent_loop_tq import AgentLoopWorkerTQ
from verl.trainer.ppo.v1.replay_buffer import rollout_diagnostics

Worker = AgentLoopWorkerTQ.__ray_metadata__.modified_class


@pytest.mark.parametrize(
    "enabled,validate,task_rewards",
    [(True, False, False), (True, True, False), (False, False, False), (False, False, True), (False, True, True)],
)
def test_postprocess_skips_only_training_reward_before_teacher(monkeypatch, enabled, validate, task_rewards):
    monkeypatch.setenv("SWE_AGENT_REPEAT_REWARD_MODE", "off")
    monkeypatch.setenv("VERL_GRPO_GOLDEN_REPLAY_ENABLED", "0")
    worker = object.__new__(Worker)
    worker.config = OmegaConf.create(
        {
            "distillation": {
                "enabled": True,
                "skip_hidden_reward": enabled,
                "distillation_loss": {"use_task_rewards": task_rewards},
            },
            "trainer": {"use_v1": True},
        }
    )
    worker.repeat_coordinator = None
    worker._repeat_debug_event = lambda *args, **kwargs: None
    calls = []

    async def score(outputs, **kwargs):
        calls.append("reward")

    class TeacherReached(Exception):
        pass

    async def teacher(*args, **kwargs):
        calls.append("teacher")
        raise TeacherReached

    worker._compute_score = score
    worker._compute_teacher_logprobs = teacher
    output = SimpleNamespace(
        prompt_ids=[1],
        response_ids=[2],
        reward_score=0.0,
        extra_fields={"hidden_reward_skipped": int(enabled and not validate)},
    )
    with pytest.raises(TeacherReached):
        asyncio.run(worker._agent_loop_postprocess(output, validate, uid="u", session_id=0, global_steps=1))
    assert calls == (["teacher"] if enabled and not validate else ["reward", "teacher"])


@pytest.mark.parametrize("extra,score", [({}, None), ({}, 0.0), ({"hidden_reward_skipped": 1}, None)])
def test_outdated_agent_cannot_silently_run_fallback_evaluator(monkeypatch, extra, score):
    monkeypatch.setenv("SWE_AGENT_REPEAT_REWARD_MODE", "off")
    monkeypatch.setenv("VERL_GRPO_GOLDEN_REPLAY_ENABLED", "0")
    worker = object.__new__(Worker)
    worker.config = OmegaConf.create(
        {
            "distillation": {
                "enabled": True,
                "skip_hidden_reward": True,
                "distillation_loss": {"use_task_rewards": False},
            },
            "trainer": {"use_v1": True},
        }
    )
    worker.repeat_coordinator = None
    worker._repeat_debug_event = lambda *args, **kwargs: None

    async def unexpected(*args, **kwargs):
        pytest.fail("An incompatible source must fail before reward or teacher scoring")

    worker._compute_score = worker._compute_teacher_logprobs = unexpected
    output = SimpleNamespace(extra_fields=extra, reward_score=score)
    with pytest.raises(RuntimeError, match="update the SWE source checkout"):
        asyncio.run(worker._agent_loop_postprocess(output, False, uid="u", session_id=0, global_steps=1))


def test_rollout_diagnostics_preserves_explicit_unevaluated_state():
    result = rollout_diagnostics(
        {
            "reward_extra_info": {
                "hidden_reward_skipped": 1,
                "hidden_reward_scored": 0,
                "reward_placeholder": 1,
                "hidden_reward_status": "intentionally_unevaluated",
                "raw_score": None,
                "hidden_pass": None,
            }
        }
    )
    assert result["hidden_reward_skipped"] == 1
    assert result["hidden_reward_scored"] == 0
    assert result["reward_placeholder"] == 1
    assert result["hidden_reward_status"] == "intentionally_unevaluated"
    assert result["raw_score"] is result["hidden_pass"] is None

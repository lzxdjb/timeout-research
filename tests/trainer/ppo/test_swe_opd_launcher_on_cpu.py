from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

LAUNCHER = Path(__file__).resolve().parents[3] / "running_script_6" / "opd_397b_external.sh"


def _stub_launch(tmp_path: Path, **overrides: str) -> subprocess.CompletedProcess:
    """Capture the real launcher contract without starting Ray or GPU workers."""
    target = tmp_path / "swe" / "verl"
    wrapper = target / "scripts" / "run_swe_agent_qwen3_5_35b_a3b.sh"
    wrapper.parent.mkdir(parents=True)
    wrapper.write_text(
        "#!/usr/bin/env bash\n"
        "python3 - \"$@\" <<'PY'\n"
        "import json, os, sys\n"
        "print(json.dumps({'args': sys.argv[1:], 'env': dict(os.environ)}))\n"
        "PY\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    for name in (
        "max_prompt_length",
        "max_response_length",
        "MAX_PROMPT_LENGTH",
        "MAX_RESPONSE_LENGTH",
        "TEACHER_MAX_MODEL_LEN",
        "TRAIN_BATCH_SIZE",
        "PPO_MINI_BATCH_SIZE",
        "ROLLOUT_N",
        "TOTAL_TRAINING_STEPS",
        "MODEL_PATH",
        "SWE_AGENT_TRAINING_TOOL_FAILURE_PENALTY_COST",
        "SWE_AGENT_TRAINING_TOOL_FAILURE_PENALTY_CAP",
        "SWE_AGENT_TRAINING_TOOL_FAILURE_PENALTY_WARMUP_STEPS",
    ):
        env.pop(name, None)
    env.update(TARGET_VERL=str(target), CONFIG_ONLY="1")
    env.update(overrides)
    return subprocess.run(["bash", str(LAUNCHER)], env=env, capture_output=True, text=True)


def test_opd_launcher_uses_external_teacher_and_clears_inherited_objectives(tmp_path: Path) -> None:
    result = _stub_launch(
        tmp_path,
        SWE_AGENT_TRAINING_TOOL_FAILURE_PENALTY_ENABLED="0",
        SWE_AGENT_TRAINING_PROTOCOL_REWARD_SHAPING="1",
        SWE_AGENT_ROLLOUT_TRAINING_UNVERIFIED_TRAJECTORY_RATIO_THRESHOLD="0.25",
        VERL_GRPO_GOLDEN_REPLAY_ENABLED="1",
    )
    assert result.returncode == 0, result.stderr
    captured = json.loads(result.stdout)
    args, env = captured["args"], captured["env"]
    assert "distillation.teacher_models.teacher_model.backend=external" in args
    assert "distillation.n_gpus_per_node=0" in args
    assert "distillation.nnodes=0" in args
    assert "distillation.distillation_loss.loss_mode=k1" in args
    assert "distillation.distillation_loss.use_policy_gradient=True" in args
    assert "distillation.distillation_loss.use_task_rewards=False" in args
    assert "trainer.resume_mode=disable" in args
    assert "actor_rollout_ref.rollout.n=8" in args
    assert "actor_rollout_ref.rollout.val_kwargs.n=1" in args
    assert env["SWE_AGENT_TRAINING_TOOL_FAILURE_PENALTY_ENABLED"] == "1"
    assert env["SWE_AGENT_TRAINING_TOOL_FAILURE_PENALTY_COST"] == "0.02"
    assert env["SWE_AGENT_TRAINING_TOOL_FAILURE_PENALTY_CAP"] == "0.30"
    assert env["SWE_AGENT_TRAINING_TOOL_FAILURE_PENALTY_WARMUP_STEPS"] == "0"
    assert env["SWE_AGENT_TRAINING_TOOL_REPEAT_PENALTY_ENABLED"] == "0"
    assert env["SWE_AGENT_TRAINING_THINKING_REPEAT_PENALTY_ENABLED"] == "0"
    assert env["SWE_AGENT_TRAINING_PROTOCOL_REWARD_SHAPING"] == "0"
    assert env["VERL_GRPO_GOLDEN_REPLAY_ENABLED"] == "0"
    assert env["SWE_AGENT_ROLLOUT_TRAINING_COMPLETION_RATIO_THRESHOLD"] == "0.85"
    assert env["SWE_AGENT_ROLLOUT_TRAINING_UNVERIFIED_TRAJECTORY_RATIO_THRESHOLD"] == "0.25"
    assert env["SWE_AGENT_TASK_FILTER_MIN_GROUP_ATTEMPTS"] == "8"
    assert env["MODEL_PATH"].endswith("/stock-rl-reflect/data/3552-checkpoint")
    for arg in (
        "data.train_batch_size=64",
        "actor_rollout_ref.actor.ppo_mini_batch_size=64",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.rollout.max_model_len=262144",
        "actor_rollout_ref.rollout.max_num_seqs=512",
        "actor_rollout_ref.rollout.max_num_batched_tokens=32678",
        "trainer.save_freq=1",
        "trainer.test_freq=2",
        "trainer.val_before_train=False",
        "trainer.total_epochs=15",
    ):
        assert arg in args
    assert env["MAX_PROMPT_LENGTH"] == "8192"
    assert env["MAX_RESPONSE_LENGTH"] == "253952"
    assert not any(arg.startswith("trainer.total_training_steps=") for arg in args)


@pytest.mark.parametrize("response,accepted", [("253951", True), ("253952", False)])
def test_opd_launcher_reserves_one_teacher_scoring_token(tmp_path: Path, response: str, accepted: bool) -> None:
    result = _stub_launch(
        tmp_path, MAX_PROMPT_LENGTH="8192", MAX_RESPONSE_LENGTH=response, TEACHER_MAX_MODEL_LEN="262144"
    )
    assert (result.returncode == 0) is accepted
    if accepted:
        assert "actor_rollout_ref.rollout.max_model_len=262143" in json.loads(result.stdout)["args"]
    else:
        assert "+ 1 scoring token exceeds the teacher context" in result.stderr


def test_opd_launcher_preserves_explicit_penalty_costs_and_step_limit(tmp_path: Path) -> None:
    result = _stub_launch(
        tmp_path,
        TOTAL_TRAINING_STEPS="20",
        SWE_AGENT_TRAINING_TOOL_FAILURE_PENALTY_COST="0.03",
        SWE_AGENT_TRAINING_TOOL_FAILURE_PENALTY_CAP="0.40",
        SWE_AGENT_TRAINING_TOOL_FAILURE_PENALTY_WARMUP_STEPS="2",
    )
    assert result.returncode == 0, result.stderr
    captured = json.loads(result.stdout)
    assert "trainer.total_training_steps=20" in captured["args"]
    assert captured["env"]["SWE_AGENT_TRAINING_TOOL_FAILURE_PENALTY_COST"] == "0.03"
    assert captured["env"]["SWE_AGENT_TRAINING_TOOL_FAILURE_PENALTY_CAP"] == "0.40"
    assert captured["env"]["SWE_AGENT_TRAINING_TOOL_FAILURE_PENALTY_WARMUP_STEPS"] == "2"

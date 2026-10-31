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
        "SWE_AGENT_TRAINING_SKIP_HIDDEN_REWARD",
        "SWE_AGENT_OPD_USE_TASK_REWARDS",
        "SWE_AGENT_OPD_DISTILLATION_LOSS_COEF",
        "SWE_AGENT_TRAINING_PARTIAL_HIDDEN_TEST_REWARD",
        "SWE_AGENT_ROLLOUT_TRAINING_COMPLETION_RATIO_THRESHOLD",
        "SWE_AGENT_OPD_CANCEL_ON_CUTOFF",
        "SWE_AGENT_OPD_CUTOFF_WAIT_FOR_SCORING",
        "SWE_AGENT_OPD_PHASE_TIMING",
        "SWE_AGENT_OPD_CUTOFF_CLEANUP_TIMEOUT_SECONDS",
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
        SWE_AGENT_TRAINING_PARTIAL_HIDDEN_TEST_REWARD="1",
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
    assert "distillation.distillation_loss.distillation_loss_coef=1.0" in args
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
    assert env["SWE_AGENT_TRAINING_PARTIAL_HIDDEN_TEST_REWARD"] == "0"
    assert env["SWE_AGENT_OPD_USE_TASK_REWARDS"] == "0"
    assert env["SWE_AGENT_OPD_DISTILLATION_LOSS_COEF"] == "1.0"
    assert env["VERL_GRPO_GOLDEN_REPLAY_ENABLED"] == "0"
    assert env["SWE_AGENT_ROLLOUT_TRAINING_COMPLETION_RATIO_THRESHOLD"] == "0.95"
    assert env["SWE_AGENT_OPD_CANCEL_ON_CUTOFF"] == "1"
    assert env["SWE_AGENT_OPD_CUTOFF_WAIT_FOR_SCORING"] == "0"
    assert env["SWE_AGENT_OPD_PHASE_TIMING"] == "1"
    assert env["SWE_AGENT_OPD_CUTOFF_CLEANUP_TIMEOUT_SECONDS"] == "120"
    assert env["SWE_AGENT_ROLLOUT_TRAINING_UNVERIFIED_TRAJECTORY_RATIO_THRESHOLD"] == "0.25"
    assert env["SWE_AGENT_TASK_FILTER_MIN_GROUP_ATTEMPTS"] == "8"
    assert env["MODEL_PATH"].endswith("/stock-rl-reflect/data/3552-checkpoint")
    for arg in (
        "data.train_batch_size=64",
        "actor_rollout_ref.actor.ppo_mini_batch_size=64",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.rollout.max_model_len=262143",
        "actor_rollout_ref.rollout.max_num_seqs=512",
        "actor_rollout_ref.rollout.max_num_batched_tokens=32678",
        "trainer.save_freq=1",
        "trainer.test_freq=2",
        "trainer.val_before_train=False",
        "trainer.total_epochs=15",
    ):
        assert arg in args
    assert env["MAX_PROMPT_LENGTH"] == "8192"
    assert env["MAX_RESPONSE_LENGTH"] == "253951"
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


def test_opd_launcher_preserves_explicit_cutoff_and_instrumentation_overrides(tmp_path: Path) -> None:
    overrides = {
        "SWE_AGENT_ROLLOUT_TRAINING_COMPLETION_RATIO_THRESHOLD": "1.0",
        "SWE_AGENT_OPD_CANCEL_ON_CUTOFF": "0",
        "SWE_AGENT_OPD_PHASE_TIMING": "0",
        "SWE_AGENT_OPD_CUTOFF_CLEANUP_TIMEOUT_SECONDS": "45",
    }
    result = _stub_launch(tmp_path, **overrides)

    assert result.returncode == 0, result.stderr
    env = json.loads(result.stdout)["env"]
    for name, expected in overrides.items():
        assert env[name] == expected
    assert env["SWE_AGENT_ROLLOUT_VALIDATION_COMPLETION_RATIO_THRESHOLD"] == "0.98"


@pytest.mark.parametrize("task_rewards", ["0", "1"])
def test_scoring_drain_is_opt_in_for_both_opd_objectives(tmp_path: Path, task_rewards: str) -> None:
    result = _stub_launch(
        tmp_path,
        SWE_AGENT_OPD_USE_TASK_REWARDS=task_rewards,
        SWE_AGENT_OPD_CUTOFF_WAIT_FOR_SCORING="1",
    )
    assert result.returncode == 0, result.stderr
    env = json.loads(result.stdout)["env"]
    assert env["SWE_AGENT_OPD_CUTOFF_WAIT_FOR_SCORING"] == "1"
    assert env["SWE_AGENT_OPD_CANCEL_ON_CUTOFF"] == "1"
    assert env["SWE_AGENT_TRAINING_SKIP_HIDDEN_REWARD"] == ("0" if task_rewards == "1" else "1")


@pytest.mark.parametrize("override,expected", [(None, "1"), ("0", "0")])
def test_opd_launcher_skips_hidden_reward_by_default_with_explicit_opt_out(
    tmp_path: Path, override: str | None, expected: str
) -> None:
    overrides = {"MAX_RESPONSE_LENGTH": "253951", "TEACHER_MAX_MODEL_LEN": "262144"}
    if override is not None:
        overrides["SWE_AGENT_TRAINING_SKIP_HIDDEN_REWARD"] = override
    result = _stub_launch(tmp_path, **overrides)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["env"]["SWE_AGENT_TRAINING_SKIP_HIDDEN_REWARD"] == expected


@pytest.mark.parametrize("task_rewards", ["1", "true", "True"])
def test_opd_launcher_task_rewards_enable_hidden_evaluation(tmp_path: Path, task_rewards: str) -> None:
    result = _stub_launch(
        tmp_path,
        SWE_AGENT_OPD_USE_TASK_REWARDS=task_rewards,
        SWE_AGENT_TRAINING_PARTIAL_HIDDEN_TEST_REWARD="1",
    )

    assert result.returncode == 0, result.stderr
    captured = json.loads(result.stdout)
    assert "distillation.distillation_loss.use_task_rewards=True" in captured["args"]
    assert captured["env"]["SWE_AGENT_OPD_USE_TASK_REWARDS"] == "1"
    assert captured["env"]["SWE_AGENT_TRAINING_SKIP_HIDDEN_REWARD"] == "0"
    assert captured["env"]["SWE_AGENT_TRAINING_PARTIAL_HIDDEN_TEST_REWARD"] == "0"


@pytest.mark.parametrize("skip_hidden", ["1", "true", "TRUE"])
def test_opd_launcher_rejects_hidden_skip_with_task_rewards(tmp_path: Path, skip_hidden: str) -> None:
    result = _stub_launch(
        tmp_path,
        SWE_AGENT_OPD_USE_TASK_REWARDS="1",
        SWE_AGENT_TRAINING_SKIP_HIDDEN_REWARD=skip_hidden,
    )

    assert result.returncode != 0
    assert "requires hidden evaluation" in result.stderr


@pytest.mark.parametrize("task_rewards,skip_hidden", [("false", "false"), ("true", "FALSE"), ("0", "true")])
def test_opd_launcher_normalizes_explicit_boolean_settings(
    tmp_path: Path, task_rewards: str, skip_hidden: str
) -> None:
    result = _stub_launch(
        tmp_path,
        SWE_AGENT_OPD_USE_TASK_REWARDS=task_rewards,
        SWE_AGENT_TRAINING_SKIP_HIDDEN_REWARD=skip_hidden,
    )

    assert result.returncode == 0, result.stderr
    env = json.loads(result.stdout)["env"]
    assert env["SWE_AGENT_OPD_USE_TASK_REWARDS"] == str(int(task_rewards == "true"))
    assert env["SWE_AGENT_TRAINING_SKIP_HIDDEN_REWARD"] == str(int(skip_hidden == "true"))


@pytest.mark.parametrize("variable", ["SWE_AGENT_OPD_USE_TASK_REWARDS", "SWE_AGENT_TRAINING_SKIP_HIDDEN_REWARD"])
def test_opd_launcher_rejects_invalid_booleans(tmp_path: Path, variable: str) -> None:
    result = _stub_launch(tmp_path, **{variable: "maybe"})

    assert result.returncode != 0
    assert f"{variable} must be 0, 1, false, or true" in result.stderr


@pytest.mark.parametrize("coefficient", ["0", "0.5", "2", "1e-3"])
def test_opd_launcher_accepts_distillation_loss_coefficient(tmp_path: Path, coefficient: str) -> None:
    result = _stub_launch(tmp_path, SWE_AGENT_OPD_DISTILLATION_LOSS_COEF=coefficient)

    assert result.returncode == 0, result.stderr
    captured = json.loads(result.stdout)
    normalized = str(float(coefficient))
    assert f"distillation.distillation_loss.distillation_loss_coef={normalized}" in captured["args"]
    assert captured["env"]["SWE_AGENT_OPD_DISTILLATION_LOSS_COEF"] == normalized


@pytest.mark.parametrize("coefficient", ["-0.1", "nan", "inf", "-inf", "broken"])
def test_opd_launcher_rejects_invalid_distillation_loss_coefficient(tmp_path: Path, coefficient: str) -> None:
    result = _stub_launch(tmp_path, SWE_AGENT_OPD_DISTILLATION_LOSS_COEF=coefficient)

    assert result.returncode != 0
    assert "SWE_AGENT_OPD_DISTILLATION_LOSS_COEF must be finite and nonnegative" in result.stderr

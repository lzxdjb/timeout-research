from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
LAUNCHER = REPO_ROOT / "scripts" / "run_swe_agent_qwen3_5_35b_a3b_async.sh"


def _create_fake_environment(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path, str]:
    target = tmp_path / "verl"
    (target / "verl").mkdir(parents=True)

    async_deps = tmp_path / "async-deps"
    async_deps.mkdir()

    source = tmp_path / "stock-rl-reflect"
    agent_loop = source / "recipe" / "swe_agent" / "agent_loop.py"
    agent_loop.parent.mkdir(parents=True)
    agent_loop.write_text("# test fixture\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    subprocess.run(["git", "-C", str(source), "add", "recipe/swe_agent/agent_loop.py"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(source),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()

    model = tmp_path / "model"
    model.mkdir()

    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "for arg in \"$@\"; do printf 'ARG=%s\\n' \"$arg\"; done\n"
        "/usr/bin/env\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    return target, async_deps, source, model, fake_python, revision


def _launcher_env(tmp_path: Path) -> tuple[dict[str, str], str, Path, Path, Path]:
    target, async_deps, source, model, fake_python, revision = _create_fake_environment(tmp_path)
    env = os.environ.copy()
    env.update(
        {
            "TARGET_VERL": str(target),
            "ASYNC_DEPS": str(async_deps),
            "SWE_SOURCE": str(source),
            "MODEL_PATH": str(model),
            "PYTHON_BIN": str(fake_python),
            "TRAIN_FILES": '["/tmp/train.parquet"]',
            "VAL_FILES": '["/tmp/val.parquet"]',
            "SWE_AGENT_EXECUTION_URLS": "http://127.0.0.1:18080",
            "SWE_SOURCE_EXPECTED_COMMIT": revision[:12],
            "CONFIG_ONLY": "1",
        }
    )
    return env, revision, target, async_deps, source


def _arguments(output: str) -> set[str]:
    return {line.removeprefix("ARG=") for line in output.splitlines() if line.startswith("ARG=")}


def _exported_environment(output: str) -> dict[str, str]:
    result = {}
    for line in output.splitlines():
        if "=" in line and not line.startswith("ARG="):
            name, value = line.split("=", 1)
            result[name] = value
    return result


def test_async_launcher_uses_integrated_swe_configuration(tmp_path: Path):
    env, revision, target, async_deps, source = _launcher_env(tmp_path)

    result = subprocess.run([str(LAUNCHER)], env=env, check=True, text=True, capture_output=True)
    arguments = _arguments(result.stdout)
    exported = _exported_environment(result.stdout)

    assert "verl.experimental.fully_async_policy.fully_async_main" in arguments
    assert "--config-name=fully_async_ppo_megatron_trainer.yaml" in arguments
    assert "--cfg" in arguments
    assert "actor_rollout_ref.actor.megatron.context_parallel_size=2" in arguments
    assert "actor_rollout_ref.actor.megatron.expert_model_parallel_size=4" in arguments
    assert "actor_rollout_ref.actor.ppo_mini_batch_size=128" in arguments
    assert "actor_rollout_ref.rollout.n=8" in arguments
    assert "actor_rollout_ref.rollout.mode=async" in arguments
    assert "async_training.concurrent_samples_per_replica=2" in arguments
    assert "algorithm.filter_groups.enable=True" in arguments
    assert "algorithm.filter_groups.metric=acc" in arguments
    assert "algorithm.filter_groups.max_inflight_gen_batches=1" in arguments
    agent_loop_config = (
        f"actor_rollout_ref.rollout.agent.agent_loop_config_path="
        f"{source}/recipe/swe_agent/config/agent_loop_config.yaml"
    )
    assert agent_loop_config in arguments
    assert f"reward.custom_reward_function.path={source}/recipe/swe_agent/reward_function.py" in arguments
    assert exported["PYTHONPATH"].startswith(f"{async_deps}:{target}:{source}")
    assert exported["SWE_AGENT_EXECUTION_CLAIM_HTTP_TIMEOUT_SECONDS"] == "900"
    assert exported["SWE_AGENT_ROLLOUT_MAX_CONCURRENT_CLAIM_HTTP_REQUESTS"] == "8"
    assert exported["SWE_AGENT_ROLLOUT_TRAINING_TRAJECTORY_TIMEOUT_SECONDS"] == "2400"
    assert exported["SWE_AGENT_ROLLOUT_VALIDATION_TRAJECTORY_TIMEOUT_SECONDS"] == "3600"
    assert exported["SWE_AGENT_TRAINING_IMAGE_PREFETCH"] == "1"
    assert f"({revision}, recipe=clean)" in result.stdout


def test_async_launcher_preserves_concurrency_and_timeout_overrides(tmp_path: Path):
    env, _revision, _target, _async_deps, _source = _launcher_env(tmp_path)
    env["ASYNC_CONCURRENT_SAMPLES_PER_REPLICA"] = "3"
    env["SWE_AGENT_EXECUTION_CLAIM_HTTP_TIMEOUT_SECONDS"] = "123"
    env["SWE_AGENT_ROLLOUT_MAX_CONCURRENT_CLAIM_HTTP_REQUESTS"] = "3"
    env["SWE_AGENT_ROLLOUT_TRAINING_TRAJECTORY_TIMEOUT_SECONDS"] = "777"

    result = subprocess.run([str(LAUNCHER)], env=env, check=True, text=True, capture_output=True)
    arguments = _arguments(result.stdout)
    exported = _exported_environment(result.stdout)

    assert "async_training.concurrent_samples_per_replica=3" in arguments
    assert exported["SWE_AGENT_EXECUTION_CLAIM_HTTP_TIMEOUT_SECONDS"] == "123"
    assert exported["SWE_AGENT_ROLLOUT_MAX_CONCURRENT_CLAIM_HTTP_REQUESTS"] == "3"
    assert exported["SWE_AGENT_ROLLOUT_TRAINING_TRAJECTORY_TIMEOUT_SECONDS"] == "777"


def test_async_launcher_does_not_embed_wandb_credentials():
    contents = LAUNCHER.read_text(encoding="utf-8")

    assert "WANDB_API_KEY" not in contents
    assert "wandb login" not in contents

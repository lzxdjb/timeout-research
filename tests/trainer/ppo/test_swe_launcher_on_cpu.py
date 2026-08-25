from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
LAUNCHER = REPO_ROOT / "scripts" / "run_swe_agent_qwen3_5_35b_a3b.sh"

TIMEOUT_DEFAULTS = {
    "SWE_AGENT_EXECUTION_HTTP_TIMEOUT": "2100",
    "SWE_AGENT_EXECUTION_CLAIM_HTTP_TIMEOUT_SECONDS": "900",
    "SWE_AGENT_EXECUTION_EXECUTE_HTTP_TIMEOUT_SECONDS": "750",
    "SWE_AGENT_EXECUTION_REWARD_HTTP_TIMEOUT_SECONDS": "750",
    "SWE_AGENT_EXECUTION_RELEASE_HTTP_TIMEOUT_SECONDS": "60",
    "SWE_AGENT_EXECUTION_PREFETCH_HTTP_TIMEOUT_SECONDS": "60",
    "SWE_AGENT_EXECUTION_OPERATION_POLL_HTTP_TIMEOUT_SECONDS": "30",
    "SWE_AGENT_EXECUTION_OPERATION_POLL_INTERVAL_SECONDS": "1",
    "SWE_AGENT_EXECUTION_TRAINING_HARD_TIMEOUT_SECONDS": "600",
    "SWE_AGENT_EXECUTION_VALIDATION_HARD_TIMEOUT_SECONDS": "900",
    "SWE_AGENT_ROLLOUT_CLAIM_RETRY_TIMEOUT_SECONDS": "600",
    "SWE_AGENT_ROLLOUT_EXECUTE_CAPACITY_RETRY_TIMEOUT_SECONDS": "900",
    "SWE_AGENT_ROLLOUT_REWARD_RETRY_TIMEOUT_SECONDS": "900",
    "SWE_AGENT_ROLLOUT_RELEASE_RETRY_TIMEOUT_SECONDS": "300",
    "SWE_AGENT_TRAINING_IMAGE_PREFETCH_TIMEOUT": "60",
    "SWE_AGENT_TRAINING_IMAGE_PREFETCH_RETRY_TIMEOUT_SECONDS": "600",
    "SWE_AGENT_VALIDATION_IMAGE_PREFETCH_TIMEOUT": "60",
    "SWE_AGENT_VALIDATION_IMAGE_PREFETCH_RETRY_TIMEOUT_SECONDS": "600",
}


def _create_fake_checkouts(tmp_path: Path) -> tuple[Path, Path, Path, str]:
    target = tmp_path / "verl"
    (target / "verl").mkdir(parents=True)
    example = target / "examples" / "grpo_trainer" / "run_qwen3_5_35b_megatron.sh"
    example.parent.mkdir(parents=True)
    example.write_text("#!/usr/bin/env bash\n/usr/bin/env\n", encoding="utf-8")

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
    return target, source, model, revision


def _launcher_env(target: Path, source: Path, model: Path) -> dict[str, str]:
    env = os.environ.copy()
    for name in TIMEOUT_DEFAULTS:
        env.pop(name, None)
    env.update(
        {
            "TARGET_VERL": str(target),
            "SWE_SOURCE": str(source),
            "MODEL_PATH": str(model),
            "TRAIN_FILES": '["/tmp/train.parquet"]',
            "VAL_FILES": '["/tmp/val.parquet"]',
            "SWE_AGENT_EXECUTION_URLS": "http://127.0.0.1:18080",
        }
    )
    return env


def _exported_environment(output: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in output.splitlines():
        if "=" in line:
            name, value = line.split("=", 1)
            result[name] = value
    return result


def test_launcher_exports_granular_timeout_defaults_and_accepts_commit_prefix(tmp_path: Path) -> None:
    target, source, model, revision = _create_fake_checkouts(tmp_path)
    env = _launcher_env(target, source, model)
    env["SWE_SOURCE_EXPECTED_COMMIT"] = revision[:12]

    result = subprocess.run([str(LAUNCHER)], env=env, check=True, text=True, capture_output=True)
    exported = _exported_environment(result.stdout)

    for name, expected in TIMEOUT_DEFAULTS.items():
        assert exported[name] == expected
    assert f"({revision}, recipe=clean)" in result.stdout


def test_launcher_preserves_explicit_timeout_overrides(tmp_path: Path) -> None:
    target, source, model, _revision = _create_fake_checkouts(tmp_path)
    env = _launcher_env(target, source, model)
    env["SWE_AGENT_EXECUTION_HTTP_TIMEOUT"] = "999"
    env["SWE_AGENT_EXECUTION_PREFETCH_HTTP_TIMEOUT_SECONDS"] = "17"

    result = subprocess.run([str(LAUNCHER)], env=env, check=True, text=True, capture_output=True)
    exported = _exported_environment(result.stdout)

    assert exported["SWE_AGENT_EXECUTION_HTTP_TIMEOUT"] == "999"
    assert exported["SWE_AGENT_EXECUTION_PREFETCH_HTTP_TIMEOUT_SECONDS"] == "17"


def test_launcher_rejects_source_revision_mismatch(tmp_path: Path) -> None:
    target, source, model, expected_revision = _create_fake_checkouts(tmp_path)
    agent_loop = source / "recipe" / "swe_agent" / "agent_loop.py"
    agent_loop.write_text("# second revision\n", encoding="utf-8")
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
            "second fixture",
        ],
        check=True,
    )
    env = _launcher_env(target, source, model)
    env["SWE_SOURCE_EXPECTED_COMMIT"] = expected_revision

    result = subprocess.run([str(LAUNCHER)], env=env, text=True, capture_output=True)

    assert result.returncode == 2
    assert "SWE source revision mismatch:" in result.stderr

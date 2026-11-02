import os
import shlex
import subprocess
from pathlib import Path

import pytest


LAUNCHER = Path(__file__).resolve().parents[2] / "scripts/serve_swe_repeat_judge.sh"


@pytest.fixture
def launcher_env(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    env = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith(("SWE_REPEAT_JUDGE_", "SWE_AGENT_REPEAT_JUDGE_"))
    }
    env.update({
        "SWE_REPEAT_JUDGE_CHECKPOINT": str(tmp_path),
        "SWE_REPEAT_JUDGE_ADVERTISE_HOST": "judge.service.example",
    })
    return env


def dry_run(env):
    return subprocess.run(
        ["bash", str(LAUNCHER), "--dry-run"], env=env, capture_output=True, text=True, timeout=5
    )


@pytest.mark.parametrize("gpus,tp", [(None, None), ("0,1,2,3,4,5,6,7", "8")])
def test_single_replica_keeps_legacy_vllm_arguments(launcher_env, gpus, tp):
    if gpus is not None:
        launcher_env["SWE_REPEAT_JUDGE_GPUS"] = gpus
        launcher_env["SWE_REPEAT_JUDGE_TP"] = tp
    result = dry_run(launcher_env)
    assert result.returncode == 0, result.stderr
    args = shlex.split(result.stdout)
    assert args[:3] == [f"CUDA_VISIBLE_DEVICES={gpus or '0,1,2,3'}", "vllm", "serve"]
    assert args[args.index("--tensor-parallel-size") + 1] == (tp or "4")
    assert "--data-parallel-size" not in args
    assert args[args.index("--dtype") + 1] == "bfloat16"
    assert args[args.index("--max-model-len") + 1] == "131072"
    assert "Judge client endpoint: http://judge.service.example:18090" in result.stderr


@pytest.mark.parametrize("explicit_dp", [False, True])
def test_all_eight_gpus_can_serve_two_tp4_replicas(launcher_env, explicit_dp):
    launcher_env.update({"SWE_REPEAT_JUDGE_GPUS": "0,1,2,3,4,5,6,7", "SWE_REPEAT_JUDGE_TP": "4"})
    if explicit_dp:
        launcher_env["SWE_REPEAT_JUDGE_DP"] = "2"
    result = dry_run(launcher_env)
    assert result.returncode == 0, result.stderr
    args = shlex.split(result.stdout)
    assert args[0] == "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7"
    assert args[args.index("--tensor-parallel-size") + 1] == "4"
    assert args[args.index("--data-parallel-size") + 1] == "2"
    assert args.count("--data-parallel-size") == 1


@pytest.mark.parametrize("settings", [
    {"SWE_REPEAT_JUDGE_GPUS": "0,1,2", "SWE_REPEAT_JUDGE_TP": "2"},
    {"SWE_REPEAT_JUDGE_GPUS": "0,1,2,3", "SWE_REPEAT_JUDGE_TP": "8"},
    {"SWE_REPEAT_JUDGE_TP": "0"},
    {"SWE_REPEAT_JUDGE_TP": "invalid"},
    {"SWE_REPEAT_JUDGE_DP": "0"},
    {"SWE_REPEAT_JUDGE_DP": "invalid"},
    {"SWE_REPEAT_JUDGE_DP": "2"},
    {"SWE_REPEAT_JUDGE_GPUS": "0,1,2,3,4,5,6,7", "SWE_REPEAT_JUDGE_TP": "4", "SWE_REPEAT_JUDGE_DP": "1"},
])
def test_invalid_parallel_layout_fails_before_launch(launcher_env, settings):
    result = dry_run({**launcher_env, **settings})
    assert result.returncode == 2
    assert result.stderr
    assert "vllm serve" not in result.stdout

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from verl.trainer.distillation.config_utils import (
    opd_cancel_on_cutoff,
    opd_phase_timing_enabled,
    opd_wait_for_scoring_on_cutoff,
    should_skip_hidden_reward,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("audio_only", [True, False])
@pytest.mark.parametrize("inherited_flags", [False, True])
def test_audio_launcher_keeps_distillation_startup_options_consistent(
    tmp_path: Path, audio_only: bool, inherited_flags: bool
) -> None:
    """Exercise both launchers without starting Ray, models, or log writers."""
    target = tmp_path / "verl"
    budget_helper = target / "verl" / "utils" / "swe_response_budget.py"
    budget_helper.parent.mkdir(parents=True)
    budget_helper.write_bytes((REPO_ROOT / "verl" / "utils" / "swe_response_budget.py").read_bytes())
    wrapper = target / "scripts" / "run_swe_agent_qwen3_5_35b_a3b.sh"
    wrapper.parent.mkdir()
    wrapper.write_bytes((REPO_ROOT / "scripts" / wrapper.name).read_bytes())
    example = target / "examples" / "grpo_trainer" / "run_qwen3_5_35b_megatron.sh"
    example.parent.mkdir(parents=True)
    example.write_text(
        "#!/usr/bin/env bash\n"
        "python3 - \"$@\" <<'PY'\n"
        "import json, sys\n"
        "print(json.dumps(sys.argv[1:]))\n"
        "PY\n",
        encoding="utf-8",
    )
    source = tmp_path / "stock-rl-reflect"
    agent_loop = source / "recipe" / "swe_agent" / "agent_loop.py"
    agent_loop.parent.mkdir(parents=True)
    agent_loop.write_text("# fixture\n", encoding="utf-8")
    tool_config = agent_loop.parent / "config" / "tool_config_claude_like.yaml"
    tool_config.parent.mkdir()
    tool_config.write_text("tools: []\n", encoding="utf-8")
    model = tmp_path / "model"
    model.mkdir()

    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("SWE_AGENT_", "AUDIO_BENCHMARK_", "TEACHER_"))
    }
    for key in ("max_prompt_length", "max_response_length", "MAX_PROMPT_LENGTH", "MAX_RESPONSE_LENGTH"):
        env.pop(key, None)
    env.update(
        TARGET_VERL=str(target),
        SWE_SOURCE=str(source),
        MODEL_PATH=str(model),
        CONFIG_ONLY="1",
        AUDIO_BENCHMARK_ONLY=str(audio_only),
    )
    if inherited_flags:
        env.update(
            SWE_AGENT_TRAINING_SKIP_HIDDEN_REWARD="1",
            SWE_AGENT_OPD_CANCEL_ON_CUTOFF="1",
            SWE_AGENT_OPD_CUTOFF_WAIT_FOR_SCORING="1",
            SWE_AGENT_OPD_PHASE_TIMING="1",
        )
    result = subprocess.run(
        ["bash", str(REPO_ROOT / "running_script_6" / "audio_test.sh")],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    args = json.loads(result.stdout.splitlines()[-1])
    startup_config = OmegaConf.from_dotlist(
        [arg for arg in args if arg.startswith(("distillation.", "trainer.use_v1="))]
    )
    assert startup_config.distillation.enabled is (not audio_only)
    assert should_skip_hidden_reward(startup_config) is (not audio_only)
    assert opd_cancel_on_cutoff(startup_config) is (not audio_only)
    assert opd_phase_timing_enabled(startup_config) is (not audio_only)
    assert opd_wait_for_scoring_on_cutoff(startup_config) is (not audio_only and inherited_flags)

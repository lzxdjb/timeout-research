"""Exercise the frozen and projector launch contracts without workers or GPUs."""

import json
import os
import subprocess
from pathlib import Path

import pytest
from hydra.core.override_parser.overrides_parser import OverridesParser

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def launch_env(tmp_path):
    target = tmp_path / "verl"
    helper = target / "scripts/audio_data/multi_teacher_config.py"
    helper.parent.mkdir(parents=True)
    helper.write_bytes((ROOT / "scripts/audio_data/multi_teacher_config.py").read_bytes())
    wrapper = target / "scripts/run_swe_agent_qwen3_5_35b_a3b.sh"
    wrapper.write_text(
        "#!/usr/bin/env bash\n"
        "python3 - \"$@\" <<'PY'\n"
        "import json, os, sys\n"
        "print(json.dumps({'argv':sys.argv[1:], 'model':os.environ['MODEL_PATH'], "
        "'train':os.environ['TRAIN_FILES'], 'val':os.environ['VAL_FILES'], "
        "'expected_tasks':os.environ['SWE_AGENT_VALIDATION_EXPECTED_TASKS']}))\n"
        "PY\n"
    )
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(
            ("SWE_AGENT_", "AUDIO_", "TEACHER_", "OPD_", "TRAIN_", "PPO_", "ROLLOUT_", "TOTAL_", "MODEL_")
        )
    }
    for key in (
        "max_prompt_length", "max_response_length", "MAX_PROMPT_LENGTH", "MAX_RESPONSE_LENGTH",
        "VAL_FILES", "TEST_FREQ", "SAVE_FREQ", "VAL_BEFORE_TRAIN", "SMOKE_VALIDATE", "ACTOR_LR",
    ):
        env.pop(key, None)
    env.update(TARGET_VERL=str(target), SWE_SOURCE=str(tmp_path / "source"), CONFIG_ONLY="1")
    return env


def _launch(launch_env, *, smoke=False, projector=False, extra=()):
    suffix = "_projector" if projector else ""
    suffix += "_smoke" if smoke else ""
    name = f"opd_397b_multi_teacher{suffix}.sh"
    result = subprocess.run(
        ["bash", str(ROOT / "running_script_6" / name), *extra],
        env=launch_env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    captured = json.loads(result.stdout.splitlines()[-1])
    parser = OverridesParser.create()
    overrides = [parser.parse_override(value) for value in captured["argv"]]
    values = {override.key_or_group: override.value() for override in overrides}
    return captured, values


@pytest.mark.parametrize("smoke", [False, True])
def test_stage1_launches_both_routes_with_frozen_audio_and_safe_one_step_gate(launch_env, smoke):
    # The smoke gate must enforce one step even if a copied command has a stale limit.
    extra = ["trainer.total_training_steps=999", "trainer.val_before_train=True"] if smoke else []
    captured, values = _launch(launch_env, smoke=smoke, extra=extra)
    assert values["actor_rollout_ref.actor.megatron.frozen_audio"] is True
    assert values["actor_rollout_ref.actor.megatron.train_audio_projector"] is False
    assert values["actor_rollout_ref.ref.megatron.frozen_audio"] is True
    assert values["actor_rollout_ref.ref.megatron.train_audio_projector"] is False
    assert "actor_rollout_ref.rollout.engine_kwargs.vllm.hf_overrides.verl_audio_projector_trainable" not in values
    assert values["actor_rollout_ref.model.enable_audio_processor"] is True
    assert values["data.audio_sampling_rate"] == 16000
    assert values["data.custom_cls.path"] == "pkg://verl.utils.dataset.audio_opd_dataset"
    assert values["distillation.distillation_loss.audio_sequence_surrogate"] is True
    assert values["distillation.distillation_loss.use_task_rewards"] is False
    assert values["algorithm.rollout_correction.rollout_is"] is None
    assert values["algorithm.rollout_correction.rollout_rs"] is None
    assert values["data.dataloader_num_workers"] == 0
    assert values["data.gen_batch_size"] == 1
    assert values["distillation.teacher_key"] == "teacher_route"
    assert values["trainer.total_training_steps"] == (1 if smoke else 100)
    assert values["trainer.val_before_train"] is (not smoke)
    assert values["trainer.save_freq"] == (1 if smoke else 10)
    assert values["trainer.test_freq"] == (0 if smoke else 10)
    assert values["actor_rollout_ref.rollout.n"] == (1 if smoke else 2)
    swe = values["distillation.teacher_models.swe_teacher"]
    audio = values["distillation.teacher_models.audio_teacher"]
    assert swe["key"] == "swe" and audio["key"] == "audio"
    assert swe["external"]["protocol"] == "text_token_ids"
    assert audio["external"]["protocol"] == "audio_candidate_score"
    assert audio["external"]["base_url"] == "http://10.248.101.72:18090"
    assert audio["external"]["audio_response_length"] == 2048
    assert captured["model"].endswith("global_step_18/actor/model/huggingface")
    assert len(json.loads(captured["train"])) == 2
    assert captured["expected_tasks"] == "357"
    assert json.loads(captured["val"])[0].endswith(
        "swe_bench_verified_reference_verified_repeat2_500_images_v1.parquet"
    )
    assert values["distillation.enabled"] is True
    assert values["trainer.val_only"] is False
    assert values["trainer.audio_benchmark.only"] is False
    assert values["trainer.audio_benchmark.enabled"] is (not smoke)
    assert values["trainer.audio_benchmark.data_root"].endswith("/data/restored/audio_benchmark")
    assert values["trainer.audio_benchmark.benchmarks"] == [
        "aishell2", "chartqa", "clothoaqa", "mmbench", "mmstar", "ocrbench", "realworldqa",
        "rul_muchomusic", "voicebench_bbh",
    ]
    assert values["trainer.audio_benchmark.max_samples"] == 100
    assert values["trainer.audio_benchmark.concurrency"] == 32
    assert values["trainer.audio_benchmark.load_balance"] is True
    assert values["trainer.audio_benchmark.failure_policy"] == "warn"


@pytest.mark.parametrize("smoke", [False, True])
def test_mounted_audio_profile_retains_swe_validation_and_training(launch_env, smoke):
    launch_env.update(
        AUDIO_BENCHMARK_DATA_ROOT="/mnt",
        AUDIO_BENCHMARK_BENCHMARKS="[aishell1,librispeech,wenetspeech,clothoaqa,rul_muchomusic,voicebench_bbh]",
        AUDIO_BENCHMARK_MAX_SAMPLES="5" if smoke else "null",
        AUDIO_BENCHMARK_CONCURRENCY="128",
        AUDIO_BENCHMARK_LOAD_BALANCE="True",
        AUDIO_BENCHMARK_FAILURE_POLICY="error",
        SMOKE_VALIDATE="1",
        VAL_BEFORE_TRAIN="False",
        TEST_FREQ="4",
        SAVE_FREQ="7",
    )
    captured, values = _launch(launch_env, smoke=smoke)
    assert values["trainer.audio_benchmark.enabled"] is True
    assert values["trainer.audio_benchmark.data_root"] == "/mnt"
    assert values["trainer.audio_benchmark.benchmarks"] == [
        "aishell1", "librispeech", "wenetspeech", "clothoaqa", "rul_muchomusic", "voicebench_bbh",
    ]
    assert values["trainer.audio_benchmark.max_samples"] == (5 if smoke else None)
    assert values["trainer.audio_benchmark.concurrency"] == 128
    assert values["trainer.audio_benchmark.load_balance"] is True
    assert values["trainer.audio_benchmark.failure_policy"] == "error"
    assert values["trainer.total_training_steps"] == (1 if smoke else 100)
    assert values["trainer.val_before_train"] is False
    assert values["trainer.test_freq"] == (1 if smoke else 4)
    assert values["trainer.save_freq"] == (1 if smoke else 7)
    assert values["trainer.val_only"] is False
    assert values["trainer.audio_benchmark.only"] is False
    assert values["distillation.enabled"] is True
    assert values["actor_rollout_ref.actor.megatron.frozen_audio"] is True
    assert values["actor_rollout_ref.ref.megatron.frozen_audio"] is True
    assert len(json.loads(captured["train"])) == 2
    assert captured["expected_tasks"] == "357"
    assert json.loads(captured["val"])[0].endswith(
        "swe_bench_verified_reference_verified_repeat2_500_images_v1.parquet"
    )


@pytest.mark.parametrize("validate", ["0", "1", "fAlSe", "TrUe"])
def test_smoke_validation_gate_enforces_one_training_step_over_stale_flags(launch_env, validate):
    launch_env.update(
        SMOKE_VALIDATE=validate,
        AUDIO_FREEZE_ENCODER="0",
        AUDIO_FREEZE_PROJECTOR="0",
        AUDIO_TRAIN_PROJECTOR="1",
        TOTAL_TRAINING_STEPS="999",
        VAL_BEFORE_TRAIN="True",
        TEST_FREQ="99",
        SAVE_FREQ="99",
    )
    _, values = _launch(
        launch_env,
        smoke=True,
        extra=[
            "trainer.total_training_steps=999", "trainer.val_before_train=True", "trainer.val_only=True",
            "trainer.test_freq=99", "trainer.save_freq=99", "trainer.audio_benchmark.only=True",
            "actor_rollout_ref.actor.megatron.frozen_audio=False",
            "actor_rollout_ref.ref.megatron.frozen_audio=False",
            "actor_rollout_ref.actor.megatron.train_audio_projector=True",
            "actor_rollout_ref.ref.megatron.train_audio_projector=True",
        ],
    )
    enabled = validate.lower() in {"1", "true"}
    assert values["trainer.total_training_steps"] == 1
    assert values["trainer.val_before_train"] is False
    assert values["trainer.val_only"] is False
    assert values["trainer.test_freq"] == int(enabled)
    assert values["trainer.save_freq"] == 1
    assert values["trainer.audio_benchmark.only"] is False
    assert values["trainer.audio_benchmark.enabled"] is enabled
    assert values["actor_rollout_ref.actor.megatron.frozen_audio"] is True
    assert values["actor_rollout_ref.actor.megatron.train_audio_projector"] is False
    assert values["actor_rollout_ref.ref.megatron.frozen_audio"] is True
    assert values["actor_rollout_ref.ref.megatron.train_audio_projector"] is False


def test_smoke_validation_can_run_swe_without_audio(launch_env):
    launch_env.update(SMOKE_VALIDATE="True", AUDIO_BENCHMARK_ENABLED="False")
    captured, values = _launch(launch_env, smoke=True)
    assert values["trainer.test_freq"] == 1
    assert values["trainer.audio_benchmark.enabled"] is False
    assert values["trainer.audio_benchmark.only"] is False
    assert json.loads(captured["val"])


def test_stage1_preserves_explicit_benchmark_and_validation_choices(launch_env):
    launch_env.update(
        AUDIO_BENCHMARK_ENABLED="False",
        AUDIO_BENCHMARK_LOAD_BALANCE="False",
        VAL_FILES='["/custom/swe_verified.parquet"]',
        VAL_BEFORE_TRAIN="False",
        TEST_FREQ="3",
        SAVE_FREQ="6",
    )
    captured, values = _launch(launch_env)
    assert json.loads(captured["val"]) == ["/custom/swe_verified.parquet"]
    assert values["trainer.audio_benchmark.enabled"] is False
    assert values["trainer.audio_benchmark.load_balance"] is False
    assert values["trainer.val_before_train"] is False
    assert values["trainer.test_freq"] == 3
    assert values["trainer.save_freq"] == 6


def test_smoke_rejects_invalid_validation_gate_before_launch(launch_env):
    launch_env["SMOKE_VALIDATE"] = "sometimes"
    result = subprocess.run(
        ["bash", str(ROOT / "running_script_6/opd_397b_multi_teacher_smoke.sh")],
        env=launch_env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "SMOKE_VALIDATE" in result.stderr


@pytest.mark.parametrize(
    "flag",
    ["AUDIO_FREEZE_ENCODER", "AUDIO_FREEZE_PROJECTOR", "SWE_AGENT_OPD_USE_TASK_REWARDS"],
)
def test_stage1_rejects_unfreezing_or_task_reward_before_launch(launch_env, flag):
    launch_env[flag] = "1" if flag == "SWE_AGENT_OPD_USE_TASK_REWARDS" else "0"
    result = subprocess.run(
        ["bash", str(ROOT / "running_script_6/opd_397b_multi_teacher.sh")],
        env=launch_env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "Stage 1" in result.stderr


@pytest.mark.parametrize("smoke", [False, True])
def test_stage2_trains_only_actor_projector_with_frozen_reference_and_fresh_optimizer(launch_env, smoke):
    captured, values = _launch(launch_env, smoke=smoke, projector=True)
    assert captured["model"].endswith(
        "qwen35_multi_teacher_frozen_step1_audio_20261102T153523Z/global_step_1/actor/model/huggingface"
    )
    assert values["actor_rollout_ref.actor.megatron.frozen_audio"] is False
    assert values["actor_rollout_ref.actor.megatron.train_audio_projector"] is True
    assert values["actor_rollout_ref.ref.megatron.frozen_audio"] is True
    assert values["actor_rollout_ref.ref.megatron.train_audio_projector"] is False
    assert values["actor_rollout_ref.rollout.engine_kwargs.vllm.hf_overrides.verl_audio_projector_trainable"] is True
    assert values["trainer.resume_mode"] == "disable"
    assert values["actor_rollout_ref.actor.optim.lr"] == 5e-7
    assert values["actor_rollout_ref.rollout.n"] == (1 if smoke else 2)
    assert values["trainer.total_training_steps"] == (1 if smoke else 100)
    assert values["trainer.save_freq"] == (1 if smoke else 10)
    assert values["trainer.test_freq"] == (0 if smoke else 10)
    assert values["distillation.distillation_loss.audio_sequence_surrogate"] is True
    assert values["distillation.distillation_loss.use_task_rewards"] is False
    assert len(json.loads(captured["train"])) == 2
    assert captured["expected_tasks"] == "357"


@pytest.mark.parametrize("validate", ["0", "1", "fAlSe", "TrUe"])
def test_projector_smoke_enforces_mode_and_one_update_over_stale_environment_and_overrides(launch_env, validate):
    launch_env.update(
        SMOKE_VALIDATE=validate,
        AUDIO_TRAIN_PROJECTOR="0",
        AUDIO_FREEZE_ENCODER="0",
        AUDIO_FREEZE_PROJECTOR="1",
        TOTAL_TRAINING_STEPS="999",
        ROLLOUT_N="99",
        TEST_FREQ="99",
        SAVE_FREQ="99",
    )
    _, values = _launch(
        launch_env,
        smoke=True,
        projector=True,
        extra=[
            "trainer.resume_mode=auto", "trainer.total_training_steps=999", "trainer.val_before_train=True",
            "trainer.val_only=True", "trainer.audio_benchmark.only=True", "trainer.test_freq=99",
            "trainer.save_freq=99", "actor_rollout_ref.actor.megatron.frozen_audio=True",
            "actor_rollout_ref.actor.megatron.train_audio_projector=False",
            "actor_rollout_ref.ref.megatron.frozen_audio=False",
            "actor_rollout_ref.ref.megatron.train_audio_projector=True", "actor_rollout_ref.rollout.n=99",
            "actor_rollout_ref.rollout.engine_kwargs.vllm.hf_overrides.verl_audio_projector_trainable=False",
        ],
    )
    enabled = validate.lower() in {"1", "true"}
    assert values["trainer.resume_mode"] == "disable"
    assert values["trainer.total_training_steps"] == 1
    assert values["trainer.val_before_train"] is False
    assert values["trainer.val_only"] is False
    assert values["trainer.test_freq"] == int(enabled)
    assert values["trainer.save_freq"] == 1
    assert values["trainer.audio_benchmark.only"] is False
    assert values["trainer.audio_benchmark.enabled"] is enabled
    assert values["actor_rollout_ref.actor.megatron.frozen_audio"] is False
    assert values["actor_rollout_ref.actor.megatron.train_audio_projector"] is True
    assert values["actor_rollout_ref.ref.megatron.frozen_audio"] is True
    assert values["actor_rollout_ref.ref.megatron.train_audio_projector"] is False
    assert values["actor_rollout_ref.rollout.n"] == 1
    assert values["actor_rollout_ref.rollout.engine_kwargs.vllm.hf_overrides.verl_audio_projector_trainable"] is True


def test_stage2_preserves_custom_checkpoint_and_mounted_validation_profile(launch_env):
    launch_env.update(
        MODEL_PATH="/custom/frozen/actor/model/huggingface",
        SMOKE_VALIDATE="1",
        AUDIO_BENCHMARK_DATA_ROOT="/mnt",
        AUDIO_BENCHMARK_BENCHMARKS="[aishell1,librispeech,wenetspeech,clothoaqa,rul_muchomusic,voicebench_bbh]",
        AUDIO_BENCHMARK_MAX_SAMPLES="5",
        AUDIO_BENCHMARK_CONCURRENCY="128",
        AUDIO_BENCHMARK_FAILURE_POLICY="error",
    )
    captured, values = _launch(launch_env, smoke=True, projector=True)
    assert captured["model"] == "/custom/frozen/actor/model/huggingface"
    assert values["trainer.audio_benchmark.enabled"] is True
    assert values["trainer.audio_benchmark.data_root"] == "/mnt"
    assert values["trainer.audio_benchmark.benchmarks"] == [
        "aishell1", "librispeech", "wenetspeech", "clothoaqa", "rul_muchomusic", "voicebench_bbh",
    ]
    assert values["trainer.audio_benchmark.max_samples"] == 5
    assert values["trainer.audio_benchmark.concurrency"] == 128
    assert values["trainer.audio_benchmark.failure_policy"] == "error"
    assert json.loads(captured["val"])[0].endswith(
        "swe_bench_verified_reference_verified_repeat2_500_images_v1.parquet"
    )


@pytest.mark.parametrize("flag,value", [("AUDIO_FREEZE_ENCODER", "0"), ("AUDIO_FREEZE_PROJECTOR", "1")])
def test_stage2_shared_opt_in_rejects_conflicting_freeze_flags(launch_env, flag, value):
    launch_env.update(AUDIO_TRAIN_PROJECTOR="1", **{flag: value})
    result = subprocess.run(
        ["bash", str(ROOT / "running_script_6/opd_397b_multi_teacher.sh")],
        env=launch_env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "Stage 2 requires" in result.stderr


def test_audio_engine_modes_are_opt_in_and_mutually_exclusive():
    from verl.workers.config.engine import McoreEngineConfig

    config = McoreEngineConfig(tensor_model_parallel_size=2)
    assert config.frozen_audio is False and config.train_audio_projector is False
    assert McoreEngineConfig(train_audio_projector=True, tensor_model_parallel_size=2).train_audio_projector is True
    with pytest.raises(ValueError, match="mutually exclusive"):
        McoreEngineConfig(frozen_audio=True, train_audio_projector=True)


@pytest.mark.parametrize("mode", ["frozen_audio", "train_audio_projector"])
def test_audio_engine_modes_reject_dynamic_context_parallel(mode):
    from verl.workers.config.engine import McoreEngineConfig

    with pytest.raises(ValueError, match=f"{mode} does not support dynamic_context_parallel"):
        McoreEngineConfig(**{mode: True}, dynamic_context_parallel=True, max_seqlen_per_dp_cp_rank=4096)

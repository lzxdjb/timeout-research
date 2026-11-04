"""Verify profile isolation and launch contracts without starting workers."""

import json
import os
import subprocess
from pathlib import Path

import pytest
from hydra.core.override_parser.overrides_parser import OverridesParser
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def launch_env(tmp_path):
    target = tmp_path / "workspace/swe/verl"
    helper_dir = target / "scripts/audio_data"
    helper_dir.mkdir(parents=True)
    for name in ("multi_teacher_config.py", "reasoning_multi_teacher_config.py"):
        (helper_dir / name).write_bytes((ROOT / "scripts/audio_data" / name).read_bytes())
    wrapper = target / "scripts/run_swe_agent_qwen3_5_35b_a3b.sh"
    wrapper.write_text(
        "#!/usr/bin/env bash\n"
        "python3 - \"$@\" <<'PY'\n"
        "import json, os, sys\n"
        "names = ['TRAIN_FILES', 'VAL_FILES', 'SWE_AGENT_VALIDATION_EXPECTED_TASKS', "
        "'SWE_AGENT_TRAINING_TOOL_FAILURE_PENALTY_ENABLED', "
        "'SWE_AGENT_TRAINING_TOOL_FAILURE_PENALTY_COST', "
        "'SWE_AGENT_TRAINING_TOOL_FAILURE_PENALTY_CAP', 'OPD_MIX_SCHEDULE', 'MODEL_PATH', "
        "'SWE_AGENT_ROLLOUT_TRAINING_COMPLETION_RATIO_THRESHOLD', "
        "'SWE_AGENT_ROLLOUT_VALIDATION_COMPLETION_RATIO_THRESHOLD', "
        "'SWE_AGENT_ROLLOUT_TRAINING_RESPONSE_LENGTH', 'SWE_AGENT_ROLLOUT_VALIDATION_RESPONSE_LENGTH', "
        "'MAX_RESPONSE_LENGTH', 'MAX_PROMPT_LENGTH', 'SWE_AGENT_TRAINING_SKIP_HIDDEN_REWARD', "
        "'SWE_AGENT_OPD_CANCEL_ON_CUTOFF', 'SWE_AGENT_OPD_CUTOFF_WAIT_FOR_SCORING', "
        "'SWE_AGENT_OPD_PHASE_TIMING']\n"
        "print(json.dumps({'argv':sys.argv[1:], 'env':{name:os.environ.get(name) for name in names}}))\n"
        "PY\n"
    )
    prefixes = (
        "SWE_AGENT_",
        "AUDIO_",
        "TEACHER_",
        "OPD_",
        "TRAIN_",
        "PPO_",
        "ROLLOUT_",
        "TOTAL_",
        "MODEL_",
        "GENERAL_",
        "MAX_",
        "VAL_",
        "OUTPUT_",
        "VALIDATION_",
        "EXPERIMENT_",
        "SMOKE_",
        "TEST_",
        "SAVE_",
    )
    env = {key: value for key, value in os.environ.items() if not key.startswith(prefixes)}
    for key in ("max_prompt_length", "max_response_length", "ACTOR_LR"):
        env.pop(key, None)
    env.update(TARGET_VERL=str(target), SWE_SOURCE=str(tmp_path / "swe_source"), CONFIG_ONLY="1")
    return env


def launch(env, suffix="reasoning", extra=(), *, success=True):
    result = subprocess.run(
        [
            "bash",
            str(ROOT / "running_script_7" / f"opd_397b_multi_teacher{('_' + suffix) if suffix else ''}.sh"),
            *extra,
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    if not success:
        return result
    assert result.returncode == 0, result.stderr
    captured = json.loads(result.stdout.splitlines()[-1])
    parser = OverridesParser.create()
    overrides = [parser.parse_override(value) for value in captured["argv"]]
    return captured["env"], {override.key_or_group: override.value() for override in overrides}


def test_new_profile_adds_four_reasoning_sources_without_replacing_swe_validation(launch_env):
    exported, values = launch(launch_env)
    train = json.loads(exported["TRAIN_FILES"])
    assert len(train) == 6
    assert train[0].endswith("swe_rebench_v2_reference_verified_snapshot_2000.parquet")
    assert train[1].endswith("audio_opd_pilot_v1/train.opd.parquet")
    root = Path(launch_env["TARGET_VERL"]).parents[1] / "stock-rl-reflect/data/restored"
    assert train[2:] == [
        str(root / "frontier" / f"{name}.parquet")
        for name in ("openmath_29", "math_mixture_42", "arxivqa_136", "deepvision_227")
    ]
    assert exported["SWE_AGENT_VALIDATION_EXPECTED_TASKS"] == "357"
    assert len(json.loads(exported["VAL_FILES"])) == 1
    assert "swe_bench_verified_reference_verified_repeat2_500_images_v1" in exported["VAL_FILES"]
    config = values["trainer.general_reasoning"]
    assert config["enabled"] is True
    assert config["max_samples"] == -1
    assert [Path(path).stem for path in config["val_files"]] == [
        "aime_2024",
        "aime_2025",
        "gpqa_diamond",
        "mmlu_pro",
        "bbh",
    ]
    assert config["max_prompt_length"] == 8192
    assert config["response_length"] == 253951
    assert config["max_model_len"] == 262144
    assert values["data.custom_cls.path"] == "pkg://verl.utils.dataset.reasoning_opd_dataset"
    assert values["data.custom_cls.name"] == "ReasoningOPDMixedDataset"
    assert values["reward.custom_reward_function.path"].endswith("/swe_audio_reasoning_opd.py")
    assert values["actor_rollout_ref.rollout.agent.agent_loop_config_path"].endswith("reasoning_agent_loop_config.yaml")


def test_actual_megatron_launcher_keeps_six_training_files_in_one_hydra_argument(launch_env, tmp_path):
    exported, _ = launch(launch_env)
    # Exercise the real downstream Bash array expansion while replacing only
    # the final Python module, so this cannot create training workers.
    (tmp_path / "capture_reasoning_argv.py").write_text("import json, sys\nprint(json.dumps(sys.argv[1:]))\n")
    env = {
        **launch_env,
        "DEVICE": "gpu",
        "VERL_TRAINER_MODULE": "capture_reasoning_argv",
        "PYTHONPATH": str(tmp_path),
        "train_path": exported["TRAIN_FILES"],
        "test_path": exported["VAL_FILES"],
    }
    result = subprocess.run(
        ["bash", str(ROOT / "examples/grpo_trainer/run_qwen3_5_35b_megatron.sh")],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    argv = json.loads(result.stdout.splitlines()[-1])
    parsed = OverridesParser.create().parse_overrides(argv)
    values = {override.key_or_group: override.value() for override in parsed}
    assert values["data.train_files"] == json.loads(exported["TRAIN_FILES"])
    assert len(values["data.train_files"]) == 6


def test_curriculum_preserves_swe_audio_phases_and_reserves_four_per_reasoning_file(launch_env):
    exported, _ = launch(launch_env)
    phases = json.loads(exported["OPD_MIX_SCHEDULE"])
    assert [phase["steps"] for phase in phases] == [20, 60, 20]
    assert [phase["quotas"]["swe"] for phase in phases] == [12, 24, 36]
    for phase, expected_audio in zip(phases, (36, 24, 12), strict=True):
        quotas = phase["quotas"]
        assert sum(quotas.values()) == 64
        assert sum(quota for name, quota in quotas.items() if name.startswith("audio_")) == expected_audio
        assert {
            name: quotas[name]
            for name in (
                "reasoning_openmath",
                "reasoning_math_mixture",
                "reasoning_arxivqa",
                "reasoning_deepvision",
            )
        } == dict.fromkeys(
            ("reasoning_openmath", "reasoning_math_mixture", "reasoning_arxivqa", "reasoning_deepvision"), 4
        )


@pytest.mark.parametrize("projector", [False, True])
def test_profile_preserves_existing_swe_audio_objectives_and_penalty(launch_env, projector):
    suffix = "_projector" if projector else ""
    old_env, old = launch(launch_env, suffix="projector" if projector else "")
    new_env, new = launch(launch_env, suffix=f"reasoning{suffix}")
    invariant_keys = (
        "actor_rollout_ref.actor.megatron.frozen_audio",
        "actor_rollout_ref.actor.megatron.train_audio_projector",
        "actor_rollout_ref.ref.megatron.frozen_audio",
        "actor_rollout_ref.ref.megatron.train_audio_projector",
        "actor_rollout_ref.model.enable_audio_processor",
        "actor_rollout_ref.rollout.n",
        "actor_rollout_ref.rollout.max_model_len",
        "distillation.distillation_loss.loss_mode",
        "distillation.distillation_loss.audio_sequence_surrogate",
        "distillation.distillation_loss.use_task_rewards",
        "trainer.audio_benchmark.enabled",
        "trainer.audio_benchmark.benchmarks",
        "trainer.audio_benchmark.max_samples",
        "trainer.val_before_train",
        "trainer.test_freq",
    )
    assert {key: new[key] for key in invariant_keys} == {key: old[key] for key in invariant_keys}
    for key in (
        "SWE_AGENT_TRAINING_TOOL_FAILURE_PENALTY_ENABLED",
        "SWE_AGENT_TRAINING_TOOL_FAILURE_PENALTY_COST",
        "SWE_AGENT_TRAINING_TOOL_FAILURE_PENALTY_CAP",
        "VAL_FILES",
    ):
        assert new_env[key] == old_env[key]
    swe = new["distillation.teacher_models.swe_teacher"]["external"]
    audio = new["distillation.teacher_models.audio_teacher"]["external"]
    assert swe["protocol"] == "text_token_ids"
    assert swe["base_url"] == "http://10.248.102.92:18090"
    assert audio["base_url"] == "http://10.248.102.8:18090"
    assert audio["protocol"] == "audio_candidate_score"
    assert (
        audio["audio_response_length"]
        == old["distillation.teacher_models.audio_teacher"]["external"]["audio_response_length"]
    )
    assert new["distillation.teacher_models.swe_teacher.external.enable_image_candidate_scoring"] is True
    assert "trainer.general_reasoning" not in old
    assert old["distillation.teacher_models.swe_teacher"]["external"]["base_url"] == "http://10.248.102.141:18090"


@pytest.mark.parametrize("validate", ["0", "1"])
def test_projector_smoke_enforces_gate_and_caps_only_reasoning_validation_samples(launch_env, validate):
    launch_env.update(SMOKE_VALIDATE=validate, TOTAL_TRAINING_STEPS="999", TEST_FREQ="99", SAVE_FREQ="99")
    _, values = launch(
        launch_env,
        suffix="reasoning_projector_smoke",
        extra=("trainer.total_training_steps=999", "trainer.val_before_train=True", "trainer.test_freq=99"),
    )
    assert values["trainer.total_training_steps"] == 1
    assert values["trainer.val_before_train"] is False
    assert values["trainer.test_freq"] == int(validate)
    assert values["trainer.save_freq"] == 1
    assert values["actor_rollout_ref.rollout.n"] == 1
    assert values["actor_rollout_ref.actor.megatron.train_audio_projector"] is True
    assert values["actor_rollout_ref.ref.megatron.train_audio_projector"] is False
    assert values["trainer.general_reasoning"]["max_samples"] == 5
    assert values["trainer.general_reasoning"]["response_length"] == 253951


def test_profile_honors_explicit_data_endpoints_and_limits_without_changing_audio(launch_env):
    launch_env.update(
        GENERAL_REASONING_DATA_ROOT="/custom/reasoning data",
        GENERAL_REASONING_VAL_FILES='["/custom/test.parquet"]',
        GENERAL_REASONING_MAX_SAMPLES="null",
        GENERAL_TEACHER_TOKENIZER_PATH="/custom/tokenizer",
        TEACHER_BASE_URL="http://swe-teacher:18090",
        AUDIO_TEACHER_BASE_URL="http://audio-teacher:18090",
        max_prompt_length="1024",
        max_response_length="4096",
    )
    exported, values = launch(launch_env)
    assert json.loads(exported["TRAIN_FILES"])[2] == "/custom/reasoning data/frontier/openmath_29.parquet"
    config = values["trainer.general_reasoning"]
    assert config["val_files"] == ["/custom/test.parquet"]
    assert config["max_samples"] is None
    assert config["max_prompt_length"] == 1024
    assert config["response_length"] == 4096
    assert values["actor_rollout_ref.rollout.max_model_len"] == 5120
    assert values["distillation.teacher_models.swe_teacher.external.tokenizer_path"] == "/custom/tokenizer"
    assert values["distillation.teacher_models.swe_teacher"]["external"]["base_url"] == "http://swe-teacher:18090"
    audio = values["distillation.teacher_models.audio_teacher"]["external"]
    assert audio["base_url"] == "http://audio-teacher:18090"
    assert audio["audio_response_length"] == 2048


@pytest.mark.parametrize("bad_samples", ["0", "-2", "true", '"5"'])
def test_invalid_reasoning_validation_caps_fail_before_launch(launch_env, bad_samples):
    launch_env["GENERAL_REASONING_MAX_SAMPLES"] = bad_samples
    result = launch(launch_env, success=False)
    assert result.returncode != 0
    assert "GENERAL_REASONING_MAX_SAMPLES" in result.stderr


def test_formal_profile_applies_requested_budgets_and_evaluates_all_examples(launch_env):
    launch_env.update(
        ROLLOUT_N="1", TRAIN_BATCH_SIZE="32", MAX_RESPONSE_LENGTH="100", max_response_length="200",
        AUDIO_MAX_RESPONSE_LENGTH="2048", AUDIO_BENCHMARK_MAX_SAMPLES="5", GENERAL_REASONING_MAX_SAMPLES="5",
        VAL_FILES='["/wrong/validation.parquet"]', GENERAL_REASONING_VAL_FILES='["/wrong/general.parquet"]',
    )
    exported, values = launch(launch_env, suffix="reasoning_projector_formal")
    assert values["actor_rollout_ref.rollout.n"] == 8
    assert values["data.train_batch_size"] == values["actor_rollout_ref.actor.ppo_mini_batch_size"] == 64
    assert values["actor_rollout_ref.actor.optim.lr"] == 1e-6
    assert values["trainer.total_training_steps"] == 100
    assert values["trainer.save_freq"] == 1 and values["trainer.test_freq"] == 2
    assert values["trainer.val_before_train"] is False
    assert int(exported["MAX_RESPONSE_LENGTH"]) == values["trainer.general_reasoning"]["response_length"] == 252000
    assert exported["MAX_PROMPT_LENGTH"] == "8192"
    assert values["actor_rollout_ref.rollout.max_model_len"] == 260192
    audio = values["distillation.teacher_models.audio_teacher"]
    assert audio["external"]["audio_response_length"] == audio["inference"]["response_length"] == 4096
    assert values["trainer.audio_benchmark.max_tokens"] == 4096
    assert values["trainer.audio_benchmark.temperature"] == 0.0
    assert values["trainer.audio_benchmark.top_k"] == 1
    assert values["trainer.audio_benchmark.top_p"] == 1.0
    # Audio benchmark overrides must not alter training or reasoning decoding.
    assert values["actor_rollout_ref.rollout.temperature"] == 1.0
    assert values["actor_rollout_ref.rollout.top_k"] == -1
    assert values["actor_rollout_ref.rollout.top_p"] == 1.0
    assert values["trainer.general_reasoning"]["temperature"] == 1.0
    assert values["trainer.general_reasoning"]["top_k"] == -1
    assert values["trainer.general_reasoning"]["top_p"] == 1.0
    assert "trainer.general_reasoning.temperature" not in values
    assert values["trainer.audio_benchmark.max_samples"] is None
    assert values["trainer.general_reasoning"]["max_samples"] == -1
    assert len(values["trainer.audio_benchmark.benchmarks"]) == 6
    assert len(values["trainer.general_reasoning"]["val_files"]) == 5
    assert "swe_bench_verified_reference_verified_repeat2_500_images_v1" in exported["VAL_FILES"]
    for phase, cutoff in (("TRAINING", "0.96"), ("VALIDATION", "0.98")):
        assert exported[f"SWE_AGENT_ROLLOUT_{phase}_COMPLETION_RATIO_THRESHOLD"] == cutoff
        assert exported[f"SWE_AGENT_ROLLOUT_{phase}_RESPONSE_LENGTH"] == "252000"


def test_swe_audio_formal_removes_reasoning_and_preserves_current_formal_settings(launch_env):
    # Simulate switching profiles in a shell with stale reasoning paths/quotas.
    inherited, _ = launch(launch_env, suffix="reasoning_projector_formal")
    launch_env.update(
        MODEL_PATH="/models/chosen-hf-checkpoint",
        TRAIN_FILES=inherited["TRAIN_FILES"],
        VAL_FILES='["/old/reasoning-validation.parquet"]',
        OPD_MIX_SCHEDULE=inherited["OPD_MIX_SCHEDULE"],
        GENERAL_REASONING_VAL_FILES='["/old/aime_2024.parquet"]',
    )
    exported, values = launch(launch_env, suffix="projector_formal")
    files = json.loads(exported["TRAIN_FILES"])
    assert len(files) == 2
    assert files[0].endswith("swe_rebench_v2_reference_verified_snapshot_2000.parquet")
    assert files[1].endswith("audio_opd_pilot_v1/train.opd.parquet")
    assert len(json.loads(exported["VAL_FILES"])) == 1
    assert "swe_bench_verified_reference_verified_repeat2_500_images_v1" in exported["VAL_FILES"]
    assert exported["MODEL_PATH"] == "/models/chosen-hf-checkpoint"
    assert values["trainer.resume_mode"] == "disable"
    assert values["trainer.general_reasoning.enabled"] is False
    assert "trainer.general_reasoning" not in values
    assert values["data.custom_cls.path"] == "pkg://verl.utils.dataset.audio_opd_dataset"
    assert values["data.custom_cls.name"] == "AudioOPDMixedDataset"
    assert values["reward.custom_reward_function.path"].endswith("/swe_audio_opd.py")
    assert "actor_rollout_ref.rollout.agent.agent_loop_config_path" not in values
    phases = json.loads(exported["OPD_MIX_SCHEDULE"])
    assert [phase["steps"] for phase in phases] == [20, 60, 20]
    assert [phase["quotas"] for phase in phases] == [
        {"swe": 16, "audio_librispeech": 18, "audio_aishell1": 18, "audio_clotho_aqa": 12},
        {"swe": 32, "audio_librispeech": 12, "audio_aishell1": 12, "audio_clotho_aqa": 8},
        {"swe": 48, "audio_librispeech": 6, "audio_aishell1": 6, "audio_clotho_aqa": 4},
    ]
    assert values["trainer.audio_opd.exclude_incomplete_sequences"] is True
    assert values["trainer.training_metrics_snapshot.enabled"] is True
    assert values["actor_rollout_ref.rollout.n"] == 8
    assert values["data.train_batch_size"] == values["actor_rollout_ref.actor.ppo_mini_batch_size"] == 64
    assert values["actor_rollout_ref.actor.optim.lr"] == 1e-6
    assert values["trainer.total_training_steps"] == 100
    assert values["trainer.save_freq"] == 1 and values["trainer.test_freq"] == 2
    assert values["trainer.val_before_train"] is False
    assert exported["MAX_RESPONSE_LENGTH"] == "252000" and exported["MAX_PROMPT_LENGTH"] == "8192"
    assert values["actor_rollout_ref.rollout.max_model_len"] == 260192
    for phase, cutoff in (("TRAINING", "0.96"), ("VALIDATION", "0.98")):
        assert exported[f"SWE_AGENT_ROLLOUT_{phase}_COMPLETION_RATIO_THRESHOLD"] == cutoff
        assert exported[f"SWE_AGENT_ROLLOUT_{phase}_RESPONSE_LENGTH"] == "252000"
    audio = values["distillation.teacher_models.audio_teacher"]
    assert audio["external"]["audio_response_length"] == audio["inference"]["response_length"] == 8192
    assert audio["external"]["base_url"] == "http://10.248.102.8:18090"
    assert values["distillation.teacher_models.swe_teacher"]["external"]["base_url"] == "http://10.248.102.92:18090"
    assert values["trainer.audio_benchmark.max_tokens"] == 4096
    assert values["trainer.audio_benchmark.max_samples"] is None
    assert values["trainer.audio_benchmark.enabled"] is True
    assert values["trainer.audio_benchmark.benchmarks"] == [
        "aishell1", "librispeech", "wenetspeech", "clothoaqa", "rul_muchomusic", "voicebench_bbh",
    ]
    assert values["actor_rollout_ref.actor.megatron.train_audio_projector"] is True
    assert values["actor_rollout_ref.ref.megatron.frozen_audio"] is True
    assert values["distillation.distillation_loss.audio_sequence_surrogate"] is True
    assert values["distillation.distillation_loss.use_task_rewards"] is False
    assert exported["SWE_AGENT_TRAINING_TOOL_FAILURE_PENALTY_ENABLED"] == "1"
    assert exported["SWE_AGENT_TRAINING_TOOL_FAILURE_PENALTY_COST"] == "0.02"
    assert exported["SWE_AGENT_TRAINING_TOOL_FAILURE_PENALTY_CAP"] == "0.30"


def test_swe_audio_formal_accepts_audio_cap_override_and_requires_explicit_weights(launch_env):
    missing = launch(launch_env, suffix="projector_formal", success=False)
    assert missing.returncode != 0
    assert "Set MODEL_PATH" in missing.stderr
    launch_env.update(MODEL_PATH="/models/chosen-hf-checkpoint", AUDIO_MAX_RESPONSE_LENGTH="16384")
    _, values = launch(launch_env, suffix="projector_formal")
    audio = values["distillation.teacher_models.audio_teacher"]
    assert audio["external"]["audio_response_length"] == audio["inference"]["response_length"] == 16384
    assert values["trainer.audio_benchmark.max_tokens"] == 4096


def test_validation_only_profile_skips_swe_and_sets_both_benchmark_decoders(launch_env):
    exported, values = launch(launch_env, suffix="reasoning_validation_only")
    assert json.loads(exported["TRAIN_FILES"]) == json.loads(exported["VAL_FILES"]) == []
    assert values["trainer.val_only"] is True
    assert values["trainer.audio_benchmark.only"] is True
    assert values["trainer.val_before_train"] is True
    assert values["distillation.enabled"] is False
    assert values["distillation.skip_hidden_reward"] is False
    assert values["distillation.cancel_pending_on_cutoff"] is False
    assert values["distillation.wait_for_scoring_on_cutoff"] is False
    assert values["distillation.phase_timing"] is False
    assert values["trainer.save_freq"] == 0
    assert values["trainer.audio_benchmark.max_samples"] is None
    assert values["trainer.general_reasoning"]["max_samples"] == -1
    assert values["trainer.audio_benchmark.max_tokens"] == 4096
    for decoder in ("actor_rollout_ref.rollout.val_kwargs", "trainer.general_reasoning"):
        assert values[f"{decoder}.temperature"] == 0.6
        assert values[f"{decoder}.top_k"] == 20
        assert values[f"{decoder}.top_p"] == 0.95
    assert values["trainer.audio_benchmark.temperature"] == 0.0
    assert values["trainer.audio_benchmark.top_k"] == 1
    assert values["trainer.audio_benchmark.top_p"] == 1.0


def test_validation_only_profile_passes_runtime_preflight_without_starting_ray(launch_env, monkeypatch):
    # Other CPU suites may select the CUDA platform; this test needs no GPU.
    monkeypatch.setattr("verl.utils.device.get_device_capability", lambda *args: (None, None))
    from verl.trainer import main_ppo
    from verl.trainer.distillation.config_utils import opd_wait_for_scoring_on_cutoff

    exported, values = launch(launch_env, suffix="reasoning_validation_only")
    # Supply Hydra defaults and environment-backed switches normally supplied
    # by the downstream SWE wrapper, then apply the actual profile overrides.
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {"actor": {"policy_loss": {}}, "rollout": {"full_determinism": False}},
            "reward": {"reward_model": {"enable": False, "rollout": {}}},
        }
    )
    for flag, variable in (
        ("skip_hidden_reward", "SWE_AGENT_TRAINING_SKIP_HIDDEN_REWARD"),
        ("cancel_pending_on_cutoff", "SWE_AGENT_OPD_CANCEL_ON_CUTOFF"),
        ("wait_for_scoring_on_cutoff", "SWE_AGENT_OPD_CUTOFF_WAIT_FOR_SCORING"),
        ("phase_timing", "SWE_AGENT_OPD_PHASE_TIMING"),
    ):
        OmegaConf.update(config, f"distillation.{flag}", exported[variable] == "1", force_add=True)
    for key, value in values.items():
        OmegaConf.update(config, key, value, force_add=True)

    class ReachedRayBoundary(Exception):
        pass

    def stop_before_ray():
        raise ReachedRayBoundary

    monkeypatch.setattr(main_ppo.ray, "is_initialized", stop_before_ray)
    assert opd_wait_for_scoring_on_cutoff(config) is False
    with pytest.raises(ReachedRayBoundary):
        main_ppo.run_ppo(config, task_runner_class=None)

"""Check the isolated mixed RL launcher and fail-closed audited data preflight."""

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.audio_data.mixed_rl_preflight import QUOTAS, validate_mixed_rl

ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "running_script_8/mixed_swe_audio_rl.sh"


@pytest.fixture
def launcher_environment(tmp_path):
    target = tmp_path / "target repo"
    scripts = target / "scripts"
    scripts.mkdir(parents=True)
    capture = scripts / "run_swe_agent_qwen3_5_35b_a3b.sh"
    capture.write_text(
        'exec "' + sys.executable + '" - "$@" <<\'PY\'\n'
        "import json, os, sys\n"
        "keys = [key for key in os.environ if key.startswith('SWE_AGENT_')]\n"
        "keys += ['TRAIN_FILES', 'VAL_FILES', 'MODEL_PATH', 'TP', 'CP', 'EP', 'GEN_TP', 'CONFIG_ONLY']\n"
        "print(json.dumps({'args': sys.argv[1:], 'env': {key: os.environ[key] for key in keys}}))\n"
        "PY\n"
    )
    return {
        **os.environ,
        "TARGET_VERL": str(target),
        "SWE_SOURCE": str(tmp_path / "stock source"),
        "AUDIO_RL_ROOT": str(tmp_path / "audited audio"),
        "CONFIG_ONLY": "1",
        "PREFLIGHT_ONLY": "0",
        "ROLLOUT_DATA_DIR": str(tmp_path / "must not create rollout"),
        "VALIDATION_DATA_DIR": str(tmp_path / "must not create validation"),
        "AUDIO_BENCHMARK_OUTPUT_DIR": str(tmp_path / "must not create audio validation"),
    }


def _launch(environment, *args):
    result = subprocess.run(["bash", str(LAUNCHER), *args], env=environment, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def _overrides(captured):
    return {argument.lstrip("+").split("=", 1)[0]: argument.split("=", 1)[1] for argument in captured["args"]}


def test_mixed_rl_preserves_formal_sampling_and_validation(launcher_environment):
    captured = _launch(launcher_environment)
    config = _overrides(captured)
    expected = {
        "trainer.total_training_steps": "100",
        "trainer.resume_mode": "disable",
        "trainer.save_freq": "1",
        "trainer.test_freq": "2",
        "trainer.val_before_train": "False",
        "data.train_batch_size": "64",
        "data.gen_batch_size": "1",
        "data.dataloader_num_workers": "0",
        "data.seed": "42",
        "data.custom_cls.name": "AudioOPDMixedDataset",
        "actor_rollout_ref.actor.ppo_mini_batch_size": "64",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": "1",
        "actor_rollout_ref.actor.optim.lr": "1e-6",
        "actor_rollout_ref.rollout.n": "8",
        "actor_rollout_ref.rollout.temperature": "1.0",
        "actor_rollout_ref.rollout.top_p": "1.0",
        "actor_rollout_ref.rollout.top_k": "-1",
        "actor_rollout_ref.rollout.val_kwargs.n": "1",
        "actor_rollout_ref.rollout.val_kwargs.do_sample": "True",
        "actor_rollout_ref.rollout.val_kwargs.temperature": "1.0",
        "actor_rollout_ref.rollout.val_kwargs.top_p": "1.0",
        "actor_rollout_ref.rollout.val_kwargs.top_k": "-1",
        "actor_rollout_ref.actor.megatron.frozen_audio": "False",
        "actor_rollout_ref.actor.megatron.train_audio_projector": "True",
        "trainer.audio_rl.enabled": "True",
        "trainer.audio_rl.response_length": "2048",
        "trainer.audio_rl.enable_thinking": "False",
        "trainer.audio_rl.exclude_incomplete_sequences": "True",
        "trainer.audio_benchmark.max_samples": "null",
        "trainer.audio_benchmark.concurrency": "128",
        "trainer.audio_benchmark.load_balance": "True",
        "trainer.audio_benchmark.failure_policy": "error",
        "trainer.audio_benchmark.temperature": "0.0",
        "trainer.audio_benchmark.top_k": "1",
        "trainer.audio_benchmark.top_p": "1.0",
        "trainer.audio_benchmark.max_tokens": "512",
        "trainer.general_reasoning.enabled": "False",
    }
    assert {name: config[name] for name in expected} == expected
    assert config["data.opd_source_mix"] == (
        "[{steps:100,quotas:{swe:16,audio_librispeech:18,audio_aishell1:18,audio_clotho_aqa:12}}]"
    )
    assert config["trainer.audio_benchmark.benchmarks"] == (
        "[aishell1,librispeech,wenetspeech,clothoaqa,rul_muchomusic,voicebench_bbh]"
    )
    train = json.loads(captured["env"]["TRAIN_FILES"])
    assert len(train) == 2
    assert train[0].endswith("swe_rebench_v2_reference_verified_snapshot_2000.parquet")
    assert train[1] == str(Path(launcher_environment["AUDIO_RL_ROOT"]) / "train.rl.parquet")
    assert "opd.parquet" not in captured["env"]["TRAIN_FILES"]
    assert len(json.loads(captured["env"]["VAL_FILES"])) == 1
    assert captured["env"]["SWE_AGENT_VALIDATION_EXPECTED_TASKS"] == "357"
    assert captured["env"]["SWE_AGENT_ROLLOUT_TRAINING_COMPLETION_RATIO_THRESHOLD"] == "0.96"
    assert captured["env"]["SWE_AGENT_ROLLOUT_VALIDATION_COMPLETION_RATIO_THRESHOLD"] == "0.98"
    for name in ("ROLLOUT_DATA_DIR", "VALIDATION_DATA_DIR", "AUDIO_BENCHMARK_OUTPUT_DIR"):
        assert not Path(launcher_environment[name]).exists()


def test_inherited_opd_flags_and_penalties_are_reset(launcher_environment):
    environment = dict(launcher_environment)
    for name in (
        "SWE_AGENT_TRAINING_SKIP_HIDDEN_REWARD",
        "SWE_AGENT_OPD_CANCEL_ON_CUTOFF",
        "SWE_AGENT_OPD_CUTOFF_WAIT_FOR_SCORING",
        "SWE_AGENT_OPD_PHASE_TIMING",
    ):
        environment[name] = "1"
    environment.update(
        TRAIN_FILES='["wrong.opd.parquet"]',
        VAL_FILES='["wrong.validation.parquet"]',
        SWE_AGENT_TRAINING_VERIFICATION_REWARD_SHAPING="0",
        SWE_AGENT_TRAINING_TOOL_FAILURE_PENALTY_CAP="0.30",
    )
    captured = _launch(environment)
    config = _overrides(captured)
    for name in (
        "enabled",
        "skip_hidden_reward",
        "cancel_pending_on_cutoff",
        "wait_for_scoring_on_cutoff",
        "phase_timing",
        "distillation_loss.use_policy_gradient",
        "distillation_loss.audio_sequence_surrogate",
    ):
        assert config[f"distillation.{name}"] == "False"
    assert config["distillation.distillation_loss.use_task_rewards"] == "True"
    assert config["distillation.distillation_loss.distillation_loss_coef"] == "0.0"
    assert captured["env"]["SWE_AGENT_TRAINING_SKIP_HIDDEN_REWARD"] == "0"
    assert captured["env"]["SWE_AGENT_TRAINING_VERIFICATION_REWARD_SHAPING"] == "1"
    for penalty in ("TOOL_FAILURE", "THINKING_REPEAT", "TOOL_REPEAT"):
        assert captured["env"][f"SWE_AGENT_TRAINING_{penalty}_PENALTY_ENABLED"] == "1"
        assert captured["env"][f"SWE_AGENT_TRAINING_{penalty}_PENALTY_CAP"] == "0.10"
    assert config["reward.custom_reward_function.path"].endswith("/verl/utils/reward_score/swe_audio_opd.py")
    assert not any("teacher_models" in argument for argument in captured["args"])


def test_explicit_model_and_resume_overrides_are_preserved(launcher_environment):
    environment = dict(launcher_environment, MODEL_PATH="/explicit/student/checkpoint")
    captured = _launch(environment, "trainer.resume_mode=resume_path", "trainer.resume_from_path=/explicit/resume")
    assert captured["env"]["MODEL_PATH"] == "/explicit/student/checkpoint"
    assert _overrides(captured)["trainer.resume_mode"] == "resume_path"


def test_preflight_only_never_delegates_to_training(launcher_environment, tmp_path):
    fake_python = tmp_path / "fake preflight python"
    calls = tmp_path / "preflight_calls.json"
    fake_python.write_text(
        f"#!{sys.executable}\nimport json, sys\nfrom pathlib import Path\n"
        f"Path({str(calls)!r}).write_text(json.dumps(sys.argv[1:]))\n"
    )
    fake_python.chmod(0o755)
    environment = dict(launcher_environment, PREFLIGHT_ONLY="1", AUDIO_PYTHON=str(fake_python))
    result = subprocess.run(["bash", str(LAUNCHER)], env=environment, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    args = json.loads(calls.read_text())
    assert args[:2] == ["-m", "scripts.audio_data.mixed_rl_preflight"]
    assert args[args.index("--expected-swe-validation") + 1] == "357"


@pytest.fixture
def mixed_files(tmp_path):
    media = tmp_path / "audio.flac"
    media.write_bytes(b"fixture media; preflight checks availability, not decoding")
    rows = []
    for dataset, task, language, reference in (
        ("librispeech", "asr", "en", "clear speech"),
        ("aishell1", "asr", "zh", "清楚的语音"),
        ("clotho_aqa", "closed_qa", "en", "a dog barking"),
    ):
        rows.append(
            {
                "data_source": "audio_" + dataset,
                "agent_name": "single_turn_agent",
                "ability": "audio",
                "audios": [{"audio": str(media)}],
                "prompt": [{"role": "user", "content": "Answer about the recording.\n<audio>"}],
                "reward_model": {
                    "style": "rule",
                    "ground_truth": json.dumps({"task": task, "language": language, "reference": reference}),
                },
                "extra_info": {
                    "sample_id": dataset + ":sample",
                    "dataset": dataset,
                    "split": "train",
                    "task": task,
                    "language": language,
                    "export_mode": "rl",
                    "rl_eligible": True,
                    "audio_sha256": "a" * 64,
                    "qc_json": json.dumps(
                        {
                            "offline": {"decision": "accept", "input_fingerprint": "b" * 64},
                            "qwen": {
                                "decision": "accept",
                                "reference_supported": True,
                                "input_fingerprint": "b" * 64,
                                "audio_sha256": "a" * 64,
                            },
                        }
                    ),
                },
            }
        )
    audio = tmp_path / "train.rl.parquet"
    pq.write_table(pa.Table.from_pylist(rows), audio)
    swe_rows = [
        {
            "data_source": "swe_rebench",
            "agent_name": "swe_tool_agent",
            "extra_info": {"task_id": "original-task-" + str(index)},
            "reward_model": {"ground_truth": "original hidden SWE task specification"},
        }
        for index in range(2)
    ]
    swe = tmp_path / "swe.parquet"
    validation = tmp_path / "validation.parquet"
    pq.write_table(pa.Table.from_pylist(swe_rows), swe)
    pq.write_table(pa.Table.from_pylist(swe_rows), validation)
    return audio, swe, validation, rows


def _validate(files):
    return validate_mixed_rl(*files[:3], expected_swe_validation=2)


def test_preflight_validates_audited_pools_and_reports_cycling(mixed_files):
    report = _validate(mixed_files)
    assert report["source_pools"] == {"swe": 2, "audio_librispeech": 1, "audio_aishell1": 1, "audio_clotho_aqa": 1}
    assert report["prompt_quotas_per_64"] == QUOTAS
    assert report["source_sampling"]["audio_librispeech"] == {"prompt_draws": 1800, "pool_passes": 1800}
    assert len(report["audio_train"]["sha256"]) == 64


@pytest.mark.parametrize(
    "failure", ["opd", "ineligible", "test_split", "missing_media", "bad_reward", "unreviewed", "hash"]
)
def test_preflight_rejects_invalid_audio_rows(mixed_files, failure):
    audio, _, _, rows = mixed_files
    rows = copy.deepcopy(rows)
    row = rows[0]
    if failure == "opd":
        row["extra_info"]["export_mode"] = "opd"
    elif failure == "ineligible":
        row["extra_info"]["rl_eligible"] = False
    elif failure == "test_split":
        row["extra_info"]["split"] = "test"
    elif failure == "missing_media":
        row["audios"][0]["audio"] = "/missing/audio.flac"
    elif failure == "bad_reward":
        row["reward_model"]["ground_truth"] = json.dumps({"task": "asr", "language": "en", "reference": "!!!"})
    else:
        qc = json.loads(row["extra_info"]["qc_json"])
        qc["qwen"]["decision" if failure == "unreviewed" else "audio_sha256"] = (
            "reject" if failure == "unreviewed" else "c" * 64
        )
        row["extra_info"]["qc_json"] = json.dumps(qc)
    pq.write_table(pa.Table.from_pylist(rows), audio)
    with pytest.raises(ValueError):
        _validate(mixed_files)


def test_preflight_rejects_empty_requested_source(mixed_files):
    audio, _, _, rows = mixed_files
    pq.write_table(pa.Table.from_pylist(rows[:-1]), audio)
    with pytest.raises(ValueError, match="source pools are empty"):
        _validate(mixed_files)


def test_preflight_rejects_changed_swe_validation_roster(mixed_files):
    with pytest.raises(ValueError, match="exactly 357"):
        validate_mixed_rl(*mixed_files[:3])
    table = pq.read_table(mixed_files[2]).to_pylist()
    table[1]["extra_info"]["task_id"] = table[0]["extra_info"]["task_id"]
    pq.write_table(pa.Table.from_pylist(table), mixed_files[2])
    with pytest.raises(ValueError, match="Duplicate SWE validation"):
        _validate(mixed_files)

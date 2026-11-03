from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
from omegaconf import OmegaConf

LAUNCHER = Path(__file__).resolve().parents[3] / "running_script_6" / "full_audio_benchmark.sh"
CHECKPOINT_HELPER = LAUNCHER.with_name("full_audio_benchmark_checkpoints.py")


@pytest.fixture
def full_launcher(tmp_path: Path) -> tuple[Path, dict[str, str], Path]:
    scripts = tmp_path / "verl" / "running_script_6"
    scripts.mkdir(parents=True)
    launcher = scripts / LAUNCHER.name
    shutil.copyfile(LAUNCHER, launcher)
    shutil.copyfile(CHECKPOINT_HELPER, scripts / CHECKPOINT_HELPER.name)
    (scripts / "audio_test.sh").write_text(
        "#!/usr/bin/env bash\n"
        "python3 - \"$@\" <<'PY'\n"
        "import json, os, shutil, sys\n"
        "from pathlib import Path\n"
        "captured = {'args': sys.argv[1:], 'env': dict(os.environ)}\n"
        "if os.environ.get('STUB_SUMMARY_PATH'):\n"
        "    captured['summary'] = json.loads(Path(os.environ['STUB_SUMMARY_PATH']).read_text())\n"
        "print(json.dumps(captured))\n"
        "if os.environ.get('STUB_ADD_STEP'):\n"
        "    root = Path(os.environ['CHECKPOINT_ROOT'])\n"
        "    destination = root / ('global_step_' + os.environ['STUB_ADD_STEP'])\n"
        "    if not destination.exists():\n"
        "        shutil.copytree(root / 'global_step_17', destination)\n"
        "if os.environ.get('STUB_FAIL_STEP') and os.environ['STUB_FAIL_STEP'] == os.environ.get('CHECKPOINT_STEP'):\n"
        "    sys.exit(7)\n"
        "PY\n"
    )
    checkpoint = tmp_path / "checkpoints"
    model = checkpoint / "global_step_17" / "actor" / "model" / "huggingface"
    model.mkdir(parents=True)
    (model / "config.json").write_text('{"audio_config": {"num_mel_bins": 128}}')
    (model / "processor_config.json").write_text('{"processor_class": "Qwen3VLProcessor"}')
    (model / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.audio_tower.weight": "audio_tower.safetensors"}})
    )
    (model / "audio_tower.safetensors").touch()
    (checkpoint / "latest_checkpointed_iteration.txt").write_text("17\n")
    # A newer directory must not override the completed-checkpoint marker.
    (checkpoint / "global_step_18").mkdir()
    processor_source = tmp_path / "original-audio-model"
    processor_source.mkdir()
    shutil.copyfile(model / "config.json", processor_source / "config.json")
    (processor_source / "preprocessor_config.json").write_text(
        json.dumps({"feature_extractor_type": "WhisperFeatureExtractor", "feature_size": 128, "sampling_rate": 16000})
    )
    datasets = tmp_path / "data"
    for name in ("aishell1", "librispeech", "wenetspeech", "clothoaqa", "muchomusic", "voicebench_bbh"):
        (datasets / name).mkdir(parents=True)
    env = {
        **os.environ,
        "CHECKPOINT_ROOT": str(checkpoint),
        "CHECKPOINT_STEP": "17",
        "TARGET_VERL": str(scripts.parent),
        "AUDIO_BENCHMARK_DATA_ROOT": str(datasets),
        "AUDIO_PROCESSOR_SOURCE": str(processor_source),
        "AUDIO_FULL_RUN_DIR": str(tmp_path / "evaluation"),
        "CONFIG_ONLY": "1",
    }
    return launcher, env, model


def test_full_audio_preview_selects_completed_checkpoint_and_clears_inherited_limits(full_launcher):
    launcher, env, model = full_launcher
    env.update(
        MODEL_PATH="/old/sft/model",
        GAGE_EVAL_MAX_SAMPLES="5",
        AUDIO_BENCHMARK_MAX_SAMPLES="5",
        AUDIO_BENCHMARK_ONLY="False",
    )
    result = subprocess.run(
        ["bash", str(launcher), "trainer.val_only=False", "trainer.audio_benchmark.max_samples=5"],
        env=env, text=True, capture_output=True, check=True,
    )
    captured = json.loads(result.stdout.splitlines()[-1])
    config = OmegaConf.from_dotlist(captured["args"])
    assert config.actor_rollout_ref.model.path == str(model)
    assert config.trainer.val_only and config.trainer.audio_benchmark.only
    assert config.trainer.audio_benchmark.max_samples is None
    assert config.trainer.resume_mode == "disable"
    assert config.data.train_files == config.data.val_files == []
    assert config.distillation.enabled is False
    assert captured["env"]["MODEL_PATH"] == str(model)
    assert captured["env"]["AUDIO_BENCHMARK_MAX_SAMPLES"] == "null"
    assert "GAGE_EVAL_MAX_SAMPLES" not in captured["env"]
    assert not Path(env["AUDIO_FULL_RUN_DIR"]).exists()


def test_full_audio_launcher_preserves_existing_run_logs(full_launcher):
    launcher, env, model = full_launcher
    env["CONFIG_ONLY"] = "0"
    result = subprocess.run(["bash", str(launcher)], env=env, text=True, capture_output=True, check=True)
    log = Path(env["AUDIO_FULL_RUN_DIR"]) / "launcher.log"
    previous = log.read_bytes()
    assert str(model) in result.stdout
    serving_model = Path(json.loads(log.read_text().splitlines()[-1])["env"]["MODEL_PATH"])
    assert serving_model == log.parent / "model"
    assert (serving_model / "audio_tower.safetensors").is_symlink()
    assert (serving_model / "audio_tower.safetensors").resolve() == model / "audio_tower.safetensors"
    assert (serving_model / "processor_config.json").resolve() == model / "processor_config.json"
    assert (serving_model / "preprocessor_config.json").read_bytes() == (
        Path(env["AUDIO_PROCESSOR_SOURCE"]) / "preprocessor_config.json"
    ).read_bytes()
    assert not (model / "preprocessor_config.json").exists()

    rerun = subprocess.run(["bash", str(launcher)], env=env, text=True, capture_output=True)

    assert rerun.returncode != 0
    assert "Refusing to reuse" in rerun.stderr
    assert log.read_bytes() == previous


def test_full_audio_launcher_rejects_incomplete_checkpoint(full_launcher):
    launcher, env, model = full_launcher
    (model / "audio_tower.safetensors").unlink()

    result = subprocess.run(["bash", str(launcher)], env=env, text=True, capture_output=True)

    assert result.returncode != 0
    assert "missing weight shards" in result.stderr
    assert not Path(env["AUDIO_FULL_RUN_DIR"]).exists()


@pytest.mark.parametrize("marker_present", [True, False])
def test_full_audio_launcher_can_select_an_older_step(full_launcher, marker_present):
    launcher, env, model = full_launcher
    previous = model.parents[3] / "global_step_16" / "actor" / "model" / "huggingface"
    shutil.copytree(model, previous)
    env["CHECKPOINT_STEP"] = "16"
    if not marker_present:
        (Path(env["CHECKPOINT_ROOT"]) / "latest_checkpointed_iteration.txt").unlink()

    result = subprocess.run(["bash", str(launcher)], env=env, text=True, capture_output=True, check=True)

    captured = json.loads(result.stdout.splitlines()[-1])
    assert captured["env"]["MODEL_PATH"] == str(previous)


def test_full_audio_launcher_rejects_incompatible_processor_source(full_launcher):
    launcher, env, _model = full_launcher
    (Path(env["AUDIO_PROCESSOR_SOURCE"]) / "config.json").write_text('{"audio_config": {"num_mel_bins": 80}}')

    result = subprocess.run(["bash", str(launcher)], env=env, text=True, capture_output=True)

    assert result.returncode != 0
    assert "audio_config does not match" in result.stderr
    assert not Path(env["AUDIO_FULL_RUN_DIR"]).exists()


def test_complete_audio_checkpoint_does_not_need_processor_source(full_launcher):
    launcher, env, model = full_launcher
    shutil.copyfile(
        Path(env["AUDIO_PROCESSOR_SOURCE"]) / "preprocessor_config.json", model / "preprocessor_config.json"
    )
    env.update(CONFIG_ONLY="0", AUDIO_PROCESSOR_SOURCE="/nonexistent")

    result = subprocess.run(["bash", str(launcher)], env=env, text=True, capture_output=True, check=True)

    captured = json.loads(result.stdout.splitlines()[-1])
    assert captured["env"]["MODEL_PATH"] == str(model)
    assert not (Path(env["AUDIO_FULL_RUN_DIR"]) / "model").exists()


def _copy_checkpoint(model: Path, step: int) -> Path:
    destination = model.parents[3] / f"global_step_{step}" / "actor" / "model" / "huggingface"
    if destination.parents[2].exists():
        shutil.rmtree(destination.parents[2])
    return Path(shutil.copytree(model, destination))


def _captured_runs(result: subprocess.CompletedProcess) -> list[dict]:
    return [json.loads(line) for line in result.stdout.splitlines() if line.startswith('{"args":')]


def test_full_audio_all_preview_uses_numeric_order_and_skips_unsafe_exports(full_launcher):
    launcher, env, model = full_launcher
    for step in (2, 10, 18):
        _copy_checkpoint(model, step)
    root = Path(env["CHECKPOINT_ROOT"])
    (root / "global_step_6" / "actor" / "model" / "huggingface").mkdir(parents=True)
    invalid_config = _copy_checkpoint(model, 7)
    (invalid_config / "config.json").write_text("{broken json")
    no_audio_tower = _copy_checkpoint(model, 8)
    (no_audio_tower / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.weight": "audio_tower.safetensors"}})
    )
    (root / "global_step_invalid").mkdir()
    env["CHECKPOINT_STEP"] = ""

    result = subprocess.run(["bash", str(launcher)], env=env, text=True, capture_output=True, check=True)

    captures = _captured_runs(result)
    assert [Path(item["env"]["MODEL_PATH"]).parents[2].name for item in captures] == [
        "global_step_2", "global_step_10", "global_step_17"
    ]
    for skipped in ("global_step_6", "global_step_7", "global_step_8", "global_step_18", "global_step_invalid"):
        assert skipped in result.stderr
    assert not Path(env["AUDIO_FULL_RUN_DIR"]).exists()


@pytest.mark.parametrize("config_only", ["0", "1"])
def test_full_audio_all_rejects_missing_marker_before_creating_run(full_launcher, config_only):
    launcher, env, model = full_launcher
    _copy_checkpoint(model, 1)
    _copy_checkpoint(model, 23)
    (Path(env["CHECKPOINT_ROOT"]) / "latest_checkpointed_iteration.txt").unlink()
    env.pop("CHECKPOINT_STEP")
    env["CONFIG_ONLY"] = config_only

    result = subprocess.run(["bash", str(launcher)], env=env, text=True, capture_output=True)

    assert result.returncode != 0
    assert "latest_checkpointed_iteration.txt" in result.stderr
    assert not _captured_runs(result)
    assert not Path(env["AUDIO_FULL_RUN_DIR"]).exists()


def test_full_audio_all_rejects_no_eligible_exports_before_creating_run(full_launcher):
    launcher, env, model = full_launcher
    (model / "audio_tower.safetensors").unlink()
    env.update(CHECKPOINT_STEP="", CONFIG_ONLY="0")

    result = subprocess.run(["bash", str(launcher)], env=env, text=True, capture_output=True)

    assert result.returncode != 0
    assert not _captured_runs(result)
    assert not Path(env["AUDIO_FULL_RUN_DIR"]).exists()


def test_full_audio_all_uses_isolated_child_runs_and_forwards_exact_arguments(full_launcher):
    launcher, env, model = full_launcher
    previous = _copy_checkpoint(model, 16)
    env.update(CHECKPOINT_STEP="", CONFIG_ONLY="0", EXPERIMENT_NAME="custom-audio-evaluation")
    overrides = [
        "actor_rollout_ref.rollout.temperature=0.4",
        'trainer.project_name="audio eval $literal `literal`"',
        "trainer.val_only=False",
    ]

    result = subprocess.run(["bash", str(launcher), *overrides], env=env, text=True, capture_output=True, check=True)

    run_dir = Path(env["AUDIO_FULL_RUN_DIR"])
    captures = _captured_runs(result)
    assert len(captures) == 2
    for step, source, captured in zip((16, 17), (previous, model), captures, strict=True):
        child_dir = run_dir / f"global_step_{step}"
        assert captured["args"][:len(overrides)] == overrides
        config = OmegaConf.from_dotlist(captured["args"])
        assert config.trainer.val_only
        assert captured["env"]["EXPERIMENT_NAME"] == f"custom-audio-evaluation_step{step}"
        for variable, suffix in (
            ("OUTPUT_DIR", "trainer"),
            ("ROLLOUT_DATA_DIR", "rollouts"),
            ("VALIDATION_DATA_DIR", "validation"),
            ("AUDIO_BENCHMARK_OUTPUT_DIR", "benchmarks"),
        ):
            assert captured["env"][variable] == str(child_dir / suffix)
        serving_model = Path(captured["env"]["MODEL_PATH"])
        assert serving_model == child_dir / "model"
        assert (serving_model / "audio_tower.safetensors").resolve() == source / "audio_tower.safetensors"
        assert (child_dir / "launcher.log").is_file()
    summary = json.loads((run_dir / "checkpoint_summary.json").read_text())
    assert summary["checkpoint_root"] == env["CHECKPOINT_ROOT"]
    assert summary["status"] == "completed"
    expected = [
        {
            "step": step,
            "checkpoint_path": str(source),
            "run_dir": str(run_dir / f"global_step_{step}"),
            "experiment_name": f"custom-audio-evaluation_step{step}",
            "status": "completed",
            "exit_code": 0,
        }
        for step, source in ((16, previous), (17, model))
    ]
    assert [
        {key: checkpoint[key] for key in expected_checkpoint}
        for checkpoint, expected_checkpoint in zip(summary["checkpoints"], expected, strict=True)
    ] == expected


def test_full_audio_all_continues_after_failure_and_reports_nonzero(full_launcher):
    launcher, env, model = full_launcher
    _copy_checkpoint(model, 16)
    run_dir = Path(env["AUDIO_FULL_RUN_DIR"])
    env.update(
        CHECKPOINT_STEP="",
        CONFIG_ONLY="0",
        STUB_FAIL_STEP="16",
        STUB_SUMMARY_PATH=str(run_dir / "checkpoint_summary.json"),
    )

    result = subprocess.run(["bash", str(launcher)], env=env, text=True, capture_output=True)

    assert result.returncode != 0
    captures = _captured_runs(result)
    assert [capture["env"]["CHECKPOINT_STEP"] for capture in captures] == ["16", "17"]
    assert [item["status"] for item in captures[0]["summary"]["checkpoints"]] == ["running", "pending"]
    assert [item["status"] for item in captures[1]["summary"]["checkpoints"]] == ["failed", "running"]
    summary = json.loads((run_dir / "checkpoint_summary.json").read_text())
    assert summary["status"] == "completed_with_failures"
    assert [(item["step"], item["status"], item["exit_code"]) for item in summary["checkpoints"]] == [
        (16, "failed", 7),
        (17, "completed", 0),
    ]


def test_full_audio_all_refuses_existing_parent_and_preserves_logs(full_launcher):
    launcher, env, model = full_launcher
    _copy_checkpoint(model, 16)
    env.update(CHECKPOINT_STEP="", CONFIG_ONLY="0")
    subprocess.run(["bash", str(launcher)], env=env, text=True, capture_output=True, check=True)
    run_dir = Path(env["AUDIO_FULL_RUN_DIR"])
    previous = {path: path.read_bytes() for path in run_dir.rglob("*.log")}
    summary_path = run_dir / "checkpoint_summary.json"
    previous[summary_path] = summary_path.read_bytes()

    result = subprocess.run(["bash", str(launcher)], env=env, text=True, capture_output=True)

    assert result.returncode != 0
    assert "Refusing to reuse" in result.stderr
    assert not _captured_runs(result)
    assert previous
    assert {path: path.read_bytes() for path in previous} == previous


def test_full_audio_all_preview_preserves_existing_run_directory(full_launcher):
    launcher, env, model = full_launcher
    _copy_checkpoint(model, 16)
    env.update(CHECKPOINT_STEP="", EXPERIMENT_NAME="preview-parent")
    run_dir = Path(env["AUDIO_FULL_RUN_DIR"])
    run_dir.mkdir()
    sentinel = run_dir / "existing.log"
    sentinel.write_text("preserve existing results\n")

    result = subprocess.run(["bash", str(launcher)], env=env, text=True, capture_output=True, check=True)

    captures = _captured_runs(result)
    assert [item["env"]["EXPERIMENT_NAME"] for item in captures] == [
        "preview-parent_step16", "preview-parent_step17"
    ]
    assert list(run_dir.iterdir()) == [sentinel]
    assert sentinel.read_text() == "preserve existing results\n"


def test_full_audio_all_uses_initial_checkpoint_snapshot(full_launcher):
    launcher, env, model = full_launcher
    _copy_checkpoint(model, 10)
    env.update(CHECKPOINT_STEP="", CONFIG_ONLY="0", STUB_ADD_STEP="16")

    result = subprocess.run(["bash", str(launcher)], env=env, text=True, capture_output=True, check=True)

    assert (Path(env["CHECKPOINT_ROOT"]) / "global_step_16").is_dir()
    assert [item["env"]["CHECKPOINT_STEP"] for item in _captured_runs(result)] == ["10", "17"]
    run_dir = Path(env["AUDIO_FULL_RUN_DIR"])
    summary = json.loads((run_dir / "checkpoint_summary.json").read_text())
    assert [item["step"] for item in summary["checkpoints"]] == [10, 17]
    assert not (run_dir / "global_step_16").exists()


def test_full_audio_all_rejects_invalid_completion_marker_before_creating_run(full_launcher):
    launcher, env, _model = full_launcher
    (Path(env["CHECKPOINT_ROOT"]) / "latest_checkpointed_iteration.txt").write_text("unfinished\n")
    env.update(CHECKPOINT_STEP="", CONFIG_ONLY="0")

    result = subprocess.run(["bash", str(launcher)], env=env, text=True, capture_output=True)

    assert result.returncode != 0
    assert "latest_checkpointed_iteration.txt" in result.stderr
    assert not _captured_runs(result)
    assert not Path(env["AUDIO_FULL_RUN_DIR"]).exists()

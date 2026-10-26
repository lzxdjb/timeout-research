"""Check HF integrity and orchestration without contacting executors or GPUs."""

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from safetensors.numpy import save_file

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "running_script_4/validate_thinking_penalty_python_tests_hf.sh"


@pytest.fixture
def runner(tmp_path, monkeypatch):
    source = tmp_path / "source"
    train = source / "data/swe_benchmarks/verl/swe_rebench_v2_reference_verified_snapshot_2000.parquet"
    train.parent.mkdir(parents=True)
    train.touch()
    validation = tmp_path / "validation.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [{"extra_info": {"task_id": f"task-{i}"}, "metric_data_source": "swe_bench_verified"} for i in range(2)]
        ),
        validation,
    )
    monkeypatch.setenv("TARGET_VERL", str(ROOT))
    monkeypatch.setenv("SWE_SOURCE", str(source))
    monkeypatch.setenv("VAL_FILES", json.dumps([str(validation)]))
    monkeypatch.setenv("SWE_AGENT_VALIDATION_EXPECTED_TASKS", "2")
    monkeypatch.delenv("CONFIG_ONLY", raising=False)
    module = types.ModuleType("hf_checkpoint_validation")
    body = SCRIPT.read_text().split("<<'PY'\n", 1)[1].rsplit("\nPY\n", 1)[0]
    exec(compile(body, str(SCRIPT), "exec"), module.__dict__)
    return module


def make_hf(path):
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}")
    (path / "tokenizer_config.json").write_text("{}")
    (path / "tokenizer.json").write_text("{}")
    save_file({"weight": np.zeros(2, dtype=np.float32)}, path / "model-1.safetensors")
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"weight": "model-1.safetensors"}}))
    return path


def test_discovery_handles_empty_dcp_legacy_layout_and_explicit_steps(runner, tmp_path):
    make_hf(tmp_path / "global_step_34/actor/model/huggingface")
    make_hf(tmp_path / "global_step_35/actor/huggingface")
    (tmp_path / "global_step_34/actor/model/dist_ckpt").mkdir()
    dcp = tmp_path / "global_step_35/actor/dist_ckpt"
    dcp.mkdir()
    (dcp / ".metadata").touch()
    (dcp / "__0_0.distcp").touch()
    args = SimpleNamespace(hf_path=None, checkpoint_root=tmp_path, steps=None)
    assert [step for step, _ in runner.select_checkpoints(args)] == [34]
    args.steps = "34-35"
    selected = runner.select_checkpoints(args)
    assert [step for step, _ in selected] == [34, 35]
    assert selected[1][1] == tmp_path / "global_step_35/actor/huggingface"
    args.hf_path = selected[1][1]
    assert runner.select_checkpoints(args) == [selected[1]]


@pytest.mark.parametrize("damage", ["missing_shard", "truncated_shard", "missing_tensor", "escaping_shard"])
def test_integrity_rejects_incomplete_exports(runner, tmp_path, damage):
    hf = make_hf(tmp_path / "hf")
    assert runner.check_hf(hf) == {"shards": 1, "tensors": 1}
    shard = hf / "model-1.safetensors"
    if damage == "missing_shard":
        shard.unlink()
    elif damage == "truncated_shard":
        shard.write_bytes(shard.read_bytes()[:-1])
    else:
        mapping = {"missing": "model-1.safetensors"} if damage == "missing_tensor" else {"weight": "../outside"}
        (hf / "model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}))
    with pytest.raises((OSError, ValueError)):
        runner.check_hf(hf)


def test_integrity_accepts_hf_cache_symlinks(runner, tmp_path):
    hf = make_hf(tmp_path / "hf")
    shard = hf / "model-1.safetensors"
    blob = tmp_path / "blob"
    shard.rename(blob)
    shard.symlink_to(blob)
    assert runner.check_hf(hf) == {"shards": 1, "tensors": 1}


@pytest.mark.parametrize("existing_output", [True, False])
def test_all_inputs_checked_before_any_launch(runner, tmp_path, monkeypatch, existing_output):
    checkpoints = tmp_path / "checkpoints"
    make_hf(checkpoints / "global_step_34/actor/model/huggingface")
    invalid = make_hf(checkpoints / "global_step_35/actor/model/huggingface")
    (invalid / "model-1.safetensors").unlink()
    output = tmp_path / "output"
    if existing_output:
        output.mkdir()
    monkeypatch.setattr(runner, "service_preflight", lambda env: pytest.fail("Preflight ran before input checks"))
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **kw: pytest.fail("Invalid inputs reached launch"))
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--checkpoint-root", str(checkpoints), "--output-dir", str(output)])
    with pytest.raises((OSError, ValueError)):
        runner.main()
    assert output.exists() == existing_output
    if existing_output:
        assert not list(output.iterdir())


@pytest.mark.parametrize("mode", ["--dry-run", "--config-only"])
def test_preview_never_contacts_services_or_writes_artifacts(runner, tmp_path, monkeypatch, mode):
    hf = make_hf(tmp_path / "hf")
    output = tmp_path / "output"
    calls = []

    def no_services(*args):
        pytest.fail("Preview contacted execution services")

    def launch(command, **kwargs):
        calls.append(command)
        assert kwargs["env"]["CONFIG_ONLY"] == "1"
        assert "trainer.val_only=True" in command
        assert "trainer.resume_mode=disable" in command

    monkeypatch.setattr(runner, "service_preflight", no_services)
    monkeypatch.setattr(runner.subprocess, "run", launch)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--hf-path", str(hf), "--output-dir", str(output), mode])
    runner.main()
    assert bool(calls) == (mode == "--config-only")
    assert not output.exists()


@pytest.mark.parametrize("outcome", ["incomplete", "failed", "missing_audit", "invalid_audit"])
def test_sequential_results_preserve_bounds_and_stop_after_failure(runner, tmp_path, monkeypatch, outcome):
    checkpoints = tmp_path / "checkpoints"
    for step in (34, 35):
        make_hf(checkpoints / f"global_step_{step}/actor/model/huggingface")
    output = tmp_path / "output"
    calls = []
    monkeypatch.setattr(runner, "service_preflight", lambda env: [{"ok": True}])

    def launch(command, **kwargs):
        assert kwargs["env"]["CONFIG_ONLY"] == "0"
        assert "trainer.val_only=True" in command
        assert "trainer.val_before_train=True" in command
        assert "trainer.resume_mode=disable" in command
        assert "actor_rollout_ref.actor.megatron.use_dist_checkpointing=False" in command
        calls.append(kwargs["env"]["MODEL_PATH"])
        audit = Path(next(arg.split("=", 1)[1] for arg in command if arg.startswith("trainer.validation_data_dir=")))
        audit /= "validation_audit/0.json"
        if outcome != "missing_audit":
            audit.parent.mkdir(parents=True)
            prefix = "val-core/swe_bench_verified/fixed_roster/"
            metrics = {
                prefix + key: value
                for key, value in {
                    "success_lower@1": 0.5,
                    "success_upper@1": 1.0,
                    "coverage": 0.5,
                    "successes": 1,
                    "known_failures": 0,
                    "unresolved": 1,
                }.items()
            }
            audit.write_text(
                "{" if outcome == "invalid_audit" else json.dumps({"status": "incomplete", "metrics": metrics})
            )
        return SimpleNamespace(returncode=7 if outcome == "failed" else 0)

    monkeypatch.setattr(runner.subprocess, "run", launch)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--checkpoint-root", str(checkpoints), "--output-dir", str(output)])
    if outcome == "incomplete":
        runner.main()
        assert len(calls) == 2
    else:
        with pytest.raises(ValueError, match="stopped before the next checkpoint"):
            runner.main()
        assert len(calls) == 1
    manifest = json.loads((output / "manifest.json").read_text())
    assert [item["original_step"] for item in manifest["checkpoints"]] == [34, 35]
    rows = json.loads((output / "summary.json").read_text())
    assert [row["original_step"] for row in rows] == ([34, 35] if outcome == "incomplete" else [34])
    if outcome in ("incomplete", "failed"):
        assert rows[0]["success_lower"] == 0.5
        assert rows[0]["success_upper"] == 1.0
        assert rows[0]["unresolved"] == 1
    assert (output / "summary.csv").is_file()

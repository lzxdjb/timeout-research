"""Reasoning profile validation selection and early training-health visibility."""

import json
import math
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from verl.trainer.ppo import general_reasoning_benchmark
from verl.trainer.ppo.v1 import trainer_base
from verl.trainer.ppo.v1.trainer_base import PPOTrainer


class _StubTrainer(PPOTrainer):
    def on_step_end(self):
        pytest.fail("Validation-only evaluation must not train")

    def on_sample_end(self):
        pytest.fail("Validation-only evaluation must not sample training data")


def _trainer(reasoning=None):
    trainer = _StubTrainer.__new__(_StubTrainer)
    trainer.config = OmegaConf.create(
        {
            "trainer": {
                "audio_benchmark": {"enabled": True, "only": True},
                "validation_data_dir": "/results/validation",
                "val_only": True,
                "val_before_train": False,
                "project_name": "reasoning-tests",
                "experiment_name": "validation-only",
                "logger": ["console"],
            },
            # Standalone evaluation does not need teacher endpoints or scoring.
            "distillation": {"enabled": False},
            "data": {"max_prompt_length": 8192, "max_response_length": 252000},
            "actor_rollout_ref": {"rollout": {"max_model_len": 260192}},
        }
    )
    if reasoning is not None:
        trainer.config.trainer.general_reasoning = reasoning
    trainer.global_steps = 0
    trainer.tokenizer = object()
    trainer.llm_server_manager = SimpleNamespace(get_addresses=lambda: ["student-a:8000", "student-b:8000"])
    trainer._run_optional_audio_validation = Mock(return_value={"val-audio/completed": 1.0})
    return trainer


def test_audio_and_reasoning_validate_without_swe_or_teachers(monkeypatch):
    trainer = _trainer({"enabled": True, "temperature": 0.6, "top_k": 20, "top_p": 0.95})
    reasoning = Mock(return_value={"val-general/completed": 1.0})
    monkeypatch.setattr(general_reasoning_benchmark, "run_general_reasoning_validation", reasoning)

    # No replay buffer or SWE validation dataset exists on this trainer. The
    # standalone path must run both suites against the current student replicas.
    metrics = trainer._validate()

    assert metrics == {"val-audio/completed": 1.0, "val-general/completed": 1.0}
    trainer._run_optional_audio_validation.assert_called_once_with()
    reasoning.assert_called_once_with(
        {
            "enabled": True,
            "temperature": 0.6,
            "top_k": 20,
            "top_p": 0.95,
            "response_length": 252000,
            "max_prompt_length": 8192,
            "max_model_len": 260192,
            "output_dir": "/results/validation/general_reasoning",
        },
        ["student-a:8000", "student-b:8000"],
        0,
        tokenizer=trainer.tokenizer,
    )
    assert "output_dir" not in trainer.config.trainer.general_reasoning


@pytest.mark.parametrize("reasoning", [None, {"enabled": False}])
@pytest.mark.parametrize("audio_exclusion", [False, True])
def test_existing_audio_only_profile_still_runs_only_audio(monkeypatch, reasoning, audio_exclusion):
    trainer = _trainer(reasoning)
    trainer.config.trainer.audio_opd = {"exclude_incomplete_sequences": audio_exclusion}
    runner = Mock(side_effect=AssertionError("Disabled reasoning must not run"))
    monkeypatch.setattr(general_reasoning_benchmark, "run_general_reasoning_validation", runner)
    assert trainer._validate() == {"val-audio/completed": 1.0}
    runner.assert_not_called()
    trainer._run_optional_audio_validation.assert_called_once_with()


def test_reasoning_helper_keeps_explicit_limits_and_output(monkeypatch):
    config = {
        "enabled": True,
        "response_length": 1234,
        "max_prompt_length": 567,
        "max_model_len": 2000,
        "output_dir": "/results/reasoning-custom",
        "max_samples": -1,
    }
    trainer = _trainer(config)
    runner = Mock(return_value={"val-general/completed": 1.0})
    monkeypatch.setattr(general_reasoning_benchmark, "run_general_reasoning_validation", runner)

    assert trainer._run_optional_general_reasoning_validation() == {"val-general/completed": 1.0}
    assert runner.call_args.args[0] == config
    assert OmegaConf.to_container(trainer.config.trainer.general_reasoning) == config


def test_standalone_validation_skips_training_and_swe_datasets():
    trainer = _trainer({"enabled": True})
    trainer._set_training_steps = Mock()
    # Deliberately omit file paths: standalone evaluation must not read them.
    trainer._init_dataloader()
    trainer._set_training_steps.assert_called_once_with(1)
    assert trainer.train_dataset is trainer.val_dataset is None
    assert trainer.train_dataloader is trainer.val_dataloader is None
    assert trainer.train_dataloader_it is None
    assert trainer.steps_per_epoch == 1


def test_standalone_fit_validates_and_exits_with_no_initial_validation_flag(monkeypatch):
    trainer = _trainer({"enabled": True})
    trainer.replay_buffer = SimpleNamespace(cancel_fn=None)
    trainer.on_validate_begin = Mock()
    trainer.on_validate_end = Mock()
    trainer._shutdown_dump_executor = Mock()
    logger = Mock()
    monkeypatch.setattr(trainer_base, "Tracking", Mock(return_value=logger))
    monkeypatch.setattr(trainer_base, "ValidationGenerationsLogger", Mock())
    monkeypatch.setattr(trainer_base, "DapoFilteredRewardTableLogger", Mock())
    monkeypatch.setattr(trainer_base.SkipManager, "init", Mock())
    monkeypatch.setattr(
        general_reasoning_benchmark,
        "run_general_reasoning_validation",
        Mock(return_value={"val-general/completed": 1.0}),
    )
    agent_loop = SimpleNamespace(cancel_sequences=Mock())

    trainer.fit(agent_loop)

    trainer.on_validate_begin.assert_called_once_with()
    trainer.on_validate_end.assert_called_once_with()
    trainer._shutdown_dump_executor.assert_called_once_with()
    logger.log.assert_called_once_with(
        data={"val-audio/completed": 1.0, "val-general/completed": 1.0}, step=0, commit=True
    )
    assert trainer.global_steps == 0


@pytest.mark.parametrize("reasoning,snapshot", [(True, False), (False, True)])
def test_training_snapshot_preserves_scalar_metrics_and_reports_nonfinite(tmp_path, caplog, reasoning, snapshot):
    trainer = _trainer({"enabled": reasoning})
    trainer.config.trainer.training_metrics_snapshot = {"enabled": snapshot}
    trainer.config.trainer.default_local_dir = str(tmp_path)
    trainer.global_steps = 3
    metrics = {
        "actor/pg_loss": -0.125,
        "actor/grad_norm": np.float32(1.25),
        "actor/lr": torch.tensor(1e-6, dtype=torch.float64),
        "actor/distillation/loss": float("nan"),
        "actor/distillation/audio/loss": float("inf"),
        "actor/distillation/general/loss": -float("inf"),
        "distillation/general/accepted_sequences": np.int64(4),
        "sync/updated": True,
        "private_table": [{"response": "not a scalar metric"}],
        "vector": torch.tensor([1.0, 2.0]),
    }
    with caplog.at_level("INFO", logger=trainer_base.__name__):
        trainer._write_reasoning_training_metrics_snapshot(metrics)

    payload = json.loads((tmp_path / "training_metrics/step-3.json").read_text())
    assert payload == {
        "step": 3,
        "metrics": {
            "actor/pg_loss": -0.125,
            "actor/grad_norm": 1.25,
            "actor/lr": 1e-6,
            "actor/distillation/loss": "NaN",
            "actor/distillation/audio/loss": "Infinity",
            "actor/distillation/general/loss": "-Infinity",
            "distillation/general/accepted_sequences": 4,
            "sync/updated": True,
        },
        "non_finite_metrics": [
            "actor/distillation/audio/loss",
            "actor/distillation/general/loss",
            "actor/distillation/loss",
        ],
    }
    assert math.isnan(metrics["actor/distillation/loss"])
    assert "Pre-validation training metrics: step=3" in caplog.text
    assert "actor/grad_norm" in caplog.text
    assert "NaN" in caplog.text


@pytest.mark.parametrize("reasoning", [None, {"enabled": False}])
@pytest.mark.parametrize("audio_exclusion", [False, True])
def test_existing_profiles_do_not_write_training_snapshots(tmp_path, reasoning, audio_exclusion):
    trainer = _trainer(reasoning)
    trainer.config.trainer.audio_opd = {"exclude_incomplete_sequences": audio_exclusion}
    trainer.config.trainer.default_local_dir = str(tmp_path)
    trainer._write_reasoning_training_metrics_snapshot({"actor/pg_loss": 1.0})
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("reasoning,snapshot", [(True, False), (False, True)])
def test_snapshot_is_available_before_validation_and_includes_weight_sync(tmp_path, monkeypatch, reasoning, snapshot):
    trainer = _trainer({"enabled": reasoning})
    trainer.config.trainer.training_metrics_snapshot = {"enabled": snapshot}
    trainer.config.trainer.update(
        {"default_local_dir": str(tmp_path), "val_only": False, "total_epochs": 1, "save_freq": 1, "test_freq": 2}
    )
    trainer.config.global_profiler = {"steps": None}
    trainer.steps_per_epoch = trainer.total_training_steps = 1
    trainer.replay_buffer = SimpleNamespace(cancel_fn=None)
    trainer._reissue_inflight_prompts = Mock()
    trainer.on_train_begin = Mock()
    trainer.on_step_begin = Mock()
    trainer._start_profiling = Mock()
    trainer._stop_profiling = Mock()
    events = []

    def step(metrics, timing_raw):
        events.append("train")
        metrics.update({"actor/grad_norm": 0.25, "actor/pg_loss": -0.1})
        return Mock()

    trainer.step = step
    trainer._save_checkpoint = lambda: events.append("checkpoint")

    def on_step_end():
        events.append("sync")
        trainer._pending_sync_metrics = {"weight_sync/changed_ratio": 0.75}

    trainer.on_step_end = on_step_end

    def on_validate_begin():
        events.append("validation")
        payload = json.loads((tmp_path / "training_metrics/step-1.json").read_text())
        assert payload["metrics"]["actor/grad_norm"] == 0.25
        assert payload["metrics"]["weight_sync/changed_ratio"] == 0.75
        assert events == ["train", "checkpoint", "sync", "validation"]
        raise RuntimeError("Stop before long validation")

    trainer.on_validate_begin = on_validate_begin
    monkeypatch.setattr(trainer_base, "Tracking", Mock())
    monkeypatch.setattr(trainer_base, "ValidationGenerationsLogger", Mock())
    monkeypatch.setattr(trainer_base, "DapoFilteredRewardTableLogger", Mock())
    monkeypatch.setattr(trainer_base.SkipManager, "init", Mock())
    monkeypatch.setattr(trainer_base.SkipManager, "set_step", Mock())
    monkeypatch.setattr(trainer_base, "tqdm", Mock())

    with pytest.raises(RuntimeError, match="Stop before long validation"):
        trainer.fit(SimpleNamespace(cancel_sequences=Mock()))


@pytest.mark.parametrize("reasoning,audio_exclusion", [(False, False), (True, False), (False, True)])
def test_training_exclusion_count_is_opted_in_and_ignores_padding(monkeypatch, reasoning, audio_exclusion):
    monkeypatch.setenv("SWE_AGENT_REPEAT_REWARD_MODE", "off")
    monkeypatch.setenv("VERL_GRPO_GOLDEN_REPLAY_ENABLED", "0")
    trainer = _trainer({"enabled": reasoning})
    trainer.config.trainer.audio_opd = {"exclude_incomplete_sequences": audio_exclusion}
    trainer.config.actor_rollout_ref.rollout.temperature = 1
    trainer.use_reference_policy = trainer.use_critic = False
    batch = trainer_base.KVBatchMeta(
        partition_id="train",
        keys=["complete_0_0", "excluded_0_0", "padding_0_0"],
        tags=[{}, {"audio_opd_candidate_excluded": True}, {"audio_opd_candidate_excluded": True, "is_padding": True}],
    )
    trainer.replay_buffer = SimpleNamespace(sample=lambda **_: (batch, {}), golden_replay_enabled=False)
    trainer.reward_loop_manager = SimpleNamespace(reward_loop_worker_handles=None)
    trainer.on_sample_begin = trainer.on_sample_end = lambda: None
    trainer._balance_batch = trainer._compute_old_log_prob = trainer._compute_advantage = lambda batch, **_: batch
    trainer._compute_reward_colocate = lambda batch, **_: batch
    trainer._has_active_training_samples = lambda _: False

    metrics = {}
    trainer._step_once(metrics, {}, 1)

    name = "training/audio_opd/excluded_incomplete_sequences"
    if reasoning or audio_exclusion:
        assert metrics[name] == 1.0
    else:
        assert name not in metrics

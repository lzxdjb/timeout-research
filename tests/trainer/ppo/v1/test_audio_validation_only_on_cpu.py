"""Exercise audio evaluation without requiring or running SWE validation data."""

from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf

from verl.trainer.ppo.v1 import trainer_base


class _AudioTrainer(trainer_base.PPOTrainer):
    def on_step_end(self):
        pass

    def on_sample_end(self):
        pass


def _trainer(*, val_only=True, audio_only=True):
    trainer = _AudioTrainer.__new__(_AudioTrainer)
    trainer.config = OmegaConf.create(
        {
            "algorithm": {"filter_groups": {"enable": False}},
            "data": {
                "train_files": [],
                "val_files": [],
                "train_batch_size": 2,
                "gen_batch_size": 2,
                "val_batch_size": 1,
                "dataloader_num_workers": 0,
            },
            "trainer": {
                "audio_benchmark": {"enabled": True, "only": audio_only},
                "val_only": val_only,
                "total_epochs": 3,
                "total_training_steps": None,
                "v1": {"sampler": {"sync_refill_failed_groups": False}},
            },
            "actor_rollout_ref": {"actor": {"optim": {"lr": 1e-6}}},
            "critic": {"optim": {"lr": 1e-6}},
        }
    )
    trainer.trainer_mode = "sync"
    trainer.parameter_sync_step = 2
    trainer.tokenizer = trainer.processor = None
    return trainer


def test_audio_evaluation_does_not_read_empty_swe_datasets_or_strict_roster(monkeypatch):
    monkeypatch.setenv("SWE_AGENT_VALIDATION_STRICT_ACCOUNTING", "1")
    create_dataset = MagicMock(side_effect=AssertionError("SWE dataset must not be opened"))
    load_roster = MagicMock(side_effect=AssertionError("SWE roster must not be opened"))
    dataloader = MagicMock(side_effect=AssertionError("SWE dataloader must not be initialized"))
    monkeypatch.setattr(trainer_base, "create_rl_dataset", create_dataset)
    monkeypatch.setattr(trainer_base, "load_roster", load_roster)
    monkeypatch.setattr(trainer_base, "StatefulDataLoader", dataloader)
    trainer = _trainer()

    trainer._init_dataloader()

    create_dataset.assert_not_called()
    load_roster.assert_not_called()
    dataloader.assert_not_called()
    assert trainer.train_dataset is trainer.val_dataset is None
    assert trainer.train_dataloader is trainer.val_dataloader is None
    assert trainer.total_training_steps == 1
    assert trainer.config.actor_rollout_ref.actor.optim.total_training_steps == 2
    assert trainer.config.critic.optim.total_training_steps == 2


@pytest.mark.parametrize("audio_only", [False, True])
def test_training_keeps_its_dataset_and_only_skips_swe_validation_when_requested(monkeypatch, audio_only):
    monkeypatch.setenv("SWE_AGENT_VALIDATION_STRICT_ACCOUNTING", "1")
    monkeypatch.delenv("SWE_AGENT_ROLLOUT_TRAINING_INFRA_FAILURE_GROUP_RATIO_THRESHOLD", raising=False)
    monkeypatch.delenv("VERL_GRPO_GOLDEN_REPLAY_ENABLED", raising=False)
    train_dataset = [{}, {}, {}, {}]
    val_dataset = [{}]
    create_dataset = MagicMock(side_effect=[train_dataset, val_dataset])
    load_roster = MagicMock(return_value=([{}], []))
    dataloader = MagicMock()
    monkeypatch.setattr(trainer_base, "create_rl_dataset", create_dataset)
    monkeypatch.setattr(trainer_base, "load_roster", load_roster)
    monkeypatch.setattr(trainer_base, "StatefulDataLoader", dataloader)
    monkeypatch.setattr(trainer_base, "create_rl_sampler", MagicMock(return_value=None))
    trainer = _trainer(val_only=False, audio_only=audio_only)

    trainer._init_dataloader()

    assert trainer.train_dataset is train_dataset
    assert create_dataset.call_args_list[0].kwargs["is_train"] is True
    assert len(create_dataset.call_args_list) == (1 if audio_only else 2)
    assert dataloader.call_count == (1 if audio_only else 2)
    assert load_roster.call_count == (0 if audio_only else 1)
    assert (trainer.val_dataloader is None) is audio_only
    assert trainer.total_training_steps == 6
    assert trainer.config.actor_rollout_ref.actor.optim.total_training_steps == 12


def test_audio_validation_dispatches_without_swe_roster_or_dataloader(monkeypatch):
    monkeypatch.setenv("SWE_AGENT_VALIDATION_STRICT_ACCOUNTING", "1")
    load_roster = MagicMock(side_effect=AssertionError("SWE roster must not be opened"))
    monkeypatch.setattr(trainer_base, "load_roster", load_roster)
    trainer = _trainer()
    trainer._run_optional_audio_validation = MagicMock(return_value={"val-audio/completed": 1.0})

    assert trainer._validate() == {"val-audio/completed": 1.0}

    trainer._run_optional_audio_validation.assert_called_once_with()
    load_roster.assert_not_called()


def test_audio_only_cannot_silently_disable_all_validation():
    trainer = _trainer()
    trainer.config.trainer.audio_benchmark.enabled = False

    with pytest.raises(ValueError, match=r"only=True requires.*enabled=True"):
        trainer._init_dataloader()


@pytest.mark.parametrize("audio_config", [None, {"enabled": False}, {"enabled": True}])
def test_audio_only_is_opt_in(audio_config):
    config = OmegaConf.create({"trainer": {"audio_benchmark": audio_config}})

    assert trainer_base._audio_validation_only(config) is False


def test_audio_evaluation_runs_once_and_commits_metrics_before_training(monkeypatch):
    tracking = MagicMock()
    monkeypatch.setattr(trainer_base, "Tracking", MagicMock(return_value=tracking))
    monkeypatch.setattr(trainer_base, "ValidationGenerationsLogger", MagicMock())
    monkeypatch.setattr(trainer_base, "DapoFilteredRewardTableLogger", MagicMock())
    monkeypatch.setattr(trainer_base, "SkipManager", MagicMock())
    trainer = MagicMock()
    trainer.config = OmegaConf.create(
        {
            "trainer": {
                "project_name": "audio-debug",
                "experiment_name": "audio-debug",
                "logger": ["console"],
                "val_before_train": False,
                "val_only": True,
                "audio_benchmark": {"enabled": True, "only": True},
            }
        }
    )
    trainer.global_steps = 0
    metrics = {"val-audio/completed": 1.0}
    trainer._validate.return_value = metrics

    trainer_base.PPOTrainer.fit(trainer, MagicMock())

    trainer._validate.assert_called_once_with()
    trainer.on_validate_begin.assert_called_once_with()
    trainer.on_validate_end.assert_called_once_with()
    tracking.log.assert_called_once_with(data=metrics, step=0, commit=True)
    trainer._shutdown_dump_executor.assert_called_once_with()
    trainer.on_train_begin.assert_not_called()
    trainer.step.assert_not_called()

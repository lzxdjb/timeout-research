"""Check completed-step W&B commits without losing same-step tables."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf

from verl.trainer.ppo.v1 import trainer_base
from verl.utils.tracking import Tracking


@pytest.mark.parametrize("val_only", [True, False])
@pytest.mark.parametrize("filtered_counts", [{}, {0.0: 2, 1.0: 3}])
def test_fit_commits_after_validation_and_reward_tables(monkeypatch, val_only, filtered_counts):
    events = []
    tracking = Tracking.__new__(Tracking)
    tracking.logger = {"wandb": MagicMock()}
    tracking.logger["wandb"].log.side_effect = lambda **kwargs: events.append(("metrics", kwargs))
    monkeypatch.setattr(trainer_base, "Tracking", lambda **kwargs: tracking)
    monkeypatch.setattr(trainer_base, "SkipManager", MagicMock())
    monkeypatch.setattr(trainer_base, "tq", MagicMock())
    monkeypatch.setattr(trainer_base, "tqdm", MagicMock())
    monkeypatch.setattr(trainer_base, "marked_timer", lambda *args, **kwargs: nullcontext())
    reward_logger = MagicMock()
    reward_logger.log.side_effect = lambda loggers, counts, step: events.append(("reward_table", step))
    monkeypatch.setattr(trainer_base, "DapoFilteredRewardTableLogger", lambda **kwargs: reward_logger)

    trainer = MagicMock()
    trainer.config = OmegaConf.create(
        {
            "trainer": {
                "project_name": "test",
                "experiment_name": "test",
                "logger": ["wandb"],
                "val_before_train": True,
                "val_only": val_only,
                "total_epochs": 1,
                "save_freq": 0,
                "test_freq": 1,
            },
            "global_profiler": {"steps": None},
        }
    )
    trainer.global_steps = 17
    trainer.steps_per_epoch = 100
    trainer.total_training_steps = 18
    trainer._consume_sync_metrics.return_value = {}

    def validate():
        events.append(("validation_table", trainer.global_steps))
        return {"val-audio/score": 70.0}

    def step(metrics, timing):
        metrics.update({"loss": 0.5, trainer_base.DAPO_FILTERED_REWARD_COUNTS_KEY: filtered_counts})
        return SimpleNamespace(keys=["sample"], partition_id="train")

    trainer._validate.side_effect = validate
    trainer.step.side_effect = step
    trainer_base.PPOTrainer.fit(trainer, MagicMock())

    expected = [
        ("validation_table", 17),
        ("metrics", {"data": {"val-audio/score": 70.0}, "step": 17, "commit": True}),
    ]
    if not val_only:
        expected.append(("validation_table", 18))
        if filtered_counts:
            expected.append(("reward_table", 18))
        expected.append(("metrics", {"data": {"loss": 0.5, "val-audio/score": 70.0}, "step": 18, "commit": True}))
    assert events == expected

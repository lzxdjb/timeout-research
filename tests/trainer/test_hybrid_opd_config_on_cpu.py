"""Mixed OPD retains evaluator requirements and opt-in cutoff cleanup."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from verl.trainer.distillation.config_utils import opd_cancel_on_cutoff, should_skip_hidden_reward
from verl.workers.config.distillation import (
    DistillationConfig,
    DistillationLossConfig,
    DistillationTeacherModelConfig,
    ExternalTeacherConfig,
)
from verl.workers.config.rollout import RolloutConfig


def _namespace(value):
    return (
        SimpleNamespace(**{key: _namespace(item) for key, item in value.items()}) if isinstance(value, dict) else value
    )


@pytest.mark.parametrize("convert", [dict, OmegaConf.create, _namespace])
def test_hybrid_cutoff_accepts_enabled_distillation_without_skipping_evaluation(convert):
    raw = {
        "distillation": {
            "enabled": True,
            "cancel_pending_on_cutoff": True,
            "skip_hidden_reward": False,
            "cutoff_cleanup_timeout_s": 120,
            "distillation_loss": {"use_task_rewards": True},
        },
        "trainer": {"use_v1": True},
    }
    original = deepcopy(raw)
    config = convert(raw)
    assert opd_cancel_on_cutoff(config)
    assert not opd_cancel_on_cutoff(config, validate=True)
    assert not should_skip_hidden_reward(config)
    assert raw == original


def _hybrid_config(**overrides):
    teacher = DistillationTeacherModelConfig(
        backend="external",
        external=ExternalTeacherConfig(base_url="http://127.0.0.1:1", model_name="teacher", max_model_len=16),
        inference=RolloutConfig(name="vllm", prompt_length=3, response_length=4),
    )
    return DistillationConfig(
        enabled=True,
        cancel_pending_on_cutoff=True,
        teacher_models={"teacher_model": teacher},
        distillation_loss=DistillationLossConfig(loss_mode="k1", use_task_rewards=True),
        **overrides,
    )


def test_hybrid_dataclass_keeps_external_teacher_without_gpu_pool():
    config = _hybrid_config()
    assert config.cancel_pending_on_cutoff
    assert config.distillation_loss.use_task_rewards
    assert not config.skip_hidden_reward
    assert not config.has_managed_teachers
    assert config.n_gpus_per_node == config.nnodes == 0


def test_hybrid_still_rejects_skip_hidden_reward():
    with pytest.raises(ValueError, match="skip_hidden_reward requires"):
        _hybrid_config(skip_hidden_reward=True)


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_hybrid_still_requires_positive_finite_cleanup_deadline(timeout):
    with pytest.raises(ValueError, match="positive and finite"):
        _hybrid_config(cutoff_cleanup_timeout_s=timeout)


def test_cancellation_still_rejects_disabled_distillation_and_v0():
    with pytest.raises(ValueError, match="requires enabled distillation"):
        DistillationConfig(cancel_pending_on_cutoff=True)
    with pytest.raises(ValueError, match="requires enabled distillation"):
        opd_cancel_on_cutoff({"distillation": {"cancel_pending_on_cutoff": True}})
    with pytest.raises(ValueError, match="trainer.use_v1=True"):
        opd_cancel_on_cutoff({"distillation": {"enabled": True, "cancel_pending_on_cutoff": True}})

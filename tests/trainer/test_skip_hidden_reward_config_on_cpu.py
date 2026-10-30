"""Guard the opt-in evaluator bypass against reward-dependent training modes."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from verl.trainer.distillation.config_utils import should_skip_hidden_reward
from verl.workers.config.distillation import DistillationConfig, DistillationLossConfig


@pytest.fixture(autouse=True)
def _clear_reward_mode_environment(monkeypatch):
    monkeypatch.delenv("VERL_GRPO_GOLDEN_REPLAY_ENABLED", raising=False)
    monkeypatch.delenv("SWE_AGENT_REPEAT_REWARD_MODE", raising=False)


def _valid_config():
    return {
        "distillation": {
            "enabled": True,
            "skip_hidden_reward": True,
            "distillation_loss": {"use_task_rewards": False},
        },
        "trainer": {"use_v1": True},
        "actor_rollout_ref": {
            "actor": {"timeout_prediction": {"enabled": False}},
            "rollout": {"agent": {"default_agent_loop": "swe_tool_agent"}},
        },
        "algorithm": {"adv_estimator": "grpo", "filter_groups": {"enable": False}},
        "critic": {"enable": False},
    }


def _namespace(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{key: _namespace(item) for key, item in value.items()})
    return value


@pytest.mark.parametrize("convert", [dict, OmegaConf.create, _namespace], ids=["mapping", "hydra", "namespace"])
def test_valid_pure_opd_skips_training_evaluator_and_preserves_validation(convert):
    raw = _valid_config()
    original = deepcopy(raw)
    config = convert(raw)

    assert should_skip_hidden_reward(config) is True
    assert should_skip_hidden_reward(config, validate=True) is False
    assert raw == original


@pytest.mark.parametrize("config", [None, {}, {"distillation": {}}, {"distillation": {"enabled": True}}])
def test_legacy_configurations_keep_hidden_evaluation(config):
    assert should_skip_hidden_reward(config) is False


def test_explicit_opt_out_keeps_reward_dependent_training_unchanged(monkeypatch):
    config = _valid_config()
    config["distillation"]["skip_hidden_reward"] = False
    config["distillation"]["distillation_loss"]["use_task_rewards"] = True
    config["trainer"]["use_v1"] = False
    config["algorithm"]["filter_groups"]["enable"] = True
    config["critic"]["enable"] = True
    monkeypatch.setenv("VERL_GRPO_GOLDEN_REPLAY_ENABLED", "1")
    monkeypatch.setenv("SWE_AGENT_REPEAT_REWARD_MODE", "reward")

    assert should_skip_hidden_reward(config) is False


@pytest.mark.parametrize(
    "path,value,message",
    [
        (("distillation", "enabled"), False, "enabled distillation"),
        (("distillation", "distillation_loss", "use_task_rewards"), True, "use_task_rewards=False"),
        (("trainer", "use_v1"), False, "trainer.use_v1=True"),
        (("actor_rollout_ref", "rollout", "agent", "default_agent_loop"), "tool_agent", "swe_tool_agent"),
        (("algorithm", "filter_groups", "enable"), True, "reward-based group filtering"),
        (("actor_rollout_ref", "actor", "timeout_prediction", "enabled"), True, "timeout reward prediction"),
        (("critic", "enable"), True, "task-reward critic training"),
        (("algorithm", "adv_estimator"), "gae", "task-reward critic training"),
    ],
)
@pytest.mark.parametrize("convert", [dict, _namespace], ids=["mapping", "namespace"])
def test_incompatible_training_modes_raise_before_rollout(path, value, message, convert):
    config = _valid_config()
    node = config
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value

    with pytest.raises(ValueError, match=message):
        should_skip_hidden_reward(convert(config))


@pytest.mark.parametrize("value", ["1", "true", "yes", "on", " TRUE "])
def test_reward_based_golden_replay_is_rejected(monkeypatch, value):
    monkeypatch.setenv("VERL_GRPO_GOLDEN_REPLAY_ENABLED", value)

    with pytest.raises(ValueError, match="golden replay"):
        should_skip_hidden_reward(_valid_config())


@pytest.mark.parametrize("value", ["reward", "hybrid", "judge"])
def test_repetition_reward_mode_is_rejected(monkeypatch, value):
    monkeypatch.setenv("SWE_AGENT_REPEAT_REWARD_MODE", value)

    with pytest.raises(ValueError, match="SWE_AGENT_REPEAT_REWARD_MODE=off"):
        should_skip_hidden_reward(_valid_config())


@pytest.mark.parametrize("enabled,use_task_rewards", [(False, False), (True, True)])
def test_dataclass_rejects_skip_without_pure_distillation(enabled, use_task_rewards):
    with pytest.raises(ValueError, match="enabled distillation with use_task_rewards=False"):
        DistillationConfig(
            enabled=enabled,
            skip_hidden_reward=True,
            distillation_loss=DistillationLossConfig(use_task_rewards=use_task_rewards),
        )


def test_dataclass_default_keeps_hidden_evaluation():
    assert DistillationConfig().skip_hidden_reward is False

# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Read-only predicates for both unresolved Hydra and dataclass configurations."""

import math
import os


def requires_teacher_resource_pool(distillation_config) -> bool:
    """Whether enabled distillation includes a trainer-managed GPU teacher.

    Do not instantiate DistillationConfig here: its preparation rewrites teacher
    inference lengths. Resource allocation must only inspect the configuration.
    """
    if not distillation_config or not distillation_config.get("enabled", False):
        return False
    teachers = dict(distillation_config.get("teacher_models", {}))
    if len(teachers) > 1:
        teachers.pop("teacher_model", None)  # Same default-entry resolution as DistillationConfig.
    # Preserve legacy validation for an enabled but incomplete configuration.
    return not teachers or any(teacher.get("backend", "managed") != "external" for teacher in teachers.values())


def is_pure_distillation(distillation_config) -> bool:
    """True only when task rewards do not participate in the actor objective."""
    return bool(
        distillation_config
        and distillation_config.get("enabled", False)
        and not distillation_config.get("distillation_loss", {}).get("use_task_rewards", True)
    )


def _config_get(config, key, default=None):
    if config is None:
        return default
    return config.get(key, default) if hasattr(config, "get") else getattr(config, key, default)


def opd_cleanup_timeout_s(config) -> float:
    """Bound cancellation acknowledgement, independently of teacher request timeout."""
    value = float(_config_get(_config_get(config, "distillation", {}), "cutoff_cleanup_timeout_s", 120.0))
    if not math.isfinite(value) or value <= 0:
        raise ValueError("distillation.cutoff_cleanup_timeout_s must be positive and finite")
    return value


def opd_cancel_on_cutoff(config, validate: bool = False) -> bool:
    """Enable explicit RPC cancellation for opt-in pure or mixed OPD training."""
    distillation = _config_get(config, "distillation", {})
    if not _config_get(distillation, "cancel_pending_on_cutoff", False):
        return False
    if not _config_get(distillation, "enabled", False):
        raise ValueError("cancel_pending_on_cutoff requires enabled distillation")
    if not _config_get(_config_get(config, "trainer", {}), "use_v1", False):
        raise ValueError("cancel_pending_on_cutoff currently requires trainer.use_v1=True")
    opd_cleanup_timeout_s(config)
    return not validate


def opd_wait_for_scoring_on_cutoff(config, validate: bool = False) -> bool:
    """Opt-in generation-only cutoff; completed candidates drain their scoring."""
    distillation = _config_get(config, "distillation", {})
    if not _config_get(distillation, "wait_for_scoring_on_cutoff", False):
        return False
    if not _config_get(distillation, "cancel_pending_on_cutoff", False):
        raise ValueError("wait_for_scoring_on_cutoff requires cancel_pending_on_cutoff=True")
    return opd_cancel_on_cutoff(config, validate=validate)


def opd_phase_timing_enabled(config, validate: bool = False) -> bool:
    """Phase timing is independent of cutoff and never records validation work."""
    distillation = _config_get(config, "distillation", {})
    if not _config_get(distillation, "phase_timing", False):
        return False
    if not _config_get(distillation, "enabled", False):
        raise ValueError("phase_timing requires enabled distillation")
    if not _config_get(_config_get(config, "trainer", {}), "use_v1", False):
        raise ValueError("phase_timing currently requires trainer.use_v1=True")
    return not validate


def should_skip_hidden_reward(config, validate: bool = False) -> bool:
    """Validate the opt-in SWE/V1 bypass without changing ordinary RL defaults.

    Accept a full trainer config (mapping, dataclass, or namespace). Intentional
    non-evaluation supplies no outcome for reward-dependent training features.
    """
    distillation = _config_get(config, "distillation", {})
    if not _config_get(distillation, "skip_hidden_reward", False):
        return False
    loss = _config_get(distillation, "distillation_loss", {})
    if not _config_get(distillation, "enabled", False) or _config_get(loss, "use_task_rewards", True):
        raise ValueError("skip_hidden_reward requires enabled distillation with use_task_rewards=False")
    if validate:
        return False
    trainer = _config_get(config, "trainer", {})
    if not _config_get(trainer, "use_v1", False):
        raise ValueError("skip_hidden_reward currently requires trainer.use_v1=True")
    actor_rollout = _config_get(config, "actor_rollout_ref", {})
    rollout = _config_get(actor_rollout, "rollout", {})
    agent = _config_get(rollout, "agent", {})
    if _config_get(agent, "default_agent_loop", "swe_tool_agent") != "swe_tool_agent":
        raise ValueError("skip_hidden_reward currently supports swe_tool_agent only")
    algorithm = _config_get(config, "algorithm", {})
    if _config_get(_config_get(algorithm, "filter_groups", {}), "enable", False):
        raise ValueError("skip_hidden_reward is incompatible with reward-based group filtering")
    actor = _config_get(actor_rollout, "actor", {})
    if _config_get(_config_get(actor, "timeout_prediction", {}), "enabled", False):
        raise ValueError("skip_hidden_reward is incompatible with timeout reward prediction")
    critic = _config_get(config, "critic", {})
    if _config_get(critic, "enable", False) or _config_get(algorithm, "adv_estimator") == "gae":
        raise ValueError("skip_hidden_reward is incompatible with task-reward critic training")
    if os.environ.get("VERL_GRPO_GOLDEN_REPLAY_ENABLED", "0").strip().lower() in {"1", "true", "yes", "on"}:
        raise ValueError("skip_hidden_reward is incompatible with reward-based golden replay")
    if os.environ.get("SWE_AGENT_REPEAT_REWARD_MODE", "off").strip().lower() != "off":
        raise ValueError("skip_hidden_reward requires SWE_AGENT_REPEAT_REWARD_MODE=off")
    return True

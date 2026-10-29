"""External OPD resource, filtering, and fatal-error integration without GPUs."""

import asyncio
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.experimental.teacher_loop.errors import TeacherScoringError
from verl.trainer.distillation.config_utils import is_pure_distillation, requires_teacher_resource_pool
from verl.trainer.main_ppo_v0 import BaseTaskRunner
from verl.trainer.ppo.utils import Role
from verl.trainer.ppo.v1 import agent_loop_tq, replay_buffer
from verl.trainer.ppo.v1.trainer_base import PPOTrainer


def _config(*, backend="external", task_rewards=False):
    return OmegaConf.create(
        {
            "actor_rollout_ref": {"model": {}, "actor": {"use_kl_loss": False}, "rollout": {"n": 1}},
            "critic": {"enable": False},
            "algorithm": {"adv_estimator": "grpo", "use_kl_in_reward": False},
            "reward": {"reward_model": {"enable": False, "enable_resource_pool": False}},
            "trainer": {"nnodes": 1, "n_gpus_per_node": 8},
            "distillation": {
                "enabled": True,
                "nnodes": 0 if backend == "external" else 1,
                "n_gpus_per_node": 0 if backend == "external" else 8,
                "teacher_models": {"teacher_model": {"backend": backend}},
                "distillation_loss": {"use_task_rewards": task_rewards},
            },
        }
    )


@pytest.mark.parametrize("backend", ["managed", "external"])
def test_v1_and_legacy_allocate_gpus_only_for_managed_teachers(backend):
    config = _config(backend=backend)
    trainer = SimpleNamespace(config=config)
    PPOTrainer._init_resource_pool_mgr(trainer)
    runner = BaseTaskRunner()
    runner.add_teacher_model_resource_pool(config)
    legacy_pool = runner.init_resource_pool_mgr(config)
    expected = backend == "managed"
    for pools, mapping in (
        (trainer.resource_pool_manager.resource_pool_spec, trainer.mapping),
        (legacy_pool.resource_pool_spec, runner.mapping),
    ):
        assert ("teacher_pool" in pools) is expected
        assert (Role.TeacherModel in mapping) is expected
        assert pools["global_pool"] == [8]


def test_resource_predicates_resolve_default_entry_without_mutating_config():
    distillation = _config().distillation
    distillation.teacher_models.other = {"backend": "external"}
    distillation.teacher_models.teacher_model.backend = "managed"
    original = OmegaConf.to_container(distillation)
    assert not requires_teacher_resource_pool(distillation)
    assert OmegaConf.to_container(distillation) == original
    distillation.teacher_models.managed_teacher = {"backend": "managed"}
    assert requires_teacher_resource_pool(distillation)
    assert is_pure_distillation(distillation)
    distillation.enabled = False
    assert not requires_teacher_resource_pool(distillation)
    assert not is_pure_distillation(distillation)


@pytest.mark.parametrize("task_rewards", [False, True])
def test_only_pure_opd_uses_per_sample_filter_with_grpo_n_one(monkeypatch, task_rewards):
    monkeypatch.setenv("SWE_AGENT_TASK_FILTER_TRAINING_ENABLED", "1")
    monkeypatch.setenv("SWE_AGENT_TASK_FILTER_MIN_GROUP_ATTEMPTS", "8")
    data = DataProto.from_single_dict({"response_mask": torch.ones((2, 3), dtype=torch.int64)})
    data.non_tensor_batch["uid"] = np.asarray(["a", "b"], dtype=object)
    trainer = SimpleNamespace(
        config=_config(task_rewards=task_rewards),
        _extract_v1_infrastructure_metadata=lambda *_args: (
            np.asarray([0, 1], dtype=object),
            np.asarray([0, 3], dtype=object),
        ),
    )
    PPOTrainer._apply_v1_task_filter(trainer, data, [], {})
    assert data.batch["train_sample_mask"].tolist() == [True, task_rewards]


def test_teacher_failure_is_published_as_fatal_postprocessing_error(monkeypatch):
    worker_class = agent_loop_tq.AgentLoopWorkerTQ.__ray_metadata__.modified_class
    worker = object.__new__(worker_class)
    worker.config = SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(n=1, val_kwargs=SimpleNamespace(n=1)))
    )
    worker.repeat_coordinator = None
    worker.repeat_finalizers = {}
    worker.repeat_generation_sessions = {}
    worker.repeat_debug_prompts = {}
    worker.repeat_debug = False
    worker.cutoff_session_states = {}
    tags = {}

    async def put(*, key, partition_id, tag):
        tags.setdefault(key, {}).update(tag)

    async def fail(*_args, **_kwargs):
        raise TeacherScoringError("teacher unavailable after bounded retries")

    monkeypatch.setattr(agent_loop_tq.tq, "async_kv_put", put)
    worker._run_agent_loop = fail
    asyncio.run(worker._run_prompt({"uid": "u"}, {}, {"validate": False}))
    assert tags["u"]["status"] == "reward_error"
    assert "bounded retries" in tags["u"]["teacher_error"]


def test_teacher_error_stops_replay_sampling_instead_of_being_refilled(monkeypatch):
    buffer = object.__new__(replay_buffer.ReplayBuffer)
    for name in (
        "partitions",
        "pending_keys",
        "running_keys",
        "generation_complete_keys",
        "reward_pending_keys",
        "finished_keys",
        "failure_keys",
        "completion_ratio_cutoff_keys",
        "prompt_global_steps",
    ):
        setattr(buffer, name, defaultdict(dict))
    monkeypatch.setattr(
        replay_buffer.tq,
        "kv_list",
        lambda: {
            "train": {
                "u": {
                    "is_prompt": True,
                    "status": "reward_error",
                    "global_steps": 0,
                    "teacher_error": "invalid score alignment",
                }
            }
        },
    )
    with pytest.raises(RuntimeError, match="Teacher scoring failed.*invalid score alignment"):
        buffer._sync_metadata_from_transfer_queue()

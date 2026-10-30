"""V1 OPD retains sparse local costs without manufacturing hidden outcomes."""

import json
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from transfer_queue import KVBatchMeta

from verl.trainer.ppo.v1 import trainer_base
from verl.trainer.ppo.v1.trainer_base import PPOTrainer
from verl.trainer.ppo.v1.utils import (
    MetricsAggregator,
    compute_v1_trajectory_debug_metrics,
    hidden_reward_was_skipped,
    v1_success_values,
)
from verl.utils import tensordict_utils as tu
from verl.utils.swe_tool_failure import rejection_metadata, token_digest


class StubTrainer(PPOTrainer):
    def on_step_end(self):
        pass

    def on_sample_end(self):
        pass


def config():
    return OmegaConf.create(
        {
            "trainer": {"use_v1": True},
            "distillation": {
                "enabled": True,
                "skip_hidden_reward": True,
                "distillation_loss": {"use_task_rewards": False},
            },
            "algorithm": {"adv_estimator": "grpo", "use_kl_in_reward": False, "gamma": 1, "lam": 1},
            "actor_rollout_ref": {
                "model": {},
                "rollout": {"n": 2},
                "actor": {
                    "tool_failure_penalty_enabled": True,
                    "tool_failure_penalty_cost": 0.02,
                    "tool_failure_penalty_cap": 0.30,
                    "tool_failure_penalty_warmup_steps": 0,
                },
            },
        }
    )


def rejection(ids):
    return {
        "version": 1,
        "start": 1,
        "end": 3,
        "token_sha256": token_digest(ids[1:3]),
        "prefix_sha256": token_digest(ids[:3]),
        "proof": rejection_metadata("unknown_tool")["tool_contract_rejection"],
    }


@pytest.mark.parametrize("tag", [{}, {"is_padding": True}, {"completion_ratio_cutoff": True}])
def test_skip_advantage_keeps_costs_and_exclusions(monkeypatch, tag):
    monkeypatch.setenv("SWE_AGENT_TASK_FILTER_TRAINING_ENABLED", "1")
    monkeypatch.setenv("SWE_AGENT_TASK_FILTER_MIN_GROUP_ATTEMPTS", "2")
    monkeypatch.setenv("SWE_AGENT_REPEAT_REWARD_MODE", "off")
    monkeypatch.setenv("VERL_GRPO_GOLDEN_REPLAY_ENABLED", "0")
    trainer = StubTrainer.__new__(StubTrainer)
    trainer.config = config()
    trainer.timeout_prediction_enabled = False
    trainer.global_steps = 1
    ids = [10, 11, 12, 13]
    batch = KVBatchMeta(partition_id="train", keys=["g_0_0", "other_0_0", "infra_0_0"], tags=[tag, {}, {}])
    extras = [
        {
            "tool_failure_events": [rejection(ids)],
            "reward_extra_info": {
                "hidden_reward_skipped": 1,
                "hidden_reward_scored": 0,
                "observed_infrastructure_failure": int(row == 2),
            },
        }
        for row in range(3)
    ]
    queue = tu.get_tensordict(
        {
            "uid": torch.arange(3),
            "responses": torch.nested.as_nested_tensor([torch.tensor(ids)] * 3, layout=torch.jagged),
            "response_mask": torch.nested.as_nested_tensor([torch.ones(4)] * 3, layout=torch.jagged),
            "rm_scores": torch.nested.as_nested_tensor([torch.zeros(4)] * 3, layout=torch.jagged),
            "extra_fields": extras,
        }
    )
    captured = {}

    def get(**kwargs):
        fields = kwargs["select_fields"]
        if len(fields) == 1:
            raise KeyError(fields[0])
        return queue.select(*(field for field in fields if field in queue)).clone()

    def put(**kwargs):
        captured.update(kwargs["fields"])
        return batch

    monkeypatch.setattr(trainer_base.tq, "kv_batch_get", get)
    monkeypatch.setattr(trainer_base.tq, "kv_batch_put", put)
    monkeypatch.setattr(
        trainer_base,
        "compute_advantage_for_multi_trajectories",
        lambda *a, **kw: pytest.fail("Hidden outcome GRPO must not run"),
    )
    metrics = {}
    trainer._compute_advantage(batch, metrics)
    active = not bool(tag)
    assert captured["tool_failure_active"].tolist() == [active, True, False]
    assert captured["train_sample_mask"].tolist() == [active, True, False]
    assert captured["tool_failure_weights"].unbind()[1].sum().item() == pytest.approx(0.02)
    assert captured["tool_failure_weights"].unbind()[2].eq(0).all()
    assert all(row.eq(0).all() for row in captured["advantages"].unbind())
    assert all(row.eq(0).all() for row in captured["returns"].unbind())
    assert metrics["distillation/task_advantage_skipped"] == 1
    assert not any(key.startswith("grpo/") for key in metrics)


def test_skipped_outcome_never_falls_back_to_transport_score():
    extras = [{"reward_extra_info": {"hidden_reward_skipped": 1}}, {}]
    assert hidden_reward_was_skipped(extras[0])
    assert v1_success_values(extras, [0.0, 1.0]) == [None, 1.0]
    extras[0]["reward_extra_info"]["acc"] = 0
    extras[1]["reward_extra_info"] = {"acc": 1}
    assert v1_success_values(extras, [0.0, 1.0]) == [None, 1.0]


def test_dump_writes_null_outcome_for_unscored_row(monkeypatch, tmp_path):
    trainer = StubTrainer.__new__(StubTrainer)
    trainer.tokenizer = SimpleNamespace(pad_token_id=0, decode=lambda ids, **_: str(ids.tolist()))
    trainer._dump_generations = lambda **kw: trainer._write_generations(**kw, global_steps=1)
    batch = KVBatchMeta(partition_id="train", keys=["g_0_0"], tags=[{}])

    def get(**kwargs):
        fields = kwargs["select_fields"]
        if fields == ["extra_fields"]:
            return {
                "extra_fields": SimpleNamespace(
                    tolist=lambda: [
                        {
                            "reward_extra_info": {
                                "hidden_reward_skipped": 1,
                                "hidden_reward_scored": 0,
                            }
                        }
                    ]
                )
            }
        if len(fields) == 1:
            raise KeyError(fields[0])
        return {
            "uid": torch.tensor([0]),
            "prompts": torch.nested.as_nested_tensor([torch.tensor([1])], layout=torch.jagged),
            "responses": torch.nested.as_nested_tensor([torch.tensor([2, 3])], layout=torch.jagged),
            "rm_scores": torch.zeros(1, 2),
        }

    monkeypatch.setattr(trainer_base.tq, "kv_batch_get", get)
    trainer._log_rollout_data(batch, {}, str(tmp_path))
    row = json.loads((tmp_path / "1.jsonl").read_text())
    assert row["score"] is None and row["success_value"] is None


def test_metrics_omit_outcomes_and_preserve_lengths_and_infrastructure(monkeypatch):
    monkeypatch.setenv("SWE_AGENT_REPEAT_REWARD_MODE", "off")
    monkeypatch.setenv("VERL_GRPO_GOLDEN_REPLAY_ENABLED", "0")
    trainer = StubTrainer.__new__(StubTrainer)
    trainer.config, trainer.use_critic, trainer.trainer_mode = config(), False, "sync"
    trainer._rollout_moe_lb_metrics_accumulator = None
    trainer._get_n_gpus_for_throughput = lambda: 8
    batch = KVBatchMeta(partition_id="train", keys=["g_0_0"], tags=[{}])
    data = tu.get_tensordict(
        {
            "prompts": torch.nested.as_nested_tensor([torch.tensor([1])], layout=torch.jagged),
            "responses": torch.nested.as_nested_tensor([torch.tensor([2, 3])], layout=torch.jagged),
            **{
                k: torch.nested.as_nested_tensor([torch.zeros(2)], layout=torch.jagged)
                for k in ("advantages", "returns", "rm_scores")
            },
            "response_mask": torch.nested.as_nested_tensor([torch.ones(2)], layout=torch.jagged),
            "num_turns": torch.ones(1),
        }
    )

    def get(**kwargs):
        if kwargs["select_fields"] == ["extra_fields"]:
            return {
                "extra_fields": SimpleNamespace(
                    tolist=lambda: [
                        {
                            "reward_extra_info": {
                                "hidden_reward_skipped": 1,
                                "hidden_reward_scored": 0,
                            }
                        }
                    ]
                )
            }
        raise KeyError(kwargs["select_fields"][0])

    monkeypatch.setattr(trainer_base.tq, "kv_batch_get", get)
    monkeypatch.setattr(trainer_base, "get_metric_data_with_optional_routed_experts", lambda **_: data.clone())
    monkeypatch.setattr(trainer_base, "compute_moe_lb_metrics", lambda **_: {})
    monkeypatch.setattr(trainer_base, "compute_timing_metrics", lambda **_: {"timing_s/gen": 1.0})
    monkeypatch.setattr(trainer_base, "compute_throughout_metrics", lambda **_: {})
    monkeypatch.setattr(
        trainer_base,
        "compute_data_metrics",
        lambda **_: {
            "critic/score/mean": 0,
            "critic/rewards/mean": 0,
            "critic/advantages/mean": 0,
            "training/partial_reward/score/mean": 0,
            "response_length/mean": 2,
            "training/infrastructure_failure/mean": 0,
        },
    )
    monkeypatch.setattr(trainer_base, "compute_v1_success_ratio_metrics", lambda **_: pytest.fail("No hidden outcome"))
    monkeypatch.setattr(trainer_base, "compute_variance_proxy_metrics", lambda **_: pytest.fail("No PPO signal"))
    metrics = {}
    trainer._compute_metrics(batch, metrics, {}, global_steps=1, epoch=0)
    assert metrics["training/hidden_reward_scored"] == 0
    assert metrics["training/hidden_reward_skipped"] == 1
    assert metrics["trajectory_debug/hidden_reward_skipped_count"] == 1
    assert metrics["trajectory_debug/hidden_reward_scored_count"] == 0
    assert metrics["trajectory_debug/hidden_reward_skipped_fraction"] == 1
    assert metrics["response_length/mean"] == 2
    assert metrics["training/infrastructure_failure/mean"] == 0
    assert not any(k.startswith(("critic/", "training/partial_reward/", "training/success_ratio/")) for k in metrics)
    assert not any(k.startswith(("trajectory_debug/success/", "trajectory_debug/failure/")) for k in metrics)
    assert not any("grpo" in key for key in metrics)


@pytest.mark.parametrize("skip", [False, True])
def test_training_step_omits_colocated_reward_only_when_opted_in(monkeypatch, skip):
    monkeypatch.setenv("SWE_AGENT_REPEAT_REWARD_MODE", "off")
    monkeypatch.setenv("VERL_GRPO_GOLDEN_REPLAY_ENABLED", "0")
    trainer = StubTrainer.__new__(StubTrainer)
    trainer.config = config()
    trainer.config.distillation.skip_hidden_reward = skip
    trainer.config.actor_rollout_ref.rollout.temperature = 1
    trainer.global_steps = 1
    trainer.use_reference_policy = trainer.use_critic = False
    batch = KVBatchMeta(partition_id="train", keys=["g_0_0"], tags=[{}])
    trainer.replay_buffer = SimpleNamespace(sample=lambda **_: (batch, {}), golden_replay_enabled=False)
    trainer.reward_loop_manager = SimpleNamespace(reward_loop_worker_handles=None)
    trainer.on_sample_begin = trainer.on_sample_end = lambda: None
    trainer._balance_batch = trainer._compute_old_log_prob = trainer._compute_advantage = lambda batch, **_: batch
    trainer._has_active_training_samples = lambda _: False
    calls = []

    def reward(batch, **_):
        calls.append(True)
        return batch

    trainer._compute_reward_colocate = reward
    trainer._step_once({}, {}, 1)
    assert len(calls) == int(not skip)


def test_coverage_counts_nonpadding_final_sessions_with_explicit_evidence():
    metrics = compute_v1_trajectory_debug_metrics(
        batch_keys=["a_0_0", "a_0_1", "b_0_0", "c_0_0", "pad_0_0"],
        batch_tags=[{}, {}, {}, {}, {"is_padding": True}],
        extra_fields=[
            {"hidden_reward_scored": 1},
            {"reward_extra_info": {"hidden_reward_skipped": 1, "hidden_reward_scored": 0}},
            {"reward_extra_info": {"hidden_reward_scored": 1}},
            {},
            {"hidden_reward_skipped": 1},
        ],
        scores=[0] * 5,
        success_values=[None] * 5,
        response_tokens=[10] * 5,
        train_mask=[True] * 5,
        expected_rollout_count=1,
        include_grpo_groups=False,
    )
    assert metrics["trajectory_debug/final_sessions"] == 3
    for label in ("skipped", "scored", "unknown"):
        assert metrics[f"trajectory_debug/hidden_reward_{label}_count"] == 1
        assert metrics[f"trajectory_debug/hidden_reward_{label}_fraction"] == pytest.approx(1 / 3)
    assert metrics["trajectory_debug/success/count"] == metrics["trajectory_debug/failure/count"] == 0


def test_coverage_aggregation_sums_counts_and_recomputes_fractions():
    aggregator = MetricsAggregator()
    for sessions, skipped, scored in ((2, 2, 0), (8, 0, 4)):
        metrics = {"trajectory_debug/final_sessions": sessions}
        for label, count in (("skipped", skipped), ("scored", scored), ("unknown", sessions - skipped - scored)):
            metrics[f"trajectory_debug/hidden_reward_{label}_count"] = count
            metrics[f"trajectory_debug/hidden_reward_{label}_fraction"] = count / sessions
        aggregator.add_step_metrics(metrics, sample_count=100)
    # An older worker may omit markers, which supplies no evidence of scoring.
    aggregator.add_step_metrics({"trajectory_debug/final_sessions": 2}, sample_count=100)
    metrics = aggregator.get_aggregated_metrics()
    assert metrics["trajectory_debug/final_sessions"] == 12
    for label, count in (("skipped", 2), ("scored", 4), ("unknown", 6)):
        assert metrics[f"trajectory_debug/hidden_reward_{label}_count"] == count
        assert metrics[f"trajectory_debug/hidden_reward_{label}_fraction"] == pytest.approx(count / 12)

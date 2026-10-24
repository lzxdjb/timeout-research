"""Network/GPU-free contracts for censored SWE validation and its watchdogs."""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from verl.trainer.ppo.v1 import replay_buffer as replay
from verl.trainer.ppo.v1.trainer_base import PPOTrainer
from verl.utils import tensordict_utils as tu
from verl.utils.swe_validation import evaluation_reason, fixed_roster_report, load_roster


def valid(passed=0):
    return dict(evaluation_valid=True, hidden_pass=passed, partial_hidden_reward_counts_complete=True)


def test_fixed_roster_never_drops_unpublished_or_invalid_tasks():
    roster = [dict(task_id=str(i), source="bench") for i in range(5)]
    observations = [
        dict(task_id="0", session_id="0", **valid(1)),
        dict(task_id="1", session_id="0", **valid(0)),
        dict(
            task_id="2",
            session_id="0",
            evaluation_valid=False,
            hidden_pass=0,
            evaluator_invalid_reason="official_test_coverage_incomplete",
        ),
        dict(task_id="3", session_id="0", **valid(1), observed_infrastructure_failure=1),
    ]
    metrics, tasks = fixed_roster_report(roster, observations, 1)
    prefix = "val-core/bench/fixed_roster/"
    assert metrics[prefix + "success_lower@1"] == 0.2
    assert metrics[prefix + "success_upper@1"] == 0.8
    assert metrics[prefix + "coverage"] == 0.4
    assert metrics[prefix + "known_failures"] == 1
    assert metrics[prefix + "unresolved"] == 3
    assert not any("reward/mean" in key for key in metrics)
    assert tasks[-1]["unresolved_reasons"] == {"not_published": 1}
    with pytest.raises(ValueError, match="duplicate"):
        fixed_roster_report(roster, observations + observations[:1], 1)


def test_complete_multiple_samples_are_single_sample_mean_not_pass_at_k():
    roster = [dict(task_id="a", source="bench"), dict(task_id="b", source="bench")]
    observations = [
        dict(task_id=task, session_id=str(session), **valid(passed))
        for task, values in [("a", [1, 0]), ("b", [0, 0])]
        for session, passed in enumerate(values)
    ]
    metrics, _ = fixed_roster_report(roster, observations, 2)
    assert metrics["val-core/bench/reward/mean@2"] == 0.25
    assert metrics["val-core/bench/fixed_roster/complete"] == 1


@pytest.mark.parametrize(
    "patch",
    [
        dict(evaluation_valid=False),
        dict(evaluation_valid=None),
        dict(evaluator_invalid_reason="official_test_coverage_incomplete"),
        dict(observed_infrastructure_failure=1),
        dict(infrastructure_failure=1),
        dict(evaluator_failure_excluded=True),
        dict(unverified_trajectory=True),
        dict(completion_ratio_cutoff=True),
        dict(hidden_pass=None),
    ],
)
def test_strict_validity_rejects_incomplete_evidence(patch):
    assert evaluation_reason({**valid(1), **patch}) is not None
    assert evaluation_reason(valid(0)) is None


def test_partial_reward_parser_coverage_is_not_official_evaluation_coverage():
    assert (
        evaluation_reason(
            {
                **valid(1),
                "partial_hidden_reward_counts_complete": False,
                "partial_hidden_reward_count_reason": "unsupported_format",
            }
        )
        is None
    )


@pytest.mark.parametrize("phase", ["patch_check", "patch_apply"])
def test_model_patch_rejection_is_a_confirmed_failure_without_running_tests(monkeypatch, phase):
    info = dict(
        evaluation_valid=False,
        hidden_pass=0,
        hidden_failure_phase=phase,
        hidden_verifier_staging_ok=True,
        hidden_verifier_setup_ok=True,
    )
    assert evaluation_reason(info) is None
    assert evaluation_reason({**info, "hidden_verifier_setup_ok": False}) is not None
    assert evaluation_reason({**info, "evaluator_invalid_reason": "official_test_coverage_incomplete"}) is not None
    harness = Harness(monkeypatch, 1, retries=10)
    harness.finish("0", **{k: v for k, v in info.items() if k != "hidden_pass"})
    batch, _ = harness.sample(1)
    assert len(batch.keys) == 1 and not harness.refills


def test_official_model_failure_adjudication_survives_legacy_metadata_adapter():
    from verl.trainer.ppo.v1.utils import validation_reward_metadata

    info = validation_reward_metadata(
        {
            "reward_extra_info": dict(evaluation_valid=False, hidden_pass=0),
            "swe_hidden_reward": {"metadata": {"rebench_failure_kind": "model_failure"}},
        }
    )
    assert evaluation_reason(info) is None
    assert evaluation_reason({**info, "observed_infrastructure_failure": 1}) is not None


def test_roster_identity_and_expected_size(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = tmp_path / "official.parquet"
    rows = [dict(extra_info=dict(task_id=tid), metric_data_source="bench") for tid in ["a", "b"]]
    pq.write_table(pa.Table.from_pylist(rows), path)
    monkeypatch.setenv("SWE_AGENT_VALIDATION_EXPECTED_TASKS", "2")
    roster, provenance = load_roster([path])
    assert len(roster) == 2 and len(provenance[0]["sha256"]) == 64
    monkeypatch.setenv("SWE_AGENT_VALIDATION_EXPECTED_TASKS", "3")
    with pytest.raises(ValueError, match="Expected 3"):
        load_roster([path])
    monkeypatch.delenv("SWE_AGENT_VALIDATION_EXPECTED_TASKS")
    pq.write_table(pa.Table.from_pylist(rows + rows[:1]), path)
    with pytest.raises(ValueError, match="unique"):
        load_roster(path)


class Harness:
    def __init__(self, monkeypatch, count, threshold=0.96, retries=2):
        monkeypatch.setenv("SWE_AGENT_VALIDATION_STRICT_ACCOUNTING", "1")
        monkeypatch.setenv("SWE_AGENT_ROLLOUT_VALIDATION_COMPLETION_RATIO_THRESHOLD", str(threshold))
        monkeypatch.setenv("SWE_AGENT_ROLLOUT_VALIDATION_INFRA_FAILURE_GROUP_RATIO_THRESHOLD", ".25")
        monkeypatch.setenv("SWE_AGENT_ROLLOUT_VALIDATION_MAX_GROUP_RETRIES", str(retries))
        monkeypatch.setenv("SWE_AGENT_VALIDATION_GROUP_TIMEOUT_SECONDS", "10")
        monkeypatch.setenv("SWE_AGENT_VALIDATION_TOTAL_TIMEOUT_SECONDS", "100")
        self.clock, self.state, self.cancelled, self.refills = [0.0], {"val": {}}, [], []
        self.on_wait = lambda: None
        self.on_refill = lambda old, new: None
        self.buffer = replay.ReplayBuffer(
            trainer_mode="sync",
            trainer_config={},
            max_off_policy_threshold=1,
            max_off_policy_strategy="drop",
            sampler_kwargs={},
            expected_validation_rollout_n=1,
            validation_refill_fn=self.refill,
            cancel_fn=self.cancel,
        )
        for i in range(count):
            self.state["val"][str(i)] = dict(
                is_prompt=True,
                status="running",
                global_steps=1,
                task_id=str(i),
                validation_root_uid=str(i),
                validation_attempt=0,
                validation_submitted_at=0.0,
            )
        monkeypatch.setattr(replay.tq, "kv_list", lambda *args, **kwargs: copy.deepcopy(self.state))
        monkeypatch.setattr(replay.tq, "kv_clear", self.clear)
        monkeypatch.setattr(replay.time, "monotonic", lambda: self.clock[0])
        monkeypatch.setattr(self.buffer, "_persist_infrastructure_discard", lambda *args: True)
        monkeypatch.setattr(self.buffer, "_wait_for_next_poll", self.wait)

    def finish(self, uid, passed=0, **patch):
        self.state["val"][uid]["status"] = "finished"
        self.state["val"][uid + "_0_0"] = {**valid(passed), **patch}

    def clear(self, *, keys, partition_id):
        for key in keys:
            self.state[partition_id].pop(key, None)

    def cancel(self, uids, validate):
        assert validate
        self.cancelled.extend(uids)
        for uid in uids:
            self.state["val"][uid].update(status="failure", completion_ratio_cutoff=True)
        return len(uids)

    def refill(self, uids):
        result = {}
        for old in uids:
            prior = next(r for r in reversed(self.buffer.validation_attempt_records) if r["uid"] == old)
            new = old + "r"
            self.state["val"][new] = dict(
                is_prompt=True,
                status="running",
                global_steps=1,
                task_id=prior["task_id"],
                validation_root_uid=prior["root_uid"],
                validation_attempt=prior["attempt"] + 1,
                validation_submitted_at=self.clock[0],
            )
            self.refills.append((old, new))
            result[old] = new
            self.on_refill(old, new)
        return result

    def wait(self, *args):
        self.clock[0] += 1
        assert self.clock[0] <= 105, "sampler failed to terminate"
        self.on_wait()
        return 0.0

    def sample(self, count):
        return self.buffer.sample(global_steps=1, partition_id="val", batch_size=count)


@pytest.mark.parametrize("count,confirmed", [(500, 480), (357, 343)])
def test_cutoff_counts_fixed_roster_and_cancels_tail(monkeypatch, count, confirmed):
    harness = Harness(monkeypatch, count)
    for i in range(confirmed):
        harness.finish(str(i), i % 2)
    batch, metrics = harness.sample(count)
    assert len(batch.keys) == confirmed
    assert len(harness.cancelled) == count - confirmed
    assert metrics["validation/fixed_roster/unresolved_groups"] == count - confirmed
    assert len(harness.buffer.validation_attempt_records) == count
    assert len(harness.state["val"]) == confirmed  # selected trajectory records only


def test_reward_pending_does_not_count_toward_cutoff(monkeypatch):
    harness = Harness(monkeypatch, 3, threshold=2 / 3)
    harness.finish("0", 1)
    harness.state["val"]["1"].update(status="reward_pending", generation_complete=True)

    def release():
        assert harness.cancelled == []
        harness.finish("1", 0)

    harness.on_wait = release
    batch, _ = harness.sample(3)
    assert set(batch.keys) == {"0_0_0", "1_0_0"}
    assert harness.cancelled == ["2"]


def test_invalid_official_coverage_retries_but_valid_failure_stops(monkeypatch):
    harness = Harness(monkeypatch, 1, threshold=1, retries=10)
    harness.finish("0", 0, evaluation_valid=False, evaluator_invalid_reason="official_test_coverage_incomplete")
    harness.on_refill = lambda old, new: harness.finish(new, 0)
    batch, metrics = harness.sample(1)
    assert len(harness.refills) == 1
    assert batch.keys == ["0r_0_0"]
    assert metrics["validation/fixed_roster/confirmed_groups"] == 1
    assert [r["reason"] for r in harness.buffer.validation_attempt_records] == [
        "official_test_coverage_incomplete",
        None,
    ]


def test_exhausted_retries_leave_unknown_without_aborting(monkeypatch):
    harness = Harness(monkeypatch, 1, threshold=1, retries=3)
    harness.finish("0", observed_infrastructure_failure=1)
    harness.on_refill = lambda old, new: harness.finish(new, observed_infrastructure_failure=1)
    batch, metrics = harness.sample(1)
    assert batch.keys == [] and len(harness.refills) == 3
    assert len(harness.buffer.validation_attempt_records) == 4
    assert metrics["validation/fixed_roster/unresolved_groups"] == 1
    assert harness.buffer.validation_attempt_records[-1]["disposition"] == "retry_exhausted"


def test_watchdog_terminates_when_more_than_four_percent_hang(monkeypatch):
    harness = Harness(monkeypatch, 100)
    for i in range(90):
        harness.finish(str(i), 1)
    batch, metrics = harness.sample(100)
    assert len(batch.keys) == 90 and len(harness.cancelled) == 10
    assert harness.clock[0] == 10
    assert not harness.refills  # watchdog cancellations are not retry-to-success
    assert metrics["validation/fixed_roster/unresolved_groups"] == 10


def test_global_deadline_cancels_even_reward_pending(monkeypatch):
    harness = Harness(monkeypatch, 1)
    harness.buffer.validation_deadline_at = 2.0
    harness.state["val"]["0"].update(status="reward_pending", generation_complete=True)
    batch, metrics = harness.sample(1)
    assert batch.keys == [] and harness.cancelled == ["0"]
    assert metrics["validation/fixed_roster/deadline_reached"] == 1


def test_retry_activity_cannot_starve_other_groups_watchdog(monkeypatch):
    harness = Harness(monkeypatch, 2, threshold=1, retries=3)
    harness.finish("0", observed_infrastructure_failure=1)

    def retry(old, new):
        harness.clock[0] = 11.0
        harness.finish(new, observed_infrastructure_failure=int(len(harness.refills) < 3))

    harness.on_refill = retry
    batch, _ = harness.sample(2)
    assert len(harness.refills) == 3
    assert len(batch.keys) == 1 and harness.cancelled == ["1"]
    assert harness.buffer.validation_attempt_records[2]["reason"] == "validation_attempt_timeout"


def test_deadline_already_expired_preserves_roster_without_new_retries(monkeypatch):
    harness = Harness(monkeypatch, 1)
    harness.buffer.validation_deadline_at = -1.0
    harness.finish("0", evaluation_valid=False)
    batch, _ = harness.sample(1)
    assert batch.keys == [] and not harness.refills


def test_failed_cancellation_stops_instead_of_contaminating_training(monkeypatch):
    harness = Harness(monkeypatch, 1)
    harness.buffer.validation_deadline_at = -1.0

    def fail(*args, **kwargs):
        raise RuntimeError("cleanup was not acknowledged")

    harness.buffer.cancel_fn = fail
    with pytest.raises(RuntimeError, match="not acknowledged"):
        harness.sample(1)
    assert "0" in harness.state["val"]


class StubTrainer(PPOTrainer):
    def on_step_end(self):
        pass

    def on_sample_end(self):
        pass


def test_real_tensor_batch_retry_preserves_task_and_attempt_lineage(monkeypatch):
    from verl.trainer.ppo.v1 import trainer_base as module

    monkeypatch.setenv("SWE_AGENT_VALIDATION_STRICT_ACCOUNTING", "1")
    monkeypatch.setenv("SWE_AGENT_ROLLOUT_VALIDATION_MAX_GROUP_RETRIES", "10")
    trainer = StubTrainer.__new__(StubTrainer)
    trainer.global_steps = 4
    trainer._validation_roster = [dict(task_id="a", source="bench")]
    trainer._validation_prompt_templates = {}
    trainer._validation_lineage = {}
    publications, dispatches = [], []
    monkeypatch.setattr(module.tq, "kv_batch_put", lambda **kwargs: publications.append(kwargs))
    trainer.agent_loop_manager = SimpleNamespace(generate_sequences=lambda batch: dispatches.append(batch))
    batch = tu.get_tensordict(
        dict(
            uid=np.array(["old"], dtype=object),
            reward_model=np.array([dict(ground_truth=dict(instance_id="a"))], dtype=object),
            extra_info=np.array([dict(task_id="a")], dtype=object),
            data_source=np.array(["bench"], dtype=object),
        )
    )
    trainer._submit_validation_batch(batch)
    replacements = trainer._refill_validation_groups(["old"])
    new = replacements["old"]
    assert trainer._validation_lineage[new] == dict(task_id="a", source="bench", root_uid="old", attempt=1)
    assert publications[-1]["tags"][0]["validation_root_uid"] == "old"
    assert publications[-1]["tags"][0]["validation_attempt"] == 1
    assert len(dispatches) == 2


def test_validation_cleanup_has_a_real_manager_deadline(monkeypatch):
    from verl.trainer.ppo.v1 import agent_loop_tq as module

    monkeypatch.setenv("SWE_AGENT_VALIDATION_STRICT_ACCOUNTING", "1")
    monkeypatch.setenv("SWE_AGENT_VALIDATION_CLEANUP_TIMEOUT_SECONDS", "2")
    manager = object.__new__(module.AgentLoopManagerTQ)
    manager.agent_loop_workers = [SimpleNamespace(cancel_sequences=SimpleNamespace(remote=lambda *a, **kw: "worker"))]
    manager.repeat_coordinator = None
    tick = [0.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: tick[0])

    def wait(refs, **kwargs):
        tick[0] += kwargs["timeout"]
        return [], refs

    monkeypatch.setattr(module.ray, "wait", wait)
    monkeypatch.setattr(module.ray, "get", lambda refs: [])
    with pytest.raises(RuntimeError, match="refusing to resume training"):
        manager.cancel_sequences(["a"], validate=True)
    assert tick[0] == 2


def test_strict_protocol_is_forwarded_to_ray_and_resets_when_disabled(monkeypatch):
    from verl.trainer.constants_ppo import get_ppo_ray_runtime_env

    monkeypatch.setenv("SWE_AGENT_VALIDATION_STRICT_ACCOUNTING", "1")
    monkeypatch.setenv("SWE_AGENT_VALIDATION_TOTAL_TIMEOUT_SECONDS", "21600")
    monkeypatch.setenv("SWE_AGENT_ROLLOUT_VALIDATION_MAX_GROUP_RETRIES", "10")
    values = get_ppo_ray_runtime_env()["env_vars"]
    assert values["SWE_AGENT_VALIDATION_STRICT_ACCOUNTING"] == "1"
    assert values["SWE_AGENT_VALIDATION_TOTAL_TIMEOUT_SECONDS"] == "21600"
    assert values["SWE_AGENT_ROLLOUT_VALIDATION_MAX_GROUP_RETRIES"] == "10"
    monkeypatch.delenv("SWE_AGENT_VALIDATION_STRICT_ACCOUNTING")
    assert get_ppo_ray_runtime_env()["env_vars"]["SWE_AGENT_VALIDATION_STRICT_ACCOUNTING"] == "0"


def test_validation_dispatch_is_bounded_when_worker_is_unresponsive(monkeypatch):
    from verl.trainer.ppo.v1 import agent_loop_tq as module

    monkeypatch.setenv("SWE_AGENT_VALIDATION_STRICT_ACCOUNTING", "1")
    monkeypatch.setenv("SWE_AGENT_VALIDATION_CLEANUP_TIMEOUT_SECONDS", "2")
    manager = object.__new__(module.AgentLoopManagerTQ)
    worker = SimpleNamespace(generate_sequences=SimpleNamespace(remote=lambda batch: "dispatch"))
    manager.agent_loop_workers = [worker]
    batch = tu.get_tensordict(dict(uid=np.array(["a"], dtype=object)))
    tu.assign_non_tensor_data(batch, "validate", True)

    def unresponsive(refs, timeout):
        assert refs == ["dispatch"] and timeout == 2
        raise module.ray.exceptions.GetTimeoutError("unresponsive worker")

    monkeypatch.setattr(module.ray, "get", unresponsive)
    with pytest.raises(RuntimeError, match="dispatch was not acknowledged"):
        manager.generate_sequences(batch)


def validation_trainer(monkeypatch, tmp_path, confirmed, successes):
    """Drive the actual trainer and sampler with in-memory queue storage."""
    import torch
    from omegaconf import OmegaConf

    from verl.trainer.ppo.v1 import trainer_base as module

    harness = Harness(monkeypatch, 0, threshold=0.5, retries=0)
    trainer = StubTrainer.__new__(StubTrainer)
    trainer.global_steps = 12
    trainer.config = OmegaConf.create(
        {
            "trainer": {"validation_data_dir": str(tmp_path), "default_local_dir": str(tmp_path / "checkpoints")},
            "actor_rollout_ref": {"rollout": {"val_kwargs": {"n": 1, "temperature": 1.0}}},
        }
    )
    trainer._validation_roster = [dict(task_id=str(i), source="bench") for i in range(4)]
    trainer._validation_roster_provenance = [dict(path="official.parquet", sha256="fixed", rows=4)]
    trainer._validation_prompt_templates = {}
    trainer.replay_buffer = harness.buffer
    harness.buffer.validation_refill_fn = trainer._refill_validation_groups
    trainer.reward_loop_manager = SimpleNamespace(reward_loop_worker_handles=[object()])
    trainer.timeout_prediction_enabled = False
    trainer.tokenizer = SimpleNamespace(pad_token_id=0, decode=lambda *a, **kw: "decoded")
    trainer._maybe_log_val_generations = lambda **kw: None
    trainer._dump_generations = lambda **kw: None
    trainer._run_optional_audio_validation = lambda: {}
    trainer.val_dataloader = [
        dict(
            raw_prompt=np.array(["prompt"] * 2, dtype=object),
            extra_info=np.array([dict(task_id=str(i)) for i in ids], dtype=object),
            reward_model=np.array([dict(ground_truth=dict(instance_id=str(i))) for i in ids], dtype=object),
            data_source=np.array(["bench"] * 2, dtype=object),
        )
        for ids in [(0, 1), (2, 3)]
    ]
    fields, dispatches = {}, []

    def put(*, keys, partition_id, tags, **kwargs):
        for key, tag in zip(keys, tags, strict=True):
            harness.state[partition_id].setdefault(key, {}).update(tag)

    def dispatch(batch):
        dispatches.append(batch)
        for uid in batch["uid"]:
            uid = str(uid)
            tag = harness.state["val"][uid]
            tid = tag["task_id"]
            if tid not in confirmed:
                continue
            passed = int(tid in successes)
            harness.finish(uid, passed)
            fields[uid + "_0_0"] = dict(
                uid=uid,
                num_turns=3,
                rm_scores=[passed],
                data_source="bench",
                reward_model=dict(ground_truth=dict(instance_id=tid)),
                extra_fields=dict(reward_extra_info={**valid(passed), "score": passed}),
            )

    def get(*, keys, select_fields, **kwargs):
        if select_fields == ["prompts", "responses"]:
            return {name: torch.nested.nested_tensor([torch.tensor([1, 2])] * len(keys)) for name in select_fields}
        result = {}
        for name in select_fields:
            values = [fields[key][name] for key in keys]
            result[name] = torch.tensor(values) if name == "rm_scores" else np.array(values, dtype=object)
        return result

    trainer.agent_loop_manager = SimpleNamespace(generate_sequences=dispatch)
    monkeypatch.setattr(module.tq, "kv_batch_put", put)
    monkeypatch.setattr(module.tq, "kv_batch_get", get)
    return trainer, harness, dispatches


@pytest.mark.parametrize(
    "confirmed,successes", [({"0", "2"}, {"0"}), (set(), set()), ({"0", "1", "2", "3"}, {"0", "2"})]
)
def test_full_validation_accounts_for_multiple_batches_and_empty_results(monkeypatch, tmp_path, confirmed, successes):
    trainer, harness, _ = validation_trainer(monkeypatch, tmp_path, confirmed, successes)
    metrics = trainer._validate()
    prefix = "val-core/bench/fixed_roster/"
    assert metrics[prefix + "expected_tasks"] == 4
    assert metrics[prefix + "coverage"] == len(confirmed) / 4
    assert metrics[prefix + "success_lower@1"] == len(successes) / 4
    assert metrics[prefix + "success_upper@1"] == (len(successes) + 4 - len(confirmed)) / 4
    assert ("val-core/bench/reward/mean@1" in metrics) == (len(confirmed) == 4)
    audit = json.loads((tmp_path / "validation_audit/12.json").read_text())
    assert len(audit["tasks"]) == 4 and len(audit["attempts"]) == 4
    assert audit["status"] == ("complete" if len(confirmed) == 4 else "incomplete")
    assert not harness.state["val"]


def test_validation_deadline_skips_later_batches_without_changing_denominator(monkeypatch, tmp_path):
    trainer, harness, dispatches = validation_trainer(monkeypatch, tmp_path, {"0", "2"}, {"0", "2"})
    batches = trainer.val_dataloader

    def batches_with_delay():
        yield batches[0]
        harness.clock[0] = 101.0
        yield batches[1]

    trainer.val_dataloader = batches_with_delay()
    metrics = trainer._validate()
    assert len(dispatches) == 1
    assert metrics["val-core/bench/fixed_roster/expected_tasks"] == 4
    assert metrics["val-core/bench/fixed_roster/unresolved"] == 3
    assert metrics["validation/fixed_roster/deadline_reached"] == 1


def test_validation_cleanup_failure_leaves_durable_error_audit(monkeypatch, tmp_path):
    trainer, harness, _ = validation_trainer(monkeypatch, tmp_path, {"0"}, {"0"})

    def fail(*args, **kwargs):
        raise RuntimeError("cleanup was not acknowledged")

    harness.buffer.cancel_fn = fail
    with pytest.raises(RuntimeError, match="not acknowledged"):
        trainer._validate()
    audit = json.loads((tmp_path / "validation_audit/12.json").read_text())
    assert audit["status"] == "error" and "not acknowledged" in audit["error"]
    assert any(a["status"] == "cancellation_requested" for a in audit["attempts"])


def test_strict_sampler_uses_one_snapshot_per_poll(monkeypatch):
    harness = Harness(monkeypatch, 1)
    calls = []

    def snapshot(*args, **kwargs):
        calls.append(None)
        # The prompt becomes terminal after the first snapshot was captured.
        previous = copy.deepcopy(harness.state)
        harness.finish("0", 1)
        return previous

    monkeypatch.setattr(replay.tq, "kv_list", snapshot)
    batch, _ = harness.sample(1)
    assert batch.keys == ["0_0_0"] and not harness.refills
    assert len(calls) == 3  # one running poll, one terminal poll, final materialization

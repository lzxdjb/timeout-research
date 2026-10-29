# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""CPU tests for ReplayBuffer against a real TransferQueue instance.

Producers write every trajectory before marking its prompt terminal. Consumers run in a separate
thread when a test needs to observe blocking behavior. Off-policy staleness is measured in model
versions as ``global_steps - prompt_global_steps + 1``; ``drop`` and ``wait`` apply only to async
trainers.
"""

import threading
import time
import uuid
from dataclasses import dataclass, field

import pytest
import torch
import transfer_queue as tq
from transfer_queue import KVBatchMeta

from verl.trainer.ppo.v1.replay_buffer import (
    ReplayBuffer,
    ReplayBufferAsync,
    golden_group_is_eligible,
)


def test_golden_group_admission_excludes_zero_variance_and_rollout_discards():
    common = {
        "infrastructure_failure_count": 0,
        "expected_rollout_count": 8,
        "infrastructure_failure_threshold": 0.25,
    }
    assert golden_group_is_eligible([0.0, 1.0, 0.0, 1.0], completion_ratio_cutoff=False, **common)
    assert not golden_group_is_eligible([1.0, 1.0, 1.0, 1.0], completion_ratio_cutoff=False, **common)
    assert not golden_group_is_eligible([0.0, 1.0, 0.0, 1.0], completion_ratio_cutoff=True, **common)
    assert not golden_group_is_eligible(
        [0.0, 1.0, 0.0, 1.0],
        completion_ratio_cutoff=False,
        infrastructure_failure_count=3,
        **{key: value for key, value in common.items() if key != "infrastructure_failure_count"},
    )


@pytest.mark.parametrize("dapo_enabled", [False, True])
def test_strict_validation_retry_and_cutoff_against_real_queue(tq_init, monkeypatch, tmp_path, dapo_enabled):
    monkeypatch.setenv("SWE_AGENT_VALIDATION_STRICT_ACCOUNTING", "1")
    monkeypatch.setenv("SWE_AGENT_ROLLOUT_VALIDATION_COMPLETION_RATIO_THRESHOLD", str(2 / 3))
    monkeypatch.setenv("SWE_AGENT_ROLLOUT_VALIDATION_INFRA_FAILURE_GROUP_RATIO_THRESHOLD", ".25")
    monkeypatch.setenv("SWE_AGENT_ROLLOUT_VALIDATION_MAX_GROUP_RETRIES", "10")
    _clear_partition("val")
    roots = [_uid() for _ in range(3)]
    retried, cancelled = [], []

    def publish(uid, task, *, root=None, attempt=0, passed=0, valid=True):
        tq.kv_put(
            key=f"{uid}_0_0",
            partition_id="val",
            fields={"input_ids": torch.tensor([1]), "extra_fields": {"reward_extra_info": {}}},
            tag={"evaluation_valid": valid, "hidden_pass": passed, "infrastructure_failure": not valid},
        )
        tq.kv_put(
            key=uid,
            partition_id="val",
            tag={
                "is_prompt": True,
                "status": "finished",
                "global_steps": 0,
                "task_id": task,
                "validation_root_uid": root or uid,
                "validation_attempt": attempt,
                "validation_submitted_at": time.monotonic(),
            },
        )

    def refill(uids):
        assert uids == [roots[1]]
        replacement = _uid()
        retried.append(replacement)
        publish(replacement, "b", root=roots[1], attempt=1, passed=0)
        return {roots[1]: replacement}

    def cancel(uids, validate):
        assert validate
        cancelled.extend(uids)
        for uid in uids:
            tq.kv_put(key=uid, partition_id="val", tag={"status": "failure", "completion_ratio_cutoff": True})
        return len(uids)

    try:
        publish(roots[0], "a", passed=1)
        publish(roots[1], "b", valid=False)
        tq.kv_put(
            key=roots[2],
            partition_id="val",
            tag={
                "is_prompt": True,
                "status": "reward_pending",
                "global_steps": 0,
                "task_id": "c",
                "validation_root_uid": roots[2],
                "validation_attempt": 0,
                "validation_submitted_at": time.monotonic(),
            },
        )
        rb = _make_rb(
            expected_validation_rollout_n=1,
            validation_refill_fn=refill,
            cancel_fn=cancel,
            filter_groups_metric="score" if dapo_enabled else None,
            refill_fn=(lambda _n: pytest.fail("Validation must not refill training prompts")) if dapo_enabled else None,
        )
        rb.trainer_config = {"validation_data_dir": str(tmp_path)}
        batch, metrics = rb.sample(global_steps=0, partition_id="val", batch_size=3)
        assert set(batch.keys) == {f"{roots[0]}_0_0", f"{retried[0]}_0_0"}
        assert cancelled == [roots[2]]
        assert metrics["validation/fixed_roster/recovered_groups"] == 1
        assert metrics["validation/fixed_roster/unresolved_groups"] == 1
        assert len(rb.validation_attempt_records) == 4
        assert len(list((tmp_path / "infrastructure_discard").glob("*.json"))) == 1
    finally:
        _clear_partition("val")


# Small poll interval so the blocking consumer reacts to producer writes quickly.
POLL_INTERVAL = 0.05


@pytest.fixture(scope="module")
def tq_init():
    tq.init()
    yield
    tq.close()


@pytest.fixture
def partition_id():
    """A unique partition per test to isolate TransferQueue state across tests."""
    return f"test-{uuid.uuid4().hex}"


def _make_rb(
    *,
    max_off_policy_threshold: int = 8,
    max_off_policy_strategy: str = "drop",
    poll_interval: float = POLL_INTERVAL,
    refill_fn=None,
    trainer_mode: str = "sync",
    filter_groups_metric: str | None = None,
    train_batch_size: int = 2,
    gen_batch_size: int = 1,
    max_inflight_gen_batches: int = 1,
    sync_refill_failed_groups: bool = False,
    cancel_fn=None,
    expected_rollout_n: int | None = None,
    expected_validation_rollout_n: int | None = None,
    validation_refill_fn=None,
) -> ReplayBuffer:
    """Construct a ReplayBuffer with defaults that keep generic samples on-policy."""
    replay_buffer_cls = ReplayBuffer if trainer_mode == "sync" else ReplayBufferAsync
    return replay_buffer_cls(
        trainer_mode=trainer_mode,
        trainer_config={},
        max_off_policy_threshold=max_off_policy_threshold,
        max_off_policy_strategy=max_off_policy_strategy,
        sampler_kwargs={},
        poll_interval=poll_interval,
        refill_fn=refill_fn,
        filter_groups_metric=filter_groups_metric,
        train_batch_size=train_batch_size,
        gen_batch_size=gen_batch_size,
        max_inflight_gen_batches=max_inflight_gen_batches,
        sync_refill_failed_groups=sync_refill_failed_groups,
        cancel_fn=cancel_fn,
        expected_rollout_n=expected_rollout_n,
        expected_validation_rollout_n=expected_validation_rollout_n,
        validation_refill_fn=validation_refill_fn,
    )


def _configure_infrastructure_partition(rb: ReplayBuffer, partition_id: str, expected_count: int) -> None:
    """Apply the train policy to a synthetic TransferQueue partition."""
    rb.infrastructure_failure_ratio_thresholds[partition_id] = 0.25
    rb.expected_rollout_counts[partition_id] = expected_count


class FakeRefiller:
    """Produce fresh terminal prompts when the replay buffer requests replacements."""

    def __init__(self, partition_id: str, global_steps: int, sessions: int = 1, rewards: list[float] | None = None):
        self.partition_id = partition_id
        self.global_steps = global_steps
        self.sessions = sessions
        # When set, refilled groups carry these per-session rewards so they survive zero-variance filtering.
        self.rewards = rewards
        self.calls: list[int] = []
        self.produced_uids: list[str] = []

    def __call__(self, num_prompts: int) -> int:
        self.calls.append(num_prompts)
        specs = [
            PromptSpec(
                uid=_uid(),
                status="finished",
                sessions=self.sessions,
                global_steps=self.global_steps,
                rewards=self.rewards,
            )
            for _ in range(num_prompts)
        ]
        # Synchronous produce (already-terminal) so the sample loop sees them on its next poll.
        _produce(self.partition_id, specs).join_and_check()
        self.produced_uids.extend(spec.uid for spec in specs)
        return num_prompts


def _sample(rb: ReplayBuffer, partition_id: str, batch_size: int, global_steps: int = 0) -> KVBatchMeta:
    """Call ``sample`` and return just the batch (dropping the metrics dict)."""
    batch, _metrics = rb.sample(global_steps=global_steps, partition_id=partition_id, batch_size=batch_size)
    return batch


def _uid() -> str:
    # uid must not contain "_" because ReplayBuffer derives it via key.split("_")[0].
    return uuid.uuid4().hex


def _trajectory_key(uid: str, session_id: int = 0, index: int = 0) -> str:
    return f"{uid}_{session_id}_{index}"


def _set_prompt_status(partition_id: str, uid: str, status: str, global_steps: int) -> None:
    """Transition an existing prompt to a new status (e.g. running -> finished).

    Mirrors the rollout side flipping a GRPO group's status once it terminates.
    The prompt tag is updated in place; its trajectory values are untouched.
    """
    tq.kv_put(
        key=uid,
        partition_id=partition_id,
        tag={"is_prompt": True, "status": status, "global_steps": global_steps},
    )


@dataclass
class PromptSpec:
    """A prompt group and the trajectories that precede its status update."""

    uid: str
    status: str
    sessions: int = 1
    global_steps: int = 0
    rewards: list[float] | None = None
    infrastructure_failures: list[int] | None = None
    unverified: list[bool] | None = None
    trajectory_keys: list[str] = field(default_factory=list)


class RolloutProducer(threading.Thread):
    """Write complete trajectory groups before publishing their prompt status."""

    def __init__(self, partition_id: str, specs: list[PromptSpec]):
        super().__init__(daemon=True)
        self.partition_id = partition_id
        self.specs = specs
        self.error: Exception | None = None

    def run(self) -> None:
        try:
            for spec in self.specs:
                for session_id in range(spec.sessions):
                    key = _trajectory_key(spec.uid, session_id)
                    fields = {"input_ids": torch.tensor([1, 2, 3])}
                    tag = {"is_prompt": False, "seq_len": 3, "global_steps": spec.global_steps}
                    if spec.unverified is not None:
                        tag["unverified_trajectory"] = spec.unverified[session_id]
                    if spec.rewards is not None:
                        fields["extra_fields"] = {"reward_extra_info": {"acc": float(spec.rewards[session_id])}}
                    if spec.infrastructure_failures is not None:
                        fields.setdefault("extra_fields", {}).setdefault("reward_extra_info", {}).update(
                            {
                                "observed_infrastructure_failure": int(
                                    spec.infrastructure_failures[session_id]
                                ),
                                "observed_infrastructure_failure_code": 3,
                            }
                        )
                        tag["infrastructure_failure"] = bool(spec.infrastructure_failures[session_id])
                    tq.kv_put(
                        key=key,
                        partition_id=self.partition_id,
                        fields=fields,
                        tag=tag,
                    )
                    spec.trajectory_keys.append(key)
                tq.kv_put(
                    key=spec.uid,
                    partition_id=self.partition_id,
                    tag={"is_prompt": True, "status": spec.status, "global_steps": spec.global_steps},
                )
        except Exception as e:  # surfaced to the test via join_and_check()
            self.error = e

    def join_and_check(self, timeout: float = 10.0) -> None:
        self.join(timeout)
        assert not self.is_alive(), "RolloutProducer thread did not finish in time"
        if self.error is not None:
            raise self.error


class SampleConsumer(threading.Thread):
    """Runs the blocking ``ReplayBuffer.sample`` in a background thread so the test
    can assert that it stays blocked until the producer supplies enough data."""

    def __init__(self, rb: ReplayBuffer, partition_id: str, batch_size: int, global_steps: int = 0):
        super().__init__(daemon=True)
        self.rb = rb
        self.partition_id = partition_id
        self.batch_size = batch_size
        self.global_steps = global_steps
        self.result: KVBatchMeta | None = None
        self.metrics: dict | None = None
        self.error: Exception | None = None

    def run(self) -> None:
        try:
            self.result, self.metrics = self.rb.sample(
                global_steps=self.global_steps,
                partition_id=self.partition_id,
                batch_size=self.batch_size,
            )
        except Exception as e:
            self.error = e

    def result_or_raise(self, timeout: float = 10.0) -> KVBatchMeta:
        self.join(timeout)
        assert not self.is_alive(), "SampleConsumer thread did not finish in time"
        if self.error is not None:
            raise self.error
        assert self.result is not None
        return self.result


def _produce(partition_id: str, specs: list[PromptSpec]) -> RolloutProducer:
    producer = RolloutProducer(partition_id, specs)
    producer.start()
    return producer


def _clear_partition(partition_id: str) -> None:
    """Best-effort cleanup of every key written into a partition."""
    keys = list(tq.kv_list(partition_id=partition_id).get(partition_id, {}).keys())
    if keys:
        tq.kv_clear(keys=keys, partition_id=partition_id)


def _uids_of(keys: list[str]) -> set[str]:
    return {key.split("_")[0] for key in keys}


# --------------------------------------------------------------------------- #
# __init__: configuration validation.
# --------------------------------------------------------------------------- #


def test_init_rejects_non_positive_threshold():
    """max_off_policy_threshold must be a positive integer."""
    with pytest.raises(AssertionError, match="max off policy threshold"):
        _make_rb(max_off_policy_threshold=0)


def test_init_rejects_unknown_strategy():
    """max_off_policy_strategy must be one of {drop, wait}."""
    with pytest.raises(AssertionError, match="max off policy strategy"):
        _make_rb(max_off_policy_strategy="bogus")


@pytest.mark.parametrize("value", ["0", "1.1", "nan", "not-a-number"])
def test_completion_ratio_rejects_invalid_values(monkeypatch, value: str) -> None:
    monkeypatch.setenv("SWE_AGENT_ROLLOUT_VALIDATION_COMPLETION_RATIO_THRESHOLD", value)

    with pytest.raises(ValueError, match="SWE_AGENT_ROLLOUT_VALIDATION_COMPLETION_RATIO_THRESHOLD"):
        _make_rb()


def test_sync_completion_ratio_cancels_tail_and_returns_complete_groups(
    tq_init, partition_id: str, monkeypatch
) -> None:
    monkeypatch.setenv("SWE_AGENT_ROLLOUT_TRAINING_COMPLETION_RATIO_THRESHOLD", "0.9")
    finished = [PromptSpec(uid=_uid(), status="finished", sessions=2) for _ in range(9)]
    running_uid = _uid()
    _produce(partition_id, finished).join_and_check()
    tq.kv_put(
        key=running_uid,
        partition_id=partition_id,
        tag={"is_prompt": True, "status": "running", "global_steps": 0},
    )
    cancelled: list[tuple[list[str], bool]] = []

    def cancel_fn(uids: list[str], validate: bool = False) -> int:
        cancelled.append((uids, validate))
        for uid in uids:
            tq.kv_put(
                key=uid,
                partition_id=partition_id,
                tag={
                    "is_prompt": True,
                    "status": "failure",
                    "global_steps": 0,
                    "completion_ratio_cutoff": True,
                },
            )
        return len(uids)

    rb = _make_rb(train_batch_size=10, cancel_fn=cancel_fn)
    rb.completion_ratio_thresholds[partition_id] = 0.9
    try:
        batch, metrics = rb.sample(global_steps=0, partition_id=partition_id, batch_size=10)

        assert cancelled == [([running_uid], False)]
        assert len(batch.keys) == 18
        assert _uids_of(batch.keys) == {spec.uid for spec in finished}
        assert metrics["validation/completion_ratio/terminal_groups"] == 9.0
        assert metrics["validation/completion_ratio/cutoff_groups"] == 1.0
    finally:
        _clear_partition(partition_id)


def test_golden_replay_groups_count_toward_training_completion_ratio(tq_init, monkeypatch) -> None:
    partition_id = "train"
    _clear_partition(partition_id)
    monkeypatch.setenv("SWE_AGENT_ROLLOUT_TRAINING_COMPLETION_RATIO_THRESHOLD", "0.5")
    golden = [PromptSpec(uid=_uid(), status="finished", sessions=2) for _ in range(4)]
    fresh = PromptSpec(uid=_uid(), status="finished", sessions=2)
    _produce(partition_id, [*golden, fresh]).join_and_check()
    for spec in golden:
        tq.kv_put(
            key=spec.uid,
            partition_id=partition_id,
            tag={
                "is_prompt": True,
                "status": "finished",
                "global_steps": 0,
                "golden_replay_source": True,
            },
        )
    running_uids = [_uid() for _ in range(5)]
    for uid in running_uids:
        tq.kv_put(
            key=uid,
            partition_id=partition_id,
            tag={"is_prompt": True, "status": "running", "global_steps": 0},
        )
    cancelled: list[list[str]] = []

    def cancel_fn(uids: list[str], validate: bool = False) -> int:
        assert not validate
        cancelled.append(uids)
        for uid in uids:
            tq.kv_put(
                key=uid,
                partition_id=partition_id,
                tag={
                    "is_prompt": True,
                    "status": "failure",
                    "global_steps": 0,
                    "completion_ratio_cutoff": True,
                },
            )
        return len(uids)

    rb = _make_rb(train_batch_size=10, cancel_fn=cancel_fn)
    rb.golden_replay_enabled = True
    try:
        batch, metrics = rb.sample(global_steps=0, partition_id=partition_id, batch_size=10)

        assert cancelled == [sorted(running_uids)]
        assert _uids_of(batch.keys) == {spec.uid for spec in golden} | {fresh.uid}
        assert metrics["training/completion_ratio/required_groups"] == 5.0
        assert metrics["training/completion_ratio/replayed_groups"] == 4.0
        assert metrics["training/completion_ratio/fresh_terminal_groups"] == 1.0
        assert metrics["training/completion_ratio/terminal_groups"] == 5.0
        assert metrics["training/completion_ratio/total_groups"] == 10.0
    finally:
        _clear_partition(partition_id)


def test_golden_replay_completion_ratio_requires_one_fresh_group(tq_init, monkeypatch) -> None:
    partition_id = "train"
    _clear_partition(partition_id)
    monkeypatch.setenv("SWE_AGENT_ROLLOUT_TRAINING_COMPLETION_RATIO_THRESHOLD", "0.5")
    golden = [PromptSpec(uid=_uid(), status="finished", sessions=2) for _ in range(5)]
    _produce(partition_id, golden).join_and_check()
    for spec in golden:
        tq.kv_put(
            key=spec.uid,
            partition_id=partition_id,
            tag={
                "is_prompt": True,
                "status": "finished",
                "global_steps": 0,
                "golden_replay_source": True,
            },
        )
    running_uids = [_uid() for _ in range(5)]
    for uid in running_uids:
        tq.kv_put(
            key=uid,
            partition_id=partition_id,
            tag={"is_prompt": True, "status": "running", "global_steps": 0},
        )
    cancelled: list[list[str]] = []

    def cancel_fn(uids: list[str], validate: bool = False) -> int:
        cancelled.append(uids)
        return len(uids)

    rb = _make_rb(train_batch_size=10, cancel_fn=cancel_fn)
    rb.golden_replay_enabled = True
    try:
        rb._sync_metadata_from_transfer_queue()
        cutoff_requested, metrics = rb._maybe_apply_completion_ratio_cutoff(
            partition_id, batch_size=10, cutoff_requested=False
        )
        assert not cutoff_requested
        assert not metrics
        assert not cancelled

        fresh_uid = running_uids.pop()
        tq.kv_put(
            key=_trajectory_key(fresh_uid),
            partition_id=partition_id,
            fields={"input_ids": torch.tensor([1, 2, 3])},
            tag={"is_prompt": False, "seq_len": 3, "global_steps": 0},
        )
        _set_prompt_status(partition_id, fresh_uid, "finished", global_steps=0)
        rb._sync_metadata_from_transfer_queue()
        cutoff_requested, metrics = rb._maybe_apply_completion_ratio_cutoff(
            partition_id, batch_size=10, cutoff_requested=False
        )

        assert cutoff_requested
        assert cancelled == [sorted(running_uids)]
        assert metrics["training/completion_ratio/replayed_groups"] == 5.0
        assert metrics["training/completion_ratio/fresh_terminal_groups"] == 1.0
    finally:
        _clear_partition(partition_id)


def test_completion_ratio_keeps_cutoff_groups_with_materialized_sessions() -> None:
    rb = _make_rb()
    complete_uid = _uid()
    no_output_uid = _uid()
    rb.failure_keys["train"].update({complete_uid, no_output_uid})
    rb.completion_ratio_cutoff_keys["train"].update({complete_uid, no_output_uid})
    rb.partitions["train"] = {
        _trajectory_key(complete_uid, 0): {"completion_ratio_cutoff": True},
        _trajectory_key(complete_uid, 1): {"completion_ratio_cutoff": True},
    }

    eviction_reasons = rb._terminal_eviction_reasons(global_steps=0, partition_id="train")

    assert complete_uid not in eviction_reasons[2]
    assert no_output_uid in eviction_reasons[2]
    assert complete_uid in rb._sampleable_terminal_keys("train", eviction_reasons)


@pytest.mark.parametrize("value", ["-0.1", "1.1", "nan", "not-a-number"])
def test_infrastructure_group_filter_rejects_invalid_values(monkeypatch, value: str) -> None:
    monkeypatch.setenv("SWE_AGENT_ROLLOUT_TRAINING_COMPLETION_RATIO_THRESHOLD", "0.9")
    monkeypatch.setenv("SWE_AGENT_ROLLOUT_TRAINING_INFRA_FAILURE_GROUP_RATIO_THRESHOLD", value)

    with pytest.raises(ValueError, match="SWE_AGENT_ROLLOUT_TRAINING_INFRA_FAILURE_GROUP_RATIO_THRESHOLD"):
        _make_rb(refill_fn=lambda _n: None, expected_rollout_n=8)


def test_infrastructure_group_filter_requires_completion_ratio(monkeypatch) -> None:
    monkeypatch.setenv("SWE_AGENT_ROLLOUT_TRAINING_INFRA_FAILURE_GROUP_RATIO_THRESHOLD", "0.25")

    with pytest.raises(ValueError, match="SWE_AGENT_ROLLOUT_TRAINING_COMPLETION_RATIO_THRESHOLD"):
        _make_rb(refill_fn=lambda _n: None, expected_rollout_n=8)


def test_dapo_training_keeps_failure_filters_without_completion_cutoff(monkeypatch):
    monkeypatch.delenv("SWE_AGENT_ROLLOUT_TRAINING_COMPLETION_RATIO_THRESHOLD", raising=False)
    monkeypatch.setenv("SWE_AGENT_ROLLOUT_TRAINING_INFRA_FAILURE_GROUP_RATIO_THRESHOLD", "0.25")
    monkeypatch.setenv("SWE_AGENT_ROLLOUT_TRAINING_UNVERIFIED_TRAJECTORY_RATIO_THRESHOLD", "0.25")
    monkeypatch.setenv("SWE_AGENT_ROLLOUT_VALIDATION_COMPLETION_RATIO_THRESHOLD", "0.98")
    monkeypatch.setenv("SWE_AGENT_ROLLOUT_VALIDATION_INFRA_FAILURE_GROUP_RATIO_THRESHOLD", "0.25")
    rb = _make_rb(
        filter_groups_metric="score",
        refill_fn=lambda _n: None,
        expected_rollout_n=8,
        expected_validation_rollout_n=1,
        validation_refill_fn=lambda _uids: None,
    )
    assert rb.completion_ratio_thresholds == {"train": None, "val": 0.98}
    assert rb.infrastructure_failure_ratio_thresholds["train"] == 0.25
    assert rb.unverified_ratio_thresholds["train"] == 0.25


@pytest.mark.parametrize("completion_ratio", ["0.85", "1.0"])
def test_dapo_still_rejects_training_completion_cutoff(monkeypatch, completion_ratio):
    monkeypatch.setenv("SWE_AGENT_ROLLOUT_TRAINING_COMPLETION_RATIO_THRESHOLD", completion_ratio)
    with pytest.raises(ValueError, match="Training completion-ratio"):
        _make_rb(filter_groups_metric="score", refill_fn=lambda _n: None)


def test_dapo_preserves_validation_infrastructure_cutoff_requirement(monkeypatch):
    monkeypatch.delenv("SWE_AGENT_ROLLOUT_VALIDATION_COMPLETION_RATIO_THRESHOLD", raising=False)
    monkeypatch.setenv("SWE_AGENT_ROLLOUT_VALIDATION_INFRA_FAILURE_GROUP_RATIO_THRESHOLD", "0.25")
    with pytest.raises(ValueError, match="SWE_AGENT_ROLLOUT_VALIDATION_COMPLETION_RATIO_THRESHOLD"):
        _make_rb(
            filter_groups_metric="score",
            refill_fn=lambda _n: None,
            expected_validation_rollout_n=1,
            validation_refill_fn=lambda _uids: None,
        )


def test_validation_infrastructure_filter_does_not_require_training_refill(monkeypatch) -> None:
    monkeypatch.setenv("SWE_AGENT_ROLLOUT_VALIDATION_COMPLETION_RATIO_THRESHOLD", "0.9")
    monkeypatch.setenv("SWE_AGENT_ROLLOUT_VALIDATION_INFRA_FAILURE_GROUP_RATIO_THRESHOLD", "0.25")

    rb = _make_rb(
        refill_fn=None,
        expected_validation_rollout_n=8,
        validation_refill_fn=lambda _uids: None,
    )
    assert rb.infrastructure_failure_ratio_thresholds["val"] == 0.25


def test_infrastructure_group_filter_uses_expected_sessions_and_strict_threshold(
    tq_init, partition_id: str, monkeypatch
) -> None:
    monkeypatch.setenv("SWE_AGENT_ROLLOUT_TRAINING_COMPLETION_RATIO_THRESHOLD", "0.9")
    monkeypatch.setenv("SWE_AGENT_ROLLOUT_TRAINING_INFRA_FAILURE_GROUP_RATIO_THRESHOLD", "0.25")
    discarded = PromptSpec(
        uid=_uid(), status="finished", sessions=3, infrastructure_failures=[1, 1, 1]
    )
    boundary = PromptSpec(
        uid=_uid(), status="finished", sessions=2, infrastructure_failures=[1, 1]
    )
    _produce(partition_id, [discarded, boundary]).join_and_check()
    rb = _make_rb(
        refill_fn=lambda _n: None,
        expected_rollout_n=8,
        expected_validation_rollout_n=8,
    )
    _configure_infrastructure_partition(rb, partition_id, 8)
    try:
        rb._sync_metadata_from_transfer_queue()
        discarded_uids, ratios = rb._infrastructure_filtered_uids(partition_id)

        assert ratios[discarded.uid] == 3 / 8
        assert ratios[boundary.uid] == 2 / 8
        assert discarded_uids == {discarded.uid}
    finally:
        _clear_partition(partition_id)


def test_infrastructure_group_filter_counts_only_final_output_per_session(
    tq_init, partition_id: str, monkeypatch
) -> None:
    monkeypatch.setenv("SWE_AGENT_ROLLOUT_TRAINING_COMPLETION_RATIO_THRESHOLD", "0.9")
    monkeypatch.setenv("SWE_AGENT_ROLLOUT_TRAINING_INFRA_FAILURE_GROUP_RATIO_THRESHOLD", "0.25")
    uid = _uid()
    for index, failure in enumerate([1, 0]):
        tq.kv_put(
            key=_trajectory_key(uid, session_id=0, index=index),
            partition_id=partition_id,
            fields={
                "input_ids": torch.tensor([1, 2, 3]),
                "extra_fields": {
                    "reward_extra_info": {"observed_infrastructure_failure": failure}
                },
            },
            tag={
                "is_prompt": False,
                "seq_len": 3,
                "global_steps": 0,
                "infrastructure_failure": bool(failure),
            },
        )
    _set_prompt_status(partition_id, uid, "finished", global_steps=0)
    rb = _make_rb(refill_fn=lambda _n: None, expected_rollout_n=1, expected_validation_rollout_n=1)
    _configure_infrastructure_partition(rb, partition_id, 1)
    try:
        rb._sync_metadata_from_transfer_queue()
        discarded_uids, ratios = rb._infrastructure_filtered_uids(partition_id)

        assert ratios[uid] == 0.0
        assert not discarded_uids
    finally:
        _clear_partition(partition_id)


@pytest.mark.parametrize("threshold", [None, 0.25])
def test_evaluator_group_exclusion_is_filtered_even_without_infrastructure_failure(
    tq_init, partition_id: str, threshold
) -> None:
    uid = _uid()
    tq.kv_put(
        key=_trajectory_key(uid),
        partition_id=partition_id,
        fields={"input_ids": torch.tensor([1, 2, 3])},
        tag={"is_prompt": False, "seq_len": 3, "global_steps": 0, "evaluator_group_excluded": True},
    )
    _set_prompt_status(partition_id, uid, "finished", global_steps=0)
    rb = _make_rb(refill_fn=lambda _n: None, expected_rollout_n=1)
    _configure_infrastructure_partition(rb, partition_id, 1)
    rb.infrastructure_failure_ratio_thresholds[partition_id] = threshold
    try:
        rb._sync_metadata_from_transfer_queue()
        discarded_uids, ratios = rb._infrastructure_filtered_uids(partition_id)
        assert discarded_uids == {uid}
        assert ratios[uid] == (1.0 if threshold is None else 0.0)
    finally:
        _clear_partition(partition_id)


def test_infrastructure_groups_are_replaced_before_completion_cutoff(
    tq_init, partition_id: str, monkeypatch
) -> None:
    monkeypatch.setenv("SWE_AGENT_ROLLOUT_TRAINING_COMPLETION_RATIO_THRESHOLD", "0.75")
    monkeypatch.setenv("SWE_AGENT_ROLLOUT_TRAINING_INFRA_FAILURE_GROUP_RATIO_THRESHOLD", "0.25")
    invalid = PromptSpec(
        uid=_uid(), status="finished", sessions=4, infrastructure_failures=[1, 1, 0, 0]
    )
    valid = [
        PromptSpec(uid=_uid(), status="finished", sessions=4, infrastructure_failures=[0, 0, 0, 0])
        for _ in range(2)
    ]
    running_uid = _uid()
    _produce(partition_id, [invalid, *valid]).join_and_check()
    tq.kv_put(
        key=running_uid,
        partition_id=partition_id,
        tag={"is_prompt": True, "status": "running", "global_steps": 0},
    )
    refiller = FakeRefiller(partition_id, global_steps=0, sessions=4)
    cancelled: list[list[str]] = []

    def cancel_fn(uids: list[str], validate: bool = False) -> int:
        cancelled.append(uids)
        for uid in uids:
            tq.kv_put(
                key=uid,
                partition_id=partition_id,
                tag={
                    "is_prompt": True,
                    "status": "failure",
                    "global_steps": 0,
                    "completion_ratio_cutoff": True,
                },
            )
        return len(uids)

    rb = _make_rb(
        refill_fn=refiller,
        cancel_fn=cancel_fn,
        train_batch_size=4,
        expected_rollout_n=4,
        expected_validation_rollout_n=4,
    )
    _configure_infrastructure_partition(rb, partition_id, 4)
    rb.completion_ratio_thresholds[partition_id] = 0.75
    try:
        batch, metrics = rb.sample(global_steps=0, partition_id=partition_id, batch_size=4)

        assert invalid.uid not in _uids_of(batch.keys)
        assert refiller.calls == [1]
        assert cancelled == [[running_uid]]
        assert metrics["validation/infrastructure_group_filter/discarded_groups"] == 1.0
        assert metrics["validation/infrastructure_group_filter/refilled_groups"] == 1
        assert metrics["validation/completion_ratio/terminal_groups"] == 3.0
        assert metrics["validation/completion_ratio/total_groups"] == 4.0
    finally:
        _clear_partition(partition_id)


def test_init_rejects_non_positive_sync_dapo_inflight_limit():
    with pytest.raises(ValueError, match="max_inflight_gen_batches"):
        _make_rb(
            refill_fn=lambda _n: None,
            filter_groups_metric="acc",
            max_inflight_gen_batches=0,
        )


# --------------------------------------------------------------------------- #
# _sync_metadata_from_transfer_queue: classification of polled metadata.
# --------------------------------------------------------------------------- #


def test_sync_metadata_classifies_keys(tq_init, partition_id):
    """The poll splits prompts by status and collects trajectory tags."""
    pending = PromptSpec(uid=_uid(), status="pending", sessions=0)
    running = PromptSpec(uid=_uid(), status="running", sessions=1)
    finished = PromptSpec(uid=_uid(), status="finished", sessions=2)
    failure = PromptSpec(uid=_uid(), status="failure", sessions=1)
    _produce(partition_id, [pending, running, finished, failure]).join_and_check()

    rb = _make_rb()
    try:
        rb._sync_metadata_from_transfer_queue()

        assert rb.pending_keys[partition_id] == {pending.uid}
        assert rb.running_keys[partition_id] == {running.uid}
        assert rb.finished_keys[partition_id] == {finished.uid}
        assert rb.failure_keys[partition_id] == {failure.uid}

        # All trajectory keys (and only those) land in the partition value map.
        expected_traj = set(running.trajectory_keys) | set(finished.trajectory_keys) | set(failure.trajectory_keys)
        assert set(rb.partitions[partition_id].keys()) == expected_traj
        for key in expected_traj:
            assert rb.partitions[partition_id][key] == {"is_prompt": False, "seq_len": 3, "global_steps": 0}
    finally:
        _clear_partition(partition_id)


def test_infrastructure_discard_persists_aligned_evidence_before_deletion(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    from verl.trainer.ppo.v1.replay_buffer import rollout_diagnostics
    monkeypatch.setenv("SWE_AGENT_INFRA_DISCARD_AUDIT_DIR", str(tmp_path))
    rb = _make_rb()
    rb.partitions["train"] = {"drop_0_0": {"infrastructure_failure": 1}, "keep_0_0": {}}
    monkeypatch.setattr(rb, "_infrastructure_filtered_uids", lambda pid: ({"drop"}, {"drop": 1.0}))
    monkeypatch.setattr(tq, "kv_batch_get", lambda **kw: {"extra_fields": SimpleNamespace(tolist=lambda: [
        {"reward_extra_info": {"evaluation_valid": False, "private_evaluator_artifact_path": "/private/evidence.json"},
         "secret_token": "do not persist"}])})
    def clear(pid, uids):
        files = list((tmp_path / "infrastructure_discard").glob("*.json"))
        assert len(files) == 1
        evidence = json.loads(files[0].read_text())
        assert evidence["diagnostic_coverage"] == "available"
        assert evidence["trajectories"][0]["uid"] == "drop_0_0"
        assert evidence["trajectories"][0]["evaluation_valid"] is False
        assert evidence["trajectories"][0]["infrastructure_failure"] == 1
        assert "secret_token" not in files[0].read_text()
    monkeypatch.setattr(rb, "_clear_groups", clear)
    _, metrics = rb._discard_infrastructure_groups("train")
    assert metrics["training/infrastructure_group_filter/evidence_persisted"] == 1
    assert rollout_diagnostics({"reward_extra_info": {"raw_score": 1}, "raw_score": 0})["raw_score"] == 0


def test_sync_metadata_unknown_status_raises(tq_init, partition_id):
    """An unrecognized prompt status is a hard error during the poll."""
    _produce(partition_id, [PromptSpec(uid=_uid(), status="bogus", sessions=0)]).join_and_check()

    rb = _make_rb()
    try:
        with pytest.raises(ValueError, match="Unknown status"):
            rb._sync_metadata_from_transfer_queue()
    finally:
        # The bogus prompt must be removed: every poll lists *all* partitions, so
        # leaving it behind would break unrelated tests.
        _clear_partition(partition_id)


def test_clear_groups_updates_active_snapshot(monkeypatch):
    rb = _make_rb()
    pid, dropped_uid, kept_uid = "p", "drop", "keep"
    dropped_trajectory = _trajectory_key(dropped_uid)
    kept_trajectory = _trajectory_key(kept_uid)
    rb.partitions[pid] = {dropped_trajectory: {}, kept_trajectory: {}}
    for status_keys in (rb.pending_keys, rb.running_keys, rb.finished_keys, rb.failure_keys):
        status_keys[pid] |= {dropped_uid, kept_uid}
    rb.prompt_global_steps[pid] = {dropped_uid: 1, kept_uid: 2}
    rb._dapo_classification_cache[pid] = {dropped_uid: 0.0, kept_uid: None}

    cleared: list[set[str]] = []
    monkeypatch.setattr(tq, "kv_clear", lambda *, keys, partition_id: cleared.append(set(keys)))

    rb._clear_groups(pid, {dropped_uid})

    assert cleared == [{dropped_uid, dropped_trajectory}]
    assert rb.partitions[pid] == {kept_trajectory: {}}
    for status_keys in (rb.pending_keys, rb.running_keys, rb.finished_keys, rb.failure_keys):
        assert status_keys[pid] == {kept_uid}
    assert rb.prompt_global_steps[pid] == {kept_uid: 2}
    assert rb._dapo_classification_cache[pid] == {kept_uid: None}


# --------------------------------------------------------------------------- #
# sample: end-to-end against a real TransferQueue.
# --------------------------------------------------------------------------- #


def test_sample_returns_finished_and_failure_trajectories(tq_init, partition_id):
    """sample picks trajectories belonging to finished/failure prompts and clears
    the sampled prompt keys from TransferQueue."""
    finished = PromptSpec(uid=_uid(), status="finished", sessions=2)
    failure = PromptSpec(uid=_uid(), status="failure", sessions=1)
    # Running prompt's trajectory must NOT be sampled.
    running = PromptSpec(uid=_uid(), status="running", sessions=1)

    _produce(partition_id, [finished, failure, running]).join_and_check()
    expected_keys = set(finished.trajectory_keys) | set(failure.trajectory_keys)

    rb = _make_rb()
    try:
        batch = _sample(rb, partition_id, batch_size=2)

        assert batch.partition_id == partition_id
        assert set(batch.keys) == expected_keys
        assert len(batch.tags) == len(batch.keys)

        # The two sampled prompt keys are consumed from TransferQueue; the running
        # prompt and all trajectory values remain.
        remaining = tq.kv_list(partition_id=partition_id).get(partition_id, {})
        assert finished.uid not in remaining
        assert failure.uid not in remaining
        assert running.uid in remaining
    finally:
        _clear_partition(partition_id)


def test_sync_failure_without_trajectories_uses_padding_path_by_default(tq_init, partition_id):
    finished = PromptSpec(uid=_uid(), status="finished", sessions=1)
    empty_failure = PromptSpec(uid=_uid(), status="failure", sessions=0)
    _produce(partition_id, [finished, empty_failure]).join_and_check()

    rb = _make_rb()
    try:
        batch, metrics = rb.sample(global_steps=0, partition_id=partition_id, batch_size=2)

        assert _uids_of(batch.keys) == {finished.uid}
        assert metrics == {}
    finally:
        _clear_partition(partition_id)


def test_sync_all_empty_failures_raise_without_refill(tq_init, partition_id):
    failures = [PromptSpec(uid=_uid(), status="failure", sessions=0) for _ in range(2)]
    _produce(partition_id, failures).join_and_check()

    rb = _make_rb()
    try:
        with pytest.raises(RuntimeError, match="sync_refill_failed_groups"):
            rb.sample(global_steps=0, partition_id=partition_id, batch_size=2)
    finally:
        _clear_partition(partition_id)


def test_sample_prioritizes_smallest_global_steps(tq_init, partition_id):
    """When more prompts are ready than ``batch_size``, sample must hand out the
    oldest ones first (smallest ``global_steps``), leaving the newer surplus."""
    # Produced out of step order on purpose; selection must follow global_steps.
    oldest = PromptSpec(uid=_uid(), status="finished", sessions=1, global_steps=1)
    middle = PromptSpec(uid=_uid(), status="finished", sessions=1, global_steps=2)
    newest = PromptSpec(uid=_uid(), status="finished", sessions=1, global_steps=5)
    _produce(partition_id, [newest, oldest, middle]).join_and_check()

    rb = _make_rb()
    try:
        batch = _sample(rb, partition_id, batch_size=2)

        # The two smallest-step prompts are picked; the newest is left behind.
        assert _uids_of(batch.keys) == {oldest.uid, middle.uid}

        remaining = tq.kv_list(partition_id=partition_id).get(partition_id, {})
        assert oldest.uid not in remaining
        assert middle.uid not in remaining
        assert newest.uid in remaining
    finally:
        _clear_partition(partition_id)


def test_sample_blocks_until_enough_then_unblocks(tq_init, partition_id):
    """sample stays blocked while fewer than batch_size prompts are ready and
    returns once the producer thread supplies the missing group."""
    # One group ready up front -> not enough for batch_size=2.
    _produce(partition_id, [PromptSpec(uid=_uid(), status="finished", sessions=1)]).join_and_check()

    rb = _make_rb()
    consumer = SampleConsumer(rb, partition_id, batch_size=2)
    try:
        consumer.start()

        # Give the consumer time to poll a few times; it must still be blocked.
        time.sleep(POLL_INTERVAL * 5)
        assert consumer.is_alive(), "sample returned before batch_size prompts were ready"

        # The producer thread supplies a second group; sample can now complete.
        _produce(partition_id, [PromptSpec(uid=_uid(), status="finished", sessions=1)]).join_and_check()

        batch = consumer.result_or_raise()
        assert len(batch.keys) == 2
    finally:
        consumer.join(timeout=10.0)
        _clear_partition(partition_id)


def test_sync_grpo_step_returns_complete_groups(tq_init, partition_id):
    """A synchronous PPO/GRPO step submits batch_size prompts, each a GRPO group of
    n sessions; one sample must return every trajectory as whole groups."""
    n_prompts = 3
    n_sessions = 4  # GRPO rollout.n
    specs = [PromptSpec(uid=_uid(), status="finished", sessions=n_sessions) for _ in range(n_prompts)]
    _produce(partition_id, specs).join_and_check()

    rb = _make_rb()
    try:
        batch = _sample(rb, partition_id, batch_size=n_prompts)

        # Every prompt's full GRPO group is present, nothing more, nothing less.
        assert len(batch.keys) == n_prompts * n_sessions
        assert set(batch.keys) == {k for spec in specs for k in spec.trajectory_keys}
        # Each sampled prompt contributes exactly n_sessions trajectories.
        per_uid: dict[str, int] = {}
        for key in batch.keys:
            per_uid[key.split("_")[0]] = per_uid.get(key.split("_")[0], 0) + 1
        assert set(per_uid.values()) == {n_sessions}
    finally:
        _clear_partition(partition_id)


def test_async_overproduction_drains_in_batches_without_duplicates(tq_init, partition_id):
    """An async rollouter over-produces; sequential samples drain the surplus
    batch_size complete groups at a time without ever re-selecting a prompt.

    ``sample`` only re-polls TransferQueue while it is under-filled, and it clears
    just the sampled *prompt* keys (trajectory values stay). The real trainer
    re-polls metadata in the gap between two ``sample`` calls; we reproduce that
    gap deterministically by re-syncing so a follow-up ``sample`` cannot re-select
    consumed prompts.
    """
    n_prompts = 5
    n_sessions = 2
    batch_size = 2
    specs = [PromptSpec(uid=_uid(), status="finished", sessions=n_sessions) for _ in range(n_prompts)]
    _produce(partition_id, specs).join_and_check()
    all_uids = {spec.uid for spec in specs}

    rb = _make_rb()
    try:
        collected_keys: list[str] = []
        consumed_uids: set[str] = set()

        # Drain 2 + 2 + 1 prompts across three samples.
        for bs in (batch_size, batch_size, n_prompts - 2 * batch_size):
            batch = _sample(rb, partition_id, batch_size=bs)

            sampled_uids = _uids_of(batch.keys)
            assert len(sampled_uids) == bs
            assert len(batch.keys) == bs * n_sessions
            assert not (sampled_uids & consumed_uids), "a prompt was handed out twice"

            collected_keys.extend(batch.keys)
            consumed_uids |= sampled_uids
            # Mimic the trainer's metadata refresh between two sample() calls.
            rb._sync_metadata_from_transfer_queue()

        # The whole surplus was drained exactly once.
        assert consumed_uids == all_uids
        assert len(collected_keys) == n_prompts * n_sessions
        assert len(set(collected_keys)) == len(collected_keys)
    finally:
        _clear_partition(partition_id)


# --------------------------------------------------------------------------- #
# sample: off-policy "drop" strategy.
# --------------------------------------------------------------------------- #


def test_drop_refills_stale_groups_and_reports_metrics(tq_init, partition_id):
    """drop discards finished groups whose staleness strictly exceeds the threshold *inside the
    polling loop*, calls ``refill_fn`` for an equal count, and returns a full, fresh batch."""
    # staleness = (global_steps - prompt_global_steps + 1); drop when staleness > threshold(=2).
    # At global_steps=5:
    #   stale    gs=0 -> 6 > 2 -> dropped (and refilled)
    #   boundary gs=4 -> 2 not > 2 -> kept (boundary is inclusive on the keep side)
    #   fresh    gs=5 -> 1 -> kept
    stale = PromptSpec(uid=_uid(), status="finished", sessions=1, global_steps=0)
    boundary = PromptSpec(uid=_uid(), status="finished", sessions=1, global_steps=4)
    fresh = PromptSpec(uid=_uid(), status="finished", sessions=1, global_steps=5)
    inflight = PromptSpec(uid=_uid(), status="running", sessions=1, global_steps=0)
    _produce(partition_id, [stale, boundary, fresh, inflight]).join_and_check()

    refiller = FakeRefiller(partition_id, global_steps=5)
    rb = _make_rb(
        trainer_mode="colocate_async",
        max_off_policy_strategy="drop",
        max_off_policy_threshold=2,
        refill_fn=refiller,
    )
    try:
        batch, metrics = rb.sample(global_steps=5, partition_id=partition_id, batch_size=3)

        # Full batch (batch_size=3), no stale group, dropped slot backfilled by the refiller.
        sampled_uids = _uids_of(batch.keys)
        assert len(sampled_uids) == 3
        assert stale.uid not in sampled_uids
        assert inflight.uid not in sampled_uids
        assert {boundary.uid, fresh.uid} <= sampled_uids
        assert len(sampled_uids & set(refiller.produced_uids)) == 1

        # The dropped group's prompt and trajectory keys are gone from TransferQueue.
        remaining = tq.kv_list(partition_id=partition_id).get(partition_id, {})
        assert stale.uid not in remaining
        assert stale.trajectory_keys[0] not in remaining

        assert refiller.calls == [1]

        # partition_id is a random test id (not "train") -> "validation" prefix.
        assert metrics["validation/off_policy/evicted_samples"] == 1
        assert metrics["validation/off_policy/evicted_samples_staleness/mean"] == 6
        assert metrics["validation/off_policy/evicted_samples_staleness/max"] == 6
        assert metrics["validation/off_policy/evicted_samples_staleness/min"] == 6
    finally:
        _clear_partition(partition_id)


def test_drop_uses_one_snapshot_per_poll_iteration(tq_init, partition_id):
    """A stale running prompt that finishes during refill is dropped on the next snapshot.

    This reproduces the production race where a second metadata sync after refill exposed newly
    finished stale prompts to selection without running the drop pass again.
    """
    stale_finished = PromptSpec(uid=_uid(), status="finished", sessions=1, global_steps=0)
    stale_running = PromptSpec(uid=_uid(), status="running", sessions=1, global_steps=0)
    _produce(partition_id, [stale_finished, stale_running]).join_and_check()

    refiller = FakeRefiller(partition_id, global_steps=5)

    def finish_during_first_refill(num_prompts: int) -> int:
        if not refiller.calls:
            _set_prompt_status(partition_id, stale_running.uid, "finished", global_steps=0)
        return refiller(num_prompts)

    rb = _make_rb(
        trainer_mode="colocate_async",
        max_off_policy_strategy="drop",
        max_off_policy_threshold=1,
        refill_fn=finish_during_first_refill,
    )
    try:
        batch, metrics = rb.sample(global_steps=5, partition_id=partition_id, batch_size=1)

        sampled_uids = _uids_of(batch.keys)
        assert len(sampled_uids) == 1
        assert not (sampled_uids & {stale_finished.uid, stale_running.uid})
        assert sampled_uids <= set(refiller.produced_uids)
        assert refiller.calls == [1, 1]
        assert metrics["validation/off_policy/evicted_samples"] == 2
    finally:
        _clear_partition(partition_id)


# --------------------------------------------------------------------------- #
# sample: off-policy "wait" (dropless) strategy.
# --------------------------------------------------------------------------- #


def test_wait_blocks_until_stale_inflight_finishes(tq_init, partition_id):
    """wait holds back a full batch while a stale in-flight prompt exists, then
    proceeds (without dropping it) once it terminates."""
    threshold, g = 2, 5
    fresh = [PromptSpec(uid=_uid(), status="finished", sessions=1, global_steps=g) for _ in range(2)]
    # In-flight prompt: staleness (5 - 0 + 1) = 6 >= 2 -> blocks sampling.
    stale = PromptSpec(uid=_uid(), status="running", sessions=1, global_steps=0)
    _produce(partition_id, fresh + [stale]).join_and_check()

    rb = _make_rb(trainer_mode="colocate_async", max_off_policy_strategy="wait", max_off_policy_threshold=threshold)
    consumer = SampleConsumer(rb, partition_id, batch_size=2, global_steps=g)
    try:
        consumer.start()

        time.sleep(POLL_INTERVAL * 5)
        assert consumer.is_alive(), "wait must block while a stale in-flight prompt exists"

        # The stale group terminates -> no in-flight prompt at threshold -> unblock.
        # (clear+put is not atomic, so the consumer may proceed on the two fresh
        # groups the instant the running prompt disappears; either way it returns a
        # full batch_size batch -- droplessness is asserted separately below.)
        _set_prompt_status(partition_id, stale.uid, "finished", global_steps=0)

        batch = consumer.result_or_raise()
        assert len(_uids_of(batch.keys)) == 2
    finally:
        consumer.join(timeout=10.0)
        _clear_partition(partition_id)


def test_wait_keeps_stale_terminal_trajectories(tq_init, partition_id):
    """wait is dropless: a finished-but-very-stale group that ``drop`` would
    discard is still returned, with no drop metrics."""
    # staleness (100 - 0 + 1) = 101, far above threshold=2; "drop" would
    # remove it, "wait" keeps it. No in-flight prompts, so sampling never blocks.
    stale = PromptSpec(uid=_uid(), status="finished", sessions=1, global_steps=0)
    _produce(partition_id, [stale]).join_and_check()

    rb = _make_rb(trainer_mode="colocate_async", max_off_policy_strategy="wait", max_off_policy_threshold=2)
    try:
        batch, metrics = rb.sample(global_steps=100, partition_id=partition_id, batch_size=1)

        assert set(batch.keys) == set(stale.trajectory_keys)
        assert metrics == {}
    finally:
        _clear_partition(partition_id)


def test_wait_does_not_block_when_inflight_is_fresh(tq_init, partition_id):
    """wait proceeds immediately when every in-flight prompt is below threshold,
    and never drops (no drop metrics)."""
    threshold, g = 4, 3
    finished = [PromptSpec(uid=_uid(), status="finished", sessions=1, global_steps=g) for _ in range(2)]
    # In-flight prompt staleness (3 - 3 + 1) = 1 < 4 -> does not block.
    inflight = PromptSpec(uid=_uid(), status="running", sessions=1, global_steps=g)
    _produce(partition_id, finished + [inflight]).join_and_check()

    rb = _make_rb(trainer_mode="colocate_async", max_off_policy_strategy="wait", max_off_policy_threshold=threshold)
    try:
        batch, metrics = rb.sample(global_steps=g, partition_id=partition_id, batch_size=2)

        assert _uids_of(batch.keys) == {spec.uid for spec in finished}
        assert metrics == {}
    finally:
        _clear_partition(partition_id)


# --------------------------------------------------------------------------- #
# mode matrix: stale drop, DAPO, and rollout failure.
# --------------------------------------------------------------------------- #


def test_init_rejects_dapo_without_refill_fn():
    with pytest.raises(ValueError, match="requires refill_fn"):
        _make_rb(filter_groups_metric="acc", refill_fn=None)


def test_init_rejects_sync_failure_refill_without_refill_fn():
    with pytest.raises(ValueError, match="requires refill_fn"):
        _make_rb(sync_refill_failed_groups=True, refill_fn=None)


def test_init_rejects_batched_sync_failure_refill():
    with pytest.raises(ValueError, match="requires gen_batch_size=1"):
        _make_rb(sync_refill_failed_groups=True, refill_fn=lambda _n: None, gen_batch_size=2)


def test_validation_never_drops_terminal_groups(tq_init):
    partition_id = "val"
    _clear_partition(partition_id)
    stale_dapo = PromptSpec(uid=_uid(), status="finished", sessions=2, global_steps=0, rewards=[1.0, 1.0])
    stale_failure = PromptSpec(uid=_uid(), status="failure", global_steps=0)
    _produce(partition_id, [stale_dapo, stale_failure]).join_and_check()

    refiller = FakeRefiller(partition_id, global_steps=5, sessions=2, rewards=[0.0, 1.0])
    rb = _make_rb(
        trainer_mode="colocate_async",
        max_off_policy_strategy="drop",
        max_off_policy_threshold=1,
        refill_fn=refiller,
        filter_groups_metric="acc",
    )
    try:
        batch, metrics = rb.sample(global_steps=5, partition_id=partition_id, batch_size=2)
        assert _uids_of(batch.keys) == {stale_dapo.uid, stale_failure.uid}
        assert refiller.calls == []
        assert metrics == {}
    finally:
        _clear_partition(partition_id)


def test_validation_ignores_sync_failure_refill(tq_init):
    partition_id = "val"
    _clear_partition(partition_id)
    finished = PromptSpec(uid=_uid(), status="finished")
    empty_failure = PromptSpec(uid=_uid(), status="failure", sessions=0)
    _produce(partition_id, [finished, empty_failure]).join_and_check()

    refiller = FakeRefiller(partition_id, global_steps=1)
    rb = _make_rb(sync_refill_failed_groups=True, refill_fn=refiller)
    try:
        batch, metrics = rb.sample(global_steps=1, partition_id=partition_id, batch_size=2)

        assert _uids_of(batch.keys) == {finished.uid}
        assert refiller.calls == []
        assert metrics == {}
    finally:
        _clear_partition(partition_id)


@pytest.mark.parametrize("strategy", ["drop", "wait"])
def test_sync_ignores_off_policy_strategy(tq_init, partition_id, strategy):
    stale = PromptSpec(uid=_uid(), status="finished", global_steps=0)
    fresh = PromptSpec(uid=_uid(), status="finished", global_steps=5)
    _produce(partition_id, [stale, fresh]).join_and_check()

    rb = _make_rb(max_off_policy_strategy=strategy, max_off_policy_threshold=1)
    try:
        batch, metrics = rb.sample(global_steps=5, partition_id=partition_id, batch_size=2)
        assert _uids_of(batch.keys) == {stale.uid, fresh.uid}
        assert metrics == {}
    finally:
        _clear_partition(partition_id)


def test_async_failure_refills_exact_missing_count(tq_init, partition_id):
    finished = PromptSpec(uid=_uid(), status="finished")
    failure = PromptSpec(uid=_uid(), status="failure")
    _produce(partition_id, [finished, failure]).join_and_check()

    refiller = FakeRefiller(partition_id, global_steps=1)
    rb = _make_rb(trainer_mode="colocate_async", refill_fn=refiller)
    try:
        batch, metrics = rb.sample(global_steps=1, partition_id=partition_id, batch_size=2)
        sampled_uids = _uids_of(batch.keys)
        assert failure.uid not in sampled_uids
        assert finished.uid in sampled_uids
        assert refiller.calls == [1]
        assert metrics["validation/rollout_failure/evicted_samples"] == 1
    finally:
        _clear_partition(partition_id)


def test_sync_failure_refill_replaces_empty_failed_groups(tq_init, partition_id):
    finished = PromptSpec(uid=_uid(), status="finished")
    failure = PromptSpec(uid=_uid(), status="failure", sessions=0)
    _produce(partition_id, [finished, failure]).join_and_check()

    refiller = FakeRefiller(partition_id, global_steps=1)
    rb = _make_rb(sync_refill_failed_groups=True, refill_fn=refiller)
    try:
        batch, metrics = rb.sample(global_steps=1, partition_id=partition_id, batch_size=2)
        sampled_uids = _uids_of(batch.keys)

        assert len(sampled_uids) == 2
        assert finished.uid in sampled_uids
        assert failure.uid not in sampled_uids
        assert refiller.calls == [1]
        assert metrics["validation/rollout_failure/evicted_samples"] == 1
    finally:
        _clear_partition(partition_id)


def test_sync_failure_refill_replaces_multiple_empty_failed_groups(tq_init, partition_id):
    failures = [PromptSpec(uid=_uid(), status="failure", sessions=0) for _ in range(3)]
    _produce(partition_id, failures).join_and_check()

    refiller = FakeRefiller(partition_id, global_steps=1)
    rb = _make_rb(sync_refill_failed_groups=True, refill_fn=refiller, train_batch_size=3)
    try:
        batch, metrics = rb.sample(global_steps=1, partition_id=partition_id, batch_size=3)

        assert _uids_of(batch.keys) == set(refiller.produced_uids)
        assert refiller.calls == [3]
        assert metrics["validation/rollout_failure/evicted_samples"] == 3
    finally:
        _clear_partition(partition_id)


def test_sync_failure_refill_keeps_materializable_failed_groups(tq_init, partition_id):
    finished = PromptSpec(uid=_uid(), status="finished")
    failure = PromptSpec(uid=_uid(), status="failure", sessions=1)
    _produce(partition_id, [finished, failure]).join_and_check()

    refiller = FakeRefiller(partition_id, global_steps=1)
    rb = _make_rb(sync_refill_failed_groups=True, refill_fn=refiller)
    try:
        batch, metrics = rb.sample(global_steps=1, partition_id=partition_id, batch_size=2)

        assert _uids_of(batch.keys) == {finished.uid, failure.uid}
        assert refiller.calls == []
        assert metrics == {}
    finally:
        _clear_partition(partition_id)


def test_sync_failure_refill_does_not_drain_unrelated_inflight(tq_init, partition_id):
    finished = PromptSpec(uid=_uid(), status="finished")
    empty_failure = PromptSpec(uid=_uid(), status="failure", sessions=0)
    running = PromptSpec(uid=_uid(), status="running", sessions=0)
    _produce(partition_id, [finished, empty_failure, running]).join_and_check()

    refiller = FakeRefiller(partition_id, global_steps=1)
    rb = _make_rb(sync_refill_failed_groups=True, refill_fn=refiller)
    consumer = SampleConsumer(rb, partition_id, batch_size=1, global_steps=1)
    try:
        consumer.start()
        batch = consumer.result_or_raise(timeout=2)

        assert _uids_of(batch.keys) == {finished.uid}
        assert refiller.calls == []
        assert consumer.metrics is not None
        assert consumer.metrics["validation/rollout_failure/evicted_samples"] == 1
        remaining = tq.kv_list(partition_id=partition_id).get(partition_id, {})
        assert running.uid in remaining
    finally:
        if consumer.is_alive():
            _set_prompt_status(partition_id, running.uid, "finished", global_steps=1)
            consumer.join(timeout=2)
        _clear_partition(partition_id)


def test_async_dapo_refills_exact_missing_count(tq_init, partition_id):
    all_same = PromptSpec(uid=_uid(), status="finished", sessions=2, rewards=[1.0, 1.0])
    mixed = PromptSpec(uid=_uid(), status="finished", sessions=2, rewards=[0.0, 1.0])
    _produce(partition_id, [all_same, mixed]).join_and_check()

    refiller = FakeRefiller(partition_id, global_steps=1, sessions=2, rewards=[0.0, 1.0])
    rb = _make_rb(
        trainer_mode="colocate_async",
        refill_fn=refiller,
        filter_groups_metric="acc",
    )
    try:
        batch, metrics = rb.sample(global_steps=1, partition_id=partition_id, batch_size=2)
        sampled_uids = _uids_of(batch.keys)
        assert all_same.uid not in sampled_uids
        assert mixed.uid in sampled_uids
        assert refiller.calls == [1]
        assert metrics["validation/filter_groups/evicted_samples"] == 1
    finally:
        _clear_partition(partition_id)


def test_dapo_classification_cache_fetches_only_new_finished_groups(tq_init, partition_id, monkeypatch):
    first_filtered = PromptSpec(uid=_uid(), status="finished", sessions=2, rewards=[1.0, 1.0])
    mixed = PromptSpec(uid=_uid(), status="finished", sessions=2, rewards=[0.0, 1.0])
    _produce(partition_id, [first_filtered, mixed]).join_and_check()

    requested_keys: list[set[str]] = []
    original_kv_batch_get = tq.kv_batch_get

    def recording_kv_batch_get(*args, **kwargs):
        requested_keys.append(set(kwargs["keys"]))
        return original_kv_batch_get(*args, **kwargs)

    monkeypatch.setattr(tq, "kv_batch_get", recording_kv_batch_get)
    rb = _make_rb(filter_groups_metric="acc", refill_fn=lambda _n: None)
    try:
        rb._sync_metadata_from_transfer_queue()
        first_result = rb._dapo_filtered_keys(partition_id)
        assert first_result[0] == {first_filtered.uid}
        assert dict(first_result[1]) == {1.0: 1}
        assert rb._dapo_filtered_keys(partition_id) == first_result

        second_filtered = PromptSpec(uid=_uid(), status="finished", sessions=2, rewards=[0.0, 0.0])
        _produce(partition_id, [second_filtered]).join_and_check()
        rb._sync_metadata_from_transfer_queue()
        filtered_uids, filtered_counts = rb._dapo_filtered_keys(partition_id)

        assert filtered_uids == {first_filtered.uid, second_filtered.uid}
        assert dict(filtered_counts) == {0.0: 1, 1.0: 1}
        assert requested_keys == [
            set(first_filtered.trajectory_keys + mixed.trajectory_keys),
            set(second_filtered.trajectory_keys),
        ]
    finally:
        _clear_partition(partition_id)


@pytest.mark.parametrize("status", ["finished", "failure"])
@pytest.mark.parametrize("shared_reward", [0.0, 1.0, -0.1])
@pytest.mark.parametrize("mixed", [False, True])
def test_dapo_compares_only_verified_final_session_rewards(monkeypatch, status, shared_reward, mixed):
    rb = _make_rb(filter_groups_metric="score", refill_fn=lambda _n: None)
    getattr(rb, f"{status}_keys")["train"].add("group")
    rb.partitions["train"] = {
        **{f"group_{i}_0": {} for i in range(8)},
        "group_0_1": {},
    }
    rb.partitions["train"]["group_6_0"] = {"unverified_trajectory": True}
    rb.partitions["train"]["group_7_0"] = {"evaluator_group_excluded": True}
    scores = {key: shared_reward for key in rb.partitions["train"]}
    scores["group_0_0"] = -1000.0  # Intermediate output must not affect final-session variance.
    scores["group_6_0"] = float("nan")
    scores["group_7_0"] = 1000.0
    if mixed:
        scores["group_5_0"] = shared_reward + 1.0
    requested_keys = []

    def fetch(*, keys, **kwargs):
        requested_keys.extend(keys)
        return {"extra_fields": [{"reward_extra_info": {"score": scores[key]}} for key in keys]}

    monkeypatch.setattr(tq, "kv_batch_get", fetch)
    filtered, counts = rb._dapo_filtered_keys("train")
    assert filtered == (set() if mixed else {"group"})
    assert dict(counts) == ({} if mixed else {shared_reward: 1})
    assert set(requested_keys) == {"group_0_1", *(f"group_{i}_0" for i in range(1, 6))}


@pytest.mark.parametrize("rejection", ["infrastructure", "unverified", "both"])
def test_dapo_failure_filters_do_not_refill_when_batch_is_already_full(
    tq_init, monkeypatch, tmp_path, rejection
):
    monkeypatch.setenv("SWE_AGENT_ROLLOUT_TRAINING_INFRA_FAILURE_GROUP_RATIO_THRESHOLD", "0.25")
    monkeypatch.setenv("SWE_AGENT_ROLLOUT_TRAINING_UNVERIFIED_TRAJECTORY_RATIO_THRESHOLD", "0.25")
    bad = PromptSpec(
        uid=_uid(), status="finished", sessions=8, rewards=[0.0] * 8,
        infrastructure_failures=[1] * 3 + [0] * 5 if rejection != "unverified" else None,
        unverified=[False] * 3 + [True] * 3 + [False] * 2 if rejection != "infrastructure" else None,
    )
    good = [PromptSpec(uid=_uid(), status="finished", sessions=8, rewards=[0.0, 1.0] * 4) for _ in range(2)]
    refiller = FakeRefiller("train", global_steps=0, sessions=8, rewards=[0.0, 1.0] * 4)
    rb = _make_rb(filter_groups_metric="acc", refill_fn=refiller, expected_rollout_n=8, train_batch_size=3)
    rb.trainer_config = {"rollout_data_dir": str(tmp_path)}
    try:
        _produce("train", [bad, *good]).join_and_check()
        batch, metrics = rb.sample(global_steps=0, partition_id="train", batch_size=2)
        assert _uids_of(batch.keys) == {spec.uid for spec in good}
        assert refiller.calls == []
        assert metrics["training/infrastructure_group_filter/discarded_groups"] == 1
        assert metrics.get("training/filter_groups/evicted_samples", 0) == 0
        assert metrics.get("training/infrastructure_group_filter/refilled_groups", 0) == 0
        if rejection != "infrastructure":
            assert metrics["training/unverified_group_filter/discarded_groups"] == 1
    finally:
        _clear_partition("train")


def test_dapo_combines_failure_and_zero_advantage_filters(tq_init, monkeypatch, tmp_path):
    monkeypatch.setenv("SWE_AGENT_ROLLOUT_TRAINING_INFRA_FAILURE_GROUP_RATIO_THRESHOLD", "0.25")
    monkeypatch.setenv("SWE_AGENT_ROLLOUT_TRAINING_UNVERIFIED_TRAJECTORY_RATIO_THRESHOLD", "0.25")
    bad_infra = PromptSpec(
        uid=_uid(), status="finished", sessions=8, rewards=[0.0, 1.0] * 4,
        infrastructure_failures=[1] * 3 + [0] * 5,
    )
    bad_unverified = PromptSpec(
        uid=_uid(), status="failure", sessions=8, rewards=[0.0, 1.0] * 4,
        unverified=[True] * 3 + [False] * 5,
    )
    uniform_verified = PromptSpec(
        uid=_uid(), status="finished", sessions=8, rewards=[1.0] * 6 + [0.0, 0.0],
        unverified=[False] * 6 + [True] * 2,
    )
    boundary = PromptSpec(
        uid=_uid(), status="finished", sessions=8, rewards=[0.0, 1.0] * 4,
        infrastructure_failures=[1, 1] + [0] * 6,
    )
    refiller = FakeRefiller("train", global_steps=0, sessions=8, rewards=[0.0, 1.0] * 4)
    rb = _make_rb(filter_groups_metric="acc", refill_fn=refiller, expected_rollout_n=8, train_batch_size=4)
    rb.trainer_config = {"rollout_data_dir": str(tmp_path)}
    try:
        _produce("train", [bad_infra, bad_unverified, uniform_verified, boundary]).join_and_check()
        batch, metrics = rb.sample(global_steps=0, partition_id="train", batch_size=4)
        selected = _uids_of(batch.keys)
        assert len(selected) == 4 and boundary.uid in selected
        assert not selected & {bad_infra.uid, bad_unverified.uid, uniform_verified.uid}
        assert refiller.calls == [4]  # One per failure-filter rejection, two per uniform group.
        assert metrics["training/infrastructure_group_filter/discarded_groups"] == 2
        assert metrics["training/infrastructure_group_filter/refilled_groups"] == 2
        assert metrics["training/unverified_group_filter/discarded_groups"] == 1
        assert metrics["training/filter_groups/evicted_samples"] == 1
        assert metrics["training/filter_groups/discarded_surplus_samples"] == 1
    finally:
        _clear_partition("train")


def test_dapo_bounds_combined_refills_and_does_not_refill_failures_while_draining(tq_init, monkeypatch, tmp_path):
    monkeypatch.setenv("SWE_AGENT_ROLLOUT_TRAINING_INFRA_FAILURE_GROUP_RATIO_THRESHOLD", "0.25")
    bad = PromptSpec(
        uid=_uid(), status="finished", sessions=8, rewards=[0.0, 1.0] * 4,
        infrastructure_failures=[1] * 3 + [0] * 5,
    )
    uniform = PromptSpec(uid=_uid(), status="finished", sessions=8, rewards=[0.0] * 8)
    slow = PromptSpec(
        uid=_uid(), status="running", sessions=8, rewards=[0.0, 1.0] * 4,
        infrastructure_failures=[1] * 3 + [0] * 5,
    )
    refiller = FakeRefiller("train", global_steps=0, sessions=8, rewards=[0.0, 1.0] * 4)
    drain_started = threading.Event()

    def bounded_refill(count):
        pending = len(rb.pending_keys["train"]) + len(rb.running_keys["train"])
        assert pending + count <= 3
        return refiller(count)

    rb = _make_rb(
        filter_groups_metric="acc", refill_fn=bounded_refill, expected_rollout_n=8, train_batch_size=3,
    )
    rb.trainer_config = {"rollout_data_dir": str(tmp_path)}
    original_wait = rb._wait_for_next_poll

    def wait(*args):
        if len(rb.finished_keys["train"]) >= 3:
            drain_started.set()
        return original_wait(*args)

    monkeypatch.setattr(rb, "_wait_for_next_poll", wait)
    consumer = SampleConsumer(rb, "train", batch_size=3)
    try:
        _produce("train", [bad, uniform, slow]).join_and_check()
        consumer.start()
        assert drain_started.wait(timeout=10)
        assert refiller.calls == [2, 1]
        assert consumer.is_alive()
        _set_prompt_status("train", slow.uid, "finished", global_steps=0)
        batch = consumer.result_or_raise()
        assert _uids_of(batch.keys) == set(refiller.produced_uids)
        assert refiller.calls == [2, 1]
        assert consumer.metrics["training/infrastructure_group_filter/discarded_groups"] == 2
        assert consumer.metrics["training/infrastructure_group_filter/refilled_groups"] == 1
        assert consumer.metrics["training/filter_groups/evicted_samples"] == 1
    finally:
        if consumer.is_alive():
            _set_prompt_status("train", slow.uid, "finished", global_steps=0)
            consumer.join(timeout=2)
        _clear_partition("train")


def test_terminal_eviction_reasons_has_no_cross_call_hidden_state(tq_init, partition_id):
    """An interleaved validation lookup must not overwrite a pending training breakdown."""
    all_zero = PromptSpec(uid=_uid(), status="finished", sessions=2, rewards=[0.0, 0.0])
    all_one = PromptSpec(uid=_uid(), status="finished", sessions=2, rewards=[1.0, 1.0])
    _produce(partition_id, [all_zero, all_one]).join_and_check()

    rb = _make_rb(trainer_mode="colocate_async", filter_groups_metric="acc", refill_fn=lambda _n: None)
    try:
        rb._sync_metadata_from_transfer_queue()
        train_reasons = rb._terminal_eviction_reasons(global_steps=0, partition_id=partition_id)
        rb._terminal_eviction_reasons(global_steps=0, partition_id="val")

        _evicted, _stale_count, dapo_count, metrics = rb._evict_terminal_groups(
            global_steps=0, partition_id=partition_id, eviction_reasons=train_reasons
        )
        from verl.trainer.ppo.v1.replay_buffer import DAPO_FILTERED_REWARD_COUNTS_KEY

        assert dapo_count == 2
        assert metrics[DAPO_FILTERED_REWARD_COUNTS_KEY] == {0.0: 1, 1.0: 1}
    finally:
        _clear_partition(partition_id)


def test_accumulate_eviction_metrics_merges_filtered_reward_counts():
    """The dict-valued DAPO diagnostic accumulates additively across poll iterations."""
    from verl.trainer.ppo.v1.replay_buffer import (
        DAPO_FILTERED_REWARD_COUNTS_KEY,
        _accumulate_eviction_metrics,
    )

    acc: dict = {}
    _accumulate_eviction_metrics(
        acc,
        {DAPO_FILTERED_REWARD_COUNTS_KEY: {0.0: 3, 1.0: 1}, "training/filter_groups/evicted_samples": 4},
        stale_count=0,
    )
    _accumulate_eviction_metrics(
        acc,
        {DAPO_FILTERED_REWARD_COUNTS_KEY: {0.0: 2}, "training/filter_groups/evicted_samples": 2},
        stale_count=0,
    )

    assert acc[DAPO_FILTERED_REWARD_COUNTS_KEY] == {0.0: 5, 1.0: 1}
    assert acc["training/filter_groups/evicted_samples"] == 6


def test_dapo_reports_filtered_reward_value_breakdown(tq_init, partition_id):
    """The DAPO diagnostic maps each no-signal group's shared metric value to a count."""
    from verl.trainer.ppo.v1.replay_buffer import DAPO_FILTERED_REWARD_COUNTS_KEY

    # Two groups collapse to acc=0.0, one to acc=1.0; one mixed group survives filtering.
    all_zero_a = PromptSpec(uid=_uid(), status="finished", sessions=2, rewards=[0.0, 0.0])
    all_zero_b = PromptSpec(uid=_uid(), status="finished", sessions=2, rewards=[0.0, 0.0])
    all_one = PromptSpec(uid=_uid(), status="finished", sessions=2, rewards=[1.0, 1.0])
    mixed = PromptSpec(uid=_uid(), status="finished", sessions=2, rewards=[0.0, 1.0])
    _produce(partition_id, [all_zero_a, all_zero_b, all_one, mixed]).join_and_check()

    refiller = FakeRefiller(partition_id, global_steps=1, sessions=2, rewards=[0.0, 1.0])
    rb = _make_rb(trainer_mode="colocate_async", refill_fn=refiller, filter_groups_metric="acc")
    try:
        _batch, metrics = rb.sample(global_steps=1, partition_id=partition_id, batch_size=1)
        assert metrics["validation/filter_groups/evicted_samples"] == 3
        assert metrics[DAPO_FILTERED_REWARD_COUNTS_KEY] == {0.0: 2, 1.0: 1}
    finally:
        _clear_partition(partition_id)


def test_sync_dapo_refills_twice_and_clears_surplus(tq_init, partition_id):
    all_same = PromptSpec(uid=_uid(), status="finished", sessions=2, rewards=[0.0, 0.0])
    mixed = PromptSpec(uid=_uid(), status="finished", sessions=2, rewards=[0.0, 1.0])
    _produce(partition_id, [all_same, mixed]).join_and_check()

    refiller = FakeRefiller(partition_id, global_steps=1, sessions=2, rewards=[0.0, 1.0])
    rb = _make_rb(refill_fn=refiller, filter_groups_metric="acc")
    try:
        batch, metrics = rb.sample(global_steps=1, partition_id=partition_id, batch_size=2)
        sampled_uids = _uids_of(batch.keys)
        refill_uids = set(refiller.produced_uids)

        assert all_same.uid not in sampled_uids
        assert mixed.uid in sampled_uids
        assert refiller.calls == [2]
        assert len(sampled_uids & refill_uids) == 1
        assert metrics["validation/filter_groups/evicted_samples"] == 1
        assert metrics["validation/filter_groups/discarded_surplus_samples"] == 1

        surplus_uid = (refill_uids - sampled_uids).pop()
        remaining = tq.kv_list(partition_id=partition_id).get(partition_id, {})
        assert surplus_uid not in remaining
        assert _trajectory_key(surplus_uid, 0) not in remaining
        assert _trajectory_key(surplus_uid, 1) not in remaining
    finally:
        _clear_partition(partition_id)


def test_sync_dapo_and_failure_refill_credits_are_combined(tq_init, partition_id):
    all_same = PromptSpec(uid=_uid(), status="finished", sessions=2, rewards=[0.0, 0.0])
    empty_failure = PromptSpec(uid=_uid(), status="failure", sessions=0)
    mixed = PromptSpec(uid=_uid(), status="finished", sessions=2, rewards=[0.0, 1.0])
    _produce(partition_id, [all_same, empty_failure, mixed]).join_and_check()

    refiller = FakeRefiller(partition_id, global_steps=1, sessions=2, rewards=[0.0, 1.0])
    rb = _make_rb(
        refill_fn=refiller,
        filter_groups_metric="acc",
        sync_refill_failed_groups=True,
        train_batch_size=3,
    )
    try:
        batch, metrics = rb.sample(global_steps=1, partition_id=partition_id, batch_size=3)
        sampled_uids = _uids_of(batch.keys)

        assert len(sampled_uids) == 3
        assert mixed.uid in sampled_uids
        assert all_same.uid not in sampled_uids
        assert empty_failure.uid not in sampled_uids
        assert refiller.calls == [3]
        assert metrics["validation/filter_groups/evicted_samples"] == 1
        assert metrics["validation/rollout_failure/evicted_samples"] == 1
        assert metrics["validation/filter_groups/discarded_surplus_samples"] == 1
    finally:
        _clear_partition(partition_id)


def test_sync_dapo_does_not_refill_when_filtered_surplus_leaves_a_full_batch(tq_init, partition_id):
    all_same = PromptSpec(uid=_uid(), status="finished", sessions=2, rewards=[0.0, 0.0])
    mixed = [PromptSpec(uid=_uid(), status="finished", sessions=2, rewards=[0.0, 1.0]) for _ in range(2)]
    _produce(partition_id, [all_same, *mixed]).join_and_check()

    refiller = FakeRefiller(partition_id, global_steps=1, sessions=2, rewards=[0.0, 1.0])
    rb = _make_rb(
        refill_fn=refiller,
        filter_groups_metric="acc",
        train_batch_size=3,
    )
    try:
        batch, metrics = rb.sample(global_steps=1, partition_id=partition_id, batch_size=2)

        assert _uids_of(batch.keys) == {spec.uid for spec in mixed}
        assert refiller.calls == []
        assert metrics["validation/filter_groups/evicted_samples"] == 1
    finally:
        _clear_partition(partition_id)


def test_sync_dapo_streams_refill_credit_with_bounded_inflight(tq_init, partition_id):
    all_same = PromptSpec(uid=_uid(), status="finished", sessions=2, rewards=[0.0, 0.0])
    slow_valid = PromptSpec(uid=_uid(), status="running", sessions=2, rewards=[0.0, 1.0])
    _produce(partition_id, [all_same, slow_valid]).join_and_check()

    refiller = FakeRefiller(partition_id, global_steps=1, sessions=2, rewards=[0.0, 1.0])
    # Sync DAPO is on-policy, so max_off_policy_strategy is irrelevant here; the default (drop) is a NO-OP.
    rb = _make_rb(
        refill_fn=refiller,
        filter_groups_metric="acc",
        train_batch_size=2,
        gen_batch_size=1,
        max_inflight_gen_batches=1,
    )
    consumer = SampleConsumer(rb, partition_id, batch_size=2, global_steps=1)
    try:
        consumer.start()
        deadline = time.time() + 5
        while len(refiller.calls) < 2 and consumer.is_alive() and time.time() < deadline:
            time.sleep(POLL_INTERVAL)

        # One original prompt remains in flight, so the 2x credit is dispatched one slot at a time.
        assert refiller.calls == [1, 1]
        assert consumer.is_alive()

        _set_prompt_status(partition_id, slow_valid.uid, "finished", global_steps=0)
        batch = consumer.result_or_raise()

        assert len(_uids_of(batch.keys)) == 2
        assert all_same.uid not in _uids_of(batch.keys)
        assert consumer.metrics is not None
        assert consumer.metrics["validation/filter_groups/evicted_samples"] == 1
        assert consumer.metrics["validation/filter_groups/discarded_surplus_samples"] == 1
    finally:
        _clear_partition(partition_id)


def test_sync_dapo_discards_unsent_credit_when_batch_fills(tq_init, partition_id):
    all_same = PromptSpec(uid=_uid(), status="finished", sessions=2, rewards=[0.0, 0.0])
    slow_valid = PromptSpec(uid=_uid(), status="running", sessions=2, rewards=[0.0, 1.0])
    _produce(partition_id, [all_same, slow_valid]).join_and_check()

    refiller = FakeRefiller(partition_id, global_steps=1, sessions=2, rewards=[0.0, 1.0])

    def refill_and_finish_original(num_prompts: int) -> int:
        result = refiller(num_prompts)
        _set_prompt_status(partition_id, slow_valid.uid, "finished", global_steps=0)
        return result

    rb = _make_rb(
        refill_fn=refill_and_finish_original,
        filter_groups_metric="acc",
        train_batch_size=2,
        gen_batch_size=1,
        max_inflight_gen_batches=1,
    )
    try:
        batch, metrics = rb.sample(global_steps=1, partition_id=partition_id, batch_size=2)

        assert len(_uids_of(batch.keys)) == 2
        assert all_same.uid not in _uids_of(batch.keys)
        assert refiller.calls == [1]
        assert metrics["validation/filter_groups/evicted_samples"] == 1
        assert "validation/filter_groups/discarded_surplus_samples" not in metrics
    finally:
        _clear_partition(partition_id)


def test_overlapping_async_eviction_reasons_refill_once(tq_init, partition_id):
    stale_dapo = PromptSpec(uid=_uid(), status="finished", sessions=2, global_steps=0, rewards=[1.0, 1.0])
    _produce(partition_id, [stale_dapo]).join_and_check()

    refiller = FakeRefiller(partition_id, global_steps=5, sessions=2, rewards=[0.0, 1.0])
    rb = _make_rb(
        trainer_mode="colocate_async",
        max_off_policy_strategy="drop",
        max_off_policy_threshold=1,
        refill_fn=refiller,
        filter_groups_metric="acc",
    )
    try:
        _, metrics = rb.sample(global_steps=5, partition_id=partition_id, batch_size=1)
        assert refiller.calls == [1]
        assert metrics["validation/off_policy/evicted_samples"] == 1
        assert metrics["validation/filter_groups/evicted_samples"] == 1
    finally:
        _clear_partition(partition_id)


def test_repeat_diagnostics_keep_valid_json_and_artifact_pointer():
    import json
    from verl.trainer.ppo.v1.replay_buffer import rollout_diagnostics
    verdict = {"verdict": "uncertain", "reason": "unsupported span", "votes": [{"reason": "x" * 600}] * 8}
    result = rollout_diagnostics({"repeat_reward_verdict_json": json.dumps(verdict),
                                  "repeat_reward_audit_path": "/shared/audit/abc.json.gz",
                                  "repeat_reward_audit_sha256": "a" * 64})
    assert json.loads(result["repeat_reward_verdict_json"])["verdict"] == "uncertain"
    assert "votes" not in json.loads(result["repeat_reward_verdict_json"])
    assert result["repeat_reward_audit_path"] == "/shared/audit/abc.json.gz"
    assert result["repeat_reward_audit_sha256"] == "a" * 64


@pytest.mark.parametrize("status", ["reward_pending", "running"])
def test_deferred_cutoff_counts_rollout_completion_but_waits_for_rewards(monkeypatch, status):
    """No Ray needed: drive the real sampler across generation/reward state changes."""
    monkeypatch.setenv("SWE_AGENT_ROLLOUT_TRAINING_COMPLETION_RATIO_THRESHOLD", "0.6666666666666666")
    state = {"train": {
        "ready": {"is_prompt": True, "status": "finished", "global_steps": 0},
        "ready_0_0": {"seq_len": 3},
        "judging": {"is_prompt": True, "status": status, "global_steps": 0, "generation_complete": True},
        "tail": {"is_prompt": True, "status": "running", "global_steps": 0},
    }}
    cancelled = []
    polls = []
    def cancel(uids, validate=False):
        cancelled.extend(uids)
        for uid in uids:
            state["train"][uid].update(status="failure", completion_ratio_cutoff=True)
        return len(uids)
    rb = _make_rb(train_batch_size=3, cancel_fn=cancel)
    def clear(*, keys, partition_id):
        for key in keys:
            state[partition_id].pop(key, None)
    monkeypatch.setattr(tq, "kv_clear", clear)
    monkeypatch.setattr(tq, "kv_list", lambda *args, **kwargs: state)
    def wait(partition_id, last_debug_time, sample_started=None):
        assert cancelled == ["tail"]
        polls.append(1)
        state["train"]["judging"]["status"] = "finished"
        state["train"]["judging_0_0"] = {"seq_len": 3}
        return last_debug_time
    monkeypatch.setattr(rb, "_wait_for_next_poll", wait)
    monkeypatch.setattr(rb, "_materialize_batch", lambda partition_id, selected, snapshot: selected)
    selected, metrics = rb.sample(global_steps=0, partition_id="train", batch_size=3)
    assert set(selected) == {"ready", "judging"}
    assert polls == [1] and cancelled == ["tail"]
    assert metrics["training/completion_ratio/terminal_groups"] == 2
    assert metrics["training/repeat_queue/cancellation_drain_seconds"] >= 0


def test_deferred_error_stops_sampling_instead_of_skipping_penalty(monkeypatch):
    rb = _make_rb()
    monkeypatch.setattr(tq, "kv_list", lambda *args, **kwargs: {"train": {
        "lost": {"is_prompt": True, "status": "reward_error", "global_steps": 0,
                 "repeat_error": "coordinator died"}}})
    with pytest.raises(RuntimeError, match="coordinator died"):
        rb.sample(global_steps=0, partition_id="train", batch_size=1)

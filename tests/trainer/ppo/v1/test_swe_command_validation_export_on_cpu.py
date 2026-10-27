"""Exercise command evidence through the real validation sampler and JSONL writer."""

import copy
import hashlib
import json

import numpy as np
import pytest

from recipe.swe_agent import command_evidence as ce
from tests.trainer.ppo.test_swe_command_evidence_on_cpu import card, dispatch
from tests.trainer.ppo.test_swe_edit_recovery_on_cpu import session
from tests.trainer.ppo.v1.test_swe_validation_on_cpu import validation_trainer
from verl.trainer.ppo.v1 import trainer_base as module


def dump_validation(monkeypatch, tmp_path, extras, *, fragmented=False):
    trainer, harness, _ = validation_trainer(monkeypatch, tmp_path, set(extras), {"0", "2"})
    trainer.config.trainer.val_only = True
    trainer.config.actor_rollout_ref.actor = dict(
        tool_command_diagnostics_enabled=True,
        tool_failure_penalty_enabled=False,
        tool_repeat_penalty_enabled=False,
        thinking_repeat_penalty_enabled=False,
    )
    # Deliberately sort across batches and within each batch at dump time.
    uids = iter(["z", "b", "y", "a"])
    monkeypatch.setattr(module.uuid, "uuid4", lambda: next(uids))
    original_dispatch = trainer.agent_loop_manager.generate_sequences

    def generate(batch):
        original_dispatch(batch)
        if fragmented:
            for uid in batch["uid"]:
                final = harness.state["val"][uid + "_0_0"]
                # The highest-index fragment is deliberately not the last row.
                for index in (2, 1):
                    harness.state["val"][f"{uid}_0_{index}"] = dict(final)

    trainer.agent_loop_manager.generate_sequences = generate
    original_get = module.tq.kv_batch_get
    published_extras = []

    def get(*, keys, select_fields, **kwargs):
        base_keys = [key.rsplit("_", 1)[0] + "_0" for key in keys]
        data = original_get(keys=base_keys, select_fields=select_fields, **kwargs)
        if "extra_fields" in select_fields:
            values = []
            for key, extra, reward in zip(keys, data["extra_fields"], data["reward_model"], strict=True):
                tid = reward["ground_truth"]["instance_id"]
                extra = copy.deepcopy(extra)
                if fragmented:
                    assert key.endswith("_2"), "Only the final cumulative evidence belongs to the session"
                extra.update(copy.deepcopy(extras[tid]))
                published_extras.append((extra, copy.deepcopy(extra)))
                values.append(extra)
            data["extra_fields"] = np.array(values, dtype=object)
        return data

    monkeypatch.setattr(module.tq, "kv_batch_get", get)
    original_metrics = trainer._val_metrics_update
    metric_inputs = []

    def metrics(sources, ids, infos, turns):
        assert "command_evidence_audit_json" not in infos
        metric_inputs.append(copy.deepcopy(infos))
        return original_metrics(sources, ids, infos, turns)

    trainer._val_metrics_update = metrics
    trainer._compute_advantage = lambda *a, **kw: pytest.fail("Validation must not apply local loss penalties")
    # Use the production asynchronous writer, including its completion barrier.
    del trainer._dump_generations
    trainer._init_dump_executor()
    try:
        result = trainer._validate()
    finally:
        trainer._shutdown_dump_executor()
    assert published_extras and all(before == after for before, after in published_extras)
    assert metric_inputs[0]["reward"] == [1, 0, 1, 0]
    assert result["val-core/bench/reward/mean@1"] == 0.5
    assert not any("command_evidence_audit_json" in key for key in result)
    rows = [json.loads(line) for line in (tmp_path / "12.jsonl").read_text().splitlines()]
    assert [r["uid"] for r in rows] == sorted(r["uid"] for r in rows)
    assert len(rows) == (12 if fragmented else 4)
    for row in rows:
        expected_reward = int(row["gts"]["instance_id"] in {"0", "2"})
        assert row["score"] == row["reward"] == expected_reward
        assert "tool_repeat_weights" not in row
    return rows


def command_audit(events):
    _, counts = ce.analyze(events)
    return {**ce.audit_json(events), **{"command_evidence_" + k: v for k, v in counts.items()}}


@pytest.mark.parametrize("fragmented", [False, True])
def test_validation_exports_real_command_evidence_without_changing_rewards(tmp_path, monkeypatch, fragmented):
    extras = {}
    ready = card("mwaskom__seaborn-3069", 211)["arguments"]["command"]
    for tid in ("0", "1", "2", "3"):
        workspace = tmp_path / ("workspace_" + tid)
        workspace.mkdir()
        executor, events = session(workspace, monkeypatch), []
        if tid == "2":
            # Expected assertion failures are a negative control from the audit.
            (workspace / "test_public.py").write_text(
                "import unittest\nclass Regression(unittest.TestCase):\n"
                " def test_behavior(self): self.assertEqual(1, 2)\n"
            )
        command = {
            "0": ready,
            "1": "python -c 'if True print(1)'",
            "2": "python -B -m unittest test_public",
            "3": ready,
        }[tid]
        for attempt in range(3):
            if tid == "3" and attempt == 2:
                # A real change makes this retry meaningful; it restarts protection.
                (workspace / "fix.py").write_text("value = 1\n")
            dispatch(executor, events, command)
        selected, _ = ce.analyze(events)
        assert selected == ({3: ["command_no_progress" if tid == "0" else "command_errors"]}
                            if tid in {"0", "1"} else {})
        extras[tid] = command_audit(events)

    rows = dump_validation(monkeypatch, tmp_path / "validation", extras, fragmented=fragmented)
    for row in rows:
        expected = extras[row["gts"]["instance_id"]]
        for key, value in expected.items():
            assert row[key] == value
        encoded = row["command_evidence_audit_json"]
        assert len(encoded) > 2048  # Ordinary diagnostic string truncation must not apply.
        assert hashlib.sha256(encoded.encode()).hexdigest() == row["command_evidence_audit_sha256"]
        assert len(json.loads(encoded)["events"]) == 3


def test_validation_keeps_missing_and_oversized_evidence_distinct(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    executor, events = session(workspace, monkeypatch), []
    dispatch(executor, events, 'python -c "print(123)"', collect=False)
    assert events[0]["command_evidence"] is None
    # Bound is measured in bytes, including non-ASCII content.
    oversized = json.dumps({"events": [], "padding": "é" * (1024 * 1024)}, ensure_ascii=False)
    assert len(oversized) < 2 * 1024 * 1024 < len(oversized.encode())
    extras = {
        "0": {},  # An older worker or collection disabled: no fabricated proof/counts.
        "1": {"command_evidence_audit_json": oversized},
        "2": ce.audit_json(events, max_bytes=1),  # Producer-side bound also survives export.
        "3": command_audit(events),  # Audit exists, but the executor supplied no command proof.
    }
    rows = dump_validation(monkeypatch, tmp_path / "validation", extras)
    by_task = {row["gts"]["instance_id"]: row for row in rows}
    for tid in ("0", "1", "2"):
        assert by_task[tid].get("command_evidence_audit_json") is None
    assert by_task["0"].get("command_evidence_audit_error") is None
    for tid in ("1", "2"):
        assert by_task[tid]["command_evidence_audit_error"] == "size_limit"
    saved = json.loads(by_task["3"]["command_evidence_audit_json"])
    assert saved["events"][0]["command_evidence"] is None
    assert by_task["3"]["command_evidence_barrier_progress_or_insufficient_evidence"] == 1
    assert not any(k.startswith("command_evidence_selected_") for k in by_task["3"])


def test_validation_without_command_diagnostics_preserves_legacy_dump(tmp_path, monkeypatch):
    rows = dump_validation(monkeypatch, tmp_path, {str(i): {} for i in range(4)})
    assert all(not any(key.startswith("command_evidence_") for key in row) for row in rows)

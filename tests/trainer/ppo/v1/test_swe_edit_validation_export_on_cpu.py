"""Edit signals through the real validation sampler and asynchronous JSONL dump."""

import copy
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from recipe.swe_agent.edit_recovery import audit_json

from tests.trainer.ppo.test_swe_edit_causes_on_cpu import align, config, slice_events
from tests.trainer.ppo.test_swe_edit_causes_on_cpu import tokenizer as tokenizer
from tests.trainer.ppo.v1.test_swe_command_validation_export_on_cpu import dump_validation
from verl.trainer.ppo.v1.trainer_base import PPOTrainer
from verl.utils.swe_edit_diagnostics import collect_edit_diagnostics


@pytest.mark.parametrize("fragmented", [False, True])
def test_validation_exports_edit_proofs_and_signals_without_reward_or_parameter_changes(
    tmp_path, monkeypatch, tokenizer, fragmented
):
    def forbidden(*a, **kw):
        pytest.fail("Validation must not compute policy advantages or update parameters")

    monkeypatch.setattr(PPOTrainer, "_compute_advantage", forbidden)
    monkeypatch.setattr(PPOTrainer, "_update_actor", forbidden)
    monkeypatch.setattr(torch.optim.Optimizer, "step", forbidden)
    extras = {}
    for tid in ("0", "1", "2", "3"):
        events = slice_events(3, 423, 90, 93) if tid in {"0", "1"} else slice_events(1, 60, 80, 92)
        output = align(events, tokenizer)
        extras[tid] = {**audit_json(events), **collect_edit_diagnostics(events, output, config(), validation=True)}
    original = copy.deepcopy(extras)
    rows = dump_validation(monkeypatch, tmp_path, extras, fragmented=fragmented)
    assert extras == original
    for row in rows:
        tid = row["gts"]["instance_id"]
        assert row["score"] == row["reward"] == int(tid in {"0", "2"})
        assert row["edit_diagnostics_would_cost"] == pytest.approx(0.02 if tid in {"0", "1"} else 0.0)
        assert row["edit_diagnostics_actual_optimization_cost"] == 0.0
        for name in ("edit_diagnostics_audit_json", "tool_repeat_recovery_audit_json"):
            assert row[name] == extras[tid][name]
            assert json.loads(row[name])["events"]
        assert (
            hashlib.sha256(row["edit_diagnostics_audit_json"].encode()).hexdigest()
            == row["edit_diagnostics_audit_sha256"]
        )
        assert not any(k in row for k in ("advantages", "tool_repeat_weights", "token_level_rewards"))


def test_validation_without_edit_collection_and_oversized_proofs_are_distinct(tmp_path, monkeypatch):
    extras = {
        "0": {},
        "1": {"edit_diagnostics_audit_json": "é" * (1024 * 1024 + 1)},
        "2": {"tool_repeat_recovery_audit_json": "x" * (512 * 1024 + 1)},
        "3": {"edit_diagnostics_failed": 1, "edit_diagnostics_error": "MissingProof"},
    }
    rows = dump_validation(monkeypatch, tmp_path, extras)
    by_task = {r["gts"]["instance_id"]: r for r in rows}
    assert by_task["0"].get("edit_diagnostics_audit_error") is None
    assert by_task["1"]["edit_diagnostics_audit_error"] == "size_limit"
    assert by_task["2"]["tool_repeat_recovery_audit_error"] == "size_limit"
    assert by_task["3"]["edit_diagnostics_failed"] == 1
    assert by_task["3"]["edit_diagnostics_error"] == "MissingProof"
    assert all(r.get("edit_diagnostics_audit_json") is None for r in rows)


def test_val_only_fit_returns_before_training_or_parameter_updates(monkeypatch):
    from verl.trainer.ppo.v1 import trainer_base as module

    calls = []
    parameter = torch.nn.Parameter(torch.tensor([3.0]))
    before = parameter.detach().clone()

    def forbidden(*args, **kwargs):
        pytest.fail("Validation-only fit must return before training")

    monkeypatch.setattr(module.SkipManager, "init", lambda config: None)
    monkeypatch.setattr(module, "Tracking", lambda **kwargs: SimpleNamespace(log=lambda **kw: calls.append("log")))
    monkeypatch.setattr(module, "ValidationGenerationsLogger", lambda **kwargs: object())
    monkeypatch.setattr(module, "DapoFilteredRewardTableLogger", lambda **kwargs: object())
    monkeypatch.setattr(torch.optim.Optimizer, "step", forbidden)
    trainer = SimpleNamespace(
        config=OmegaConf.create(
            dict(
                trainer=dict(
                    val_only=True, val_before_train=True, project_name="offline", experiment_name="edit", logger=[]
                )
            )
        ),
        replay_buffer=SimpleNamespace(),
        global_steps=12,
        on_validate_begin=lambda: calls.append("begin"),
        on_validate_end=lambda: calls.append("end"),
        _validate=lambda: {"val-core/bench/reward/mean@1": 0.5},
        _shutdown_dump_executor=lambda: calls.append("dump_flush"),
        _compute_advantage=forbidden,
        _update_actor=forbidden,
        _reissue_inflight_prompts=forbidden,
    )
    # No steps_per_epoch or training state: entering the training path fails.
    PPOTrainer.fit(trainer, SimpleNamespace(cancel_sequences=lambda *a, **kw: None))
    assert calls == ["begin", "end", "log", "dump_flush"]
    assert trainer.global_steps == 12
    torch.testing.assert_close(parameter.detach(), before)

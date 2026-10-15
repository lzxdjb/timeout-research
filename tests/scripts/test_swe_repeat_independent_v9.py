"""Stage independence, evidence safety and validation reward isolation."""
import asyncio
import copy
import gzip
import hashlib
import json
from pathlib import Path

import jsonschema
import pytest

from recipe.swe_agent import repetition_reward as policy
from scripts.swe_repeat_contracts import independent_stage2_contracts
from scripts.swe_repeat_regression import replay_events


def event(i, name="Edit", args=None, result=policy.NOOP_EDIT_RESULT):
    args = args or {"file_path": "a.py", "old_string": "x=1", "new_string": "x=1"}
    return {"id": i, "name": name, "arguments": args, "serial": True,
            **policy.observation_fields(result), **policy.capture_observation(result),
            **policy.noop_edit_evidence(name, args, result)}


def positive(ids=(1, 3)):
    return {"reason": "Events 1 and 3 repeat the same explicitly rejected edit without changing the file.",
            "verdict": "repetitive", "category": "ineffective_action_loop", "event_ids": list(ids)}


def test_evidence_review_is_independent_and_not_primed_by_vote_reasons():
    claims = [positive() for _ in range(8)]
    messages = policy.judge_messages("", [event(1), event(3)], review_claims=claims)
    assert messages[0]["content"] == policy.EVIDENCE_REVIEW_PROMPT
    body = json.loads(messages[1]["content"])
    assert body["evidence_review"] == [{"category": "ineffective_action_loop", "event_ids": [1, 3]}]
    assert claims[0]["reason"] not in messages[1]["content"]
    assert body["events"] == json.loads(policy.judge_messages("", [event(1), event(3)])[1]["content"])["events"]


def test_nonconsecutive_repetition_is_available_and_judged_without_stage1(monkeypatch):
    events = [event(1), event(2, "Read", {"file_path": "a.py"}, "x=1"), event(3)]
    assert policy.conservative_detect(events)["verdict"] == "uncertain"
    assert policy.candidate_spans(events) == []
    jsonschema.validate(positive(), policy.schema_for_events(events))
    client = policy.JudgeClient(policy.RepeatConfig(vote_count=8, vote_quorum=7))
    requests = []
    async def request(method, path, payload=None):
        if path == "/v1/models":
            return {"data": [{"id": client.config.model, "max_model_len": client.config.max_context}]}
        if path == "/tokenize":
            return {"count": 100}
        body = json.loads(payload["messages"][1]["content"])
        assert "candidate_spans" not in body
        assert len(body["events"]) == 3
        requests.append(payload)
        return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(positive())}}]}
    monkeypatch.setattr(client, "_request", request)
    verdict = asyncio.run(client.judge_consensus("", events))
    assert verdict["verdict"] == "repetitive" and verdict["evidence_review_passed"] == 1
    assert verdict["vote_count"] == 8 and len(requests) == 9
    assert "Evidence review:" in requests[-1]["messages"][0]["content"]


@pytest.mark.parametrize("validation_enabled", [False, True])
@pytest.mark.parametrize("mode", ["off", "shadow", "apply"])
@pytest.mark.parametrize("validation", [False, True])
@pytest.mark.parametrize("valid", [False, True])
@pytest.mark.parametrize("raw,existing", [(1, 1), (1, -.1), (0, 0)])
def test_validation_flag_never_changes_training_or_validation_rewards(validation_enabled, mode, validation, valid, raw, existing):
    config = policy.RepeatConfig(mode=mode, validation_enabled=validation_enabled)
    selected = policy.judgment_eligible(config, validation=validation, valid=valid, raw=raw, existing=existing)
    assert selected == (valid and (validation_enabled if validation else mode != "off" and raw == 1 and existing > 0))
    result = policy.compose_reward(config, validation=validation, valid=valid, raw=raw, existing=existing, verdict="repetitive")
    assert result == (existing * config.multiplier if mode == "apply" and not validation and valid and raw == 1 and existing > 0 else existing)
    if raw == 0:
        assert policy.hypothetical_reward(config, valid=valid, raw=raw, existing=existing, verdict="repetitive") == existing


@pytest.mark.parametrize("value,expected", [(None, False), ("0", False), ("1", True), ("True", True), ("off", False)])
def test_validation_config_default_and_boolean_parsing(monkeypatch, value, expected):
    monkeypatch.setenv("SWE_AGENT_REPEAT_REWARD_MODE", "off")
    if value is None:
        monkeypatch.delenv("SWE_AGENT_REPEAT_JUDGE_VALIDATION_ENABLED", raising=False)
    else:
        monkeypatch.setenv("SWE_AGENT_REPEAT_JUDGE_VALIDATION_ENABLED", value)
    assert policy.RepeatConfig.from_env().validation_enabled is expected


def test_validation_config_rejects_typo(monkeypatch):
    monkeypatch.setenv("SWE_AGENT_REPEAT_JUDGE_VALIDATION_ENABLED", "treu")
    with pytest.raises(ValueError, match="boolean"):
        policy.RepeatConfig.from_env()


@pytest.mark.parametrize("change", ["result", "args", "proof", "status", "gap"])
def test_stage1_rejects_stale_or_inconsistent_proof(change):
    events = [event(i, "Read", {"file_path": "a.py"}, "x=1") for i in range(1, 6)]
    for e in events:
        e["evidence"] = {"version": policy.EVIDENCE_VERSION, "complete": True, "before": "state", "after": "state",
                         "arguments_sha256": "server-normalized-args", "observation_sha256": hashlib.sha256(b"x=1").hexdigest()}
    assert policy.conservative_detect(events)["verdict"] == "repetitive"
    if change == "result":
        events[2]["result"] = "x=2"
    elif change == "args":
        events[2]["arguments"] = {"file_path": "b.py"}
    elif change == "proof":
        events[2]["evidence"]["observation_sha256"] = "stale"
    elif change == "status":
        events[2]["observation_status"] = "artifact_incomplete"
    else:
        events[2]["id"] = 999
    assert policy.conservative_detect(events)["verdict"] == "uncertain"


def test_stage1_noop_proof_requires_actual_rejection():
    events = [event(i) for i in range(1, 6)]
    events[2]["result"] = "Edited a.py (1 replacement)."
    assert policy.conservative_detect(events)["verdict"] == "uncertain"


@pytest.mark.parametrize("reason,expected", [
    ("This constitutes an unproductive loop of failed commands.", "uncertain"),
    ("The agent engaged in a sustained ineffective action loop.", "uncertain"),
    ("No unproductive loop was established.", "clean"),
    ("These attempts do not constitute an unproductive loop.", "clean"),
    ("The agent corrected the command and verified the repair.", "clean"),
])
def test_contradictory_clean_reasons_never_become_penalties(reason, expected):
    verdict = {"verdict": "clean", "category": "none", "event_ids": [], "reason": reason}
    assert policy.consistent_reason(verdict)["verdict"] == expected


@pytest.mark.parametrize("reason,expected", [
    ("The proposed loop is disproven by intervening recovery events.", "uncertain"),
    ("The cited repetition loop was not established. A separate loop occurs later.", "uncertain"),
    ("No unproductive loop is established by these events.", "uncertain"),
    ("The proposed loop is not supported by these observations.", "uncertain"),
    ("The proposed loop is not disproven by earlier useful work.", "repetitive"),
    ("Events 1-100 form a long uninterrupted loop despite earlier recovery.", "repetitive"),
])
def test_positive_reason_rejecting_its_own_loop_is_downgraded(reason, expected):
    verdict = {**positive(), "reason": reason}
    result = policy.consistent_reason(verdict)
    assert result["verdict"] == expected
    if expected == "uncertain":
        assert result["event_ids"] == [] and result["proposed_verdict"] == verdict


@pytest.mark.parametrize("confirmation", ["uncertain", "clean", "error", "different_loop"])
def test_positive_evidence_review_can_veto_unanimous_votes(monkeypatch, confirmation):
    client = policy.JudgeClient(policy.RepeatConfig(vote_count=8, vote_quorum=7))
    async def judge(transcript, events, *, seed=0, review_claims=None):
        if review_claims is None:
            return positive()
        assert len(review_claims) == 8
        if confirmation == "error":
            return {**policy.uncertain("judge_error"), "error": 1, "failure_stage": "completion"}
        if confirmation == "different_loop":
            return positive((20, 22))
        if confirmation == "clean":
            return {"verdict": "clean", "category": "none", "event_ids": [], "reason": "Relevant recovery"}
        return policy.uncertain("Cited events do not demonstrate the asserted loop")
    monkeypatch.setattr(client, "judge", judge)
    result = asyncio.run(client.judge_consensus("", []))
    assert result["verdict"] == "uncertain" and result["repetitive_votes"] == 8
    assert result["evidence_review_passed"] == 0


def test_votes_for_disjoint_loops_do_not_reach_a_common_quorum(monkeypatch):
    client = policy.JudgeClient(policy.RepeatConfig(vote_count=8, vote_quorum=7))
    ids = iter([(1, 3)] * 4 + [(20, 22)] * 4)
    async def judge(transcript, events, *, seed=0, review_claims=None):
        assert review_claims is None
        return positive(next(ids))
    monkeypatch.setattr(client, "judge", judge)
    result = asyncio.run(client.judge_consensus("", []))
    assert result["repetitive_votes"] == 8 and result["aligned_repetitive_votes"] == 4
    assert result["verdict"] == "uncertain" and result["evidence_review_called"] == 0


def test_new_contracts_cover_interleaving_and_protected_recovery():
    contracts = independent_stage2_contracts()
    assert len(contracts) == 16 and len({c["id"] for c in contracts}) == 16
    for contract in contracts:
        case = {"events": [{"index": i, "batch": i, "name": name, "arguments": args, "response": response}
                           for i, (name, args, response) in enumerate(contract["calls"])], "raw": {"input": "", "output": ""}}
        events, _, _ = replay_events(case, policy)
        if contract["expected"] != "repetitive":
            assert policy.conservative_detect(events)["verdict"] != "repetitive", contract["id"]
        if contract["id"].endswith(("interleaved_noops", "alternating_failed_commands", "edit_revert_cycles")):
            assert policy.conservative_detect(events)["verdict"] == "uncertain"
            assert any(b["properties"]["verdict"]["const"] == "repetitive" for b in policy.schema_for_events(events)["oneOf"])


def test_validation_flag_forwarded_to_existing_ray_workers(monkeypatch):
    from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
    monkeypatch.setenv("SWE_AGENT_REPEAT_REWARD_MODE", "off")
    monkeypatch.setenv("SWE_AGENT_REPEAT_JUDGE_VALIDATION_ENABLED", "1")
    assert get_ppo_ray_runtime_env()["env_vars"]["SWE_AGENT_REPEAT_JUDGE_VALIDATION_ENABLED"] == "1"


def test_positive_review_audit_is_reproducible(tmp_path, monkeypatch):
    monkeypatch.setenv("SWE_AGENT_REPEAT_AUDIT_DIR", str(tmp_path))
    events = [event(1), event(2), event(3)]
    claims = [positive()] * 7
    verdict = {**positive(), "votes": [{**positive(), "seed": 1}],
               "evidence_review": {**positive(), "seed": 123}, "evidence_review_claims": claims}
    config = policy.RepeatConfig()
    saved = policy.save_audit("", events, verdict, config)
    with gzip.open(saved["repeat_reward_audit_path"], "rt") as stream:
        audit = json.load(stream)
    assert audit["evidence_review_request"] == policy.completion_request(config, "", events, seed=123, review_claims=claims)
    assert "uniqueItems" not in json.dumps(audit["completion_request_template"]["structured_outputs"])

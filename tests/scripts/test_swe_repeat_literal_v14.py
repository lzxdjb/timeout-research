"""Literal rendering, raw evidence preservation and offline early-exit accounting."""
import asyncio
import copy
import gzip
import json
from pathlib import Path

import pytest

from recipe.swe_agent import repetition_reward as p
from recipe.swe_agent import repeat_legacy_render_v6 as old
from scripts.swe_repeat_regression import evaluate_two_level, file_digest, replay_evidence


def event(i=1):
    return {"id": i, "name": "Edit", "arguments": {"file_path": "a.py", "old_string": f"old{i}", "new_string": f"new{i}"},
            "completed": True, "serial": True, "observation_status": "observed",
            **p.capture_observation("Edited a.py (1 replacement).\nfeedback",
                                    {"repeat_public_result": "Edited a.py (1 replacement).", "repeat_public_feedback": "\nfeedback"}),
            "observation_view": {"text": "preview", "omitted": True},
            "evidence": {"before": "STATE_SECRET", "after": "STATE_SECRET", "complete": True}}


def test_literal_single_result_keeps_arguments_thinking_and_terminal_order():
    events = [event(1), event(2)]
    segments = [{"id": "a1", "after_event_id": 0, "text": "<think>First.</think>", "captured": True},
                {"id": "a2", "after_event_id": 2, "text": "Terminal.", "captured": True}]
    original = copy.deepcopy((events, segments))
    for claims in (None, [{"category": "ineffective_action_loop", "event_ids": [1, 2], "reason": "PRIVATE_VOTE"}]):
        messages = p.judge_messages("DUPLICATED_TRANSCRIPT", events, actor_segments=segments, review_claims=claims)
        body = json.loads(messages[1]["content"])
        assert [(x["kind"], x["id"]) for x in body["trajectory"]] == [("actor", "a1"), ("tool", 1), ("tool", 2), ("actor", "a2")]
        assert body["trajectory"][0]["text"] == "First."
        for e in body["trajectory"][1:3]:
            assert set(e) == {"kind", "id", "name", "arguments", "result", "completed", "serial", "observation_status"}
            assert e["result"] == "Edited a.py (1 replacement)."
        assert body["evidence_complete"] and body["tool_results_included"]
        assert all(value not in messages[1]["content"] for value in ("STATE_SECRET", "DUPLICATED_TRANSCRIPT", "feedback", "PRIVATE_VOTE"))
    assert (events, segments) == original


@pytest.mark.parametrize("source", ["result", "observation", "observation_view", "missing"])
def test_complete_available_result_once_and_missing_stays_missing(source):
    e = {"id": 1, "name": "Read", "arguments": {}, "completed": True, "observation_status": "observed"}
    if source == "observation_view":
        e[source] = {"text": "literal", "omitted": False}
    elif source != "missing":
        e[source] = "literal"
    body = json.loads(p.judge_messages("fallback context", [e])[1]["content"])
    assert body["trajectory"][0]["result"] == (None if source == "missing" else "literal")
    assert body["evidence_complete"] == (source != "missing")
    assert body["public_transcript"] == ""


def test_current_live_renderer_rejects_reference_fallback():
    with pytest.raises(ValueError, match="historical replay only"):
        p.judge_messages("", [event()], encoding=p.COMPACT_ENCODING_VERSION)


@pytest.mark.parametrize("view", ["full", "actor_calls"])
def test_view_config_audit_request_and_replay_agree(monkeypatch, tmp_path, view):
    monkeypatch.setenv("SWE_AGENT_REPEAT_JUDGE_INPUT_VIEW", view)
    monkeypatch.setenv("SWE_AGENT_REPEAT_AUDIT_DIR", str(tmp_path))
    config = p.RepeatConfig.from_env()
    e = event()
    saved = p.save_audit("original context", [e], p.uncertain("fixture"), config)
    with gzip.open(saved["repeat_reward_audit_path"], "rt") as stream:
        audit = json.load(stream)
    body = json.loads(audit["messages"][1]["content"])
    assert ("result" in body["trajectory"][0]) == (view == "full")
    assert body["evidence_complete"] == (view == "full")
    assert audit["events"][0]["feedback"] == "\nfeedback"
    assert audit["events"][0]["evidence"]["before"] == "STATE_SECRET"
    assert audit["original_public_context"] == "original context"
    assert audit["messages"] == p.JudgeClient(config).messages("original context", [e])
    assert p.JudgeClient(config).provenance()["input_view"] == view
    assert replay_evidence(saved, p)[0] == [e]


def test_bad_view_is_explicit_error(monkeypatch):
    monkeypatch.setenv("SWE_AGENT_REPEAT_JUDGE_INPUT_VIEW", "typo")
    with pytest.raises(ValueError, match="INPUT_VIEW"):
        p.RepeatConfig.from_env()


@pytest.mark.parametrize("encoding", [old.SERIALIZER_VERSION, old.COMPACT_ENCODING_VERSION])
def test_v6_requests_remain_verifiable_with_frozen_serializer(tmp_path, encoding):
    e = event()
    artifact = {"version": old.SERIALIZER_VERSION, "events": [e], "original_public_context": "context",
                "actor_segments": None, "verdict": {"encoding_version": encoding},
                "messages": old.judge_messages("context", [e], encoding=encoding)}
    path = tmp_path / "old.json.gz"
    with gzip.open(path, "wt") as stream:
        json.dump(artifact, stream)
    pointer = {"repeat_reward_audit_path": str(path), "repeat_reward_audit_sha256": file_digest(path)}
    assert replay_evidence(pointer, p)[2] == "captured_render_v6_migrated"


def test_offline_early_exit_reports_executed_and_skipped_votes(monkeypatch, tmp_path):
    case = {"id": "fixture", "task_key": "fixture", "partition": "test", "public_events": [event(1), event(2)],
            "actor_segments": [], "public_context": "", "raw": {"raw_score": 1}}
    fixtures, labels = tmp_path / "fixtures.jsonl.gz", tmp_path / "labels.json"
    with gzip.open(fixtures, "wt") as stream:
        stream.write(json.dumps(case) + "\n")
    labels.write_text(json.dumps({"policy_version": p.POLICY_VERSION,
                                 "cases": [{"id": "fixture", "expected": "uncertain", "reason": "No proven loop."}]}))
    monkeypatch.setenv("SWE_AGENT_REPEAT_JUDGE_VOTE_COUNT", "8")
    monkeypatch.setenv("SWE_AGENT_REPEAT_JUDGE_VOTE_QUORUM", "7")
    monkeypatch.setenv("SWE_AGENT_REPEAT_JUDGE_EARLY_EXIT_ENABLED", "1")
    monkeypatch.setenv("SWE_AGENT_REPEAT_AUDIT_DIR", str(tmp_path / "audit"))
    async def preflight(self):
        self.ready = True
    calls = []
    async def judge(self, *args, **kwargs):
        calls.append(kwargs)
        return {"verdict": "clean", "category": "none", "event_ids": [], "reason": "Useful distinct patches."}
    monkeypatch.setattr(p.JudgeClient, "preflight", preflight)
    monkeypatch.setattr(p.JudgeClient, "judge", judge)
    report = asyncio.run(evaluate_two_level(fixtures, labels, tmp_path / "report.json", online=True, audit_level2=True))
    assert len(calls) == 2 and report["judge_voting"]["skipped_votes"] == 6
    assert report["judge_voting"]["executed_votes"] == 2
    assert report["judge_voting"]["configured_votes"] == 8
    assert report["judge_voting"]["early_exit_cases"] == 1
    assert report["rows"][0]["verdict"] == "uncertain" and report["penalty_safety_pass"]


def test_new_matched_controls_preserve_stage1_safety():
    from scripts.swe_repeat_dataset_v14 import contracts
    for name, events, segments, expected in contracts():
        result = p.conservative_detect(events, segments)
        assert (result["verdict"] == "repetitive") == (expected == "repetitive"), name

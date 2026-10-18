"""Network-free penalty equivalence and accounting for optional negative early exit."""
import asyncio
import gzip
import itertools
import json
from dataclasses import replace

import pytest

from recipe.swe_agent import repetition_reward as repeat


async def consensus(config, sequence, review_verdict="repetitive"):
    client = repeat.JudgeClient(config)
    calls = []

    async def judge(*args, seed, review_claims=None, **kwargs):
        calls.append((seed, review_claims))
        verdict = review_verdict if review_claims is not None else sequence[len(calls) - 1]
        return {"verdict": verdict, "category": "none", "event_ids": [], "reason": "fixture"}

    client.judge = judge
    return await client.judge_consensus("public task", []), calls


@pytest.mark.parametrize("count,quorum,labels", [
    (8, 5, ("clean", "repetitive")),
    (8, 6, ("clean", "repetitive")),
    (8, 7, ("clean", "repetitive")),
    (4, 3, ("clean", "uncertain", "repetitive")),
    (1, 1, ("clean", "uncertain", "repetitive")),
])
@pytest.mark.parametrize("reviewer_enabled", [False, True])
def test_early_exit_preserves_penalties_and_positive_review_inputs(count, quorum, labels, reviewer_enabled):
    async def run():
        config = repeat.RepeatConfig(vote_count=count, vote_quorum=quorum, reviewer_enabled=reviewer_enabled)
        for sequence in itertools.product(labels, repeat=count):
            full, full_calls = await consensus(config, sequence)
            early, early_calls = await consensus(replace(config, early_exit_enabled=True), sequence)
            assert (early["verdict"] == "repetitive") == (full["verdict"] == "repetitive")
            assert early["configured_vote_count"] == count
            assert early["vote_count"] + early["votes_skipped"] == count
            assert early["vote_count"] == sum(early[k] for k in ("repetitive_votes", "clean_votes", "uncertain_votes"))
            assert early_calls == full_calls[:len(early_calls)]
            if full["verdict"] == "repetitive":
                assert early_calls == full_calls
                assert early["votes"] == full["votes"]
                assert not early["early_exit_taken"]
            if early["early_exit_taken"]:
                assert early["verdict"] == "uncertain" and early["reason"] == "positive_quorum_impossible"
                assert early["repetitive_votes"] + early["votes_skipped"] < quorum
                assert not early["evidence_review_called"]
    asyncio.run(run())


def test_default_disabled_and_runtime_env_forwarding(monkeypatch):
    from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
    monkeypatch.delenv("SWE_AGENT_REPEAT_JUDGE_EARLY_EXIT_ENABLED", raising=False)
    assert not repeat.RepeatConfig.from_env().early_exit_enabled
    monkeypatch.setenv("SWE_AGENT_REPEAT_JUDGE_EARLY_EXIT_ENABLED", "1")
    assert repeat.RepeatConfig.from_env().early_exit_enabled
    assert get_ppo_ray_runtime_env()["env_vars"]["SWE_AGENT_REPEAT_JUDGE_EARLY_EXIT_ENABLED"] == "1"
    monkeypatch.setenv("SWE_AGENT_REPEAT_JUDGE_EARLY_EXIT_ENABLED", "maybe")
    with pytest.raises(ValueError, match="boolean"):
        repeat.RepeatConfig.from_env()


@pytest.mark.parametrize("review_verdict", ["clean", "uncertain", "repetitive"])
def test_potential_positive_never_shortens_review(review_verdict):
    async def run():
        config = repeat.RepeatConfig(vote_count=8, vote_quorum=7)
        sequence = ("repetitive",) * 7 + ("clean",)
        full, full_calls = await consensus(config, sequence, review_verdict)
        early, early_calls = await consensus(replace(config, early_exit_enabled=True), sequence, review_verdict)
        assert early["verdict"] == full["verdict"]
        assert early_calls == full_calls and len(early_calls) == 9
        assert early["evidence_review_called"] == 1 and not early["early_exit_taken"]
    asyncio.run(run())


def test_early_exit_retains_errors_guards_and_actual_audit_seeds(monkeypatch, tmp_path):
    async def run():
        config = repeat.RepeatConfig(mode="apply", reward_scope="all_valid", vote_count=8, vote_quorum=7,
                                     early_exit_enabled=True)
        client = repeat.JudgeClient(config)
        votes = iter([
            {**repeat.uncertain("judge_error:TimeoutError"), "error": 1, "failure_stage": "completion",
             "attempt_errors": 1, "service_seconds": 2},
            {**repeat.uncertain("unsupported evidence"), "guard_blocked": 1, "service_seconds": 3},
        ])
        async def judge(*args, **kwargs):
            return next(votes)
        client.judge = judge
        verdict = await client.judge_consensus("public task", [])
        assert verdict["vote_count"] == 2 and verdict["votes_skipped"] == 6
        assert verdict["uncertain_votes"] == 2 and verdict["error_count"] == 1
        assert verdict["guard_blocked"] == 1 and verdict["service_seconds"] == 5
        assert verdict["failure_stage"] == "completion" and verdict["attempt_errors"] == 1
        for base in (1., 0., -.1):
            job = dict(config=config, metadata=dict(repeat_reward_outcome_bucket="submitted_success",
                       repeat_reward_judge_called=1, repeat_reward_stage=2), validation=False, valid=True,
                       raw_score=base, existing_score=base, transcript="public task", events=[], actor_segments=[])
            score, meta = await repeat.finish_repeat_reward(job, verdict)
            assert score == base and not meta["repeat_reward_applied"]
            assert meta["repeat_reward_judge_vote_count"] == 2
            assert meta["repeat_reward_judge_configured_vote_count"] == 8
            assert meta["repeat_reward_judge_votes_skipped"] == 6
            with gzip.open(meta["repeat_reward_audit_path"], "rt") as stream:
                audit = json.load(stream)
            assert audit["config"]["early_exit_enabled"]
            assert audit["vote_request_seeds"] == [vote["seed"] for vote in verdict["votes"]]
            assert len(audit["verdict"]["votes"]) == 2
            assert json.loads(meta["repeat_reward_verdict_json"])["votes_skipped"] == 6
    monkeypatch.setenv("SWE_AGENT_REPEAT_AUDIT_DIR", str(tmp_path))
    asyncio.run(run())


def test_effective_config_manifest_records_actual_server_without_secrets(monkeypatch, tmp_path, caplog):
    config = repeat.RepeatConfig(url="http://user:secret@host:18090/path?token=hidden", deferred_enabled=True,
                                 concurrency=32, max_pending=1024, early_exit_enabled=True)
    client = repeat.JudgeClient(config)
    async def request(*args):
        return {"data": [{"id": config.model, "root": "/actual/model", "max_model_len": 253952}]}
    client._request = request
    monkeypatch.setenv("SWE_AGENT_REPEAT_AUDIT_DIR", str(tmp_path))
    asyncio.run(client.preflight())
    manifests = list(tmp_path.glob("runtime-config-*.json"))
    assert len(manifests) == 1
    body = json.loads(manifests[0].read_text())
    assert body["config_sha256"] == repeat.digest(body["provenance"])
    provenance = body["provenance"]
    assert provenance["endpoint"] == "http://host:18090/path"
    assert provenance["served_root"] == "/actual/model" and provenance["server_max_context"] == 253952
    assert provenance["early_exit_enabled"] and provenance["concurrency"] == 32 and provenance["max_pending"] == 1024
    assert not any(secret in manifests[0].read_text() + caplog.text for secret in ("user:", "secret", "hidden"))
    assert "effective runtime config" in caplog.text


def test_manifest_io_failure_does_not_break_preflight(monkeypatch, tmp_path, caplog):
    client = repeat.JudgeClient(repeat.RepeatConfig())
    async def request(*args):
        return {"data": [{"id": client.config.model, "max_model_len": 253952}]}
    def fail_replace(*args):
        raise OSError("write failure")
    client._request = request
    monkeypatch.setenv("SWE_AGENT_REPEAT_AUDIT_DIR", str(tmp_path))
    monkeypatch.setattr(repeat.os, "replace", fail_replace)
    asyncio.run(client.preflight())
    assert client.ready and not list(tmp_path.iterdir())
    assert "runtime config persistence failed: OSError" in caplog.text

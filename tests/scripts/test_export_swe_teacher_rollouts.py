import argparse
import fcntl
import json

import pytest

from scripts import export_swe_teacher_rollouts as adapter
from scripts import make_swe_sft_data as builder
from scripts import swe_sft_quality as quality


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return "system\n" + messages[0]["content"] + "\nuser\n" + messages[1]["content"] + "\nassistant\n"

    def decode(self, value, **kwargs):
        return value

    def encode(self, value, **kwargs):
        return list(value.encode())


def trajectory():
    return {
        "task_id": "owner__repo-1",
        "attempt": 1,
        "raw_score": 1,
        "hidden_pass": True,
        "evaluation_valid": True,
        "submission_signal_seen": True,
        "shaped_score": -0.1,
        "source_extra_info": {"benchmark": "swe_rebench_v2", "repo": "owner/repo", "base_sha": "abc"},
        "extra_fields": {"reward_extra_info": {"final_patch_empty": False}},
        "messages": [
            {"role": "system", "content": "Use repository-relative paths as the only model-facing file-tool format."},
            {"role": "user", "content": "Fix a.py."},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "Change the faulty branch.",
                "tool_calls": [
                    {
                        "id": "call1",
                        "type": "function",
                        "function": {
                            "name": "Write",
                            "arguments": json.dumps({"file_path": "a.py", "content": "fixed"}),
                        },
                    }
                ],
            },
            {"role": "tool", "name": "Write", "tool_call_id": "call1", "content": "Wrote a.py."},
            {
                "role": "assistant",
                "content": "Implemented the repair.",
                "reasoning_content": "The observed change addresses the bug.",
            },
        ],
    }


def test_adapter_connects_actual_teacher_evidence_to_existing_quality_builder(tmp_path):
    record = tmp_path / "record.json"
    record.write_text("{}")
    row = adapter.rollout_row(trajectory(), Tokenizer(), record, {"model": "397B"})
    candidate = builder._candidate(
        row,
        run_name="teacher",
        source_file=str(record),
        source_line=1,
        require_path_policy=True,
        max_response_tokens=10000,
        max_tool_dispatches=120,
        max_penalized_repeated_tool_calls=0,
        repeated_tool_detection_mode="mutation_aware_v2",
        repeated_tool_read_window=8,
        repeated_tool_exec_window=4,
        require_swe_mutation=True,
        quality_mode=True,
    )
    assert candidate.raw_score == 1 and candidate.shaped_score == -0.1
    assert candidate.task_id == "owner__repo-1" and candidate.mutation_evidence
    events, segments, _ = quality.load_evidence(row)
    assert events[0]["name"] == "Write" and events[0]["observation"].strip() == "Wrote a.py."
    assert any("faulty branch" in segment["text"] for segment in segments)
    assert "secret_hidden_test" not in row["input"] + row["output"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("evaluation_valid", None),
        ("evaluation_valid", False),
        ("hidden_pass", None),
        ("hidden_pass", False),
        ("raw_score", 0),
        ("infrastructure_failure", 1),
        ("completion_ratio_cutoff", True),
        ("observed_infrastructure_failure", True),
        ("synthesis_context_pruned", True),
        ("synthesis_protocol_retries", 1),
    ],
)
def test_adapter_never_admits_unknown_or_invalid_outcomes(field, value):
    value_row = trajectory()
    value_row["extra_fields"][field] = value
    if field in {
        "evaluation_valid",
        "hidden_pass",
        "raw_score",
        "infrastructure_failure",
        "observed_infrastructure_failure",
    }:
        value_row[field] = value
    assert adapter.admission_reason(value_row) is not None


def test_adapter_checks_observation_identity_and_completeness():
    value = trajectory()
    value["messages"][3]["tool_call_id"] = "wrong"
    with pytest.raises(ValueError, match="ID"):
        adapter.render_transcript(value, Tokenizer())
    value = trajectory()
    value["messages"].pop(3)
    with pytest.raises(ValueError, match="Missing observation"):
        adapter.render_transcript(value, Tokenizer())
    value = trajectory()
    value["messages"].pop()
    with pytest.raises(ValueError, match="final assistant"):
        adapter.render_transcript(value, Tokenizer())


def test_adapter_leaves_hidden_material_out_of_model_messages(tmp_path):
    value = trajectory()
    value["reward_model"] = {
        "ground_truth": json.dumps(
            {
                "benchmark": "swe_rebench_v2",
                "instance_id": value["task_id"],
                "hidden_tests": "secret_hidden_test",
            }
        )
    }
    value["extra_fields"]["swe_hidden_reward"] = {"metadata": {"test_output": "secret_hidden_test"}}
    record = tmp_path / "record.json"
    record.write_text("{}")
    row = adapter.rollout_row(value, Tokenizer(), record, {})
    assert "secret_hidden_test" not in json.dumps(row)


def test_adapter_preserves_public_base_commit_without_ground_truth(tmp_path):
    value = trajectory()
    value["source_extra_info"].pop("base_sha")
    value["source_extra_info"]["base_commit"] = "public-base-commit"
    record = tmp_path / "record.json"
    record.write_text("{}")
    row = adapter.rollout_row(value, Tokenizer(), record, {})
    assert row["gts"]["base_commit"] == "public-base-commit"


@pytest.mark.parametrize("attempt_url", [None, "http://original:18090/v1", "http://replacement:18090/v1"])
def test_adapter_preserves_attempt_endpoint_after_address_change(tmp_path, attempt_url):
    value = trajectory()
    contract = {"model": "original-model", "api_url": "http://original:18090/v1"}
    if attempt_url is not None:
        value.update(teacher_model="attempt-model", teacher_api_url=attempt_url)
    record = tmp_path / "record.json"
    record.write_text(json.dumps(value))
    row = adapter.rollout_row(value, Tokenizer(), record, contract)
    converted = adapter.sft_row(row)
    expected_url = attempt_url or contract["api_url"]
    expected_model = value.get("teacher_model") or contract["model"]
    assert row["teacher_api_url"] == converted["teacher_api_url"] == expected_url
    assert row["teacher_model"] == converted["teacher_model"] == expected_model


def test_finished_snapshot_while_collecting_is_explicit_and_ignores_incomplete_attempts(tmp_path, monkeypatch):
    import pandas as pd
    from transformers import AutoTokenizer

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args, **kwargs: Tokenizer())
    source = tmp_path / "synthesis"
    source.mkdir()
    (source / "run_contract.json").write_text(json.dumps({"model": "397B"}))
    complete = source / "attempts" / "owner__repo-1" / "0001"
    complete.mkdir(parents=True)
    (complete / "record.json").write_text(json.dumps({"kind": "solve", "trajectory": trajectory()}))
    incomplete = complete.with_name("0002")
    incomplete.mkdir()
    (incomplete / "trajectory.json").write_text(json.dumps(trajectory()))
    invalid = complete.with_name("0003")
    invalid.mkdir()
    pruned = trajectory()
    pruned["extra_fields"]["synthesis_context_pruned"] = True
    (invalid / "record.json").write_text(json.dumps({"kind": "solve", "trajectory": pruned}))
    args = argparse.Namespace(
        synthesis_dir=source,
        output_dir=tmp_path / "finished",
        tokenizer_path=tmp_path / "tokenizer",
        allow_running=False,
        write_sft_parquet=True,
    )
    with (source / ".lock").open("w") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="--allow-running"):
            adapter.export(args)
        args.allow_running = True
        report = adapter.export(args)
    assert report["collection_running_at_snapshot_start"] is True
    assert report["counts"]["completed_attempts"] == 2
    assert report["counts"]["exported_trajectories"] == 1
    assert report["counts"]["rejected:synthesis_context_pruned"] == 1
    frame = pd.read_parquet(args.output_dir / "successful.parquet")
    assert len(frame) == 1 and frame.iloc[0]["score"] == 1
    messages = list(frame.iloc[0]["messages"])
    assert [message["role"] for message in messages] == ["system", "user", "assistant", "user", "assistant"]
    assert "<function=Write>" in messages[2]["content"]
    assert "<tool_response>" in messages[3]["content"]
    assert "tools" not in frame.columns
    assert frame.iloc[0]["quality_status"] == "hidden_success_before_quality_filter"
    with pytest.raises(ValueError, match="immutable"):
        adapter.export(args)


def test_export_preserves_mixed_runtime_sessions_without_misattributing_legacy_code(tmp_path, monkeypatch):
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args, **kwargs: Tokenizer())
    source = tmp_path / "synthesis"
    source.mkdir()
    contract = {
        "model": "397B",
        "api_url": "http://original:18090/v1",
        "loop_hashes": {"agent_loop": "initial-contract-hash"},
    }
    (source / "run_contract.json").write_text(json.dumps(contract))
    expected = {}
    for index in range(4):
        value = trajectory()
        value["task_id"] = f"owner__repo-{index}"
        if index in {1, 2}:
            session_id = f"session-{index}"
            provenance = {
                "session_id": session_id,
                "concurrency": index * 8,
                "runtime_revisions": {"verl": f"revision-{index}"},
                "loop_hashes": {"agent_loop": f"attempt-hash-{index}"},
                "resolved_actor_config": {"diagnostics": True},
                "harness_environment": {},
            }
            value.update(
                runtime_session_id=session_id,
                runtime_provenance=provenance,
                teacher_api_url=f"http://teacher-{index}:18090/v1",
            )
        expected[value["task_id"]] = value
        attempt_dir = source / "attempts" / value["task_id"] / "0001"
        attempt_dir.mkdir(parents=True)
        (attempt_dir / "record.json").write_text(json.dumps({"kind": "solve", "trajectory": value}))
    # The latest runtime session must not be used to fill older attempt fields.
    (source / "runtime_sessions.jsonl").write_text(json.dumps({"session_id": "latest-unrelated-session"}) + "\n")
    args = argparse.Namespace(
        synthesis_dir=source,
        output_dir=tmp_path / "export",
        tokenizer_path=tmp_path / "tokenizer",
        write_sft_parquet=True,
    )
    report = adapter.export(args)
    assert report["counts"]["exported_trajectories"] == 4
    rows = [json.loads(line) for line in (args.output_dir / "rollouts" / "1.jsonl").read_text().splitlines()]
    sft_rows = pq.read_table(args.output_dir / "successful.parquet").to_pylist()
    by_task = {row["task_id"]: row for row in sft_rows}
    for row in rows:
        attempt = expected[row["task_id"]]
        sft = by_task[row["task_id"]]
        assert row["runtime_session_id"] == sft["runtime_session_id"] == attempt.get("runtime_session_id")
        assert row["runtime_provenance"] == attempt.get("runtime_provenance")
        if attempt.get("runtime_provenance") is None:
            assert sft["runtime_provenance"] is None
        else:
            assert isinstance(sft["runtime_provenance"], str)
            assert json.loads(sft["runtime_provenance"]) == attempt["runtime_provenance"]
        assert row["teacher_api_url"] == sft["teacher_api_url"] == attempt.get("teacher_api_url", contract["api_url"])
        assert "initial-contract-hash" not in json.dumps(row)
        assert "latest-unrelated-session" not in json.dumps(row)

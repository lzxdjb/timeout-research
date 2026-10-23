import json

from verl.utils.swe_cutoff_debug import (
    DiagnosticRepeatTracker,
    MAX_EVENT_BYTES,
    reconcile_cutoff_sessions,
    summarize_agent_loop_state,
    summarize_repeat_events,
    write_cutoff_debug_event,
)


def test_reconcile_cutoff_sessions_counts_unstarted_and_unpublished():
    result = reconcile_cutoff_sessions(
        8,
        started={0, 1, 2, 3, 4, 5},
        returned={0, 1},
        cancelled={2, 3, 4},
        failed={5},
        published={0},
    )
    assert result["expected_sessions"] == 8
    assert result["not_started_sessions"] == 2
    assert result["cancelled_sessions"] == 3
    assert result["failed_sessions"] == 1
    assert result["returned_without_publication_sessions"] == 1
    assert result["not_started_session_ids"] == [6, 7]


def test_summarize_agent_loop_state_emits_counts_without_transcript_data():
    class State:
        response_mask = [1, 0, 1, 0]
        assistant_turns = 2
        user_turns = 2
        tool_calls = [object()]
        extra_fields = {
            "trajectory_tool_dispatches": 7,
            "trajectory_repeated_tool_calls": 3,
            "thinking_repeat_events": [{"thought": "private transcript"}],
            "trajectory_unretained_tool_responses_json": "private observation",
        }

    result = summarize_agent_loop_state(State())
    assert result["partial_response_tokens"] == 4
    assert result["partial_assistant_tokens"] == 2
    assert result["partial_observation_tokens"] == 2
    assert result["partial_parsed_tool_calls"] == 1
    assert result["trajectory_repeated_tool_calls"] == 3
    assert "thinking_repeat_events" not in result
    assert "trajectory_unretained_tool_responses_json" not in result
    assert "private" not in json.dumps(result)


def test_summarize_repeat_events_hashes_calls_and_thoughts_only():
    events = [
        {
            "name": "Bash",
            "arguments": {"command": "pytest tests/test_x.py"},
            "repeat_python_test_evidence": {"kind": "pytest"},
            "thinking_candidate": {"thought_sha256": "a" * 64},
        },
        {
            "name": "Bash",
            "arguments": {"command": "pytest tests/test_x.py"},
            "repeat_python_test_evidence": {"kind": "pytest"},
            "thinking_candidate": {"thought_sha256": "a" * 64},
        },
        {
            "name": "Bash",
            "arguments": {"command": "git status --short"},
        },
    ]

    result = summarize_repeat_events(events)
    assert result["partial_tool_call_events"] == 3
    assert result["partial_tool_signature_repeat_surplus"] == 1
    assert result["partial_tool_max_consecutive_signature_copies"] == 2
    assert result["partial_python_test_evidence_calls"] == 2
    assert result["partial_python_test_signature_repeat_surplus"] == 1
    assert result["partial_thinking_candidate_events"] == 2
    assert result["partial_thinking_hash_repeat_surplus"] == 1
    assert "pytest" not in json.dumps(result)
    assert "command" not in json.dumps(result)


def test_cutoff_event_writer_is_opt_in_and_writes_compact_jsonl(tmp_path, monkeypatch):
    monkeypatch.setenv("SWE_AGENT_CUTOFF_DEBUG", "0")
    monkeypatch.setenv("SWE_AGENT_CUTOFF_DEBUG_DIR", str(tmp_path))
    assert not write_cutoff_debug_event("session_cancelled", partition="train", session_id=4)
    assert list(tmp_path.iterdir()) == []

    monkeypatch.setenv("SWE_AGENT_CUTOFF_DEBUG", "1")
    assert write_cutoff_debug_event(
        "session_cancelled", partition="train", session_id=4, partial_response_tokens=100
    )
    paths = list(tmp_path.glob("cutoff_train_*.jsonl"))
    assert len(paths) == 1
    rows = [json.loads(line) for line in paths[0].read_text().splitlines()]
    assert rows[0]["event"] == "session_cancelled"
    assert rows[0]["session_id"] == 4
    assert rows[0]["partial_response_tokens"] == 100
    assert rows[0]["schema_version"] == 1


def test_independent_tracker_captures_short_thoughts_without_native_candidates():
    class Tokenizer:
        def decode(self, ids, **_kwargs):
            return "<think>Try   the test again.</think>" if ids[0] == 1 else "Try the test again.</think>"

    tracker = DiagnosticRepeatTracker()
    tracker.observe_generation(Tokenizer(), [1] * 64)
    tracker.observe_generation(Tokenizer(), [1] * 64)
    tracker.observe_tool("Bash", '{"command": "pytest", "timeout": 1}')
    tracker.observe_tool("Bash", {"timeout": 1, "command": "pytest"})
    result = tracker.summary()
    assert result["diagnostic_thought_repeat_surplus"] == 1
    assert result["diagnostic_tool_max_consecutive_copies"] == 2
    assert result["diagnostic_assistant_block64_repeat_surplus"] == 1
    assert "pytest" not in json.dumps(result)
    assert "Try" not in json.dumps(result)


def test_event_writer_bounds_size_and_handles_short_writes(tmp_path, monkeypatch):
    from verl.utils import swe_cutoff_debug as debug
    monkeypatch.setenv("SWE_AGENT_CUTOFF_DEBUG", "1")
    monkeypatch.setenv("SWE_AGENT_CUTOFF_DEBUG_DIR", str(tmp_path))
    original_write = debug.os.write
    monkeypatch.setattr(debug.os, "write", lambda fd, data: original_write(fd, data[:17]))
    assert write_cutoff_debug_event("large", partition="train", uid="case",
                                    **{f"counter{i}": "x" * 5000 for i in range(190)})
    data = next(tmp_path.glob("*.jsonl")).read_bytes()
    assert len(data) <= MAX_EVENT_BYTES
    row = json.loads(data)
    assert row["payload_truncated"] is True
    assert row["uid"] == "case"


def test_event_writer_fails_open_on_bad_serialization_and_filesystem(tmp_path, monkeypatch):
    monkeypatch.setenv("SWE_AGENT_CUTOFF_DEBUG", "1")
    path = tmp_path / "not_a_directory"
    path.write_text("x")
    monkeypatch.setenv("SWE_AGENT_CUTOFF_DEBUG_DIR", str(path))
    assert not write_cutoff_debug_event("cannot_write", partition="train")

    class BrokenScalar:
        def item(self):
            raise RuntimeError("bad scalar")

    monkeypatch.setenv("SWE_AGENT_CUTOFF_DEBUG_DIR", str(tmp_path))
    assert not write_cutoff_debug_event("bad_scalar", global_step=BrokenScalar())


def test_debug_environment_is_forwarded_and_reset(monkeypatch):
    from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
    monkeypatch.setenv("SWE_AGENT_CUTOFF_DEBUG", "1")
    monkeypatch.setenv("SWE_AGENT_CUTOFF_DEBUG_DIR", "/shared/debug")
    monkeypatch.setenv("SWE_AGENT_ROLLOUT_TRAINING_COMPLETION_RATIO_THRESHOLD", "0.96")
    env = get_ppo_ray_runtime_env()["env_vars"]
    assert env["SWE_AGENT_CUTOFF_DEBUG"] == "1"
    assert env["SWE_AGENT_CUTOFF_DEBUG_DIR"] == "/shared/debug"
    assert env["SWE_AGENT_ROLLOUT_TRAINING_COMPLETION_RATIO_THRESHOLD"] == "0.96"
    monkeypatch.delenv("SWE_AGENT_CUTOFF_DEBUG")
    monkeypatch.delenv("SWE_AGENT_CUTOFF_DEBUG_DIR")
    env = get_ppo_ray_runtime_env()["env_vars"]
    assert env["SWE_AGENT_CUTOFF_DEBUG"] == "0"
    assert env["SWE_AGENT_CUTOFF_DEBUG_DIR"] == ""

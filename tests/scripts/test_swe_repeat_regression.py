import copy
import gzip
import json

import pytest

from scripts import swe_repeat_regression as regression
from scripts.swe_repeat_contracts import contracts, event, filesystem_evidence


def indexed(events):
    return [{"index": i, "batch": i, **e} for i, e in enumerate(events)]


def call(name="Read", **arguments):
    return "<tool_call><function=" + name + ">" + "".join(
        f"<parameter={k}>{v}</parameter>" for k, v in arguments.items()
    ) + "</function></tool_call>"


def test_parser_preserves_role_words_and_calls_inside_observation():
    output = call(path="a.py") + "\nuser\n<tool_response>\nassistant\n" + call(path="quoted.py") + "\nuser\n</tool_response>\nassistant\nDone."
    events, messages, issues = regression.parse_transcript(output)
    assert len(events) == 1
    assert len(messages) == 3
    assert "quoted.py" in events[0]["response"]
    assert issues == []


def test_parser_never_counts_undispatched_prefix_tail():
    output = call(path="a.py") + call(path="b.py") + "\nuser\n<tool_response>ok</tool_response>\nassistant\nDone."
    events, _, issues = regression.parse_transcript(output)
    assert [e["arguments"]["path"] for e in events] == ["a.py"]
    assert issues == ["undispatched_or_unobserved_calls"]


def test_terminal_unanswered_call_is_not_invented_execution():
    events, _, issues = regression.parse_transcript(call(path="a.py"))
    assert events == []
    assert issues == ["undispatched_or_unobserved_calls"]


@pytest.mark.parametrize("output,reason", [
    ("", "empty_output"),
    (call(path="a") + "\nuser\n<tool_response>unterminated", "unbalanced_tool_response"),
    ("start\nassistant\nagain", "nonalternating_roles"),
    (call(path="a") + "\nuser\n<tool_response>1</tool_response><tool_response>2</tool_response>", "more_responses_than_calls"),
])
def test_ambiguous_protocol_fails_closed(output, reason):
    events, _, issues = regression.parse_transcript(output)
    assert events == []
    assert reason in issues


@pytest.mark.parametrize("command", [
    "pytest -q", "python -m pytest tests/test_a.py -v", "go test ./...",
    "npm test 2>&1 | head -100", "cargo test", "python3 -m pytest -q 2>&1",
])
def test_evidence_runner_recognition(command):
    assert regression.test_command(command)


@pytest.mark.parametrize("command", [
    "echo pytest", "python -c 'print(\"pytest\")'", "pytest; touch x",
    "pytest && rm a.py", "pytest > src/a.py", "echo 'npm test'",
])
def test_evidence_never_infers_test_from_keywords(command):
    assert not regression.test_command(command)


def test_handwritten_pair_ground_truth_is_independent_of_detector():
    cases = {c["id"]: c for c in contracts()}
    for name, expected in [
        ("same_file_read_after_actual_edit", "allow"),
        ("failed_edit_cannot_exempt_read", "penalize"),
        ("noop_edit_cannot_exempt_read", "penalize"),
        ("unrelated_edit_cannot_exempt_read", "penalize"),
        ("ordinary_pytest_after_edit", "allow"),
    ]:
        labels, _ = regression.annotate(cases[name]["events"])
        assert labels[-1]["expected"] == expected


def test_unknown_shell_state_is_not_fabricated_progress_or_nonsense():
    events = indexed([
        event("Read", {"file_path": "a.py"}),
        event("Bash", {"command": "python update_files.py"}, "exit_code: 0"),
        event("Read", {"file_path": "a.py"}),
    ])
    labels, observations = regression.annotate(events)
    assert labels[-1]["expected"] == "unknown"
    assert observations[0]["expected_progress"] is None


def test_successful_write_is_not_proof_content_changed():
    assert regression.effect(event("Write", {"file_path": "a.py", "content": "x"}, "Wrote 1 characters to a.py")) == "unknown"


def test_changed_reply_must_match_target_and_nonidentical_replacement():
    edit = event("Edit", {"file_path": "a.py", "old_string": "a", "new_string": "b"}, "Edited other.py (1 replacement).")
    assert regression.effect(edit) == "unknown"
    edit["response"] = "Edited a.py (1 replacement)."
    assert regression.effect(edit) == "changed"
    edit["arguments"]["new_string"] = "a"
    assert regression.effect(edit) == "unknown"


def test_fingerprint_preserves_internal_whitespace_and_input():
    original = event("Bash", {"command": "  printf 'a  b'\n"})
    before = copy.deepcopy(original)
    assert regression.fingerprint(original) == regression.fingerprint(event("Bash", {"command": "printf 'a  b'"}))
    assert regression.fingerprint(original) != regression.fingerprint(event("Bash", {"command": "printf 'a b'"}))
    assert original == before


def test_task_partition_ignores_run_checkpoint_and_outcome():
    a = {"gts": {"repo": "owner/repo", "instance_id": "task"}, "output": "Done.", "raw_score": 1}
    b = {**a, "raw_score": 0}
    source = {"run": "grpo", "split": "rollout_data", "step": 1, "path": "a", "line": 1}
    ca = regression.make_case(a, source)
    cb = regression.make_case(b, {**source, "run": "sft", "step": 12})
    assert ca["partition"] == cb["partition"]
    assert ca["task_key"] == cb["task_key"]


def test_real_temporary_files_establish_noop_and_failure_ground_truth():
    result = filesystem_evidence()
    assert result["passed"]
    assert result["before"] != result["changed"]
    assert result["changed"] == result["noop"] == result["failed"]


def test_frozen_checksum_rejects_relabelling(tmp_path):
    fixture = tmp_path / "fixtures.jsonl.gz"
    with gzip.open(fixture, "wt") as stream:
        stream.write("{}\n")
    (tmp_path / "manifest.json").write_text(json.dumps({"fixtures_sha256": "not-the-hash"}))
    with pytest.raises(ValueError, match="checksum mismatch"):
        regression.read_fixtures(tmp_path)


def test_build_refuses_to_overwrite_frozen_directory(tmp_path):
    with pytest.raises(ValueError, match="immutable"):
        regression.build(tmp_path, 160, 12)


def test_selection_is_order_independent_and_caps_tasks():
    cases = []
    for i in range(30):
        source = {"run": "sft", "split": "validation_data", "step": 2, "path": "x", "line": i + 1}
        row = {"gts": {"repo": "r", "instance_id": f"t{i // 3}"}, "output": "Done.", "raw_score": i % 2}
        cases.append(regression.make_case(row, source))
    left = regression.select_cases(cases, 16)
    right = regression.select_cases(list(reversed(cases)), 16)
    assert [c["id"] for c in left] == [c["id"] for c in right]
    assert max(__import__("collections").Counter(c["task_key"] for c in left).values()) <= 2

import json
import sys

import pytest

from scripts import make_swe_sft_data as sft


def _call(name: str, **parameters: object) -> str:
    body = "\n".join(
        f"<parameter={key}>\n{str(value).lower() if isinstance(value, bool) else value}\n</parameter>"
        for key, value in parameters.items()
    )
    return f"<tool_call>\n<function={name}>\n{body}\n</function>\n</tool_call>"


def _phase(name: str, response: str = "ok", **parameters: object) -> list[dict[str, str]]:
    if name.lower() == "edit" and response == "ok":
        target = parameters.get("file_path", parameters.get("path", "file"))
        response = f"Edited {target} (1 replacement)."
    return [
        {"role": "assistant", "content": _call(name, **parameters)},
        {"role": "user", "content": f"<tool_response>\n{response}\n</tool_response>"},
    ]


def _repeat_metrics(messages: list[dict[str, str]], read_window: int = 8, exec_window: int = 16):
    return sft._reconstructed_repeat_metrics(
        messages,
        read_window=read_window,
        exec_window=exec_window,
    )


def _valid_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "input": (
            "system\nUse repository-relative paths as the only model-facing file-tool format."
            "\nuser\nFix the bug.\nassistant\n"
        ),
        "output": (
            f"{_call('Write', file_path='src/fix.py', content='fixed')}\n"
            "user\n<tool_response>ok</tool_response>\n"
            "assistant\nImplemented and verified."
        ),
        "gts": {
            "benchmark": "swe_rebench_v2",
            "instance_id": "owner__repo-1",
            "repo": "owner/repo",
            "base_commit": "abc123",
        },
        "raw_score": 1.0,
        "shaped_score": 1.0,
        "train_sample_mask": True,
        "submission_signal_seen": True,
        "final_patch_empty": False,
        "trajectory_response_tokens": 100,
        "trajectory_tool_dispatches": 1,
        "trajectory_tool_error_returns": 0,
    }
    row.update(overrides)
    return row


def _base_rejection(row: dict[str, object]):
    return sft._base_rejection(
        row,
        require_path_policy=True,
        max_response_tokens=1000,
        max_tool_dispatches=120,
        max_penalized_repeated_tool_calls=0,
        repeated_tool_detection_mode="mutation_aware",
        repeated_tool_read_window=8,
        repeated_tool_exec_window=16,
        require_swe_mutation=True,
    )


def test_successful_exact_repeat_is_penalized():
    messages = _phase("Read", file_path="src/a.py") + _phase("Read", file_path="src/a.py")

    metrics = _repeat_metrics(messages)

    assert metrics["raw_exact_repeat_count"] == 1
    assert metrics["penalized_repeat_count"] == 1


def test_repeat_after_mutation_is_suppressed():
    messages = (
        _phase("Read", file_path="src/a.py")
        + _phase("Edit", file_path="src/a.py", old_string="a", new_string="b")
        + _phase("Read", file_path="src/a.py")
    )

    metrics = _repeat_metrics(messages)

    assert metrics["penalized_repeat_count"] == 0
    assert metrics["repeats_suppressed_after_mutation"] == 1


def test_repeated_mutation_does_not_exempt_itself():
    edit = _phase("Edit", file_path="src/a.py", old_string="a", new_string="b")
    metrics = _repeat_metrics(edit + edit)

    assert metrics["penalized_repeat_count"] == 1


def test_retry_after_tool_error_is_suppressed():
    messages = _phase(
        "Read", "Remote execution error: service unavailable", file_path="src/a.py"
    ) + _phase("Read", file_path="src/a.py")

    metrics = _repeat_metrics(messages)

    assert metrics["penalized_repeat_count"] == 0
    assert metrics["repeats_suppressed_after_error"] == 1


def test_duplicate_calls_in_one_assistant_phase_are_inflight():
    call = _call("Read", file_path="src/a.py")
    messages = [
        {"role": "assistant", "content": f"{call}\n{call}"},
        {
            "role": "user",
            "content": "<tool_response>one</tool_response>\n<tool_response>two</tool_response>",
        },
    ]

    metrics = _repeat_metrics(messages)

    assert metrics["raw_exact_repeat_count"] == 1
    assert metrics["penalized_repeat_count"] == 0
    assert metrics["repeats_suppressed_inflight"] == 1


def test_undispatched_extra_call_is_excluded_from_repeat_history():
    target = _call("Read", file_path="src/a.py")
    extra = _call("Read", file_path="src/b.py")
    messages = [
        {"role": "assistant", "content": f"{target}\n{extra}"},
        {"role": "user", "content": "<tool_response>only the first call ran</tool_response>"},
        {"role": "assistant", "content": extra},
        {"role": "user", "content": "<tool_response>now it ran</tool_response>"},
    ]

    metrics = _repeat_metrics(messages)

    assert metrics["raw_exact_repeat_count"] == 0
    assert metrics["penalized_repeat_count"] == 0
    assert metrics["repeats_suppressed_after_error"] == 0


@pytest.mark.parametrize(
    ("tool", "intervening"),
    [("Read", 8), ("run_tests", 16)],
)
def test_repeat_does_not_expire_after_intervening_calls(tool: str, intervening: int):
    parameters = {"file_path": "src/a.py"} if tool == "Read" else {"target": "unit"}
    messages = _phase(tool, **parameters)
    for index in range(intervening):
        messages += _phase("Read", file_path=f"src/filler_{index}.py")
    messages += _phase(tool, **parameters)

    metrics = _repeat_metrics(messages, read_window=8, exec_window=16)

    assert metrics["penalized_repeat_count"] == 1
    assert metrics["repeats_suppressed_outside_window"] == 0


def test_bash_verification_does_not_create_a_mutation_revision():
    messages = (
        _phase("Edit", file_path="src/a.py", old_string="a", new_string="b")
        + _phase("Bash", command="pytest -q", verification=True)
        + _phase("Bash", command="pytest -q", verification=True)
    )

    metrics = _repeat_metrics(messages)
    tool_metrics = sft._tool_metrics(
        {}, messages, detection_mode="mutation_aware", read_window=8, exec_window=16
    )

    assert metrics["penalized_repeat_count"] == 1
    assert tool_metrics["has_verification"] is True


def test_stale_stored_mutation_aware_counters_do_not_override_reconstruction():
    reconstructed = _repeat_metrics(_phase("Read", file_path="src/a.py"))
    row = {
        "trajectory_repeated_tool_detection_mode": "mutation_aware",
        "trajectory_repeated_tool_read_window": 8,
        "trajectory_repeated_tool_exec_window": 16,
        "trajectory_repeated_tool_calls": 7,
        "trajectory_penalized_repeated_tool_calls": 2,
        "trajectory_repeats_suppressed_after_mutation": 3,
        "trajectory_repeats_suppressed_after_error": 4,
        "trajectory_repeats_suppressed_outside_window": 5,
        "trajectory_repeats_suppressed_inflight": 6,
    }

    metrics = sft._repeat_metrics(
        row,
        reconstructed,
        detection_mode="mutation_aware",
        read_window=8,
        exec_window=16,
    )

    assert metrics["repeat_detection_source"] == "reconstructed"
    assert metrics["raw_exact_repeat_count"] == 0
    assert metrics["penalized_repeat_count"] == 0
    assert metrics["repeats_suppressed_after_mutation"] == 0


def test_stored_counters_with_different_windows_are_not_reused():
    reconstructed = {**_repeat_metrics(_phase("Read", file_path="src/a.py")), "penalized_repeat_count": 1}
    row = {
        "trajectory_repeated_tool_detection_mode": "mutation_aware",
        "trajectory_repeated_tool_read_window": 4,
        "trajectory_repeated_tool_exec_window": 16,
        "trajectory_penalized_repeated_tool_calls": 9,
    }

    metrics = sft._repeat_metrics(
        row,
        reconstructed,
        detection_mode="mutation_aware",
        read_window=8,
        exec_window=16,
    )

    assert metrics["repeat_detection_source"] == "reconstructed"
    assert metrics["penalized_repeat_count"] == 1


def test_legacy_exact_mode_uses_raw_repeat_count():
    reconstructed = _repeat_metrics(_phase("Read", file_path="src/a.py"))

    metrics = sft._repeat_metrics(
        {"trajectory_repeated_tool_calls": 4},
        reconstructed,
        detection_mode="legacy_exact",
        read_window=8,
        exec_window=16,
    )

    assert metrics["repeat_detection_source"] == "stored_legacy_exact"
    assert metrics["penalized_repeat_count"] == 4


def test_raw_reward_one_with_shaped_penalty_is_accepted():
    metadata, _prompt, _metrics, _messages = _base_rejection(_valid_row(shaped_score=-0.1))

    assert metadata["raw_score"] == 1.0
    assert metadata["shaped_score"] == -0.1


@pytest.mark.parametrize("raw_score", [0.0, 0.999, 1.1, None])
def test_raw_reward_must_equal_one(raw_score: object):
    with pytest.raises(sft.DataReject, match="not_reward_one"):
        _base_rejection(_valid_row(raw_score=raw_score))


def test_explicit_missing_submission_is_rejected():
    with pytest.raises(sft.DataReject, match="missing_submission"):
        _base_rejection(_valid_row(submission_signal_seen=False))


def test_unknown_submission_value_is_not_reported_as_explicit():
    metadata, _prompt, _metrics, _messages = _base_rejection(
        _valid_row(submission_signal_seen="unknown")
    )

    assert metadata["submission_check"] == "inferred_final_assistant"


def test_transcript_tool_error_is_rejected_without_stored_counter():
    output = (
        f"{_call('Write', file_path='src/fix.py', content='fixed')}\n"
        "user\n<tool_response>Remote execution error: disconnected</tool_response>\n"
        "assistant\nDone."
    )
    row = _valid_row(output=output)
    row.pop("trajectory_tool_error_returns")

    with pytest.raises(sft.DataReject, match="tool_error"):
        _base_rejection(row)


@pytest.mark.parametrize(
    "text",
    [
        "Use repository-relative paths only.",
        "Use a repository-relative path as the only model-facing format.",
        "Repository-relative paths are the only allowed form.",
    ],
)
def test_path_policy_wording_variants(text: str):
    assert sft._path_policy_ok(text)


def test_old_repeat_flags_are_compatible_aliases():
    args = sft.build_parser().parse_args(
        [
            "--output-dir",
            "out",
            "--max-tool-dispatches",
            "120",
            "--max-repeated-bash-dispatches",
            "1",
            "--relaxed-max-repeated-bash-dispatches",
            "3",
        ]
    )

    assert args.max_tool_dispatches == 120
    assert args.relaxed_max_tool_dispatches == 120
    assert args.max_penalized_repeated_tool_calls == 1
    assert args.relaxed_max_penalized_repeated_tool_calls == 3


def test_builder_writes_dataset_with_user_dispatch_limit(tmp_path, monkeypatch):
    rollout_root = tmp_path / "rollouts"
    run_dir = rollout_root / "fixture"
    run_dir.mkdir(parents=True)
    (run_dir / "1.jsonl").write_text(json.dumps(_valid_row()) + "\n", encoding="utf-8")
    output_dir = tmp_path / "output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "make_swe_sft_data.py",
            "--rollout-root",
            str(rollout_root),
            "--run",
            "fixture",
            "--output-dir",
            str(output_dir),
            "--val-fraction",
            "0",
            "--max-response-tokens",
            "95536",
            "--max-tool-dispatches",
            "120",
        ],
    )

    assert sft.main() == 0
    statistics = json.loads((output_dir / "statistics.json").read_text(encoding="utf-8"))

    assert statistics["selected_counts"]["core_train"] == 1
    assert statistics["selected_counts"]["relaxed_train"] == 1
    assert statistics["filters"]["max_tool_dispatches"] == 120
    assert statistics["filters"]["relaxed_max_tool_dispatches"] == 120
    assert (output_dir / "core_train.parquet").is_file()

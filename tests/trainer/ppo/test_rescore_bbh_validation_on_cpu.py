"""Offline report generation preserves historical trajectories and scores."""

import json

import pytest

from verl.trainer.ppo.rescore_bbh_validation import rescore_trajectory


def _row(uid="bbh:logical_deduction_seven_objects:1"):
    return {
        "uid": uid,
        "benchmark": "bbh",
        "prompt": [{"role": "user", "content": "Select one.\nOptions:\n(A) first\n(B) second"}],
        "ground_truth": "B",
        "response": r"The answer is A. </think> \boxed{(B)}",
        "reward": {"acc": 0.0, "deterministic_acc": 0.0, "format_valid": 0.0, "pred": ""},
        "finish_reason": "stop",
        "response_tokens": 10,
    }


def test_report_retains_legacy_metrics_and_does_not_modify_trajectory(tmp_path):
    source = tmp_path / "samples.jsonl"
    original = json.dumps(_row()) + "\n"
    source.write_text(original)
    destination = tmp_path / "report"
    summary = rescore_trajectory(source, destination)
    assert source.read_text() == original
    assert summary["count"] == 1
    assert summary["metrics"]["original_acc"] == 0.0
    assert summary["metrics"]["answer_acc"] == 1.0
    assert summary["metrics"]["boxed_answer_acc"] == 1.0
    record = json.loads((destination / "samples.jsonl").read_text())
    assert record["source_line"] == 1
    assert record["task"] == "logical_deduction_seven_objects"
    assert record["task_source"] == "uid"
    assert (destination / "report.md").is_file()
    before = (destination / "summary.json").read_bytes()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        rescore_trajectory(source, destination)
    assert (destination / "summary.json").read_bytes() == before


def test_no_bbh_or_duplicate_uids_leave_no_output_directory(tmp_path):
    source = tmp_path / "samples.jsonl"
    source.write_text(json.dumps({"benchmark": "gpqa_diamond"}) + "\n")
    destination = tmp_path / "report"
    with pytest.raises(ValueError, match="No BBH"):
        rescore_trajectory(source, destination)
    assert not destination.exists()
    source.write_text((json.dumps(_row()) + "\n") * 2)
    with pytest.raises(ValueError, match="Duplicate BBH"):
        rescore_trajectory(source, destination)
    assert not destination.exists()


def test_optional_metadata_must_match_saved_prompt_and_reference(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    source = tmp_path / "samples.jsonl"
    row = _row()
    source.write_text(json.dumps(row) + "\n")
    metadata = tmp_path / "bbh.parquet"
    prepared = {
        "uid": row["uid"],
        "prompt": row["prompt"],
        "reward_model": {"ground_truth": "A"},
        "extra_info": {"task": "logical_deduction_seven_objects"},
    }
    pq.write_table(pa.Table.from_pylist([prepared]), metadata)
    with pytest.raises(ValueError, match="differs from supplied metadata"):
        rescore_trajectory(source, tmp_path / "report", bbh_parquet=metadata)
    assert not (tmp_path / "report").exists()

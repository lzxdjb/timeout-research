"""Offline ASR aggregation must retain failed outputs and corpus weighting."""

import json
import sys

import pytest

from scripts.rescore_audio_asr import ASRCounts, build_report, main, read_samples, rescore_suite


def _sample(sample_id, reference_count, edits, *, task_id="net", answer="a", **output):
    return {
        "sample_id": sample_id,
        "task_id": task_id,
        "model_output": {"answer": answer, **output},
        "evaluation": {"include_in_aggregate": False},
        "metrics": {
            "wenetspeech_cer": {
                "metadata": {
                    "edit_distance": edits,
                    "reference_char_count": reference_count,
                    "prediction_normalized": answer,
                }
            }
        },
    }


def _suite(tmp_path, samples, expected_count=None):
    root = tmp_path / "suite"
    directory = root / "wenetspeech"
    directory.mkdir(parents=True)
    (directory / "samples.jsonl").write_text(
        "".join(json.dumps(sample, ensure_ascii=False) + "\n" for sample in samples), encoding="utf-8"
    )
    (root / "suite_summary.json").write_text(
        json.dumps(
            {
                "run_id": "saved-run",
                "benchmarks": [
                    {
                        "benchmark_id": "wenetspeech",
                        "summary": {
                            "tasks": [
                                {
                                    "task_id": "net",
                                    "sample_count": len(samples) if expected_count is None else expected_count,
                                }
                            ]
                        },
                    }
                ],
            }
        )
    )
    return root


def test_corpus_weighting_includes_length_errors_and_unclamped_insertions(tmp_path):
    samples = [
        _sample("short", 1, 40, answer="嗯" * 40, finish_reason="length", usage={"completion_tokens": 4096}),
        _sample("long", 99, 0),
    ]
    root = _suite(tmp_path, samples)
    result = rescore_suite(root)["benchmarks"][0]["tasks"][0]
    assert result["corpus_percent"] == 40.0
    assert result["utterance_mean_percent"] == 2000.0
    assert result["sample_count"] == 2
    assert result["total_edit_distance"] == 40
    assert result["total_reference_units"] == 100
    assert result["diagnostics"]["finish_reason_length_count"] == 1
    assert result["diagnostics"]["completion_tokens_ge_4096_count"] == 1
    assert result["diagnostics"]["repeated_block_count"] == 1
    assert result["diagnostics"]["excluded_in_original_aggregate_count"] == 2
    assert build_report([root])["runs"][0]["run_id"] == "saved-run"


@pytest.mark.parametrize("format", ["jsonl", "pretty_object", "pretty_array", "wrapped_array"])
def test_saved_sample_reader_preserves_unicode_separators(tmp_path, format):
    sample = _sample("one", 2, 0, answer="文\u2028字\u2029")
    if format == "jsonl":
        content = json.dumps(sample, ensure_ascii=False) + "\n" + json.dumps(sample, ensure_ascii=False) + "\n"
        expected = [sample, sample]
    else:
        payload = (
            sample if format == "pretty_object" else {"samples": [sample]} if format == "wrapped_array" else [sample]
        )
        content = json.dumps(payload, ensure_ascii=False, indent=2)
        expected = [sample]
    path = tmp_path / "samples.jsonl"
    path.write_text(content, encoding="utf-8")
    assert list(read_samples(path)) == expected


def test_incomplete_or_duplicate_artifacts_fail_instead_of_dropping_samples(tmp_path):
    root = _suite(tmp_path, [_sample("one", 1, 1)], expected_count=2)
    with pytest.raises(ValueError, match="refusing to publish an incomplete score"):
        rescore_suite(root)
    sample_file = root / "wenetspeech" / "samples.jsonl"
    sample_file.write_text(sample_file.read_text() * 2)
    with pytest.raises(ValueError, match="duplicate saved sample"):
        rescore_suite(root)


def test_missing_counts_and_conflicting_metric_counts_fail(tmp_path):
    sample = _sample("one", 1, 1)
    del sample["metrics"]["wenetspeech_cer"]["metadata"]["edit_distance"]
    root = _suite(tmp_path, [sample])
    with pytest.raises(ValueError, match="edit_distance must be a nonnegative integer"):
        rescore_suite(root)
    sample = _sample("one", 1, 1)
    sample["metrics"]["wenetspeech_corpus_cer"] = {"metadata": {"edit_distance": 2, "reference_char_count": 1}}
    (root / "wenetspeech" / "samples.jsonl").write_text(json.dumps(sample) + "\n")
    with pytest.raises(ValueError, match="disagree on edit/reference counts"):
        rescore_suite(root)


def test_zero_reference_insertions_remain_in_pooled_score():
    counts = ASRCounts()
    for sample in [_sample("empty", 0, 5), _sample("nonempty", 5, 0)]:
        counts.add(sample, sample["metrics"]["wenetspeech_cer"]["metadata"], "reference_char_count")
    assert counts.result("cer")["corpus_percent"] == 100.0
    empty_counts = ASRCounts()
    sample = _sample("empty", 0, 5)
    empty_counts.add(sample, sample["metrics"]["wenetspeech_cer"]["metadata"], "reference_char_count")
    with pytest.raises(ValueError, match="zero"):
        empty_counts.result("cer")


def test_librispeech_pools_word_counts_instead_of_averaging_splits(tmp_path):
    root = tmp_path / "suite"
    benchmark_dir = root / "librispeech"
    benchmark_dir.mkdir(parents=True)
    samples = []
    for task_id, reference_words, edits in [("clean", 10, 0), ("other", 90, 90)]:
        sample = _sample(task_id, reference_words, edits, task_id=task_id)
        sample["metrics"] = {
            "librispeech_corpus_wer": {
                "metadata": {
                    "edit_distance": edits,
                    "reference_word_count": reference_words,
                }
            }
        }
        samples.append(sample)
    (benchmark_dir / "samples.jsonl").write_text("".join(json.dumps(sample) + "\n" for sample in samples))
    (root / "suite_summary.json").write_text(
        json.dumps(
            {
                "benchmarks": [
                    {
                        "benchmark_id": "librispeech",
                        "summary": {
                            "tasks": [{"task_id": task_id, "sample_count": 1} for task_id in ["clean", "other"]]
                        },
                    }
                ],
            }
        )
    )
    result = rescore_suite(root)["benchmarks"][0]
    assert result["pooled"]["value_key"] == "wer"
    assert result["pooled"]["corpus_percent"] == 90.0
    assert result["pooled"]["utterance_mean_percent"] == 50.0
    assert result["pooled"]["sample_count"] == 2


def test_cli_refuses_overwriting_existing_file_before_reading_inputs(tmp_path, monkeypatch):
    output = tmp_path / "report.json"
    output.write_text("original source")
    monkeypatch.setattr(sys, "argv", ["rescore_audio_asr.py", str(tmp_path / "missing"), "--output", str(output)])
    with pytest.raises(SystemExit, match="2"):
        main()
    assert output.read_text() == "original source"

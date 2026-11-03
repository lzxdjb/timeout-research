"""Benchmark exclusions fail closed and are independent of training duration gates."""

import numpy as np
import pytest
import soundfile as sf

from scripts.audio_data.common import write_jsonl
from scripts.audio_data.filter_offline import (
    _database,
    _register_holdouts,
    filter_manifest,
    filter_records,
    inspect_audio,
)


def _tone(path, duration=0.25):
    rate = 8000
    samples = 0.2 * np.sin(2 * np.pi * 440 * np.arange(round(rate * duration)) / rate)
    sf.write(path, samples, rate, subtype="PCM_16")
    return path


def _candidate(path):
    return {
        "schema_version": 1,
        "sample_id": "candidate",
        "dataset": "fixture",
        "revision": "fixture-1",
        "source_split": "train",
        "split": "train",
        "source_recording_id": "original-1",
        "group_id": "freesound:1234",
        "audio_path": str(path),
        "task": "closed_qa",
        "language": "en",
        "question": "Can you hear a tone?",
        "reference": "yes",
        "reference_aliases": [],
        "audio_license": "CC0-1.0",
        "annotation_license": "MIT",
        "annotation_type": "human",
        "reference_verified": True,
        "metadata": {},
    }


@pytest.mark.parametrize("audio_path", [None, "", "   ", 3, False, [], {}])
def test_supplied_invalid_audio_path_cannot_be_hidden_by_valid_group(audio_path):
    with pytest.raises(ValueError):
        filter_records([], holdout_records=[{"group_id": "freesound:1234", "audio_path": audio_path}])


def test_missing_audio_path_raises_even_with_valid_group(tmp_path):
    with pytest.raises(ValueError):
        filter_records(
            [],
            holdout_records=[
                {
                    "group_id": "freesound:1234",
                    "audio_path": str(tmp_path / "missing.wav"),
                }
            ],
        )


def test_corrupt_audio_raises_even_with_valid_group(tmp_path):
    corrupt = tmp_path / "corrupt.wav"
    corrupt.write_bytes(b"not an audio container")
    with pytest.raises(ValueError):
        filter_records([], holdout_records=[{"group_id": "freesound:1234", "audio_path": str(corrupt)}])


@pytest.mark.parametrize("identity", [{}, {"group_id": ""}, {"group_id": "  "}, {"pcm_sha256": None}])
def test_holdout_requires_usable_group_or_audio_identity(identity):
    with pytest.raises(ValueError):
        filter_records([], holdout_records=[identity])


@pytest.mark.parametrize("field", ["group_id", "pcm_sha256"])
@pytest.mark.parametrize("value", [7, True, [], {}, "", "  "])
def test_invalid_identity_types_raise_even_when_audio_is_usable(tmp_path, field, value):
    audio = _tone(tmp_path / "benchmark.wav")
    with pytest.raises(ValueError):
        filter_records([], holdout_records=[{"audio_path": str(audio), field: value}])


def test_none_optional_identity_fields_allow_usable_audio(tmp_path):
    audio = _tone(tmp_path / "benchmark.wav")
    assert (
        filter_records(
            [],
            holdout_records=[
                {
                    "audio_path": str(audio),
                    "group_id": None,
                    "pcm_sha256": None,
                }
            ],
        )
        == []
    )


@pytest.mark.parametrize("digest", ["A" * 64, "0" * 63, "g" * 64])
def test_prehashed_holdout_requires_canonical_digest(digest):
    with pytest.raises(ValueError):
        filter_records([], holdout_records=[{"group_id": "valid-group", "pcm_sha256": digest}])


def test_prehashed_holdout_quarantines_matching_candidate(tmp_path):
    audio = _tone(tmp_path / "candidate.wav")
    pcm = inspect_audio(audio)["pcm_sha256"]
    (result,) = filter_records([_candidate(audio)], holdout_records=[{"pcm_sha256": pcm}])
    assert result["qc"]["offline"]["decision"] == "quarantine"
    assert "cross_split_pcm" in result["qc"]["offline"]["reasons"]


def test_supplied_pcm_must_match_supplied_audio(tmp_path):
    audio = _tone(tmp_path / "benchmark.wav")
    with pytest.raises(ValueError, match="does not match"):
        filter_records(
            [],
            holdout_records=[
                {
                    "audio_path": str(audio),
                    "group_id": "valid-group",
                    "pcm_sha256": "0" * 64,
                }
            ],
        )


def test_group_only_holdout_quarantines_matching_candidate(tmp_path):
    candidate = _candidate(_tone(tmp_path / "candidate.wav"))
    (result,) = filter_records([candidate], holdout_records=[{"group_id": candidate["group_id"]}])
    assert result["qc"]["offline"]["decision"] == "quarantine"
    assert "cross_split_recording" in result["qc"]["offline"]["reasons"]


def test_pcm_holdout_matches_differently_packaged_audio_without_shared_group(tmp_path):
    original = _tone(tmp_path / "candidate.wav")
    samples, rate = sf.read(original, dtype="int16")
    benchmark = tmp_path / "benchmark.flac"
    sf.write(benchmark, samples, rate, subtype="PCM_16")
    (result,) = filter_records([_candidate(original)], holdout_records=[{"audio_path": str(benchmark)}])
    assert result["qc"]["offline"]["decision"] == "quarantine"
    assert "cross_split_pcm" in result["qc"]["offline"]["reasons"]
    assert "cross_split_recording" not in result["qc"]["offline"]["reasons"]


@pytest.mark.parametrize("duration", [0.001, 121.0])
def test_holdout_index_ignores_training_duration_limits(tmp_path, duration):
    benchmark = _tone(tmp_path / "benchmark.wav", duration=duration)
    connection = _database(":memory:")
    try:
        _register_holdouts(
            connection,
            [{"audio_path": str(benchmark)}],
            {
                "min_duration": 0.1,
                "max_duration": 120.0,
                "max_file_bytes": 256 * 1024 * 1024,
            },
        )
        entry = connection.execute("SELECT value, split FROM media WHERE kind='pcm'").fetchone()
        assert entry is not None
        assert len(entry[0]) == 64
        assert entry[1] == "test"
    finally:
        connection.close()


def test_holdout_file_size_limit_is_retained(tmp_path):
    audio = _tone(tmp_path / "benchmark.wav")
    with pytest.raises(ValueError):
        filter_records(
            [],
            holdout_records=[{"audio_path": str(audio), "group_id": "valid-group"}],
            max_file_bytes=audio.stat().st_size - 1,
        )


def test_failed_holdout_does_not_publish_partially_checked_manifest(tmp_path):
    candidate = _candidate(_tone(tmp_path / "candidate.wav"))
    input_path, output_path = tmp_path / "input.jsonl", tmp_path / "checked.jsonl"
    holdout_path = tmp_path / "holdouts.jsonl"
    write_jsonl(input_path, [candidate])
    write_jsonl(holdout_path, [{"audio_path": str(tmp_path / "missing-benchmark.wav")}])
    output_path.write_text("previous accepted export\n", encoding="utf-8")
    with pytest.raises(ValueError):
        filter_manifest(input_path, output_path, holdout_manifest=holdout_path)
    assert output_path.read_text(encoding="utf-8") == "previous accepted export\n"

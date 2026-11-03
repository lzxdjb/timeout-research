# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Audio curation contract tests; no GPU, model download, or remote endpoint required.

Small generated waveforms exercise real decoding and content hashes. These tests
verify evidence boundaries and leakage prevention, rather than treating a model's
agreement with a reference as proof that the reference is correct.
"""

import copy
import hashlib
import importlib.util
import json
import math
import struct
import subprocess
import sys
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from scripts.audio_data.common import record_fingerprint
from scripts.audio_data.convert_verl import convert_manifest, convert_records
from scripts.audio_data.filter_offline import filter_records

REPO_ROOT = Path(__file__).resolve().parents[2]
# The reward is stdlib-only; importing verl itself would load the training stack.
_REWARD_SPEC = importlib.util.spec_from_file_location(
    "_audio_data_test_reward", REPO_ROOT / "verl/utils/reward_score/audio.py"
)
_REWARD_MODULE = importlib.util.module_from_spec(_REWARD_SPEC)
_REWARD_SPEC.loader.exec_module(_REWARD_MODULE)
compute_score = _REWARD_MODULE.compute_score


def _write_wav(path, *, frequency=440, amplitude=0.2, duration=0.25, rate=16000):
    samples = [
        round(32767 * amplitude * math.sin(2 * math.pi * frequency * index / rate))
        for index in range(round(duration * rate))
    ]
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(rate)
        writer.writeframes(struct.pack(f"<{len(samples)}h", *samples))
    return path


def _record(path, sample_id="sample-1", **changes):
    row = {
        "schema_version": 1,
        "sample_id": sample_id,
        "dataset": "test_audio",
        "revision": "fixture-v1",
        "source_split": "train",
        "split": "train",
        "source_recording_id": f"recording-{sample_id}",
        "group_id": f"test_audio/recording-{sample_id}",
        "audio_path": str(path.resolve()),
        "task": "closed_qa",
        "language": "en",
        "question": "Is there a sustained tone? Answer yes or no.",
        "reference": "yes",
        "reference_aliases": [],
        "audio_license": "CC-BY-4.0",
        "annotation_license": "CC-BY-4.0",
        "annotation_type": "human",
        "reference_verified": True,
        "metadata": {},
        "qc": {},
    }
    row.update(changes)
    return row


def _decision(row):
    return row["qc"]["offline"]["decision"]


def _ground_truth(task="closed_qa", reference="yes", **changes):
    payload = {
        "task": task,
        "reference": reference,
        "reference_aliases": [],
        "language": "en",
        "label_set_complete": False,
    }
    payload.update(changes)
    return json.dumps(payload, ensure_ascii=False)


def _accept_qwen(row):
    row["qc"]["qwen"] = {
        "decision": "accept",
        "reference_supported": True,
        "input_fingerprint": record_fingerprint(row),
        "reasons": [],
    }
    return row


def test_corrupt_audio_rejected_and_silence_quarantined(tmp_path):
    corrupt = tmp_path / "corrupt.wav"
    corrupt.write_bytes(b"this is not an audio container")
    silent = _write_wav(tmp_path / "silence.wav", amplitude=0)
    rows = filter_records([_record(corrupt, "corrupt"), _record(silent, "silent")])

    assert _decision(rows[0]) == "reject"
    assert _decision(rows[1]) == "quarantine"
    assert "silent_audio" in rows[1]["qc"]["offline"]["reasons"]
    assert rows[1]["qc"]["offline"]["metrics"]["duration_seconds"] == pytest.approx(0.25)


def test_low_energy_environmental_sound_is_not_rejected_by_generic_speech_rules(tmp_path):
    path = _write_wav(tmp_path / "quiet.wav", amplitude=0.001)
    row = filter_records([_record(path, task="caption", reference="A faint sustained tone.")])[0]

    assert _decision(row) == "accept"
    assert len(row["audio_sha256"]) == 64
    assert len(row["pcm_sha256"]) == 64
    assert row["qc"]["offline"]["metrics"]["duration_seconds"] == pytest.approx(0.25)


def test_same_decoded_audio_across_containers_quarantines_both_splits(tmp_path):
    import soundfile as sf

    wav = _write_wav(tmp_path / "clip.wav")
    pcm, rate = sf.read(wav, dtype="int16")
    flac = tmp_path / "clip.flac"
    sf.write(flac, pcm, rate, subtype="PCM_16")
    rows = filter_records(
        [
            _record(wav, "train"),
            _record(flac, "test", split="test", source_split="test"),
        ]
    )

    assert rows[0]["audio_sha256"] != rows[1]["audio_sha256"]
    assert rows[0]["pcm_sha256"] == rows[1]["pcm_sha256"]
    assert [_decision(row) for row in rows] == ["quarantine", "quarantine"]


def test_different_questions_on_one_training_recording_are_retained(tmp_path):
    path = _write_wav(tmp_path / "clip.wav")
    common = {"group_id": "test_audio/shared", "source_recording_id": "shared"}
    rows = filter_records(
        [
            _record(path, "question-1", **common),
            _record(path, "question-2", question="Is there human speech?", reference="no", **common),
        ]
    )

    assert [_decision(row) for row in rows] == ["accept", "accept"]
    assert len(convert_records(rows, mode="opd")) == 2


def test_identical_example_is_retained_once_without_discarding_other_questions(tmp_path):
    path = _write_wav(tmp_path / "clip.wav")
    rows = filter_records([_record(path, "first-copy"), _record(path, "second-copy")])

    assert [_decision(row) for row in rows].count("accept") == 1
    assert any("duplicate_example" in row["qc"]["offline"]["reasons"] for row in rows)
    assert len(convert_records(rows, mode="opd")) == 1


def test_ineligible_duplicate_does_not_suppress_usable_copy(tmp_path):
    path = _write_wav(tmp_path / "clip.wav")
    rows = filter_records(
        [
            _record(path, "a-unlicensed", audio_license="unknown"),
            _record(path, "z-permitted"),
        ]
    )

    assert _decision(rows[0]) != "accept"
    assert _decision(rows[1]) == "accept"
    assert len(convert_records(rows, mode="opd")) == 1


def test_recording_group_leakage_is_blocked_even_when_segment_audio_differs(tmp_path):
    first = _write_wav(tmp_path / "segment-1.wav", frequency=440)
    second = _write_wav(tmp_path / "segment-2.wav", frequency=880)
    rows = filter_records(
        [
            _record(first, "train", group_id="shared-original-recording"),
            _record(
                second,
                "validation",
                split="validation",
                source_split="validation",
                group_id="shared-original-recording",
            ),
        ]
    )

    assert rows[0]["pcm_sha256"] != rows[1]["pcm_sha256"]
    assert [_decision(row) for row in rows] == ["quarantine", "quarantine"]


@pytest.mark.parametrize("source_split", ["test-clean", "validation", "dev", "dev:val", "nsynth-valid"])
def test_official_evaluation_split_cannot_be_relabelled_as_training(tmp_path, source_split):
    row = _record(_write_wav(tmp_path / "clip.wav"), source_split=source_split, split="train")
    filtered = filter_records([row])

    assert _decision(filtered[0]) != "accept"
    assert not convert_records(filtered, mode="opd", split="train")


def test_changed_media_cannot_be_exported_using_old_filter_evidence(tmp_path):
    path = _write_wav(tmp_path / "clip.wav", frequency=440)
    filtered = filter_records([_record(path)])
    assert _decision(filtered[0]) == "accept"
    _write_wav(path, frequency=880)

    with pytest.raises(ValueError, match="(?i)(hash|changed|integrity|digest)"):
        convert_records(filtered, mode="opd")


def test_edited_reference_invalidates_previous_filter_evidence(tmp_path):
    row = filter_records([_record(_write_wav(tmp_path / "clip.wav"))])[0]
    row["reference"] = "a different answer"

    with pytest.raises(ValueError, match="(?i)(manifest|changed|fingerprint)"):
        convert_records([row], mode="rl")


def test_separate_benchmark_manifest_blocks_pcm_and_group_overlap(tmp_path):
    first = _write_wav(tmp_path / "first.wav", frequency=400)
    second = _write_wav(tmp_path / "second.wav", frequency=600)
    rows = filter_records(
        [
            _record(first, "pcm-match", group_id="different-group"),
            _record(second, "group-match", group_id="benchmark-original-recording"),
        ],
        holdout_records=[{"audio_path": str(first), "group_id": "benchmark-original-recording"}],
    )

    assert [_decision(row) for row in rows] == ["quarantine", "quarantine"]
    assert "cross_split_pcm" in rows[0]["qc"]["offline"]["reasons"]
    assert "cross_split_recording" in rows[1]["qc"]["offline"]["reasons"]


def test_independent_asr_evidence_must_match_media_and_configured_threshold(tmp_path):
    path = _write_wav(tmp_path / "speech-fixture.wav")
    row = _record(path, task="asr", question="Transcribe the speech.", reference="hello world")
    hypothesis = {
        "audio_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "text": "hello world",
        "model": "fixture-independent-asr",
    }
    policy = {"max_wer": 0.1, "require_reference_check": True}
    matched = filter_records([row], asr_hypotheses={row["sample_id"]: hypothesis}, **policy)[0]
    assert _decision(matched) == "accept"
    assert matched["qc"]["offline"]["metrics"]["wer"] == 0

    changed_audio = dict(hypothesis, audio_sha256="0" * 64)
    mismatch = filter_records([row], asr_hypotheses={row["sample_id"]: changed_audio}, **policy)[0]
    assert _decision(mismatch) == "quarantine"
    assert "asr_hypothesis_audio_mismatch" in mismatch["qc"]["offline"]["reasons"]

    disagreement = dict(hypothesis, text="different words")
    disputed = filter_records([row], asr_hypotheses={row["sample_id"]: disagreement}, **policy)[0]
    assert _decision(disputed) == "quarantine"
    assert "asr_reference_disagreement" in disputed["qc"]["offline"]["reasons"]


@pytest.mark.parametrize(
    ("language", "reference", "text", "metric", "rate"),
    [
        ("en", "one two", "one", "wer", 0.5),
        ("zh-CN", "北京欢迎你", "北 京", "cer", 0.6),
        ("cmn", "北京欢迎你", "北 京", "cer", 0.6),
    ],
)
def test_asr_checks_use_language_metric_and_keep_threshold_boundary(tmp_path, language, reference, text, metric, rate):
    path = _write_wav(tmp_path / "clip.wav")
    row = _record(path, task="asr", language=language, reference=reference)
    hypothesis = {"audio_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "text": text}
    policy = {"require_reference_check": True, f"max_{metric}": rate}

    at_boundary = filter_records([row], asr_hypotheses={row["sample_id"]: hypothesis}, **policy)[0]
    assert _decision(at_boundary) == "accept"
    assert at_boundary["qc"]["offline"]["metrics"][metric] == pytest.approx(rate)
    assert at_boundary["reference"] == reference

    policy[f"max_{metric}"] = rate - 0.01
    disagreement = filter_records([row], asr_hypotheses={row["sample_id"]: hypothesis}, **policy)[0]
    assert _decision(disagreement) == "quarantine"
    assert "asr_reference_disagreement" in disagreement["qc"]["offline"]["reasons"]
    assert disagreement["reference"] == reference
    assert row["qc"] == {}


def test_asr_reference_aliases_support_agreement_without_changing_primary_reference(tmp_path):
    path = _write_wav(tmp_path / "clip.wav")
    row = _record(path, task="asr", reference="colour", reference_aliases=["color", "", "!!!"])
    hypothesis = {"audio_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "text": "color"}
    checked = filter_records(
        [row], asr_hypotheses={row["sample_id"]: hypothesis}, max_wer=0, require_reference_check=True
    )[0]

    assert _decision(checked) == "accept"
    assert checked["reference"] == "colour"
    assert checked["qc"]["offline"]["metrics"]["wer"] == 0
    assert len(convert_records([checked], mode="opd")) == 1
    assert not convert_records([checked], mode="rl")


def test_asr_alias_cannot_replace_missing_usable_primary_reference(tmp_path):
    path = _write_wav(tmp_path / "clip.wav")
    row = _record(path, task="asr", reference="!!!", reference_aliases=["recognized words"])
    hypothesis = {"audio_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "text": "recognized words"}
    checked = filter_records(
        [row], asr_hypotheses={row["sample_id"]: hypothesis}, max_wer=0, require_reference_check=True
    )[0]

    assert _decision(checked) == "quarantine"
    assert "missing_asr_reference" in checked["qc"]["offline"]["reasons"]
    assert "wer" not in checked["qc"]["offline"]["metrics"]


@pytest.mark.parametrize("text", ["", " ", "!!!"])
def test_normalized_empty_asr_hypothesis_never_establishes_reference_agreement(tmp_path, text):
    path = _write_wav(tmp_path / "clip.wav")
    row = _record(path, task="asr", reference="valid words")
    hypothesis = {"audio_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "text": text}
    checked = filter_records([row], asr_hypotheses={row["sample_id"]: hypothesis}, max_wer=0)[0]

    assert _decision(checked) == "quarantine"
    assert "invalid_asr_hypothesis" in checked["qc"]["offline"]["reasons"]
    assert "wer" not in checked["qc"]["offline"]["metrics"]


@pytest.mark.parametrize(
    ("extra", "accepted"),
    [
        ({}, True),
        ({"language": "en-US"}, True),
        ({"language": "English"}, True),
        ({"language": "zh"}, False),
        ({"language": None}, False),
        ({"language": ["en"]}, False),
    ],
)
def test_asr_hypothesis_language_must_match_when_declared(tmp_path, extra, accepted):
    path = _write_wav(tmp_path / "clip.wav")
    row = _record(path, task="asr", reference="valid words")
    hypothesis = {"audio_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "text": "valid words", **extra}
    checked = filter_records(
        [row], asr_hypotheses={row["sample_id"]: hypothesis}, max_wer=0, require_reference_check=True
    )[0]

    assert _decision(checked) == ("accept" if accepted else "quarantine")
    if not accepted:
        assert "asr_hypothesis_language_mismatch" in checked["qc"]["offline"]["reasons"]


@pytest.mark.parametrize("task", ["closed_qa", "classification", "music_family", "caption", "open_qa"])
def test_asr_reference_policy_does_not_gate_other_audio_tasks(tmp_path, task):
    row = _record(_write_wav(tmp_path / "clip.wav"), task=task)
    checked = filter_records(
        [row],
        asr_hypotheses={row["sample_id"]: {"text": None, "audio_sha256": "wrong", "language": ["bad"]}},
        max_wer=0,
        max_cer=0,
        require_reference_check=True,
    )[0]

    assert _decision(checked) == "accept"
    assert "wer" not in checked["qc"]["offline"]["metrics"]
    assert "cer" not in checked["qc"]["offline"]["metrics"]


def test_rerunning_asr_check_invalidates_qwen_review_and_preserves_gold(tmp_path):
    path = _write_wav(tmp_path / "clip.wav")
    row = _record(path, task="asr", reference="original transcript")
    original = _accept_qwen(filter_records([row])[0])
    hypothesis = {"audio_sha256": original["audio_sha256"], "text": "different words"}
    checked = filter_records(
        [original], asr_hypotheses={row["sample_id"]: hypothesis}, max_wer=0, require_reference_check=True
    )[0]

    assert _decision(checked) == "quarantine"
    assert "qwen" not in checked["qc"]
    assert checked["reference"] == "original transcript"
    assert record_fingerprint(checked) == record_fingerprint(original)
    assert original["qc"]["qwen"]["decision"] == "accept"


def test_concurrent_offline_runs_do_not_share_asr_policy_or_mutate_input(tmp_path):
    path = _write_wav(tmp_path / "clip.wav")
    row = _record(path, task="asr", reference="one two")
    hypothesis = {"audio_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "text": "one"}

    def check(threshold):
        return filter_records(
            [row], asr_hypotheses={row["sample_id"]: hypothesis}, max_wer=threshold, require_reference_check=True
        )[0]

    with ThreadPoolExecutor(max_workers=2) as executor:
        at_boundary, below_boundary = list(executor.map(check, [0.5, 0.49]))

    assert _decision(at_boundary) == "accept"
    assert _decision(below_boundary) == "quarantine"
    assert row["qc"] == {}


def test_reference_checks_can_target_training_without_skipping_global_leakage_scan(tmp_path):
    train_path = _write_wav(tmp_path / "train.wav", frequency=400)
    validation_path = _write_wav(tmp_path / "validation.wav", frequency=600)
    shared_path = _write_wav(tmp_path / "shared.wav", frequency=800)
    rows = [
        _record(train_path, "train-asr", task="asr", reference="valid words"),
        _record(
            validation_path,
            "validation-asr",
            task="asr",
            reference="valid words",
            split="validation",
            source_split="validation",
        ),
        _record(shared_path, "leak-train", task="asr", reference="valid words"),
        _record(shared_path, "leak-test", task="asr", reference="valid words", split="test", source_split="test"),
    ]
    hypotheses = {
        row["sample_id"]: {
            "audio_sha256": hashlib.sha256(Path(row["audio_path"]).read_bytes()).hexdigest(),
            "text": "valid words",
        }
        for row in (rows[0], rows[2])
    }
    checked = filter_records(
        rows, asr_hypotheses=hypotheses, max_wer=0, require_reference_check=True, reference_check_splits=["train"]
    )

    assert [_decision(row) for row in checked] == ["accept", "accept", "quarantine", "quarantine"]
    assert "wer" not in checked[1]["qc"]["offline"]["metrics"]
    assert "cross_split_pcm" in checked[2]["qc"]["offline"]["reasons"]
    assert "cross_split_pcm" in checked[3]["qc"]["offline"]["reasons"]


@pytest.mark.parametrize("reference", [None, [], ""])
def test_required_asr_reference_check_quarantines_missing_references(tmp_path, reference):
    path = _write_wav(tmp_path / "speech-fixture.wav")
    row = _record(path, task="asr", question="Transcribe the speech.", reference=reference)
    hypothesis = {
        "audio_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "text": "recognized words",
        "model": "fixture-independent-asr",
    }
    checked = filter_records(
        [row],
        asr_hypotheses={row["sample_id"]: hypothesis},
        max_wer=0.1,
        require_reference_check=True,
    )[0]

    assert _decision(checked) == "quarantine"


def test_programmatic_exports_default_to_training_split(tmp_path):
    import pyarrow.parquet as pq

    rows = filter_records(
        [
            _record(
                _write_wav(tmp_path / f"{split}.wav", frequency=400 + index * 200),
                split,
                split=split,
                source_split=split,
            )
            for index, split in enumerate(("train", "validation", "test"))
        ]
    )
    assert all(_decision(row) == "accept" for row in rows)
    exported = convert_records(rows)
    assert [row["extra_info"]["split"] for row in exported] == ["train"]

    manifest = tmp_path / "all-splits.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    output = tmp_path / "default.parquet"
    convert_manifest(manifest, output)

    assert [row["extra_info"]["split"] for row in pq.read_table(output).to_pylist()] == ["train"]


def test_opd_and_rl_have_distinct_reference_requirements(tmp_path):
    first = _write_wav(tmp_path / "closed.wav", frequency=400)
    second = _write_wav(tmp_path / "caption.wav", frequency=600)
    third = _write_wav(tmp_path / "unverified.wav", frequency=800)
    rows = filter_records(
        [
            _record(first, "closed"),
            _record(second, "caption", task="caption", reference="A continuous tone."),
            _record(third, "unverified", reference_verified=False),
        ]
    )

    assert len(convert_records(rows, mode="opd")) == 3
    assert len(convert_records(rows, mode="rl")) == 1


def test_required_review_and_synthetic_examples_exclude_unreviewed_rows(tmp_path):
    rows = filter_records(
        [
            _record(_write_wav(tmp_path / "human.wav", frequency=400), "human"),
            _record(
                _write_wav(tmp_path / "synthetic.wav", frequency=800),
                "synthetic",
                annotation_type="synthetic",
                reference_verified=False,
            ),
        ]
    )

    assert len(convert_records(rows, mode="opd")) == 1
    assert not convert_records(rows, mode="opd", require_qwen=True)

    reviewed = [_accept_qwen(row) for row in rows]
    assert len(convert_records(reviewed, mode="opd", require_qwen=True)) == 2
    assert len(convert_records(reviewed, mode="rl", require_qwen=True)) == 2


def test_stale_qwen_acceptance_cannot_be_exported(tmp_path):
    row = _accept_qwen(filter_records([_record(_write_wav(tmp_path / "clip.wav"))])[0])
    row["qc"]["qwen"]["input_fingerprint"] = "0" * 64

    with pytest.raises(ValueError, match="(?i)(qwen|review|changed|fingerprint)"):
        convert_records([row], mode="rl")


@pytest.mark.parametrize("invalid_stage", ["accept", None, [], 42])
def test_invalid_qc_stage_type_is_excluded(invalid_stage, tmp_path):
    row = filter_records([_record(_write_wav(tmp_path / "clip.wav"))])[0]
    row["qc"]["qwen"] = invalid_stage

    assert not convert_records([row], mode="opd")


def test_incomplete_positive_only_sound_labels_are_not_closed_world_rl_targets(tmp_path):
    path = _write_wav(tmp_path / "clip.wav")
    row = _record(path, task="classification", reference=["Tone"], metadata={"label_set_complete": False})
    filtered = filter_records([row])

    assert len(convert_records(filtered, mode="opd")) == 1
    assert not convert_records(filtered, mode="rl")


def test_reference_and_verification_metadata_do_not_enter_student_prompt(tmp_path):
    reference = "REFERENCE_SENTINEL_87432"
    alias = "ALIAS_SENTINEL_45621"
    transcript = "TRANSCRIPT_SENTINEL_91264"
    row = _record(
        _write_wav(tmp_path / "clip.wav"),
        reference=reference,
        reference_aliases=[alias],
        metadata={"asr_transcript": transcript},
    )
    row = _accept_qwen(filter_records([row])[0])
    row["qc"]["qwen"]["reason"] = "JUDGE_SENTINEL_23871"
    exported = convert_records([row], mode="rl")[0]
    prompt = json.dumps(exported["prompt"], ensure_ascii=False)

    assert "<audio>" in prompt
    for secret in (reference, alias, transcript, "JUDGE_SENTINEL_23871"):
        assert secret not in prompt
    assert reference in exported["reward_model"]["ground_truth"]


def test_noncommercial_media_requires_explicit_opt_in_at_both_stages(tmp_path):
    row = _record(_write_wav(tmp_path / "clip.wav"), audio_license="CC-BY-NC-4.0")
    assert _decision(filter_records([row])[0]) != "accept"

    allowed = filter_records([copy.deepcopy(row)], allow_noncommercial=True)
    assert _decision(allowed[0]) == "accept"
    assert not convert_records(allowed, mode="opd")
    assert len(convert_records(allowed, mode="opd", allow_noncommercial=True)) == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", True),
        ("schema_version", 1.0),
        ("sample_id", {"id": "bad"}),
        ("sample_id", ["bad"]),
        ("source_recording_id", {"id": "bad"}),
        ("group_id", {"id": "bad"}),
        ("group_id", ["bad"]),
        ("split", {"partition": "train"}),
        ("split", ["train"]),
        ("task", {"name": "asr"}),
        ("task", ["asr"]),
        ("annotation_type", {"name": "human"}),
        ("annotation_type", ["human"]),
        ("metadata", []),
        ("metadata", {"quarantine_reasons": None}),
        ("metadata", {"quarantine_reasons": "invalid source"}),
        ("metadata", {"quarantine_reasons": [{"reason": "invalid source"}]}),
        ("metadata", {"annotation_consensus": []}),
        ("metadata", {"label_set_complete": "true"}),
        ("reference", {"answer": "yes"}),
        ("reference_aliases", {"alias": "yes"}),
    ],
)
def test_malformed_manifest_field_types_quarantine_without_registry_errors(tmp_path, field, value):
    row = _record(_write_wav(tmp_path / "clip.wav"))
    row[field] = value

    checked = filter_records([row])[0]

    assert _decision(checked) == "quarantine"
    assert checked["qc"]["offline"]["reasons"]
    assert not convert_records([checked], mode="opd")


def test_malformed_optional_pcm_hash_on_corrupt_audio_does_not_break_registry(tmp_path):
    path = tmp_path / "corrupt.wav"
    path.write_bytes(b"not an audio file")
    row = _record(path, pcm_sha256={"unexpected": "object"})

    checked = filter_records([row])[0]

    assert _decision(checked) == "reject"
    assert not convert_records([checked], mode="opd")


def test_malformed_license_url_is_unapproved_instead_of_crashing(tmp_path):
    row = _record(_write_wav(tmp_path / "clip.wav"), audio_license="https://[invalid")

    checked = filter_records([row])[0]

    assert _decision(checked) == "quarantine"
    assert "unapproved_audio_license" in checked["qc"]["offline"]["reasons"]


@pytest.mark.parametrize("task", ["asr", "closed_qa", "music_family", "classification"])
def test_normalized_empty_references_are_opd_only(tmp_path, task):
    reference = ["!!!"] if task == "classification" else "!!!"
    row = _record(
        _write_wav(tmp_path / "clip.wav"),
        task=task,
        reference=reference,
        metadata={"label_set_complete": True},
    )
    checked = filter_records([row])

    exported = convert_records(checked, mode="opd")
    assert len(exported) == 1
    assert exported[0]["extra_info"]["rl_eligible"] is False
    assert not convert_records(checked, mode="rl")


@pytest.mark.parametrize("aliases", [[""], ["!!!"], ["valid", "!!!"]])
def test_normalized_empty_aliases_cannot_create_unscorable_rl_exports(tmp_path, aliases):
    row = _record(_write_wav(tmp_path / "clip.wav"), reference_aliases=aliases)
    checked = filter_records([row])

    assert len(convert_records(checked, mode="opd")) == 1
    assert not convert_records(checked, mode="rl")


def test_task_rewards_respect_language_and_fail_closed_for_malformed_answers():
    english = _ground_truth(task="asr", reference="Hello, world!")
    assert compute_score("audio/asr", "hello world", english) == pytest.approx(1.0)
    assert compute_score("audio/asr", "hello", english) == pytest.approx(0.5)

    mandarin = _ground_truth(task="asr", reference="北京欢迎你", language="zh")
    assert compute_score("audio/asr", "北 京 欢 迎 你", mandarin) == pytest.approx(1.0)
    assert compute_score("audio/asr", "北京", mandarin) == pytest.approx(0.4)

    yes_no = _ground_truth(reference_aliases=["yeah"])
    assert compute_score("audio/closed_qa", "yeah", yes_no) == pytest.approx(1.0)
    assert compute_score("audio/closed_qa", '{"answer": yes}', yes_no) == pytest.approx(0.0)
    assert compute_score("audio/closed_qa", "no", yes_no) == pytest.approx(0.0)


def test_incomplete_sound_labels_cannot_reward_absence_of_unannotated_events():
    ground_truth = _ground_truth(task="classification", reference=["Dog", "Bird"])
    with pytest.raises(ValueError, match="(?i)(incomplete|complete|label)"):
        compute_score("audio/classification", '["Dog", "Bird"]', ground_truth)

    complete = _ground_truth(task="classification", reference=["Dog", "Bird"], label_set_complete=True)
    assert compute_score("audio/classification", '["Bird", "Dog"]', complete) == pytest.approx(1.0)


@pytest.mark.parametrize("aliases", [[""], ["!!!"], ["", "!!!"]])
def test_empty_normalized_reward_aliases_are_invalid_specifications(aliases):
    with pytest.raises(ValueError, match="(?i)(alias|reference|normaliz|empty)"):
        compute_score("audio/closed_qa", "yes", _ground_truth(reference_aliases=aliases))


@pytest.mark.parametrize("task", ["asr", "closed_qa", "music_family", "classification"])
def test_punctuation_only_reward_reference_is_invalid(task):
    reference = ["!!!"] if task == "classification" else "!!!"
    truth = _ground_truth(task=task, reference=reference, label_set_complete=True)

    with pytest.raises(ValueError, match="(?i)(reference|normaliz|empty)"):
        compute_score(f"audio/{task}", "!!!", truth)


@pytest.mark.parametrize("task", ["asr", "closed_qa", "music_family", "classification"])
def test_punctuation_only_prediction_never_earns_reward(task):
    reference = ["Dog"] if task == "classification" else "yes"
    prediction = '["!!!"]' if task == "classification" else "!!!"
    truth = _ground_truth(task=task, reference=reference, label_set_complete=True)

    assert compute_score(f"audio/{task}", prediction, truth) == pytest.approx(0.0)


def test_parquet_cli_round_trip_preserves_audio_and_reward_contract(tmp_path):
    import pyarrow.parquet as pq

    path = _write_wav(tmp_path / "clip.wav")
    manifest = tmp_path / "input.jsonl"
    manifest.write_text(json.dumps(_record(path)) + "\n", encoding="utf-8")
    filtered = tmp_path / "filtered.jsonl"
    output = tmp_path / "train.parquet"

    for module, args in (
        ("filter_offline", ["--input", str(manifest), "--output", str(filtered)]),
        (
            "convert_verl",
            ["--input", str(filtered), "--output", str(output), "--mode", "rl", "--split", "train"],
        ),
    ):
        result = subprocess.run(
            [sys.executable, "-m", f"scripts.audio_data.{module}", *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stdout + result.stderr

    rows = pq.read_table(output).to_pylist()
    assert len(rows) == 1
    row = rows[0]
    assert "<audio>" in json.dumps(row["prompt"])
    assert str(path.resolve()) in json.dumps(row["audios"])
    truth = json.loads(row["reward_model"]["ground_truth"])
    assert truth["reference"] == "yes"
    assert compute_score(row["data_source"], "yes", row["reward_model"]["ground_truth"]) == pytest.approx(1.0)

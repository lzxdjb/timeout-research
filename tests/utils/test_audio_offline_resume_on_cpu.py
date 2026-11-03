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
"""Crash/resume contracts using real tiny waveforms, without models or network."""

import copy
import hashlib
import json
import math
import os
import struct
import subprocess
import sys
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.audio_data import filter_offline as offline


def _write_wave(path, frequency):
    rate = 16000
    samples = [round(6000 * math.sin(2 * math.pi * frequency * index / rate)) for index in range(3200)]
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(rate)
        writer.writeframes(struct.pack(f"<{len(samples)}h", *samples))
    return path


def _write_manifest(path, records):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")


def _read_manifest(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.fixture
def resume_case(tmp_path):
    paths = [_write_wave(tmp_path / f"clip-{index}.wav", 400 + index * 100) for index in range(5)]
    paths.append(paths[0])  # A late test row must invalidate the earlier training row.
    rows = []
    hypotheses = {}
    for index, path in enumerate(paths):
        split = "test" if index == 5 else "train"
        row = {
            "schema_version": 1,
            "sample_id": f"sample-{index:03}",
            "dataset": "resume_fixture",
            "revision": "fixture-v1",
            "source_split": split,
            "split": split,
            "source_recording_id": f"recording-{index}",
            "group_id": f"original-recording:{index}",
            "audio_path": str(path.resolve()),
            "task": "asr",
            "language": "en",
            "question": "Transcribe the English speech.",
            "reference": "known transcript",
            "reference_aliases": [],
            "audio_license": "CC0-1.0",
            "annotation_license": "CC-BY-4.0",
            "annotation_type": "human",
            "reference_verified": True,
            "metadata": {},
        }
        rows.append(row)
        hypotheses[row["sample_id"]] = {
            "text": "known transcript",
            "language": "en",
            "audio_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "model": "fixture-independent-asr",
        }
    source = tmp_path / "source.jsonl"
    _write_manifest(source, rows)
    holdout = tmp_path / "holdout.jsonl"
    _write_manifest(holdout, [{"group_id": rows[1]["group_id"]}])
    return SimpleNamespace(
        source=source,
        output=tmp_path / "filtered.jsonl",
        checkpoint=tmp_path / "checkpoint",
        rows=rows,
        hypotheses=hypotheses,
        holdout=holdout,
    )


def _run(case, *, output=None, checkpoint=True, **policy_changes):
    policy = {
        "asr_hypotheses": case.hypotheses,
        "holdout_manifest": case.holdout,
        "max_wer": 0,
        "require_reference_check": True,
    }
    policy.update(policy_changes)
    return offline.filter_manifest(
        case.source,
        output or case.output,
        checkpoint_dir=case.checkpoint if checkpoint else None,
        checkpoint_interval=2,
        progress="off",
        **policy,
    )


def _interrupt_scan(case, monkeypatch, successful_rows, **policy_changes):
    original = offline._check_record
    checked_ids = []

    def interrupted(record, **policy):
        if len(checked_ids) == successful_rows:
            raise RuntimeError("simulated scan interruption")
        checked_ids.append(record["sample_id"])
        return original(record, **policy)

    with monkeypatch.context() as patch:
        patch.setattr(offline, "_check_record", interrupted)
        with pytest.raises(RuntimeError, match="simulated scan interruption"):
            _run(case, **policy_changes)
    return checked_ids


def test_resume_reuses_committed_rows_and_preserves_global_leakage_decisions(resume_case, monkeypatch):
    case = resume_case
    expected_path = case.output.with_name("uninterrupted.jsonl")
    _run(case, output=expected_path, checkpoint=False)
    expected = _read_manifest(expected_path)
    previous_output = b"previous published output\n"
    case.output.write_bytes(previous_output)

    assert _interrupt_scan(case, monkeypatch, successful_rows=4) == [row["sample_id"] for row in case.rows[:4]]
    assert case.output.read_bytes() == previous_output

    original = offline._check_record
    resumed_ids = []

    def counted(record, **policy):
        resumed_ids.append(record["sample_id"])
        return original(record, **policy)

    monkeypatch.setattr(offline, "_check_record", counted)
    _run(case)

    assert resumed_ids == [row["sample_id"] for row in case.rows[4:]]
    actual = _read_manifest(case.output)
    assert actual == expected
    assert "cross_split_pcm" in actual[0]["qc"]["offline"]["reasons"]
    assert "cross_split_pcm" in actual[-1]["qc"]["offline"]["reasons"]
    assert "cross_split_recording" in actual[1]["qc"]["offline"]["reasons"]


def test_final_pass_failure_preserves_output_and_resume_does_not_redecode(resume_case, monkeypatch):
    case = resume_case
    previous_output = b"previous published output\n"
    case.output.write_bytes(previous_output)
    original_finish = offline._finish
    finished = []

    def interrupted_finish(connection, record):
        if len(finished) == 2:
            raise RuntimeError("simulated final pass interruption")
        finished.append(record["sample_id"])
        return original_finish(connection, record)

    with monkeypatch.context() as patch:
        patch.setattr(offline, "_finish", interrupted_finish)
        with pytest.raises(RuntimeError, match="simulated final pass interruption"):
            _run(case)
    assert case.output.read_bytes() == previous_output

    def should_not_recheck(*args, **kwargs):
        raise AssertionError("committed media must not be decoded again")

    monkeypatch.setattr(offline, "_check_record", should_not_recheck)
    _run(case)
    assert len(_read_manifest(case.output)) == len(case.rows)


def test_partial_uncommitted_batch_is_recomputed_without_duplicate_rows(resume_case, monkeypatch):
    case = resume_case
    _interrupt_scan(case, monkeypatch, successful_rows=3)
    original = offline._check_record
    resumed_ids = []

    def counted(record, **policy):
        resumed_ids.append(record["sample_id"])
        return original(record, **policy)

    monkeypatch.setattr(offline, "_check_record", counted)
    _run(case)

    assert resumed_ids == [row["sample_id"] for row in case.rows[2:]]
    actual = _read_manifest(case.output)
    assert [row["sample_id"] for row in actual] == [row["sample_id"] for row in case.rows]


@pytest.mark.parametrize("changed", ["input", "hypotheses", "policy", "holdout", "media"])
def test_resume_rejects_changed_inputs_before_reusing_decisions(resume_case, monkeypatch, changed):
    case = resume_case
    previous_output = b"previous published output\n"
    case.output.write_bytes(previous_output)
    _interrupt_scan(case, monkeypatch, successful_rows=2)
    policy_changes = {}
    if changed == "input":
        edited = copy.deepcopy(case.rows)
        edited[0]["reference"] = "edited reference"
        _write_manifest(case.source, edited)
    elif changed == "hypotheses":
        edited = copy.deepcopy(case.hypotheses)
        edited[case.rows[0]["sample_id"]]["text"] = "edited hypothesis"
        policy_changes["asr_hypotheses"] = edited
    elif changed == "policy":
        policy_changes["max_wer"] = 0.1
    elif changed == "holdout":
        _write_manifest(case.holdout, [{"group_id": case.rows[1]["group_id"]}, {"group_id": "new-benchmark-recording"}])
    else:
        _write_wave(Path(case.rows[0]["audio_path"]), frequency=1600)

    def should_not_recheck(*args, **kwargs):
        raise AssertionError("resume compatibility must be checked before processing candidates")

    monkeypatch.setattr(offline, "_check_record", should_not_recheck)
    with pytest.raises(ValueError):
        _run(case, **policy_changes)
    assert case.output.read_bytes() == previous_output


def test_finished_checkpoint_still_rejects_changed_media(resume_case, monkeypatch):
    case = resume_case
    _run(case)
    published = case.output.read_bytes()
    _write_wave(Path(case.rows[0]["audio_path"]), frequency=1700)

    def should_not_recheck(*args, **kwargs):
        raise AssertionError("finished checkpoint must validate media before reuse")

    monkeypatch.setattr(offline, "_check_record", should_not_recheck)
    with pytest.raises(ValueError):
        _run(case)
    assert case.output.read_bytes() == published


def test_resume_rejects_retargeted_candidate_audio_symlink(resume_case, monkeypatch):
    case = resume_case
    original_target = Path(case.rows[0]["audio_path"])
    alternate_target = Path(case.rows[2]["audio_path"])
    link = case.source.parent / "candidate-symlink.wav"
    link.symlink_to(original_target)
    case.rows[0]["audio_path"] = str(link)
    _write_manifest(case.source, case.rows)
    case.output.write_bytes(b"previous published output\n")
    _interrupt_scan(case, monkeypatch, successful_rows=2)
    link.unlink()
    link.symlink_to(alternate_target)

    def should_not_recheck(*args, **kwargs):
        raise AssertionError("symlink identity must be checked before reusing candidates")

    monkeypatch.setattr(offline, "_check_record", should_not_recheck)
    with pytest.raises(ValueError):
        _run(case)
    assert case.output.read_bytes() == b"previous published output\n"


def test_finished_checkpoint_rejects_mutated_holdout_media(resume_case, monkeypatch):
    case = resume_case
    holdout_audio = _write_wave(case.source.parent / "benchmark.wav", 2000)
    _write_manifest(case.holdout, [{"group_id": case.rows[1]["group_id"]}, {"audio_path": str(holdout_audio)}])
    _run(case)
    published = case.output.read_bytes()
    _write_wave(holdout_audio, frequency=2200)

    def should_not_recheck(*args, **kwargs):
        raise AssertionError("holdout media must be checked before reusing candidate decisions")

    monkeypatch.setattr(offline, "_check_record", should_not_recheck)
    with pytest.raises(ValueError):
        _run(case)
    assert case.output.read_bytes() == published


def test_abrupt_process_exit_rolls_back_partial_batch_and_resumes(resume_case, monkeypatch):
    case = resume_case
    hypotheses_path = case.source.parent / "hypotheses.json"
    hypotheses_path.write_text(json.dumps(case.hypotheses), encoding="utf-8")
    script = r"""
import json
import os
import sys
from scripts.audio_data import filter_offline as offline

source, output, checkpoint, holdout, hypotheses = sys.argv[1:]
original = offline._check_record
checked = 0
def interrupted(record, **policy):
    global checked
    if checked == 3:
        os._exit(23)  # Bypass connection cleanup with the third row uncommitted.
    checked += 1
    return original(record, **policy)
offline._check_record = interrupted
with open(hypotheses, encoding="utf-8") as handle:
    predictions = json.load(handle)
offline.filter_manifest(
    source, output, checkpoint_dir=checkpoint, checkpoint_interval=2, progress="off",
    holdout_manifest=holdout, asr_hypotheses=predictions, max_wer=0, require_reference_check=True,
)
"""
    environment = dict(os.environ)
    root = str(Path(__file__).resolve().parents[2])
    environment["PYTHONPATH"] = os.pathsep.join(filter(None, [root, environment.get("PYTHONPATH")]))
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            *map(str, [case.source, case.output, case.checkpoint, case.holdout, hypotheses_path]),
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 23, result.stdout + result.stderr
    assert not case.output.exists()
    original = offline._check_record
    resumed_ids = []

    def counted(record, **policy):
        resumed_ids.append(record["sample_id"])
        return original(record, **policy)

    monkeypatch.setattr(offline, "_check_record", counted)
    _run(case)

    assert resumed_ids == [row["sample_id"] for row in case.rows[2:]]
    assert [row["sample_id"] for row in _read_manifest(case.output)] == [row["sample_id"] for row in case.rows]


def test_checkpoint_lock_blocks_reset_without_destroying_saved_work(resume_case, monkeypatch):
    fcntl = pytest.importorskip("fcntl")
    case = resume_case
    _interrupt_scan(case, monkeypatch, successful_rows=2)
    with (case.checkpoint / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="(?i)(another|using|lock)"):
            _run(case, reset_checkpoint=True)

    original = offline._check_record
    resumed_ids = []

    def counted(record, **policy):
        resumed_ids.append(record["sample_id"])
        return original(record, **policy)

    monkeypatch.setattr(offline, "_check_record", counted)
    _run(case)
    assert resumed_ids == [row["sample_id"] for row in case.rows[2:]]


def test_explicit_reset_restarts_changed_manifest_without_retaining_old_rows(resume_case, monkeypatch):
    case = resume_case
    _interrupt_scan(case, monkeypatch, successful_rows=2)
    case.rows[0]["question"] = "Transcribe this English recording exactly."
    _write_manifest(case.source, case.rows)
    original = offline._check_record
    checked_ids = []

    def counted(record, **policy):
        checked_ids.append(record["sample_id"])
        return original(record, **policy)

    monkeypatch.setattr(offline, "_check_record", counted)
    _run(case, reset_checkpoint=True)

    assert checked_ids == [row["sample_id"] for row in case.rows]
    actual = _read_manifest(case.output)
    assert len(actual) == len(case.rows)
    assert actual[0]["question"] == case.rows[0]["question"]


@pytest.mark.parametrize("quarantine", ["schema", "duration"])
def test_stale_incoming_audio_hash_cannot_poison_checkpoint_media_identity(resume_case, monkeypatch, quarantine):
    case = resume_case
    case.rows[0]["audio_sha256"] = "0" * 64
    policy = {}
    if quarantine == "schema":
        case.rows[0]["schema_version"] = 99
    else:
        policy["max_duration"] = 0.1
    _write_manifest(case.source, case.rows)
    _interrupt_scan(case, monkeypatch, successful_rows=2, **policy)
    original = offline._check_record
    resumed_ids = []

    def counted(record, **options):
        resumed_ids.append(record["sample_id"])
        return original(record, **options)

    monkeypatch.setattr(offline, "_check_record", counted)
    _run(case, **policy)

    assert resumed_ids == [row["sample_id"] for row in case.rows[2:]]
    assert _read_manifest(case.output)[0]["qc"]["offline"]["decision"] == "quarantine"

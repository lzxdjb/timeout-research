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
"""Qwen audit recovery contracts, with real media but no inference service."""

import copy
import fcntl
import hashlib
import json
import math
import os
import struct
import subprocess
import sys
import threading
import time
import wave
from types import SimpleNamespace

import pytest

from scripts.audio_data import filter_qwen as audit
from scripts.audio_data.common import read_jsonl, record_fingerprint, write_jsonl


def _write_wave(path, frequency):
    samples = [round(6000 * math.sin(2 * math.pi * frequency * index / 16000)) for index in range(1600)]
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(struct.pack(f"<{len(samples)}h", *samples))


@pytest.fixture
def audit_case(tmp_path, monkeypatch):
    rows = []
    for index in range(4):
        path = tmp_path / f"clip-{index}.wav"
        _write_wave(path, 400 + index * 100)
        record = {
            "sample_id": f"sample-{index}",
            "dataset": "audit_resume_fixture",
            "audio_path": str(path),
            "audio_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "task": "closed_qa",
            "language": "en",
            "question": "What sound is audible?",
            "reference": "a tone",
            "reference_aliases": [],
            "annotation_type": "human",
            "reference_verified": True,
            "metadata": {},
            "qc": {"offline": {"decision": "accept", "reasons": [], "metrics": {"duration_seconds": 0.1}}},
        }
        record["qc"]["offline"]["input_fingerprint"] = record_fingerprint(record)
        rows.append(record)
    source = tmp_path / "source.jsonl"
    write_jsonl(source, rows)
    state = SimpleNamespace(
        calls=[],
        completed=[],
        fail=set(),
        errors=set(),
        delays={},
        decisions={},
        after_review={},
        identity={"id": "fixture-model", "root": "fixture-model", "revision": "v1"},
        lock=threading.Lock(),
    )

    class FakeReviewer(audit.QwenReviewer):
        def __init__(self, config):
            self.config = config
            self.model = "fixture-model"
            self.identity = copy.deepcopy(state.identity)
            self.advertised_model_card = copy.deepcopy(self.identity)
            self.discovery_error = None
            self.service_error = None

        def review(self, record):
            sample_id = record["sample_id"]
            with state.lock:
                state.calls.append(sample_id)
            time.sleep(state.delays.get(sample_id, 0))
            if sample_id in state.fail:
                raise RuntimeError("simulated audit interruption")
            if sample_id in state.errors:
                return audit._quarantine(record, "review_failed:request_failed:TimeoutError", model=self.model)
            result = copy.deepcopy(record)
            decision = state.decisions.get(sample_id, "accept")
            result["qc"]["qwen"] = {
                "decision": decision,
                "reasons": [
                    "uncertain_reference_support_or_ambiguous_question"
                    if decision == "quarantine"
                    else "reference_supported_by_audio"
                    if decision == "accept"
                    else "reference_directly_contradicted_by_audio"
                ],
                "cache_hit": False,
                "request_config": self._request_config(),
                "audio_sha256": record["audio_sha256"],
                "input_fingerprint": record_fingerprint(record),
                "prompt_version": audit.PROMPT_VERSION,
            }
            callback = state.after_review.get(sample_id)
            if callback is not None:
                callback(record)
            with state.lock:
                state.completed.append(sample_id)
            return result

    monkeypatch.setattr(audit, "QwenReviewer", FakeReviewer)
    return SimpleNamespace(
        source=source,
        output=tmp_path / "reviewed.jsonl",
        report=tmp_path / "reviewed.report.json",
        checkpoint=tmp_path / "checkpoint",
        cache=tmp_path / "cache",
        rows=rows,
        state=state,
    )


def _run(case, *extra):
    audit.main(
        [
            "--input",
            str(case.source),
            "--output",
            str(case.output),
            "--report",
            str(case.report),
            "--cache-dir",
            str(case.cache),
            "--checkpoint-dir",
            str(case.checkpoint),
            "--checkpoint-interval",
            "1",
            "--concurrency",
            "1",
            "--retries",
            "0",
            "--max-samples",
            "4",
            "--progress",
            "off",
            *extra,
        ]
    )


def _report(case):
    return json.loads(case.report.read_text())


def test_completed_audit_rerun_avoids_reviews_and_audio_decoding(audit_case, monkeypatch):
    case = audit_case
    _run(case)
    expected = case.output.read_bytes()
    assert case.state.calls == [row["sample_id"] for row in case.rows]
    case.state.calls.clear()

    def unexpected_decode(*args, **kwargs):
        raise AssertionError("completed checkpoint must not decode audio")

    import soundfile

    monkeypatch.setattr(soundfile, "info", unexpected_decode)
    monkeypatch.setattr(soundfile, "read", unexpected_decode)
    _run(case)
    assert case.state.calls == []
    assert case.output.read_bytes() == expected
    assert _report(case)["records"] == 4
    assert _report(case)["selected"] == 4


def test_interrupted_audit_preserves_output_and_resumes_completed_records(audit_case):
    case = audit_case
    case.output.write_text("previous successful output\n")
    case.report.write_text("previous successful report\n")
    case.state.fail.add("sample-1")
    case.state.delays["sample-1"] = 0.08
    with pytest.raises(RuntimeError, match="simulated audit interruption"):
        _run(case)
    assert case.output.read_text() == "previous successful output\n"
    assert case.report.read_text() == "previous successful report\n"
    assert "sample-0" in case.state.completed
    case.state.calls.clear()
    case.state.fail.clear()
    _run(case)
    assert "sample-0" not in case.state.calls
    assert "sample-1" in case.state.calls
    assert [row["sample_id"] for row in read_jsonl(case.output)] == [row["sample_id"] for row in case.rows]
    assert _report(case)["review_errors"] == 0


def test_out_of_order_completion_is_durable_without_publishing_partial_output(audit_case):
    case = audit_case
    case.output.write_text("previous output\n")
    case.state.delays.update({"sample-0": 0.25, "sample-2": 0.08})
    case.state.fail.add("sample-2")
    with pytest.raises(RuntimeError, match="simulated audit interruption"):
        _run(case, "--concurrency", "2")
    assert case.state.completed[0] == "sample-1"
    assert case.output.read_text() == "previous output\n"
    case.state.calls.clear()
    case.state.fail.clear()
    _run(case, "--concurrency", "2")
    assert "sample-1" not in case.state.calls
    assert "sample-2" in case.state.calls
    assert [row["sample_id"] for row in read_jsonl(case.output)] == [row["sample_id"] for row in case.rows]


def test_request_failures_are_retried_while_uncertain_and_rejected_decisions_resume(audit_case):
    case = audit_case
    case.state.errors.add("sample-1")
    case.state.decisions.update({"sample-2": "quarantine", "sample-3": "reject"})
    _run(case)
    assert _report(case)["review_errors"] == 1
    assert _report(case)["qwen_decisions"] == {"accept": 1, "quarantine": 2, "reject": 1}
    case.state.calls.clear()
    case.state.errors.clear()
    _run(case)
    assert case.state.calls == ["sample-1"]
    report = _report(case)
    assert report["review_errors"] == 0
    assert report["qwen_decisions"] == {"accept": 2, "quarantine": 1, "reject": 1}


@pytest.mark.parametrize(
    "change", ["manifest", "audio", "model_identity", "model_revision", "prompt", "tokens", "duration", "budget"]
)
def test_incompatible_checkpoint_is_rejected_before_review_and_preserves_output(audit_case, monkeypatch, change):
    case = audit_case
    _run(case)
    expected = case.output.read_bytes()
    case.state.calls.clear()
    extra = []
    if change == "manifest":
        case.source.write_text(case.source.read_text() + "\n")
    elif change == "audio":
        path = case.source.parent / "clip-0.wav"
        previous_stat = path.stat()
        contents = bytearray(path.read_bytes())
        contents[-1] ^= 1
        path.write_bytes(contents)
        os.utime(path, ns=(previous_stat.st_atime_ns, previous_stat.st_mtime_ns))
    elif change == "model_identity":
        case.state.identity["revision"] = "v2"
    elif change == "model_revision":
        extra = ["--model-revision", "immutable-v2"]
    elif change == "prompt":
        monkeypatch.setattr(audit, "SUPPORT_PROMPT", audit.SUPPORT_PROMPT + "\nnew rubric")
    elif change == "tokens":
        extra = ["--max-tokens", "512"]
    elif change == "duration":
        extra = ["--max-duration", "30"]
    elif change == "budget":
        extra = ["--max-samples", "3"]
    with pytest.raises(ValueError):
        _run(case, *extra)
    assert case.state.calls == []
    assert case.output.read_bytes() == expected


def test_execution_controls_can_change_without_invalidating_checkpoint(audit_case, capsys):
    case = audit_case
    _run(case)
    expected = case.output.read_bytes()
    case.state.calls.clear()
    _run(
        case,
        "--concurrency",
        "3",
        "--timeout",
        "10",
        "--retries",
        "1",
        "--checkpoint-interval",
        "2",
        "--progress",
        "on",
        "--progress-interval",
        "0.01",
    )
    assert case.state.calls == []
    assert case.output.read_bytes() == expected
    assert "100.0%" in capsys.readouterr().err


def test_explicit_reset_allows_a_new_budget(audit_case):
    case = audit_case
    _run(case, "--max-samples", "2")
    assert _report(case)["budget_skipped"] == 2
    case.state.calls.clear()
    _run(case, "--max-samples", "4", "--reset-checkpoint")
    assert set(case.state.calls) == {row["sample_id"] for row in case.rows}
    assert _report(case)["budget_skipped"] == 0


def test_progress_heartbeats_while_requests_are_waiting(audit_case, capsys):
    case = audit_case
    case.state.delays["sample-0"] = 0.25
    _run(case, "--progress", "on", "--progress-interval", "0.03")
    captured = capsys.readouterr()
    assert "100.0%" in captured.err
    assert "elapsed" in captured.err
    # Initial/final renders alone cannot satisfy this: waiting must emit updates.
    assert len([line for line in captured.err.splitlines() if "0/4" in line and "elapsed" in line]) >= 3
    assert json.loads(captured.out.splitlines()[-1])["records"] == 4


def test_abrupt_process_exit_resumes_last_durable_review(audit_case):
    case = audit_case
    script = r"""
import copy
import os
import time
from scripts.audio_data import filter_qwen as audit
from scripts.audio_data.common import record_fingerprint

class FakeReviewer(audit.QwenReviewer):
    def __init__(self, config):
        self.config = config
        self.model = "fixture-model"
        self.identity = {"id": "fixture-model", "root": "fixture-model", "revision": "v1"}
        self.advertised_model_card = copy.deepcopy(self.identity)
        self.discovery_error = None
        self.service_error = None

    def review(self, record):
        if record["sample_id"] == "sample-1":
            time.sleep(0.3)
            os._exit(77)
        result = copy.deepcopy(record)
        result["qc"]["qwen"] = {
            "decision": "accept",
            "reasons": ["reference_supported_by_audio"],
            "cache_hit": False,
            "request_config": self._request_config(),
            "audio_sha256": record["audio_sha256"],
            "input_fingerprint": record_fingerprint(record),
            "prompt_version": audit.PROMPT_VERSION,
        }
        return result

audit.QwenReviewer = FakeReviewer
audit.main()
"""
    command = [
        sys.executable,
        "-c",
        script,
        "--input",
        str(case.source),
        "--output",
        str(case.output),
        "--report",
        str(case.report),
        "--cache-dir",
        str(case.cache),
        "--checkpoint-dir",
        str(case.checkpoint),
        "--checkpoint-interval",
        "1",
        "--concurrency",
        "1",
        "--retries",
        "0",
        "--max-samples",
        "4",
        "--progress",
        "off",
    ]
    interrupted = subprocess.run(command, capture_output=True, text=True, timeout=15)
    assert interrupted.returncode == 77, interrupted.stderr
    assert not case.output.exists()
    _run(case)
    assert "sample-0" not in case.state.calls
    assert set(case.state.calls) == {"sample-1", "sample-2", "sample-3"}
    assert _report(case)["records"] == 4


@pytest.mark.parametrize("reset", [False, True])
def test_checkpoint_lock_rejects_concurrent_run_before_work_or_reset(audit_case, reset):
    case = audit_case
    _run(case)
    expected = case.output.read_bytes()
    status = (case.checkpoint / "status.json").read_bytes()
    case.state.calls.clear()
    with (case.checkpoint / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="checkpoint|Checkpoint"):
            _run(case, *(["--reset-checkpoint"] if reset else []))
    assert case.state.calls == []
    assert case.output.read_bytes() == expected
    assert (case.checkpoint / "status.json").read_bytes() == status


def test_uncommitted_checkpoint_batch_may_repeat_after_interruption(audit_case):
    case = audit_case
    case.state.fail.add("sample-2")
    case.state.delays["sample-2"] = 0.08
    with pytest.raises(RuntimeError, match="simulated audit interruption"):
        _run(case, "--checkpoint-interval", "3")
    assert {"sample-0", "sample-1"}.issubset(case.state.completed)
    case.state.calls.clear()
    case.state.fail.clear()
    _run(case, "--checkpoint-interval", "3")
    assert set(case.state.calls) == {row["sample_id"] for row in case.rows}
    assert _report(case)["records"] == 4


def test_zero_review_budget_skips_discovery_and_preserves_offline_rejections(audit_case, monkeypatch):
    case = audit_case
    case.rows[-1]["qc"]["offline"]["decision"] = "reject"
    write_jsonl(case.source, case.rows)

    def unexpected_discovery(config):
        raise AssertionError("zero review budget must not discover an audit model")

    monkeypatch.setattr(audit, "QwenReviewer", unexpected_discovery)
    _run(case, "--max-samples", "0")
    results = list(read_jsonl(case.output))
    assert [row["sample_id"] for row in results] == [row["sample_id"] for row in case.rows]
    for result in results[:-1]:
        assert result["qc"]["qwen"]["decision"] == "quarantine"
        assert result["qc"]["qwen"]["reasons"] == ["qwen_sample_limit_not_reviewed"]
    assert results[-1] == case.rows[-1]
    report = _report(case)
    assert report["selected"] == 0
    assert report["budget_skipped"] == 3
    assert report["offline_skipped"] == 1
    assert report["model_config"] is None
    assert case.state.calls == []


def test_audio_changed_during_review_is_not_reusable_and_preserves_public_files(audit_case):
    case = audit_case
    audio_path = case.source.parent / "clip-0.wav"
    original_audio = audio_path.read_bytes()
    case.output.write_text("previous successful output\n")
    case.report.write_text("previous successful report\n")

    def mutate_audio(record):
        changed = bytearray(original_audio)
        changed[-1] ^= 1
        audio_path.write_bytes(changed)

    case.state.after_review["sample-0"] = mutate_audio
    with pytest.raises(ValueError, match="[Aa]udio"):
        _run(case)
    assert case.output.read_text() == "previous successful output\n"
    assert case.report.read_text() == "previous successful report\n"
    # Restoring the original offline-audited bytes permits retry. A result from
    # the mutation window must not have become a reusable checkpoint entry.
    audio_path.write_bytes(original_audio)
    case.state.after_review.clear()
    case.state.calls.clear()
    _run(case)
    assert "sample-0" in case.state.calls
    assert _report(case)["review_errors"] == 0
    assert _report(case)["records"] == 4

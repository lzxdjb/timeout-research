"""ASR log heartbeats and bounded concurrency without calling a model service."""

import json
import re
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from scripts.audio_data import transcribe_qwen as qwen_asr


class _LogStream:
    def __init__(self, on_write=None):
        self.lines = []
        self.on_write = on_write

    def isatty(self):
        return False

    def write(self, value):
        self.lines.append(value)
        if self.on_write is not None:
            self.on_write(value)
        return len(value)

    def flush(self):
        pass


def _counter(line, name):
    match = re.search(rf"\b{re.escape(name)}\s*[=:]\s*(\d+)", line)
    return int(match[1]) if match else None


def _setup(tmp_path, monkeypatch, transcribe, *, size=1):
    source, destination = tmp_path / "offline.jsonl", tmp_path / "hypotheses.jsonl"
    records = [
        {
            "sample_id": f"speech-{index}",
            "task": "asr",
            "split": "train",
            "qc": {"offline": {"decision": "accept"}},
        }
        for index in range(size)
    ]
    source.write_text("".join(json.dumps(record) + "\n" for record in records))
    producer = SimpleNamespace(
        transcribe=transcribe,
        _discovered=False,
        discovery_error=None,
        service_error=None,
    )
    monkeypatch.setattr(qwen_asr, "QwenASRProducer", lambda config: producer)
    arguments = [
        "--input",
        str(source),
        "--output",
        str(destination),
        "--max-samples",
        str(size),
        "--concurrency",
        "2",
        "--progress-interval",
        "0.01",
    ]
    return arguments, destination


def _hypothesis(record, *, cache_hit=False):
    return {"sample_id": record["sample_id"], "text": "test transcript", "cache_hit": cache_hit}


def test_redirected_progress_heartbeats_before_first_request_finishes(tmp_path, monkeypatch, capsys):
    started, release, finished = threading.Event(), threading.Event(), threading.Event()
    waiting_heartbeats = []

    def transcribe(record):
        started.set()
        release.wait(1.5)
        finished.set()
        return _hypothesis(record)

    def observe(line):
        if started.is_set() and not finished.is_set() and _counter(line, "completed") == 0:
            waiting_heartbeats.append(line)
            release.set()

    arguments, destination = _setup(tmp_path, monkeypatch, transcribe)
    stream = _LogStream(observe)
    monkeypatch.setattr(sys, "stderr", stream)
    qwen_asr.main([*arguments, "--progress", "on"])

    assert waiting_heartbeats, "A quiet, in-flight request must still produce periodic log updates"
    assert len(stream.lines) >= 3, "Initial, waiting, and final progress must all be visible"
    assert _counter(stream.lines[0], "completed") == 0
    assert _counter(stream.lines[-1], "completed") == 1
    assert _counter(stream.lines[-1], "transcribed") == 1
    assert json.loads(capsys.readouterr().out)["transcribed"] == 1
    assert json.loads(destination.read_text())["sample_id"] == "speech-0"


def test_fast_completion_is_reported_while_earlier_request_is_waiting(tmp_path, monkeypatch, capsys):
    slow_started, release_slow, slow_finished = threading.Event(), threading.Event(), threading.Event()
    observed = []

    def transcribe(record):
        if record["sample_id"] == "speech-0":
            slow_started.set()
            release_slow.wait(1.5)
            slow_finished.set()
            return _hypothesis(record)
        assert slow_started.wait(1)
        return _hypothesis(record, cache_hit=True)

    def observe(line):
        if _counter(line, "completed") == 1 and not slow_finished.is_set():
            observed.append(line)
            release_slow.set()

    arguments, destination = _setup(tmp_path, monkeypatch, transcribe, size=2)
    monkeypatch.setattr(sys, "stderr", _LogStream(observe))
    qwen_asr.main([*arguments, "--progress", "on"])

    assert observed, "An earlier slow request must not hide a later completed request"
    assert any(_counter(line, "transcribed") == 1 and _counter(line, "cache_hits") == 1 for line in observed)
    assert len(destination.read_text().splitlines()) == 2
    report = json.loads(destination.with_suffix(".report.json").read_text())
    assert report["attempted"] == 2 and report["transcribed"] == 2 and report["cache_hits"] == 1
    assert report["concurrency"] == 2 and report["max_in_flight"] == 4
    assert json.loads(capsys.readouterr().out)["errors"] == 0


@pytest.mark.parametrize("mode", ["off", "auto"])
def test_disabled_or_redirected_auto_progress_leaves_stderr_silent(tmp_path, monkeypatch, capsys, mode):
    arguments, destination = _setup(tmp_path, monkeypatch, _hypothesis)
    stream = _LogStream()
    monkeypatch.setattr(sys, "stderr", stream)
    qwen_asr.main([*arguments, "--progress", mode])
    assert stream.lines == []
    assert json.loads(capsys.readouterr().out)["transcribed"] == 1
    assert destination.exists()


def test_progress_keeps_worker_and_pending_queue_limits(tmp_path, monkeypatch, capsys):
    state = {"active": 0, "peak_active": 0, "read": 0, "written": 0, "peak_pending": 0}
    lock = threading.Lock()

    def transcribe(record):
        with lock:
            state["active"] += 1
            state["peak_active"] = max(state["peak_active"], state["active"])
        try:
            time.sleep(0.01)
            return _hypothesis(record)
        finally:
            with lock:
                state["active"] -= 1

    arguments, destination = _setup(tmp_path, monkeypatch, transcribe, size=17)
    original_read, original_write = qwen_asr.read_jsonl, qwen_asr.write_jsonl

    def observed_read(path):
        for record in original_read(path):
            state["read"] += 1
            state["peak_pending"] = max(state["peak_pending"], state["read"] - state["written"])
            assert state["read"] - state["written"] <= 4
            yield record

    def observed_write(path, records):
        def observed():
            for record in records:
                state["written"] += 1
                yield record

        original_write(path, observed())

    monkeypatch.setattr(qwen_asr, "read_jsonl", observed_read)
    monkeypatch.setattr(qwen_asr, "write_jsonl", observed_write)
    stream = _LogStream()
    monkeypatch.setattr(sys, "stderr", stream)
    qwen_asr.main([*arguments, "--progress", "on"])

    assert state["peak_active"] == 2
    assert state["peak_pending"] == 4
    assert state["written"] == 17 and len(destination.read_text().splitlines()) == 17
    assert json.loads(capsys.readouterr().out)["transcribed"] == 17
    assert _counter(stream.lines[-1], "pending") == 0


@pytest.mark.parametrize("interval", ["0", "-1", "nan", "inf"])
def test_progress_interval_must_be_positive_and_finite(tmp_path, monkeypatch, interval):
    arguments, _ = _setup(tmp_path, monkeypatch, _hypothesis)
    with pytest.raises(SystemExit) as error:
        qwen_asr.main([*arguments, "--progress-interval", interval])
    assert error.value.code == 2

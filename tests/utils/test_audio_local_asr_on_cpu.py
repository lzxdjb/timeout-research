"""Verify local-only ASR contracts using an injected model and real WAV bytes."""

import copy
import hashlib
import io
import json
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from scripts.audio_data import transcribe_local as asr_module
from scripts.audio_data.common import record_fingerprint
from scripts.audio_data.transcribe_local import (
    LocalASRConfig,
    LocalASRProducer,
    TranscriptionError,
    main,
    snapshot_identity,
)


@pytest.fixture
def snapshot(tmp_path):
    path = tmp_path / "whisper-local"
    path.mkdir()
    for name, data in (("model.bin", b"fake model weights"), ("config.json", b"{}"), ("tokenizer.json", b"{}")):
        (path / name).write_bytes(data)
    return path


@pytest.fixture
def record(tmp_path):
    path = tmp_path / "speech.wav"
    sf.write(path, np.sin(np.arange(3200) / 10) * 0.2, 16000, subtype="PCM_16")
    record = {
        "sample_id": "speech-1",
        "task": "asr",
        "language": "en",
        "question": "Transcribe the speech.",
        "reference": "hello world",
        "audio_path": str(path),
        "audio_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "qc": {"offline": {"decision": "accept", "reasons": [], "metrics": {}}},
    }
    _bind(record)
    return record


def _bind(record):
    record["qc"]["offline"]["input_fingerprint"] = record_fingerprint(record)


@pytest.fixture
def fake_engine(monkeypatch):
    state = {"initializations": [], "calls": [], "mode": "normal"}

    class WhisperModel:
        def __init__(self, model, **kwargs):
            state["initializations"].append((model, kwargs))

        def transcribe(self, audio, **kwargs):
            raw = audio.read()
            decoded, rate = sf.read(io.BytesIO(raw))
            assert rate == 16000 and len(decoded) == 3200
            state["calls"].append((raw, kwargs))
            info = SimpleNamespace(language=kwargs["language"], language_probability=0.99)
            if state["mode"] == "empty":
                return iter([]), info
            if state["mode"] == "fail":

                def failed_segments():
                    yield SimpleNamespace(text="partial")
                    raise RuntimeError("mock lazy decoder failure")

                return failed_segments(), info
            text = ["你好", "世界"] if kwargs["language"] == "zh" else ["hello", "world"]
            return (SimpleNamespace(text=piece) for piece in text), info

    module = ModuleType("faster_whisper")
    module.WhisperModel = WhisperModel
    module.__version__ = "mock-local-only"
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    return state


def test_local_only_model_receives_exact_verified_audio_and_native_language(snapshot, record, fake_engine):
    producer = LocalASRProducer(LocalASRConfig(model=snapshot))
    assert fake_engine["initializations"] == []  # Model loading is lazy.
    english = producer.transcribe(record)
    assert english["text"] == "hello world"
    assert english["audio_sha256"] == record["audio_sha256"]
    assert english["language"] == "en"
    assert english["model_identity"]["kind"] == "local_snapshot_sha256"
    assert len(english["model_identity"]["content_sha256"]) == 64
    assert fake_engine["initializations"] == [
        (str(snapshot), {"device": "cpu", "compute_type": "int8", "local_files_only": True})
    ]
    assert hashlib.sha256(fake_engine["calls"][0][0]).hexdigest() == record["audio_sha256"]
    chinese = copy.deepcopy(record)
    chinese.update(sample_id="speech-zh", language="zh-CN", reference="你好世界")
    _bind(chinese)
    result = producer.transcribe(chinese)
    assert result["language"] == "zh" and result["text"] == "你好世界"
    assert fake_engine["calls"][1][1]["language"] == "zh"
    assert fake_engine["calls"][1][1]["task"] == "transcribe"
    assert len(fake_engine["initializations"]) == 1


def test_stale_semantics_or_changed_media_never_reaches_model(snapshot, record, fake_engine):
    producer = LocalASRProducer(LocalASRConfig(model=snapshot))
    stale = copy.deepcopy(record)
    stale["reference"] = "edited without offline review"
    with pytest.raises(TranscriptionError, match="record_changed_since_offline_review"):
        producer.transcribe(stale)
    with open(record["audio_path"], "ab") as stream:
        stream.write(b"changed")
    with pytest.raises(TranscriptionError, match="audio_changed_since_offline_review"):
        producer.transcribe(record)
    assert fake_engine["initializations"] == []
    assert fake_engine["calls"] == []


def test_repeat_media_cache_is_language_specific_and_failures_are_retried(snapshot, record, fake_engine):
    producer = LocalASRProducer(LocalASRConfig(model=snapshot, model_revision="immutable-test-checkpoint"))
    fake_engine["mode"] = "fail"
    with pytest.raises(TranscriptionError, match="local_transcription_failed:RuntimeError"):
        producer.transcribe(record)
    fake_engine["mode"] = "normal"
    assert producer.transcribe(record)["text"] == "hello world"
    repeated = copy.deepcopy(record)
    repeated["sample_id"] = "same-media-different-prompt"
    _bind(repeated)
    result = producer.transcribe(repeated)
    assert result["sample_id"] == repeated["sample_id"]
    assert result["model_identity"]["revision"] == "immutable-test-checkpoint"
    assert len(fake_engine["calls"]) == 2


def test_snapshot_hash_tracks_content_and_incomplete_local_tokenizer_is_rejected(snapshot):
    first = snapshot_identity(snapshot, None)
    (snapshot / "config.json").write_text('{"changed": true}')
    assert snapshot_identity(snapshot, None)["content_sha256"] != first["content_sha256"]
    (snapshot / "tokenizer.json").unlink()
    with pytest.raises(ValueError, match="missing tokenizer.json"):
        LocalASRProducer(LocalASRConfig(model=snapshot))
    with pytest.raises(ValueError, match="existing local"):
        LocalASRProducer(LocalASRConfig(model=snapshot / "nonexistent-model"))


def test_cli_reports_failures_separately_without_fake_transcripts(tmp_path, snapshot, record, fake_engine, capsys):
    input_path = tmp_path / "offline.jsonl"
    output_path = tmp_path / "hypotheses.jsonl"
    input_path.write_text(json.dumps(record) + "\n")
    fake_engine["mode"] = "empty"
    main(["--input", str(input_path), "--output", str(output_path), "--model", str(snapshot)])
    assert output_path.read_text() == ""
    errors = [json.loads(line) for line in (tmp_path / "hypotheses.errors.jsonl").read_text().splitlines()]
    assert errors[0]["sample_id"] == record["sample_id"] and errors[0]["reason"] == "empty_transcription"
    report = json.loads(capsys.readouterr().out)
    assert report["errors"] == 1


def test_limits_and_offline_ineligible_records_do_not_start_inference(snapshot, record, fake_engine, tmp_path):
    producer = LocalASRProducer(LocalASRConfig(model=snapshot, max_duration=0.01))
    with pytest.raises(TranscriptionError, match="audio_duration_outside_asr_limit"):
        producer.transcribe(record)
    rejected = copy.deepcopy(record)
    rejected["qc"]["offline"]["decision"] = "reject"
    assert producer.transcribe(rejected) is None
    other_task = copy.deepcopy(record)
    other_task["task"] = "closed_qa"
    assert producer.transcribe(other_task) is None
    input_path, output_path = tmp_path / "offline.jsonl", tmp_path / "none.jsonl"
    input_path.write_text(json.dumps(record) + "\n")
    main(["--input", str(input_path), "--output", str(output_path), "--model", str(snapshot), "--max-samples", "0"])
    assert output_path.read_text() == ""
    assert fake_engine["initializations"] == []


def test_repeated_media_memory_cache_is_bounded(snapshot, record, fake_engine, monkeypatch):
    monkeypatch.setattr(asr_module, "MAX_CACHED_PREDICTIONS", 2)
    producer = LocalASRProducer(LocalASRConfig(model=snapshot))
    originals = []
    for index in range(3):
        sf.write(record["audio_path"], np.sin(np.arange(3200) / (10 + index)) * 0.2, 16000, subtype="PCM_16")
        with open(record["audio_path"], "rb") as stream:
            raw = stream.read()
        item = copy.deepcopy(record)
        item["audio_sha256"] = hashlib.sha256(raw).hexdigest()
        _bind(item)
        originals.append((item, raw))
        producer.transcribe(item)
    assert len(producer._predictions) == 2
    first, raw = originals[0]
    with open(first["audio_path"], "wb") as stream:
        stream.write(raw)
    producer.transcribe(first)
    assert len(fake_engine["calls"]) == 4  # The oldest cached prediction was evicted.
    assert len(producer._predictions) == 2

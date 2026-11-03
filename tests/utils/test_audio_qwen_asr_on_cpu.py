"""Reference-blind Qwen ASR boundary tests with actual media and a local mock server."""

import base64
import copy
import hashlib
import io
import json
import struct
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from scripts.audio_data import transcribe_qwen as qwen_asr
from scripts.audio_data.common import record_fingerprint
from scripts.audio_data.filter_qwen import ReviewError
from scripts.audio_data.transcribe_qwen import QwenASRConfig, QwenASRProducer, main


@contextmanager
def _server():
    state = {
        "requests": [],
        "models_requests": 0,
        "content": None,
        "content_sequence": [],
        "finish_reason": "stop",
        "http_status": 200,
        "model_card": {
            "id": "swe-repeat-judge",
            "root": "Qwen3-Omni-30B-A3B-Instruct",
            "owned_by": "vllm",
            "max_model_len": 65536,
        },
    }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send_json(self, status, value):
            body = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            assert self.path == "/v1/models"
            state["models_requests"] += 1
            card = copy.deepcopy(state["model_card"])
            card.update(created=state["models_requests"], permission=[{"id": f"perm-{state['models_requests']}"}])
            self.send_json(200, {"data": [card]})

        def do_POST(self):
            assert self.path == "/v1/chat/completions"
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["requests"].append(request)
            if state["http_status"] != 200:
                self.send_json(
                    state["http_status"],
                    {
                        "error": {
                            "message": (
                                "At most 0 audio(s) may be provided in one prompt. (parameter=audio) SECRET_SERVER_BODY"
                            )
                        }
                    },
                )
                return
            instruction = request["messages"][1]["content"][0]["text"]
            text = "你好世界" if "Mandarin" in instruction else "hello world"
            content = state["content"] if state["content"] is not None else json.dumps({"text": text})
            if state["content_sequence"]:
                content = state["content_sequence"].pop(0)
            self.send_json(
                200,
                {
                    "id": "mock-asr",
                    "model": "Qwen3-Omni",
                    "choices": [{"finish_reason": state["finish_reason"], "message": {"content": content}}],
                },
            )

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", state
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.fixture
def record(tmp_path):
    path = tmp_path / "speech.wav"
    sf.write(path, 0.2 * np.sin(np.arange(1600) / 10), 16000, subtype="PCM_16")
    record = {
        "sample_id": "fixture-speech",
        "task": "asr",
        "split": "train",
        "language": "en",
        "audio_path": str(path),
        "audio_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "question": "SECRET_ORIGINAL_QUESTION",
        "reference": "SECRET_ORIGINAL_REFERENCE",
        "reference_aliases": ["SECRET_REFERENCE_ALIAS"],
        "metadata": {"transcript": "SECRET_METADATA_TRANSCRIPT"},
        "qc": {"offline": {"decision": "accept", "reasons": [], "metrics": {}}},
    }
    _bind(record)
    return record


def _bind(record):
    record["qc"]["offline"]["input_fingerprint"] = record_fingerprint(record)


def _producer(url, tmp_path, **kwargs):
    return QwenASRProducer(QwenASRConfig(base_url=url, cache_dir=tmp_path / "cache", retries=0, **kwargs))


def test_actual_audio_is_transcribed_without_original_question_or_reference(tmp_path, record):
    with _server() as (url, state):
        producer = _producer(url, tmp_path)
        result = producer.transcribe(record)
    assert result["text"] == "hello world" and result["audio_sha256"] == record["audio_sha256"]
    assert result["model"] == "swe-repeat-judge" and result["producer"] == "qwen-asr"
    assert result["input_fingerprint"] == record_fingerprint(record)
    assert len(state["requests"]) == 1
    request = state["requests"][0]
    assert request["temperature"] == 0
    assert "SECRET_" not in json.dumps(request)
    payload = request["messages"][1]["content"][1]["input_audio"]
    raw = base64.b64decode(payload["data"], validate=True)
    assert raw == (tmp_path / "speech.wav").read_bytes()
    decoded, rate = sf.read(io.BytesIO(raw))
    assert rate == 16000 and len(decoded) == 1600
    assert "reference" not in result and "qwen" not in record["qc"]


def test_cache_reuses_audio_after_label_edit_but_binds_language_model_and_media(tmp_path, record):
    with _server() as (url, state):
        producer = _producer(url, tmp_path)
        assert producer.transcribe(record)["cache_hit"] is False
        edited = copy.deepcopy(record)
        edited.update(sample_id="new-prompt-same-audio", reference="edited gold", question="edited task")
        with pytest.raises(ReviewError, match="record_changed_since_offline_review"):
            producer.transcribe(edited)
        _bind(edited)
        cached = producer.transcribe(edited)
        assert cached["cache_hit"] is True and cached["sample_id"] == edited["sample_id"]
        assert cached["input_fingerprint"] == record_fingerprint(edited)
        assert len(state["requests"]) == 1
        chinese = copy.deepcopy(edited)
        chinese["language"] = "zh-CN"
        _bind(chinese)
        assert producer.transcribe(chinese)["text"] == "你好世界"
        assert len(state["requests"]) == 2
        assert _producer(url, tmp_path, model_revision="immutable-v2").transcribe(edited)["cache_hit"] is False
        assert len(state["requests"]) == 3
        sf.write(record["audio_path"], np.ones(1600) * 0.1, 16000)
        with pytest.raises(ReviewError, match="audio_changed_since_offline_review"):
            producer.transcribe(record)
        replacement = copy.deepcopy(record)
        replacement["audio_sha256"] = hashlib.sha256((tmp_path / "speech.wav").read_bytes()).hexdigest()
        _bind(replacement)
        assert producer.transcribe(replacement)["cache_hit"] is False
        assert len(state["requests"]) == 4


def test_fresh_asr_producers_reuse_cache_despite_ephemeral_model_card_fields(tmp_path, record):
    with _server() as (url, state):
        first = _producer(url, tmp_path)
        assert first.transcribe(record)["cache_hit"] is False
        fresh = _producer(url, tmp_path)
        assert fresh.transcribe(record)["cache_hit"] is True
        assert first.advertised_model_card != fresh.advertised_model_card
        assert first.identity == fresh.identity and len(state["requests"]) == 1
        state["model_card"]["root"] = "Qwen3-Omni-new-checkpoint"
        assert _producer(url, tmp_path).transcribe(record)["cache_hit"] is False
        assert len(state["requests"]) == 2
        assert (
            _producer(url, tmp_path, model_revision="immutable-deployment-two").transcribe(record)["cache_hit"] is False
        )
        assert len(state["requests"]) == 3


def test_flac_cache_ignores_transport_peak_timestamps_but_binds_original_media(tmp_path, record, monkeypatch):
    path = tmp_path / "speech.flac"
    waveform = (0.2 * np.sin(np.arange(1600) / 10)).astype("float32")
    sf.write(path, waveform, 16000, subtype="PCM_24")
    record["audio_path"] = str(path)
    record["audio_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    _bind(record)
    original_prepare = QwenASRProducer._prepare_audio
    serialized = []

    def prepare_with_different_peak_timestamp(self, item):
        audio, digest, media = original_prepare(self, item)
        raw = bytearray(base64.b64decode(audio["data"]))
        assert raw[:4] == b"RIFF" and raw[8:12] == b"WAVE"
        timestamp = len(serialized) + 1
        offset, found = 12, False
        while offset + 8 <= len(raw):
            chunk_size = struct.unpack_from("<I", raw, offset + 4)[0]
            if raw[offset : offset + 4] == b"PEAK":
                assert chunk_size >= 8
                struct.pack_into("<I", raw, offset + 12, timestamp)
                found = True
                break
            offset += 8 + chunk_size + (chunk_size % 2)
        if not found:
            # Some libsndfile versions omit PEAK; add a valid ancillary chunk
            # so this regression still exercises time-varying container bytes.
            peak = struct.pack("<II", 1, timestamp) + struct.pack("<fI", 0.2, 0) * media["channels"]
            raw.extend(b"PEAK" + struct.pack("<I", len(peak)) + peak)
            struct.pack_into("<I", raw, 4, len(raw) - 8)
        serialized.append(bytes(raw))
        audio["data"] = base64.b64encode(raw).decode("ascii")
        return audio, digest, media

    monkeypatch.setattr(QwenASRProducer, "_prepare_audio", prepare_with_different_peak_timestamp)
    with _server() as (url, state):
        first = _producer(url, tmp_path).transcribe(record)
        fresh = _producer(url, tmp_path).transcribe(record)
        assert first["cache_hit"] is False and fresh["cache_hit"] is True
        assert first["cache_key"] == fresh["cache_key"] and len(state["requests"]) == 1
        assert hashlib.sha256(serialized[0]).digest() != hashlib.sha256(serialized[1]).digest()
        first_pcm, first_rate = sf.read(io.BytesIO(serialized[0]))
        fresh_pcm, fresh_rate = sf.read(io.BytesIO(serialized[1]))
        assert first_rate == fresh_rate == 16000
        np.testing.assert_array_equal(first_pcm, fresh_pcm)
        sf.write(path, waveform * 0.5, 16000, subtype="PCM_24")
        changed = copy.deepcopy(record)
        changed["audio_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        _bind(changed)
        assert _producer(url, tmp_path).transcribe(changed)["cache_hit"] is False
        assert len(state["requests"]) == 2


@pytest.mark.parametrize(
    "content,finish",
    [
        ('{"text": ""}', "stop"),
        ('{"text": 3}', "stop"),
        ("plain text without required JSON", "stop"),
        ('{"text": "partial"}', "length"),
    ],
)
def test_invalid_empty_or_truncated_results_are_errors_not_cached(tmp_path, record, content, finish):
    with _server() as (url, state):
        state.update(content=content, finish_reason=finish)
        producer = _producer(url, tmp_path)
        with pytest.raises(ReviewError):
            producer.transcribe(record)
        assert not list((tmp_path / "cache").rglob("*.json"))
        state.update(content='```json\n{"text": "valid transcript"}\n```', finish_reason="stop")
        assert producer.transcribe(record)["text"] == "valid transcript"
        assert len(state["requests"]) == 2


def test_malformed_first_completion_is_retried_and_only_valid_transcript_is_cached(tmp_path, record):
    with _server() as (url, state):
        state["content_sequence"] = ["transient malformed response", '\n{"text": "valid recovered transcript"}\n']
        producer = QwenASRProducer(QwenASRConfig(base_url=url, cache_dir=tmp_path / "cache", retries=1))
        result = producer.transcribe(record)
        assert result["text"] == "valid recovered transcript" and result["cache_hit"] is False
        assert len(state["requests"]) == 2
        paths = list((tmp_path / "cache").rglob("*.json"))
        assert len(paths) == 1
        cached = json.loads(paths[0].read_text())
        assert cached["prediction"]["text"] == "valid recovered transcript"
        assert "transient malformed response" not in paths[0].read_text()
        assert producer.transcribe(record)["cache_hit"] is True
        assert len(state["requests"]) == 2


def test_stale_media_and_fingerprint_fail_before_discovery_and_audio_disabled_is_safe(tmp_path, record):
    with _server() as (url, state):
        producer = _producer(url, tmp_path)
        stale = copy.deepcopy(record)
        stale["question"] = "changed"
        with pytest.raises(ReviewError, match="record_changed_since_offline_review"):
            producer.transcribe(stale)
        changed_hash = copy.deepcopy(record)
        changed_hash["audio_sha256"] = "0" * 64
        _bind(changed_hash)
        with pytest.raises(ReviewError, match="audio_changed_since_offline_review"):
            producer.transcribe(changed_hash)
        assert state["models_requests"] == 0 and state["requests"] == []
        state["http_status"] = 400
        with pytest.raises(ReviewError, match="server_audio_disabled:http_status_400") as error:
            producer.transcribe(record)
        assert "SECRET_SERVER_BODY" not in str(error.value)
        with pytest.raises(ReviewError, match="server_audio_disabled:http_status_400"):
            producer.transcribe(record)
        assert len(state["requests"]) == 1


def test_cache_corruption_and_prompt_change_force_new_transcription(tmp_path, record, monkeypatch):
    with _server() as (url, state):
        producer = _producer(url, tmp_path)
        first = producer.transcribe(record)
        key = first["cache_key"]
        path = tmp_path / "cache" / key[:2] / (key + ".json")
        cached = json.loads(path.read_text())
        cached["prediction"]["text"] = "corrupted transcript"
        path.write_text(json.dumps(cached))
        assert producer.transcribe(record)["text"] == "hello world"
        assert len(state["requests"]) == 2
        monkeypatch.setitem(qwen_asr.LANGUAGE_INSTRUCTIONS, "en", "Speech language: English. Transcribe exactly.")
        assert producer.transcribe(record)["cache_hit"] is False
        assert len(state["requests"]) == 3


def test_cli_streams_with_split_budget_errors_and_no_gold_leak(tmp_path, record, monkeypatch, capsys):
    records = []
    for index in range(14):
        item = copy.deepcopy(record)
        item["sample_id"] = f"speech-{index}"
        if index == 0:
            item["split"] = "validation"
        elif index == 1:
            item["qc"]["offline"]["decision"] = "reject"
        elif index == 3:
            item["language"] = "fr"
        elif index == 4:
            item["language"] = "zh"
        _bind(item)
        records.append(item)
    source, destination = tmp_path / "offline.jsonl", tmp_path / "hypotheses.jsonl"
    source.write_text("".join(json.dumps(item) + "\n" for item in records))
    progress = {"selected_read": 0, "written": 0, "errors": 0}
    original_read, original_write = qwen_asr.read_jsonl, qwen_asr.write_jsonl
    original_errors = qwen_asr._atomic_error_stream

    def checked_read(path):
        for item in original_read(path):
            if item["split"] == "train" and item["qc"]["offline"]["decision"] == "accept":
                progress["selected_read"] += 1
                assert progress["selected_read"] - progress["written"] - progress["errors"] <= 4
            yield item

    def checked_write(path, items):
        def observed():
            for item in items:
                progress["written"] += 1
                yield item

        original_write(path, observed())

    @contextmanager
    def counted_errors(path):
        with original_errors(path) as stream:

            def write(line):
                progress["errors"] += 1
                return stream.write(line)

            yield SimpleNamespace(write=write)

    monkeypatch.setattr(qwen_asr, "read_jsonl", checked_read)
    monkeypatch.setattr(qwen_asr, "write_jsonl", checked_write)
    monkeypatch.setattr(qwen_asr, "_atomic_error_stream", counted_errors)
    with _server() as (url, state):
        main(
            [
                "--input",
                str(source),
                "--output",
                str(destination),
                "--base-url",
                url,
                "--cache-dir",
                str(tmp_path / "cache"),
                "--max-samples",
                "7",
                "--concurrency",
                "2",
                "--retries",
                "0",
            ]
        )
    output = list(original_read(destination))
    assert [item["sample_id"] for item in output] == [f"speech-{i}" for i in [2, 4, 5, 6, 7, 8]]
    errors = list(original_read(tmp_path / "hypotheses.errors.jsonl"))
    assert len(errors) == 1 and errors[0]["sample_id"] == "speech-3"
    report = json.loads((tmp_path / "hypotheses.report.json").read_text())
    assert report["attempted"] == 7 and report["transcribed"] == 6 and report["errors"] == 1
    assert report["split"] == "train" and report["skipped_ineligible"] == 2
    assert report["cache_hits"] == 4 and len(state["requests"]) == 2
    assert "input_audio" not in json.dumps(report) and "SECRET_" not in json.dumps(report)
    assert json.loads(capsys.readouterr().out)["transcribed"] == 6


def test_zero_budget_and_expired_runtime_make_no_requests(tmp_path, record, monkeypatch, capsys):
    source, destination = tmp_path / "offline.jsonl", tmp_path / "hypotheses.jsonl"
    source.write_text(json.dumps(record) + "\n")
    with _server() as (url, state):
        main(["--input", str(source), "--output", str(destination), "--base-url", url, "--max-samples", "0"])
        assert destination.read_text() == "" and state["models_requests"] == 0
        clock = iter([0.0, 2.0, 2.0])
        monkeypatch.setattr(qwen_asr.time, "monotonic", lambda: next(clock, 2.0))
        main(["--input", str(source), "--output", str(destination), "--base-url", url, "--max-runtime-seconds", "1"])
        report = json.loads((tmp_path / "hypotheses.report.json").read_text())
        assert report["stopped_at_runtime_limit"] == 1 and report["attempted"] == 0
        assert state["models_requests"] == 0 and not state["requests"]
    assert len(capsys.readouterr().out.splitlines()) == 2


def test_cli_all_failed_requests_exit_nonzero_after_persisting_diagnostics(tmp_path, record):
    source, destination = tmp_path / "offline.jsonl", tmp_path / "hypotheses.jsonl"
    source.write_text(json.dumps(record) + "\n")
    with _server() as (url, state):
        state["http_status"] = 400
        with pytest.raises(SystemExit) as error:
            main(
                [
                    "--input",
                    str(source),
                    "--output",
                    str(destination),
                    "--base-url",
                    url,
                    "--cache-dir",
                    str(tmp_path / "cache"),
                    "--retries",
                    "0",
                ]
            )
        assert error.value.code == 1
    assert destination.read_text() == ""
    report = json.loads((tmp_path / "hypotheses.report.json").read_text())
    assert report["attempted"] == 1 and report["errors"] == 1 and report["transcribed"] == 0
    assert report["model_service_error"] == "server_audio_disabled:http_status_400"
    errors = (tmp_path / "hypotheses.errors.jsonl").read_text()
    assert "server_audio_disabled" in errors and "SECRET_SERVER_BODY" not in errors

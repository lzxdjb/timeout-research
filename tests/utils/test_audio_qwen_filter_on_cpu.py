"""Exercise the real HTTP/audio boundary without contacting an inference service."""

import base64
import copy
import hashlib
import io
import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest
import soundfile as sf

from scripts.audio_data import filter_qwen as qwen_module
from scripts.audio_data.common import record_fingerprint
from scripts.audio_data.filter_qwen import QwenReviewer, ReviewConfig, main, stable_advertised_model_identity


def _blind(**overrides):
    return {
        "answer": "a bell",
        "question_answerable": True,
        "ambiguous": False,
        "evidence": "A resonant metallic strike is audible.",
        "reason": "The tone decays after the impact.",
        **overrides,
    }


def _support(**overrides):
    return {
        "reference_supported": True,
        "question_answerable": True,
        "ambiguous": False,
        "unsupported_claims": [],
        "clear_contradiction": False,
        "evidence": "The audio contains a short resonant chime.",
        "reason": "The reference describes the audible event.",
        **overrides,
    }


@contextmanager
def _server(responses=None, http_error=False, error_status=503, error_message="temporary failure"):
    state = {
        "requests": [],
        "responses": responses,
        "http_error": http_error,
        "models_requests": 0,
        "model_card": {
            "id": "swe-repeat-judge",
            "root": "Qwen3-Omni",
            "parent": None,
            "max_model_len": 32768,
            "owned_by": "vllm",
            "deployment": {"revision": "one"},
        },
    }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, status, payload):
            body = json.dumps(payload).encode()
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
            self._send(200, {"data": [card]})

        def do_POST(self):
            assert self.path == "/v1/chat/completions"
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["requests"].append(request)
            if state["http_error"]:
                self._send(error_status, {"error": {"message": error_message}})
                return
            index = len(state["requests"]) - 1
            if state["responses"] is None:
                context = json.loads(request["messages"][1]["content"][0]["text"])
                response = _support() if "reference" in context else _blind()
            else:
                response = state["responses"][index % len(state["responses"])]
            self._send(
                200,
                {
                    "id": f"mock-{index}",
                    "model": "Qwen3-Omni",
                    "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(response)}}],
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
    path = tmp_path / "chime.wav"
    sample_rate = 16000
    signal = (0.3 * np.sin(2 * np.pi * 440 * np.arange(1600) / sample_rate)).astype("float32")
    sf.write(path, signal, sample_rate, subtype="PCM_16")
    record = {
        "sample_id": "mock-chime",
        "dataset": "fixture",
        "audio_path": str(path),
        "audio_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "task": "closed_qa",
        "language": "en",
        "question": "What sound is audible?",
        "reference": "UNIQUE_GOLD_CHIME",
        "reference_aliases": ["UNIQUE_GOLD_ALIAS"],
        "annotation_type": "human",
        "reference_verified": True,
        "metadata": {"transcript": "UNIQUE_HIDDEN_TRANSCRIPT"},
        "qc": {"offline": {"decision": "accept", "reasons": [], "metrics": {"duration_seconds": 0.1}}},
    }
    _bind_offline(record)
    return record


def _bind_offline(record):
    record["qc"]["offline"]["input_fingerprint"] = record_fingerprint(record)


def _reviewer(url, tmp_path, **overrides):
    return QwenReviewer(ReviewConfig(base_url=url, cache_dir=tmp_path / "cache", retries=0, **overrides))


def test_real_audio_payload_blind_pass_and_hard_reference_acceptance(tmp_path, record):
    with _server() as (url, state):
        result = _reviewer(url, tmp_path).review(record)
    assert result["qc"]["qwen"]["decision"] == "accept"
    assert len(state["requests"]) == 2
    blind, support = state["requests"]
    blind_text = json.dumps(blind)
    assert "UNIQUE_GOLD_CHIME" not in blind_text
    assert "UNIQUE_GOLD_ALIAS" not in blind_text
    assert "UNIQUE_HIDDEN_TRANSCRIPT" not in blind_text
    assert "UNIQUE_GOLD_CHIME" in json.dumps(support)
    for request in state["requests"]:
        payload = request["messages"][1]["content"][1]["input_audio"]
        decoded = base64.b64decode(payload["data"], validate=True)
        assert decoded == (tmp_path / "chime.wav").read_bytes()
        audio, rate = sf.read(io.BytesIO(decoded))
        assert rate == 16000 and len(audio) == 1600
        assert payload["format"] == "wav"
    assert result["qc"]["qwen"]["blind_prediction"]["answer"] != record["reference"]
    assert result["qc"]["qwen"]["request_config"]["model"] == "swe-repeat-judge"
    assert "qwen" not in record["qc"]  # The input manifest record is not mutated.


def test_cache_keys_all_semantics_and_actual_media(tmp_path, record):
    with _server() as (url, state):
        reviewer = _reviewer(url, tmp_path)
        assert reviewer.review(record)["qc"]["qwen"]["cache_hit"] is False
        assert reviewer.review(record)["qc"]["qwen"]["cache_hit"] is True
        assert len(state["requests"]) == 2
        changed = copy.deepcopy(record)
        changed["metadata"]["options"] = ["bell", "dog"]
        stale = reviewer.review(changed)
        assert stale["qc"]["qwen"]["decision"] == "quarantine"
        assert "record_changed_since_offline_review" in stale["qc"]["qwen"]["reasons"][0]
        assert len(state["requests"]) == 2
        _bind_offline(changed)
        assert reviewer.review(changed)["qc"]["qwen"]["cache_hit"] is False
        assert len(state["requests"]) == 4
        changed["reference"] = "other ground truth"
        _bind_offline(changed)
        assert reviewer.review(changed)["qc"]["qwen"]["cache_hit"] is False
        assert len(state["requests"]) == 6
        # Changing bytes under the same path cannot return an earlier cached pass.
        sf.write(record["audio_path"], np.ones(1600) * 0.1, 16000)
        result = reviewer.review(record)
        assert result["qc"]["qwen"]["decision"] == "quarantine"
        assert "audio_changed_since_offline_review" in result["qc"]["qwen"]["reasons"][0]
        assert len(state["requests"]) == 6
        # A newly offline-audited replacement must receive a fresh model review.
        replacement = copy.deepcopy(record)
        replacement["audio_sha256"] = hashlib.sha256((tmp_path / "chime.wav").read_bytes()).hexdigest()
        _bind_offline(replacement)
        assert reviewer.review(replacement)["qc"]["qwen"]["cache_hit"] is False
        assert len(state["requests"]) == 8


def test_cache_checksum_and_model_config_changes_force_new_review(tmp_path, record):
    with _server() as (url, state):
        reviewer = _reviewer(url, tmp_path)
        first = reviewer.review(record)
        key = first["qc"]["qwen"]["cache_key"]
        path = tmp_path / "cache" / key[:2] / (key + ".json")
        cached = json.loads(path.read_text())
        cached["qwen"]["support_review"]["reference_supported"] = False
        path.write_text(json.dumps(cached))
        assert reviewer.review(record)["qc"]["qwen"]["cache_hit"] is False
        assert len(state["requests"]) == 4
        different_config = _reviewer(url, tmp_path, max_tokens=512)
        assert different_config.review(record)["qc"]["qwen"]["cache_hit"] is False
        assert len(state["requests"]) == 6


def test_model_identity_ignores_only_ephemeral_card_fields_and_fresh_reviewers_share_cache(tmp_path, record):
    with _server() as (url, state):
        first = _reviewer(url, tmp_path)
        assert first.review(record)["qc"]["qwen"]["cache_hit"] is False
        fresh = _reviewer(url, tmp_path)
        assert fresh.advertised_model_card != first.advertised_model_card
        assert fresh.identity == first.identity
        assert fresh.review(record)["qc"]["qwen"]["cache_hit"] is True
        assert len(state["requests"]) == 2
        assert "created" not in fresh.identity and "permission" not in fresh.identity
        assert fresh.identity["max_model_len"] == 32768
        assert fresh.identity["deployment"] == {"revision": "one"}
        state["model_card"]["root"] = "Qwen3-Omni-checkpoint-two"
        assert _reviewer(url, tmp_path).review(record)["qc"]["qwen"]["cache_hit"] is False
        assert len(state["requests"]) == 4
        assert (
            _reviewer(url, tmp_path, model_revision="immutable-three").review(record)["qc"]["qwen"]["cache_hit"]
            is False
        )
        assert len(state["requests"]) == 6
    nested = {
        "id": "alias",
        "created": 1,
        "permission": [{"id": "dynamic"}],
        "deployment": {"created": "meaningful nested configuration", "custom": [1, 2]},
    }
    stable = stable_advertised_model_identity(nested)
    assert stable == {"id": "alias", "deployment": nested["deployment"]}
    stable["deployment"]["custom"].append(3)
    assert nested["deployment"]["custom"] == [1, 2]


def test_request_failures_are_not_cached_and_offline_reject_is_preserved(tmp_path, record):
    with _server(http_error=True) as (url, state):
        reviewer = _reviewer(url, tmp_path)
        first = reviewer.review(record)
        assert first["qc"]["qwen"]["decision"] == "quarantine"
        assert not list((tmp_path / "cache").rglob("*.json"))
        state["http_error"] = False
        # Reset counting to keep the default blind/support mock sequence aligned.
        state["requests"].clear()
        assert reviewer.review(record)["qc"]["qwen"]["decision"] == "accept"
        rejected = copy.deepcopy(record)
        rejected["qc"]["offline"]["decision"] = "reject"
        assert reviewer.review(rejected) == rejected
        assert len(state["requests"]) == 2


def test_audio_disabled_deployment_error_is_actionable_and_never_echoes_error_body(tmp_path, record):
    message = "At most 0 audio(s) may be provided in one prompt. (parameter=audio) SECRET_ECHOED_PAYLOAD"
    with _server(http_error=True, error_status=400, error_message=message) as (url, state):
        reviewer = _reviewer(url, tmp_path)
        result = reviewer.review(record)
        assert result["qc"]["qwen"]["decision"] == "quarantine"
        assert "server_audio_disabled:http_status_400" in result["qc"]["qwen"]["reasons"][0]
        assert "SECRET_ECHOED_PAYLOAD" not in json.dumps(result)
        assert reviewer.review(record)["qc"]["qwen"]["decision"] == "quarantine"
        assert len(state["requests"]) == 1  # Do not repeatedly call a deployment that forbids audio.
        assert not list((tmp_path / "cache").rglob("*.json"))
    with _server(http_error=True, error_status=400, error_message="SECRET_UNRELATED_SERVER_BODY") as (url, _state):
        result = _reviewer(url, tmp_path).review(record)
        assert "http_status_400" in result["qc"]["qwen"]["reasons"][0]
        assert "SECRET_UNRELATED_SERVER_BODY" not in json.dumps(result)


def test_schema_validation_rejects_truthy_strings_and_preserves_uncertainty(tmp_path, record):
    with _server([_blind(question_answerable="true")]) as (url, _state):
        result = _reviewer(url, tmp_path).review(record)
        assert result["qc"]["qwen"]["decision"] == "quarantine"
        assert not list((tmp_path / "cache").rglob("*.json"))
    with _server([_blind(), _support(reference_supported=False, ambiguous=True)]) as (url, _state):
        result = _reviewer(url, tmp_path).review(record)
        assert result["qc"]["qwen"]["decision"] == "quarantine"


def test_clear_audible_contradiction_can_reject(tmp_path, record):
    support = _support(
        reference_supported=False,
        unsupported_claims=["A dog is barking."],
        clear_contradiction=True,
        evidence="A sustained electronic tone is audible, with no bark-like transients.",
    )
    with _server([_blind(), support]) as (url, _state):
        assert _reviewer(url, tmp_path).review(record)["qc"]["qwen"]["decision"] == "reject"


def test_flac_is_losslessly_decoded_without_resampling(tmp_path, record):
    path = tmp_path / "source.flac"
    samples = np.array([[0.25, -0.125], [-0.5, 0.375]] * 100, dtype="float32")
    sf.write(path, samples, 22050, subtype="PCM_24")
    record["audio_path"] = str(path)
    record["audio_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    _bind_offline(record)
    with _server() as (url, state):
        result = _reviewer(url, tmp_path).review(record)
    assert result["qc"]["qwen"]["decision"] == "accept"
    payload = state["requests"][0]["messages"][1]["content"][1]["input_audio"]
    actual, rate = sf.read(io.BytesIO(base64.b64decode(payload["data"])), dtype="float32", always_2d=True)
    expected, _rate = sf.read(path, dtype="float32", always_2d=True)
    assert rate == 22050
    np.testing.assert_array_equal(actual, expected)


def test_cli_sample_limit_quarantines_unreviewed_without_request(tmp_path, record, capsys):
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    input_path.write_text(json.dumps(record) + "\n")
    main(["--input", str(input_path), "--output", str(output_path), "--max-samples", "0"])
    result = json.loads(output_path.read_text())
    assert result["qc"]["qwen"]["decision"] == "quarantine"
    assert "qwen_sample_limit_not_reviewed" in result["qc"]["qwen"]["reasons"]
    assert json.loads(capsys.readouterr().out)["selected"] == 0
    report = json.loads((tmp_path / "output.report.json").read_text())
    assert report["budget_skipped"] == 1 and report["model_config"] is None


def test_cli_streams_in_order_with_bounded_pending_work_and_reports_budget(tmp_path, record, monkeypatch, capsys):
    records = []
    for index in range(12):
        item = copy.deepcopy(record)
        item["sample_id"] = f"example-{index}"
        _bind_offline(item)
        if index == 1:
            item["qc"]["offline"]["decision"] = "reject"
            item["qc"]["offline"]["reasons"] = ["fixture_rejected"]
        records.append(item)
    source = tmp_path / "input.jsonl"
    destination = tmp_path / "filtered.jsonl"
    source.write_text("".join(json.dumps(item) + "\n" for item in records))
    progress = {"read": 0, "written": 0}
    original_read, original_write = qwen_module.read_jsonl, qwen_module.write_jsonl

    def checked_read(path):
        for item in original_read(path):
            progress["read"] += 1
            # This fails if the CLI eagerly lists the manifest or submits an
            # unbounded sequence of futures before consuming output records.
            assert progress["read"] - progress["written"] <= 4
            yield item

    def checked_write(path, items):
        def observed():
            for item in items:
                progress["written"] += 1
                yield item

        return original_write(path, observed())

    monkeypatch.setattr(qwen_module, "read_jsonl", checked_read)
    monkeypatch.setattr(qwen_module, "write_jsonl", checked_write)
    with _server() as (url, state):
        arguments = [
            "--input",
            str(source),
            "--output",
            str(destination),
            "--base-url",
            url,
            "--cache-dir",
            str(tmp_path / "cache"),
            "--max-samples",
            "2",
            "--concurrency",
            "2",
            "--retries",
            "0",
        ]
        main(arguments)
        assert len(state["requests"]) == 4
        output = list(original_read(destination))
        assert [item["sample_id"] for item in output] == [item["sample_id"] for item in records]
        assert output[1] == records[1]  # Preserve offline rejection byte-for-byte as JSON data.
        assert output[0]["qc"]["qwen"]["decision"] == "accept"
        assert output[2]["qc"]["qwen"]["decision"] == "accept"
        assert all(item["qc"]["qwen"]["decision"] == "quarantine" for item in output[3:])
        report = json.loads((tmp_path / "filtered.report.json").read_text())
        assert report["records"] == 12 and report["selected"] == 2
        assert report["budget_skipped"] == 9 and report["offline_skipped"] == 1
        assert report["cache_hits"] == 0 and report["review_errors"] == 0
        assert report["max_in_flight"] == 4
        assert report["model_config"]["model"] == "swe-repeat-judge"
        assert "input_audio" not in json.dumps(report) and "audio_payload" not in json.dumps(report)
        progress.update(read=0, written=0)
        main(arguments)
        assert len(state["requests"]) == 4
        report = json.loads((tmp_path / "filtered.report.json").read_text())
        assert report["cache_hits"] == 2
    assert len(capsys.readouterr().out.splitlines()) == 2

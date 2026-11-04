"""Exercise opt-in sampling at the HTTP boundary used by every GAGE suite."""

from __future__ import annotations

import http.client
import json
import queue
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

import pytest

from verl.trainer.ppo import audio_benchmark
from verl.trainer.ppo.audio_benchmark_router import AudioBenchmarkRouter

SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "max_tokens": 4096}
GREEDY = {"temperature": 0.0, "top_p": 1.0, "top_k": 1, "max_tokens": 4096}


@pytest.fixture
def endpoint():
    received = queue.Queue()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            received.put((self.path, body, self.headers))
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"{}")

        def do_GET(self):
            self.send_response(200)
            body = b'{"data":[{"id":"student"}]}'
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}/v1", received
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def _post(api_base, path, body):
    url = urlsplit(api_base)
    connection = http.client.HTTPConnection(url.hostname, url.port, timeout=5)
    try:
        connection.request(
            "POST",
            url.path + path,
            body=body,
            headers={"Content-Type": "application/json", "Authorization": "Bearer x"},
        )
        response = connection.getresponse()
        return response.status, response.read()
    finally:
        connection.close()


@pytest.mark.parametrize("path", ["/chat/completions", "/completions"])
@pytest.mark.parametrize("sampling", [SAMPLING, GREEDY])
def test_explicit_sampling_reaches_upstream_and_preserves_input(endpoint, path, sampling):
    api_base, received = endpoint
    original = {
        "model": "student",
        "messages": [{"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": "A" * 10000}}]}],
        "temperature": 0,
        "top_p": 1,
        "top_k": -1,
        "max_completion_tokens": 64,
        "chat_template_kwargs": {"enable_thinking": False},
        "stream": True,
    }
    with AudioBenchmarkRouter([api_base], generation_overrides=sampling) as router:
        assert _post(router.api_base, path, json.dumps(original).encode())[0] == 200
    forwarded_path, body, headers = received.get(timeout=5)
    assert forwarded_path == f"/v1{path}"
    assert headers["Authorization"] == "Bearer x"
    assert int(headers["Content-Length"]) == len(body)
    expected = {key: value for key, value in original.items() if key != "max_completion_tokens"}
    assert json.loads(body) == {**expected, **sampling}


def test_sampling_does_not_change_non_generation_requests(endpoint):
    api_base, received = endpoint
    body = b'{ "arbitrary": "retain exact bytes" }'
    with AudioBenchmarkRouter([api_base], generation_overrides=SAMPLING) as router:
        assert _post(router.api_base, "/tokenize", body)[0] == 200
    assert received.get(timeout=5)[1] == body


def test_invalid_generation_json_fails_without_sending_to_upstream(endpoint):
    api_base, received = endpoint
    with AudioBenchmarkRouter([api_base], generation_overrides=SAMPLING) as router:
        assert _post(router.api_base, "/chat/completions", b"[]")[0] == 400
    assert received.empty()


@pytest.mark.parametrize(
    "values",
    [
        {"temperature": -0.1},
        {"temperature": float("nan")},
        {"temperature": True},
        {"top_p": 0},
        {"top_p": 1.1},
        {"top_k": 0},
        {"top_k": -2},
        {"top_k": 20.5},
        {"max_tokens": 0},
        {"max_tokens": True},
        {"unsupported": 1},
    ],
)
def test_invalid_sampling_fails_before_starting_router(values):
    with pytest.raises(ValueError, match="audio generation override"):
        AudioBenchmarkRouter(["http://127.0.0.1:8000/v1"], generation_overrides=values)


@pytest.mark.parametrize("load_balance", [False, True])
@pytest.mark.parametrize("sampling", [SAMPLING, GREEDY])
def test_all_six_suites_use_sampling_on_single_replica(monkeypatch, tmp_path, endpoint, load_balance, sampling):
    api_base, received = endpoint
    runner = tmp_path / "gage/scripts/run/run_suite.sh"
    runner.parent.mkdir(parents=True)
    runner.touch()
    suites = ["aishell1", "librispeech", "wenetspeech", "clothoaqa", "rul_muchomusic", "voicebench_bbh"]

    def fake_run(command, **kwargs):
        target = command[command.index("--external-api-base") + 1]
        assert target != api_base
        for suite in suites:
            flag = audio_benchmark._BENCHMARK_SPECS[suite][0]
            assert flag in command
            request = {"model": "student", "messages": [{"role": "user", "content": suite}], "max_tokens": 64}
            assert _post(target, "/chat/completions", json.dumps(request).encode())[0] == 200
        run_id = command[command.index("--run-id") + 1]
        summary = tmp_path / "results" / run_id / "suite_summary.json"
        summary.parent.mkdir(parents=True)
        summary.write_text("{}")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(audio_benchmark.subprocess, "run", fake_run)
    monkeypatch.setattr(audio_benchmark, "_read_metrics", lambda *args, **kwargs: {"val-audio/completed": 1.0})
    result = audio_benchmark.run_audio_benchmarks(
        {
            "enabled": True,
            "gage_root": str(tmp_path / "gage"),
            "data_root": "/mnt",
            "output_dir": str(tmp_path / "results"),
            "benchmarks": suites,
            "load_balance": load_balance,
            **sampling,
        },
        server_addresses=[api_base],
        model_path="/model",
        experiment_name="sampling",
        global_step=1,
    )
    assert result == {"val-audio/completed": 1.0}
    for _ in suites:
        payload = json.loads(received.get(timeout=5)[1])
        assert {key: payload[key] for key in sampling} == sampling

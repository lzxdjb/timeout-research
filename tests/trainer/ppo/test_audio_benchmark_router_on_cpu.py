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

from __future__ import annotations

import gzip
import http.client
import json
import queue
import socket
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

import pytest

from verl.trainer.ppo.audio_benchmark_router import AudioBenchmarkRouter


class _Backend:
    def __init__(self, name: str, received: queue.Queue):
        self.name = name
        self.release = threading.Event()
        self.status = 200
        self.body: bytes | None = None
        self.headers: dict[str, str] = {}
        backend = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_GET(self):
                self._handle()

            def do_POST(self):
                self._handle()

            def _handle(self):
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                received.put(
                    {
                        "backend": backend.name,
                        "method": self.command,
                        "path": self.path,
                        "headers": {key.lower(): value for key, value in self.headers.items()},
                        "body": body,
                    }
                )
                if self.path.startswith("/v1/hold"):
                    if not backend.release.wait(10):
                        return
                if self.path.startswith("/v1/stream"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Transfer-Encoding", "chunked")
                    self.send_header("X-Replica", backend.name)
                    self.end_headers()
                    try:
                        for chunk in (b"data: first\n\n", b"data: [DONE]\n\n"):
                            self.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
                            self.wfile.flush()
                            if chunk.startswith(b"data: first") and not backend.release.wait(10):
                                return
                        self.wfile.write(b"0\r\n\r\n")
                        self.wfile.flush()
                    except ConnectionError:
                        # A client-disconnect test intentionally cancels the
                        # proxy while this backend waits to finish its stream.
                        pass
                    return
                payload = backend.body if backend.body is not None else json.dumps({"backend": backend.name}).encode()
                self.send_response(backend.status)
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("X-Replica", backend.name)
                for key, value in backend.headers.items():
                    self.send_header(key, value)
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
        self.thread.start()
        self.api_base = f"http://127.0.0.1:{self.server.server_port}/v1"

    def close(self):
        self.release.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


@pytest.fixture
def backends():
    received = queue.Queue()
    instances = []

    def make(name):
        backend = _Backend(name, received)
        instances.append(backend)
        return backend

    yield make, received
    for backend in instances:
        backend.close()


def _request(api_base, suffix="/chat/completions", *, method="POST", body=b"{}", headers=None):
    url = urlsplit(api_base)
    connection = http.client.HTTPConnection(url.hostname, url.port, timeout=5)
    try:
        connection.request(method, url.path.rstrip("/") + suffix, body=body, headers=headers or {})
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        connection.close()


def test_sequential_requests_use_every_replica_evenly(backends):
    make, received = backends
    replicas = [make(str(index)) for index in range(3)]
    with AudioBenchmarkRouter([replica.api_base for replica in replicas]) as router:
        results = [json.loads(_request(router.api_base)[2])["backend"] for _ in range(12)]

    assert Counter(results) == {"0": 4, "1": 4, "2": 4}
    assert received.qsize() == 12


def test_requests_prefer_replica_without_inflight_work(backends):
    make, received = backends
    replicas = [make(str(index)) for index in range(2)]
    with AudioBenchmarkRouter([replica.api_base for replica in replicas]) as router:
        with ThreadPoolExecutor(max_workers=1) as executor:
            slow = executor.submit(_request, router.api_base, "/hold")
            occupied = received.get(timeout=5)["backend"]
            try:
                fast_results = [json.loads(_request(router.api_base)[2])["backend"] for _ in range(5)]
                assert all(name != occupied for name in fast_results)
                assert not slow.done()
            finally:
                for replica in replicas:
                    replica.release.set()
            assert slow.result(timeout=5)[0] == 200


def test_proxy_preserves_large_body_auth_path_query_and_error_response(backends):
    make, received = backends
    replica = make("one")
    replica.status = 429
    replica.body = b'{"error":{"message":"capacity reached"}}'
    replica.headers = {"Content-Type": "application/json", "Retry-After": "7"}
    body = b'{"audio":"' + b"A" * (2 * 1024 * 1024) + b'","metadata":"\\u4f60\\u597d"}'
    with AudioBenchmarkRouter([replica.api_base]) as router:
        status, headers, response_body = _request(
            router.api_base,
            "/chat/completions?trace=one%2Ftwo&repeat=1&repeat=2",
            body=body,
            headers={"Authorization": "Bearer local-test-key", "Content-Type": "application/json"},
        )

    forwarded = received.get(timeout=5)
    assert forwarded["method"] == "POST"
    assert forwarded["path"] == "/v1/chat/completions?trace=one%2Ftwo&repeat=1&repeat=2"
    assert forwarded["headers"]["authorization"] == "Bearer local-test-key"
    assert forwarded["headers"]["content-type"] == "application/json"
    assert forwarded["body"] == body
    assert status == 429
    assert response_body == replica.body
    assert headers["Retry-After"] == "7"
    assert headers["Content-Type"] == "application/json"


def test_proxy_preserves_compressed_response_bytes_and_get_method(backends):
    make, received = backends
    replica = make("one")
    replica.body = gzip.compress(b'{"data":[{"id":"served-model"}]}')
    replica.headers = {"Content-Encoding": "gzip", "Content-Type": "application/json"}
    with AudioBenchmarkRouter([replica.api_base]) as router:
        status, headers, body = _request(router.api_base, "/models?detail=true", method="GET", body=None)

    forwarded = received.get(timeout=5)
    assert forwarded["method"] == "GET"
    assert forwarded["path"] == "/v1/models?detail=true"
    assert status == 200
    assert headers["Content-Encoding"] == "gzip"
    assert body == replica.body


def test_streaming_response_is_forwarded_before_completion_and_remains_inflight(backends):
    make, received = backends
    replicas = [make(str(index)) for index in range(2)]
    with AudioBenchmarkRouter([replica.api_base for replica in replicas]) as router:
        url = urlsplit(router.api_base)
        connection = http.client.HTTPConnection(url.hostname, url.port, timeout=5)
        try:
            connection.request("POST", url.path + "/stream", body=b"{}")
            response = connection.getresponse()
            occupied = received.get(timeout=5)["backend"]
            assert response.status == 200
            assert response.getheader("Content-Type") == "text/event-stream"
            assert response.read(len(b"data: first\n\n")) == b"data: first\n\n"
            fast_result = json.loads(_request(router.api_base)[2])["backend"]
            assert fast_result != occupied
            for replica in replicas:
                replica.release.set()
            assert response.read() == b"data: [DONE]\n\n"
        finally:
            for replica in replicas:
                replica.release.set()
            connection.close()


def test_client_disconnect_releases_inflight_work_before_upstream_finishes(backends, monkeypatch):
    make, received = backends
    replicas = [make(str(index)) for index in range(2)]
    stream_finished = threading.Event()
    inflight_after_stream = []
    original_forward = AudioBenchmarkRouter._forward

    async def observe_forward(self, request):
        try:
            return await original_forward(self, request)
        finally:
            if request.path == "/v1/stream":
                inflight_after_stream.append(list(self._inflight))
                stream_finished.set()

    monkeypatch.setattr(AudioBenchmarkRouter, "_forward", observe_forward)
    with AudioBenchmarkRouter([replica.api_base for replica in replicas]) as router:
        url = urlsplit(router.api_base)
        connection = http.client.HTTPConnection(url.hostname, url.port, timeout=5)
        response = None
        try:
            connection.request("POST", url.path + "/stream", body=b"{}")
            response = connection.getresponse()
            received.get(timeout=5)
            assert response.read(len(b"data: first\n\n")) == b"data: first\n\n"
            assert not stream_finished.is_set()
            response.close()
            connection.close()

            assert stream_finished.wait(5), "Disconnected client left its streaming request running"
            assert inflight_after_stream == [[0, 0]]
            assert all(not replica.release.is_set() for replica in replicas)
            results = [json.loads(_request(router.api_base)[2])["backend"] for _ in range(4)]
            assert Counter(results) == {"0": 2, "1": 2}
        finally:
            if response is not None:
                response.close()
            connection.close()
            for replica in replicas:
                replica.release.set()


def test_unavailable_upstream_returns_502_without_breaking_healthy_requests(backends):
    make, _ = backends
    healthy = make("healthy")
    with socket.socket() as unavailable:
        unavailable.bind(("127.0.0.1", 0))
        unavailable_base = f"http://127.0.0.1:{unavailable.getsockname()[1]}/v1"
        with AudioBenchmarkRouter([unavailable_base, healthy.api_base]) as router:
            responses = [_request(router.api_base) for _ in range(6)]

    assert [status for status, _, _ in responses].count(502) >= 1
    successful = [json.loads(body)["backend"] for status, _, body in responses if status == 200]
    assert successful
    assert set(successful) == {"healthy"}


@pytest.mark.parametrize("raise_in_context", [False, True])
def test_context_closes_listening_socket_and_background_thread(backends, raise_in_context):
    make, _ = backends
    replica = make("one")
    previous_threads = set(threading.enumerate())

    @contextmanager
    def expected_error():
        if raise_in_context:
            with pytest.raises(RuntimeError, match="caller failed"):
                yield
        else:
            yield

    with expected_error():
        with AudioBenchmarkRouter([replica.api_base]) as router:
            url = urlsplit(router.api_base)
            router_threads = set(threading.enumerate()) - previous_threads
            assert router_threads
            if raise_in_context:
                raise RuntimeError("caller failed")

    for thread in router_threads:
        thread.join(timeout=5)
        assert not thread.is_alive()
    with pytest.raises(OSError):
        with socket.create_connection((url.hostname, url.port), timeout=1):
            pass


def test_startup_failure_closes_allocated_session_socket_and_thread(monkeypatch):
    resources = {}
    original_start = AudioBenchmarkRouter._start

    async def fail_after_allocating_resources(self):
        await original_start(self)
        resources.update(session=self._session, loop=self._loop, thread=self._thread, url=urlsplit(self.api_base))
        raise OSError("injected startup failure")

    monkeypatch.setattr(AudioBenchmarkRouter, "_start", fail_after_allocating_resources)
    with pytest.raises(RuntimeError, match="Could not start the audio benchmark router") as error:
        with AudioBenchmarkRouter(["http://127.0.0.1:1/v1"]):
            pytest.fail("A failed router must not enter its context")

    assert isinstance(error.value.__cause__, OSError)
    assert resources["session"].closed
    assert resources["loop"].is_closed()
    assert not resources["thread"].is_alive()
    url = resources["url"]
    with pytest.raises(OSError):
        with socket.create_connection((url.hostname, url.port), timeout=1):
            pass

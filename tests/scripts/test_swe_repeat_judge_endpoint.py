import os
import signal
import socket
import subprocess
import time
from pathlib import Path

import pytest

from scripts import swe_repeat_judge_endpoint as endpoint


@pytest.fixture
def auto_detection(monkeypatch):
    for name in ("POD_IP", "MY_POD_IP", "SWE_REPEAT_JUDGE_ADVERTISE_HOST", "SWE_REPEAT_JUDGE_ADVERTISE_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(endpoint, "route_source", lambda: "10.23.4.5")
    monkeypatch.setattr(endpoint, "hostname_addresses", lambda: iter(["10.23.4.6"]))


def test_detection_prefers_pod_environment_then_route_then_hostname(auto_detection, monkeypatch):
    assert endpoint.resolve_endpoint("0.0.0.0", 18090) == "http://10.23.4.5:18090"
    monkeypatch.setenv("POD_IP", "10.23.4.7")
    assert endpoint.detect_host() == "10.23.4.7"
    monkeypatch.setenv("POD_IP", "127.0.0.1")
    monkeypatch.setenv("MY_POD_IP", "10.23.4.8")
    assert endpoint.detect_host() == "10.23.4.8"
    monkeypatch.delenv("MY_POD_IP")
    assert endpoint.detect_host() == "10.23.4.5"
    monkeypatch.setattr(endpoint, "route_source", lambda: "")
    monkeypatch.setattr(endpoint, "hostname_addresses", lambda: iter([
        "0.0.0.0", "127.0.0.1", "::1", "169.254.1.2", "224.0.0.1", "10.23.4.6",
    ]))
    assert endpoint.detect_host() == "10.23.4.6"


def test_detection_failure_is_actionable(auto_detection, monkeypatch):
    monkeypatch.setattr(endpoint, "route_source", lambda: "")
    monkeypatch.setattr(endpoint, "hostname_addresses", lambda: iter(["127.0.0.1", "::1"]))
    with pytest.raises(ValueError, match="Cannot detect a usable pod IPv4"):
        endpoint.resolve_endpoint("0.0.0.0", 18090)


def test_overrides_and_concrete_bind_host(auto_detection, monkeypatch):
    assert endpoint.resolve_endpoint("10.23.4.9", 18090) == "http://10.23.4.9:18090"
    monkeypatch.setenv("SWE_REPEAT_JUDGE_ADVERTISE_HOST", "judge.service.example")
    assert endpoint.resolve_endpoint("0.0.0.0", 18090) == "http://judge.service.example:18090"
    monkeypatch.setenv("SWE_REPEAT_JUDGE_ADVERTISE_HOST", "fd00::123")
    assert endpoint.resolve_endpoint("::", 18090) == "http://[fd00::123]:18090"
    monkeypatch.setenv("SWE_REPEAT_JUDGE_ADVERTISE_URL", "https://judge.service.example/proxy")
    monkeypatch.setattr(endpoint, "detect_host", lambda: pytest.fail("URL override must bypass detection"))
    assert endpoint.resolve_endpoint("0.0.0.0", 18090) == "https://judge.service.example/proxy"


@pytest.mark.parametrize("url", [
    "http://0.0.0.0:18090", "http://[::]:18090", "http://127.0.0.1:18090",
    "http://localhost:18090", "http://169.254.1.2:18090", "http://224.0.0.1:18090",
    "[http://10.23.4.5:18090](http://10.23.4.5:18090)", "http://unavailable host:18090",
    "ftp://judge:18090", "http://user:password@judge:18090", "http://judge:65536",
    "http://judge:0", "http://judge:abc", "http://judge:", "http://judge?x=1", "http://judge#fragment",
])
def test_invalid_override_is_rejected(auto_detection, monkeypatch, url):
    monkeypatch.setenv("SWE_REPEAT_JUDGE_ADVERTISE_URL", url)
    with pytest.raises(ValueError, match="Invalid judge client endpoint"):
        endpoint.resolve_endpoint("0.0.0.0", 18090)


def test_host_override_cannot_contain_url_path(auto_detection, monkeypatch):
    monkeypatch.setenv("SWE_REPEAT_JUDGE_ADVERTISE_HOST", "judge.service.example/proxy")
    with pytest.raises(ValueError, match="expected only an IP address or DNS name"):
        endpoint.resolve_endpoint("0.0.0.0", 18090)


def test_readiness_watcher_exits_when_server_parent_exits(monkeypatch, capsys):
    parent_ids = iter([42, 1])
    monkeypatch.setattr(endpoint.os, "getppid", lambda: next(parent_ids))
    monkeypatch.setattr(endpoint.time, "sleep", lambda seconds: None)

    class Unready:
        def open(self, *args, **kwargs):
            raise OSError("Server exited before loading")

    monkeypatch.setattr(endpoint, "build_opener", lambda *args: Unready())
    endpoint.announce_when_ready(42, "0.0.0.0", 18090, "http://10.23.4.5:18090")
    assert "Judge ready." not in capsys.readouterr().err


@pytest.fixture
def fake_launcher(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    fake_vllm = tmp_path / "vllm"
    fake_vllm.write_text('''#!/usr/bin/env python3
import os
import signal
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

if os.environ.get("FAKE_VLLM_FAIL"):
    print("model loading failed", flush=True)
    sys.exit(17)

def stop(signum, frame):
    print("vllm received signal " + str(signum), flush=True)
    sys.exit(23)

signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)

class Handler(BaseHTTPRequestHandler):
    calls = 0
    def do_GET(self):
        if self.path != "/health":
            self.send_error(404)
            return
        Handler.calls += 1
        status = 503 if Handler.calls == 1 else 200
        print("health=" + str(status), flush=True)
        self.send_response(status)
        self.end_headers()
    def log_message(self, *args):
        pass

port = int(sys.argv[sys.argv.index("--port") + 1])
server = HTTPServer(("127.0.0.1", port), Handler)
print("vllm startup: http://0.0.0.0:" + str(port), flush=True)
server.serve_forever()
''')
    fake_vllm.chmod(0o755)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("SWE_REPEAT_JUDGE_", "SWE_AGENT_REPEAT_JUDGE_"))}
    env.update({
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "SWE_REPEAT_JUDGE_CHECKPOINT": str(tmp_path),
        "SWE_REPEAT_JUDGE_PORT": str(port),
        "POD_IP": "10.23.4.5",
        # Readiness must bypass proxy settings.
        "http_proxy": "http://127.0.0.1:1",
        "HTTP_PROXY": "http://127.0.0.1:1",
        "NO_PROXY": "",
        "no_proxy": "",
    })
    launcher = Path(__file__).resolve().parents[2] / "scripts/serve_swe_repeat_judge.sh"
    return launcher, env, port


@pytest.mark.parametrize("shutdown_signal", [signal.SIGTERM, signal.SIGINT])
def test_ready_message_follows_health_and_signals_reach_vllm(fake_launcher, tmp_path, shutdown_signal):
    launcher, env, port = fake_launcher
    log = tmp_path / "server.log"
    with log.open("w") as output:
        process = subprocess.Popen(["bash", str(launcher)], env=env, stdout=output, stderr=output)
        try:
            deadline = time.monotonic() + 10
            while "Judge ready." not in log.read_text():
                assert process.poll() is None, log.read_text()
                assert time.monotonic() < deadline, log.read_text()
                time.sleep(0.05)
            text = log.read_text()
            assert text.index("vllm startup:") < text.index("health=503") < text.index("health=200")
            assert text.index("health=200") < text.index("Judge ready.")
            assert f"export SWE_AGENT_REPEAT_JUDGE_URL=http://10.23.4.5:{port}" in text
            process.send_signal(shutdown_signal)
            assert process.wait(timeout=5) == 23
            assert f"vllm received signal {int(shutdown_signal)}" in log.read_text()
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)


def test_failed_startup_keeps_exit_status_and_never_announces_ready(fake_launcher):
    launcher, env, _ = fake_launcher
    result = subprocess.run(["bash", str(launcher)], env={**env, "FAKE_VLLM_FAIL": "1"},
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 17
    assert "model loading failed" in result.stdout
    assert "Judge ready." not in result.stderr


def test_dry_run_does_not_launch_or_wait(fake_launcher):
    launcher, env, port = fake_launcher
    result = subprocess.run(["bash", str(launcher), "--dry-run"], env=env,
                            capture_output=True, text=True, timeout=5)
    assert result.returncode == 0
    assert f"Judge client endpoint: http://10.23.4.5:{port}" in result.stderr
    assert "--host 0.0.0.0" in result.stdout
    assert "vllm startup:" not in result.stdout
    assert "Judge ready." not in result.stderr

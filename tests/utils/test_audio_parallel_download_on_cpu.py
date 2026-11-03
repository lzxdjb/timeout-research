"""Exercise segmented archive downloads over real local HTTP connections."""

from __future__ import annotations

import fcntl
import hashlib
import json
import re
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from scripts.audio_data import download


class ArchiveServer:
    """Small controllable origin; transfer tests do not replace urllib calls."""

    def __init__(self, *, data=None, ranges=True, validator=True, fault=None):
        self.data = data if data is not None else bytes(range(251)) * 4096
        self.ranges = ranges
        self.validator = validator
        self.fault = fault
        self.requests = []
        self.completed = []
        self.interrupted = []
        self.bytes_sent = 0
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()
        self.one_complete = threading.Event()
        self.failed_once = False
        owner = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_HEAD(self):
                self.serve(head=True)

            def do_GET(self):
                self.serve(head=False)

            def serve(self, *, head):
                header = self.headers.get("Range")
                match = re.fullmatch(r"bytes=(\d+)-(\d*)", header or "")
                start = int(match.group(1)) if match and owner.ranges else 0
                end = min(int(match.group(2)), len(owner.data) - 1) if match and match.group(2) else len(owner.data) - 1
                if not owner.ranges:
                    end = len(owner.data) - 1
                bounded = bool(match and match.group(2) and end > start)
                ranged_body = bool(match and owner.ranges and end > start)
                record = {
                    "method": "HEAD" if head else "GET",
                    "path": self.path,
                    "range": header,
                    "if_range": self.headers.get("If-Range"),
                    "start": start,
                    "end": end,
                }
                with owner.lock:
                    owner.requests.append(record)
                    fail = bounded and owner.fault == "disconnect" and not owner.failed_once
                    if fail:
                        owner.failed_once = True
                self.send_response(206 if match and owner.ranges else 200)
                if owner.validator:
                    if owner.validator == "last-modified":
                        self.send_header("Last-Modified", "Wed, 21 Oct 2015 07:28:00 GMT")
                    else:
                        etag = 'W/"archive-v1"' if owner.validator == "weak" else '"archive-v1"'
                        if ranged_body and start > 0 and owner.fault == "validator":
                            etag = '"archive-v2"'
                        self.send_header("ETag", etag)
                self.send_header("Accept-Ranges", "bytes" if owner.ranges else "none")
                self.send_header("Content-Length", str(end - start + 1))
                if ranged_body and owner.fault == "encoding":
                    self.send_header("Content-Encoding", "gzip")
                if match and owner.ranges:
                    actual_start = start + 1 if ranged_body and owner.fault == "range-start" else start
                    actual_end = end - 1 if ranged_body and owner.fault == "range-end" else end
                    actual_total = (
                        len(owner.data) + 1 if ranged_body and owner.fault == "range-total" else len(owner.data)
                    )
                    self.send_header("Content-Range", f"bytes {actual_start}-{actual_end}/{actual_total}")
                self.end_headers()
                if head:
                    return
                with owner.lock:
                    owner.active += 1
                    owner.max_active = max(owner.max_active, owner.active)
                try:
                    if fail:
                        owner.one_complete.wait(timeout=3)
                        time.sleep(0.1)  # Let the client durably record the other completed chunk.
                    payload = owner.data[start : end + 1]
                    if fail:
                        payload = payload[: max(1, len(payload) // 2)]
                    # A short delay makes overlapping client connections observable.
                    for offset in range(0, len(payload), 16 * 1024):
                        block = payload[offset : offset + 16 * 1024]
                        self.wfile.write(block)
                        self.wfile.flush()
                        with owner.lock:
                            owner.bytes_sent += len(block)
                        if bounded:
                            time.sleep(0.002)
                    if fail:
                        with owner.lock:
                            owner.interrupted.append(record)
                        self.close_connection = True
                        self.connection.shutdown(socket.SHUT_RDWR)
                    else:
                        with owner.lock:
                            owner.completed.append(record)
                        if bounded:
                            owner.one_complete.set()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
                finally:
                    with owner.lock:
                        owner.active -= 1

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def asset(self, path="/archive", filename="archive.bin"):
        return {
            "filename": filename,
            "url": f"http://127.0.0.1:{self.server.server_port}{path}",
            "size_bytes": len(self.data),
            "sha256": hashlib.sha256(self.data).hexdigest(),
        }

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=3)


@pytest.fixture
def archive_server():
    servers = []

    def create(**kwargs):
        server = ArchiveServer(**kwargs)
        servers.append(server)
        return server

    yield create
    for server in servers:
        server.close()


@pytest.fixture(autouse=True)
def small_parallel_chunks(monkeypatch):
    # Production thresholds keep tiny metadata downloads serial. Tests transfer
    # small local archives while retaining real socket and filesystem behavior.
    from scripts.audio_data import parallel_download

    monkeypatch.setattr(parallel_download, "CHUNK_BYTES", 16 * 1024)


def fetch(server, directory, *, asset=None, budget=None, connections=4):
    return download.download_asset(
        asset or server.asset(),
        directory,
        len(server.data) * 2 if budget is None else budget,
        timeout=5,
        progress="off",
        connections=connections,
    )


def test_parallel_download_uses_simultaneous_connections_and_publishes_receipt(tmp_path, archive_server):
    server = archive_server()
    transferred = fetch(server, tmp_path, budget=len(server.data))
    assert (tmp_path / "archive.bin").read_bytes() == server.data
    assert server.max_active > 1
    assert transferred == server.bytes_sent
    assert transferred == len(server.data)
    receipt = json.loads((tmp_path / "archive.bin.receipt.json").read_text())
    assert receipt["publisher_checksum_verified"] is True
    request_count = len(server.requests)
    assert fetch(server, tmp_path, budget=0) == 0
    assert len(server.requests) == request_count


def test_parallel_last_modified_validator_is_preserved_across_ranges(tmp_path, archive_server):
    server = archive_server(validator="last-modified")
    assert fetch(server, tmp_path, budget=len(server.data)) == len(server.data)
    assert (tmp_path / "archive.bin").read_bytes() == server.data
    assert server.max_active > 1
    following = [r for r in server.requests if r["method"] == "GET" and r["start"] > 0]
    assert following
    assert all(r["if_range"] == "Wed, 21 Oct 2015 07:28:00 GMT" for r in following)


def test_parallel_mode_preserves_legacy_sequential_prefix(tmp_path, archive_server):
    server = archive_server()
    prefix = 12345
    (tmp_path / "archive.bin.part").write_bytes(server.data[:prefix])
    (tmp_path / "archive.bin.part.json").write_text(
        json.dumps({"url": server.asset()["url"], "validator": '"archive-v1"'})
    )
    transferred = fetch(server, tmp_path, budget=len(server.data) - prefix)
    assert (tmp_path / "archive.bin").read_bytes() == server.data
    assert transferred == server.bytes_sent
    assert transferred == len(server.data) - prefix
    body_requests = [r for r in server.requests if r["method"] == "GET" and r["end"] > r["start"]]
    assert body_requests
    assert all(r["start"] >= prefix for r in body_requests)
    assert all(r["if_range"] == '"archive-v1"' for r in body_requests)


def test_parallel_known_size_rejects_insufficient_budget_before_network(tmp_path, archive_server):
    server = archive_server()
    with pytest.raises(ValueError, match="(?i)budget"):
        fetch(server, tmp_path, budget=len(server.data) - 1)
    assert not server.requests
    assert not (tmp_path / "archive.bin").exists()


def test_parallel_cli_applies_one_network_budget_across_assets(tmp_path, archive_server):
    server = archive_server()
    first = server.asset("/first", "first.bin")
    second = server.asset("/second", "second.bin")
    config = tmp_path / "datasets.yaml"
    config.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "datasets": {
                    "fixture": {"enabled": True, "revision": "local", "assets": [first, second]},
                },
            }
        )
    )
    root = tmp_path / "data"
    with pytest.raises(ValueError, match="(?i)budget"):
        download.main(
            [
                "--config",
                str(config),
                "--root",
                str(root),
                "--execute",
                "--max-download-bytes",
                str(2 * len(server.data) - 1),
                "--connections",
                "4",
                "--progress",
                "off",
            ]
        )
    assert (root / "fixture" / "downloads" / "first.bin").read_bytes() == server.data
    assert not (root / "fixture" / "downloads" / "second.bin").exists()
    assert all(r["path"] == "/first" for r in server.requests)


@pytest.mark.parametrize("fallback", ["no-ranges", "no-validator", "weak-validator"])
def test_parallel_falls_back_to_one_serial_body_without_safe_ranges(tmp_path, archive_server, fallback):
    validator = "weak" if fallback == "weak-validator" else fallback != "no-validator"
    server = archive_server(ranges=fallback != "no-ranges", validator=validator)
    transferred = fetch(server, tmp_path)
    assert (tmp_path / "archive.bin").read_bytes() == server.data
    assert transferred == len(server.data)
    full_bodies = [r for r in server.requests if r["method"] == "GET" and r["range"] is None]
    bounded_requests = [r for r in server.requests if r["method"] == "GET" and r["range"] is not None]
    assert len(full_bodies) == 1
    assert len(bounded_requests) <= 1


@pytest.mark.parametrize("fault", ["range-start", "range-end", "range-total", "validator"])
def test_parallel_rejects_inconsistent_chunk_headers_without_publishing(tmp_path, archive_server, fault):
    server = archive_server(fault=fault)
    with pytest.raises(ValueError):
        fetch(server, tmp_path)
    assert not (tmp_path / "archive.bin").exists()
    assert not (tmp_path / "archive.bin.receipt.json").exists()


def test_parallel_checksum_failure_never_publishes(tmp_path, archive_server):
    server = archive_server()
    with pytest.raises(ValueError, match="sha256 mismatch"):
        fetch(server, tmp_path, asset={**server.asset(), "sha256": "0" * 64})
    assert not (tmp_path / "archive.bin").exists()
    assert not (tmp_path / "archive.bin.receipt.json").exists()


def test_parallel_interruption_resumes_only_completed_chunks(tmp_path, archive_server):
    server = archive_server(fault="disconnect")
    with pytest.raises((ValueError, OSError)):
        fetch(server, tmp_path)
    assert not (tmp_path / "archive.bin").exists()
    completed = {(r["start"], r["end"]) for r in server.completed if r["end"] > r["start"]}
    interrupted = {(r["start"], r["end"]) for r in server.interrupted}
    assert completed
    assert interrupted
    request_count = len(server.requests)
    server.fault = None
    fetch(server, tmp_path)
    retried = {
        (r["start"], r["end"])
        for r in server.requests[request_count:]
        if r["method"] == "GET" and r["end"] > r["start"]
    }
    assert completed.isdisjoint(retried)
    assert interrupted <= retried
    assert (tmp_path / "archive.bin").read_bytes() == server.data


def test_publisher_digest_allows_same_object_prefix_resume_from_another_mirror(tmp_path, archive_server):
    server = archive_server()
    prefix = 12345
    asset = server.asset("/official-mirror")
    (tmp_path / "archive.bin.part").write_bytes(server.data[:prefix])
    (tmp_path / "archive.bin.part.json").write_text(
        json.dumps({"url": server.asset("/original")["url"], "validator": '"archive-v1"'})
    )
    transferred = fetch(server, tmp_path, asset=asset)
    assert (tmp_path / "archive.bin").read_bytes() == server.data
    assert all(r["path"] == "/official-mirror" for r in server.requests)
    assert transferred < len(server.data)


def test_mirror_change_without_publisher_digest_preserves_partial_and_rejects_resume(tmp_path, archive_server):
    server = archive_server()
    prefix = server.data[:12345]
    asset = server.asset("/official-mirror")
    del asset["sha256"]
    partial = tmp_path / "archive.bin.part"
    sidecar = tmp_path / "archive.bin.part.json"
    partial.write_bytes(prefix)
    old_state = json.dumps({"url": server.asset("/original")["url"], "validator": '"archive-v1"'})
    sidecar.write_text(old_state)
    with pytest.raises(ValueError):
        fetch(server, tmp_path, asset=asset)
    assert partial.read_bytes() == prefix
    assert sidecar.read_text() == old_state
    assert not (tmp_path / "archive.bin").exists()


@pytest.mark.parametrize("fault", ["validator", "range-start", "range-end", "range-total", "encoding"])
def test_serial_resume_rejects_inconsistent_headers_before_changing_saved_prefix(tmp_path, archive_server, fault):
    server = archive_server(fault=fault)
    partial = tmp_path / "archive.bin.part"
    sidecar = tmp_path / "archive.bin.part.json"
    prefix = server.data[:12345]
    old_state = json.dumps({"url": server.asset()["url"], "validator": '"archive-v1"'})
    partial.write_bytes(prefix)
    sidecar.write_text(old_state)

    with pytest.raises(ValueError):
        fetch(server, tmp_path, connections=1)

    assert partial.read_bytes() == prefix
    assert sidecar.read_text() == old_state
    assert not (tmp_path / "archive.bin").exists()
    assert not (tmp_path / "archive.bin.receipt.json").exists()


def test_serial_resume_restarts_weak_only_validator_without_publisher_digest(tmp_path, archive_server):
    server = archive_server(validator="weak")
    asset = server.asset()
    del asset["sha256"]
    (tmp_path / "archive.bin.part").write_bytes(b"old object prefix")
    (tmp_path / "archive.bin.part.json").write_text(json.dumps({"url": asset["url"], "validator": 'W/"archive-v1"'}))

    transferred = fetch(server, tmp_path, asset=asset, connections=1)

    assert transferred == len(server.data)
    assert (tmp_path / "archive.bin").read_bytes() == server.data
    assert server.requests
    assert all(request["range"] is None for request in server.requests)
    assert all(request["if_range"] is None for request in server.requests)


@pytest.mark.parametrize("connections", [1, 4])
def test_asset_download_lock_contention_preserves_partial_data(tmp_path, archive_server, connections):
    server = archive_server()
    partial = tmp_path / "archive.bin.part"
    sidecar = tmp_path / "archive.bin.part.json"
    prefix = server.data[:12345]
    old_state = json.dumps({"url": server.asset()["url"], "validator": '"archive-v1"'})
    partial.write_bytes(prefix)
    sidecar.write_text(old_state)

    with (tmp_path / "archive.bin.download.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="Another downloader"):
            fetch(server, tmp_path, connections=connections)

    assert not server.requests
    assert partial.read_bytes() == prefix
    assert sidecar.read_text() == old_state
    assert not (tmp_path / "archive.bin").exists()
    assert not (tmp_path / "archive.bin.receipt.json").exists()


def test_serial_restart_crash_cannot_tag_stale_prefix_with_new_validator(tmp_path, archive_server, monkeypatch):
    server = archive_server(ranges=False)
    partial = tmp_path / "archive.bin.part"
    sidecar = tmp_path / "archive.bin.part.json"
    partial.write_bytes(b"old remote object prefix")
    old_state = json.dumps({"url": server.asset()["url"], "validator": '"archive-v0"'})
    sidecar.write_text(old_state)
    observed_empty_prefix = []

    def crash_before_state_write(path, value):
        assert path == sidecar
        assert value["validator"] == '"archive-v1"'
        assert partial.read_bytes() == b""
        observed_empty_prefix.append(True)
        raise RuntimeError("crash before new state write")

    with monkeypatch.context() as patch:
        patch.setattr(download, "_atomic_json", crash_before_state_write)
        with pytest.raises(RuntimeError, match="crash before new state write"):
            fetch(server, tmp_path, connections=1)

    assert observed_empty_prefix == [True]
    assert partial.read_bytes() == b""
    assert sidecar.read_text() == old_state
    assert not (tmp_path / "archive.bin").exists()
    assert not (tmp_path / "archive.bin.receipt.json").exists()
    assert server.requests[0]["if_range"] == '"archive-v0"'
    request_count = len(server.requests)

    assert fetch(server, tmp_path, connections=1) == len(server.data)
    assert (tmp_path / "archive.bin").read_bytes() == server.data
    assert all(request["range"] is None for request in server.requests[request_count:])

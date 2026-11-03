"""HTTP range transfers with durable chunks and legacy contiguous-prefix resume."""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import re
import shutil
import threading
import urllib.request
import uuid

from .common import file_sha256
from .progress import Progress

CHUNK_BYTES = 16 * 1024 * 1024


def _atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _validator(headers):
    etag = headers.get("ETag")
    if etag and not etag.startswith("W/"):
        return etag
    return headers.get("Last-Modified")


def _usable_validator(value):
    return isinstance(value, str) and bool(value) and not value.startswith("W/")


def _identity(asset):
    return {key: asset.get(key) for key in ("size_bytes", "sha256", "md5")}


def _chunk_paths(directory, start, end):
    base = directory / f"{start}-{end}"
    return base.with_suffix(".chunk"), base.with_suffix(".json"), base.with_suffix(".tmp")


def _valid_chunk(directory, start, end, generation):
    path, metadata, _ = _chunk_paths(directory, start, end)
    if not path.is_file() or not metadata.is_file():
        return False
    try:
        saved = json.loads(metadata.read_text(encoding="utf-8"))
        return (
            saved.get("generation") == generation
            and saved.get("start") == start
            and saved.get("end") == end
            and path.stat().st_size == end - start + 1
            and saved.get("sha256") == file_sha256(path)
        )
    except (OSError, ValueError, TypeError):
        return False


def _open_range(url, start, end, size, validator, timeout):
    headers = {"User-Agent": "verl-audio-data/2", "Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"}
    if validator:
        headers["If-Range"] = validator
    response = urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout)
    try:
        if response.getcode() == 200:
            # Do not read a potentially multi-GB body on a range refusal.
            response.close()
            return None
        if response.getcode() != 206:
            raise ValueError(f"Unexpected HTTP status for range download: {response.getcode()}")
        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
        if not match or tuple(map(int, match.groups())) != (start, end, size):
            raise ValueError(f"Invalid Content-Range: {response.headers.get('Content-Range')}")
        length = response.headers.get("Content-Length")
        if length is not None and int(length) != end - start + 1:
            raise ValueError("Range Content-Length does not match requested bytes")
        if response.headers.get("Content-Encoding", "identity").lower() not in {"", "identity"}:
            raise ValueError("Range server returned encoded bytes despite Accept-Encoding: identity")
        received_validator = _validator(response.headers)
        if validator and received_validator != validator:
            raise ValueError("Remote validator changed during range download; saved partial retained")
        if not received_validator:
            response.close()
            return None
        return response
    except BaseException:
        response.close()
        raise


def _download_chunk(url, start, end, size, validator, timeout, directory, generation, stop, counter, response=None):
    path, metadata, temporary = _chunk_paths(directory, start, end)
    try:
        if stop.is_set():
            return
        response = response or _open_range(url, start, end, size, validator, timeout)
        if response is None:
            raise ValueError("Range support or validator disappeared during parallel download; rerun to resume")
        digest = hashlib.sha256()
        with response, temporary.open("wb") as stream:
            remaining = end - start + 1
            while remaining:
                if stop.is_set():
                    return
                block = response.read(min(1024 * 1024, remaining))
                if not block:
                    raise ValueError(f"Incomplete range {start}-{end}; rerun to resume")
                # Disjoint ranges, no automatic retries, and the preflight sum
                # bound all worker payload reads by the invocation's byte cap.
                with counter["lock"]:
                    counter["bytes"] += len(block)
                stream.write(block)
                digest.update(block)
                remaining -= len(block)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        _atomic_json(
            metadata,
            {"generation": generation, "start": start, "end": end, "sha256": digest.hexdigest()},
        )
    finally:
        if response is not None:
            response.close()
        temporary.unlink(missing_ok=True)


def _merge(partial, state_path, directory, state, ranges, progress):
    parallel = state["parallel"]
    committed = parallel["committed"]
    with (
        partial.open("r+b") as target,
        Progress(
            "Merge completed download chunks", total=parallel["size"], unit="bytes", initial=committed, mode=progress
        ) as bar,
    ):
        # A crash between append/fsync and metadata commit leaves an uncommitted
        # tail. Its source chunk is still available, so safely append it again.
        target.truncate(committed)
        target.seek(committed)
        for start, end in ranges:
            if end < committed:
                continue
            if start != committed:
                raise ValueError("Noncontiguous chunk checkpoint; saved partial retained")
            path, metadata, _ = _chunk_paths(directory, start, end)
            with path.open("rb") as source:
                while block := source.read(1024 * 1024):
                    target.write(block)
                    bar.update(len(block))
            target.flush()
            os.fsync(target.fileno())
            committed = end + 1
            parallel["committed"] = committed
            _atomic_json(state_path, state)
            path.unlink(missing_ok=True)
            metadata.unlink(missing_ok=True)


def transfer(asset, partial, state_path, remaining_bytes, *, connections, timeout, progress):
    """Return payload bytes read, or None to use the serial implementation.

    Completed chunks are immutable and hashed before reuse. The saved prefix
    stays contiguous; the archive is not published until its normal verifier
    succeeds. Changing connections does not invalidate saved ranges.
    """
    size = asset.get("size_bytes")
    if not isinstance(size, int) or size <= 0:
        return None
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
    if not isinstance(state, dict):
        raise ValueError("Invalid download checkpoint; saved partial retained")
    saved = state.get("parallel")
    if saved and (saved.get("version") != 1 or saved.get("identity") != _identity(asset) or saved.get("size") != size):
        raise ValueError("Download checkpoint identity changed; use the original asset or a fresh directory")
    prefix = partial.stat().st_size if partial.is_file() else 0
    if saved:
        committed = saved["committed"]
        if prefix < committed:
            raise ValueError("Saved download prefix is truncated; use a fresh directory")
        # Ignore only a known uncommitted merge tail, never a legacy prefix.
        prefix = committed
    digest_available = bool(asset.get("sha256") or asset.get("md5"))
    same_url = state.get("url") == asset["url"]
    if (prefix or saved) and state.get("url") and not same_url and not digest_available:
        raise ValueError(
            "Cannot resume across mirrors without a publisher checksum; use the original mirror or a fresh directory"
        )
    resumable = same_url and _usable_validator(state.get("validator"))
    resumable = resumable or (digest_available and bool(state.get("url")))
    if prefix and not resumable:
        # The serial implementation may restart a legacy, unvalidated prefix.
        if saved:
            raise ValueError("Download checkpoint lacks a usable validator; saved partial retained")
        return None
    if prefix > size:
        raise ValueError("Partial file is larger than the expected archive")
    if not saved and (connections <= 1 or size - prefix <= CHUNK_BYTES):
        return None
    if saved:
        initial_prefix = saved["prefix"]
        chunk_bytes = saved["chunk_bytes"]
        generation = saved["generation"]
    else:
        initial_prefix, chunk_bytes, generation = prefix, CHUNK_BYTES, uuid.uuid4().hex
    if chunk_bytes <= 0 or not (0 <= initial_prefix <= prefix <= size):
        raise ValueError("Invalid parallel download checkpoint offsets")
    ranges = [(start, min(size - 1, start + chunk_bytes - 1)) for start in range(initial_prefix, size, chunk_bytes)]
    directory = partial.with_name(partial.name + ".segments")
    completed, pending = [], []
    with Progress("Verify saved download chunks", total=len(ranges), unit="chunks", mode=progress) as bar:
        for start, end in ranges:
            if end < prefix:
                pass
            elif _valid_chunk(directory, start, end, generation):
                completed.append((start, end))
            else:
                pending.append((start, end))
            bar.update()
    required = sum(end - start + 1 for start, end in pending)
    if required > remaining_bytes:
        raise ValueError(f"Byte budget too small for {asset['filename']}: needs {required}, has {remaining_bytes}")
    first_response = None
    validator = state.get("validator") if same_url and _usable_validator(state.get("validator")) else None
    if pending:
        start, end = pending[0]
        with Progress(f"Request ranges {asset['filename']}", total=1, unit="requests", mode=progress) as bar:
            first_response = _open_range(asset["url"], start, end, size, validator, timeout)
            bar.update()
        if first_response is None:
            if saved or completed:
                raise ValueError(
                    "Server no longer supports resumable ranges; use the original source or a fresh directory"
                )
            return None
        validator = _validator(first_response.headers)
    try:
        if not saved:
            saved = {
                "version": 1,
                "size": size,
                "prefix": prefix,
                "committed": prefix,
                "chunk_bytes": chunk_bytes,
                "generation": generation,
                "identity": _identity(asset),
            }
        state = {"url": asset["url"], "validator": validator, "parallel": saved}
        directory.mkdir(exist_ok=True)
        if not partial.exists():
            partial.touch()
        if partial.stat().st_size > prefix:
            with partial.open("r+b") as stream:
                stream.truncate(prefix)
        _atomic_json(state_path, state)
        already = prefix + sum(end - start + 1 for start, end in completed)
        stop, counter = threading.Event(), {"bytes": 0, "lock": threading.Lock()}
        with Progress(
            f"Download {asset['filename']} ({min(connections, len(pending))} connections)",
            total=size,
            unit="bytes",
            initial=already,
            mode=progress,
        ) as bar:
            pool = concurrent.futures.ThreadPoolExecutor(max_workers=connections)
            futures = set()
            try:
                for number, (start, end) in enumerate(pending):
                    futures.add(
                        pool.submit(
                            _download_chunk,
                            asset["url"],
                            start,
                            end,
                            size,
                            validator,
                            timeout,
                            directory,
                            generation,
                            stop,
                            counter,
                            first_response if number == 0 else None,
                        )
                    )
                first_response = None  # Ownership has moved to the worker.
                waiting = futures
                while waiting:
                    done, waiting = concurrent.futures.wait(
                        waiting, timeout=0.2, return_when=concurrent.futures.FIRST_COMPLETED
                    )
                    with counter["lock"]:
                        current = already + counter["bytes"]
                    bar.update(current - bar.current)
                    for future in done:
                        future.result()
            finally:
                stop.set()
                for future in futures:
                    future.cancel()
                pool.shutdown(wait=True, cancel_futures=True)
                with counter["lock"]:
                    current = already + counter["bytes"]
                bar.update(current - bar.current)
        _merge(partial, state_path, directory, state, ranges, progress)
        return counter["bytes"]
    finally:
        if first_response is not None:
            first_response.close()


def cleanup(partial):
    """Remove only this asset's range state after successful final verification."""
    directory = partial.with_name(partial.name + ".segments")
    if directory.is_dir():
        shutil.rmtree(directory)

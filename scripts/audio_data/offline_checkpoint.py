"""Transactional, bounded-memory checkpoints for the offline audio filter."""

from __future__ import annotations

import fcntl
import json
import os
import stat
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

from .common import file_sha256, read_jsonl, stable_hash, write_jsonl
from .progress import Progress

# Keep UI/cadence options out of identity; spelling an effective default explicitly
# must not invalidate an otherwise identical run.
DEFAULT_POLICY = {
    "allow_noncommercial": False,
    "min_duration": 0.1,
    "max_duration": 120.0,
    "asr_hypotheses": None,
    "max_wer": None,
    "max_cer": None,
    "require_reference_check": False,
    "reference_check_splits": None,
    "max_file_bytes": 256 * 1024 * 1024,
}


def _file_identity(path):
    if path is None:
        return None
    requested = Path(os.path.abspath(Path(path).expanduser()))
    return {"path": str(requested), "resolved": str(requested.resolve()), "sha256": file_sha256(requested)}


def _identity(input_path, output_path, holdout_manifest, hypotheses_path, policy):
    effective = DEFAULT_POLICY | policy
    effective["asr_hypotheses"] = stable_hash(effective["asr_hypotheses"] or {})
    if effective["reference_check_splits"] is not None:
        effective["reference_check_splits"] = sorted(set(effective["reference_check_splits"]))
    root = Path(__file__).resolve().parents[2]
    code = [
        Path(__file__),
        Path(__file__).with_name("filter_offline.py"),
        Path(__file__).with_name("common.py"),
        root / "verl/utils/reward_score/audio.py",
    ]
    return {
        "version": 1,
        "input": _file_identity(input_path),
        "output": str(Path(output_path).resolve()),
        "holdout": _file_identity(holdout_manifest),
        "hypotheses_file": _file_identity(hypotheses_path),
        "policy": effective,
        "code": {str(path): file_sha256(path) for path in code},
        "runtime": [np.__version__, sf.__version__, sf.__libsndfile_version__],
        "cwd": os.getcwd(),
    }


def _snapshot(path):
    requested = Path(os.path.abspath(Path(path).expanduser()))
    result = {"path": str(requested)}
    try:
        result["resolved"] = str(requested.resolve())
        info = requested.stat()
    except (OSError, RuntimeError) as exc:
        result["error"] = [type(exc).__name__, getattr(exc, "errno", None)]
        return result
    result["stat"] = [
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
        info.st_mode,
    ]
    return result


def _hashable(snapshot, max_file_bytes):
    info = snapshot.get("stat")
    return info is not None and stat.S_ISREG(info[-1]) and info[2] <= max_file_bytes


def _remember_media(connection, before, *, max_file_bytes, known_hash=None):
    if before is None:
        return
    path = before["path"]
    if _snapshot(path) != before:
        raise ValueError(f"Audio changed during checking: {path}; start a fresh checkpoint")
    existing = connection.execute("SELECT snapshot, sha256 FROM files WHERE path=?", (path,)).fetchone()
    if existing:
        if json.loads(existing[0]) != before:
            raise ValueError(f"Audio changed during this run: {path}; start a fresh checkpoint")
        return
    digest = None
    if _hashable(before, max_file_bytes):
        digest = known_hash or file_sha256(path)
        if _snapshot(path) != before:
            raise ValueError(f"Audio changed during hashing: {path}; start a fresh checkpoint")
    connection.execute("INSERT INTO files VALUES (?, ?, ?)", (path, json.dumps(before), digest))


def _record_snapshot(record):
    path = record.get("audio_path")
    return _snapshot(path) if isinstance(path, str) and path.strip() else None


def _validate_media(connection, *, progress, rehash):
    total = connection.execute("SELECT COUNT(*) FROM files").fetchone()[0]
    label = "Verify checkpoint audio" if rehash else "Validate audio before publication"
    with Progress(label, total=total, unit="files", mode=progress) as bar:
        for path, encoded, digest in connection.execute("SELECT path, snapshot, sha256 FROM files ORDER BY path"):
            previous = json.loads(encoded)
            if _snapshot(path) != previous:
                raise ValueError(f"Checkpoint audio has changed: {path}; use a fresh checkpoint directory or reset it")
            if rehash and digest and (file_sha256(path) != digest or _snapshot(path) != previous):
                raise ValueError(f"Checkpoint audio content has changed: {path}; use a fresh checkpoint or reset it")
            bar.update()


def _save_state(connection, state):
    connection.execute("INSERT OR REPLACE INTO state VALUES ('run', ?)", (json.dumps(state),))
    connection.commit()


def _status(directory, state):
    # Informational only; SQLite is authoritative if a crash leaves this behind.
    payload = {key: state[key] for key in ("phase", "checked", "finalized", "offset", "line")}
    fd, temporary = tempfile.mkstemp(prefix=".status-", dir=directory)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
        os.replace(temporary, directory / "status.json")
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _commit(connection, directory, state):
    _save_state(connection, state)
    _status(directory, state)


def _scan(api, connection, directory, state, input_path, policy, interval, progress):
    if state["phase"] != "scan":
        return
    cache = {}
    with (
        open(input_path, "rb") as source,
        Progress(
            f"Inspect audio / ASR references ({state['checked']:,} records)",
            total=Path(input_path).stat().st_size,
            unit="bytes",
            initial=state["offset"],
            mode=progress,
        ) as bar,
    ):
        source.seek(state["offset"])
        for line in source:
            state["line"] += 1
            state["offset"] += len(line)
            if not line.strip():
                bar.update(len(line))
                continue
            try:
                original = json.loads(line)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{input_path}:{state['line']}: invalid JSON") from exc
            if not isinstance(original, dict):
                raise ValueError(f"{input_path}:{state['line']}: expected a JSON object")
            before = _record_snapshot(original)
            checked = api._check_record(original, audio_cache=cache, **policy)
            # Only trust hashes produced by this run's inspector. An incoming
            # quarantined record can still carry old hashes from a prior scan.
            inspected = {}
            if before:
                info = before.get("stat")
                key = (before.get("resolved"), info[3], info[2]) if info else (before.get("resolved"),)
                inspected = cache.get(key, {})
            _remember_media(
                connection,
                before,
                max_file_bytes=policy.get("max_file_bytes", DEFAULT_POLICY["max_file_bytes"]),
                known_hash=inspected.get("audio_sha256"),
            )
            api._register(connection, checked)
            state["checked"] += 1
            connection.execute("INSERT INTO checked VALUES (?, ?)", (state["checked"], json.dumps(checked)))
            bar.label = f"Inspect audio / ASR references ({state['checked']:,} records)"
            bar.update(len(line))
            if state["checked"] % interval == 0:
                _commit(connection, directory, state)
        state["phase"] = "holdouts"
        _commit(connection, directory, state)


def _holdouts(api, connection, directory, state, manifest, policy, progress):
    if state["phase"] != "holdouts":
        return
    # One transaction includes all holdout registrations; a partial pass must not
    # mark an external benchmark as covered. Progress still updates during it.
    if manifest:
        with Progress("Index benchmark holdouts", mode=progress) as bar:

            def tracked_records():
                for original in read_jsonl(manifest):
                    before = _record_snapshot(original)
                    yield original
                    _remember_media(
                        connection,
                        before,
                        max_file_bytes=policy.get("max_file_bytes", DEFAULT_POLICY["max_file_bytes"]),
                    )
                    bar.update()

            api._register_holdouts(connection, tracked_records(), policy)
    state["phase"] = "index"
    _commit(connection, directory, state)


def _finalize(api, connection, directory, state, interval, progress):
    if state["phase"] == "index":
        with Progress("Build global duplicate/leakage indexes", total=1, unit="passes", mode=progress) as bar:
            api._index(connection)
            bar.update()
        state["phase"] = "finalize"
        _commit(connection, directory, state)
    if state["phase"] != "finalize":
        return
    with Progress(
        "Global duplicate/leakage decisions", total=state["checked"], initial=state["finalized"], mode=progress
    ) as bar:
        # Fetch one batch at a time so commits never leave a long-lived cursor
        # depending on partially updated tables.
        while state["finalized"] < state["checked"]:
            rows = connection.execute(
                "SELECT ordinal, record FROM checked WHERE ordinal>? ORDER BY ordinal LIMIT ?",
                (state["finalized"], interval),
            ).fetchall()
            for ordinal, encoded in rows:
                record = api._finish(connection, json.loads(encoded))
                connection.execute("INSERT INTO finished VALUES (?, ?)", (ordinal, json.dumps(record)))
                state["finalized"] = ordinal
                bar.update()
            _commit(connection, directory, state)
        state["phase"] = "publish"
        _commit(connection, directory, state)


def _publish(connection, directory, state, output_path, policy, holdout_manifest, progress, verify):
    counts, reasons = Counter(), Counter()
    with Progress("Write filtered manifest", total=state["checked"], mode=progress) as bar:

        def records():
            for (encoded,) in connection.execute("SELECT record FROM finished ORDER BY ordinal"):
                record = json.loads(encoded)
                stage = record["qc"]["offline"]
                counts[stage["decision"]] += 1
                reasons.update(stage["reasons"])
                yield record
                bar.update()
            # Runs inside the atomic writer before its final os.replace.
            verify()

        write_jsonl(output_path, records())
    report = {
        "records": sum(counts.values()),
        "decisions": dict(counts),
        "reasons": dict(reasons),
        "policy": {key: value for key, value in policy.items() if key != "asr_hypotheses"},
        "holdout_manifest": str(holdout_manifest) if holdout_manifest else None,
    }
    # Use the same atomic writer contract for the report (a single JSON object
    # is valid JSON as well as a one-line JSONL file).
    write_jsonl(str(output_path) + ".report.json", [report])
    state["phase"] = "complete"
    _commit(connection, directory, state)
    return report


def run_filter(
    api,
    input_path,
    output_path,
    *,
    directory,
    checkpoint_interval,
    progress,
    holdout_manifest,
    hypotheses_path,
    reset,
    policy,
):
    if checkpoint_interval <= 0:
        raise ValueError("checkpoint_interval must be positive")
    unknown = set(policy) - set(DEFAULT_POLICY)
    if unknown:
        raise TypeError(f"Unexpected filter policy options: {sorted(unknown)}")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f"Another filter is using checkpoint directory {directory}") from exc
        if reset:
            for name in (
                "checkpoint.sqlite3",
                "checkpoint.sqlite3-journal",
                "checkpoint.sqlite3-wal",
                "checkpoint.sqlite3-shm",
                "status.json",
            ):
                (directory / name).unlink(missing_ok=True)
        with Progress("Fingerprint manifests and filter settings", total=1, unit="passes", mode=progress) as bar:
            identity = _identity(input_path, output_path, holdout_manifest, hypotheses_path, policy)
            bar.update()
        connection = api._database(directory / "checkpoint.sqlite3")
        try:
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT)")
            connection.execute("CREATE TABLE IF NOT EXISTS checked (ordinal INTEGER PRIMARY KEY, record TEXT)")
            connection.execute("CREATE TABLE IF NOT EXISTS finished (ordinal INTEGER PRIMARY KEY, record TEXT)")
            connection.execute("CREATE TABLE IF NOT EXISTS files (path TEXT PRIMARY KEY, snapshot TEXT, sha256 TEXT)")
            row = connection.execute("SELECT value FROM state WHERE key='run'").fetchone()
            if row:
                state = json.loads(row[0])
                if state["identity"] != identity:
                    raise ValueError(
                        "Checkpoint inputs, policy, or filter code changed; "
                        "use a fresh checkpoint directory or --reset-checkpoint"
                    )
                if progress == "on" or (progress == "auto" and sys.stderr.isatty()):
                    print(
                        f"Resume checkpoint {directory}: phase={state['phase']}, "
                        f"checked={state['checked']:,}, finalized={state['finalized']:,}",
                        file=sys.stderr,
                    )
                _validate_media(connection, progress=progress, rehash=True)
            else:
                state = {"identity": identity, "phase": "scan", "checked": 0, "finalized": 0, "offset": 0, "line": 0}
                _commit(connection, directory, state)
            _scan(api, connection, directory, state, input_path, policy, checkpoint_interval, progress)
            _holdouts(api, connection, directory, state, holdout_manifest, policy, progress)
            _finalize(api, connection, directory, state, checkpoint_interval, progress)

            def verify():
                if _identity(input_path, output_path, holdout_manifest, hypotheses_path, policy) != identity:
                    raise ValueError(
                        "Filter inputs changed during execution; no new output published; use a fresh checkpoint"
                    )
                _validate_media(connection, progress=progress, rehash=False)

            return _publish(connection, directory, state, output_path, policy, holdout_manifest, progress, verify)
        finally:
            connection.close()  # rolls back any unfinished batch on exceptions

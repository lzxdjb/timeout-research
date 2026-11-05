"""Durable, ordered Qwen audit results with independent concurrent completion."""

from __future__ import annotations

import fcntl
import json
import os
import sqlite3
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import soundfile as sf

from .common import file_sha256, read_jsonl
from .offline_checkpoint import _snapshot
from .progress import Progress


def failed_review(result):
    reasons = result.get("qc", {}).get("qwen", {}).get("reasons", [])
    return any(reason.startswith(("review_failed:", "model_discovery_failed:")) for reason in reasons)


def discover_reviewer(api, config, *, progress, interval):
    """Keep logs alive while model discovery waits through retries/timeouts."""
    with Progress("Qwen audit: discovering model", mode=progress, interval=interval, unit="models") as bar:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(api.QwenReviewer, config)
            while not future.done():
                wait([future], timeout=interval)
                bar.refresh()
            reviewer = future.result()
            bar.update()
    return reviewer


def _identity(api, args, report_path):
    input_path = args.input.resolve()
    paths = [
        Path(__file__),
        Path(api.__file__),
        Path(__file__).with_name("common.py"),
        Path(__file__).with_name("offline_checkpoint.py"),
    ]
    return {
        "version": 1,
        "input": {"path": os.path.abspath(args.input), "resolved": str(input_path), "sha256": file_sha256(input_path)},
        "output": str(args.output.resolve()),
        "report": str(report_path.resolve()),
        "base_url": args.base_url.rstrip("/"),
        "model": args.model,
        "model_revision": args.model_revision,
        "max_samples": args.max_samples,
        "max_tokens": args.max_tokens,
        "max_audio_bytes": args.max_audio_bytes,
        "max_duration": args.max_duration,
        "prompt_version": api.PROMPT_VERSION,
        "prompt_sha256": api.stable_hash([api.BLIND_PROMPT, api.SUPPORT_PROMPT]),
        "code": {str(path.resolve()): file_sha256(path) for path in paths},
        "runtime": [np.__version__, sf.__version__, sf.__libsndfile_version__],
    }


@contextmanager
def _connection(directory, *, reset):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f"Another audit is using checkpoint directory {directory}") from exc
        if reset:
            for name in (
                "audit.sqlite3",
                "audit.sqlite3-journal",
                "audit.sqlite3-wal",
                "audit.sqlite3-shm",
                "status.json",
            ):
                (directory / name).unlink(missing_ok=True)
        connection = sqlite3.connect(directory / "audit.sqlite3")
        try:
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT)")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS jobs (ordinal INTEGER PRIMARY KEY, original TEXT, selected INTEGER, "
                "kind TEXT, result TEXT, reusable INTEGER DEFAULT 0, media TEXT, digest TEXT)"
            )
            yield connection
        finally:
            connection.close()


def _get(connection, key):
    row = connection.execute("SELECT value FROM state WHERE key=?", (key,)).fetchone()
    return json.loads(row[0]) if row else None


def _put(connection, key, value):
    connection.execute("INSERT OR REPLACE INTO state VALUES (?, ?)", (key, json.dumps(value, ensure_ascii=False)))


def _summary(connection):
    result = {
        "records": 0,
        "selected": 0,
        "completed": 0,
        "review_errors": 0,
        "accept": 0,
        "reject": 0,
        "quarantine": 0,
    }
    result["records"], result["selected"] = connection.execute(
        "SELECT COUNT(*), COALESCE(SUM(selected), 0) FROM jobs"
    ).fetchone()
    for encoded, reusable in connection.execute(
        "SELECT result, reusable FROM jobs WHERE selected=1 AND result IS NOT NULL"
    ):
        record = json.loads(encoded)
        if reusable:
            result["completed"] += 1
            decision = record["qc"]["qwen"]["decision"]
            result[decision] += 1
        else:
            result["review_errors"] += 1
    return result


def _commit(api, connection, directory, *, phase, resumed, summary=None):
    _put(connection, "phase", phase)
    connection.commit()
    if summary is None:
        summary = _summary(connection)
    api._atomic_cache_write(directory / "status.json", {"phase": phase, **summary, "resumed": resumed})


def _index(api, connection, args):
    if _get(connection, "indexed"):
        return
    selected = 0
    with Progress("Index Qwen audit candidates", mode=args.progress, interval=args.progress_interval) as bar:
        for ordinal, original in enumerate(read_jsonl(args.input), 1):
            eligible = original.get("qc", {}).get("offline", {}).get("decision") == "accept"
            chosen = eligible and selected < args.max_samples
            kind = "review" if chosen else ("budget" if eligible else "offline")
            selected += chosen
            result = (
                None
                if chosen
                else (api._quarantine(original, "qwen_sample_limit_not_reviewed") if eligible else original)
            )
            connection.execute(
                "INSERT INTO jobs (ordinal, original, selected, kind, result, reusable) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    ordinal,
                    json.dumps(original, ensure_ascii=False),
                    int(chosen),
                    kind,
                    json.dumps(result, ensure_ascii=False) if result is not None else None,
                    int(not chosen),
                ),
            )
            bar.update()
        _put(connection, "indexed", True)
        connection.commit()


def _media_before(original, config):
    path = original.get("audio_path")
    if not isinstance(path, str) or not path.strip():
        return None
    path = Path(path)
    if not path.is_absolute():
        path = config.audio_base / path
    return _snapshot(path)


def _media_evidence(original, result, before):
    if before is None:
        raise ValueError("Successful audit has no audio path")
    if _snapshot(before["path"]) != before:
        raise ValueError(f"Audio changed during Qwen audit: {before['path']}; use a fresh checkpoint")
    digest = file_sha256(before["path"])
    if _snapshot(before["path"]) != before:
        raise ValueError(f"Audio changed during checkpoint hashing: {before['path']}")
    if digest != original.get("audio_sha256") or digest != result.get("qc", {}).get("qwen", {}).get("audio_sha256"):
        raise ValueError("Successful review does not match the actual audio bytes; rerun offline checks")
    return digest


def _verify_media(connection, args, *, rehash):
    total = connection.execute("SELECT COUNT(*) FROM jobs WHERE selected=1 AND reusable=1").fetchone()[0]
    label = "Verify audit checkpoint audio" if rehash else "Validate audit audio before publication"
    with Progress(label, total=total, mode=args.progress, interval=args.progress_interval, unit="files") as bar:
        for encoded, digest in connection.execute("SELECT media, digest FROM jobs WHERE selected=1 AND reusable=1"):
            before = json.loads(encoded)
            path = before["path"]
            if _snapshot(path) != before:
                raise ValueError(f"Checkpoint audio has changed: {path}; use a fresh checkpoint or --reset-checkpoint")
            if rehash and (file_sha256(path) != digest or _snapshot(path) != before):
                raise ValueError(f"Checkpoint audio content has changed: {path}")
            bar.update()


def _run_jobs(api, connection, args, config, directory, reviewer, resumed):
    summary = _summary(connection)
    # Update these counters alongside SQL mutations: scanning full result JSON
    # at every default per-record commit would make status reporting quadratic.
    _commit(api, connection, directory, phase="review", resumed=resumed, summary=summary)
    selected, completed = summary["selected"], summary["completed"]
    decisions = Counter({key: summary[key] for key in ("accept", "reject", "quarantine")})
    errors, cache_hits, since_commit, attempted = 0, 0, 0, 0
    pending = {}
    last_ordinal = 0
    exhausted = False
    with Progress(
        "Qwen audit", total=selected, initial=completed, mode=args.progress, interval=args.progress_interval
    ) as bar:

        def refresh(*, force=False, amount=0):
            bar.label = (
                f"Qwen audit processed={completed} selected={selected} resumed={resumed} "
                f"accept={decisions['accept']} reject={decisions['reject']} quarantine={decisions['quarantine']} "
                f"errors={errors} cache_hits={cache_hits} pending={len(pending)} concurrency={args.concurrency}"
            )
            if amount:
                bar.update(amount)
            else:
                bar.refresh(force=force)

        refresh(force=True)
        executor = ThreadPoolExecutor(max_workers=args.concurrency)
        try:
            while pending or not exhausted:
                while not exhausted and len(pending) < 2 * args.concurrency:
                    # Fetch a bounded batch before mutating jobs; do not hold a
                    # live SELECT cursor across updates/commits on its table.
                    rows = connection.execute(
                        "SELECT ordinal, original FROM jobs WHERE selected=1 AND reusable=0 AND ordinal>? "
                        "ORDER BY ordinal LIMIT ?",
                        (last_ordinal, 2 * args.concurrency - len(pending)),
                    ).fetchall()
                    if not rows:
                        exhausted = True
                        break
                    for ordinal, encoded in rows:
                        last_ordinal = ordinal
                        original = json.loads(encoded)
                        before = _media_before(original, config)
                        pending[executor.submit(reviewer.review, original)] = (ordinal, original, before)
                        attempted += 1
                if not pending:
                    break
                done, _ = wait(pending, timeout=args.progress_interval, return_when=FIRST_COMPLETED)
                for future in done:
                    ordinal, original, before = pending.pop(future)
                    result = future.result()
                    failure = failed_review(result)
                    stage = result.get("qc", {}).get("qwen", {})
                    digest = None if failure else _media_evidence(original, result, before)
                    connection.execute(
                        "UPDATE jobs SET result=?, reusable=?, media=?, digest=? WHERE ordinal=?",
                        (
                            json.dumps(result, ensure_ascii=False),
                            int(not failure),
                            json.dumps(before) if before else None,
                            digest,
                            ordinal,
                        ),
                    )
                    decisions[stage.get("decision", "quarantine")] += 1
                    errors += int(failure)
                    if failure:
                        summary["review_errors"] += 1
                    else:
                        summary["completed"] += 1
                        summary[stage["decision"]] += 1
                    cache_hits += stage.get("cache_hit") is True
                    completed += 1
                    refresh(amount=1)
                    since_commit += 1
                    if since_commit >= args.checkpoint_interval:
                        _commit(api, connection, directory, phase="review", resumed=resumed, summary=summary)
                        since_commit = 0
                refresh(force=not done)
            _commit(api, connection, directory, phase="publish", resumed=resumed, summary=summary)
            refresh(force=True)
        finally:
            for future in pending:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
    return attempted, cache_hits


def run_audit(api, args, config, report_path):
    directory = Path(args.checkpoint_dir)
    started = time.monotonic()
    with _connection(directory, reset=args.reset_checkpoint) as connection:
        with Progress(
            "Fingerprint Qwen audit inputs",
            mode=args.progress,
            interval=args.progress_interval,
            total=1,
            unit="manifests",
        ) as bar:
            identity = _identity(api, args, report_path)
            bar.update()
        saved = _get(connection, "identity")
        if saved is not None and saved != identity:
            raise ValueError(
                "Checkpoint input, audit policy, or code changed; use a fresh checkpoint or --reset-checkpoint"
            )
        if saved is None:
            _put(connection, "identity", identity)
            connection.commit()
        _index(api, connection, args)
        summary = _summary(connection)
        resumed = summary["completed"]
        _verify_media(connection, args, rehash=True)
        selected = summary["selected"]
        reviewer = (
            discover_reviewer(api, config, progress=args.progress, interval=args.progress_interval)
            if selected
            else None
        )
        if reviewer is not None:
            model_config = reviewer._request_config()
            old_model = _get(connection, "model_config")
            if old_model is not None and (reviewer.discovery_error or old_model != model_config):
                raise ValueError("Checkpoint model identity changed or cannot be verified; no saved reviews reused")
            if not reviewer.discovery_error:
                _put(connection, "model_config", model_config)
                connection.commit()
            # Request failures remain retryable. Recompute only those and
            # unfinished rows; substantive uncertain/reject decisions survive.
            connection.execute("UPDATE jobs SET result=NULL, media=NULL, digest=NULL WHERE selected=1 AND reusable=0")
            connection.commit()
            attempted, cache_hits = _run_jobs(api, connection, args, config, directory, reviewer, resumed)
        else:
            attempted, cache_hits, model_config = 0, 0, None
        counts, decisions = Counter(), Counter()
        with Progress(
            "Write audited manifest",
            total=_summary(connection)["records"],
            mode=args.progress,
            interval=args.progress_interval,
        ) as bar:

            def records():
                for encoded, chosen, kind in connection.execute(
                    "SELECT result, selected, kind FROM jobs ORDER BY ordinal"
                ):
                    if encoded is None:
                        raise ValueError("Incomplete audit checkpoint; final manifest was not published")
                    record = json.loads(encoded)
                    counts["records"] += 1
                    counts["selected"] += chosen
                    counts["budget_skipped"] += kind == "budget"
                    counts["offline_skipped"] += kind == "offline"
                    counts["review_errors"] += chosen and failed_review(record)
                    decisions[record.get("qc", {}).get("qwen", {}).get("decision", "not_reviewed")] += 1
                    yield record
                    bar.update()
                if _identity(api, args, report_path) != identity:
                    raise ValueError("Audit inputs changed during execution; old output preserved")
                _verify_media(connection, args, rehash=False)

            api.write_jsonl(args.output, records())
        report = {
            "schema_version": 1,
            "input": str(args.input),
            "output": str(args.output),
            **{
                key: counts[key]
                for key in ("records", "selected", "offline_skipped", "budget_skipped", "review_errors")
            },
            "cache_hits": cache_hits,
            "resumed": resumed,
            "attempted_this_run": attempted,
            "qwen_decisions": dict(decisions),
            "concurrency": args.concurrency,
            "max_in_flight": 2 * args.concurrency,
            "max_samples": args.max_samples,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "model_config": model_config,
            "model_discovery_error": reviewer.discovery_error if reviewer else None,
            "model_service_error": reviewer.service_error if reviewer else None,
            "checkpoint_dir": str(directory.resolve()),
        }
        api._atomic_cache_write(report_path, report)
        _commit(
            api, connection, directory, phase="retry_errors" if counts["review_errors"] else "complete", resumed=resumed
        )
        return report

"""Checkpoint-backed, provisional live outputs for the ReBench filter."""

from __future__ import annotations

import bisect
import collections
import json
import os
import queue
import threading
import time
from pathlib import Path

from scripts import swe_tool_online_benchmark as audit
from scripts.swe_reference_reliability import validate_repeat_qualification


class LiveResults:
    """One bounded writer; terminal checkpoints, not live files, own the proof."""

    def __init__(
        self,
        output: Path,
        dataset: Path,
        signature: str,
        source_sha: str,
        cases: list[dict],
        states: dict,
        stop: threading.Event,
        *,
        flush_seconds=5.0,
        part_rows=256,
        part_seconds=60.0,
    ):
        import pyarrow.parquet as pq

        self.output, self.dataset = output, dataset
        self.signature, self.source_sha, self.stop = signature, source_sha, stop
        self.flush_seconds, self.part_rows, self.part_seconds = flush_seconds, part_rows, part_seconds
        self.items = {c["inventory"]["id"]: c["inventory"] for c in cases}
        self.recovered = [(key, state["qualified_sha256"]) for key, state in states.items() if state.get("terminal")]
        self.checkpoints = dict(self.recovered)
        self.parts_dir = output / "live_parquet_parts"
        self.parts_dir.mkdir(exist_ok=True)
        self.parquet = pq.ParquetFile(dataset)
        self.source_stat = self._source_stat()
        self.offsets = [0]
        for i in range(self.parquet.num_row_groups):
            self.offsets.append(self.offsets[-1] + self.parquet.metadata.row_group(i).num_rows)
        self.parts, self.exported = [], set()
        self._recover_parts()
        self.events = queue.Queue(maxsize=256)
        self.error = None
        self.seen, self.pending = set(), []
        self.counts = collections.Counter()
        self.last_flush = self.last_part = time.monotonic()
        self.closed = False
        self.thread = threading.Thread(target=self._run, name="rebench-live-results", daemon=True)
        self.thread.start()

    def _source_stat(self):
        stat = self.dataset.stat()
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns

    def _record(self, key, expected):
        path = self.output / "cases" / key / "qualified_record.json"
        if audit.file_sha256(path) != expected:
            raise ValueError("live qualification checkpoint changed")
        record = json.loads(path.read_text())
        item = self.items[key]
        if (
            record.get("id") != key
            or record.get("row") != item["row"]
            or record.get("task_id") != item["task_id"]
            or record.get("signature") != self.signature
            or record.get("dataset_sha256") != self.source_sha
            or record.get("complete") is not True
            or record.get("source_match") is not True
        ):
            raise ValueError("live qualification identity changed")
        if record["status"] == "reference_verified":
            validate_repeat_qualification(record)
        return record

    def _recover_parts(self):
        """Reuse closed parts only if bound to current durable terminal proofs."""
        import pyarrow.parquet as pq

        for receipt_path in sorted(self.parts_dir.glob("part-*.parquet.audit.json")):
            receipt = json.loads(receipt_path.read_text())
            path = receipt_path.with_suffix("").with_suffix("")
            if (
                receipt.get("version") != "swe-rebench-live-part-v1"
                or receipt.get("signature") != self.signature
                or receipt.get("source_sha256") != self.source_sha
                or receipt.get("provisional") is not True
                or receipt.get("repeat_required") is not True
                or audit.file_sha256(path) != receipt.get("output_sha256")
            ):
                raise ValueError("live parquet part identity or content changed")
            proofs = receipt["checkpoints"]
            if not proofs or len({p["id"] for p in proofs}) != len(proofs):
                raise ValueError("empty or duplicate live parquet part proofs")
            rows = []
            for proof in proofs:
                key, sha = proof["id"], proof["sha256"]
                if key in self.exported or self.checkpoints.get(key) != sha:
                    raise ValueError("live parquet part is not bound to a terminal checkpoint")
                record = self._record(key, sha)
                if record["status"] != "reference_verified":
                    raise ValueError("live parquet part contains an unqualified task")
                rows.append(record["row"])
                self.exported.add(key)
            parquet = pq.ParquetFile(path)
            if (
                rows != receipt.get("retained_row_indices")
                or rows != sorted(rows)
                or parquet.metadata.num_rows != len(rows)
                or not parquet.schema_arrow.equals(self.parquet.schema_arrow, check_metadata=True)
            ):
                raise ValueError("live parquet part schema or row count changed")
            self.parts.append(self._part_description(path, receipt))
        # A crash between publishing a part and its receipt leaves no admitted
        # proof. Discard that orphan and regenerate from terminal checkpoints.
        for path in self.parts_dir.glob("part-*.parquet"):
            if not path.with_suffix(path.suffix + ".audit.json").exists():
                path.unlink()
        for path in self.parts_dir.glob(".part-*.tmp"):
            path.unlink()

    @staticmethod
    def _part_description(path, receipt):
        return {
            "path": str(path.resolve()),
            "rows": receipt["rows"],
            "output_sha256": receipt["output_sha256"],
            "receipt": str(path.with_suffix(path.suffix + ".audit.json").resolve()),
        }

    def check(self):
        if self.error is not None:
            raise RuntimeError("live result writer failed; no final dataset exported") from self.error

    def submit(self, key: str, sha: str):
        self.check()
        if self.closed:
            raise RuntimeError("live result writer is closed")
        while True:
            self.check()
            try:
                self.events.put((key, sha), timeout=0.1)
                break
            except queue.Full:
                continue
        self.check()

    def close(self):
        if not self.closed:
            self.closed = True
            while self.thread.is_alive():
                try:
                    self.events.put(None, timeout=0.1)
                    break
                except queue.Full:
                    continue
            self.thread.join()
        self.check()

    def _summary(self, *, rebuilding=False, error=None):
        snapshot = {
            "version": "swe-rebench-live-results-v1",
            "signature": self.signature,
            "source_sha256": self.source_sha,
            "complete": False,
            "provisional": True,
            "rebuilding": rebuilding,
            "total": len(self.items),
            "terminal": len(self.seen),
            "reference_verified": self.counts["reference_verified"],
            "rejected": len(self.seen) - self.counts["reference_verified"],
            "counts": dict(self.counts),
            "parquet_rows": len(self.exported),
            "pending_parquet_rows": len(self.pending),
            "parts": self.parts,
            "updated_at_unix": time.time(),
        }
        if error:
            snapshot["writer_error"] = error
        audit.audit_write_json(self.output / "live_summary.json", snapshot)

    def _append(self, event, ledger, ids):
        key, sha = event
        if key in self.seen:
            if self.checkpoints[key] != sha:
                raise ValueError("duplicate live event changed qualification")
            return
        record = self._record(key, sha)
        ledger.write(json.dumps(record, sort_keys=True) + "\n")
        if record["status"] == "reference_verified":
            ids.write(record["task_id"] + "\n")
            if key not in self.exported:
                self.pending.append((key, sha, record["row"]))
        self.checkpoints[key] = sha
        self.seen.add(key)
        self.counts[record["status"]] += 1

    def _flush(self, ledger, ids):
        for stream in (ledger, ids):
            stream.flush()
            os.fsync(stream.fileno())
        self.last_flush = time.monotonic()

    def _part(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        if not self.pending:
            return
        if self._source_stat() != self.source_stat:
            raise ValueError("source dataset changed during live export")
        pending = sorted(self.pending, key=lambda p: p[2])
        rows = [p[2] for p in pending]
        groups = collections.defaultdict(set)
        for row in rows:
            group = bisect.bisect_right(self.offsets, row) - 1
            groups[group].add(row)
        number = len(self.parts)
        path = self.parts_dir / f"part-{number:06d}.parquet"
        temporary = self.parts_dir / f".part-{number:06d}.tmp"
        written = 0
        try:
            with pq.ParquetWriter(temporary, self.parquet.schema_arrow) as writer:
                for group, selected in sorted(groups.items()):
                    offset = self.offsets[group]
                    for batch in self.parquet.iter_batches(batch_size=64, row_groups=[group]):
                        keep = [i for i in range(batch.num_rows) if offset + i in selected]
                        offset += batch.num_rows
                        if keep:
                            table = pa.Table.from_batches([batch]).take(pa.array(keep, type=pa.int64()))
                            writer.write_table(table)
                            written += len(keep)
            for key, sha, _ in pending:
                self._record(key, sha)
            if written != len(rows) or self._source_stat() != self.source_stat:
                raise ValueError("source rows changed during live export")
            exported = pq.ParquetFile(temporary)
            if exported.metadata.num_rows != len(rows) or not exported.schema_arrow.equals(
                self.parquet.schema_arrow, check_metadata=True
            ):
                raise ValueError("live export changed schema or row count")
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())
            os.link(temporary, path)
            receipt = {
                "version": "swe-rebench-live-part-v1",
                "signature": self.signature,
                "source_sha256": self.source_sha,
                "complete": False,
                "provisional": True,
                "repeat_required": True,
                "path": str(path.resolve()),
                "rows": len(rows),
                "retained_row_indices": rows,
                "retained_rows_unchanged": True,
                "output_sha256": audit.file_sha256(path),
                "checkpoints": [{"id": key, "sha256": sha} for key, sha, _ in pending],
            }
            audit.audit_write_json(path.with_suffix(path.suffix + ".audit.json"), receipt)
            self.parts.append(self._part_description(path, receipt))
            self.exported.update(key for key, _, _ in pending)
            self.pending.clear()
            self.last_part = time.monotonic()
        finally:
            temporary.unlink(missing_ok=True)

    def _run(self):
        ledger_tmp = self.output / ".live_results.jsonl.tmp"
        ids_tmp = self.output / ".live_reference_verified_task_ids.txt.tmp"
        try:
            self._summary(rebuilding=True)
            with ledger_tmp.open("w") as ledger, ids_tmp.open("w") as ids:
                # Rebuild from the hash-checked authority; torn lines, missing
                # writes and duplicate live events never require another audit.
                for event in self.recovered:
                    self._append(event, ledger, ids)
                    if len(self.pending) >= self.part_rows:
                        self._part()
                self._flush(ledger, ids)
                os.replace(ledger_tmp, self.output / "live_results.jsonl")
                os.replace(ids_tmp, self.output / "live_reference_verified_task_ids.txt")
                self._summary()
                while True:
                    try:
                        event = self.events.get(timeout=min(self.flush_seconds, self.part_seconds))
                    except queue.Empty:
                        event = ()
                    if event is None:
                        self._flush(ledger, ids)
                        self._part()
                        self._summary()
                        return
                    if event:
                        self._append(event, ledger, ids)
                    now = time.monotonic()
                    if (
                        len(self.pending) >= self.part_rows
                        or self.pending
                        and now - self.last_part >= self.part_seconds
                    ):
                        self._flush(ledger, ids)
                        self._part()
                        self._summary()
                    if now - self.last_flush >= self.flush_seconds:
                        self._flush(ledger, ids)
                        self._summary()
        except BaseException as exc:
            self.error = exc
            self.stop.set()
            try:
                self._summary(error=f"{type(exc).__name__}: {exc}"[:1500])
            except Exception:
                pass
        finally:
            ledger_tmp.unlink(missing_ok=True)
            ids_tmp.unlink(missing_ok=True)

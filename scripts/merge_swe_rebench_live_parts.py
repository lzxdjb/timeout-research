"""Merge a frozen list of published, repeat-qualified ReBench parquet parts."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import swe_tool_online_benchmark as audit
from scripts.swe_reference_reliability import validate_repeat_qualification


def merge_snapshot(audit_dir: Path, output: Path) -> dict:
    import pyarrow.parquet as pq

    audit_dir = audit_dir.resolve()
    output = output.resolve()
    receipt_output = output.with_suffix(output.suffix + ".audit.json")
    if output.exists() or receipt_output.exists():
        raise ValueError("snapshot output exists; choose a new filename")
    if output.is_relative_to(audit_dir):
        raise ValueError("write snapshots outside the active audit directory")
    # Atomic summary publication freezes the list. Newly published parts belong
    # to the next snapshot; no locks or mutations of the running filter needed.
    summary = json.loads((audit_dir / "live_summary.json").read_text())
    if (summary.get("version") != "swe-rebench-live-results-v1"
            or summary.get("provisional") is not True
            or summary.get("rebuilding") or summary.get("writer_error")):
        raise ValueError("live writer is rebuilding or unhealthy; retry after it recovers")
    parts = summary.get("parts", [])
    if not parts:
        raise ValueError("no published parquet parts yet; wait for the first part")
    checked, task_ids, rows, keys = [], set(), set(), set()
    schema = None
    for description in parts:
        path = (audit_dir / "live_parquet_parts" / Path(description["path"]).name).resolve()
        if path.parent != audit_dir / "live_parquet_parts":
            raise ValueError("part path outside live parts directory")
        receipt_path = path.with_suffix(path.suffix + ".audit.json")
        receipt = json.loads(receipt_path.read_text())
        if (receipt.get("version") != "swe-rebench-live-part-v1"
                or receipt.get("signature") != summary.get("signature")
                or receipt.get("source_sha256") != summary.get("source_sha256")
                or receipt.get("provisional") is not True
                or receipt.get("repeat_required") is not True
                or receipt.get("retained_rows_unchanged") is not True
                or receipt.get("output_sha256") != description.get("output_sha256")
                or audit.file_sha256(path) != receipt.get("output_sha256")):
            raise ValueError("part content or verification receipt changed: " + str(path))
        proofs = receipt.get("checkpoints", [])
        part_rows = []
        for proof in proofs:
            key = proof["id"]
            if Path(key).name != key or key in {"", ".", ".."} or key in keys:
                raise ValueError("invalid or duplicate task checkpoint")
            directory = audit_dir / "cases" / key
            state = json.loads((directory / "state.json").read_text())
            checkpoint = directory / "qualified_record.json"
            if (state.get("terminal") is not True or state.get("signature") != summary["signature"]
                    or state.get("qualified_sha256") != proof["sha256"]
                    or audit.file_sha256(checkpoint) != proof["sha256"]):
                raise ValueError("part is not bound to an unchanged terminal checkpoint")
            record = json.loads(checkpoint.read_text())
            if (record.get("id") != key or record.get("signature") != summary["signature"]
                    or record.get("dataset_sha256") != summary["source_sha256"]
                    or record.get("status") != "reference_verified"
                    or record.get("complete") is not True or record.get("source_match") is not True):
                raise ValueError("part contains a task without reference verification")
            validate_repeat_qualification(record)
            for index, sha in enumerate(state.get("attempt_sha256", [])):
                if audit.file_sha256(directory / f"attempt_{index}" / "record.json") != sha:
                    raise ValueError("terminal attempt checkpoint changed")
            row, task = record["row"], record["task_id"]
            if row in rows or task in task_ids:
                raise ValueError("duplicate source row or task across parts")
            rows.add(row)
            keys.add(key)
            task_ids.add(task)
            part_rows.append(row)
        parquet = pq.ParquetFile(path)
        if (not proofs or part_rows != sorted(part_rows)
                or part_rows != receipt.get("retained_row_indices")
                or parquet.metadata.num_rows != len(proofs)
                or receipt.get("rows") != len(proofs) or description.get("rows") != len(proofs)):
            raise ValueError("part row count/order does not match checkpoint proofs")
        if schema is None:
            schema = parquet.schema_arrow
        elif not schema.equals(parquet.schema_arrow, check_metadata=True):
            raise ValueError("part schemas differ")
        checked.append((path, receipt))
    if len(rows) != summary.get("parquet_rows"):
        raise ValueError("published summary row count differs from parts")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, prefix=".rebench-snapshot-", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        with pq.ParquetWriter(temporary, schema) as writer:
            for path, receipt in checked:
                for batch in pq.ParquetFile(path).iter_batches(batch_size=128):
                    writer.write_batch(batch)
                if audit.file_sha256(path) != receipt["output_sha256"]:
                    raise ValueError("part changed during snapshot merge")
        merged = pq.ParquetFile(temporary)
        if merged.metadata.num_rows != len(rows) or not schema.equals(merged.schema_arrow, check_metadata=True):
            raise ValueError("merged schema or row count changed")
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        result = {"version": "swe-rebench-live-snapshot-v1", "complete": False, "provisional": True,
            "repeat_required": True, "rows": len(rows), "parts": len(parts), "signature": summary["signature"],
            "source_sha256": summary["source_sha256"], "audit_dir": str(audit_dir),
            "output": str(output), "output_sha256": audit.file_sha256(temporary),
            "snapshot_at_unix": summary["updated_at_unix"], "created_at_unix": time.time(),
            "part_receipts": [{"path": str(p), "output_sha256": r["output_sha256"],
                "receipt_sha256": audit.digest(r)} for p, r in checked],
            "final_filter_checks_complete": False}
        os.link(temporary, output)
        audit.audit_write_json(receipt_output, result)
        return result
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(merge_snapshot(args.audit_dir, args.output), indent=2))


if __name__ == "__main__":
    main()

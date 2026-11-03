"""Export accepted, unchanged audio records to verl Parquet without leaking references."""

import argparse
import importlib.util
import json
import os
import tempfile
from collections import Counter
from functools import lru_cache
from pathlib import Path

from .common import effective_decision, file_sha256, license_allowed, read_jsonl, record_fingerprint, validate_record

RULE_TASKS = {"asr", "closed_qa", "classification", "music_family"}


@lru_cache(maxsize=1)
def _reward_normalizer():
    # Reuse the actual reward policy without importing verl's training stack.
    location = Path(__file__).resolve().parents[2] / "verl/utils/reward_score/audio.py"
    spec = importlib.util.spec_from_file_location("_audio_export_reward", location)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.normalize_text


def _has_reference(record):
    reference = record.get("reference")
    references = [reference] if isinstance(reference, str) else reference
    aliases = record.get("reference_aliases", [])
    language = record.get("language")
    if (
        not isinstance(references, list)
        or not references
        or not all(isinstance(x, str) for x in references)
        or not isinstance(aliases, list)
        or not all(isinstance(x, str) for x in aliases)
        or not isinstance(language, str)
        or not language.strip()
    ):
        return False
    normalize = _reward_normalizer()
    return all(normalize(value, language) for value in references + aliases)


def _rl_eligible(record):
    if record["task"] not in RULE_TASKS or not _has_reference(record):
        return False
    if record["task"] == "classification" and record.get("metadata", {}).get("label_set_complete") is not True:
        return False
    reviewed = record.get("qc", {}).get("qwen", {})
    return record.get("reference_verified") is True or (
        reviewed.get("decision") == "accept" and reviewed.get("reference_supported") is True
    )


def _rows(records, *, mode="opd", split=None, allow_noncommercial=False, counters=None, require_qwen=False):
    if mode not in {"opd", "rl"}:
        raise ValueError("mode must be opd or rl")
    counts = counters if counters is not None else Counter()
    hashes = {}
    for record in records:
        counts["input"] += 1
        if effective_decision(record) != "accept":
            counts["not_accepted"] += 1
            continue
        if split is not None and record.get("split") != split:
            counts["other_split"] += 1
            continue
        if not all(
            license_allowed(record.get(key), allow_noncommercial) for key in ("audio_license", "annotation_license")
        ):
            counts["license_excluded"] += 1
            continue
        problems = validate_record(record)
        if problems:
            raise ValueError(f"Invalid accepted record {record.get('sample_id')}: {problems}")
        offline = record["qc"]["offline"]
        if offline.get("input_fingerprint") != record_fingerprint(record):
            raise ValueError(f"Manifest changed since offline checks: {record['sample_id']}; rerun filtering")
        reviewed = record.get("qc", {}).get("qwen")
        if reviewed is not None and reviewed.get("input_fingerprint") != record_fingerprint(record):
            raise ValueError(f"Manifest changed since Qwen review: {record['sample_id']}; rerun review")
        if require_qwen or record.get("annotation_type") == "synthetic":
            if not reviewed or reviewed.get("decision") != "accept":
                counts["qwen_review_required"] += 1
                continue
        path = Path(record["audio_path"]).resolve()
        try:
            stat = path.stat()
            key = (str(path), stat.st_mtime_ns, stat.st_size)
            if key not in hashes:
                if len(hashes) >= 4096:
                    hashes.clear()
                hashes[key] = file_sha256(path)
        except OSError as exc:
            raise ValueError(f"Audio integrity check failed: {record['sample_id']}") from exc
        if not record.get("audio_sha256") or hashes[key] != record["audio_sha256"]:
            raise ValueError(f"Audio hash changed since filtering: {record['sample_id']}")
        eligible = _rl_eligible(record)
        if mode == "rl" and not eligible:
            counts["rl_ineligible"] += 1
            continue
        instruction = record["question"].strip()
        if record.get("options"):
            instruction += "\nOptions:\n" + json.dumps(record["options"], ensure_ascii=False)
        if mode == "rl" and record["task"] == "classification":
            instruction += "\nReturn only a JSON array of class labels."
        specification = {
            "task": record["task"],
            "language": record["language"],
            "reference": record.get("reference"),
            "reference_aliases": record.get("reference_aliases", []),
            "label_set_complete": record.get("metadata", {}).get("label_set_complete", False),
        }
        # OPD rows retain references for later auditing, but the rollout sees only the question/audio.
        row = {
            "data_source": f"audio_{record['dataset']}",
            "agent_name": "single_turn_agent",
            "ability": "audio",
            "prompt": [{"role": "user", "content": instruction + "\n<audio>"}],
            "audios": [{"audio": str(path)}],
            "reward_model": {"style": "rule", "ground_truth": json.dumps(specification, ensure_ascii=False)},
            "extra_info": {
                "index": counts["exported"],
                "sample_id": record["sample_id"],
                "dataset": record["dataset"],
                "revision": record["revision"],
                "split": record["split"],
                "source_split": record["source_split"],
                "source_recording_id": record["source_recording_id"],
                "group_id": record["group_id"],
                "task": record["task"],
                "language": record["language"],
                "audio_sha256": record["audio_sha256"],
                "pcm_sha256": record.get("pcm_sha256", ""),
                "audio_license": record["audio_license"],
                "annotation_license": record["annotation_license"],
                "rl_eligible": eligible,
                "export_mode": mode,
                "qc_json": json.dumps(record["qc"], sort_keys=True, ensure_ascii=False),
            },
        }
        counts["exported"] += 1
        yield row


def convert_records(records, *, mode="opd", split="train", allow_noncommercial=False, require_qwen=False):
    return list(
        _rows(records, mode=mode, split=split, allow_noncommercial=allow_noncommercial, require_qwen=require_qwen)
    )


def _schema():
    import pyarrow as pa

    return pa.schema(
        [
            ("data_source", pa.string()),
            ("agent_name", pa.string()),
            ("ability", pa.string()),
            ("prompt", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))),
            ("audios", pa.list_(pa.struct([("audio", pa.string())]))),
            ("reward_model", pa.struct([("style", pa.string()), ("ground_truth", pa.string())])),
            (
                "extra_info",
                pa.struct(
                    [
                        ("index", pa.int64()),
                        ("sample_id", pa.string()),
                        ("dataset", pa.string()),
                        ("revision", pa.string()),
                        ("split", pa.string()),
                        ("source_split", pa.string()),
                        ("source_recording_id", pa.string()),
                        ("group_id", pa.string()),
                        ("task", pa.string()),
                        ("language", pa.string()),
                        ("audio_sha256", pa.string()),
                        ("pcm_sha256", pa.string()),
                        ("audio_license", pa.string()),
                        ("annotation_license", pa.string()),
                        ("rl_eligible", pa.bool_()),
                        ("export_mode", pa.string()),
                        ("qc_json", pa.string()),
                    ]
                ),
            ),
        ]
    )


def convert_manifest(
    input_path, output_path, *, mode="opd", split="train", allow_noncommercial=False, batch_size=512, require_qwen=False
):
    import pyarrow as pa
    import pyarrow.parquet as pq

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".parquet", dir=output.parent)
    os.close(fd)
    counts = Counter()
    try:
        schema = _schema()
        with pq.ParquetWriter(temporary, schema, compression="zstd") as writer:
            batch = []
            for row in _rows(
                read_jsonl(input_path),
                mode=mode,
                split=split,
                allow_noncommercial=allow_noncommercial,
                counters=counts,
                require_qwen=require_qwen,
            ):
                batch.append(row)
                if len(batch) >= batch_size:
                    writer.write_table(pa.Table.from_pylist(batch, schema=schema))
                    batch.clear()
            if batch:
                writer.write_table(pa.Table.from_pylist(batch, schema=schema))
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    report = {"mode": mode, "split": split, "counts": dict(counts), "output": str(output.resolve())}
    Path(str(output) + ".report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=("opd", "rl"), required=True)
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        required=True,
        help="Keep official partitions separate; no implicit train/test reshuffling",
    )
    parser.add_argument("--allow-noncommercial", action="store_true")
    parser.add_argument("--require-qwen", action="store_true", help="Require Qwen accept for every exported record")
    parser.add_argument("--batch-size", type=int, default=512)
    args = vars(parser.parse_args())
    input_path, output_path = args.pop("input"), args.pop("output")
    print(json.dumps(convert_manifest(input_path, output_path, **args), indent=2))


if __name__ == "__main__":
    main()

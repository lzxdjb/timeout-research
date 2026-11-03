"""Offline integrity, provenance, reference signals, and cross-split contamination checks."""

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import sqlite3
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf

from .common import (
    file_sha256,
    license_allowed,
    read_jsonl,
    record_fingerprint,
    stable_hash,
    validate_record,
)


@lru_cache(maxsize=1)
def _audio_reward_module():
    # Load the stdlib-only reward module without importing verl's GPU/Ray package.
    location = Path(__file__).resolve().parents[2] / "verl/utils/reward_score/audio.py"
    spec = importlib.util.spec_from_file_location("_audio_text_metrics", location)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _error_rate(prediction, reference, language):
    return _audio_reward_module().error_rate(prediction, reference, language)


def _asr_language(value):
    if not isinstance(value, str):
        return None
    language = value.strip().casefold().replace("_", "-")
    if language in {"zh", "cmn", "mandarin", "chinese"} or language.startswith(("zh-", "cmn-")):
        return "zh"
    if language in {"en", "english"} or language.startswith("en-"):
        return "en"
    return language or None


def inspect_audio(path, min_duration=0.1, max_duration=120.0, max_file_bytes=256 * 1024 * 1024):
    path = Path(path)
    if not path.is_file():
        return {"decision": "quarantine", "reasons": ["missing_audio"], "metrics": {}}
    before = path.stat()
    if before.st_size > max_file_bytes:
        return {"decision": "quarantine", "reasons": ["audio_file_too_large"], "metrics": {"bytes": before.st_size}}
    try:
        with sf.SoundFile(path) as stream:
            rate, channels, frames = stream.samplerate, stream.channels, len(stream)
            duration = frames / rate if rate else 0.0
            metrics = {
                "duration_seconds": duration,
                "sample_rate": rate,
                "channels": channels,
                "frames": frames,
                "bytes": before.st_size,
            }
            if not frames or not channels or not rate:
                return {"decision": "reject", "reasons": ["empty_audio"], "metrics": metrics}
            if duration < min_duration or (max_duration is not None and duration > max_duration):
                return {"decision": "quarantine", "reasons": ["duration_out_of_range"], "metrics": metrics}
            digest = hashlib.sha256(f"float32le:{rate}:{channels}:".encode())
            total, square_sum, peak, clipped, quiet, decoded_frames = 0, 0.0, 0.0, 0, 0, 0
            for block in stream.blocks(blocksize=65536, dtype="float32", always_2d=True):
                if not np.isfinite(block).all():
                    return {"decision": "reject", "reasons": ["nonfinite_audio"], "metrics": metrics}
                block[block == 0] = 0  # canonicalize negative zero
                digest.update(block.astype("<f4", copy=False).tobytes())
                absolute = np.abs(block)
                total += block.size
                decoded_frames += len(block)
                square_sum += float(np.sum(np.square(block, dtype=np.float64)))
                peak = max(peak, float(absolute.max(initial=0.0)))
                clipped += int(np.count_nonzero(absolute >= 0.999))
                quiet += int(np.count_nonzero(absolute < 1e-5))
            if decoded_frames != frames:
                return {"decision": "reject", "reasons": ["truncated_audio"], "metrics": metrics}
        raw_hash = file_sha256(path)
        after = path.stat()
        if (before.st_mtime_ns, before.st_size) != (after.st_mtime_ns, after.st_size):
            return {"decision": "quarantine", "reasons": ["audio_changed_during_check"], "metrics": metrics}
        metrics.update(
            rms=math.sqrt(square_sum / total),
            peak=peak,
            clipping_fraction=clipped / total,
            near_silence_fraction=quiet / total,
        )
        return {
            "decision": "quarantine" if peak == 0 else "accept",
            "reasons": ["silent_audio"] if peak == 0 else [],
            "metrics": metrics,
            "audio_sha256": raw_hash,
            "pcm_sha256": digest.hexdigest(),
        }
    except (OSError, RuntimeError, ValueError) as exc:
        return {"decision": "reject", "reasons": ["decode_error"], "metrics": {}, "error": str(exc)[:300]}


def _check_record(
    record,
    *,
    allow_noncommercial=False,
    min_duration=0.1,
    max_duration=120.0,
    asr_hypotheses=None,
    max_wer=None,
    max_cer=None,
    require_reference_check=False,
    reference_check_splits=None,
    max_file_bytes=256 * 1024 * 1024,
    audio_cache=None,
):
    record = copy.deepcopy(record)
    reasons = validate_record(record)
    if reasons:
        record["qc"] = {"offline": {"decision": "quarantine", "reasons": reasons, "metrics": {}}}
        return record
    for key in ("audio_license", "annotation_license"):
        if not license_allowed(record[key], allow_noncommercial):
            reasons.append(f"unapproved_{key}")
    source = record["source_split"].casefold().rsplit(":", 1)[-1].removeprefix("nsynth-")
    if record["split"] == "train" and source.startswith(("test", "eval", "val", "dev-", "dev_", "dev.", "valid")):
        reasons.append("heldout_source_labeled_train")
    if record["split"] == "train" and source == "dev":
        reasons.append("heldout_source_labeled_train")
    metadata = record.get("metadata", {})
    for reason in metadata.get("quarantine_reasons", []):
        reasons.append(f"source:{reason}")
    if metadata.get("annotation_consensus") is False:
        reasons.append("annotation_disagreement")
    audio_path = Path(record["audio_path"]).expanduser().resolve()
    record["audio_path"] = str(audio_path)
    stat = audio_path.stat() if audio_path.is_file() else None
    key = (str(audio_path), stat.st_mtime_ns, stat.st_size) if stat else (str(audio_path),)
    cache = audio_cache if audio_cache is not None else {}
    if key not in cache:
        if len(cache) >= 4096:
            cache.clear()
        cache[key] = inspect_audio(audio_path, min_duration, max_duration, max_file_bytes)
    audio = copy.deepcopy(cache[key])
    for hash_name in ("audio_sha256", "pcm_sha256"):
        if hash_name in audio:
            record[hash_name] = audio.pop(hash_name)
    reasons.extend(audio["reasons"])
    metrics = audio["metrics"]
    if record["task"] == "asr" and (reference_check_splits is None or record["split"] in reference_check_splits):
        hypothesis = (asr_hypotheses or {}).get(record["sample_id"])
        if hypothesis is not None:
            language = _asr_language(record["language"])
            normalize = _audio_reward_module().normalize_text
            if not isinstance(hypothesis, dict) or hypothesis.get("audio_sha256") != record.get("audio_sha256"):
                reasons.append("asr_hypothesis_audio_mismatch")
            elif "language" in hypothesis and _asr_language(hypothesis["language"]) != language:
                reasons.append("asr_hypothesis_language_mismatch")
            elif not isinstance(hypothesis.get("text"), str) or not normalize(hypothesis["text"], language):
                reasons.append("invalid_asr_hypothesis")
            else:
                refs = record.get("reference")
                refs = [refs] if isinstance(refs, str) else (refs or [])
                refs = [ref for ref in refs if normalize(ref, language)]
                if refs:
                    aliases = [alias for alias in record.get("reference_aliases", []) if normalize(alias, language)]
                    chinese = language == "zh"
                    name, threshold = ("cer", max_cer) if chinese else ("wer", max_wer)
                    metrics[name] = min(_error_rate(hypothesis["text"], ref, language) for ref in refs + aliases)
                    metrics["asr_model"] = hypothesis.get("model", "unspecified")
                    if threshold is None and require_reference_check:
                        reasons.append("asr_threshold_not_configured")
                    elif threshold is not None and metrics[name] > threshold:
                        reasons.append("asr_reference_disagreement")
                elif require_reference_check:
                    reasons.append("missing_asr_reference")
        elif require_reference_check:
            reasons.append("missing_asr_hypothesis")
    # Re-running offline invalidates old judge evidence and any old derived eligibility.
    record["qc"] = {
        "offline": {
            "decision": "reject" if audio["decision"] == "reject" else ("quarantine" if reasons else "accept"),
            "reasons": sorted(set(reasons)),
            "metrics": metrics,
        }
    }
    record["qc"]["offline"]["input_fingerprint"] = record_fingerprint(record)
    return record


def _register(connection, record):
    split = record.get("split")
    valid_split = isinstance(split, str) and split in {"train", "validation", "test"}
    for kind, value in (("group", record.get("group_id")), ("pcm", record.get("pcm_sha256"))):
        if isinstance(value, str) and value.strip() and valid_split:
            connection.execute("INSERT INTO media VALUES (?, ?, ?)", (kind, value, split))
    identity = record.get("sample_id")
    valid_identity = isinstance(identity, str) and bool(identity.strip())
    if valid_identity:
        connection.execute("INSERT INTO ids VALUES (?)", (identity,))
    pcm_hash = record.get("pcm_sha256")
    if (
        isinstance(pcm_hash, str)
        and pcm_hash.strip()
        and valid_identity
        and record["qc"]["offline"]["decision"] == "accept"
    ):
        duplicate = stable_hash(
            [
                record["pcm_sha256"],
                record.get("question"),
                record.get("options"),
                record.get("task"),
                record.get("language"),
                record.get("reference"),
            ]
        )
        record["qc"]["offline"]["example_key"] = duplicate
        connection.execute("INSERT INTO examples VALUES (?,?)", (duplicate, identity))


def _finish(connection, record):
    reasons = record["qc"]["offline"]["reasons"]
    for kind, value in (("group", record.get("group_id")), ("pcm", record.get("pcm_sha256"))):
        if (
            isinstance(value, str)
            and value.strip()
            and connection.execute(
                "SELECT COUNT(DISTINCT split) FROM media WHERE kind=? AND value=?", (kind, value)
            ).fetchone()[0]
            > 1
        ):
            reasons.append("cross_split_recording" if kind == "group" else "cross_split_pcm")
    identity = record.get("sample_id")
    if (
        isinstance(identity, str)
        and identity.strip()
        and connection.execute("SELECT COUNT(*) FROM ids WHERE value=?", (identity,)).fetchone()[0] > 1
    ):
        reasons.append("duplicate_sample_id")
    example = record["qc"]["offline"].get("example_key")
    if isinstance(example, str) and example:
        owner = connection.execute("SELECT MIN(sample_id) FROM examples WHERE value=?", (example,)).fetchone()[0]
        if owner != record.get("sample_id"):
            reasons.append("duplicate_example")
    record["qc"]["offline"]["reasons"] = sorted(set(reasons))
    if reasons and record["qc"]["offline"]["decision"] == "accept":
        record["qc"]["offline"]["decision"] = "quarantine"
    return record


def _database(path):
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE IF NOT EXISTS media (kind TEXT, value TEXT, split TEXT)")
    connection.execute("CREATE TABLE IF NOT EXISTS ids (value TEXT)")
    connection.execute("CREATE TABLE IF NOT EXISTS examples (value TEXT, sample_id TEXT)")
    return connection


def _index(connection):
    connection.execute("CREATE INDEX IF NOT EXISTS media_values ON media (kind,value,split)")
    connection.execute("CREATE INDEX IF NOT EXISTS sample_ids ON ids (value)")
    connection.execute("CREATE INDEX IF NOT EXISTS example_values ON examples (value,sample_id)")
    connection.commit()


def _register_holdouts(connection, records, policy):
    for number, original in enumerate(records, 1):
        record = copy.deepcopy(original)
        record["split"] = "test"
        # Holdout manifests may contain just group_id and an audio path.
        group = record.get("group_id")
        if group is not None and (not isinstance(group, str) or not group.strip()):
            raise ValueError(f"Holdout row {number}: group_id must be a nonempty string")
        pcm = record.get("pcm_sha256")
        if pcm is not None and (
            not isinstance(pcm, str) or len(pcm) != 64 or any(char not in "0123456789abcdef" for char in pcm)
        ):
            raise ValueError(f"Holdout row {number}: pcm_sha256 must be a lowercase SHA256 digest")
        if "audio_path" in record:
            if not isinstance(record["audio_path"], str) or not record["audio_path"].strip():
                raise ValueError(f"Holdout row {number}: audio_path must be a nonempty string")
            # Benchmark media need not satisfy the candidate context window.
            # A failed scan must never silently claim that the holdout was covered.
            audio = inspect_audio(
                record["audio_path"],
                min_duration=0,
                max_duration=None,
                max_file_bytes=policy.get("max_file_bytes", 256 * 1024 * 1024),
            )
            if not audio.get("pcm_sha256"):
                raise ValueError(f"Holdout row {number}: audio could not be hashed ({','.join(audio['reasons'])})")
            if pcm and pcm != audio["pcm_sha256"]:
                raise ValueError(f"Holdout row {number}: pcm_sha256 does not match supplied audio")
            record["pcm_sha256"] = audio["pcm_sha256"]
        if not group and not record.get("pcm_sha256"):
            raise ValueError(f"Holdout row {number}: provide group_id or readable audio_path")
        # Holdouts need no sample/example identity and cannot suppress unrelated
        # training examples just because an external list reuses an identifier.
        record.pop("sample_id", None)
        _register(connection, record)


def filter_records(records, *, holdout_records=None, **policy):
    connection = _database(":memory:")
    try:
        cache = {}
        checked = [_check_record(record, audio_cache=cache, **policy) for record in records]
        for record in checked:
            _register(connection, record)
        _register_holdouts(connection, holdout_records or [], policy)
        _index(connection)
        return [_finish(connection, record) for record in checked]
    finally:
        connection.close()


def filter_manifest(
    input_path,
    output_path,
    *,
    holdout_manifest=None,
    checkpoint_dir=None,
    checkpoint_interval=1000,
    progress="auto",
    hypotheses_path=None,
    reset_checkpoint=False,
    **policy,
):
    """Bounded-memory global filtering with optional transactional resume."""
    from .offline_checkpoint import run_filter

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    options = {
        "checkpoint_interval": checkpoint_interval,
        "progress": progress,
        "holdout_manifest": holdout_manifest,
        "hypotheses_path": hypotheses_path,
        "reset": reset_checkpoint,
        "policy": policy,
    }
    api = sys.modules[__name__]
    if checkpoint_dir is not None:
        if Path(input_path).resolve() == output.resolve():
            raise ValueError("Checkpointed filtering requires distinct input and output paths")
        return run_filter(api, input_path, output, directory=checkpoint_dir, **options)
    if reset_checkpoint:
        raise ValueError("reset_checkpoint requires checkpoint_dir")
    with tempfile.TemporaryDirectory(prefix=".audio-offline-", dir=output.parent) as temporary:
        return run_filter(api, input_path, output, directory=temporary, **options)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-noncommercial", action="store_true")
    parser.add_argument("--min-duration", type=float, default=0.1)
    parser.add_argument("--max-duration", type=float, default=120.0)
    parser.add_argument("--max-file-bytes", type=int, default=256 * 1024 * 1024)
    parser.add_argument("--asr-hypotheses", help="JSONL: sample_id, text, audio_sha256, model; local inference outputs")
    parser.add_argument("--max-wer", type=float, help="Calibrate on an English audit set before using as a gate")
    parser.add_argument("--max-cer", type=float, help="Calibrate on a Mandarin audit set before using as a gate")
    parser.add_argument("--require-reference-check", action="store_true")
    parser.add_argument(
        "--reference-check-split",
        dest="reference_check_splits",
        action="append",
        choices=("train", "validation", "test"),
        help="Restrict ASR reference checks to selected splits; media/leakage checks still cover all rows",
    )
    parser.add_argument("--holdout-manifest", help="Existing benchmark recording groups/media excluded from training")
    parser.add_argument("--progress", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--checkpoint-dir", help="Persistent state directory; rerun the same command to resume")
    parser.add_argument("--checkpoint-interval", type=int, default=1000, help="Commit every N checked/finalized rows")
    parser.add_argument(
        "--reset-checkpoint", action="store_true", help="Discard this filter's saved checkpoint before starting"
    )
    args = parser.parse_args()
    for value in (args.min_duration, args.max_duration, args.max_wer, args.max_cer):
        if value is not None and (not math.isfinite(value) or value < 0):
            parser.error("durations and error thresholds must be finite and nonnegative")
    if args.max_duration < args.min_duration or args.max_file_bytes <= 0:
        parser.error("invalid duration or file-size limits")
    if args.checkpoint_interval <= 0 or (args.reset_checkpoint and not args.checkpoint_dir):
        parser.error("checkpoint interval must be positive; resetting requires --checkpoint-dir")
    policy = vars(args).copy()
    input_path, output_path = policy.pop("input"), policy.pop("output")
    path = policy.pop("asr_hypotheses")
    hypotheses = {}
    if path:
        for row in read_jsonl(path):
            identity = row.get("sample_id")
            if not identity or identity in hypotheses:
                raise ValueError("ASR hypotheses require unique nonempty sample_id")
            hypotheses[identity] = row
    policy["asr_hypotheses"] = hypotheses
    policy["hypotheses_path"] = path
    print(json.dumps(filter_manifest(input_path, output_path, **policy), indent=2))


if __name__ == "__main__":
    main()

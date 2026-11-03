"""Manifest IO and policy shared by the standalone audio preparation commands."""

import hashlib
import json
import os
import tempfile
from pathlib import Path
from urllib.parse import urlparse

SCHEMA_VERSION = 1
DECISIONS = {"accept", "reject", "quarantine"}
TASKS = {"asr", "closed_qa", "classification", "music_family", "caption", "open_qa"}
PERMISSIVE_LICENSES = {
    "CC0-1.0",
    "CC-BY-2.0",
    "CC-BY-2.5",
    "CC-BY-3.0",
    "CC-BY-4.0",
    "Apache-2.0",
    "MIT",
}
NONCOMMERCIAL_LICENSES = {
    "CC-BY-NC-3.0",
    "CC-BY-NC-4.0",
    "CC-BY-NC-SA-3.0",
    "CC-BY-NC-SA-4.0",
    "NVIDIA-OneWay-Noncommercial",
    "Tampere-Noncommercial",
}


def canonical_license(value):
    """Normalize known identifiers/CC URLs without treating unknown terms as permission."""
    if not isinstance(value, str):
        return "unknown"
    value = value.strip()
    if not value:
        return "unknown"
    try:
        parsed = urlparse(value)
        hostname = parsed.hostname
    except ValueError:
        # Unknown/malformed license text is not permission, including invalid URLs.
        return value
    if hostname in {"creativecommons.org", "www.creativecommons.org"}:
        parts = parsed.path.strip("/").split("/")
        if len(parts) >= 3 and parts[0] == "licenses":
            value = f"CC-{parts[1].upper()}-{parts[2]}"
        elif len(parts) >= 3 and parts[:2] == ["publicdomain", "zero"]:
            value = "CC0-1.0"
    aliases = {"cc0": "CC0-1.0", "apache license v.2.0": "Apache-2.0", "apache 2.0": "Apache-2.0"}
    known = PERMISSIVE_LICENSES | NONCOMMERCIAL_LICENSES
    return aliases.get(value.casefold(), next((x for x in known if x.casefold() == value.casefold()), value))


def license_allowed(value, allow_noncommercial=False):
    permitted = PERMISSIVE_LICENSES | (NONCOMMERCIAL_LICENSES if allow_noncommercial else set())
    return canonical_license(value) in permitted


def stable_hash(value):
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def record_fingerprint(record):
    fields = (
        "schema_version",
        "sample_id",
        "dataset",
        "revision",
        "source_split",
        "split",
        "source_recording_id",
        "group_id",
        "audio_path",
        "task",
        "language",
        "question",
        "options",
        "reference",
        "reference_aliases",
        "audio_license",
        "annotation_license",
        "annotation_type",
        "reference_verified",
        "metadata",
        "audio_sha256",
        "pcm_sha256",
    )
    return stable_hash({key: record.get(key) for key in fields})


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path):
    with open(path, encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path}:{number}: invalid JSON") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{number}: expected a JSON object")
            yield record


def write_jsonl(path, records):
    """Atomically replace a manifest, including when input and output are the same file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def effective_decision(record):
    stages = record.get("qc", {})
    if not isinstance(stages, dict) or not isinstance(stages.get("offline"), dict):
        return "quarantine"
    if any(not isinstance(stage, dict) for stage in stages.values()):
        return "quarantine"
    decisions = [stage.get("decision") for stage in stages.values()]
    if "reject" in decisions:
        return "reject"
    if not decisions or any(x != "accept" for x in decisions):
        return "quarantine"
    return "accept"


def validate_record(record):
    reasons = []
    if type(record.get("schema_version")) is not int or record["schema_version"] != SCHEMA_VERSION:
        reasons.append("unsupported_schema")
    for name in (
        "sample_id",
        "dataset",
        "revision",
        "source_split",
        "source_recording_id",
        "group_id",
        "audio_path",
        "question",
        "language",
        "audio_license",
        "annotation_license",
    ):
        if not isinstance(record.get(name), str) or not record[name].strip():
            reasons.append(f"missing_or_invalid_{name}")
    if not isinstance(record.get("split"), str) or record["split"] not in {"train", "validation", "test"}:
        reasons.append("invalid_split")
    if not isinstance(record.get("task"), str) or record["task"] not in TASKS:
        reasons.append("invalid_task")
    if not isinstance(record.get("annotation_type"), str) or record["annotation_type"] not in {
        "human",
        "synthetic",
        "metadata",
    }:
        reasons.append("invalid_annotation_type")
    if not isinstance(record.get("reference_verified"), bool):
        reasons.append("invalid_reference_verified")
    reference = record.get("reference")
    if (
        reference is not None
        and not isinstance(reference, str)
        and not (isinstance(reference, list) and all(isinstance(x, str) for x in reference))
    ):
        reasons.append("invalid_reference")
    aliases = record.get("reference_aliases", [])
    if not isinstance(aliases, list) or not all(isinstance(x, str) for x in aliases):
        reasons.append("invalid_reference_aliases")
    metadata = record.get("metadata", {})
    if not isinstance(metadata, dict):
        reasons.append("invalid_metadata")
    else:
        source_reasons = metadata.get("quarantine_reasons", [])
        if not isinstance(source_reasons, list) or not all(isinstance(x, str) and x.strip() for x in source_reasons):
            reasons.append("invalid_metadata_quarantine_reasons")
        for name in ("annotation_consensus", "label_set_complete"):
            if name in metadata and not isinstance(metadata[name], bool):
                reasons.append(f"invalid_metadata_{name}")
    if "<audio>" in str(record.get("question", "")):
        reasons.append("question_contains_audio_placeholder")
    return reasons

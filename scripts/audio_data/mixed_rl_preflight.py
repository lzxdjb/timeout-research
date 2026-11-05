"""Validate the audited audio RL export and unchanged SWE files before mixed RL."""

import argparse
import hashlib
import importlib.util
import json
import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

QUOTAS = {"swe": 16, "audio_librispeech": 18, "audio_aishell1": 18, "audio_clotho_aqa": 12}
AUDIO_TASKS = {"audio_librispeech": "asr", "audio_aishell1": "asr", "audio_clotho_aqa": "closed_qa"}


@lru_cache(maxsize=1)
def _reward_module():
    path = Path(__file__).resolve().parents[2] / "verl/utils/reward_score/audio.py"
    spec = importlib.util.spec_from_file_location("_mixed_rl_preflight_reward", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parquet(path):
    import pyarrow.parquet as pq

    path = Path(path).resolve(strict=True)
    parquet = pq.ParquetFile(path)
    if parquet.metadata.num_rows == 0:
        raise ValueError(f"Empty training/validation parquet: {path}")
    return path, parquet


def _audio_row(row, seen):
    source = row.get("data_source")
    if source not in AUDIO_TASKS:
        raise ValueError(f"Unexpected audio training source: {source!r}")
    extra = row.get("extra_info") or {}
    sample_id = extra.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id or sample_id in seen:
        raise ValueError(f"Missing or duplicate audio sample_id: {sample_id!r}")
    seen.add(sample_id)
    if extra.get("export_mode") != "rl" or extra.get("rl_eligible") is not True:
        raise ValueError(f"{sample_id}: requires an eligible RL export, not an OPD export")
    if extra.get("split") != "train":
        raise ValueError(f"{sample_id}: audio training split must be train")
    if source != "audio_" + str(extra.get("dataset")):
        raise ValueError(f"{sample_id}: dataset and source disagree")
    if row.get("agent_name") != "single_turn_agent" or row.get("ability") != "audio":
        raise ValueError(f"{sample_id}: audio rows require single_turn_agent and audio ability")
    qc = json.loads(extra.get("qc_json", "null"))
    if not isinstance(qc, dict):
        raise ValueError(f"{sample_id}: missing audit evidence")
    offline, audit = qc.get("offline") or {}, qc.get("qwen") or {}
    if offline.get("decision") != "accept" or audit.get("decision") != "accept":
        raise ValueError(f"{sample_id}: offline and Qwen decisions must both accept")
    if audit.get("reference_supported") is not True:
        raise ValueError(f"{sample_id}: Qwen must support the hidden reference")
    fingerprint = offline.get("input_fingerprint")
    if not isinstance(fingerprint, str) or not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
        raise ValueError(f"{sample_id}: invalid offline fingerprint")
    if audit.get("input_fingerprint") != fingerprint:
        raise ValueError(f"{sample_id}: offline/Qwen fingerprint mismatch")
    audio_hash = extra.get("audio_sha256")
    if not isinstance(audio_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", audio_hash):
        raise ValueError(f"{sample_id}: invalid audio hash")
    if audit.get("audio_sha256") != audio_hash:
        raise ValueError(f"{sample_id}: exported audio differs from audited audio")
    audios = row.get("audios")
    if not isinstance(audios, list) or not audios:
        raise ValueError(f"{sample_id}: missing audio")
    for audio in audios:
        media = audio.get("audio") if isinstance(audio, dict) else None
        if not isinstance(media, str) or not Path(media).is_file() or Path(media).stat().st_size <= 0:
            raise ValueError(f"{sample_id}: missing or empty local audio: {media!r}")
    prompt = row.get("prompt")
    if not isinstance(prompt, list) or not any(
        isinstance(message, dict)
        and message.get("role") == "user"
        and isinstance(message.get("content"), str)
        and "<audio>" in message["content"]
        for message in prompt
    ):
        raise ValueError(f"{sample_id}: missing audio user prompt")
    reward = row.get("reward_model") or {}
    if reward.get("style") != "rule" or not isinstance(reward.get("ground_truth"), str):
        raise ValueError(f"{sample_id}: missing hidden rule reward specification")
    specification = json.loads(reward["ground_truth"])
    if not isinstance(specification, dict) or specification.get("task") != AUDIO_TASKS[source]:
        raise ValueError(f"{sample_id}: wrong reward task for {source}")
    if specification.get("task") != extra.get("task") or specification.get("language") != extra.get("language"):
        raise ValueError(f"{sample_id}: reward metadata disagrees with exported metadata")
    language = specification.get("language", "")
    if source == "audio_aishell1" and not language.startswith(("zh", "cmn")):
        raise ValueError(f"{sample_id}: AISHELL requires a Mandarin CER reward")
    if source == "audio_librispeech" and not language.startswith("en"):
        raise ValueError(f"{sample_id}: LibriSpeech requires an English WER reward")
    # Validate against the actual training reward implementation, without loading verl.
    _reward_module().compute_score(source, "", specification)
    return source


def _swe_file(path, *, expected_rows=None):
    path, parquet = _parquet(path)
    if expected_rows is not None and parquet.metadata.num_rows != expected_rows:
        raise ValueError(f"SWE validation must retain exactly {expected_rows} tasks: {path}")
    seen = set()
    for batch in parquet.iter_batches(batch_size=256):
        for row in batch.to_pylist():
            source = row.get("data_source")
            if not isinstance(source, str) or not source or source.startswith("audio_"):
                raise ValueError(f"Invalid SWE data source in {path}: {source!r}")
            if row.get("agent_name") not in (None, "swe_tool_agent"):
                raise ValueError(f"Unexpected SWE agent in {path}")
            extra = row.get("extra_info") or {}
            task_id = extra.get("task_id")
            if not isinstance(task_id, str) or not task_id:
                raise ValueError(f"SWE row has no original task_id in {path}")
            if expected_rows is not None and task_id in seen:
                raise ValueError(f"Duplicate SWE validation task_id: {task_id}")
            seen.add(task_id)
            reward = row.get("reward_model") or {}
            if reward.get("ground_truth") in (None, ""):
                raise ValueError(f"SWE row has no hidden ground truth: {task_id}")
    return {"path": str(path), "rows": parquet.metadata.num_rows, "unique_tasks": len(seen), "sha256": _sha256(path)}


def validate_mixed_rl(audio_train, swe_train, swe_validation, *, expected_swe_validation=357, steps=100):
    if expected_swe_validation <= 0 or steps <= 0:
        raise ValueError("Expected validation count and training steps must be positive")
    audio_path, parquet = _parquet(audio_train)
    counts, seen = Counter(), set()
    for batch in parquet.iter_batches(batch_size=256):
        for row in batch.to_pylist():
            counts[_audio_row(row, seen)] += 1
    missing = set(AUDIO_TASKS) - counts.keys()
    if missing:
        raise ValueError(f"Requested audio source pools are empty: {sorted(missing)}")
    swe = _swe_file(swe_train)
    validation = _swe_file(swe_validation, expected_rows=expected_swe_validation)
    pools = {"swe": swe["rows"], **dict(counts)}
    return {
        "audio_train": {"path": str(audio_path), "rows": sum(counts.values()), "sha256": _sha256(audio_path)},
        "swe_train": swe,
        "swe_validation": validation,
        "source_pools": pools,
        "prompt_quotas_per_64": QUOTAS,
        "training_steps": steps,
        "source_sampling": {
            source: {"prompt_draws": quota * steps, "pool_passes": math.ceil(quota * steps / pools[source])}
            for source, quota in QUOTAS.items()
        },
        "media_check": "local files exist and audit/export hashes agree; export already verified media bytes",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-train", required=True, type=Path)
    parser.add_argument("--swe-train", required=True, type=Path)
    parser.add_argument("--swe-validation", required=True, type=Path)
    parser.add_argument("--expected-swe-validation", type=int, default=357)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    print(json.dumps(validate_mixed_rl(**vars(args)), indent=2), flush=True)


if __name__ == "__main__":
    main()

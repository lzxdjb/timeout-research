"""Produce independent ASR hypotheses using an existing local faster-whisper snapshot.

No model weights or tokenizers are downloaded. The resulting JSONL is consumed by
``filter_offline --asr-hypotheses``; inference failures go to a separate error file.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import io
import json
import math
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path

from .common import file_sha256, read_jsonl, record_fingerprint, stable_hash, write_jsonl

MAX_CACHED_PREDICTIONS = 4096


class TranscriptionError(Exception):
    """A record could not produce a valid independent transcript."""


@dataclass(frozen=True)
class LocalASRConfig:
    model: Path
    device: str = "cpu"
    compute_type: str = "int8"
    model_revision: str | None = None
    max_duration: float = 120.0
    max_audio_bytes: int = 64 * 1024 * 1024
    audio_base: Path = Path(".")


def normalize_language(value: str) -> str:
    """Keep Mandarin/English language identity instead of forcing all clips to English."""
    language = value.strip().lower().replace("_", "-")
    if language in {"mandarin", "chinese", "cmn"} or language.startswith(("zh-", "cmn-")):
        return "zh"
    if language == "english" or language.startswith("en-"):
        return "en"
    if language in {"en", "zh"}:
        return language
    raise TranscriptionError(f"unsupported_language:{language}:initial_pipeline_supports_en_and_zh")


def snapshot_identity(model: Path, revision: str | None) -> dict:
    """Hash the actual local snapshot once, or use an explicitly declared immutable revision."""
    if revision is not None:
        if not revision.strip():
            raise ValueError("model-revision must be a nonempty immutable checkpoint identifier")
        return {"kind": "explicit_immutable_revision", "revision": revision}
    files = {}
    for path in sorted(model.rglob("*")):
        relative = path.relative_to(model)
        if any(part.startswith(".") for part in relative.parts) or not path.is_file():
            continue
        files[str(relative)] = file_sha256(path)
    if not files:
        raise ValueError("local model snapshot contains no readable files")
    return {"kind": "local_snapshot_sha256", "content_sha256": stable_hash(files), "files": files}


class LocalASRProducer:
    def __init__(self, config: LocalASRConfig):
        self.config = config
        self.model_path = Path(config.model).expanduser().resolve()
        if not self.model_path.is_dir():
            raise ValueError("--model must be an existing local faster-whisper snapshot directory")
        # faster-whisper may fall back to a remote tokenizer if tokenizer.json is
        # absent, even when its model weights are local. Require it explicitly.
        missing = [
            name for name in ("model.bin", "config.json", "tokenizer.json") if not (self.model_path / name).is_file()
        ]
        if missing:
            raise ValueError("incomplete local snapshot; missing " + ", ".join(missing))
        if config.device not in {"cpu", "cuda"} or config.compute_type not in {"int8", "float16"}:
            raise ValueError("device must be cpu/cuda and compute-type must be int8/float16")
        if config.device == "cpu" and config.compute_type == "float16":
            raise ValueError("CPU inference requires --compute-type int8; float16 is intended for CUDA")
        self._model = None
        self._identity = None
        self._initialization_error = None
        self._predictions = OrderedDict()

    def _get_model(self):
        if self._initialization_error:
            raise TranscriptionError(self._initialization_error)
        if self._model is None:
            try:
                identity = snapshot_identity(self.model_path, self.config.model_revision)
                engine = importlib.import_module("faster_whisper")
                self._model = engine.WhisperModel(
                    str(self.model_path),
                    device=self.config.device,
                    compute_type=self.config.compute_type,
                    local_files_only=True,
                )
                self._identity = {
                    **identity,
                    "engine": "faster-whisper",
                    "engine_version": getattr(engine, "__version__", "unavailable"),
                    "device": self.config.device,
                    "compute_type": self.config.compute_type,
                    "transcribe_config": {
                        "task": "transcribe",
                        "beam_size": 5,
                        "vad_filter": False,
                        "condition_on_previous_text": False,
                    },
                }
            except ImportError as exc:
                self._initialization_error = (
                    "missing_optional_dependency:install_scripts/audio_data/requirements-asr.txt"
                )
                raise TranscriptionError(self._initialization_error) from exc
            except Exception as exc:
                self._initialization_error = f"local_model_initialization_failed:{type(exc).__name__}"
                raise TranscriptionError(self._initialization_error) from exc
        return self._model

    def transcribe(self, record: dict) -> dict | None:
        offline = record.get("qc", {}).get("offline", {})
        if record.get("task") != "asr" or offline.get("decision") != "accept":
            return None
        try:
            if offline.get("input_fingerprint") != record_fingerprint(record):
                raise TranscriptionError("record_changed_since_offline_review:rerun_offline_filter")
            language = normalize_language(record["language"])
            path = Path(record["audio_path"])
            if not path.is_absolute():
                path = self.config.audio_base / path
            if path.stat().st_size > self.config.max_audio_bytes:
                raise TranscriptionError("audio_file_exceeds_byte_limit")
            audio = path.read_bytes()
            digest = hashlib.sha256(audio).hexdigest()
            if not record.get("audio_sha256") or record["audio_sha256"] != digest:
                raise TranscriptionError("audio_changed_since_offline_review")
            import soundfile as sf

            info = sf.info(io.BytesIO(audio))
            if info.duration <= 0 or info.duration > self.config.max_duration:
                raise TranscriptionError("audio_duration_outside_asr_limit")
            cache_key = (digest, language)
            if cache_key not in self._predictions:
                model = self._get_model()
                # BytesIO ensures inference consumes exactly the bytes we hashed;
                # faster-whisper handles decoding/resampling internally.
                segments, information = model.transcribe(
                    io.BytesIO(audio),
                    language=language,
                    task="transcribe",
                    beam_size=5,
                    vad_filter=False,
                    condition_on_previous_text=False,
                )
                pieces = []
                for segment in segments:  # Lazy inference errors can occur here.
                    if not isinstance(segment.text, str):
                        raise TranscriptionError("invalid_transcription_segment")
                    if segment.text.strip():
                        pieces.append(segment.text.strip())
                text = ("" if language == "zh" else " ").join(pieces)
                if not text.strip():
                    raise TranscriptionError("empty_transcription")
                prediction = {"text": text, "detected_language": getattr(information, "language", language)}
                probability = getattr(information, "language_probability", None)
                if isinstance(probability, (int, float)) and math.isfinite(probability):
                    prediction["language_probability"] = float(probability)
                if len(self._predictions) >= MAX_CACHED_PREDICTIONS:
                    self._predictions.popitem(last=False)
                self._predictions[cache_key] = prediction
            else:
                self._predictions.move_to_end(cache_key)
            return {
                "sample_id": record["sample_id"],
                **copy.deepcopy(self._predictions[cache_key]),
                "audio_sha256": digest,
                "language": language,
                "source_language": record["language"],
                "model": str(self.model_path),
                "model_revision": self.config.model_revision,
                "model_identity": self._identity,
                "model_identity_sha256": stable_hash(self._identity),
                "input_fingerprint": offline["input_fingerprint"],
            }
        except TranscriptionError:
            raise
        except Exception as exc:
            raise TranscriptionError(f"local_transcription_failed:{type(exc).__name__}") from exc


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--compute-type", choices=("int8", "float16"), default="int8")
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--max-duration", type=float, default=120.0)
    parser.add_argument("--max-audio-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument(
        "--max-runtime-seconds",
        type=float,
        default=3600.0,
        help="Stop starting records after this budget; an in-flight transcription finishes",
    )
    parser.add_argument(
        "--model-revision", help="Optional immutable checkpoint ID; otherwise hash local snapshot files"
    )
    parser.add_argument("--errors-output", type=Path)
    args = parser.parse_args(argv)
    if args.max_samples < 0 or min(args.max_duration, args.max_audio_bytes, args.max_runtime_seconds) <= 0:
        parser.error("max-samples must be nonnegative and all duration/byte/runtime limits must be positive")
    if args.output.resolve() == args.input.resolve():
        parser.error("hypothesis output must differ from the input manifest")
    errors_path = args.errors_output or args.output.with_suffix(".errors.jsonl")
    if errors_path.resolve() in {args.input.resolve(), args.output.resolve()}:
        parser.error("errors output must differ from input and hypothesis output")
    try:
        producer = LocalASRProducer(
            LocalASRConfig(
                model=args.model,
                device=args.device,
                compute_type=args.compute_type,
                model_revision=args.model_revision,
                max_duration=args.max_duration,
                max_audio_bytes=args.max_audio_bytes,
                audio_base=args.input.resolve().parent,
            )
        )
    except ValueError as exc:
        parser.error(str(exc))
    started = time.monotonic()
    hypotheses, errors = [], []
    counts = Counter()
    for record in read_jsonl(args.input):
        if counts["attempted"] >= args.max_samples:
            counts["stopped_at_sample_limit"] = 1
            break
        if time.monotonic() - started >= args.max_runtime_seconds:
            counts["stopped_at_runtime_limit"] = 1
            break
        counts["scanned"] += 1
        if record.get("task") != "asr" or record.get("qc", {}).get("offline", {}).get("decision") != "accept":
            counts["skipped_ineligible"] += 1
            continue
        counts["attempted"] += 1
        try:
            hypothesis = producer.transcribe(record)
            if hypothesis is not None:
                hypotheses.append(hypothesis)
                counts["transcribed"] += 1
        except TranscriptionError as exc:
            errors.append(
                {"sample_id": record.get("sample_id"), "audio_sha256": record.get("audio_sha256"), "reason": str(exc)}
            )
            counts["errors"] += 1
    write_jsonl(args.output, hypotheses)
    write_jsonl(errors_path, errors)
    print(
        json.dumps(
            {**counts, "errors_output": str(errors_path), "elapsed_seconds": round(time.monotonic() - started, 3)},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

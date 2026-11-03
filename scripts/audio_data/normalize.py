"""Normalize official extracted source layouts into auditable JSONL records.

No audio is cropped, resampled or overwritten. Missing media and unresolved
metadata remain explicit quarantine records for the offline filter.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import unquote

from .common import canonical_license, stable_hash, write_jsonl
from .download import DEFAULT_CONFIG, load_config
from .progress import Progress

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".aiff", ".aif"}
SPLITS = {
    "train": "train",
    "training": "train",
    "dev": "validation",
    "val": "validation",
    "valid": "validation",
    "validation": "validation",
    "test": "test",
    "eval": "test",
}


def _name(value) -> str:
    return unicodedata.normalize("NFC", html.unescape(unquote(str(value or "")))).strip()


def _csv(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def _split_from_path(path: Path) -> tuple[str, str]:
    for part in reversed(path.parts):
        token = part.lower()
        if token in SPLITS:
            return token, SPLITS[token]
        if token.startswith(("train-", "dev-", "test-", "nsynth-train", "nsynth-valid", "nsynth-test")):
            if token.startswith(("train", "nsynth-train")):
                return token, "train"
            if token.startswith(("dev", "nsynth-valid")):
                return token, "validation"
            return token, "test"
    return "unknown", "unknown"


def _audio_index(root: Path, progress="auto") -> dict[str, list[Path]]:
    index = defaultdict(list)
    with Progress("Index audio files", unit="entries", mode=progress) as indicator:
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
                index[_name(path.name)].append(path.resolve())
            indicator.update()
    for matches in index.values():
        matches.sort()
    return dict(index)


def _resolve_audio(index: dict, name: str, root: Path, preferred: Path | None = None) -> tuple[Path, list[str]]:
    if preferred is not None and preferred.is_file():
        return preferred.resolve(), []
    matches = index.get(_name(Path(_name(name)).name), [])
    if len(matches) == 1:
        return matches[0], []
    if len(matches) > 1:
        return matches[0], ["ambiguous_audio_path"]
    return (preferred or root / _name(name)).resolve(), ["missing_audio"]


def _base(
    dataset: str,
    source: dict,
    recording: str,
    audio: Path,
    task: str,
    split: str,
    source_split: str,
    question: str,
    reference,
    *,
    verified=False,
    group=None,
    language="und",
    metadata=None,
    audio_license=None,
) -> dict:
    metadata = dict(metadata or {})
    if split not in {"train", "validation", "test"}:
        metadata.setdefault("quarantine_reasons", []).append("unknown_official_split")
    return {
        "schema_version": 1,
        "sample_id": f"{dataset}:{recording}",
        "dataset": dataset,
        "revision": source["revision"],
        "source_split": source_split,
        "split": split,
        "source_recording_id": recording,
        "group_id": group or f"{dataset}:{recording}",
        "audio_path": str(audio.resolve()),
        "task": task,
        "language": language,
        "question": question,
        "reference": reference,
        "reference_aliases": [],
        "audio_license": canonical_license(audio_license or source.get("audio_license", "unknown")),
        "annotation_license": canonical_license(source.get("annotation_license", "unknown")),
        "annotation_type": "human",
        "reference_verified": bool(verified),
        "metadata": metadata,
    }


def iter_librispeech(root: Path, source: dict, progress="auto"):
    paths = sorted(root.rglob("*.trans.txt"))
    if not paths:
        raise FileNotFoundError("No official LibriSpeech *.trans.txt files found")
    for transcript in paths:
        source_split, split = _split_from_path(Path(root.name) / transcript.relative_to(root))
        with transcript.open(encoding="utf-8") as handle:
            for line in handle:
                fields = line.strip().split(maxsplit=1)
                if not fields:
                    continue
                recording, text = fields[0], fields[1] if len(fields) > 1 else ""
                parts = recording.split("-")
                reasons = []
                audio = transcript.parent / f"{recording}.flac"
                if not audio.is_file():
                    reasons.append("missing_audio")
                if not text:
                    reasons.append("missing_transcript")
                if len(parts) != 3:
                    reasons.append("invalid_librispeech_id")
                yield _base(
                    "librispeech",
                    source,
                    recording,
                    audio,
                    "asr",
                    split,
                    source_split,
                    "Transcribe the English speech verbatim.",
                    text,
                    language="en",
                    verified=bool(text),
                    group=f"librispeech:chapter:{'-'.join(parts[:2])}",
                    metadata={
                        "speaker_id": parts[0],
                        "chapter_id": parts[1] if len(parts) > 1 else None,
                        "transcript_path": str(transcript.resolve()),
                        "quarantine_reasons": reasons,
                    },
                )


def iter_aishell1(root: Path, source: dict, progress="auto"):
    transcripts = sorted(root.rglob("aishell_transcript_v0.8.txt"))
    if not transcripts:
        raise FileNotFoundError("No official AISHELL aishell_transcript_v0.8.txt found")
    index = _audio_index(root, progress)
    for transcript in transcripts:
        with transcript.open(encoding="utf-8-sig") as handle:
            for line in handle:
                fields = line.split()
                if not fields:
                    continue
                recording, text = fields[0], "".join(fields[1:])
                audio, reasons = _resolve_audio(index, recording + ".wav", root)
                source_split, split = _split_from_path(Path(root.name) / audio.relative_to(root.resolve()))
                if not text:
                    reasons.append("missing_transcript")
                match = re.match(r"BAC\d+(S\d+)W\d+", recording)
                speaker = match.group(1) if match else None
                yield _base(
                    "aishell1",
                    source,
                    recording,
                    audio,
                    "asr",
                    split,
                    source_split,
                    "请逐字转写音频中的普通话。",
                    text,
                    language="zh",
                    verified=bool(text),
                    group=f"aishell1:speaker:{speaker}" if speaker else f"aishell1:{recording}",
                    metadata={
                        "speaker_id": speaker,
                        "transcript_path": str(transcript.resolve()),
                        "transcription_spaces_removed": True,
                        "quarantine_reasons": reasons,
                    },
                )


def iter_fsd50k(root: Path, source: dict, progress="auto"):
    candidates = [p for p in sorted(root.rglob("*.csv")) if p.name in {"dev.csv", "eval.csv"}]
    if not candidates:
        raise FileNotFoundError("No FSD50K.ground_truth/dev.csv or eval.csv found")
    index = _audio_index(root, progress)
    for annotation in candidates:
        partition = annotation.stem
        metadata_files = sorted(root.rglob(f"{partition}_clips_info_FSD50K.json"))
        info = json.loads(metadata_files[0].read_text(encoding="utf-8")) if metadata_files else {}
        for row in _csv(annotation):
            recording = str(row.get("fname", "")).removesuffix(".wav")
            audio, reasons = _resolve_audio(index, recording + ".wav", root)
            clip = info.get(recording, {})
            if not clip:
                reasons.append("missing_source_metadata")
            if partition == "eval":
                source_split, split = "eval", "test"
            else:
                source_split = str(row.get("split", "unknown")).lower()
                split = SPLITS.get(source_split, "unknown")
            labels = [label.strip() for label in str(row.get("labels", "")).split(",") if label.strip()]
            if not labels:
                reasons.append("missing_positive_labels")
            license_value = clip.get("license", "unknown")
            if not clip.get("license"):
                reasons.append("missing_audio_license")
            record = _base(
                "fsd50k",
                source,
                recording,
                audio,
                "classification",
                split,
                f"{partition}:{source_split}",
                "Which sound events can you hear in the recording? List the clearly audible events.",
                [label.replace("_", " ") for label in labels],
                verified=bool(labels),
                group=f"freesound:{recording}",
                audio_license=license_value,
                metadata={
                    "label_set_complete": False,
                    "source_labels": labels,
                    "ontology_mids": str(row.get("mids", "")).split(","),
                    "uploader": clip.get("uploader"),
                    "source_title": clip.get("title"),
                    "source_url": f"https://freesound.org/s/{recording}/",
                    "quarantine_reasons": reasons,
                },
            )
            yield record


def iter_clotho_aqa(root: Path, source: dict, progress="auto"):
    metadata_paths = sorted(root.rglob("clotho_aqa_metadata.csv"))
    metadata = {}
    if metadata_paths:
        metadata = {_name(row.get("file_name")): row for row in _csv(metadata_paths[0])}
    annotation_paths = sorted(root.rglob("clotho_aqa_*.csv"))
    annotation_paths = [p for p in annotation_paths if re.fullmatch(r"clotho_aqa_(train|val|test)\.csv", p.name)]
    if not annotation_paths:
        raise FileNotFoundError("No official clotho_aqa_{train,val,test}.csv found")
    index = _audio_index(root, progress)
    for annotation in annotation_paths:
        source_split = annotation.stem.removeprefix("clotho_aqa_")
        split = SPLITS[source_split]
        grouped = defaultdict(list)
        for row in _csv(annotation):
            grouped[(_name(row.get("file_name")), str(row.get("QuestionText", "")).strip())].append(row)
        for (filename, question), rows in grouped.items():
            audio, reasons = _resolve_audio(index, filename, root)
            clip = metadata.get(filename, {})
            if not clip:
                reasons.append("missing_source_metadata")
            answers = [" ".join(str(row.get("answer", "")).strip().casefold().split()) for row in rows]
            confident = all(str(row.get("confidence", "")).strip().casefold() == "yes" for row in rows)
            unanimous = len(rows) == 3 and len(set(answers)) == 1 and bool(answers[0]) and confident
            if not unanimous:
                reasons.append("nonunanimous_or_unconfident_annotations")
            if not question:
                reasons.append("missing_question")
            sound_link = str(clip.get("sound_link", ""))
            match = re.search(r"/(?:sounds/|s/)(\d+)", sound_link)
            recording = f"freesound:{match.group(1)}" if match else f"clotho:{filename}"
            if not match:
                reasons.append("missing_original_recording_id")
            license_value = clip.get("license", "unknown")
            if not clip.get("license"):
                reasons.append("missing_audio_license")
            reference = answers[0] if unanimous else list(dict.fromkeys(answer for answer in answers if answer))
            record = _base(
                "clotho_aqa",
                source,
                recording,
                audio,
                "closed_qa",
                split,
                source_split,
                question,
                reference,
                language="en",
                verified=unanimous,
                group=recording,
                audio_license=license_value,
                metadata={
                    "source_file_name": filename,
                    "source_url": sound_link,
                    "source_start_end_samples": clip.get("start_end_samples", ""),
                    "audio_is_official_distributed_clip": True,
                    "annotator_count": len(rows),
                    "answer_counts": dict(Counter(answers)),
                    "all_confident": confident,
                    "uploader": clip.get("manufacturer"),
                    "quarantine_reasons": reasons,
                },
            )
            record["sample_id"] = f"clotho_aqa:{stable_hash([filename, question])[:24]}"
            yield record


def iter_nsynth(root: Path, source: dict, progress="auto"):
    annotations = sorted(root.rglob("examples.json"))
    if not annotations:
        raise FileNotFoundError("No official NSynth examples.json found (JSON/WAV archive required)")
    families = (
        "bass",
        "brass",
        "flute",
        "guitar",
        "keyboard",
        "mallet",
        "organ",
        "reed",
        "string",
        "synth_lead",
        "vocal",
    )
    for annotation in annotations:
        source_split, split = _split_from_path(Path(root.name) / annotation.relative_to(root))
        examples = json.loads(annotation.read_text(encoding="utf-8"))
        for recording, item in sorted(examples.items()):
            audio = annotation.parent / "audio" / f"{recording}.wav"
            reasons = [] if audio.is_file() else ["missing_audio"]
            family = item.get("instrument_family_str", "")
            family_id = item.get("instrument_family")
            if not family and isinstance(family_id, int) and 0 <= family_id < len(families):
                family = families[family_id]
            if family not in families:
                reasons.append("unknown_instrument_family")
            elif isinstance(family_id, int) and (
                family_id < 0 or family_id >= len(families) or families[family_id] != family
            ):
                reasons.append("inconsistent_instrument_family")
            instrument = item.get("instrument", recording.rsplit("-", 2)[0])
            record = _base(
                "nsynth",
                source,
                recording,
                audio,
                "music_family",
                split,
                source_split,
                "Which instrument family best describes the note? Answer one of: " + ", ".join(families) + ".",
                family,
                verified=family in families and "inconsistent_instrument_family" not in reasons,
                group=f"nsynth:instrument:{instrument}",
                metadata={
                    "pitch_midi": item.get("pitch"),
                    "velocity": item.get("velocity"),
                    "instrument": instrument,
                    "instrument_source": item.get("instrument_source_str"),
                    "family_id": family_id,
                    "quarantine_reasons": reasons,
                },
            )
            record["annotation_type"] = "metadata"
            if family == "synth_lead":
                record["reference_aliases"] = ["synth lead"]
            yield record


ADAPTERS = {
    "librispeech": iter_librispeech,
    "aishell1": iter_aishell1,
    "fsd50k": iter_fsd50k,
    "clotho_aqa": iter_clotho_aqa,
    "nsynth": iter_nsynth,
}


def iter_records(dataset: str, source_root: Path, source: dict, progress="auto"):
    if dataset not in ADAPTERS:
        raise ValueError(f"No pinned source adapter for {dataset}; restricted candidates remain disabled")
    yield from ADAPTERS[dataset](Path(source_root).resolve(), source, progress)


def select_records(records, *, split="all", max_samples=1000, sample_seed=None):
    """Select a prefix or uniform bounded reservoir independently per split.

    Seeded sampling scans all annotation records, without decoding audio. It
    samples records rather than balancing speakers, classes, or recordings.
    Unlimited selection always streams its input unchanged.
    """
    if max_samples < 0:
        raise ValueError("max_samples must be nonnegative")
    counts = Counter()
    reservoirs = defaultdict(list)
    generators = {}
    for ordinal, record in enumerate(records):
        record_split = record["split"]
        if split != "all" and record_split != split:
            continue
        if not max_samples:
            yield record
            continue
        counts[record_split] += 1
        if sample_seed is None:
            if counts[record_split] <= max_samples:
                yield record
            continue
        reservoir = reservoirs[record_split]
        if len(reservoir) < max_samples:
            reservoir.append((ordinal, record))
            continue
        if record_split not in generators:
            generators[record_split] = random.Random(stable_hash([sample_seed, record_split]))
        replacement = generators[record_split].randrange(counts[record_split])
        if replacement < max_samples:
            reservoir[replacement] = (ordinal, record)
    if sample_seed is not None and max_samples:
        for _, record in sorted(item for reservoir in reservoirs.values() for item in reservoir):
            yield record


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=tuple(load_config()["datasets"]))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split", choices=("train", "validation", "test", "all"), default="all")
    parser.add_argument("--max-samples", type=int, default=1000, help="Maximum per official split; 0 means unlimited")
    parser.add_argument(
        "--sample-seed",
        type=int,
        help="Sample a deterministic uniform reservoir per split instead of taking the source prefix",
    )
    parser.add_argument("--allow-noncommercial", action="store_true")
    parser.add_argument("--progress", choices=("auto", "on", "off"), default="auto")
    args = parser.parse_args(argv)
    if args.max_samples < 0:
        parser.error("--max-samples must be nonnegative")
    source = load_config(args.config)["datasets"][args.dataset]
    if source.get("restricted") and not args.allow_noncommercial:
        parser.error(f"{args.dataset} requires --allow-noncommercial")
    counts = Counter()

    def scanned():
        with Progress(f"Scan {args.dataset} metadata", mode=args.progress) as indicator:
            for record in iter_records(args.dataset, args.source_root, source, progress=args.progress):
                indicator.update()
                yield record

    def selected():
        records = select_records(
            scanned(),
            split=args.split,
            max_samples=args.max_samples,
            sample_seed=args.sample_seed,
        )
        for record in records:
            counts[record["split"]] += 1
            yield record

    write_jsonl(args.output, selected())
    if not counts:
        raise ValueError("No records matched the requested split; check the official extracted layout")
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "counts_by_split": dict(counts),
                "sampling": {
                    "method": "all"
                    if not args.max_samples
                    else "reservoir"
                    if args.sample_seed is not None
                    else "prefix",
                    "seed": args.sample_seed,
                    "max_samples_per_split": args.max_samples,
                },
            }
        )
    )


if __name__ == "__main__":
    main()

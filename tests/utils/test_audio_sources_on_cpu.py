"""Official-layout parser and transfer integrity tests; no source downloads."""

import csv
import hashlib
import io
import json
import tarfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from scripts.audio_data import normalize
from scripts.audio_data.download import download_asset, extract_assets, load_config, main, safe_extract
from scripts.audio_data.normalize import iter_records, select_records


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fixture audio")
    return path


def source(name):
    return load_config()["datasets"][name]


def test_librispeech_official_split_and_chapter(tmp_path):
    folder = tmp_path / "LibriSpeech" / "dev-clean" / "374" / "180298"
    touch(folder / "374-180298-0000.flac")
    (folder / "374-180298.trans.txt").write_text("374-180298-0000 CHAPTER SIXTEEN\n", encoding="utf-8")
    (record,) = iter_records("librispeech", tmp_path, source("librispeech"))
    assert record["split"] == "validation"
    assert record["source_split"] == "dev-clean"
    assert record["group_id"] == "librispeech:chapter:374-180298"
    assert record["reference"] == "CHAPTER SIXTEEN"
    assert record["reference_verified"]


def test_aishell_official_transcript_and_speaker(tmp_path, capsys):
    transcript = tmp_path / "data_aishell" / "transcript" / "aishell_transcript_v0.8.txt"
    transcript.parent.mkdir(parents=True)
    transcript.write_text("BAC009S0002W0122 而 对 楼市 成交\n", encoding="utf-8")
    audio = touch(tmp_path / "data_aishell" / "wav" / "train" / "S0002" / "BAC009S0002W0122.wav")
    (record,) = iter_records("aishell1", tmp_path, source("aishell1"), progress="on")
    assert record["reference"] == "而对楼市成交"
    assert record["split"] == "train"
    assert record["audio_path"] == str(audio.resolve())
    assert record["group_id"] == "aishell1:speaker:S0002"
    assert "Index audio files" in capsys.readouterr().err


def test_fsd50k_preserves_validation_and_partial_positive_labels(tmp_path):
    write_csv(
        tmp_path / "FSD50K.ground_truth" / "dev.csv",
        [
            {"fname": "64760", "labels": "Electric_guitar,Guitar", "mids": "/m/a,/m/b", "split": "val"},
        ],
    )
    info_path = tmp_path / "FSD50K.metadata" / "dev_clips_info_FSD50K.json"
    info_path.parent.mkdir()
    info_path.write_text(json.dumps({"64760": {"license": "http://creativecommons.org/licenses/by/3.0/"}}))
    touch(tmp_path / "FSD50K.dev_audio" / "64760.wav")
    (record,) = iter_records("fsd50k", tmp_path, source("fsd50k"))
    assert record["split"] == "validation"
    assert record["reference"] == ["Electric guitar", "Guitar"]
    assert record["metadata"]["label_set_complete"] is False
    assert record["group_id"] == "freesound:64760"
    assert record["audio_license"] == "CC-BY-3.0"


def test_clotho_consensus_html_names_license_and_original_id(tmp_path):
    filename = "Wind & Water.wav"
    touch(tmp_path / "audio_files" / filename)
    write_csv(
        tmp_path / "clotho_aqa_metadata.csv",
        [
            {
                "file_name": "Wind &amp; Water.wav",
                "license": "http://creativecommons.org/publicdomain/zero/1.0/",
                "sound_link": "https://freesound.org/people/example/sounds/109861",
                "start_end_samples": "[100, 200]",
            }
        ],
    )
    write_csv(
        tmp_path / "clotho_aqa_train.csv",
        [
            {
                "file_name": "Wind &amp; Water.wav",
                "QuestionText": "Is there wind?",
                "answer": "yes",
                "confidence": "yes",
            }
        ]
        * 3,
    )
    (record,) = iter_records("clotho_aqa", tmp_path, source("clotho_aqa"))
    assert record["reference_verified"]
    assert record["reference"] == "yes"
    assert record["audio_license"] == "CC0-1.0"
    assert Path(record["audio_path"]).name == filename
    assert record["group_id"] == "freesound:109861"
    assert record["metadata"]["source_start_end_samples"] == "[100, 200]"
    assert "start" not in record  # Distributed Clotho audio is already its intended clip.


def test_clotho_missing_metadata_and_disagreement_are_quarantined(tmp_path):
    rows = [
        {"file_name": "missing.wav", "QuestionText": "Any speech?", "answer": answer, "confidence": "yes"}
        for answer in ("yes", "yes", "no")
    ]
    write_csv(tmp_path / "clotho_aqa_train.csv", rows)
    (record,) = iter_records("clotho_aqa", tmp_path, source("clotho_aqa"))
    assert record["reference_verified"] is False
    assert {"missing_source_metadata", "missing_audio", "nonunanimous_or_unconfident_annotations"}.issubset(
        record["metadata"]["quarantine_reasons"]
    )


def test_nsynth_family_and_split_when_root_is_split_directory(tmp_path):
    root = tmp_path / "nsynth-valid"
    touch(root / "audio" / "bass_synthetic_033-022-050.wav")
    (root / "examples.json").write_text(
        json.dumps(
            {
                "bass_synthetic_033-022-050": {
                    "instrument_family_str": "bass",
                    "instrument_family": 0,
                    "instrument": 42,
                    "pitch": 22,
                    "velocity": 50,
                    "instrument_source_str": "synthetic",
                }
            }
        )
    )
    (record,) = iter_records("nsynth", root, source("nsynth"))
    assert record["split"] == "validation"
    assert record["reference"] == "bass"
    assert record["group_id"] == "nsynth:instrument:42"
    assert record["annotation_type"] == "metadata"


@pytest.fixture
def http_archive():
    data = b"download fixture " * 100
    ranges = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            range_header = self.headers.get("Range")
            ranges.append(range_header)
            offset = int(range_header.removeprefix("bytes=").removesuffix("-")) if range_header else 0
            self.send_response(206 if range_header else 200)
            self.send_header("Content-Length", str(len(data) - offset))
            self.send_header("ETag", '"fixture-v1"')
            if range_header:
                self.send_header("Content-Range", f"bytes {offset}-{len(data) - 1}/{len(data)}")
            self.end_headers()
            self.wfile.write(data[offset:])

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield (
            {
                "filename": "fixture.bin",
                "url": f"http://127.0.0.1:{server.server_port}/fixture.bin",
                "size_bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            },
            data,
            ranges,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_downloader_resumes_range_checksums_and_reuses_receipt(tmp_path, http_archive, capsys):
    asset, data, ranges = http_archive
    (tmp_path / "fixture.bin.part").write_bytes(data[:100])
    (tmp_path / "fixture.bin.part.json").write_text(json.dumps({"url": asset["url"], "validator": '"fixture-v1"'}))
    transferred = download_asset(asset, tmp_path, len(data) - 100, progress="on")
    assert transferred == len(data) - 100
    assert ranges == ["bytes=100-"]
    assert (tmp_path / "fixture.bin").read_bytes() == data
    assert download_asset(asset, tmp_path, 0) == 0
    assert len(ranges) == 1
    receipt = json.loads((tmp_path / "fixture.bin.receipt.json").read_text())
    assert receipt["publisher_checksum_verified"] is True
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Download fixture.bin: 100 B/" in captured.err
    assert "Checksum sha256 fixture.bin.part" in captured.err


def test_downloader_enforces_network_budget_before_transfer(tmp_path, http_archive):
    asset, data, ranges = http_archive
    with pytest.raises(ValueError, match="budget too small"):
        download_asset(asset, tmp_path, len(data) - 1)
    assert not ranges


def test_downloader_rejects_checksum_mismatch(tmp_path, http_archive):
    asset, data, _ = http_archive
    with pytest.raises(ValueError, match="sha256 mismatch"):
        download_asset({**asset, "sha256": "0" * 64}, tmp_path, len(data))
    assert not (tmp_path / "fixture.bin").exists()


@pytest.mark.parametrize("kind", ["traversal", "symlink", "oversized"])
def test_safe_extract_rejects_unsafe_tar_or_budget(tmp_path, kind):
    archive = tmp_path / "source.tar"
    with tarfile.open(archive, "w") as handle:
        item = tarfile.TarInfo("../escape" if kind == "traversal" else "safe/file.txt")
        if kind == "symlink":
            item.type = tarfile.SYMTYPE
            item.linkname = "/etc/passwd"
            handle.addfile(item)
        else:
            item.size = 20
            handle.addfile(item, io.BytesIO(b"x" * 20))
    with pytest.raises(ValueError):
        safe_extract(archive, tmp_path / "extracted", max_bytes=10 if kind == "oversized" else 100)
    assert not (tmp_path / "escape").exists()


def test_extract_metadata_and_aishell_nested_archives(tmp_path):
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    (downloads / "metadata.csv").write_text("file,license\n")
    nested = io.BytesIO()
    with tarfile.open(fileobj=nested, mode="w:gz") as handle:
        member = tarfile.TarInfo("train/S0002/sample.wav")
        member.size = 4
        handle.addfile(member, io.BytesIO(b"wave"))
    outer = downloads / "data_aishell.tgz"
    with tarfile.open(outer, "w:gz") as handle:
        member = tarfile.TarInfo("data_aishell/wav/train.tar.gz")
        member.size = len(nested.getvalue())
        handle.addfile(member, io.BytesIO(nested.getvalue()))
    destination = tmp_path / "extracted"
    extract_assets(
        [{"filename": "metadata.csv"}, {"filename": outer.name}], downloads, destination, 100_000, "aishell1"
    )
    assert (destination / "metadata.csv").exists()
    assert (destination / "data_aishell/wav/train/S0002/sample.wav").read_bytes() == b"wave"


def test_download_plan_is_network_free_and_does_not_create_root(tmp_path, capsys):
    root = tmp_path / "data"
    main(["--dataset", "librispeech", "--root", str(root)])
    assert json.loads(capsys.readouterr().out)["execute"] is False
    assert not root.exists()


def test_seeded_reservoir_reproducible_and_not_a_speaker_prefix():
    records = [{"sample_id": f"row-{i}", "split": "train", "speaker_id": i // 100} for i in range(1000)]
    selected = list(select_records(iter(records), max_samples=10, sample_seed=42))
    repeated = list(select_records(iter(records), max_samples=10, sample_seed=42))
    other_seed = list(select_records(iter(records), max_samples=10, sample_seed=43))
    assert selected == repeated
    assert selected != other_seed
    assert selected != records[:10]
    assert len({record["speaker_id"] for record in selected}) >= 4
    assert [records.index(record) for record in selected] == sorted(records.index(record) for record in selected)
    assert list(select_records(iter(records), max_samples=10)) == records[:10]


def test_seeded_reservoir_cap_and_rng_are_independent_per_split():
    splits = ("train", "validation", "test")
    records = [{"sample_id": f"{split}-{i}", "split": split} for i in range(100) for split in splits]
    selected = list(select_records(iter(records), max_samples=7, sample_seed=42))
    assert len(selected) == 21
    for split in splits:
        per_split = [record for record in selected if record["split"] == split]
        assert len(per_split) == 7
        assert per_split == list(select_records(iter(records), split=split, max_samples=7, sample_seed=42))
        assert all(record in records for record in per_split)


@pytest.mark.parametrize("sample_seed", [None, 42])
def test_unlimited_sampling_streams_original_records_without_buffering(sample_seed):
    records = [{"sample_id": f"row-{i}", "split": "train"} for i in range(10)]
    consumed = []

    def source_records():
        for record in records:
            consumed.append(record["sample_id"])
            yield record

    selected = select_records(source_records(), max_samples=0, sample_seed=sample_seed)
    assert consumed == []
    first = next(selected)
    assert first is records[0]
    assert consumed == [records[0]["sample_id"]]
    result = [first, *selected]
    assert len(result) == len(records)
    assert all(original is selected for original, selected in zip(records, result, strict=True))


def test_seeded_reservoir_preserves_small_splits_without_duplication():
    records = [{"sample_id": f"row-{i}", "split": split} for i, split in enumerate(("train", "test", "train"))]
    assert list(select_records(iter(records), max_samples=10, sample_seed=42)) == records


def test_normalizer_cli_reports_seeded_per_split_sample(tmp_path, monkeypatch, capsys):
    records = [{"sample_id": f"{split}-{i}", "split": split} for i in range(30) for split in ("train", "test")]
    monkeypatch.setattr(normalize, "iter_records", lambda *args, **kwargs: iter(records))
    output = tmp_path / "sampled.jsonl"
    normalize.main(
        [
            "--dataset",
            "librispeech",
            "--source-root",
            str(tmp_path),
            "--output",
            str(output),
            "--max-samples",
            "5",
            "--sample-seed",
            "42",
            "--progress",
            "on",
        ]
    )
    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert "Scan librispeech metadata: 60 records" in captured.err
    assert report["counts_by_split"] == {"train": 5, "test": 5}
    assert report["sampling"] == {"method": "reservoir", "seed": 42, "max_samples_per_split": 5}
    actual = [json.loads(line) for line in output.read_text().splitlines()]
    assert actual == list(select_records(iter(records), max_samples=5, sample_seed=42))

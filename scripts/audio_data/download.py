"""Plan, download and safely extract complete official audio dataset archives.

The default is a network-free plan. Execution requires an explicit total network
byte budget. Partial files survive failed requests and can be resumed on rerun.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit, urlunsplit

import yaml

from . import parallel_download
from .progress import Progress

DEFAULT_CONFIG = Path(__file__).with_name("datasets.yaml")


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("Expected a schema_version: 1 dataset configuration")
    return config


def _digest(path: Path, algorithm: str = "sha256", progress="auto") -> str:
    digest = hashlib.new(algorithm)
    with (
        path.open("rb") as handle,
        Progress(
            f"Checksum {algorithm} {path.name}",
            total=path.stat().st_size,
            unit="bytes",
            mode=progress,
        ) as indicator,
    ):
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
            indicator.update(len(block))
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _verify(path: Path, asset: dict, progress="auto") -> dict:
    if asset.get("size_bytes") is not None and path.stat().st_size != int(asset["size_bytes"]):
        raise ValueError(f"Size mismatch: {path}")
    checksums = {}
    for algorithm in ("sha256", "md5"):
        if asset.get(algorithm):
            actual = _digest(path, algorithm, progress)
            if actual.lower() != str(asset[algorithm]).lower():
                raise ValueError(f"{algorithm} mismatch: {path}; remove the corrupt file before retrying")
            checksums[algorithm] = actual
    if "sha256" not in checksums:
        checksums["sha256"] = _digest(path, progress=progress)
    return {
        "url": asset["url"],
        "size_bytes": path.stat().st_size,
        "checksums": checksums,
        "publisher_checksum_verified": bool(asset.get("sha256") or asset.get("md5")),
    }


def select_mirror(asset: dict, mirror="original") -> dict:
    """Select an official OpenSLR mirror without rewriting other publishers."""
    if mirror not in {"original", "elda"}:
        raise ValueError("mirror must be original or elda")
    selected = dict(asset)
    parsed = urlsplit(asset["url"])
    if mirror == "elda" and parsed.hostname in {"www.openslr.org", "openslr.org", "openslr.elda.org"}:
        if not parsed.path.startswith("/resources/"):
            raise ValueError("Only OpenSLR /resources/ assets support mirror selection")
        selected["url"] = urlunsplit(("https", "openslr.elda.org", parsed.path, parsed.query, parsed.fragment))
    return selected


def download_asset(
    asset: dict, directory: Path, remaining_bytes: int, timeout: float = 60, progress="auto", connections: int = 1
) -> int:
    """Download one asset, protecting its prefix/chunks from concurrent writers."""
    filename = str(asset["filename"])
    if Path(filename).name != filename or filename in {"", ".", ".."}:
        raise ValueError(f"Asset filename must be a basename: {filename}")
    if not isinstance(connections, int) or not 1 <= connections <= 64:
        raise ValueError("connections must be between 1 and 64")
    if remaining_bytes < 0:
        raise ValueError("remaining_bytes must be nonnegative")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / (filename + ".download.lock")).open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f"Another downloader is writing {filename}") from exc
        return _download_asset(asset, directory, remaining_bytes, timeout, progress, connections)


def _download_asset(asset, directory, remaining_bytes, timeout, progress, connections):
    """Return network bytes read, recording a SHA256 receipt on success.

    SHA/MD5 values in configuration are independently checked when published.
    AISHELL/NSynth have size validation and local receipts, explicitly marked as
    lacking a publisher digest; a local checksum is not a publisher checksum.
    """
    filename = str(asset["filename"])
    if Path(filename).name != filename or filename in {"", ".", ".."}:
        raise ValueError(f"Asset filename must be a basename: {filename}")
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / filename
    receipt_path = destination.with_name(filename + ".receipt.json")
    if destination.is_file():
        receipt = _verify(destination, asset, progress)
        if receipt_path.exists():
            old = json.loads(receipt_path.read_text(encoding="utf-8"))
            if old.get("checksums", {}).get("sha256") != receipt["checksums"]["sha256"]:
                raise ValueError(f"Local SHA256 receipt mismatch: {destination}")
        _atomic_json(receipt_path, receipt)
        parallel_download.cleanup(destination.with_name(filename + ".part"))
        return 0

    partial = destination.with_name(filename + ".part")
    state_path = destination.with_name(filename + ".part.json")
    saved_state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
    if connections > 1 or saved_state.get("parallel"):
        transferred = parallel_download.transfer(
            asset,
            partial,
            state_path,
            remaining_bytes,
            connections=connections,
            timeout=timeout,
            progress=progress,
        )
        if transferred is not None:
            receipt = _verify(partial, asset, progress)
            partial.replace(destination)
            _atomic_json(receipt_path, receipt)
            state_path.unlink(missing_ok=True)
            parallel_download.cleanup(partial)
            return transferred
    offset = 0
    state = {}
    if partial.exists() and state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        publisher_digest = bool(asset.get("sha256") or asset.get("md5"))
        if state.get("url") and state.get("url") != asset["url"]:
            if not publisher_digest and partial.stat().st_size:
                raise ValueError(
                    "Cannot resume across mirrors without a publisher checksum; "
                    "use the original mirror or a fresh directory"
                )
            state = {"url": asset["url"], "validator": None}
        if state.get("url") == asset["url"] and (
            parallel_download._usable_validator(state.get("validator")) or publisher_digest
        ):
            offset = partial.stat().st_size
    if asset.get("size_bytes") is not None:
        expected = int(asset["size_bytes"])
        if offset > expected:
            raise ValueError(f"Partial file is larger than the expected archive: {partial}")
        if offset == expected and offset > 0:
            receipt = _verify(partial, asset, progress)
            partial.replace(destination)
            _atomic_json(receipt_path, receipt)
            state_path.unlink(missing_ok=True)
            return 0
        if expected - offset > remaining_bytes:
            raise ValueError(f"Byte budget too small for {filename}: needs {expected - offset}, has {remaining_bytes}")
    headers = {"User-Agent": "verl-audio-data/2", "Accept-Encoding": "identity"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
        if parallel_download._usable_validator(state.get("validator")):
            headers["If-Range"] = state["validator"]
    request = urllib.request.Request(asset["url"], headers=headers)
    transferred = 0
    with Progress(f"Request {filename}", total=1, unit="requests", mode=progress) as indicator:
        response = urllib.request.urlopen(request, timeout=timeout)
        indicator.update()
    with response:
        status = response.getcode()
        if status == 206:
            content_range = response.headers.get("Content-Range", "")
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+|\*)", content_range)
            if not offset or not match or int(match.group(1)) != offset:
                raise ValueError(f"Unexpected resume Content-Range: {content_range}")
            expected = asset.get("size_bytes")
            if expected is not None and (int(match.group(2)) != int(expected) - 1 or match.group(3) != str(expected)):
                raise ValueError(f"Unexpected resume Content-Range size: {content_range}")
            if parallel_download._usable_validator(state.get("validator")):
                if parallel_download._validator(response.headers) != state["validator"]:
                    raise ValueError("Remote validator changed during serial resume; saved partial retained")
        elif status == 200:
            offset = 0  # Server ignored Range or the remote object changed.
        else:
            raise ValueError(f"Unexpected HTTP status: {status}")
        length = response.headers.get("Content-Length")
        if status == 206 and length is not None and int(length) != int(match.group(2)) - offset + 1:
            raise ValueError("Resume Content-Length does not match requested bytes")
        if response.headers.get("Content-Encoding", "identity").lower() not in {"", "identity"}:
            raise ValueError("Server returned encoded bytes despite Accept-Encoding: identity")
        if length and int(length) > remaining_bytes:
            raise ValueError(f"Byte budget too small for {filename}: response has {length} bytes")
        validator = parallel_download._validator(response.headers)
        if not offset:
            # Commit an empty prefix before labeling it with this remote
            # version. A crash must not associate old bytes with a new ETag.
            with partial.open("wb") as stream:
                stream.flush()
                os.fsync(stream.fileno())
        _atomic_json(state_path, {"url": asset["url"], "validator": validator})
        total = (
            int(asset["size_bytes"])
            if asset.get("size_bytes") is not None
            else (offset + int(length) if length else None)
        )
        with (
            partial.open("ab" if offset else "wb") as handle,
            Progress(
                f"Download {filename}",
                total=total,
                unit="bytes",
                mode=progress,
                initial=offset,
            ) as indicator,
        ):
            while True:
                if transferred == remaining_bytes:
                    if length and transferred == int(length):
                        break
                    raise ValueError("Network byte budget exhausted; partial file retained")
                block = response.read(min(1024 * 1024, remaining_bytes - transferred))
                if not block:
                    break
                transferred += len(block)
                if transferred > remaining_bytes:
                    raise ValueError("Network byte budget exhausted; partial file retained")
                handle.write(block)
                indicator.update(len(block))
        if length and transferred != int(length):
            raise ValueError(f"Incomplete response for {filename}; rerun to resume")
    receipt = _verify(partial, asset, progress)
    partial.replace(destination)
    _atomic_json(receipt_path, receipt)
    state_path.unlink(missing_ok=True)
    return transferred


def _safe_target(destination: Path, name: str) -> Path:
    normalized = name.replace("\\", "/")
    parts = PurePosixPath(normalized)
    if parts.is_absolute() or ".." in parts.parts or re.match(r"^[A-Za-z]:", normalized):
        raise ValueError(f"Unsafe archive member: {name}")
    target = destination.joinpath(*parts.parts)
    if not target.resolve().is_relative_to(destination.resolve()):
        raise ValueError(f"Archive member escapes extraction directory: {name}")
    for parent in (target, *target.parents):
        if parent == destination.parent:
            break
        if parent.is_symlink():
            raise ValueError(f"Archive target traverses a symlink: {name}")
    return target


def safe_extract(archive: Path, destination: Path, *, max_bytes: int, progress="auto") -> int:
    """Extract regular tar/zip members only, rejecting traversal and links."""
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    written = 0
    files = 0
    indicator = Progress(f"Extract {archive.name}", unit="bytes", mode=progress)

    def extract_file(name, size, source):
        nonlocal written, files
        if size < 0 or written + size > max_bytes:
            raise ValueError(f"Extraction byte budget exceeded by {name}")
        target = _safe_target(destination, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temp_name = tempfile.mkstemp(prefix=".audio-extract-", dir=target.parent)
        try:
            with os.fdopen(descriptor, "wb") as output:
                remaining = size
                while remaining:
                    block = source.read(min(1024 * 1024, remaining))
                    if not block:
                        raise ValueError(f"Truncated archive member: {name}")
                    output.write(block)
                    remaining -= len(block)
                    indicator.update(len(block))
            Path(temp_name).replace(target)
        finally:
            Path(temp_name).unlink(missing_ok=True)
        written += size
        files += 1
        indicator.label = f"Extract {archive.name} ({files:,} files)"

    try:
        if zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive) as handle:
                indicator.set_total(sum(member.file_size for member in handle.infolist() if not member.is_dir()))
                for member in handle.infolist():
                    target = _safe_target(destination, member.filename)
                    mode = member.external_attr >> 16
                    if (mode & 0o170000) == 0o120000:
                        raise ValueError(f"Archive symlink rejected: {member.filename}")
                    if member.is_dir():
                        target.mkdir(parents=True, exist_ok=True)
                    else:
                        with handle.open(member) as source:
                            extract_file(member.filename, member.file_size, source)
        else:
            with tarfile.open(archive, "r:*") as handle:
                for member in handle:
                    target = _safe_target(destination, member.name)
                    if member.isdir():
                        target.mkdir(parents=True, exist_ok=True)
                    elif member.isfile():
                        with handle.extractfile(member) as source:
                            extract_file(member.name, member.size, source)
                    else:
                        raise ValueError(f"Archive link/special file rejected: {member.name}")
    finally:
        indicator.close()
    return written


def extract_assets(
    assets: list[dict], directory: Path, destination: Path, max_bytes: int, dataset: str, progress="auto"
) -> int:
    """Handle FSD50K split ZIPs and AISHELL's nested WAV archives."""
    used = 0
    for asset in assets:
        archive = directory / asset["filename"]
        if archive.suffix.lower() not in {".zip", ".tgz", ".gz", ".tar"}:
            if not re.fullmatch(r"\.z\d+", archive.suffix.lower()):
                size = archive.stat().st_size
                if used + size > max_bytes:
                    raise ValueError("Extraction byte budget exceeded while copying metadata")
                destination.mkdir(parents=True, exist_ok=True)
                with Progress(f"Copy metadata {archive.name}", total=size, unit="bytes", mode=progress) as indicator:
                    shutil.copyfile(archive, _safe_target(destination, archive.name))
                    indicator.update(size)
                used += size
            continue
        combined = None
        if archive.suffix == ".zip" and archive.with_suffix(".z01").exists():
            if not shutil.which("zip"):
                raise ValueError("FSD50K split ZIP extraction requires the 'zip' executable (zip -s 0)")
            combined = archive.with_name(archive.stem + ".combined.zip")
            combined.unlink(missing_ok=True)
            # zip reassembles the container; Python below validates every path.
            with Progress(f"Reassemble {archive.name}", unit="bytes", mode=progress) as indicator:
                with subprocess.Popen(["zip", "-q", "-s", "0", str(archive), "--out", str(combined)]) as process:
                    while process.poll() is None:
                        size = combined.stat().st_size if combined.exists() else 0
                        indicator.update(max(0, size - indicator.current))
                        time.sleep(0.2)
                    if process.returncode:
                        raise subprocess.CalledProcessError(process.returncode, process.args)
                    size = combined.stat().st_size if combined.exists() else 0
                    indicator.update(max(0, size - indicator.current))
            archive = combined
        try:
            used += safe_extract(archive, destination, max_bytes=max_bytes - used, progress=progress)
        finally:
            if combined:
                combined.unlink(missing_ok=True)
    if dataset == "aishell1":
        for nested in sorted(destination.rglob("*.tar.gz")):
            if "wav" in nested.relative_to(destination).parts:
                used += safe_extract(nested, nested.parent, max_bytes=max_bytes - used, progress=progress)
    return used


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", action="append", help="Repeat to select sources; default: enabled sources")
    parser.add_argument("--assets", action="append", help="Exact asset basename; repeat to select a subset")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help="Without this flag print a network-free JSON plan")
    parser.add_argument("--max-download-bytes", type=int)
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--max-extracted-bytes", type=int, default=100_000_000_000)
    parser.add_argument("--allow-noncommercial", action="store_true")
    parser.add_argument("--require-publisher-checksum", action="store_true")
    parser.add_argument("--progress", choices=("auto", "on", "off"), default="auto")
    parser.add_argument(
        "--connections", type=int, default=8, help="Connections per large archive (1-64); files stay sequential"
    )
    parser.add_argument(
        "--mirror",
        choices=("original", "elda"),
        default="original",
        help="Official OpenSLR mirror; other publishers unchanged",
    )
    args = parser.parse_args(argv)
    if not 1 <= args.connections <= 64:
        parser.error("--connections must be between 1 and 64")
    config = load_config(args.config)["datasets"]
    selected = args.dataset or [name for name, value in config.items() if value.get("enabled")]
    plan = []
    matched_assets = set()
    for name in selected:
        if name not in config:
            parser.error(f"Unknown dataset: {name}")
        source = config[name]
        if source.get("restricted") and not args.allow_noncommercial:
            parser.error(f"{name} requires --allow-noncommercial")
        assets = [
            select_mirror(asset, args.mirror)
            for asset in source.get("assets", [])
            if (asset["filename"] in args.assets if args.assets else not asset.get("optional"))
        ]
        matched_assets.update(asset["filename"] for asset in assets)
        if not source.get("assets"):
            parser.error(f"{name} is listed for research only; no pinned download adapter is enabled")
        for asset in assets:
            if args.require_publisher_checksum and not (asset.get("sha256") or asset.get("md5")):
                parser.error(f"No publisher checksum is configured for {asset['filename']}")
        plan.append(
            {
                "dataset": name,
                "revision": source["revision"],
                "assets": assets,
                "download_directory": str((args.root / name / "downloads").resolve()),
                "extraction_directory": str((args.root / name / "extracted").resolve()),
                "integrity_note": source.get("integrity_note"),
            }
        )
    if args.assets and set(args.assets) - matched_assets:
        parser.error(f"Unknown/unselected assets: {sorted(set(args.assets) - matched_assets)}")
    print(
        json.dumps(
            {"execute": args.execute, "connections": args.connections, "mirror": args.mirror, "plan": plan}, indent=2
        )
    )
    if not args.execute:
        return
    if args.max_download_bytes is None or args.max_download_bytes <= 0:
        parser.error("--execute requires a positive --max-download-bytes")
    if args.max_extracted_bytes <= 0:
        parser.error("--max-extracted-bytes must be positive")
    transferred = 0
    extracted = 0
    with Progress(
        "Download assets", total=sum(len(source["assets"]) for source in plan), unit="files", mode=args.progress
    ) as indicator:
        for source in plan:
            directory = Path(source["download_directory"])
            for asset in source["assets"]:
                transferred += download_asset(
                    asset,
                    directory,
                    args.max_download_bytes - transferred,
                    progress=args.progress,
                    connections=args.connections,
                )
                indicator.update()
            if args.extract:
                extracted += extract_assets(
                    source["assets"],
                    directory,
                    Path(source["extraction_directory"]),
                    args.max_extracted_bytes - extracted,
                    source["dataset"],
                    progress=args.progress,
                )
    print(json.dumps({"network_bytes": transferred, "extracted_bytes": extracted}))


if __name__ == "__main__":
    main()

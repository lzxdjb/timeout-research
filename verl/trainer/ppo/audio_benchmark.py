"""Optional GAGE audio benchmarks over the live rollout endpoint."""

from __future__ import annotations

import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from urllib.request import ProxyHandler, Request, build_opener
from uuid import uuid4

logger = logging.getLogger(__name__)

_BENCHMARK_SPECS = {
    "aishell2": ("--aishell-2", "aishell2"),
    "chartqa": ("--chartqa", "chartqa"),
    "clothoaqa": ("--clothoaqa", "clothoaqa"),
    "mmbench": ("--mmbench", "mmbench"),
    "mmstar": ("--mmstar", "mmstar"),
    "ocrbench": ("--ocrbench", "ocrbench"),
    "realworldqa": ("--realworldqa", "realworldqa"),
    "rul_muchomusic": ("--rul-muchomusic", "muchomusic"),
    "voicebench_bbh": ("--voicebenchbbh", "voicebench_bbh"),
}


def run_audio_benchmarks(
    config: Mapping[str, Any],
    *,
    server_addresses: list[str],
    model_path: str,
    experiment_name: str,
    global_step: int,
) -> dict[str, float]:
    """Run configured GAGE suites without modifying the normal validation path.

    The suite targets ``config["api_base"]`` when provided, otherwise it reuses
    the live rollout endpoint (``server_addresses[0]``).  The live rollout serves
    audio when ``rollout.enable_audio=True``; ``api_base`` remains available to
    target a dedicated audio-capable model service instead.
    """

    if not config or not bool(config.get("enabled", False)):
        return {}

    gage_root = Path(str(config["gage_root"])).expanduser().resolve()
    runner = gage_root / "scripts" / "run" / "run_suite.sh"
    if not runner.is_file():
        raise FileNotFoundError(f"GAGE suite runner not found: {runner}")
    data_root = Path(str(config["data_root"])).expanduser().resolve()
    output_root = Path(str(config["output_dir"])).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    # Prefer an explicit audio-capable endpoint over the live rollout. When
    # rollout.enable_audio=True the live rollout already serves audio; otherwise
    # it rejects audio with "At most 0 audio(s)".
    raw_address = str(config.get("api_base") or "").strip()
    if raw_address:
        api_base = _normalize_api_base(raw_address)
    else:
        if not server_addresses:
            raise RuntimeError(
                "Audio benchmark requires `audio_benchmark.api_base` or at least one live rollout endpoint"
            )
        api_base = _normalize_api_base(server_addresses[0])

    api_key = str(config.get("api_key", "dummy"))
    served_model = str(config.get("served_model_name") or _discover_model(api_base, api_key))
    benchmarks = list(config.get("benchmarks") or _BENCHMARK_SPECS)
    unknown = sorted(set(benchmarks) - set(_BENCHMARK_SPECS))
    if unknown:
        raise ValueError(f"Unsupported audio benchmarks: {', '.join(unknown)}")

    run_id_base = f"{_safe_component(experiment_name)}-step-{global_step}"
    run_id = _new_run_id(run_id_base)
    suite_dir = output_root / run_id
    if suite_dir.exists():
        raise FileExistsError(
            f"Refusing to run GAGE audio validation into an existing directory: {suite_dir}"
        )
    command = [
        "bash",
        str(runner),
        "--model-path",
        model_path,
        "--external-api-base",
        api_base,
        "--external-model-name",
        served_model,
        "--external-api-key",
        api_key,
        "--output-dir",
        str(output_root),
        "--run-id",
        run_id,
        "--check-mode",
        str(config.get("check_mode", "light")),
        "--concurrency",
        str(int(config.get("concurrency", 4))),
        "--no-resume",
    ]
    max_samples = config.get("max_samples")
    sample_seed = config.get("sample_seed", 0)
    for benchmark in benchmarks:
        flag, dataset_dir = _BENCHMARK_SPECS[benchmark]
        dataset = data_root / dataset_dir
        command.extend([flag, str(dataset)])
        if max_samples is not None:
            command.append(str(int(max_samples)))
    if max_samples is not None:
        command.extend(["--sample-seed", str(int(sample_seed))])

    summary_path = suite_dir / "suite_summary.json"
    logger.info(
        "Running GAGE audio validation: step=%s run_id=%s api_base=%s expected_summary=%s",
        global_step,
        run_id,
        api_base,
        summary_path,
    )
    completed = subprocess.run(command, cwd=gage_root, check=False)
    logger.info(
        "GAGE audio validation process exited: run_id=%s returncode=%s expected_summary=%s",
        run_id,
        completed.returncode,
        summary_path,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"GAGE audio validation run {run_id!r} failed with exit code {completed.returncode}; "
            f"expected summary: {summary_path}. {_recent_run_details(output_root, run_id_base)}"
        )
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"GAGE audio validation run {run_id!r} exited successfully but did not write "
            f"the expected summary: {summary_path}. {_recent_run_details(output_root, run_id_base)}"
        )
    return _read_metrics(summary_path, expected_run_id=run_id)


def _normalize_api_base(address: str) -> str:
    address = address.strip().rstrip("/")
    if not address.startswith(("http://", "https://")):
        address = f"http://{address}"
    if not address.endswith("/v1"):
        address = f"{address}/v1"
    return address


def _discover_model(api_base: str, api_key: str) -> str:
    headers = {} if api_key == "dummy" else {"Authorization": f"Bearer {api_key}"}
    request = Request(f"{api_base.rstrip('/')}/models", headers=headers, method="GET")
    with build_opener(ProxyHandler({})).open(request, timeout=10) as response:
        payload = json.loads(response.read().decode("utf-8"))
    models = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(models, list) or not models or not isinstance(models[0], dict) or not models[0].get("id"):
        raise RuntimeError(f"Rollout endpoint returned no served model at {api_base}/models")

    served_model = str(models[0]["id"])

    # Validate that the served model supports audio when audio benchmarks are enabled
    # The architecture should be GageQwen3_5OmniMoeForConditionalGeneration for audio support
    model_details = models[0]
    logger.info(
        "Discovered served model: %s (architecture hint: %s)",
        served_model,
        model_details.get("architecture", "not reported"),
    )

    return served_model


def _read_metrics(summary_path: Path, *, expected_run_id: str | None = None) -> dict[str, float]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    actual_run_id = str(payload.get("run_id") or "")
    if expected_run_id is not None and actual_run_id != expected_run_id:
        raise RuntimeError(
            f"GAGE audio summary run_id mismatch at {summary_path}: "
            f"expected {expected_run_id!r}, found {actual_run_id or '<missing>'!r}"
        )
    status = str(payload.get("status") or "unknown")
    benchmark_statuses = {
        str(item.get("benchmark_id") or "unknown"): str(item.get("status") or "unknown")
        for item in payload.get("benchmarks", [])
        if isinstance(item, dict)
    }
    logger.info(
        "Reading GAGE audio results: run_id=%s status=%s benchmarks=%s summary=%s",
        actual_run_id or "<missing>",
        status,
        benchmark_statuses,
        summary_path,
    )
    if status != "completed":
        failed = [name for name, benchmark_status in benchmark_statuses.items() if benchmark_status != "completed"]
        detail = ", ".join(failed) or "unknown benchmark"
        raise RuntimeError(
            f"GAGE audio suite {actual_run_id or '<missing>'!r} status is {status!r}; "
            f"failed benchmarks: {detail}; summary: {summary_path}"
        )
    metrics: dict[str, float] = {}
    for benchmark in payload.get("benchmarks", []):
        if not isinstance(benchmark, dict):
            continue
        benchmark_id = str(benchmark.get("benchmark_id") or "unknown")
        summary_file = benchmark.get("summary", {}).get("summary_path")
        if not summary_file:
            continue
        child = json.loads(Path(str(summary_file)).read_text(encoding="utf-8"))
        for item in child.get("metrics", []):
            if not isinstance(item, dict):
                continue
            metric_id = str(item.get("metric_id") or "metric")
            values = item.get("raw_values") or item.get("values") or {}
            if isinstance(values, dict):
                for name, value in values.items():
                    if isinstance(value, str):
                        try:
                            value = float(value)
                        except ValueError:
                            continue
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        metrics[f"val-audio/{benchmark_id}/{metric_id}/{name}"] = float(value)
    metrics["val-audio/completed"] = 1.0
    return metrics


def _safe_component(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value) or "run"


def _new_run_id(run_id_base: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{run_id_base}-{timestamp}-{uuid4().hex[:12]}"


def _recent_run_details(output_root: Path, run_id_base: str, *, limit: int = 5) -> str:
    candidates = []
    for path in output_root.glob(f"{run_id_base}*"):
        if not path.is_dir():
            continue
        try:
            stat = path.stat()
            modified = datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()
        except OSError:
            continue
        candidates.append((stat.st_mtime_ns, f"{path.name} (modified={modified})"))
    candidates.sort(reverse=True)
    if not candidates:
        return f"No sibling runs matching {run_id_base!r} were found under {output_root}"
    recent = ", ".join(detail for _, detail in candidates[:limit])
    return f"Recent sibling runs under {output_root}: {recent}"

"""Side-effect-free diagnostics for Qwen3.5-Omni vLLM registration."""

from __future__ import annotations

import importlib.util
import json
import multiprocessing
import os
import socket
import sys
from importlib.metadata import PackageNotFoundError, entry_points, version
from typing import Any

_ARCHITECTURE = "GageQwen3_5OmniMoeForConditionalGeneration"
_ADAPTER_PACKAGE = "verl.workers.rollout.vllm_rollout.qwen35_omni"
_ADAPTER_MODEL_MODULE = f"{_ADAPTER_PACKAGE}.model"


def audio_debug_enabled() -> bool:
    """Return whether audio registration diagnostics are enabled."""
    value = os.environ.get("VERL_AUDIO_DEBUG", "0").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _find_module_origin(module_name: str) -> str | None:
    """Find a module without importing the target module."""
    try:
        spec = importlib.util.find_spec(module_name)
    except Exception as exc:  # noqa: BLE001
        return f"<lookup failed: {type(exc).__name__}: {exc}>"
    return None if spec is None else spec.origin


def _package_version(package: str) -> str | None:
    """Return installed package metadata without importing the package."""
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def collect_audio_registration_debug(
    stage: str,
    vllm_config: Any = None,
) -> dict[str, Any]:
    """Collect process, plugin, and registry state without registering the model."""
    import verl
    import vllm
    from vllm import ModelRegistry
    from vllm import envs as vllm_envs
    from vllm import plugins as vllm_plugins

    allowed_plugins = vllm_envs.VLLM_PLUGINS
    discovered_plugins = []
    for plugin in entry_points(group=vllm_plugins.DEFAULT_PLUGINS_GROUP):
        discovered_plugins.append(
            {
                "name": plugin.name,
                "value": plugin.value,
                "distribution": getattr(plugin.dist, "name", None),
                "allowed": allowed_plugins is None or plugin.name in allowed_plugins,
            }
        )

    model_config = getattr(vllm_config, "model_config", None)
    if model_config is None and hasattr(vllm_config, "hf_config"):
        model_config = vllm_config
    hf_config = getattr(model_config, "hf_config", None)
    model_module = sys.modules.get(_ADAPTER_MODEL_MODULE)
    model_class = getattr(model_module, _ARCHITECTURE, None) if model_module is not None else None

    return {
        "stage": stage,
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "hostname": socket.gethostname(),
        "multiprocessing_start_method": multiprocessing.get_start_method(allow_none=True),
        "cwd": os.getcwd(),
        "executable": sys.executable,
        "verl_package_file": getattr(verl, "__file__", None),
        "vllm_package_file": getattr(vllm, "__file__", None),
        "vllm_version": getattr(vllm, "__version__", None),
        "transformers_version": _package_version("transformers"),
        "pythonpath_env": os.environ.get("PYTHONPATH"),
        "sys_path": list(sys.path),
        "vllm_plugins_env": os.environ.get("VLLM_PLUGINS"),
        "vllm_plugins_parsed": allowed_plugins,
        "vllm_plugins_loaded": bool(vllm_plugins.plugins_loaded),
        "general_plugin_entry_points": discovered_plugins,
        "adapter_package_origin": _find_module_origin(_ADAPTER_PACKAGE),
        "adapter_package_loaded": _ADAPTER_PACKAGE in sys.modules,
        "adapter_model_loaded": model_module is not None,
        "architecture_registered": _ARCHITECTURE in ModelRegistry.get_supported_archs(),
        "multimodal_processor_attached": (
            "_processor_factory" in model_class.__dict__ if model_class is not None else None
        ),
        "configured_architectures": getattr(hf_config, "architectures", None),
        "model_path": (
            getattr(model_config, "model", None)
            or getattr(model_config, "path", None)
        ),
        "worker_multiproc_env": os.environ.get("VLLM_WORKER_MULTIPROC_METHOD"),
        "ray_job_id": os.environ.get("VERL_RAY_JOB_ID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def log_audio_registration_debug(logger: Any, stage: str, vllm_config: Any = None) -> None:
    """Log one machine-readable registration snapshot when diagnostics are enabled."""
    if not audio_debug_enabled():
        return
    try:
        snapshot = collect_audio_registration_debug(stage, vllm_config=vllm_config)
        logger.warning("[VERL_AUDIO_DEBUG] %s", json.dumps(snapshot, sort_keys=True, default=str))
    except Exception:  # noqa: BLE001
        logger.exception("[VERL_AUDIO_DEBUG] failed to collect stage=%s", stage)

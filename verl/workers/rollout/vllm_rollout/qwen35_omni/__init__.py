"""Register the hybrid audio architecture inside verl's vLLM rollout.

This is a minimal, self-contained port of the ``gage_vllm_qwen35_omni`` plugin.
It registers the ``GageQwen3_5OmniMoeForConditionalGeneration`` architecture
(resolved lazily to ``qwen35_omni.model``) so vLLM can dispatch the Qwen3.5-Omni
audio head when ``rollout.enable_audio`` is enabled.

Unlike the upstream plugin, this port installs no process-wide monkey patches
(auto config routing or the vLLM 0.23 shutdown guard). The verl rollout sets
``hf_overrides.architectures`` explicitly and manages engine teardown itself.
"""

from __future__ import annotations

ARCHITECTURE = "GageQwen3_5OmniMoeForConditionalGeneration"
PLUGIN_NAME = "verl_qwen35_omni"


def register() -> None:
    """Register the lazy model entry in each vLLM process."""
    import logging
    from vllm import ModelRegistry

    logger = logging.getLogger(__name__)

    if ARCHITECTURE not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            ARCHITECTURE,
            "verl.workers.rollout.vllm_rollout.qwen35_omni.model:GageQwen3_5OmniMoeForConditionalGeneration",
        )
        logger.debug(
            "Registered %s in vLLM ModelRegistry (PID=%d)",
            ARCHITECTURE,
            __import__("os").getpid(),
        )
    else:
        logger.debug(
            "Architecture %s already registered in vLLM ModelRegistry (PID=%d)",
            ARCHITECTURE,
            __import__("os").getpid(),
        )


def validate_plugin_installation() -> None:
    """Verify that vLLM can discover this adapter in every spawned process.

    vLLM's ``VLLM_PLUGINS`` value is an entry-point allowlist, not a Python
    module path. Fail before engine creation when the source checkout has not
    been installed with its entry-point metadata.
    """
    from importlib.metadata import entry_points
    import os

    discovered = {plugin.name for plugin in entry_points(group="vllm.general_plugins")}
    if PLUGIN_NAME not in discovered:
        raise RuntimeError(
            f"The {PLUGIN_NAME!r} vLLM plugin entry point is not installed. "
            "Install this verl checkout into the vLLM environment with "
            "`python -m pip install --no-deps -e .` before enabling audio rollout."
        )

    configured = os.environ.get("VLLM_PLUGINS")
    if configured is not None and configured.strip() and PLUGIN_NAME not in configured.split(","):
        raise RuntimeError(
            f"VLLM_PLUGINS={configured!r} excludes the required {PLUGIN_NAME!r} entry point. "
            f"Set VLLM_PLUGINS={PLUGIN_NAME!r} or include it in the comma-separated allowlist."
        )


# Auto-register when imported as a vLLM plugin (via VLLM_PLUGINS environment variable).
# This ensures worker processes spawned by vLLM's multiproc_executor inherit the registration.
register()

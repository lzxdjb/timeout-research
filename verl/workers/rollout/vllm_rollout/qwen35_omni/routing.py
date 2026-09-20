"""Validate hybrid audio configs and gate the supported vLLM runtime."""

from __future__ import annotations

from collections.abc import Mapping
from importlib.metadata import version
from typing import Any

from . import ARCHITECTURE


def is_hybrid_config(config: Mapping[str, Any]) -> bool:
    """Identify Qwen3.5 MoE checkpoints carrying an Omni audio encoder."""
    return (
        config.get("model_type") == "qwen3_5_moe"
        and bool(config.get("audio_config"))
        and any(
            name in {"Qwen3_5MoeForConditionalGeneration", ARCHITECTURE}
            for name in config.get("architectures", [])
        )
    )


def validate_hybrid_config(config: Mapping[str, Any]) -> None:
    """Reject incompatible audio towers before allocating model memory."""
    if not is_hybrid_config(config):
        raise ValueError("Expected a Qwen3.5 MoE checkpoint with a nonempty audio_config.")
    audio = config["audio_config"]
    if hasattr(audio, "to_dict"):
        audio = audio.to_dict()
    if not isinstance(audio, Mapping):
        raise ValueError("audio_config must be a mapping or a HuggingFace config.")
    if audio.get("model_type") != "qwen3_omni_moe_audio_encoder":
        raise ValueError("Only qwen3_omni_moe_audio_encoder is supported.")
    text = config.get("text_config", {})
    if hasattr(text, "to_dict"):
        text = text.to_dict()
    if audio.get("output_dim") != text.get("hidden_size"):
        raise ValueError("Audio output_dim must equal the text hidden_size.")
    if audio.get("n_window", 50) != 50:
        raise ValueError("This adapter requires the reference audio n_window=50.")
    if not isinstance(config.get("audio_token_id"), int):
        raise ValueError("The checkpoint must define audio_token_id.")


def check_runtime_version() -> None:
    """Reject unvalidated vLLM versions when the hybrid route is selected."""
    if version("vllm").split("+")[0] != "0.24.0":
        raise RuntimeError(
            "This adapter is validated for vLLM 0.24.0; do not replace upstream packages."
        )

"""Compose native vLLM 0.24 Qwen3.5 and Qwen3-Omni components."""

from __future__ import annotations

from typing import Any

import torch
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeAudioEncoderConfig,
)
from vllm.config import VllmConfig
from vllm.model_executor.models.interfaces import MultiModalEmbeddings
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.models.qwen2_5_omni_thinker import (
    Qwen2_5OmniConditionalGenerationMixin,
)
from vllm.model_executor.models.qwen3_5 import Qwen3_5MoeForConditionalGeneration
from vllm.model_executor.models.qwen3_omni_moe_thinker import (
    Qwen3OmniMoeAudioEncoder,
    Qwen3OmniMoeConditionalGenerationMixin,
    Qwen3OmniMoeThinkerForConditionalGeneration,
)
from vllm.model_executor.models.utils import WeightsMapper, maybe_prefix
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFeatureSpec

from .processor import (
    HybridDummyInputsBuilder,
    HybridMultiModalProcessor,
    HybridProcessingInfo,
)
from .routing import check_runtime_version, validate_hybrid_config


@MULTIMODAL_REGISTRY.register_processor(
    HybridMultiModalProcessor,
    info=HybridProcessingInfo,
    dummy_inputs=HybridDummyInputsBuilder,
)
class GageQwen3_5OmniMoeForConditionalGeneration(Qwen3_5MoeForConditionalGeneration):
    """Add an Omni audio tower without altering either upstream model class."""

    hf_to_vllm_mapper = Qwen3_5MoeForConditionalGeneration.hf_to_vllm_mapper | WeightsMapper(
        orig_to_new_prefix={"model.audio_tower.": "audio_tower."}
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model") -> None:
        # STEP 1: Validate the checkpoint before creating the unchanged visual/text modules.
        check_runtime_version()
        config = vllm_config.model_config.hf_config
        validate_hybrid_config(config.to_dict())
        compilation = vllm_config.compilation_config
        if compilation.compile_mm_encoder or compilation.cudagraph_mm_encoder:
            raise ValueError("Hybrid modality tags require eager media encoders; leave compile_mm_encoder and cudagraph_mm_encoder disabled.")
        if vllm_config.model_config.multimodal_config.video_pruning_rate:
            raise ValueError("Hybrid audio/video does not support video pruning.")
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        # STEP 2: Add only this instance's audio tower; never mutate HF config classes.
        audio_config = config.audio_config
        if isinstance(audio_config, dict):
            audio_config = Qwen3OmniMoeAudioEncoderConfig(**audio_config)
        with self._mark_tower_model(vllm_config, "audio"):
            self.audio_tower = Qwen3OmniMoeAudioEncoder(
                audio_config, prefix=maybe_prefix(prefix, "audio_tower")
            )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        """Return the checkpoint's audio markers or upstream visual markers."""
        if modality.startswith("audio"):
            return "<|audio_start|><|audio_pad|><|audio_end|>"
        return super().get_placeholder_str(modality, i)

    def _parse_and_validate_multimodal_inputs(self, **kwargs: object) -> dict:
        return Qwen3OmniMoeThinkerForConditionalGeneration._parse_and_validate_multimodal_inputs(self, **kwargs)

    def _parse_and_validate_audio_input(self, **kwargs: object) -> Any:
        return Qwen2_5OmniConditionalGenerationMixin._parse_and_validate_audio_input(self, **kwargs)

    def _process_audio_input(self, audio_input: Any) -> tuple[torch.Tensor, ...]:
        return Qwen3OmniMoeConditionalGenerationMixin._process_audio_input(self, audio_input)

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings | None:
        """Encode media and carry a modality tag through cache and prefill slicing."""
        result = []
        for modality, value in self._parse_and_validate_multimodal_inputs(**kwargs).items():
            encode = getattr(self, f"_process_{modality}_input")
            tag = {"image": 1, "video": 2, "audio": 3}[modality]
            for embedding in encode(value):
                result.append(torch.cat((embedding, embedding.new_full((len(embedding), 1), tag)), dim=-1))
        return tuple(result)

    def embed_input_ids(
        self, input_ids: torch.Tensor, multimodal_embeddings: MultiModalEmbeddings | None = None,
        *, is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Scatter by explicit modality, including mixed requests and chunked prefill."""
        result = self._embed_text_input_ids(
            input_ids, self.language_model.embed_input_ids, is_multimodal=is_multimodal,
        )
        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return result
        if is_multimodal is None:
            raise ValueError("A multimodal embedding mask is required.")
        # The V1 runner intentionally passes its scheduling mask on the CPU.
        is_multimodal = is_multimodal.to(device=input_ids.device, non_blocking=True)
        tagged = torch.cat(tuple(multimodal_embeddings), dim=0)
        for tag, token_id in [(1, self.config.image_token_id), (2, self.config.video_token_id), (3, self.config.audio_token_id)]:
            target = (input_ids == token_id) & is_multimodal
            result[target] = tagged[tagged[:, -1] == tag, :-1].to(result.dtype)
        return result

    def get_mrope_input_positions(
        self, input_tokens: list[int], mm_features: list[MultiModalFeatureSpec]
    ) -> tuple[torch.Tensor, int]:
        """Use Qwen3.5 visual positions and text-style positions for audio."""
        visual_features = [feature for feature in mm_features if feature.modality in {"image", "video"}]
        # Native Qwen3.5 treats every gap between visual features as consecutive 1-D positions.
        return super().get_mrope_input_positions(input_tokens, visual_features)

    def get_mm_mapping(self) -> MultiModelKeys:
        """Include the additional tower in vLLM's multimodal module mapping."""
        return MultiModelKeys.from_string_field(
            language_model="language_model",
            connector="visual.merger",
            tower_model=["visual.", "audio_tower."],
        )

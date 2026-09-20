"""Add audio to the installed Qwen3-VL processor without global patches."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from transformers import Qwen3VLProcessor, WhisperFeatureExtractor
from transformers.feature_extraction_utils import BatchFeature
from transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe import (
    _get_feat_extract_output_lengths,
)
from vllm.model_executor.models.qwen2_vl import Qwen2VLMultiModalDataParser
from vllm.model_executor.models.qwen3_5 import Qwen3_5MoeProcessingInfo
from vllm.model_executor.models.qwen3_vl import (
    Qwen3VLDummyInputsBuilder,
    Qwen3VLMultiModalProcessor,
)
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import MultiModalDataItems
from vllm.multimodal.processing import PromptReplacement, PromptUpdate
from vllm.multimodal.processing import PromptUpdateDetails

from .interleaved import AUDIO, VIDEO, interleave_text, pair_prompt, requested, update_tokenized, without_mode


class HybridProcessor(Qwen3VLProcessor):
    """Preserve Qwen3.5 vision processing and add Omni audio features."""

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs: Any) -> HybridProcessor:
        """Load the unchanged vision processor and checkpoint audio extractor."""
        processor = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        download_kwargs = {
            key: kwargs[key]
            for key in ("local_files_only", "revision", "cache_dir", "token")
            if key in kwargs
        }
        processor.feature_extractor = WhisperFeatureExtractor.from_pretrained(
            pretrained_model_name_or_path, **download_kwargs
        )
        processor.audio_token = "<|audio_pad|>"
        processor.audio_token_id = processor.tokenizer.convert_tokens_to_ids(processor.audio_token)
        return processor

    def __call__(
        self,
        text: str | list[str] | None = None,
        images: Any = None,
        videos: Any = None,
        audio: Any = None,
        **kwargs: Any,
    ) -> BatchFeature:
        """Expand audio tokens and delegate image/video processing to upstream."""
        if requested(kwargs):
            if videos is None or audio is None:
                raise ValueError("use_audio_in_video requires paired videos and audio waveforms.")
            texts = [text] if isinstance(text, str) else list(text or [])
            if len(texts) != 1:
                raise ValueError("Interleaved preprocessing accepts one conversation at a time.")
            count = texts[0].count(VIDEO)
            if count == 0:
                raise ValueError("use_audio_in_video requires video placeholders.")
            output = self(
                text=[pair_prompt(texts[0], count)], images=images, videos=videos,
                audio=audio, **without_mode(kwargs),
            )
            update_tokenized(output, self, count)
            return output
        if audio is None:
            return super().__call__(text=text, images=images, videos=videos, **kwargs)

        # STEP 1: Extract features using the checkpoint's sample rate and valid-frame mask.
        audio_kwargs = {
            "sampling_rate": self.feature_extractor.sampling_rate,
            "padding": True,
            "truncation": False,
            "return_attention_mask": True,
            "return_tensors": "pt",
        }
        audio_kwargs.update(kwargs.pop("audio_kwargs", {}) or {})
        if audio_kwargs.get("return_attention_mask") is not True:
            raise ValueError("Audio feature_attention_mask is required.")
        audio_kwargs["return_tensors"] = "pt"
        features = self.feature_extractor(audio, **audio_kwargs)
        features["feature_attention_mask"] = features.pop("attention_mask")
        lengths = _get_feat_extract_output_lengths(features["feature_attention_mask"].sum(-1)).tolist()

        # STEP 2: Expand each placeholder once before normal vision tokenization.
        texts = [text] if isinstance(text, str) else list(text or [])
        if sum(item.count(self.audio_token) for item in texts) != len(lengths):
            raise ValueError("The number of audio placeholders must equal the number of audio items.")
        index = 0
        expanded = []
        for item in texts:
            parts = item.split(self.audio_token)
            merged = parts[0]
            for suffix in parts[1:]:
                length = int(lengths[index])
                if length < 1:
                    raise ValueError("Audio is too short to produce encoder tokens.")
                merged += self.audio_token * length + suffix
                index += 1
            expanded.append(merged)
        result = super().__call__(text=expanded, images=images, videos=videos, **kwargs)
        result.update(features)
        return result

    @property
    def model_input_names(self) -> list[str]:
        """Include the audio fields alongside upstream processor inputs."""
        return list(dict.fromkeys([*super().model_input_names, "input_features", "feature_attention_mask"]))


class HybridProcessingInfo(Qwen3_5MoeProcessingInfo):
    """Expose audio while retaining Qwen3.5 image/video sizing and metadata."""

    def get_hf_processor(self, **kwargs: object) -> HybridProcessor:
        """Load the isolated composite processor."""
        processor = self.ctx.get_hf_processor(HybridProcessor, use_fast=kwargs.pop("use_fast", True), **kwargs)
        config = self.get_hf_config()
        if processor.audio_token_id != config.audio_token_id:
            raise ValueError("The tokenizer audio token ID does not match the model config.")
        audio = config.audio_config
        mel_bins = audio.get("num_mel_bins") if isinstance(audio, dict) else audio.num_mel_bins
        if processor.feature_extractor.feature_size != mel_bins:
            raise ValueError("Audio feature size does not match the encoder's num_mel_bins.")
        return processor

    def get_feature_extractor(self, **kwargs: object) -> WhisperFeatureExtractor:
        """Return the checkpoint's audio feature extractor."""
        return self.get_hf_processor(**kwargs).feature_extractor

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        """Advertise independent image, video and audio inputs."""
        return {"image": None, "video": None, "audio": None}

    def get_data_parser(self) -> Qwen2VLMultiModalDataParser:
        """Retain video metadata and normalize audio to mono at the target rate."""
        return Qwen2VLMultiModalDataParser(
            self.get_hf_config().vision_config.spatial_merge_size,
            video_needs_metadata=True,
            target_sr=self.get_feature_extractor().sampling_rate,
            target_channels=1,
            expected_hidden_size=self._get_expected_hidden_size(),
        )

    def get_mm_max_tokens_per_item(
        self, seq_len: int, mm_counts: Mapping[str, int]
    ) -> Mapping[str, int]:
        """Add the Omni audio token bound to upstream vision bounds."""
        bounds = dict(super().get_mm_max_tokens_per_item(seq_len, mm_counts))
        extractor = self.get_feature_extractor()
        frames = int(min(extractor.chunk_length, 30) * extractor.sampling_rate // extractor.hop_length)
        bounds["audio"] = int(_get_feat_extract_output_lengths(torch.tensor(frames)).item())
        return bounds


class HybridDummyInputsBuilder(Qwen3VLDummyInputsBuilder):
    """Extend upstream visual profiling inputs with audio waveforms."""

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        """Construct placeholders for each requested modality."""
        audio = "<|audio_start|><|audio_pad|><|audio_end|>"
        return super().get_dummy_text(mm_counts) + audio * mm_counts.get("audio", 0)

    def get_dummy_mm_data(self, seq_len: int, mm_counts: Mapping[str, int], mm_options: Mapping[str, Any]) -> dict:
        """Combine upstream image/video samples with bounded audio samples."""
        data = super().get_dummy_mm_data(seq_len, mm_counts, mm_options)
        extractor = self.info.get_feature_extractor()
        data["audio"] = self._get_dummy_audios(
            length=int(min(extractor.chunk_length, 30) * extractor.sampling_rate),
            num_audios=mm_counts.get("audio", 0),
            overrides=mm_options.get("audio"),
        )
        return data


class HybridMultiModalProcessor(Qwen3VLMultiModalProcessor):
    """Preserve upstream visual processing while adding audio field contracts."""

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        interleaved = requested(mm_kwargs)
        if interleaved:
            videos = mm_data.get("videos", [])
            audios = mm_data.get("audios", [])
            if not videos or len(videos) != len(audios):
                raise ValueError("use_audio_in_video requires equal numbers of paired videos and audio waveforms.")
            if self.info.ctx.get_mm_config().video_pruning_rate:
                raise ValueError("Video pruning is not supported for hybrid interleaved audio/video.")
            prompt = pair_prompt(prompt, len(videos))
            mm_kwargs = without_mode(mm_kwargs)
        # STEP 1: Pad audio to a hop boundary so cached items have stable lengths.
        data = dict(mm_data)
        audios = data.pop("audios", [])
        if audios:
            hop = self.info.get_feature_extractor().hop_length
            data["audio"] = [
                np.pad(np.asarray(item), (0, (-len(item)) % hop))
                for item in audios
            ]
        output = super()._call_hf_processor(prompt, data, mm_kwargs, tok_kwargs)
        if interleaved:
            update_tokenized(output, self.info.get_hf_processor(), len(videos))
        output["use_audio_in_video"] = torch.tensor(interleaved)

        # STEP 2: Flatten only valid frames, matching the native Omni encoder contract.
        if "input_features" in output:
            features = output.pop("input_features")
            mask = output["feature_attention_mask"]
            output["input_audio_features"] = features.permute(0, 2, 1)[mask.bool()].permute(1, 0)
            output["audio_feature_lengths"] = mask.sum(-1)
        return output

    def _get_mm_fields_config(
        self, hf_inputs: BatchFeature, hf_processor_mm_kwargs: Mapping[str, object]
    ) -> Mapping[str, MultiModalFieldConfig]:
        fields = dict(super()._get_mm_fields_config(hf_inputs, hf_processor_mm_kwargs))
        lengths = hf_inputs.get("audio_feature_lengths", torch.empty(0, dtype=torch.long))
        fields.update(
            input_audio_features=MultiModalFieldConfig.flat_from_sizes("audio", lengths, dim=1),
            audio_feature_lengths=MultiModalFieldConfig.batched("audio"),
            feature_attention_mask=MultiModalFieldConfig.batched("audio"),
            use_audio_in_video=MultiModalFieldConfig.shared("video", len(hf_inputs.get("video_grid_thw", []))),
        )
        return fields

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        updates = list(super()._get_prompt_updates(mm_items, hf_processor_mm_kwargs, out_mm_kwargs))
        processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)

        def replacement(index: int) -> list[int]:
            frames = out_mm_kwargs["audio"][index]["audio_feature_lengths"].data
            count = int(_get_feat_extract_output_lengths(frames).item())
            if count < 1:
                raise ValueError("Audio is too short to produce encoder tokens.")
            return [processor.audio_token_id] * count

        updates.append(PromptReplacement(modality="audio", target=processor.audio_token, replacement=replacement))
        if requested(hf_processor_mm_kwargs):
            video_update = next(update for update in updates if update.modality == "video")

            def paired_replacement(index: int) -> PromptUpdateDetails:
                regular = video_update.replacement(index)
                video_text = processor.tokenizer.decode(regular.full, skip_special_tokens=False)
                audio_text = "<|audio_start|>" + processor.audio_token * len(replacement(index)) + "<|audio_end|>"
                paired = interleave_text(video_text + audio_text, 1)
                ids = processor.tokenizer.encode(paired, add_special_tokens=False)
                return PromptUpdateDetails.select_token_id(ids, processor.video_token_id)

            updates = [update for update in updates if update.modality != "video"]
            updates.append(PromptReplacement(modality="video", target=VIDEO, replacement=paired_replacement))
        return updates

    def _cached_apply_hf_processor(self, inputs: Any, timing_ctx: Any) -> Any:
        # Paired prompt replacements depend on both modalities; partial cache hits
        # must not reconstruct a video using audio from a different request.
        if requested(inputs.hf_processor_mm_kwargs):
            return self._apply_hf_processor(inputs, timing_ctx)
        return super()._cached_apply_hf_processor(inputs, timing_ctx)

    def _maybe_apply_prompt_updates(
        self, mm_items: Any, prompt_ids: list[int], mm_kwargs: Any,
        mm_prompt_updates: Any, is_update_applied: bool,
    ) -> Any:
        paired = any(
            item and item.get("use_audio_in_video") and bool(item["use_audio_in_video"].data)
            for item in mm_kwargs.get("video", [])
        )
        if not paired:
            return super()._maybe_apply_prompt_updates(
                mm_items, prompt_ids, mm_kwargs, mm_prompt_updates, is_update_applied,
            )
        from vllm.model_executor.models.qwen3_omni_moe_thinker import Qwen3OmniMoeThinkerMultiModalProcessor

        counts = mm_items.get_all_counts()
        self._validate_mm_kwargs(mm_kwargs, counts)
        updates = {key: value for key, value in mm_prompt_updates.items() if key != "audio"}
        if is_update_applied:
            placeholders = self._find_mm_placeholders(prompt_ids, updates)
        else:
            # The paired video replacement supplies audio; remove standalone markers.
            tokenizer = self.info.get_tokenizer()
            prompt = tokenizer.decode(prompt_ids, skip_special_tokens=False).replace(AUDIO, "")
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            prompt_ids, placeholders = self._apply_prompt_updates(prompt_ids, updates)
        placeholders = Qwen3OmniMoeThinkerMultiModalProcessor._derive_audio_from_video_placeholders(
            self, placeholders, mm_prompt_updates,
        )
        self._validate_mm_placeholders(placeholders, counts)
        return prompt_ids, placeholders

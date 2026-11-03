"""Explicit audio preprocessing for Qwen3.5 training, independent of vLLM."""

import json
import math
import os
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from transformers import Qwen3VLProcessor, WhisperFeatureExtractor
from transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe import _get_feat_extract_output_lengths


def load_audio_waveform(value, *, target_sampling_rate, sampling_rate=None):
    """Load local media or a rate-qualified array as finite mono float32 audio."""
    import soundfile as sf

    if isinstance(value, Mapping):
        sampling_rate = value.get("sampling_rate", value.get("sample_rate", sampling_rate))
        value = value.get("array", value.get("audio", value.get("path")))
    if isinstance(value, tuple) and len(value) == 2:
        value, sampling_rate = value
    if isinstance(value, (str, os.PathLike)):
        if "://" in os.fspath(value):
            raise ValueError("Training audio requires local media paths")
        waveform, sampling_rate = sf.read(value, dtype="float32", always_2d=True)
    else:
        waveform = np.asarray(value, dtype=np.float32)
    if waveform.ndim == 2:
        if waveform.shape[1] <= 8:
            waveform = waveform.mean(axis=1)
        elif waveform.shape[0] <= 8:
            waveform = waveform.mean(axis=0)
        else:
            raise ValueError("Audio channel layout is ambiguous")
    if waveform.ndim != 1 or not waveform.size or not np.isfinite(waveform).all():
        raise ValueError("Training audio requires a finite nonempty waveform")
    for rate in (sampling_rate, target_sampling_rate):
        if isinstance(rate, bool) or not isinstance(rate, (int, np.integer)) or rate <= 0:
            raise ValueError("Audio sampling rate is missing or invalid")
    if sampling_rate != target_sampling_rate:
        try:
            from scipy.signal import resample_poly
        except ImportError as exc:
            raise RuntimeError(
                "Audio resampling requires scipy; install it in this environment with uv pip install scipy"
            ) from exc

        divisor = math.gcd(int(sampling_rate), int(target_sampling_rate))
        waveform = resample_poly(waveform, target_sampling_rate // divisor, sampling_rate // divisor)
    return np.ascontiguousarray(waveform, dtype=np.float32)


def load_audio_feature_extractor(source, **kwargs):
    """Prefer native extractor config; support explicitly saved hybrid exports."""
    source = Path(source)
    if (source / "preprocessor_config.json").is_file() or not source.is_dir():
        return WhisperFeatureExtractor.from_pretrained(source, **kwargs)
    # Earlier hybrid exports stored extractor fields beside image settings.
    # Reconstruct only when every defining audio field is explicitly present.
    config_file = source / "processor_config.json"
    config = json.loads(config_file.read_text()) if config_file.is_file() else {}
    saved = config.get("feature_extractor") or config.get("image_processor") or {}
    required = {"feature_size", "sampling_rate", "hop_length", "n_fft", "chunk_length"}
    if not required.issubset(saved):
        raise ValueError("Checkpoint has no complete saved audio feature-extractor configuration")
    if any(type(saved[key]) is not int or saved[key] <= 0 for key in required):
        raise ValueError("Saved audio feature-extractor dimensions and rates must be positive integers")
    fields = required | {"padding_value", "dither", "return_attention_mask"}
    extractor = WhisperFeatureExtractor(**{key: saved[key] for key in fields if key in saved})
    hf_config = json.loads((source / "config.json").read_text())
    if extractor.feature_size != hf_config.get("audio_config", {}).get("num_mel_bins"):
        raise ValueError("Saved audio feature size does not match checkpoint audio_config.num_mel_bins")
    return extractor


class AudioQwen3VLProcessor(Qwen3VLProcessor):
    """Keep Qwen3-VL vision/text semantics and explicitly add standalone audio."""

    @classmethod
    def from_pretrained(cls, source, **kwargs):
        processor = super().from_pretrained(source, **kwargs)
        extractor_kwargs = {
            key: kwargs[key] for key in ("local_files_only", "revision", "cache_dir", "token") if key in kwargs
        }
        processor.feature_extractor = load_audio_feature_extractor(source, **extractor_kwargs)
        processor.audio_token = "<|audio_pad|>"
        processor.audio_token_id = processor.tokenizer.convert_tokens_to_ids(processor.audio_token)
        return processor

    def extract_audio_features(self, audio, *, sampling_rate=None):
        """Return padded mel features/masks without retokenizing an expanded prompt."""
        if isinstance(audio, (str, os.PathLike, Mapping, np.ndarray)) or (
            isinstance(audio, tuple) and len(audio) == 2 and isinstance(audio[1], (int, np.integer))
        ):
            audio = [audio]
        if not audio:
            raise ValueError("Audio feature extraction requires at least one clip")
        extractor = self.feature_extractor
        waveforms = []
        for item in audio:
            waveform = load_audio_waveform(
                item, target_sampling_rate=extractor.sampling_rate, sampling_rate=sampling_rate
            )
            # Match the rollout processor's hop-boundary padding exactly.
            waveforms.append(np.pad(waveform, (0, (-len(waveform)) % extractor.hop_length)))
        features = extractor(
            waveforms,
            sampling_rate=extractor.sampling_rate,
            padding=True,
            truncation=False,
            return_attention_mask=True,
            return_tensors="pt",
        )
        features["feature_attention_mask"] = features.pop("attention_mask")
        return features

    def __call__(self, text=None, images=None, videos=None, audio=None, **kwargs):
        if audio is None:
            return super().__call__(text=text, images=images, videos=videos, **kwargs)
        if kwargs.pop("use_audio_in_video", False):
            raise ValueError("Training audio processor currently supports standalone audio only")
        features = self.extract_audio_features(audio, sampling_rate=kwargs.pop("sampling_rate", None))
        texts = [text] if isinstance(text, str) else list(text or [])
        lengths = iter(_get_feat_extract_output_lengths(features["feature_attention_mask"].sum(-1)).tolist())
        if sum(item.count(self.audio_token) for item in texts) != features["input_features"].shape[0]:
            raise ValueError("Each audio clip requires exactly one unexpanded audio placeholder")
        expanded = []
        for item in texts:
            parts = item.split(self.audio_token)
            expanded.append(parts[0] + "".join(self.audio_token * int(next(lengths)) + part for part in parts[1:]))
        result = super().__call__(text=expanded, images=images, videos=videos, **kwargs)
        result.update(features)
        return result

    def save_pretrained(self, save_directory, **kwargs):
        result = super().save_pretrained(save_directory, **kwargs)
        self.feature_extractor.save_pretrained(save_directory)
        return result

    def to_dict(self, **kwargs):
        result = super().to_dict(**kwargs)
        # Preserve ordinary HF/vLLM discovery; audio remains an explicit opt-in.
        result["processor_class"] = "Qwen3VLProcessor"
        return result

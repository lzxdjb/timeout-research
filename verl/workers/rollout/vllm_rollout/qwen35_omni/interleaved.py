"""Pair video frames with audio while preserving Qwen3.5 position semantics."""

from __future__ import annotations

import re
from typing import Any

import torch
from transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe import _get_feat_extract_output_lengths

VIDEO = "<|vision_start|><|video_pad|><|vision_end|>"
AUDIO = "<|audio_start|><|audio_pad|><|audio_end|>"
FRAME = re.compile(r"<([0-9]+(?:\.[0-9]+)?) seconds><\|vision_start\|>(?:<\|video_pad\|>)+<\|vision_end\|>")
PAIRED = re.compile(
    r"(?:<\|vision_start\|>)?"
    r"((?:<[0-9]+(?:\.[0-9]+)? seconds><\|vision_start\|>(?:<\|video_pad\|>)+<\|vision_end\|>)+)"
    r"(?:<\|vision_end\|>)?"
    r"<\|audio_start\|>((?:<\|audio_pad\|>)+)<\|audio_end\|>"
)


def requested(kwargs: dict | Any) -> bool:
    """Return whether this request explicitly pairs video with its audio."""
    return bool(kwargs.get("use_audio_in_video") or (kwargs.get("videos_kwargs") or {}).get("use_audio_in_video"))


def without_mode(kwargs: Any) -> dict:
    """Copy request kwargs without forwarding hybrid flags to Qwen3-VL."""
    result = dict(kwargs)
    result.pop("use_audio_in_video", None)
    if "videos_kwargs" in result:
        result["videos_kwargs"] = dict(result["videos_kwargs"] or {})
        result["videos_kwargs"].pop("use_audio_in_video", None)
    return result


def pair_prompt(text: str, count: int) -> str:
    """Place exactly one temporary audio marker immediately after each video."""
    if text.count(VIDEO) != count or text.count("<|audio_pad|>") not in (0, count):
        raise ValueError("use_audio_in_video requires one paired audio per video and no independent audio items.")
    text = text.replace(AUDIO, "")
    if "<|audio_pad|>" in text:
        raise ValueError("Paired audio must use the checkpoint's audio_start/audio_end markers.")
    return text.replace(VIDEO, VIDEO + AUDIO)


def interleave_text(text: str, expected_pairs: int) -> str:
    """Interleave complete visual frames with cumulative encoder audio tokens.

    Audio keeps the checkpoint's one-dimensional positions. Frame timestamps and
    spatial grids remain Qwen3.5 style. This is an experimental hybrid layout,
    not a claim that the checkpoint was trained with native Omni temporal RoPE.
    """
    def replace(match: re.Match) -> str:
        frames = list(FRAME.finditer(match[1]))
        count = match[2].count("<|audio_pad|>")
        consumed = 0
        pieces = ["<|audio_start|>"]
        for frame in frames:
            # The encoder emits 13 tokens per 100 mel frames, with a partial tail.
            mel_frames = int(round(float(frame[1]) * 100))
            boundary = min(count, int(_get_feat_extract_output_lengths(mel_frames)))
            pieces.append("<|audio_pad|>" * max(0, boundary - consumed))
            consumed = max(consumed, boundary)
            pieces.append(frame[0])
        pieces.extend(["<|audio_pad|>" * (count - consumed), "<|audio_end|>"])
        return "".join(pieces)

    output, count = PAIRED.subn(replace, text)
    if count != expected_pairs:
        raise ValueError(f"Could not align paired video/audio markers: expected {expected_pairs}, got {count}.")
    return output


def update_tokenized(output: Any, processor: Any, pairs: int) -> None:
    """Retokenize the reordered prompt and preserve all extracted media tensors."""
    rows = output["input_ids"]
    if len(rows) != 1:
        raise ValueError("Interleaved preprocessing accepts one conversation at a time.")
    text = processor.tokenizer.decode(rows[0], skip_special_tokens=False)
    text = interleave_text(text, pairs)
    ids = processor.tokenizer.encode(text, add_special_tokens=False)
    output["input_ids"] = torch.tensor([ids], dtype=torch.long)
    output["attention_mask"] = torch.ones_like(output["input_ids"])
    if "mm_token_type_ids" in output:
        types = torch.zeros_like(output["input_ids"])
        for token, value in [(processor.image_token_id, 1), (processor.video_token_id, 2)]:
            types[output["input_ids"] == token] = value
        output["mm_token_type_ids"] = types

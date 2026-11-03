# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Native audio-conditioned sequence scoring, loaded only by the opt-in protocol.

The teacher tokenizes the candidate in its own vocabulary. Its scalar sequence
score is not represented as aligned student token probabilities.
"""

import base64
import json
import math
import os
from pathlib import Path
from typing import Any


def _encode_audio(value: Any, sampling_rate: int | None = None) -> dict[str, str]:
    """Encode the actual waveform as float32 WAV without timestamp-bearing chunks."""
    import numpy as np
    import soundfile as sf

    if isinstance(value, dict):
        sampling_rate = value.get("sampling_rate", value.get("sample_rate", sampling_rate))
        value = value.get("array", value.get("audio", value.get("path")))
    if isinstance(value, tuple) and len(value) == 2:
        value, sampling_rate = value
    if isinstance(value, (str, os.PathLike)):
        # Remote paths would change the trust/availability semantics of training inputs.
        if "://" in os.fspath(value):
            raise ValueError("Audio candidate scoring accepts local media paths only.")
        waveform, sampling_rate = sf.read(value, dtype="float32", always_2d=True)
        waveform = waveform.mean(axis=1)
    else:
        waveform = np.asarray(value, dtype=np.float32)
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=1)
    if waveform.ndim != 1 or waveform.size == 0 or not np.isfinite(waveform).all():
        raise ValueError("Audio candidate scoring requires a finite, nonempty mono waveform.")
    if not isinstance(sampling_rate, (int, np.integer)) or sampling_rate <= 0:
        raise ValueError("Audio waveform sampling rate is missing or invalid.")
    # scipy-free RIFF IEEE float WAV avoids libsndfile's optional PEAK timestamp.
    import struct

    raw = np.asarray(waveform, dtype="<f4").tobytes()
    fmt = struct.pack("<HHIIHH", 3, 1, int(sampling_rate), int(sampling_rate) * 4, 4, 32)
    wav = b"RIFF" + struct.pack("<I", 36 + len(raw)) + b"WAVEfmt " + struct.pack("<I", 16) + fmt
    wav += b"data" + struct.pack("<I", len(raw)) + raw
    return {"data": base64.b64encode(wav).decode("ascii"), "format": "wav"}


class AudioCandidateCodec:
    def __init__(self, config, tokenizer=None):
        self.config = config
        if tokenizer is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path, local_files_only=True)
            if tokenizer.chat_template is None:
                template_path = Path(config.tokenizer_path) / "chat_template.json"
                tokenizer.chat_template = json.loads(template_path.read_text(encoding="utf-8"))["chat_template"]
        self.tokenizer = tokenizer
        self.eos_token = tokenizer.eos_token
        self.eos_token_id = tokenizer.eos_token_id
        if not self.eos_token or not isinstance(self.eos_token_id, int):
            raise ValueError("Audio teacher tokenizer must define a native EOS token.")
        if tokenizer.encode(self.eos_token, add_special_tokens=False) != [self.eos_token_id]:
            raise ValueError("Native audio teacher EOS does not encode to one verified token.")

    def prepare(self, trajectory: dict, audio_data=None, mm_processor_kwargs=None) -> tuple[dict, dict]:
        if trajectory.get("terminated") is not True or trajectory.get("truncated") is not False:
            raise ValueError("Audio sequence scoring requires verified student EOS; truncated responses are rejected.")
        text = trajectory.get("response_text")
        if not isinstance(text, str) or not text:
            raise ValueError("Audio sequence scoring requires nonempty candidate text.")
        reserved = set(self.tokenizer.all_special_tokens)
        reserved.update(token for token in self.tokenizer.get_added_vocab() if token.startswith("<|"))
        if any(token in text for token in reserved):
            raise ValueError("Audio candidate contains an embedded control token.")
        messages = trajectory.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("Audio candidate scoring requires the original prompt messages.")
        wire_messages, native_messages = [], []
        audio_count = 0
        for message in messages:
            role, content = message.get("role"), message.get("content")
            if role not in {"system", "user"}:
                raise ValueError("Initial audio sequence scoring supports system/user prompts without tools only.")
            if isinstance(content, str):
                wire_messages.append({"role": role, "content": content})
                native_messages.append({"role": role, "content": content})
                continue
            if not isinstance(content, list):
                raise ValueError("Invalid audio prompt content.")
            wire_content, native_content = [], []
            for item in content:
                kind = item.get("type")
                if kind == "text":
                    wire_content.append({"type": "text", "text": item["text"]})
                    native_content.append({"type": "text", "text": item["text"]})
                elif kind == "audio":
                    value = item.get("audio", item.get("audio_url"))
                    if value is None and audio_data is not None and audio_count < len(audio_data):
                        value = audio_data[audio_count]
                    audio = _encode_audio(value, (mm_processor_kwargs or {}).get("sampling_rate"))
                    wire_content.append({"type": "input_audio", "input_audio": audio})
                    native_content.append({"type": "audio", "audio": "waveform"})
                    audio_count += 1
                else:
                    raise ValueError(f"Unsupported prompt modality for audio teacher: {kind!r}.")
            wire_messages.append({"role": role, "content": wire_content})
            native_messages.append({"role": role, "content": native_content})
        if not audio_count:
            raise ValueError("Audio candidate scoring requires at least one actual waveform.")
        prefix = trajectory.get("assistant_prefix", "")
        if prefix not in {"", "<think>\n\n</think>\n\n", "<think>\n\n</think>\n\n\n"}:
            raise ValueError("Unsupported student assistant prefix for native audio scoring.")
        assistant = {"role": "assistant", "content": prefix + text + self.eos_token, "reasoning_content": ""}
        # Empty reasoning_content prevents native templates from deleting/reformatting <think> text.
        wire_messages.append(assistant)
        native_messages.append(assistant)
        rendered = self.tokenizer.apply_chat_template(
            native_messages, tokenize=False, continue_final_message=True, add_generation_prompt=False
        )
        header = "<|im_start|>assistant\n" + prefix
        suffix = header + text + self.eos_token
        if not rendered.endswith(suffix):
            raise ValueError("Native chat template rewrites the candidate or removes its explicit EOS.")
        encoded = self.tokenizer(suffix, add_special_tokens=False, return_offsets_mapping=True)
        suffix_ids = list(encoded["input_ids"])
        offsets = encoded["offset_mapping"]
        start = next((index for index, (left, _) in enumerate(offsets) if left >= len(header)), None)
        if start is None or offsets[start][0] != len(header):
            raise ValueError("Native assistant token boundary overlaps its chat header.")
        if suffix_ids[-1] != self.eos_token_id:
            raise ValueError("Native assistant candidate is missing its verified EOS.")
        if len(suffix_ids) - start > self.config.audio_response_length:
            raise ValueError("Native candidate exceeds the configured audio response token budget.")
        payload = {
            "model": self.config.model_name,
            "messages": wire_messages,
            "continue_final_message": True,
            "add_generation_prompt": False,
            "prompt_logprobs": 0,
            "max_tokens": 1,
            "temperature": 1.0,
            "return_token_ids": True,
        }
        return payload, {"suffix_ids": suffix_ids, "candidate_start": start, "audio_count": audio_count}

    def parse(self, payload: dict, expected: dict) -> dict:
        ids, scores = payload.get("prompt_token_ids"), payload.get("prompt_logprobs")
        suffix = expected["suffix_ids"]
        if not isinstance(ids, list) or len(ids) < len(suffix) or ids[-len(suffix) :] != suffix:
            raise ValueError("Audio teacher changed candidate text/tokenization or omitted the explicit EOS.")
        if len(ids) + 1 > self.config.max_model_len:
            raise ValueError("Native audio teacher prompt exceeds its context budget.")
        start = len(ids) - len(suffix) + expected["candidate_start"]
        if start > self.config.audio_prompt_length:
            raise ValueError("Expanded native audio prompt exceeds the configured audio prompt budget.")
        if not isinstance(scores, list) or len(scores) != len(ids):
            raise ValueError("Audio teacher prompt score count does not match its native token IDs.")
        values = []
        for position in range(start, len(ids)):
            entry = scores[position]
            if not isinstance(entry, dict) or str(ids[position]) not in entry:
                raise ValueError("Audio teacher omitted a candidate/EOS token score.")
            value = entry[str(ids[position])].get("logprob")
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value > 1e-5
            ):
                raise ValueError("Audio teacher returned an invalid candidate/EOS log probability.")
            values.append(float(value))
        total = math.fsum(values)
        if not math.isfinite(total):
            raise ValueError("Audio teacher sequence score is nonfinite.")
        return {
            "teacher_sequence_logprob": total,
            "teacher_sequence_token_count": len(values),
            "teacher_sequence_mask": True,
            "audio_teacher_termination_included": True,
            "audio_teacher_truncated": False,
        }

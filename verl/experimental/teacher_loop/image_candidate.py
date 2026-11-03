# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Render image prompts, then score the original response token IDs directly.

Generated BPE token sequences need not survive decoding and re-tokenization,
even with the same tokenizer. vLLM's render API supplies native prompt tokens
and serialized image features; its token generation API consumes those features
alongside the unmodified student response IDs. Candidate text never passes
through the server's tokenizer or assistant chat template.
"""

import base64
import copy
import io
import math
import os
from typing import Any


def _encode_image(value: Any) -> str:
    """Embed actual image pixels, without requiring the teacher to read local paths."""
    from PIL import Image

    if isinstance(value, dict):
        value = value.get("bytes") if value.get("bytes") is not None else value.get("path", value.get("image"))
    if isinstance(value, str) and value.startswith("data:image/"):
        header, separator, encoded = value.partition(",")
        if not separator or not header.endswith(";base64"):
            raise ValueError("Image candidate requires a base64 image data URL.")
        value = base64.b64decode(encoded, validate=True)
    if isinstance(value, (str, os.PathLike)):
        if "://" in os.fspath(value):
            raise ValueError("Image candidate scoring requires local images or embedded image data.")
        with Image.open(value) as source:
            source.load()
            picture = source.copy()
    elif isinstance(value, (bytes, bytearray, memoryview)):
        with Image.open(io.BytesIO(bytes(value))) as source:
            source.load()
            picture = source.copy()
    elif isinstance(value, Image.Image):
        picture = value.copy()
    else:
        raise ValueError("Image candidate scoring requires actual decoded images, image bytes, or local paths.")
    if picture.width < 1 or picture.height < 1:
        raise ValueError("Image candidate scoring requires a nonempty image.")
    if picture.mode not in {"RGB", "RGBA", "L"}:
        picture = picture.convert("RGB")
    output = io.BytesIO()
    picture.save(output, format="PNG")
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii")


class ImageCandidateCodec:
    def __init__(self, config, tokenizer=None):
        self.config = config
        if tokenizer is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path, local_files_only=True)
        self.tokenizer = tokenizer

    def prepare(
        self,
        prompt_ids: list[int],
        response_ids: list[int],
        trajectory: dict,
        image_data=None,
        mm_processor_kwargs=None,
    ) -> tuple[dict, dict]:
        for name, ids in (("prompt", prompt_ids), ("response", response_ids)):
            if not ids or any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in ids):
                raise ValueError(f"Image scoring requires nonempty nonnegative student {name} token IDs.")
        if not isinstance(trajectory, dict):
            raise ValueError("Image scoring requires the original student trajectory.")
        prefix = trajectory.get("assistant_prefix")
        if not isinstance(prefix, str):
            raise ValueError("Image scoring requires a verified student assistant prefix.")
        marker = "<|im_start|>assistant\n"
        student_tail = self.tokenizer.decode(
            prompt_ids[-64:], skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        if marker not in student_tail or student_tail.rsplit(marker, 1)[1] != prefix:
            raise ValueError("Image scoring cannot verify the student's assistant prompt boundary.")
        terminated = trajectory.get("terminated", False)
        if not isinstance(terminated, bool) or (terminated and trajectory.get("truncated", False)):
            raise ValueError("Image candidate has inconsistent termination metadata.")
        body = self.tokenizer.decode(
            response_ids[:-1] if terminated else response_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if body != trajectory.get("response_text"):
            raise ValueError("Image candidate text does not match its exact student response tokens.")
        messages = trajectory.get("messages")
        if not isinstance(messages, list) or not messages or messages[-1].get("role") != "user":
            raise ValueError("Image reasoning scoring requires original system/user messages ending in a user query.")
        wire_messages, native_messages, image_count = [], [], 0
        for message in messages:
            role, content = message.get("role"), message.get("content")
            if role not in {"system", "user"}:
                raise ValueError("Image reasoning scoring supports system/user prompts without tools only.")
            if isinstance(content, str):
                wire_messages.append({"role": role, "content": content})
                native_messages.append({"role": role, "content": content})
                continue
            if not isinstance(content, list):
                raise ValueError("Invalid image reasoning prompt content.")
            wire_content, native_content = [], []
            for item in content:
                kind = item.get("type")
                if kind == "text" and isinstance(item.get("text"), str):
                    wire_content.append({"type": "text", "text": item["text"]})
                    native_content.append({"type": "text", "text": item["text"]})
                elif kind in {"image", "image_url"}:
                    value = item.get("image", item.get("image_url"))
                    if isinstance(value, dict) and "url" in value:
                        value = value["url"]
                    if image_data is not None:
                        if image_count >= len(image_data):
                            raise ValueError("Image prompt has more placeholders than student images.")
                        value = image_data[image_count]
                    wire_content.append({"type": "image_url", "image_url": {"url": _encode_image(value)}})
                    native_content.append({"type": "image", "image": "actual-image"})
                    image_count += 1
                else:
                    raise ValueError(f"Unsupported prompt modality for image teacher: {kind!r}.")
            wire_messages.append({"role": role, "content": wire_content})
            native_messages.append({"role": role, "content": native_content})
        if not image_count or (image_data is not None and image_count != len(image_data)):
            raise ValueError("Image scoring requires exactly the images present in the student prompt.")
        original = self.tokenizer.apply_chat_template(
            native_messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
        header = marker + prefix
        if not original.endswith(header):
            raise ValueError("Native image teacher template changes the student's assistant generation prefix.")
        header_ids = list(self.tokenizer(header, add_special_tokens=False)["input_ids"])
        if not header_ids or prompt_ids[-len(header_ids) :] != header_ids:
            raise ValueError("Native image teacher assistant header is not aligned with the student prompt.")
        payload = {
            "model": self.config.model_name,
            "messages": wire_messages,
            "add_generation_prompt": True,
            "add_special_tokens": False,
            "chat_template_kwargs": {"enable_thinking": True},
            "prompt_logprobs": 0,
            "max_tokens": 1,
            "temperature": 1.0,
            "return_token_ids": True,
        }
        if mm_processor_kwargs:
            payload["mm_processor_kwargs"] = dict(mm_processor_kwargs)
        return payload, {
            "header_ids": header_ids,
            "image_count": image_count,
            "student_prompt_ids": list(prompt_ids),
            "response_ids": list(response_ids),
        }

    def prepare_scoring(self, rendered: dict, expected: dict) -> tuple[dict, dict]:
        """Append sampled tokens without changing native image features or offsets."""
        native_ids = rendered.get("token_ids")
        header_ids = expected["header_ids"]
        if (
            not isinstance(native_ids, list)
            or not native_ids
            or any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in native_ids)
            or native_ids[-len(header_ids) :] != header_ids
        ):
            raise ValueError("Native image teacher assistant header is not aligned with the student prompt.")
        features = rendered.get("features")
        if not isinstance(features, dict):
            raise ValueError("Image render omitted the native image features.")
        image_count = expected["image_count"]
        for key in ("mm_hashes", "mm_placeholders", "kwargs_data"):
            values = features.get(key)
            if (
                not isinstance(values, dict)
                or set(values) != {"image"}
                or not isinstance(values["image"], list)
                or len(values["image"]) != image_count
            ):
                raise ValueError(f"Image render returned invalid {key} for the student images.")
        for key in ("mm_hashes", "kwargs_data"):
            if any(not isinstance(value, str) or not value for value in features[key]["image"]):
                raise ValueError(f"Image render requires complete {key}, without cache-only image references.")
        previous_end = 0
        for placeholder in features["mm_placeholders"]["image"]:
            if not isinstance(placeholder, dict):
                raise ValueError("Image render returned an invalid image placeholder.")
            offset, length = placeholder.get("offset"), placeholder.get("length")
            if (
                isinstance(offset, bool)
                or not isinstance(offset, int)
                or offset < previous_end
                or isinstance(length, bool)
                or not isinstance(length, int)
                or length <= 0
                or offset + length > len(native_ids) - len(header_ids)
            ):
                raise ValueError("Image render placeholder lies outside the original user prompt.")
            previous_end = offset + length
        sequence_ids = native_ids + expected["response_ids"]
        if len(sequence_ids) + 1 > self.config.max_model_len:
            raise ValueError("Expanded image teacher prompt has no room for one scoring token.")
        sampling_params = rendered.get("sampling_params")
        if not isinstance(sampling_params, dict):
            raise ValueError("Image render omitted the native sampling parameters.")
        # Render includes serialized tensors, not merely cache hashes. Preserve
        # all native metadata so subsequent HTTP requests remain cache-independent.
        request = copy.deepcopy(rendered)
        request["token_ids"] = sequence_ids
        request["stream"] = False
        request["sampling_params"].update(
            n=1, max_tokens=1, temperature=1.0, prompt_logprobs=0, detokenize=False, truncate_prompt_tokens=None
        )
        return request, {
            "prompt_token_ids": sequence_ids,
            "candidate_start": len(native_ids),
            "student_prompt_ids": expected["student_prompt_ids"],
            "response_ids": expected["response_ids"],
        }

    def parse(self, payload: dict, expected: dict) -> dict[str, list]:
        # The token-native API consumes the submitted IDs directly; unlike the
        # chat API it returns scores without echoing prompt_token_ids.
        ids, scores = expected["prompt_token_ids"], payload.get("prompt_logprobs")
        if not isinstance(scores, list) or len(scores) != len(ids):
            raise ValueError("Image teacher prompt scores do not match its expanded token IDs.")
        start = expected["candidate_start"]
        values = []
        for position in range(start, start + len(expected["response_ids"])):
            entry = scores[position]
            if not isinstance(entry, dict) or str(ids[position]) not in entry:
                raise ValueError("Image teacher omitted a student response token score.")
            value = entry[str(ids[position])].get("logprob")
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value > 1e-5
            ):
                raise ValueError("Image teacher returned an invalid student response log probability.")
            values.append([float(value)])
        prompt_ids, response_ids = expected["student_prompt_ids"], expected["response_ids"]
        # Match the managed teacher's next-token convention. Prompt positions
        # are unused by distillation; every response position has a real score.
        sequence_ids = prompt_ids + response_ids
        return {
            "prompt_ids": [[token] for token in sequence_ids[1:]] + [[0]],
            "prompt_logprobs": [[0.0] for _ in prompt_ids[:-1]] + values + [[0.0]],
        }

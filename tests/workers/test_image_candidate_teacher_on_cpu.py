"""Image-conditioned OPD must preserve student response tokens and old protocols."""

import asyncio
import base64
import copy
import io
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import torch
from aiohttp import web
from PIL import Image

from verl.experimental.teacher_loop.errors import TeacherScoringError
from verl.experimental.teacher_loop.external_teacher import ExternalTeacherClient, ExternalTeacherScorer
from verl.experimental.teacher_loop.image_candidate import ImageCandidateCodec, _encode_image
from verl.experimental.teacher_loop.teacher_manager import AsyncTeacherLLMServerManager
from verl.utils.async_rpc import RPCCancellationError
from verl.workers.config.distillation import DistillationLossConfig, ExternalTeacherConfig
from verl.workers.config.rollout import RolloutConfig
from verl.workers.rollout.replica import TokenOutput
from verl.workers.utils.padding import no_padding_2_padding


class TinyQwenTokenizer:
    """Exercise relevant stock-template trimming without external model files."""

    eos_token = "<|im_end|>"
    eos_token_id = 900
    specials = {"<|im_end|>": 900, "<|im_start|>": 901, "<|image_pad|>": 902}

    def __call__(self, text, **kwargs):
        ids, offsets, cursor = [], [], 0
        while cursor < len(text):
            special = next((token for token in self.specials if text.startswith(token, cursor)), None)
            end = cursor + (len(special) if special else 1)
            ids.append(self.specials[special] if special else ord(text[cursor]))
            offsets.append((cursor, end))
            cursor = end
        return {"input_ids": ids, "offset_mapping": offsets}

    def encode(self, text, **kwargs):
        return self(text)["input_ids"]

    def decode(self, ids, **kwargs):
        special = {value: key for key, value in self.specials.items()}
        return "".join(special[token] if token in special else chr(token) for token in ids)

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs.get("enable_thinking") is True
        text = ""
        for message in messages:
            content = message["content"]
            if isinstance(content, list):
                content = "".join(item["text"] if item["type"] == "text" else "<|image_pad|>" for item in content)
            text += "<|im_start|>" + message["role"] + "\n" + content.strip() + "<|im_end|>\n"
        if kwargs.get("add_generation_prompt"):
            text += "<|im_start|>assistant\n<think>\n"
        return text


def _config(tmp_path, **changes):
    values = dict(
        base_url="http://127.0.0.1:1",
        model_name="teacher",
        max_model_len=4096,
        tokenizer_path=str(tmp_path),
        enable_image_candidate_scoring=True,
    )
    values.update(changes)
    return ExternalTeacherConfig(**values)


def _case(tmp_path, response="Reasoning\n</think>\n\nRed.<|im_end|>", messages=None, tokenizer=None, response_ids=None):
    tokenizer = tokenizer or TinyQwenTokenizer()
    codec = ImageCandidateCodec(_config(tmp_path), tokenizer=tokenizer)
    messages = messages or [
        {"role": "system", "content": "Think carefully."},
        {"role": "user", "content": [{"type": "text", "text": "Describe."}, {"type": "image"}]},
    ]
    prompt = tokenizer.encode(
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True),
        add_special_tokens=False,
    )
    ids = list(response_ids) if response_ids is not None else tokenizer.encode(response)
    terminated = ids[-1] == tokenizer.eos_token_id
    trajectory = {
        "messages": messages,
        "response_text": tokenizer.decode(ids[:-1] if terminated else ids, clean_up_tokenization_spaces=False),
        "assistant_prefix": "<think>\n",
        "terminated": terminated,
        "truncated": not terminated,
    }
    image = Image.new("RGB", (5, 7), (255, 0, 10))
    return codec, prompt, ids, trajectory, image


def _rendered(expected, expansion=13):
    """Represent the cache-independent native multimodal render response."""
    return {
        "request_id": "render-fixture",
        "model": "teacher",
        "stream": False,
        "token_ids": [777, 777] + [902] * expansion + expected["header_ids"],
        "features": {
            "mm_hashes": {"image": ["image-content-hash"]},
            "mm_placeholders": {"image": [{"offset": 2, "length": expansion}]},
            "kwargs_data": {"image": [base64.b64encode(b"serialized-image-tensors").decode()]},
        },
        "sampling_params": {"max_tokens": 99, "temperature": 0.3, "prompt_logprobs": None},
    }


def _native_response(expected):
    ids = expected["prompt_token_ids"]
    return {"prompt_logprobs": [None] + [{str(token): {"logprob": -0.25}} for token in ids[1:]]}


def _prepared(codec, prompt, ids, trajectory, image, expansion=13):
    request, expected = codec.prepare(prompt, ids, trajectory, [image])
    generate, final = codec.prepare_scoring(_rendered(expected, expansion), expected)
    return request, generate, final


@pytest.mark.parametrize(
    "response",
    ["Reasoning\n</think>\n\nRed.<|im_end|>", "Still thinking\n\n", " \n", "\nLeading", "<|im_end|>"],
)
def test_native_image_scoring_preserves_thinking_eos_and_truncated_candidates(tmp_path, response):
    codec, prompt, ids, trajectory, image = _case(tmp_path, response)
    before = copy.deepcopy(trajectory)
    request, generate, expected = _prepared(codec, prompt, ids, trajectory, image, expansion=57)
    assert trajectory == before
    assert request["messages"][-1]["role"] == "user"
    assert request["messages"][-1]["content"][0] == {"type": "text", "text": "Describe."}
    assert request["messages"][-1]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert request["max_tokens"] == 1 and request["prompt_logprobs"] == 0
    assert request["add_generation_prompt"] is True
    assert request["chat_template_kwargs"]["enable_thinking"] is True
    assert not request.get("continue_final_message", False)
    assert generate["token_ids"][-len(ids) :] == ids
    assert generate["sampling_params"]["max_tokens"] == 1
    assert generate["sampling_params"]["prompt_logprobs"] == 0
    assert generate["sampling_params"]["temperature"] == 1.0
    assert generate["sampling_params"]["detokenize"] is False
    result = codec.parse(_native_response(expected), expected)
    data = {
        "prompts": torch.tensor([prompt]),
        "responses": torch.tensor([ids]),
        "attention_mask": torch.ones(1, len(prompt) + len(ids), dtype=torch.long),
    }
    actual_ids = no_padding_2_padding(torch.tensor(result["prompt_ids"]), data).squeeze(-1)
    scores = no_padding_2_padding(torch.tensor(result["prompt_logprobs"]), data).squeeze(-1)
    torch.testing.assert_close(actual_ids, data["responses"])
    torch.testing.assert_close(scores, torch.full((1, len(ids)), -0.25))


def test_render_request_preserves_original_user_whitespace(tmp_path):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Question. \n"},
                {"type": "text", "text": "\n  "},
            ],
        }
    ]
    codec, prompt, ids, trajectory, image = _case(tmp_path, messages=messages)
    request, _ = codec.prepare(prompt, ids, trajectory, [image])
    assert request["messages"][-1]["content"][1:] == messages[-1]["content"][1:]
    assert messages[-1]["content"][1]["text"] == "Question. \n"


def test_image_payload_embeds_the_actual_pixels_deterministically(tmp_path):
    image = Image.new("RGB", (5, 7), (255, 0, 10))
    encoded = _encode_image(image)
    assert encoded == _encode_image(image)
    decoded = Image.open(io.BytesIO(base64.b64decode(encoded.split(",", 1)[1])))
    assert decoded.size == image.size and decoded.tobytes() == image.tobytes()
    assert _encode_image({"bytes": base64.b64decode(encoded.split(",", 1)[1]), "path": None}) == encoded
    with pytest.raises(ValueError, match="local images"):
        _encode_image("https://example.org/image.png")


def test_native_scoring_forwards_full_image_tensors_without_cache_dependency(tmp_path):
    codec, prompt, ids, trajectory, image = _case(tmp_path)
    _, expected = codec.prepare(prompt, ids, trajectory, [image])
    rendered = _rendered(expected)
    before = copy.deepcopy(rendered)
    generate, final = codec.prepare_scoring(rendered, expected)
    assert rendered == before
    assert generate["features"] == rendered["features"]
    assert generate["features"]["kwargs_data"]["image"][0]
    assert generate["features"] is not rendered["features"]
    assert generate["token_ids"] == rendered["token_ids"] + ids
    assert final["candidate_start"] == len(rendered["token_ids"])
    assert final["prompt_token_ids"] == generate["token_ids"]
    generate["features"]["kwargs_data"]["image"][0] = "changed"
    assert rendered == before


def test_noncanonical_generated_tokens_are_preserved_without_model_files(tmp_path):
    class AliasTokenizer(TinyQwenTokenizer):
        def decode(self, ids, **kwargs):
            return super().decode([ord("x") if token == 777 else token for token in ids], **kwargs)

    tokenizer = AliasTokenizer()
    ids = [777, tokenizer.eos_token_id]
    assert tokenizer.encode(tokenizer.decode(ids)) != ids
    codec, prompt, ids, trajectory, image = _case(tmp_path, tokenizer=tokenizer, response_ids=ids)
    _, generate, expected = _prepared(codec, prompt, ids, trajectory, image)
    assert generate["token_ids"][-len(ids) :] == ids
    result = codec.parse(_native_response(expected), expected)
    assert result["prompt_ids"][-len(ids) - 1 : -1] == [[777], [tokenizer.eos_token_id]]


@pytest.mark.parametrize("failure", ["short_scores", "missing", "nan", "positive", "bool", "wrong_token"])
def test_image_teacher_targets_fail_closed(tmp_path, failure):
    codec, prompt, ids, trajectory, image = _case(tmp_path)
    _, _, expected = _prepared(codec, prompt, ids, trajectory, image)
    result = _native_response(expected)
    if failure == "short_scores":
        result["prompt_logprobs"].pop()
    elif failure == "missing":
        result["prompt_logprobs"][-1] = {}
    elif failure == "wrong_token":
        result["prompt_logprobs"][-1] = {"1": {"logprob": -0.25}}
    else:
        result["prompt_logprobs"][-1][str(ids[-1])]["logprob"] = {"nan": math.nan, "positive": 0.1, "bool": True}[
            failure
        ]
    with pytest.raises(ValueError):
        codec.parse(result, expected)


@pytest.mark.parametrize("failure", ["wrong_text", "wrong_prefix", "extra_image", "missing_image"])
def test_candidate_or_prompt_mismatch_is_rejected_before_http(tmp_path, failure):
    codec, prompt, ids, trajectory, image = _case(tmp_path)
    images = [image]
    if failure == "wrong_text":
        trajectory["response_text"] += "changed"
    elif failure == "wrong_prefix":
        trajectory["assistant_prefix"] = "different"
    elif failure == "extra_image":
        images.append(image)
    else:
        images = []
    with pytest.raises(ValueError):
        codec.prepare(prompt, ids, trajectory, images)


@pytest.mark.parametrize(
    "failure",
    [
        "no_tokens",
        "negative_token",
        "bool_token",
        "header_changed",
        "no_features",
        "no_hashes",
        "no_placeholders",
        "no_tensors",
        "cache_only_tensors",
        "empty_tensor",
        "out_of_bounds",
        "overlapping_header",
        "zero_length",
        "image_count_mismatch",
    ],
)
def test_native_render_structural_mismatches_are_rejected(tmp_path, failure):
    codec, prompt, ids, trajectory, image = _case(tmp_path)
    _, expected = codec.prepare(prompt, ids, trajectory, [image])
    rendered = _rendered(expected)
    features = rendered["features"]
    if failure == "no_tokens":
        rendered["token_ids"] = []
    elif failure == "negative_token":
        rendered["token_ids"][0] = -1
    elif failure == "bool_token":
        rendered["token_ids"][0] = True
    elif failure == "header_changed":
        rendered["token_ids"][-1] += 1
    elif failure == "no_features":
        rendered.pop("features")
    elif failure == "no_hashes":
        features.pop("mm_hashes")
    elif failure == "no_placeholders":
        features.pop("mm_placeholders")
    elif failure == "no_tensors":
        features.pop("kwargs_data")
    elif failure == "cache_only_tensors":
        features["kwargs_data"]["image"] = [None]
    elif failure == "empty_tensor":
        features["kwargs_data"]["image"] = [""]
    elif failure == "out_of_bounds":
        features["mm_placeholders"]["image"][0]["offset"] = len(rendered["token_ids"])
    elif failure == "overlapping_header":
        features["mm_placeholders"]["image"][0]["length"] = len(rendered["token_ids"]) - 2
    elif failure == "zero_length":
        features["mm_placeholders"]["image"][0]["length"] = 0
    else:
        features["mm_hashes"]["image"].append("extra-image")
    with pytest.raises(ValueError):
        codec.prepare_scoring(rendered, expected)


def test_image_context_checks_native_expansion_before_inference(tmp_path):
    codec, prompt, ids, trajectory, image = _case(tmp_path)
    _, expected = codec.prepare(prompt, ids, trajectory, [image])
    with pytest.raises(ValueError, match="Expanded image"):
        codec.prepare_scoring(_rendered(expected, expansion=4096), expected)
    rendered = _rendered(expected)
    codec.config = _config(tmp_path, max_model_len=len(rendered["token_ids"]) + len(ids) + 1)
    codec.prepare_scoring(rendered, expected)
    codec.config = _config(tmp_path, max_model_len=len(rendered["token_ids"]) + len(ids))
    with pytest.raises(ValueError):
        codec.prepare_scoring(rendered, expected)


@pytest.fixture(scope="module")
def real_qwen_tokenizer():
    checkpoint = Path("/cpfs01/thscc/sharestorage/iwc/HithinkOmni/user_workspace/models/Qwen3.5-397B-A17B-FP8")
    if not (checkpoint / "tokenizer_config.json").is_file():
        pytest.skip("Local Qwen3.5 teacher tokenizer is unavailable")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)


@pytest.mark.parametrize(
    "candidate", ["split_newlines", "leading_newline", "trailing_spaces", "partial_utf8", "split_utf8"]
)
def test_real_tokenizer_noncanonical_candidates_keep_original_ids(tmp_path, real_qwen_tokenizer, candidate):
    tokenizer = real_qwen_tokenizer
    eos = tokenizer.eos_token_id
    candidates = {
        "split_newlines": [198, 198] + tokenizer.encode("Answer.", add_special_tokens=False) + [eos],
        "leading_newline": [198] + tokenizer.encode("Answer.", add_special_tokens=False) + [eos],
        "trailing_spaces": [220, 220],
        "partial_utf8": [126],
        "split_utf8": [126, 102, eos],
    }
    ids = candidates[candidate]
    codec, prompt, ids, trajectory, image = _case(tmp_path, tokenizer=tokenizer, response_ids=ids)
    header = tokenizer.encode("<|im_start|>assistant\n<think>\n", add_special_tokens=False)
    combined = tokenizer.encode(
        tokenizer.decode(header + ids, clean_up_tokenization_spaces=False), add_special_tokens=False
    )
    assert combined != header + ids, "Regression fixture must exercise a decode/re-encode mismatch"
    request, generate, expected = _prepared(codec, prompt, ids, trajectory, image)
    assert request["messages"][-1]["content"][0]["text"] == "Describe."
    assert generate["token_ids"][-len(ids) :] == ids
    parsed = codec.parse(_native_response(expected), expected)
    assert parsed["prompt_ids"][-len(ids) - 1 : -1] == [[token] for token in ids]
    assert parsed["prompt_logprobs"][-len(ids) - 1 : -1] == [[-0.25] for _ in ids]


def test_image_scoring_opt_in_does_not_load_codec_for_old_text_path(tmp_path):
    async def exercise():
        scorer = ExternalTeacherScorer(_config(tmp_path, enable_image_candidate_scoring=False))
        scorer._get_image_codec = lambda: pytest.fail("Default text scoring loaded image code")
        scorer._request = AsyncMock(
            return_value={"choices": [{"prompt_token_ids": [1, 2], "prompt_logprobs": [None, {"2": {"logprob": -1}}]}]}
        )
        assert (await scorer.score([1, 2], {}))["prompt_logprobs"] == [[-1.0], [0.0]]
        with pytest.raises(ValueError, match="explicit"):
            await scorer.score_image_candidate([1], [2], {})

    asyncio.run(exercise())


@pytest.mark.parametrize("base_suffix", ["", "/v1", "/v1/"])
def test_image_http_render_and_token_routes_preserve_tokens_across_retry(tmp_path, base_suffix):
    codec, prompt, ids, trajectory, image = _case(tmp_path, response="\n\nAnswer.<|im_end|>")
    _, render_expected = codec.prepare(prompt, ids, trajectory, [image])
    rendered = _rendered(render_expected)
    _, expected = codec.prepare_scoring(rendered, render_expected)

    async def exercise():
        received = []

        async def render(request):
            payload = await request.json()
            received.append((request.path, payload))
            assert payload["messages"][-1]["content"][0]["text"] == "Describe."
            assert payload["add_generation_prompt"] is True
            return web.json_response(rendered)

        async def generate(request):
            payload = await request.json()
            received.append((request.path, payload))
            assert payload["token_ids"] == rendered["token_ids"] + ids
            assert payload["features"] == rendered["features"]
            assert payload["sampling_params"]["prompt_logprobs"] == 0
            if len(received) == 2:
                return web.Response(status=503)
            return web.json_response(_native_response(expected))

        app = web.Application()
        app.router.add_post("/v1/chat/completions/render", render)
        app.router.add_post("/inference/v1/generate", generate)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        scorer = ExternalTeacherScorer(
            _config(
                tmp_path,
                base_url=f"http://127.0.0.1:{port}{base_suffix}",
                max_retries=1,
                retry_backoff_s=0,
            )
        )
        scorer._image_codec = codec
        try:
            result = await scorer.score_image_candidate(prompt, ids, trajectory, [image])
            assert result["prompt_ids"][-len(ids) - 1 : -1] == [[token] for token in ids]
            assert [path for path, _ in received] == [
                "/v1/chat/completions/render",
                "/inference/v1/generate",
                "/inference/v1/generate",
            ]
            assert received[1][1] == received[2][1]
        finally:
            await scorer.close()
            await runner.cleanup()

    asyncio.run(exercise())


def test_image_and_swe_text_requests_share_the_same_semaphore(tmp_path):
    codec, prompt, ids, trajectory, image = _case(tmp_path)
    _, _, expected = _prepared(codec, prompt, ids, trajectory, image)

    async def exercise():
        scorer = ExternalTeacherScorer(_config(tmp_path, max_concurrent_requests=1), phase_timing=True)
        scorer._image_codec = codec
        entered, release = asyncio.Event(), asyncio.Event()
        generation_entered, generation_release = asyncio.Event(), asyncio.Event()
        requests = []

        async def request(method, path, payload):
            requests.append(path)
            if path == "/chat/completions/render":
                entered.set()
                await release.wait()
                _, render_expected = codec.prepare(prompt, ids, trajectory, [image])
                return _rendered(render_expected)
            if path == "/inference/v1/generate":
                generation_entered.set()
                await generation_release.wait()
                return _native_response(expected)
            return {"choices": [{"prompt_token_ids": [1, 2], "prompt_logprobs": [None, {"2": {"logprob": -1}}]}]}

        scorer._request = AsyncMock(side_effect=request)
        image_task = asyncio.create_task(scorer.score_image_candidate(prompt, ids, trajectory, [image]))
        await entered.wait()
        text_task = asyncio.create_task(scorer.score([1, 2], {}))
        await asyncio.sleep(0)
        assert requests == ["/chat/completions/render"]
        release.set()
        await generation_entered.wait()
        await asyncio.sleep(0)
        assert requests == ["/chat/completions/render", "/inference/v1/generate"]
        generation_release.set()
        image_result, _ = await asyncio.gather(image_task, text_task)
        assert image_result["prompt_logprobs"][-2] == [-0.25]
        assert requests == ["/chat/completions/render", "/inference/v1/generate", "/completions"]
        metrics = await scorer.pop_opd_phase_metrics()
        assert metrics["teacher_score"]["completed_count"] == 2

    asyncio.run(exercise())


def test_image_client_uses_explicit_branch_and_keeps_fatal_failure_semantics(tmp_path, monkeypatch):
    codec, prompt, ids, trajectory, image = _case(tmp_path)
    _, _, expected = _prepared(codec, prompt, ids, trajectory, image)
    result = codec.parse(_native_response(expected), expected)
    remote = Mock(return_value=result)
    handle = SimpleNamespace(score_image_candidate=SimpleNamespace(remote=remote))
    monkeypatch.setattr(
        "verl.experimental.teacher_loop.external_teacher.await_cancellable_ray_ref", AsyncMock(return_value=result)
    )
    client = ExternalTeacherClient({}, handle, _config(tmp_path))

    async def exercise():
        output = await client.generate(
            "image",
            prompt_ids=prompt + ids,
            sampling_params={},
            image_data=[image],
            image_candidate=True,
            student_prompt_ids=prompt,
            response_ids=ids,
            trajectory=trajectory,
        )
        assert output.extra_fields == result
        assert remote.call_args.args[:3] == (prompt, ids, trajectory)
        with pytest.raises(TeacherScoringError, match="text-only"):
            await client.generate("old-path", prompt_ids=prompt + ids, sampling_params={}, image_data=[image])
        with pytest.raises(TeacherScoringError, match="audio/video"):
            await client.generate(
                "image", prompt_ids=prompt + ids, sampling_params={}, image_candidate=True, audio_data=[1]
            )

    asyncio.run(exercise())


def test_image_manager_returns_existing_tensor_schema_and_reuses_selected_client(tmp_path):
    codec, prompt, ids, trajectory, image = _case(tmp_path)
    _, _, expected = _prepared(codec, prompt, ids, trajectory, image)
    result = codec.parse(_native_response(expected), expected)
    manager = object.__new__(AsyncTeacherLLMServerManager)
    manager.teacher_model_configs = {
        "swe": SimpleNamespace(backend="external", external=_config(tmp_path), inference=RolloutConfig(name="vllm"))
    }
    manager.teacher_client = {
        "swe": SimpleNamespace(generate=AsyncMock(return_value=TokenOutput(token_ids=[], extra_fields=result)))
    }
    manager.distillation_loss_config = DistillationLossConfig(loss_mode="k1")

    async def exercise():
        teacher_ids, scores = await manager.compute_image_teacher_logprobs_single(
            prompt, ids, trajectory, {"images": [image]}, routing_key="swe"
        )
        assert teacher_ids.shape == scores.shape == (len(prompt) + len(ids), 1)
        assert teacher_ids.dtype == torch.int32 and scores.dtype == torch.float32
        kwargs = manager.teacher_client["swe"].generate.call_args.kwargs
        assert kwargs["image_candidate"] is True and kwargs["student_prompt_ids"] == prompt
        assert kwargs["response_ids"] == ids and kwargs["image_data"] == [image]

    asyncio.run(exercise())


@pytest.mark.parametrize("failure", ["disabled", "shape", "missing", "nonnumeric"])
def test_image_manager_target_failures_use_fatal_teacher_error_channel(tmp_path, failure):
    manager = object.__new__(AsyncTeacherLLMServerManager)
    manager.teacher_model_configs = {
        "swe": SimpleNamespace(
            backend="external",
            external=_config(tmp_path, enable_image_candidate_scoring=failure != "disabled"),
            inference=RolloutConfig(name="vllm"),
        )
    }
    fields = {"prompt_ids": [[2], [0]], "prompt_logprobs": [[-0.5], [0.0]]}
    if failure == "shape":
        fields["prompt_ids"].pop()
    elif failure == "missing":
        del fields["prompt_logprobs"]
    elif failure == "nonnumeric":
        fields["prompt_logprobs"][0] = ["invalid"]
    manager.teacher_client = {
        "swe": SimpleNamespace(generate=AsyncMock(return_value=TokenOutput(token_ids=[], extra_fields=fields)))
    }
    manager.distillation_loss_config = DistillationLossConfig(loss_mode="k1")

    async def exercise():
        with pytest.raises(TeacherScoringError):
            await manager.compute_image_teacher_logprobs_single([1], [2], {}, {"images": ["actual"]}, routing_key="swe")
        if failure == "disabled":
            manager.teacher_client["swe"].generate.assert_not_awaited()

    asyncio.run(exercise())


@pytest.mark.parametrize("error", [asyncio.CancelledError(), RPCCancellationError("cleanup failed")])
def test_image_manager_preserves_cancellation_and_cleanup_error_semantics(tmp_path, error):
    manager = object.__new__(AsyncTeacherLLMServerManager)
    manager.teacher_model_configs = {
        "swe": SimpleNamespace(backend="external", external=_config(tmp_path), inference=RolloutConfig(name="vllm"))
    }
    manager.teacher_client = {"swe": SimpleNamespace(generate=AsyncMock(side_effect=error))}
    manager.distillation_loss_config = DistillationLossConfig(loss_mode="k1")

    async def exercise():
        with pytest.raises(type(error)) as raised:
            await manager.compute_image_teacher_logprobs_single([1], [2], {}, routing_key="swe")
        assert raised.value is error

    asyncio.run(exercise())

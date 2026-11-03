"""Native audio candidate scoring must include only candidate text and verified EOS."""

import asyncio
import base64
import io
import math
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import torch

from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopOutput, AgentLoopWorker
from verl.experimental.teacher_loop.audio_candidate import AudioCandidateCodec, _encode_audio
from verl.experimental.teacher_loop.errors import TeacherScoringError
from verl.experimental.teacher_loop.external_teacher import ExternalTeacherClient, ExternalTeacherScorer
from verl.trainer.distillation.config_utils import audio_teacher_for_sample
from verl.workers.config.distillation import (
    DistillationConfig,
    DistillationLossConfig,
    DistillationTeacherModelConfig,
    ExternalTeacherConfig,
)
from verl.workers.config.rollout import RolloutConfig


class TinyNativeTokenizer:
    eos_token = "<eos>"
    eos_token_id = 999
    all_special_tokens = ["<eos>"]

    def get_added_vocab(self):
        return {"<|im_start|>": 998}

    def __call__(self, text, **kwargs):
        ids, offsets, cursor = [], [], 0
        while cursor < len(text):
            end = cursor + (len(self.eos_token) if text[cursor:].startswith(self.eos_token) else 1)
            ids.append(self.eos_token_id if end - cursor > 1 else ord(text[cursor]))
            offsets.append((cursor, end))
            cursor = end
        return {"input_ids": ids, "offset_mapping": offsets}

    def encode(self, text, **kwargs):
        return self(text)["input_ids"]

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"tokenize": False, "continue_final_message": True, "add_generation_prompt": False}
        assert messages[-1]["reasoning_content"] == ""
        return "native-waveform-prefix<|im_start|>assistant\n" + messages[-1]["content"]


def _config(tmp_path, **kwargs):
    return ExternalTeacherConfig(
        base_url="http://127.0.0.1:1",
        model_name="audio",
        max_model_len=1024,
        protocol="audio_candidate_score",
        tokenizer_path=str(tmp_path),
        audio_prompt_length=512,
        audio_response_length=128,
        **kwargs,
    )


def _trajectory(**changes):
    value = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is audible?"},
                    {"type": "audio", "audio": "clip.wav"},
                ],
            }
        ],
        "response_text": "Birdsong.",
        "terminated": True,
        "truncated": False,
    }
    value.update(changes)
    return value


def _codec(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "verl.experimental.teacher_loop.audio_candidate._encode_audio",
        lambda *args: {
            "data": "actual-waveform",
            "format": "wav",
        },
    )
    return AudioCandidateCodec(_config(tmp_path), tokenizer=TinyNativeTokenizer())


def _native_response(expected):
    ids = [101, 102] + expected["suffix_ids"]
    return {
        "prompt_token_ids": ids,
        "prompt_logprobs": [None] + [{str(token): {"logprob": -0.25}} for token in ids[1:]],
    }


@pytest.mark.parametrize("prefix", ["", "<think>\n\n</think>\n\n", "<think>\n\n</think>\n\n\n"])
def test_native_audio_payload_preserves_candidate_and_scores_only_body_plus_eos(tmp_path, monkeypatch, prefix):
    codec = _codec(tmp_path, monkeypatch)
    request, expected = codec.prepare(_trajectory(assistant_prefix=prefix))
    assert request["messages"][0]["content"][1] == {
        "type": "input_audio",
        "input_audio": {"data": "actual-waveform", "format": "wav"},
    }
    assert request["messages"][-1]["content"] == prefix + "Birdsong.<eos>"
    assert request["continue_final_message"] and not request["add_generation_prompt"]
    result = codec.parse(_native_response(expected), expected)
    assert result["teacher_sequence_logprob"] == -0.25 * (len("Birdsong.") + 1)
    assert result["teacher_sequence_token_count"] == len("Birdsong.") + 1
    assert result["teacher_sequence_mask"] is True
    assert result["audio_teacher_termination_included"] is True


@pytest.mark.parametrize("failure", ["eos_removed", "body_changed", "short_scores", "missing_token", "nan", "positive"])
def test_native_audio_scores_fail_closed(tmp_path, monkeypatch, failure):
    codec = _codec(tmp_path, monkeypatch)
    _, expected = codec.prepare(_trajectory())
    result = _native_response(expected)
    if failure == "eos_removed":
        result["prompt_token_ids"].pop()
        result["prompt_logprobs"].pop()
    elif failure == "body_changed":
        result["prompt_token_ids"][-2] += 1
    elif failure == "short_scores":
        result["prompt_logprobs"].pop()
    elif failure == "missing_token":
        result["prompt_logprobs"][-1] = {}
    else:
        result["prompt_logprobs"][-1]["999"]["logprob"] = math.nan if failure == "nan" else 0.1
    with pytest.raises(ValueError):
        codec.parse(result, expected)


@pytest.mark.parametrize(
    "changes",
    [
        {"terminated": False},
        {"truncated": True},
        {"response_text": "bad<eos>"},
        {"response_text": "bad<|im_start|>"},
        {"messages": [{"role": "user", "content": "No audio"}]},
        {"assistant_prefix": "unverified prefix"},
    ],
)
def test_incomplete_or_rewritten_candidates_are_never_scored(tmp_path, monkeypatch, changes):
    codec = _codec(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        codec.prepare(_trajectory(**changes))


def test_audio_budgets_do_not_shrink_global_swe_context(tmp_path):
    audio = DistillationTeacherModelConfig(
        backend="external",
        key="audio",
        external=_config(tmp_path),
        inference=RolloutConfig(name="vllm", prompt_length=8192, response_length=253951),
    )
    config = DistillationConfig(
        enabled=True,
        teacher_models={"teacher_model": audio},
        distillation_loss=DistillationLossConfig(loss_mode="k1"),
    )
    assert config.teacher_models["default"].inference.prompt_length == 640
    assert config.teacher_models["default"].external.audio_response_length == 128


def test_default_text_teacher_does_not_load_audio_codec():
    async def exercise():
        config = ExternalTeacherConfig(base_url="http://127.0.0.1:1", model_name="text", max_model_len=16)
        scorer = ExternalTeacherScorer(config)
        scorer._get_audio_codec = lambda: pytest.fail("Text teacher loaded audio implementation")
        scorer._request = AsyncMock(
            return_value={
                "choices": [
                    {
                        "prompt_token_ids": [1, 2],
                        "prompt_logprobs": [None, {"2": {"logprob": -1.0}}],
                    }
                ]
            }
        )
        assert (await scorer.score([1, 2], {}))["prompt_logprobs"] == [[-1.0], [0.0]]
        with pytest.raises(ValueError, match="explicit"):
            await scorer.score_audio_candidate(_trajectory())

    asyncio.run(exercise())


@pytest.mark.parametrize(
    ("generation_eos", "response_mask", "truncated", "accepted"),
    [
        (248044, [1, 1], False, True),
        ([248044, 55], [1, 1], False, True),
        (None, [1, 1], False, False),
        (248044, [1, 0], False, False),
        (248044, [], False, False),
        (248044, [1, 1], True, False),
    ],
)
def test_audio_student_generation_eos_can_differ_from_chat_eos(generation_eos, response_mask, truncated, accepted):
    worker = object.__new__(AgentLoopWorker)
    worker.distillation_enabled = True
    worker.teacher_key = "teacher_route"
    worker.tokenizer = StudentTokenizer()
    worker.model_config = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=generation_eos))
    worker.teacher_server_manager = SimpleNamespace(
        uses_audio_sequence_score=lambda key: key == "audio",
        has_audio_teacher=True,
        compute_teacher_sequence_score_single=AsyncMock(
            return_value={
                "teacher_sequence_logprob": -7.0,
                "teacher_sequence_mask": True,
                "teacher_sequence_token_count": 9,
            }
        ),
    )
    output = AgentLoopOutput(
        prompt_ids=[1, 2],
        response_ids=[3, 248044],
        response_mask=response_mask,
        metrics=AgentLoopMetrics(),
        extra_fields={"audio_student_truncated": truncated},
    )

    async def exercise():
        call = worker._compute_teacher_logprobs(
            output, output.prompt_ids, output.response_ids, False, {"teacher_route": "audio"}
        )
        if accepted:
            await call
            request = worker.teacher_server_manager.compute_teacher_sequence_score_single.call_args.kwargs["trajectory"]
            assert request["response_text"] == "Birdsong."
            assert request["terminated"] is True
        else:
            with pytest.raises(TeacherScoringError):
                await call
            worker.teacher_server_manager.compute_teacher_sequence_score_single.assert_not_awaited()

    asyncio.run(exercise())


def test_audio_scorer_uses_shared_http_boundary_and_exact_scalar(tmp_path, monkeypatch):
    codec = _codec(tmp_path, monkeypatch)
    _, expected = codec.prepare(_trajectory())

    async def exercise():
        scorer = ExternalTeacherScorer(_config(tmp_path))
        scorer._audio_codec = codec
        scorer._request = AsyncMock(return_value=_native_response(expected))
        result = await scorer.score_audio_candidate(_trajectory())
        assert result["teacher_sequence_logprob"] == -2.5
        assert scorer._request.call_args.args[:2] == ("POST", "/chat/completions")

    asyncio.run(exercise())


def test_waveform_payload_is_actual_audio_and_deterministic(tmp_path):
    import numpy as np
    import soundfile as sf

    waveform = np.linspace(-0.25, 0.25, 320, dtype=np.float32)
    path = tmp_path / "clip.wav"
    sf.write(path, waveform, 16000, subtype="FLOAT")
    audio = _encode_audio(path)
    assert audio == _encode_audio(path)
    recovered, rate = sf.read(io.BytesIO(base64.b64decode(audio["data"])), dtype="float32")
    assert rate == 16000
    np.testing.assert_array_equal(recovered, waveform)


def test_audio_client_passes_native_trajectory_and_preserves_fatal_errors(tmp_path, monkeypatch):
    result = {"teacher_sequence_logprob": -3.0, "teacher_sequence_mask": True}
    handle = SimpleNamespace(score_audio_candidate=SimpleNamespace(remote=lambda *args: result))
    monkeypatch.setattr(
        "verl.experimental.teacher_loop.external_teacher.await_cancellable_ray_ref", AsyncMock(return_value=result)
    )
    client = ExternalTeacherClient(config={}, scoring_handle=handle, external_config=_config(tmp_path))

    async def exercise():
        output = await client.generate(
            "candidate", prompt_ids=[9999], sampling_params={}, audio_data=["clip.wav"], trajectory=_trajectory()
        )
        assert output.extra_fields == result
        with pytest.raises(TeacherScoringError, match="image/video"):
            await client.generate("candidate", prompt_ids=[9999], sampling_params={}, image_data=["image"])

    asyncio.run(exercise())


def test_audio_protocol_routing_is_explicit_and_preserves_sources(tmp_path):
    cfg = {
        "distillation": {
            "enabled": True,
            "teacher_key": "teacher_route",
            "teacher_models": {
                "teacher_model": {},
                "swe_teacher": {"key": "swe", "backend": "external", "external": {}},
                "audio_teacher": {
                    "key": "audio",
                    "backend": "external",
                    "external": {
                        "protocol": "audio_candidate_score",
                        "tokenizer_path": str(tmp_path),
                    },
                },
            },
        }
    }
    sample = {"teacher_route": "audio", "data_source": "audio_clotho_aqa"}
    assert audio_teacher_for_sample(cfg, sample)["protocol"] == "audio_candidate_score"
    assert sample["data_source"] == "audio_clotho_aqa"
    assert audio_teacher_for_sample(cfg, {"teacher_route": "swe"}) is None
    assert audio_teacher_for_sample(cfg, {"teacher_route": "unknown"}) is None


class StudentTokenizer:
    eos_token_id = 55

    def decode(self, ids, **kwargs):
        return "<|im_start|>assistant\n<think>\n\n</think>\n\n" if ids == [1, 2] else "Birdsong."


@pytest.mark.parametrize("failure", [None, "missing_eos", "masked_eos", "truncated"])
def test_worker_rejects_partial_audio_and_preserves_scalar_metadata(failure):
    worker = object.__new__(AgentLoopWorker)
    worker.distillation_enabled = True
    worker.teacher_key = "teacher_route"
    worker.tokenizer = StudentTokenizer()
    worker.teacher_server_manager = SimpleNamespace(
        uses_audio_sequence_score=lambda key: key == "audio",
        has_audio_teacher=True,
        compute_teacher_sequence_score_single=AsyncMock(
            return_value={
                "teacher_sequence_logprob": -7.0,
                "teacher_sequence_mask": True,
                "teacher_sequence_token_count": 9,
            }
        ),
    )
    output = AgentLoopOutput(
        prompt_ids=[1, 2],
        response_ids=[3, 55] if failure != "missing_eos" else [3, 4],
        response_mask=[1, 0] if failure == "masked_eos" else [1, 1],
        metrics=AgentLoopMetrics(),
        extra_fields={"audio_student_truncated": failure == "truncated"},
    )
    kwargs = {"teacher_route": "audio", "raw_prompt": _trajectory()["messages"], "reference": "never send gold"}

    async def exercise():
        call = worker._compute_teacher_logprobs(output, output.prompt_ids, output.response_ids, False, kwargs)
        if failure:
            with pytest.raises(TeacherScoringError):
                await call
            worker.teacher_server_manager.compute_teacher_sequence_score_single.assert_not_awaited()
        else:
            await call
            request = worker.teacher_server_manager.compute_teacher_sequence_score_single.call_args.kwargs["trajectory"]
            assert request["response_text"] == "Birdsong."
            assert "reference" not in request
            assert request["assistant_prefix"] == "<think>\n\n</think>\n\n"
            fields = output.as_dict()
            assert fields["teacher_sequence_mask"].dtype == torch.bool
            assert fields["teacher_sequence_logprob"].item() == -7
            assert fields["teacher_ids"].shape == (4, 1)

    asyncio.run(exercise())

"""Opt-in audio trajectories retain scalar targets and bypass only audio task rewards."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import torch
from omegaconf import OmegaConf
from transformers.feature_extraction_utils import BatchFeature

from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopOutput, AgentLoopWorker
from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop
from verl.trainer.ppo.v1 import agent_loop_tq
from verl.workers.rollout.replica import TokenOutput


def _config():
    return OmegaConf.create(
        {
            "distillation": {
                "enabled": True,
                "teacher_key": "teacher_route",
                "teacher_models": {
                    "teacher_model": {},
                    "swe_teacher": {"key": "swe", "backend": "external", "external": {"protocol": "text_token_ids"}},
                    "audio_teacher": {
                        "key": "audio",
                        "backend": "external",
                        "external": {
                            "protocol": "audio_candidate_score",
                            "audio_response_length": 8,
                        },
                    },
                },
            }
        }
    )


@pytest.mark.parametrize("route", ["audio", "swe"])
def test_single_turn_audio_budget_and_thinking_override_are_local(route):
    loop = object.__new__(SingleTurnAgentLoop)
    loop.config = _config()
    loop.rollout_config = SimpleNamespace(full_determinism=False)
    loop.response_length = 20
    original_template = {"enable_thinking": True}
    loop.apply_chat_template_kwargs = original_template
    loop.process_multi_modal_info = AsyncMock(return_value={"audios": ["actual.wav"]})
    loop._get_mm_processor_kwargs = lambda audios: {"sampling_rate": 16000}
    loop.enable_continuous_token = False
    loop.apply_chat_template = AsyncMock(return_value=[1, 2])
    loop.server_manager = SimpleNamespace(
        generate=AsyncMock(
            return_value=TokenOutput(
                token_ids=[3, 4],
                log_probs=[-0.1, -0.2],
                stop_reason="completed",
            )
        )
    )
    sampling = {"max_tokens": 20, "temperature": 1.0}

    async def exercise():
        output = await loop.run(sampling, teacher_route=route, raw_prompt=[{"role": "user", "content": "Listen"}])
        request = loop.server_manager.generate.call_args.kwargs
        assert request["sampling_params"]["max_tokens"] == (8 if route == "audio" else 20)
        assert loop.apply_chat_template_kwargs["enable_thinking"] is (route != "audio")
        assert original_template == {"enable_thinking": True}
        assert sampling["max_tokens"] == 20
        assert ("audio_student_truncated" in output.extra_fields) is (route == "audio")

    asyncio.run(exercise())


@pytest.mark.parametrize("verified", [True, False])
def test_v1_audio_reward_bypass_requires_verified_sequence_score(monkeypatch, verified):
    worker_class = agent_loop_tq.AgentLoopWorkerTQ.__ray_metadata__.modified_class
    worker = object.__new__(worker_class)
    worker.config = _config()
    worker.repeat_coordinator = None
    worker._repeat_debug_event = lambda *args, **kwargs: None
    worker.cutoff_session_states = {}
    worker._compute_score = AsyncMock(side_effect=AssertionError("Audio pure OPD must not invoke task rewards"))

    async def teacher(output, **kwargs):
        if verified:
            output.extra_fields.update(
                teacher_sequence_mask=True, teacher_sequence_logprob=-3.5, teacher_sequence_token_count=2
            )

    worker._compute_teacher_logprobs = AsyncMock(side_effect=teacher)
    worker._compute_multi_modal_inputs = lambda *args: {}
    worker._compute_position_ids = lambda inputs, *args: torch.arange(inputs.shape[-1]).unsqueeze(0)
    published = {}

    async def publish(**kwargs):
        published.update(kwargs)

    async def await_publish(partition, uid, task):
        return await task

    worker._await_tq_publication = await_publish
    monkeypatch.setattr(agent_loop_tq.tq, "async_kv_batch_put", publish)
    output = AgentLoopOutput(
        prompt_ids=[1], response_ids=[2, 3], response_mask=[1, 1], metrics=AgentLoopMetrics(), extra_fields={}
    )

    async def exercise():
        call = worker._agent_loop_postprocess(
            output, False, uid="audio-u", session_id=0, global_steps=0, teacher_route="audio"
        )
        if not verified:
            with pytest.raises(RuntimeError, match="verified audio teacher"):
                await call
            assert not published
        else:
            await call
            assert output.reward_score == 0.0
            assert output.extra_fields["audio_opd_task_reward_skipped"] is True
            assert published["fields"]["teacher_sequence_logprob"].tolist() == [-3.5]
            assert published["fields"]["teacher_sequence_mask"].tolist() == [True]
        worker._compute_teacher_logprobs.assert_awaited_once()
        worker._compute_score.assert_not_awaited()

    asyncio.run(exercise())


def test_v1_swe_missing_hidden_reward_marker_still_fails(monkeypatch):
    worker_class = agent_loop_tq.AgentLoopWorkerTQ.__ray_metadata__.modified_class
    worker = object.__new__(worker_class)
    worker.config = _config()
    worker.repeat_coordinator = None
    worker._repeat_debug_event = lambda *args, **kwargs: None
    worker._compute_teacher_logprobs = AsyncMock()
    monkeypatch.setattr(agent_loop_tq, "should_skip_hidden_reward", lambda *args, **kwargs: True)
    output = AgentLoopOutput(prompt_ids=[1], response_ids=[2], response_mask=[1], metrics=AgentLoopMetrics())

    async def exercise():
        with pytest.raises(RuntimeError, match="SWE outputs marked hidden_reward_skipped"):
            await worker._agent_loop_postprocess(output, False, uid="swe-u", session_id=0, teacher_route="swe")
        worker._compute_teacher_logprobs.assert_not_awaited()

    asyncio.run(exercise())


@pytest.mark.parametrize("with_visual", [False, True])
def test_opt_in_audio_postprocess_extracts_waveform_without_reexpanding_tokens(monkeypatch, with_visual):
    features = torch.ones((1, 80, 12))
    feature_mask = torch.ones((1, 12), dtype=torch.long)
    waveform = [torch.linspace(-0.1, 0.1, 160)]
    extracted = []

    class Processor:
        def extract_audio_features(self, audio, *, sampling_rate=None):
            assert audio is waveform
            extracted.append(sampling_rate)
            return BatchFeature({"input_features": features, "feature_attention_mask": feature_mask})

    class Tokenizer:
        def decode(self, ids, **kwargs):
            assert with_visual, "Audio-only postprocess must not decode expanded audio token IDs"
            return "image text with audio special tokens removed"

    visual_calls = []

    def build(processor, **kwargs):
        assert with_visual, "Audio-only postprocess must not call the token-expanding processor"
        visual_calls.append(kwargs)
        return BatchFeature(
            {
                "input_ids": torch.tensor([[9]]),
                "attention_mask": torch.tensor([[1]]),
                "image_grid_thw": torch.tensor([[1, 2, 3]]),
                "pixel_values": torch.zeros((6, 4)),
            }
        )

    monkeypatch.setattr("verl.experimental.agent_loop.agent_loop.build_multimodal_processor_inputs", build)
    worker = object.__new__(AgentLoopWorker)
    worker.processor = Processor()
    worker.tokenizer = Tokenizer()
    mm_data = {"audios": waveform}
    if with_visual:
        mm_data["images"] = ["image"]
    output = SimpleNamespace(multi_modal_data=mm_data, mm_processor_kwargs={"sampling_rate": 16000})
    result = worker._compute_multi_modal_inputs(output, torch.tensor([[19, 19, 19, 7]]))

    assert extracted == [16000]
    assert result["input_features"] is features
    assert result["feature_attention_mask"] is feature_mask
    assert "input_ids" not in result and "attention_mask" not in result
    if with_visual:
        assert visual_calls[0]["audio"] is None
        assert "sampling_rate" not in visual_calls[0]["mm_processor_kwargs"]
        assert result["images_seqlens"].tolist() == [6]
        assert result["pixel_values"].shape == (6, 4)


def test_default_audio_postprocess_keeps_existing_processor_call(monkeypatch):
    calls = []
    waveform = [torch.zeros(160)]

    def build(processor, **kwargs):
        calls.append(kwargs)
        return BatchFeature({"input_ids": torch.tensor([[1]]), "feature_attention_mask": torch.tensor([[1]])})

    monkeypatch.setattr("verl.experimental.agent_loop.agent_loop.build_multimodal_processor_inputs", build)
    worker = object.__new__(AgentLoopWorker)
    worker.processor = SimpleNamespace()
    worker.tokenizer = SimpleNamespace(decode=lambda *args, **kwargs: "legacy decoded text")
    output = SimpleNamespace(multi_modal_data={"audios": waveform}, mm_processor_kwargs={"sampling_rate": 16000})
    result = worker._compute_multi_modal_inputs(output, torch.tensor([[19, 7]]))

    assert calls[0]["audio"] is waveform
    assert calls[0]["text"] == ["legacy decoded text"]
    assert calls[0]["mm_processor_kwargs"] == {"sampling_rate": 16000}
    assert "feature_attention_mask" in result
    assert "input_ids" not in result

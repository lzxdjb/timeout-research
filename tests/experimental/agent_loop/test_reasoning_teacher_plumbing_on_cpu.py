"""Exercise the actual rollout-worker branch for opt-in image reasoning."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopOutput, AgentLoopWorker
from verl.experimental.teacher_loop.errors import TeacherScoringError


class StudentTokenizer:
    eos_token_id = 55

    def decode(self, ids, **kwargs):
        assert kwargs == {"skip_special_tokens": False, "clean_up_tokenization_spaces": False}
        if ids == [1, 2]:
            return "<|im_start|>assistant\n<think>\n"
        return "".join({3: "Still thinking", 4: "\n\n", 77: "<|endoftext|>", 55: "<|im_end|>"}[token] for token in ids)


def _worker(*, enabled=True, generation_eos=77):
    worker = object.__new__(AgentLoopWorker)
    worker.config = OmegaConf.create({"trainer": {"general_reasoning": {"enabled": enabled}}})
    worker.distillation_enabled = True
    worker.teacher_key = "teacher_route"
    worker.tokenizer = StudentTokenizer()
    worker.model_config = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=generation_eos))
    worker.teacher_server_manager = SimpleNamespace(
        uses_audio_sequence_score=lambda route: False,
        has_audio_teacher=True,
        compute_image_teacher_logprobs_single=AsyncMock(),
        compute_teacher_logprobs_single=AsyncMock(),
    )
    return worker


def _output(response_ids, *, images=True):
    return AgentLoopOutput(
        prompt_ids=[1, 2],
        response_ids=response_ids,
        response_mask=[1] * len(response_ids),
        multi_modal_data={"images": ["actual-student-image"]} if images else {},
        mm_processor_kwargs={"min_pixels": 784},
        metrics=AgentLoopMetrics(),
        extra_fields={},
    )


def _sample(*, reasoning=True):
    return {
        "teacher_route": np.array("swe", dtype=object),
        "is_general_reasoning": np.array(reasoning),
        "raw_prompt": [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Question"}]}],
        "reference": "gold must never enter teacher scoring",
    }


def _targets(output):
    count = len(output.prompt_ids) + len(output.response_ids)
    return torch.arange(count, dtype=torch.int32).unsqueeze(-1), torch.full((count, 1), -0.5)


@pytest.mark.parametrize(
    ("response_ids", "generation_eos", "terminated", "body"),
    [
        ([3, 77], 77, True, "Still thinking"),
        ([3, 77], [77, 78], True, "Still thinking"),
        ([3, 55], None, True, "Still thinking"),
        ([3, 4], 77, False, "Still thinking\n\n"),
        ([3, 77], None, False, "Still thinking<|endoftext|>"),
    ],
)
def test_worker_routes_image_reasoning_with_exact_ids_and_native_eos_metadata(
    response_ids, generation_eos, terminated, body
):
    worker = _worker(generation_eos=generation_eos)
    output, sample = _output(response_ids), _sample()
    targets = _targets(output)
    manager = worker.teacher_server_manager
    manager.compute_image_teacher_logprobs_single.return_value = targets

    async def exercise():
        await worker._compute_teacher_logprobs(output, output.prompt_ids, output.response_ids, False, sample)
        manager.compute_image_teacher_logprobs_single.assert_awaited_once()
        manager.compute_teacher_logprobs_single.assert_not_awaited()
        call = manager.compute_image_teacher_logprobs_single.call_args.kwargs
        assert call["prompt_ids"] == [1, 2] and call["response_ids"] == response_ids
        assert call["routing_key"] == "swe"
        assert call["multi_modal_data"] == output.multi_modal_data
        assert call["mm_processor_kwargs"] == output.mm_processor_kwargs
        trajectory = call["trajectory"]
        assert trajectory == {
            "messages": sample["raw_prompt"],
            "response_text": body,
            "assistant_prefix": "<think>\n",
            "terminated": terminated,
            "truncated": not terminated,
        }
        assert "reference" not in trajectory
        assert output.extra_fields["teacher_ids"] is targets[0]
        assert output.extra_fields["teacher_logprobs"] is targets[1]
        assert output.extra_fields["teacher_sequence_mask"] is False
        assert output.extra_fields["teacher_sequence_token_count"] == 0

    asyncio.run(exercise())


@pytest.mark.parametrize(
    ("enabled", "reasoning", "images"),
    [(False, True, True), (True, False, True), (True, True, False)],
)
def test_existing_text_teacher_path_is_preserved_outside_image_reasoning(enabled, reasoning, images):
    worker = _worker(enabled=enabled)
    output, sample = _output([3, 77], images=images), _sample(reasoning=reasoning)
    targets = _targets(output)
    manager = worker.teacher_server_manager
    manager.compute_teacher_logprobs_single.return_value = targets

    async def exercise():
        await worker._compute_teacher_logprobs(output, output.prompt_ids, output.response_ids, False, sample)
        manager.compute_image_teacher_logprobs_single.assert_not_awaited()
        manager.compute_teacher_logprobs_single.assert_awaited_once_with(
            sequence_ids=[1, 2, 3, 77],
            multi_modal_data=output.multi_modal_data,
            mm_processor_kwargs=output.mm_processor_kwargs,
            routing_key="swe",
        )
        assert output.extra_fields["teacher_ids"] is targets[0]
        assert output.extra_fields["teacher_logprobs"] is targets[1]

    asyncio.run(exercise())


@pytest.mark.parametrize(("validate", "distillation_enabled"), [(True, True), (False, False)])
def test_validation_and_non_distillation_rollouts_do_not_request_any_teacher(validate, distillation_enabled):
    worker = _worker()
    worker.distillation_enabled = distillation_enabled
    output = _output([3, 77])

    async def exercise():
        await worker._compute_teacher_logprobs(output, output.prompt_ids, output.response_ids, validate, _sample())
        worker.teacher_server_manager.compute_image_teacher_logprobs_single.assert_not_awaited()
        worker.teacher_server_manager.compute_teacher_logprobs_single.assert_not_awaited()
        assert "teacher_ids" not in output.extra_fields

    asyncio.run(exercise())


def test_image_teacher_failure_is_not_replaced_by_text_teacher_targets():
    worker = _worker()
    worker.teacher_server_manager.compute_image_teacher_logprobs_single.side_effect = TeacherScoringError(
        "bad alignment"
    )
    output = _output([3, 77])

    async def exercise():
        with pytest.raises(TeacherScoringError, match="bad alignment"):
            await worker._compute_teacher_logprobs(output, output.prompt_ids, output.response_ids, False, _sample())
        worker.teacher_server_manager.compute_teacher_logprobs_single.assert_not_awaited()
        assert "teacher_ids" not in output.extra_fields

    asyncio.run(exercise())

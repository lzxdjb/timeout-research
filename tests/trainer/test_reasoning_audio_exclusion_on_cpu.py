"""Local audio truncation can be excluded without weakening teacher validation."""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from verl.trainer.distillation.reasoning_audio import exclude_incomplete_audio_candidate
from verl.trainer.ppo.v1.replay_buffer import rollout_diagnostics
from verl.trainer.ppo.v1.utils import hidden_reward_was_skipped, v1_success_values


def _config():
    return OmegaConf.create({
        "trainer": {"general_reasoning": {"enabled": True}},
        "distillation": {
            "enabled": True,
            "teacher_key": "teacher_route",
            "distillation_loss": {"use_task_rewards": False},
            "teacher_models": {
                "teacher_model": {},
                "audio": {"key": "audio", "backend": "external", "external": {
                    "protocol": "audio_candidate_score", "audio_response_length": 4,
                }},
                "swe": {"key": "swe", "backend": "external", "external": {"protocol": "text_token_ids"}},
            },
        },
        "actor_rollout_ref": {"rollout": {"response_length": 100}},
        "data": {"max_response_length": 100},
    })


def _output(response=None, mask=None, **extra):
    response = [3, 4, 5, 6] if response is None else response
    return SimpleNamespace(
        prompt_ids=[1, 2], response_ids=response,
        response_mask=[1] * len(response) if mask is None else mask,
        reward_score=None, extra_fields=extra,
    )


def _exclude(config=None, output=None, sample=None, **kwargs):
    return exclude_incomplete_audio_candidate(
        _config() if config is None else config,
        _output() if output is None else output,
        {"teacher_route": "audio"} if sample is None else sample,
        tokenizer=SimpleNamespace(eos_token_id=55), **kwargs,
    )


@pytest.mark.parametrize("gate", ["profile_off", "validation", "distillation_off", "task_rewards", "swe", "unknown"])
def test_unselected_paths_are_unchanged(gate):
    config = _config()
    output = _output()
    sample = {"teacher_route": "audio"}
    if gate == "profile_off":
        config.trainer.general_reasoning.enabled = False
    elif gate == "distillation_off":
        config.distillation.enabled = False
    elif gate == "task_rewards":
        config.distillation.distillation_loss.use_task_rewards = True
    elif gate in {"swe", "unknown"}:
        sample["teacher_route"] = gate
    before = deepcopy(output)
    assert not _exclude(config, output, sample, validate=gate == "validation")
    assert vars(output) == vars(before)


@pytest.mark.parametrize("response,mask", [([], []), ([3, 4], [1]), ([3, 4], [1, 0]), ([3, 4], [1, 2])])
def test_malformed_or_empty_candidates_keep_strict_path(response, mask):
    output = _output(response, mask, audio_student_truncated=True)
    assert not _exclude(output=output)
    assert output.reward_score is None
    assert "teacher_ids" not in output.extra_fields


@pytest.mark.parametrize("response", [[3, 4], [3, 4, 5, 6]])
def test_unknown_native_finish_keeps_strict_path_even_at_the_response_cap(response):
    output = _output(response, generation_finish_reason="unexpected_protocol_stop")
    assert not _exclude(output=output)
    assert output.reward_score is None
    assert output.extra_fields == {"generation_finish_reason": "unexpected_protocol_stop"}


def test_short_unexplained_missing_eos_keeps_strict_path():
    output = _output([3, 4])
    assert not _exclude(output=output)
    assert output.reward_score is None
    assert output.extra_fields == {}


def test_missing_eos_configuration_keeps_strict_path():
    output = _output()
    assert not exclude_incomplete_audio_candidate(
        _config(), output, {"teacher_route": "audio"}, tokenizer=SimpleNamespace(eos_token_id=None),
    )
    assert output.reward_score is None
    assert output.extra_fields == {}


def test_malformed_native_finish_metadata_keeps_strict_path():
    output = _output(generation_finish_reason=["length"])
    assert not _exclude(output=output)
    assert output.reward_score is None


@pytest.mark.parametrize("eos,model", [
    (55, None),
    (66, SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=66))),
    (77, {"generation_config": {"eos_token_id": [66, 77]}}),
])
def test_complete_responses_preserve_tokenizer_and_generation_eos_union(eos, model):
    output = _output([3, 4, 5, eos], generation_finish_reason="stop")
    before = deepcopy(output)
    assert not _exclude(output=output, model_config=model)
    assert vars(output) == vars(before)


@pytest.mark.parametrize("extra,reason", [
    ({}, "missing_eos_at_response_cap"),
    ({"audio_student_truncated": True}, "student_response_truncated"),
    ({"generation_finish_reason": "length"}, "generation_length_limit"),
    ({"generation_finish_reason": "abort"}, "generation_aborted"),
    ({"generation_stop_reason": "aborted"}, "generation_aborted"),
])
def test_incomplete_candidates_keep_original_ids_and_publish_only_neutral_transport(extra, reason):
    output = _output(**extra, reward_extra_info={"preserved": 9})
    response = output.response_ids
    mask = output.response_mask
    assert _exclude(output=output)
    assert output.response_ids is response
    assert output.response_mask is mask
    assert output.response_ids == [3, 4, 5, 6]
    assert output.reward_score == 0.0
    fields = output.extra_fields
    assert fields["audio_opd_candidate_excluded"] is True
    assert fields["audio_opd_exclusion_reason"] == reason
    assert fields["reward_extra_info"]["preserved"] == 9
    for key in ("audio_response_tokens", "audio_response_cap", "audio_last_token_id", "audio_response_has_eos"):
        assert fields["reward_extra_info"][key] == fields[key]
    assert fields["audio_response_tokens"] == fields["audio_response_cap"] == 4
    assert fields["audio_last_token_id"] == 6
    assert fields["audio_response_has_eos"] is False
    assert fields["teacher_sequence_mask"] is False
    assert fields["teacher_sequence_logprob"] == 0.0
    assert fields["teacher_sequence_token_count"] == 0
    assert fields["teacher_ids"].shape == fields["teacher_logprobs"].shape == (6, 1)
    assert fields["teacher_ids"].dtype == torch.int32
    assert fields["teacher_logprobs"].dtype == torch.float32
    assert not torch.count_nonzero(fields["teacher_ids"])
    assert not torch.count_nonzero(fields["teacher_logprobs"])


@pytest.mark.parametrize("limit_source", ["rollout", "data", "request"])
def test_effective_response_cap_uses_the_smallest_actual_limit(limit_source):
    config = _config()
    output = _output([3, 4])
    if limit_source == "rollout":
        config.actor_rollout_ref.rollout.response_length = 2
    elif limit_source == "data":
        config.data.max_response_length = 2
    else:
        output.extra_fields["audio_response_cap"] = 2
    assert _exclude(config=config, output=output)
    assert output.extra_fields["audio_response_cap"] == 2


def test_explicit_length_finish_can_identify_a_short_context_limited_candidate():
    output = _output([3, 4], generation_finish_reason="length")
    assert _exclude(output=output)
    assert output.extra_fields["audio_opd_exclusion_reason"] == "generation_length_limit"


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("with_accuracy", [False, True])
def test_exclusion_transport_zero_is_an_unmeasured_outcome(nested, with_accuracy):
    info = {"audio_opd_candidate_excluded": True}
    extra = {"reward_extra_info": info} if nested else info
    if with_accuracy:
        extra.setdefault("reward_extra_info", {})["acc"] = 1.0
    assert hidden_reward_was_skipped(extra)
    assert v1_success_values([extra], [0.0]) == [None]


def test_complete_audio_task_reward_placeholders_keep_existing_diagnostic_convention():
    complete = {"audio_opd_task_reward_skipped": True, "teacher_sequence_mask": True}
    assert not hidden_reward_was_skipped(complete)
    assert v1_success_values([complete], [0.0]) == [0.0]
    complete["reward_extra_info"] = {"acc": 1.0}
    assert v1_success_values([complete], [0.0]) == [1.0]


def test_rollout_dump_preserves_exclusion_evidence_without_audio_payload_or_transport_tensors():
    output = _output(generation_finish_reason="length", generation_stop_reason="completed")
    assert _exclude(output=output)
    output.extra_fields["audio_waveform"] = "sensitive waveform"
    output.extra_fields["audio_path"] = "/sensitive/recording.wav"
    result = json.loads(json.dumps(rollout_diagnostics(output.extra_fields)))
    expected = output.extra_fields["reward_extra_info"]
    assert {key: result[key] for key in expected} == expected
    assert not ({"audio_waveform", "audio_path", "teacher_ids", "teacher_logprobs"} & result.keys())


def test_audio_diagnostic_whitelist_is_enabled_only_by_the_new_exclusion_marker():
    complete = {"audio_opd_candidate_excluded": False, "audio_response_tokens": 4, "audio_student_truncated": False}
    assert rollout_diagnostics(complete) == {"rollout_diagnostic_version": 1}

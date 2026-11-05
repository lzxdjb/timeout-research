"""Teacher-free audio generation and opt-in incomplete-candidate boundaries."""

import asyncio
import copy
from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf

from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop
from verl.utils.audio_rl import audio_rl_enabled, audio_rl_for_sample, exclude_incomplete_audio_rl_candidate
from verl.workers.rollout.replica import TokenOutput


def config(**settings):
    return {
        "trainer": {"audio_rl": {"enabled": True, **settings}},
        "distillation": {"enabled": False},
        "actor_rollout_ref": {"rollout": {"response_length": 252000}},
        "data": {"max_response_length": 252000},
    }


def candidate(tokens=None, mask=None, **extra):
    tokens = [11, 12, 13, 14] if tokens is None else tokens
    return SimpleNamespace(
        prompt_ids=[1, 2],
        response_ids=tokens,
        response_mask=[1] * len(tokens) if mask is None else mask,
        reward_score=None,
        extra_fields=extra,
    )


def exclude(output, cfg=None, source="audio_asr", **kwargs):
    return exclude_incomplete_audio_rl_candidate(
        cfg or config(response_length=4), output, {"data_source": source}, SimpleNamespace(eos_token_id=0), **kwargs
    )


@pytest.mark.parametrize("cfg", [None, {}, {"trainer": {"audio_rl": {"enabled": False}}}])
def test_profile_absent_or_disabled_is_unchanged(cfg):
    assert not audio_rl_enabled(cfg)
    assert audio_rl_for_sample(cfg, {"data_source": "audio_asr"}) is None


@pytest.mark.parametrize("source", ["audio_asr", "audio_closedqa", "audio_librispeech", np.str_("audio_aishell1")])
def test_profile_routes_actual_audio_sources_without_teacher_routes(source):
    cfg = config()
    original = copy.deepcopy(cfg)
    assert audio_rl_enabled(cfg)
    assert audio_rl_for_sample(cfg, {"data_source": source}) == {
        "response_length": 2048,
        "enable_thinking": False,
        "exclude_incomplete_sequences": True,
    }
    assert cfg == original


@pytest.mark.parametrize("source", [None, "swe_agent_verl", "benchmark_swe", "122b_math", ["audio_asr"]])
def test_profile_never_caps_other_sources(source):
    assert audio_rl_for_sample(config(), {"data_source": source, "teacher_route": "audio"}) is None


@pytest.mark.parametrize(
    "settings,name",
    [
        ({"enabled": "true"}, "enabled"),
        ({"response_length": 0}, "response_length"),
        ({"response_length": -1}, "response_length"),
        ({"response_length": 4.0}, "response_length"),
        ({"response_length": True}, "response_length"),
        ({"enable_thinking": "false"}, "enable_thinking"),
        ({"exclude_incomplete_sequences": 1}, "exclude_incomplete_sequences"),
    ],
)
def test_invalid_profile_rejected_before_any_generation(settings, name):
    with pytest.raises(ValueError, match=name):
        audio_rl_enabled(config(**settings))


def test_profile_rejects_simultaneous_distillation():
    cfg = config()
    cfg["distillation"]["enabled"] = True
    with pytest.raises(ValueError, match="distillation.enabled=False"):
        audio_rl_for_sample(cfg, {"data_source": "audio_asr"})


@pytest.mark.parametrize(
    "extra,reason",
    [
        ({"generation_finish_reason": "length"}, "generation_length_limit"),
        ({"generation_stop_reason": "length"}, "generation_length_limit"),
        ({"generation_finish_reason": "abort"}, "generation_aborted"),
        ({"audio_rl_stop_reason": "aborted"}, "generation_aborted"),
        ({"audio_rl_response_truncated": True}, "student_response_truncated"),
        ({}, "missing_eos_at_response_cap"),
    ],
)
def test_incomplete_audio_excluded_without_token_rewriting_or_teacher_fields(extra, reason):
    output = candidate(**extra)
    tokens, mask = output.response_ids[:], output.response_mask[:]
    assert exclude(output)
    assert output.response_ids == tokens
    assert output.response_mask == mask
    assert output.reward_score == 0.0
    assert output.extra_fields["audio_rl_candidate_excluded"] is True
    assert output.extra_fields["audio_rl_exclusion_reason"] == reason
    assert output.extra_fields["audio_response_cap"] == 4
    assert output.extra_fields["audio_response_tokens"] == 4
    assert output.extra_fields["reward_extra_info"]["audio_rl_exclusion_reason"] == reason
    assert not any(key.startswith("teacher_") for key in output.extra_fields)


@pytest.mark.parametrize("finish", [None, "stop", "completed"])
def test_natural_eos_at_cap_remains_valid(finish):
    output = candidate([11, 12, 13, 0], generation_finish_reason=finish)
    original = copy.deepcopy(output.__dict__)
    assert not exclude(output)
    assert output.__dict__ == original


def test_explicit_native_length_limit_excluded_even_with_eos_or_unknown_tokenizer_eos():
    for tokens in ([11, 12, 13, 0], [11]):
        output = candidate(tokens, generation_finish_reason="length")
        assert exclude_incomplete_audio_rl_candidate(
            config(response_length=4), output, {"data_source": "audio_asr"}, SimpleNamespace(eos_token_id=None)
        )


def test_model_generation_config_eos_supported():
    output = candidate([11, 12, 13, 99])
    assert not exclude(output, model_config={"generation_config": {"eos_token_id": [99, 100]}})


@pytest.mark.parametrize(
    "output",
    [
        candidate([11, 12]),
        candidate([], generation_finish_reason="abort"),
        candidate(mask=[1, 0, 1, 1], generation_finish_reason="length"),
        candidate(mask=[1], generation_finish_reason="length"),
        candidate(generation_finish_reason="unknown_backend_finish"),
    ],
)
def test_ambiguous_short_empty_or_tool_masked_output_unchanged(output):
    original = copy.deepcopy(output.__dict__)
    assert not exclude(output)
    assert output.__dict__ == original


@pytest.mark.parametrize(
    "cfg,source,validate",
    [
        (config(), "audio_asr", True),
        (config(exclude_incomplete_sequences=False), "audio_asr", False),
        (config(), "swe_agent_verl", False),
        ({"trainer": {}}, "audio_asr", False),
    ],
)
def test_validation_disabled_profile_and_swe_never_excluded(cfg, source, validate):
    output = candidate(generation_finish_reason="length")
    original = copy.deepcopy(output.__dict__)
    assert not exclude(output, cfg, source, validate=validate)
    assert output.__dict__ == original


def test_exclusion_uses_actual_smaller_request_cap_and_original_token_count():
    output = candidate([11, 12], audio_response_cap=2, audio_response_tokens=7, audio_rl_response_truncated=True)
    assert exclude(output)
    assert output.extra_fields["audio_response_cap"] == 2
    assert output.extra_fields["audio_response_tokens"] == 7


def run_loop(cfg, source="audio_asr", *, sampling=None, token_ids=None, native_finish="stop"):
    loop = SingleTurnAgentLoop.__new__(SingleTurnAgentLoop)
    loop.config = cfg
    loop.rollout_config = SimpleNamespace(full_determinism=True)
    loop.response_length = 252000
    loop.apply_chat_template_kwargs = {"enable_thinking": True, "other": "preserved"}
    loop.mm_processor_kwargs = {}
    loop.enable_continuous_token = False
    loop.processor = SimpleNamespace(feature_extractor=SimpleNamespace(sampling_rate=16000))
    waveform = (np.array([0.1, -0.1], dtype="float32"), 16000)
    observations = {}

    async def multi_modal(messages):
        observations["messages"] = messages
        return {"audios": [waveform]}

    async def template(messages, **kwargs):
        observations["template_kwargs"] = dict(loop.apply_chat_template_kwargs)
        observations["template_audio"] = kwargs["audios"]
        return [5, 6]

    async def generate(**kwargs):
        observations["request"] = kwargs
        return TokenOutput(
            token_ids=[11, 0] if token_ids is None else token_ids,
            log_probs=[-0.1] * (2 if token_ids is None else len(token_ids)),
            stop_reason="completed",
            extra_fields={"generation_finish_reason": native_finish},
        )

    loop.process_multi_modal_info = multi_modal
    loop.apply_chat_template = template
    loop.server_manager = SimpleNamespace(generate=generate)
    prompt = [{"role": "user", "content": "transcribe <audio>"}]
    sampling = {"temperature": 1, "top_p": 1} if sampling is None else sampling
    original_sampling = dict(sampling)
    output = asyncio.run(loop.run(sampling, priority=np.int64(7), data_source=source, raw_prompt=prompt))
    assert sampling == original_sampling
    assert observations["template_audio"][0] is waveform
    assert observations["request"]["audio_data"][0] is waveform
    assert observations["request"]["mm_processor_kwargs"] == {"sampling_rate": 16000}
    assert observations["request"]["request_id"] == "det-7"
    return output, observations


def test_pure_rl_audio_generation_has_no_teacher_call(monkeypatch):
    def forbid_teacher(*args, **kwargs):
        pytest.fail("Pure RL audio must not resolve or call a teacher")

    monkeypatch.setattr("verl.trainer.distillation.config_utils.audio_teacher_for_sample", forbid_teacher)
    output, observed = run_loop(config())
    assert observed["request"]["sampling_params"]["max_tokens"] == 2048
    assert observed["template_kwargs"] == {"enable_thinking": False, "other": "preserved"}
    assert output.extra_fields["audio_response_cap"] == 2048
    assert output.extra_fields["audio_rl_finish_reason"] == "stop"
    assert output.extra_fields["audio_rl_stop_reason"] == "completed"
    assert not any(key.startswith("teacher_") for key in output.extra_fields)


def test_generation_honors_smaller_sampling_limit_and_tracks_before_slice():
    output, observed = run_loop(
        config(response_length=4), sampling={"max_tokens": 2, "temperature": 1}, token_ids=[10, 11, 12]
    )
    assert observed["request"]["sampling_params"]["max_tokens"] == 2
    assert output.response_ids == [10, 11]
    assert len(output.response_logprobs) == 2
    assert output.extra_fields["audio_response_cap"] == 2
    assert output.extra_fields["audio_response_tokens"] == 3
    assert output.extra_fields["audio_rl_response_truncated"] is True
    assert exclude(output, config(response_length=4))


@pytest.mark.parametrize("cfg,source", [(config(), "swe_agent_verl"), ({}, "audio_asr")])
def test_existing_default_generation_retains_sampling_and_thinking(cfg, source):
    output, observed = run_loop(cfg, source)
    assert "max_tokens" not in observed["request"]["sampling_params"]
    assert observed["template_kwargs"]["enable_thinking"] is True
    assert "audio_response_cap" not in output.extra_fields


def test_existing_opd_audio_branch_retains_teacher_cap_and_fields():
    cfg = OmegaConf.create(
        {
            "distillation": {
                "enabled": True,
                "teacher_models": {
                    "audio": {
                        "backend": "external",
                        "external": {"protocol": "audio_candidate_score", "audio_response_length": 3},
                    }
                },
            }
        }
    )
    output, observed = run_loop(cfg, token_ids=[10, 11, 12, 13])
    assert observed["request"]["sampling_params"]["max_tokens"] == 3
    assert observed["template_kwargs"]["enable_thinking"] is False
    assert output.response_ids == [10, 11, 12]
    assert output.extra_fields["audio_student_truncated"] is True
    assert "audio_response_cap" not in output.extra_fields
    assert "audio_rl_finish_reason" not in output.extra_fields

"""Exclude locally incomplete audio candidates only in opted-in OPD profiles."""

import torch

from verl.trainer.distillation.config_utils import audio_teacher_for_sample, is_pure_distillation
from verl.trainer.distillation.reasoning_utils import audio_incomplete_exclusion_enabled


def _get(config, key, default=None):
    if config is None:
        return default
    return config.get(key, default) if hasattr(config, "get") else getattr(config, key, default)


def _eos_ids(tokenizer, model_config):
    generation = _get(model_config, "generation_config")
    result = set()
    for value in (_get(tokenizer, "eos_token_id"), _get(generation, "eos_token_id")):
        result.update(value if isinstance(value, (list, tuple, set)) else [value])
    result.discard(None)
    return result


def exclude_incomplete_audio_candidate(config, output, sample, *, tokenizer, model_config=None, validate=False):
    """Publish neutral transport fields for a locally incomplete, excluded candidate.

    The caller must publish ``train_sample_mask=False`` for a true return value.
    No teacher score is claimed, and genuine teacher/protocol errors retain the
    usual strict path. Complete candidates, other profiles, and validation are
    unchanged. Response IDs and their masks are never rewritten.
    """
    if (
        validate
        or not audio_incomplete_exclusion_enabled(config)
        or not is_pure_distillation(config.get("distillation"))
    ):
        return False
    teacher = audio_teacher_for_sample(config, sample)
    if teacher is None:
        return False
    response = output.response_ids
    mask = output.response_mask
    if not response or len(mask) != len(response) or not all(value == 1 for value in mask):
        return False

    extra = output.extra_fields
    rollout = _get(_get(config, "actor_rollout_ref", {}), "rollout", {})
    data = _get(config, "data", {})
    caps = [int(_get(teacher, "audio_response_length", 2048))]
    for value in (
        _get(rollout, "response_length"),
        _get(data, "max_response_length"),
        extra.get("audio_response_cap"),
    ):
        if value is not None:
            caps.append(int(value))
    cap = min(caps)
    if cap <= 0:
        return False

    finish = extra.get("generation_finish_reason", extra.get("audio_student_finish_reason", extra.get("finish_reason")))
    stop = extra.get("generation_stop_reason", extra.get("audio_student_stop_reason", extra.get("stop_reason")))
    eos_ids = _eos_ids(tokenizer, model_config)
    if not eos_ids or (finish is not None and finish not in ("stop", "length", "abort", "aborted", "completed")):
        return False
    has_eos = response[-1] in eos_ids
    truncated = bool(extra.get("audio_student_truncated", False))
    if finish in ("abort", "aborted") or stop in ("abort", "aborted"):
        reason = "generation_aborted"
    elif truncated or len(response) > cap:
        reason = "student_response_truncated"
    elif finish == "length" or stop == "length":
        reason = "generation_length_limit"
    elif not has_eos and len(response) >= cap:
        reason = "missing_eos_at_response_cap"
    else:
        return False

    diagnostics = {
        "audio_opd_candidate_excluded": True,
        "audio_opd_exclusion_reason": reason,
        "audio_response_tokens": len(response),
        "audio_response_cap": cap,
        "audio_last_token_id": response[-1],
        "audio_native_finish_reason": finish,
        "audio_native_stop_reason": stop,
        "audio_response_has_eos": has_eos,
        "audio_student_truncated": truncated,
    }
    output.reward_score = 0.0
    extra.update(diagnostics)
    extra.setdefault("reward_extra_info", {}).update(diagnostics)
    sequence_length = len(output.prompt_ids) + len(response)
    extra.update(
        teacher_ids=torch.zeros((sequence_length, 1), dtype=torch.int32),
        teacher_logprobs=torch.zeros((sequence_length, 1), dtype=torch.float32),
        teacher_sequence_mask=False,
        teacher_sequence_logprob=0.0,
        teacher_sequence_token_count=0,
    )
    return True

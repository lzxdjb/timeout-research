"""Opt-in audio generation and incomplete-candidate handling for pure RL."""


def _get(config, key, default=None):
    if config is None:
        return default
    return config.get(key, default) if hasattr(config, "get") else getattr(config, key, default)


def _positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _settings(config):
    settings = _get(_get(config, "trainer", {}), "audio_rl", {})
    enabled = _get(settings, "enabled", False)
    if type(enabled) is not bool:
        raise ValueError("trainer.audio_rl.enabled must be a boolean")
    if not enabled:
        return None
    if _get(_get(config, "distillation", {}), "enabled", False):
        raise ValueError("trainer.audio_rl requires distillation.enabled=False")
    result = {
        "response_length": _positive_int(_get(settings, "response_length", 2048), "trainer.audio_rl.response_length"),
    }
    for key, default in (("enable_thinking", False), ("exclude_incomplete_sequences", True)):
        value = _get(settings, key, default)
        if type(value) is not bool:
            raise ValueError(f"trainer.audio_rl.{key} must be a boolean")
        result[key] = value
    return result


def audio_rl_enabled(config):
    """Validate the opt-in pure-RL profile without constructing any teacher."""
    return _settings(config) is not None


def audio_rl_for_sample(config, sample):
    """Return validated generation settings only for an opted-in audio source."""
    settings = _settings(config)
    source = _get(sample, "data_source")
    if hasattr(source, "item"):
        source = source.item()
    return settings if settings is not None and isinstance(source, str) and source.startswith("audio_") else None


def _eos_ids(tokenizer, model_config):
    generation = _get(model_config, "generation_config")
    result = set()
    for value in (_get(tokenizer, "eos_token_id"), _get(generation, "eos_token_id")):
        result.update(value if isinstance(value, (list, tuple, set)) else [value])
    result.discard(None)
    return result


def exclude_incomplete_audio_rl_candidate(config, output, sample, tokenizer, model_config=None, validate=False):
    """Mark incomplete audio responses for exclusion from both loss and GRPO.

    Callers must propagate a true return value into ``train_sample_mask=False``.
    The zero reward only transports an excluded candidate through ordinary reward
    code; it must not participate in group statistics. Tokens and their response
    masks remain intact, and no teacher fields are created.
    """
    settings = audio_rl_for_sample(config, sample)
    if validate or settings is None or not settings["exclude_incomplete_sequences"]:
        return False
    response, mask = output.response_ids, output.response_mask
    if not response or len(mask) != len(response) or not all(value == 1 for value in mask):
        return False

    extra = output.extra_fields
    rollout = _get(_get(config, "actor_rollout_ref", {}), "rollout", {})
    data = _get(config, "data", {})
    caps = [settings["response_length"]]
    for value in (
        _get(rollout, "response_length"),
        _get(data, "max_response_length"),
        extra.get("audio_response_cap"),
    ):
        if value is not None:
            caps.append(_positive_int(value, "audio response cap"))
    cap = min(caps)
    finish = extra.get("generation_finish_reason", extra.get("audio_rl_finish_reason", extra.get("finish_reason")))
    stop = extra.get("generation_stop_reason", extra.get("audio_rl_stop_reason", extra.get("stop_reason")))
    eos_ids = _eos_ids(tokenizer, model_config)
    has_eos = response[-1] in eos_ids
    truncated = bool(extra.get("audio_rl_response_truncated", False))
    if finish in ("abort", "aborted") or stop in ("abort", "aborted"):
        reason = "generation_aborted"
    elif truncated or len(response) > cap:
        reason = "student_response_truncated"
    elif finish == "length" or stop == "length":
        reason = "generation_length_limit"
    elif eos_ids and finish in (None, "stop", "completed") and not has_eos and len(response) >= cap:
        reason = "missing_eos_at_response_cap"
    else:
        return False

    diagnostics = {
        "audio_rl_candidate_excluded": True,
        "audio_rl_exclusion_reason": reason,
        "audio_response_tokens": extra.get("audio_response_tokens", len(response)),
        "audio_response_cap": cap,
        "audio_last_token_id": response[-1],
        "audio_native_finish_reason": finish,
        "audio_native_stop_reason": stop,
        "audio_response_has_eos": has_eos,
        "audio_rl_response_truncated": truncated,
    }
    output.reward_score = 0.0
    extra.update(diagnostics)
    extra.setdefault("reward_extra_info", {}).update(diagnostics)
    return True

"""Small, opt-in helpers for reasoning alongside the existing SWE/audio recipe."""


def reasoning_profile_enabled(config):
    trainer = config.get("trainer", {})
    return bool(trainer.get("general_reasoning", {}).get("enabled", False))


def is_reasoning_sample(config, sample):
    if not reasoning_profile_enabled(config):
        return False
    value = sample.get("is_general_reasoning", False)
    return bool(value.item() if hasattr(value, "item") else value)


def skip_reasoning_task_reward(config, outputs, sample, *, validate=False):
    """Supply transport placeholders only for the new profile's OPD training rows."""
    if validate or not is_reasoning_sample(config, sample):
        return False
    from verl.trainer.distillation.config_utils import is_pure_distillation

    if not is_pure_distillation(config.get("distillation")):
        raise ValueError("General reasoning reward skipping requires pure OPD")
    if len(outputs) != 1 or not all(outputs[0].response_mask):
        raise ValueError("General reasoning OPD requires a single assistant-only response")
    output = outputs[0]
    output.reward_score = 0.0
    output.extra_fields["general_opd_task_reward_skipped"] = True
    output.extra_fields.setdefault("reward_extra_info", {}).update(
        general_opd_task_reward_skipped=True, task_reward_status="intentionally_skipped"
    )
    return True


def image_reasoning_trajectory(tokenizer, prompt_ids, response_ids, messages, eos_ids):
    """Recover the exact student assistant prefix without stripping thinking tokens."""
    from verl.experimental.teacher_loop.errors import TeacherScoringError

    if not response_ids:
        raise TeacherScoringError("Image reasoning scoring requires a nonempty student response")
    terminated = response_ids[-1] in eos_ids
    candidate_ids = response_ids[:-1] if terminated else response_ids
    text = tokenizer.decode(candidate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    tail = tokenizer.decode(prompt_ids[-64:], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    marker = "<|im_start|>assistant\n"
    if marker not in tail:
        raise TeacherScoringError("Cannot verify the image reasoning assistant prompt boundary")
    return {
        "messages": list(messages),
        "response_text": text,
        "assistant_prefix": tail.rsplit(marker, 1)[1],
        "terminated": terminated,
        "truncated": not terminated,
    }


def reasoning_row_mask(data, response_mask):
    """Read the new profile's tensor discriminator, absent in existing recipes."""
    import torch

    value = data.get("general_reasoning_mask")
    if value is None:
        return torch.zeros(response_mask.shape[0], device=response_mask.device, dtype=torch.bool)
    if value.ndim != 1 or value.shape[0] != response_mask.shape[0]:
        raise ValueError("general_reasoning_mask must have shape [batch_size]")
    return value.to(device=response_mask.device, dtype=torch.bool)

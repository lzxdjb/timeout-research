"""Opt-in audio RL validity and source metrics for the synchronous trainer."""

from collections import defaultdict

import torch

from verl.utils.audio_rl import audio_rl_enabled


def audio_rl_grpo_valid_mask(config, keys, tags, evaluator_mask):
    """Exclude incomplete audio sessions from their peers' reward statistics.

    Generic row masks and existing OPD fields keep their original meaning.
    Session IDs and group membership remain unchanged.
    """
    if not audio_rl_enabled(config):
        return evaluator_mask
    if len(keys) != len(tags) or tuple(evaluator_mask.shape) != (len(keys),):
        raise ValueError("Audio RL validity requires one tag and evaluator mask entry per output")
    excluded = {
        key.rsplit("_", 1)[0]
        for key, tag in zip(keys, tags, strict=True)
        if tag.get("audio_rl_candidate_excluded", False)
    }
    if not excluded:
        return evaluator_mask
    keep = torch.tensor(
        [key.rsplit("_", 1)[0] not in excluded for key in keys],
        dtype=torch.bool,
        device=evaluator_mask.device,
    )
    return evaluator_mask.bool() & keep


def audio_rl_source_metrics(config, keys, tags, train_mask, draw_counts):
    """Report actual admitted/eligible sessions alongside cumulative prompt draws."""
    if not audio_rl_enabled(config):
        return {}
    if not (len(keys) == len(tags) == len(train_mask)):
        raise ValueError("Audio RL source metrics require one source tag and mask entry per output")
    sources = defaultdict(lambda: defaultdict(set))
    for key, tag, keep in zip(keys, tags, train_mask, strict=True):
        if tag.get("is_padding", False):
            continue
        source = tag.get("mixed_rl_source")
        if not isinstance(source, str) or not source:
            continue
        session, prompt = key.rsplit("_", 1)[0], key.rsplit("_", 2)[0]
        values = sources[source]
        values["admitted_trajectories"].add(session)
        values["admitted_prompts"].add(prompt)
        if keep:
            values["eligible_trajectories"].add(session)
            values["eligible_prompts"].add(prompt)
        if tag.get("audio_rl_candidate_excluded", False):
            values["excluded_incomplete_trajectories"].add(session)
    metrics = {}
    for source in set(sources) | set(draw_counts):
        prefix = f"training/mixed_rl/{source}"
        metrics[f"{prefix}/requested_prompts_total"] = float(draw_counts.get(source, 0))
        for name in (
            "admitted_trajectories",
            "admitted_prompts",
            "eligible_trajectories",
            "eligible_prompts",
            "excluded_incomplete_trajectories",
        ):
            metrics[f"{prefix}/{name}"] = float(len(sources[source][name]))
    return metrics

"""Conservative SWE call evidence and local PPO costs (no output-text inference)."""

import hashlib
import json
import math
import re

PROOF_CODES = frozenset({"unknown_tool", "invalid_json", "invalid_argument", "invalid_regex", "path_rejected"})


def rejection_metadata(code):
    if code not in PROOF_CODES:
        raise ValueError(code)
    return {
        "tool_contract_outcome_version": 1,
        "tool_contract_rejection": {"version": 1, "code": code, "attribution": "agent_contract", "rejected": True},
    }


def valid_proof(value):
    return (
        isinstance(value, dict)
        and type(value.get("version")) is int
        and value["version"] == 1
        and isinstance(value.get("code"), str)
        and value["code"] in PROOF_CODES
        and value.get("rejected") is True
        and value.get("attribution") == "agent_contract"
    )


def token_digest(ids):
    return hashlib.sha256(json.dumps(list(ids), separators=(",", ":")).encode()).hexdigest()


async def align_single_call(tokenizer, parser, agent_data, tools):
    """Abstain unless one complete XML call maps to actual retained generated IDs.

    Reparse the exact span using the production parser. Hash the full response
    prefix as well as the call so later template merges/truncation cannot move it.
    """
    if type(parser).__name__ != "Qwen3XMLToolParser":
        return None
    calls = agent_data.tool_calls
    ids = list(agent_data.response_ids)
    mask = agent_data.response_mask
    if len(calls) != 1 or not ids or len(mask) < len(ids):
        return None
    if list(agent_data.prompt_ids[-len(ids) :]) != ids or not all(mask[-len(ids) :]):
        return None
    start_ids = tokenizer.encode("<tool_call>", add_special_tokens=False)
    end_ids = tokenizer.encode("</tool_call>", add_special_tokens=False)
    if len(start_ids) != 1 or len(end_ids) != 1:
        return None
    if ids.count(start_ids[0]) != 1 or ids.count(end_ids[0]) != 1:
        return None
    begin, end = ids.index(start_ids[0]), ids.index(end_ids[0]) + 1
    if begin >= end:
        return None
    text = tokenizer.decode(ids)
    call_text = tokenizer.decode(ids[begin:end])
    if (
        text.count("<tool_call>") != 1
        or text.count("</tool_call>") != 1
        or call_text.count("<function=") != 1
        or call_text.count("</function>") != 1
        or re.fullmatch(r"<tool_call>\s*<function=[^>]+>.*</function>\s*</tool_call>", call_text, re.S) is None
    ):
        return None
    _, reparsed = await parser.extract_tool_calls(ids[begin:end], tools)
    if len(reparsed) != 1 or (reparsed[0].name, reparsed[0].arguments) != (calls[0].name, calls[0].arguments):
        return None
    offset = len(mask) - len(ids)
    start, stop = offset + begin, offset + end
    response = list(agent_data.prompt_ids[-len(mask) :])
    return {
        "version": 1,
        "start": start,
        "end": stop,
        "token_sha256": token_digest(response[start:stop]),
        "prefix_sha256": token_digest(response[:stop]),
    }


def verified_span(event, response, mask):
    if (
        not isinstance(event, dict)
        or type(event.get("version")) is not int
        or event["version"] != 1
        or not valid_proof(event.get("proof"))
    ):
        return None
    start, end = event.get("start"), event.get("end")
    if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(response):
        return None
    if end > len(mask) or not all(mask[start:end]):
        return None
    if token_digest(response[start:end]) != event.get("token_sha256"):
        return None
    if token_digest(response[:end]) != event.get("prefix_sha256"):
        return None
    return start, end


def build_cost_weights(
    responses, response_mask, extra_fields, active, *, cost, cap, warmup_steps, step, trajectory_ids=None,
    event_key="tool_failure_events", verifier=None, diagnostic_prefix="tool_failure_", existing_weights=None,
):
    """Validate sparse events and cap trajectory cost before any minibatch split."""
    import torch

    if not math.isfinite(cost) or not math.isfinite(cap) or min(cost, cap) < 0 or warmup_steps < 0:
        raise ValueError("Tool-failure cost/cap must be finite and nonnegative; warmup must be nonnegative")
    verifier = verifier or verified_span
    weights = torch.zeros_like(response_mask, dtype=torch.float32)
    warmup = min(1.0, max(0.0, step / warmup_steps)) if warmup_steps else 1.0
    trajectory_ids = list(range(len(extra_fields))) if trajectory_ids is None else trajectory_ids
    counts = {
        "events": 0,
        "applied": 0,
        "abstained": 0,
        "cost": 0.0,
        "active": int(active.sum()),
        "dispatches": 0,
        "proved_rejections": 0,
        "outcome_metadata_calls": 0,
        "alignment_abstentions": 0,
    }
    by_trajectory, seen = {}, set()
    response_rows = responses.tolist()
    reserved = {}
    if existing_weights is not None:
        for row, values in enumerate(existing_weights.detach().gt(0).tolist()):
            if any(values):
                reserved.setdefault(trajectory_ids[row], []).append((response_rows[row], values))
    for row, (ids, mask, extra) in enumerate(
        zip(response_rows, response_mask.tolist(), extra_fields, strict=True)
    ):
        extra = getattr(extra, "data", extra)
        if isinstance(extra, dict):
            for key in ("dispatches", "proved_rejections", "outcome_metadata_calls", "alignment_abstentions"):
                value = extra.get(diagnostic_prefix + key, 0)
                if type(value) is int and value >= 0:
                    counts[key] += value
        events = extra.get(event_key, []) if isinstance(extra, dict) else []
        if not isinstance(events, list):
            continue
        counts["events"] += len(events)
        spans, occupied = [], set()
        for event in events:
            span = verifier(event, ids, mask)
            identity = (trajectory_ids[row], span, event.get("prefix_sha256")) if span else None
            already_charged = span is not None and any(
                any(values[span[0]:span[1]]) and ids[:span[1]] == previous_ids[:span[1]]
                for previous_ids, values in reserved.get(trajectory_ids[row], [])
            )
            if (span is None or occupied.intersection(range(*span)) or identity in seen
                    or already_charged):
                counts["abstained"] += 1
                continue
            occupied.update(range(*span))
            seen.add(identity)
            spans.append(span)
        if not bool(active[row]) or not spans:
            continue
        by_trajectory.setdefault(trajectory_ids[row], []).extend((row, start, end) for start, end in spans)
    for trajectory, spans in by_trajectory.items():
        spent = (sum(float(existing_weights[i].sum()) for i, key in enumerate(trajectory_ids)
                     if key == trajectory) if existing_weights is not None else 0.0)
        per_call = (min(cost, cap / len(spans)) * warmup if existing_weights is None
                    else min(cost * warmup, max(0.0, cap * warmup - spent) / len(spans)))
        if per_call == 0:
            continue
        for row, start, end in spans:
            weights[row, start:end] = per_call / (end - start)
        counts["applied"] += len(spans)
        counts["cost"] += per_call * len(spans)
    return weights, counts


def gate_positive_credit(advantages, weights):
    import torch

    return torch.where(weights > 0, advantages.clamp_max(0), advantages)


def local_ppo_loss(
    log_prob, old_log_prob, weights, response_mask, config, *, active_global_count, dp_size=1, rollout_is_weights=None
):
    """Negative-advantage clipped PPO; weights already contain per-call token means.

    A stable global *active trajectory* denominator includes unflagged rows.
    Callers sum microbatch contributions; engine handles its usual DP/CP scaling.
    """
    import torch

    ratio = (log_prob - old_log_prob).clamp(-20, 20).exp()
    low = config.clip_ratio_low if config.clip_ratio_low is not None else config.clip_ratio
    high = config.clip_ratio_high if config.clip_ratio_high is not None else config.clip_ratio
    clipped = torch.maximum(ratio, ratio.clamp(1 - low, 1 + high))
    dual = torch.minimum(clipped, torch.full_like(clipped, config.clip_ratio_c))
    losses = dual * weights
    if rollout_is_weights is not None:
        losses = losses * rollout_is_weights
    denominator = active_global_count if active_global_count > 0 else 1
    return (losses * response_mask).sum() * dp_size / denominator


def trajectory_normalizer(active, trajectory_ids):
    """One denominator unit per session, shared across its active outputs."""
    import torch

    counts = {}
    for keep, key in zip(active.tolist(), trajectory_ids, strict=True):
        if keep:
            counts[key] = counts.get(key, 0) + 1
    return torch.tensor(
        [1.0 / counts[key] if keep else 0.0 for keep, key in zip(active.tolist(), trajectory_ids, strict=True)],
        device=active.device,
        dtype=torch.float32,
    )

"""Conservative exact thinking repetition and verified native-token local costs.

The default text API retains the frozen offline v2 rules. Training uses
capture_generation() with native IDs and verified tokenizer offsets exclusively.
"""

import re
from collections import defaultdict
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class Config:
    sentence_min_tokens: int = 16
    sentence_min_copies: int = 5
    sentence_min_surplus: int = 96
    sentence_max_gap: int = 512
    sentence_min_density: float = 0.15
    tandem_min_period: int = 8
    tandem_max_period: int = 512
    tandem_min_copies: int = 6
    tandem_min_surplus: int = 128
    protect_copies: int = 2
    aggregate_enabled: bool = False
    aggregate_max_gap: int = 4096
    aggregate_min_identities: int = 3
    aggregate_min_surplus: int = 512
    aggregate_min_fraction: float = 0.30

    def __post_init__(self):
        integers = (self.aggregate_max_gap, self.aggregate_min_identities, self.aggregate_min_surplus)
        if (
            any(type(v) is not int or v <= 0 for v in integers)
            or self.aggregate_min_identities < 2
            or not 0 < self.aggregate_min_fraction <= 1
        ):
            raise ValueError("Invalid thinking-repeat aggregate thresholds")


@dataclass(frozen=True)
class CrossConfig:
    enabled: bool = False
    min_tokens: int = 32
    min_copies: int = 8
    min_surplus: int = 256
    max_period: int = 4
    protect_copies: int = 2

    def __post_init__(self):
        integers = (self.min_tokens, self.min_copies, self.min_surplus, self.max_period, self.protect_copies)
        if (
            any(type(v) is not int or v <= 0 for v in integers)
            or self.min_copies <= self.protect_copies
            or self.max_period > 4
        ):
            raise ValueError("Invalid thinking-repeat cross-turn thresholds")


def configs_from_actor(actor):
    """Actor config is authoritative; environment switches enter via Hydra."""
    return Config(
        **{
            key: actor.get("thinking_repeat_" + key, field.default)
            for key, field in Config.__dataclass_fields__.items()
            if key.startswith("aggregate_")
        }
    ), CrossConfig(
        **{
            key: actor.get("thinking_repeat_cross_turn_" + key, field.default)
            for key, field in CrossConfig.__dataclass_fields__.items()
            if key != "protect_copies"
        }
    )


DEFAULT_CONFIG = Config()
DEFAULT_CROSS_CONFIG = CrossConfig()


def protected_intervals(text):
    """Protect code, quoted strings, and unknown/unclosed quote tails.

    Inline backtick identifiers are protected too. They may establish identity
    of an otherwise repeated sentence, but never appear in returned spans.
    """
    spans = []
    fence = None
    offset = 0
    for line in text.splitlines(keepends=True):
        stripped = line.lstrip()
        marker = re.match(r"(`{3,}|~{3,})", stripped)
        if fence or marker or stripped.startswith(">") or line.startswith(("    ", "\t")):
            spans.append((offset, offset + len(line)))
        if marker:
            if fence is None:
                fence = (marker[1][0], len(marker[1]))
            elif marker[1][0] == fence[0] and len(marker[1]) >= fence[1]:
                fence = None
        offset += len(line)
    # Scan remaining text; apostrophes within words are contractions, not quotes.
    blocked = np.zeros(len(text), dtype=bool)
    for a, b in spans:
        blocked[a:b] = True
    i = 0
    while i < len(text):
        if blocked[i]:
            i += 1
            continue
        c = text[i]
        if c not in "`\"'“‘":
            i += 1
            continue
        if c == "'" and i > 0 and text[i - 1].isalnum():
            i += 1
            continue
        closer = {"“": "”", "‘": "’"}.get(c, c)
        j = i + 1
        while j < len(text):
            if text[j] == "\\":
                j += 2
                continue
            if text[j] == closer:
                break
            j += 1
        end = min(j + 1, len(text))
        spans.append((i, end))
        blocked[i:end] = True
        i = end
    return blocked


def _runs(mask):
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return zip(edges[::2].tolist(), edges[1::2].tolist(), strict=True)


def _token_mask(offsets, blocked):
    prefix = np.r_[0, np.cumsum(blocked)]
    return np.asarray([b > a and prefix[b] == prefix[a] for a, b in offsets], dtype=bool)


def detect(text, tokenizer, config=DEFAULT_CONFIG, *, encoding=None):
    """Return independently verifiable evidence and ONLY later eligible tokens.

    All matching uses exact identities. Whitespace normalization, fuzzy matching,
    and cross-turn matching are intentionally absent. First two occurrences of
    each identity remain protected. Detection thresholds are offline hypotheses.
    """
    boundary = re.search(r"</?think\b|<tool_call\b|</?function\b", text)
    if boundary:
        text = text[: boundary.start()]
    enc = encoding if encoding is not None else tokenizer.encode(text, add_special_tokens=False)
    ids = np.asarray(enc.ids, dtype=np.int64)
    offsets = enc.offsets
    blocked = protected_intervals(text)
    eligible = _token_mask(offsets, blocked)
    proposed = np.zeros(len(ids), dtype=bool)
    protect = np.zeros(len(ids), dtype=bool)
    evidence = []
    groups = defaultdict(list)
    # Complete sentences only. Delimiters inside quotes/code do not split prose.
    cuts = [0]
    for m in re.finditer(r"[.!?](?=\s|$)|[。！？]|\n\s*\n", text):
        if not blocked[m.start()]:
            cuts.append(m.end())
    cuts.append(len(text))
    token_starts = np.asarray([a for a, _ in offsets])
    token_ends = np.asarray([b for _, b in offsets])
    for a, b in zip(cuts, cuts[1:], strict=False):
        while a < b and text[a].isspace():
            a += 1
        while b > a and text[b - 1].isspace():
            b -= 1
        if a == b or text[b - 1] not in ".!?。！？":
            continue
        # Quotes may identify legitimate fixtures: reject these sentences.
        # Inline code must be short identifiers, not a quoted code example.
        fragment = text[a:b]
        inlines = list(re.finditer(r"`([^`\n]*)`", fragment))
        inline_mask = np.zeros(b - a, dtype=bool)
        for m in inlines:
            if len(m[1]) <= 96 and not re.search(r"[{};\n]", m[1]):
                inline_mask[m.start() : m.end()] = True
        if np.any(blocked[a:b] & ~inline_mask):
            continue
        si = int(np.searchsorted(token_ends, a, side="right"))
        ei = int(np.searchsorted(token_starts, b, side="left"))
        # Retokenize exact sentence to avoid position-dependent whitespace BPE.
        identity = tuple(tokenizer.encode(fragment, add_special_tokens=False).ids)
        if len(identity) >= config.sentence_min_tokens:
            groups[(fragment, identity)].append((si, ei, a, b))
    aggregate = []
    for (fragment, identity), occurrences in groups.items():
        if config.aggregate_enabled:
            # Group exact identities across protected regions without selecting
            # any code/quote tokens. A larger window is evidence, not permission
            # to penalize the intervening text.
            relaxed = [[]]
            for item in occurrences:
                if relaxed[-1] and item[0] - relaxed[-1][-1][1] > config.aggregate_max_gap:
                    relaxed.append([])
                relaxed[-1].append(item)
            for cluster in relaxed:
                if (
                    len(cluster) >= config.sentence_min_copies
                    and (len(cluster) - config.protect_copies) * len(identity) >= config.sentence_min_surplus
                ):
                    aggregate.append((fragment, cluster))
        clusters = [[]]
        for item in occurrences:
            if clusters[-1] and item[0] - clusters[-1][-1][1] > config.sentence_max_gap:
                clusters.append([])
            clusters[-1].append(item)
        for cluster in clusters:
            copies = len(cluster)
            surplus = (copies - config.protect_copies) * len(identity)
            density = sum(e - s for s, e, _, _ in cluster) / max(1, cluster[-1][1] - cluster[0][0])
            if (
                copies < config.sentence_min_copies
                or surplus < config.sentence_min_surplus
                or density < config.sentence_min_density
            ):
                continue
            for s, e, _, _ in cluster[: config.protect_copies]:
                protect[s:e] = True
            for s, e, _, _ in cluster[config.protect_copies :]:
                proposed[s:e] = True
            evidence.append(
                {
                    "rule": "exact_sentence",
                    "copies": copies,
                    "unit_tokens": len(identity),
                    "density": density,
                    "unit": fragment,
                    "occurrences": [[a, b] for _, _, a, b in cluster],
                }
            )
    if config.aggregate_enabled and aggregate:
        # Connected bounded-gap windows; density uses ALL tokens in each window,
        # including protected regions. Sparse repeated quotations cannot inflate it.
        entries = sorted(
            (item[0], item[1], fragment, index) for fragment, cluster in aggregate for index, item in enumerate(cluster)
        )
        windows = [[]]
        end = -1
        for item in entries:
            if windows[-1] and item[0] - end > config.aggregate_max_gap:
                windows.append([])
            windows[-1].append(item)
            end = max(end, item[1])
        for window in windows:
            identities = {item[2] for item in window}
            if len(identities) < config.aggregate_min_identities:
                continue
            candidate = np.zeros(len(ids), dtype=bool)
            first_copies = np.zeros(len(ids), dtype=bool)
            for s, e, _, index in window:
                (first_copies if index < config.protect_copies else candidate)[s:e] = True
            candidate &= eligible & ~first_copies
            lo, hi = window[0][0], max(item[1] for item in window)
            count = int(candidate.sum())
            if count < config.aggregate_min_surplus or count / max(1, hi - lo) < config.aggregate_min_fraction:
                continue
            proposed |= candidate
            protect |= first_copies
            evidence.append(
                {
                    "rule": "exact_sentence_aggregate",
                    "identities": len(identities),
                    "selected_tokens": count,
                    "fraction": count / (hi - lo),
                    "start": offsets[lo][0],
                    "end": offsets[hi - 1][1],
                }
            )
    # Adjacent token cycles can catch degeneration lacking sentence boundaries.
    # Check every lag; np comparisons verify equality (no probabilistic hashes).
    for cs, ce in _runs(eligible):
        a = ids[cs:ce]
        for period in range(
            config.tandem_min_period, min(config.tandem_max_period, len(a) // config.tandem_min_copies) + 1
        ):
            eq = a[:-period] == a[period:]
            for s, e in _runs(eq):
                copies = (e - s + period) // period
                surplus = (copies - config.protect_copies) * period
                if copies < config.tandem_min_copies or surplus < config.tandem_min_surplus:
                    continue
                start = cs + s
                end = start + copies * period
                protect[start : start + config.protect_copies * period] = True
                proposed[start + config.protect_copies * period : end] = True
                # Store all intervals, but keep evidence output bounded.
                if sum(x["rule"] == "exact_tandem" for x in evidence) < 24:
                    evidence.append(
                        {
                            "rule": "exact_tandem",
                            "copies": copies,
                            "unit_tokens": period,
                            "start": offsets[start][0],
                            "end": offsets[end - 1][1],
                            "unit": text[offsets[start][0] : offsets[start + period - 1][1]],
                        }
                    )
    final = proposed & eligible & ~protect
    spans = [[offsets[s][0], offsets[e - 1][1]] for s, e in _runs(final)]
    return {
        "matched": bool(final.any()),
        "penalized_tokens": int(final.sum()),
        "tokens": len(ids),
        "spans": spans,
        "evidence": evidence,
        "config": asdict(config),
        "token_spans": list(_runs(final)),
        "provenance": "native_generated_ids" if encoding is not None else "retokenized_decoded_thinking_offline_only",
    }


def prompt_opens_thinking(tokenizer, prompt_ids):
    """Only an explicit open protocol tag at the prompt suffix grants provenance."""
    tail = tokenizer.decode(list(prompt_ids[-32:]), skip_special_tokens=False, clean_up_tokenization_spaces=False)
    return re.search(r"<think>\s*$", tail) is not None


def capture_generation(
    tokenizer,
    agent_data,
    *,
    prompt_thinking=False,
    config=DEFAULT_CONFIG,
    cross_config=DEFAULT_CROSS_CONFIG,
    candidate_out=None,
):
    """One native assistant generation -> sparse verified event, or abstention.

    Returns (event, status). No textual reconstruction is accepted unless a full
    decode/encode round trip reproduces every native ID. No assumed token offsets
    from the offline dumps enter this path. Implicit thinking requires the
    original prompt to end in an explicit <think> tag.
    """
    from types import SimpleNamespace

    from verl.utils.swe_tool_failure import token_digest

    ids = list(agent_data.response_ids)
    mask = list(agent_data.response_mask)
    if not ids or len(mask) < len(ids) or not all(mask[-len(ids) :]):
        return None, "alignment_abstentions_mask"
    if list(agent_data.prompt_ids[-len(ids) :]) != ids:
        return None, "alignment_abstentions_suffix"
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None:
        return None, "alignment_abstentions_backend"
    text = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    encoded = backend.encode(text, add_special_tokens=False)
    if encoded.ids != ids:
        return None, "alignment_abstentions_roundtrip"
    offsets = encoded.offsets
    # Native offsets must reconstruct each token without ambiguous byte pieces.
    if any(not 0 <= a < b <= len(text) for a, b in offsets):
        return None, "alignment_abstentions_offset_bounds"
    if any(a < pa or b < pb for (pa, pb), (a, b) in zip(offsets, offsets[1:], strict=False)):
        return None, "alignment_abstentions_offset_order"
    # One leading thinking region only; tool arguments and final output are out.
    opening = re.match(r"\s*<think>", text)
    if opening:
        if prompt_thinking:
            return None, "alignment_abstentions_nested_protocol"  # nested protocol ambiguity
        start = opening.end()
    elif prompt_thinking:
        start = 0
    else:
        return None, "no_thinking"
    boundary = re.search(
        r"</?think\b|<tool_call\b|</?function\b|<\|im_end\|>|<\|im_start\|>|<\|endoftext\|>", text[start:]
    )
    end = start + boundary.start() if boundary else len(text)
    selected = [i for i, (a, b) in enumerate(offsets) if start <= a < b <= end]
    if not selected:
        return None, "no_thinking"
    first, last = selected[0], selected[-1] + 1
    if selected != list(range(first, last)):
        return None, "alignment_abstentions_noncontiguous"
    # Shared-offset UTF-8 byte tokens must not straddle thinking boundaries.
    if (
        first
        and offsets[first - 1][1] > offsets[first][0]
        or last < len(offsets)
        and offsets[last][0] < offsets[last - 1][1]
    ):
        return None, "alignment_abstentions_boundary_overlap"
    chunk = text[start:end]
    native = SimpleNamespace(ids=ids[first:last], offsets=[(a - start, b - start) for a, b in offsets[first:last]])
    result = detect(chunk, backend, config, encoding=native)
    if not result["matched"] and not (cross_config.enabled and candidate_out is not None):
        return None, "unflagged"
    offset = len(mask) - len(ids)
    spans = [[offset + first + a, offset + first + b] for a, b in result["token_spans"]]
    response = list(agent_data.prompt_ids[-len(mask) :])
    stop = len(mask)
    event = {
        "version": 1,
        "detector": "exact_thinking_v3" if config.aggregate_enabled else "exact_thinking_v2",
        "start": offset,
        "end": stop,
        "thinking_start": offset + first,
        "thinking_end": offset + last,
        "spans": spans,
        "token_sha256": token_digest(ids),
        "prefix_sha256": token_digest(response[:stop]),
        "rules": sorted({e["rule"] for e in result["evidence"]}),
    }
    if cross_config.enabled and candidate_out is not None and not protected_intervals(chunk).any():
        stripped = chunk.strip()
        trim_start, trim_end = len(chunk) - len(chunk.lstrip()), len(chunk.rstrip())
        inside = [i for i, (a, b) in enumerate(native.offsets) if trim_start <= a < b <= trim_end]
        if len(inside) >= cross_config.min_tokens:
            import hashlib

            cross_event = {
                **event,
                "detector": "exact_thinking_v3",
                "rules": ["exact_cross_turn_read"],
                "spans": [[offset + first + inside[0], offset + first + inside[-1] + 1]],
            }
            if verified_event(cross_event, response, mask) is not None:
                candidate_out.update(
                    thought_sha256=hashlib.sha256(stripped.encode()).hexdigest(),
                    tokens=len(inside),
                    event=cross_event,
                    assistant_turn=int(agent_data.assistant_turns),
                )
    if not result["matched"]:
        return None, "unflagged"
    if verified_event(event, response, mask) is None:
        return None, "alignment_abstentions_event_verification"
    return event, "flagged_turns"


def detect_cross_turn(events, config=DEFAULT_CROSS_CONFIG):
    """Exact thought/read cycles; ambiguous shell/test calls are shadow-only.

    Every event must be consecutive in both tool and assistant-turn space.
    Missing native candidates, undelivered feedback, concurrency, or a mutation
    is a barrier. Returned spans still require final native transport validation.
    """
    from recipe.swe_agent.repetition_reward import digest

    from verl.utils.swe_tool_repeat import _keys

    if not config.enabled:
        return [], {"cross_turn_selected": 0, "cross_turn_shadow": 0}
    strict, shadow, turns = [], [], []
    for event in events:
        candidate = event.get("thinking_candidate") or {}
        valid = (
            isinstance(candidate, dict)
            and isinstance(candidate.get("event"), dict)
            and isinstance(candidate.get("thought_sha256"), str)
            and len(candidate["thought_sha256"]) == 64
            and type(candidate.get("tokens")) is int
            and candidate["tokens"] >= config.min_tokens
            and type(candidate.get("assistant_turn")) is int
            and type(event.get("id")) is int
            and event.get("serial") is True
            and event.get("feedback_delivered") is True
            and event.get("infrastructure_failure") is not True
        )
        read = None
        if valid:
            try:
                read = _keys(event).get("unchanged_reads")
            except (KeyError, TypeError, ValueError):
                pass  # Malformed executor evidence is never positive evidence.
        strict.append((candidate["thought_sha256"], read) if valid and read is not None else None)
        shadow.append(
            (candidate["thought_sha256"], event.get("name"), digest(event.get("arguments"))) if valid else None
        )
        turns.append(candidate.get("assistant_turn") if valid else None)

    def select(keys):
        chosen = set()
        for period in range(1, config.max_period + 1):
            i = 0
            while i + period * config.min_copies <= len(events):
                pattern = keys[i : i + period]
                if any(k is None for k in pattern):
                    i += 1
                    continue
                j = i + 1
                while (
                    j < len(events)
                    and keys[j] is not None
                    and keys[j] == pattern[(j - i) % period]
                    and events[j]["id"] == events[j - 1]["id"] + 1
                    and turns[j] == turns[j - 1] + 1
                ):
                    j += 1
                copies = (j - i) // period
                stop = i + copies * period
                start = i + config.protect_copies * period
                surplus = sum(events[k]["thinking_candidate"]["tokens"] for k in range(start, stop))
                if copies >= config.min_copies and surplus >= config.min_surplus:
                    chosen.update(range(start, stop))
                    i = stop
                else:
                    i += 1
        return chosen

    selected = select(strict)
    shadow_selected = select(shadow) - selected
    return [events[i]["thinking_candidate"]["event"] for i in sorted(selected)], {
        "cross_turn_selected": len(selected),
        "cross_turn_shadow": len(shadow_selected),
        "cross_turn_candidates": sum(k is not None for k in shadow),
        "cross_turn_proven_reads": sum(k is not None for k in strict),
    }


def verified_event(event, response, mask):
    """Reject shifted/truncated observations or stale events after template merge."""
    from verl.utils.swe_tool_failure import token_digest

    if not isinstance(event, dict) or type(event.get("version")) is not int or event["version"] != 1:
        return None
    if event.get("detector") not in {"exact_thinking_v2", "exact_thinking_v3"}:
        return None
    s, e = event.get("start"), event.get("end")
    ts, te = event.get("thinking_start"), event.get("thinking_end")
    if not all(type(x) is int for x in (s, e, ts, te)) or not 0 <= s <= ts < te <= e <= len(response):
        return None
    if e > len(mask) or not all(mask[s:e]):
        return None
    if token_digest(response[s:e]) != event.get("token_sha256") or token_digest(response[:e]) != event.get(
        "prefix_sha256"
    ):
        return None
    spans = event.get("spans")
    if not isinstance(spans, list) or not spans:
        return None
    previous = ts
    for span in spans:
        if not isinstance(span, (list, tuple)) or len(span) != 2 or not all(type(x) is int for x in span):
            return None
        a, b = span
        if not previous <= a < b <= te:
            return None
        previous = b
    return spans


def build_cost_weights(
    responses, response_mask, extra_fields, active, *, cost, cap, warmup_steps, step, trajectory_ids=None
):
    """Uniform token mean per flagged turn; capped and deduplicated per session."""
    import math

    import torch

    if not all(math.isfinite(v) and v >= 0 for v in (cost, cap)) or warmup_steps < 0:
        raise ValueError("Invalid thinking-repeat cost, cap, or warmup")
    weights = torch.zeros_like(response_mask, dtype=torch.float32)
    ramp = min(1.0, max(0.0, step / warmup_steps)) if warmup_steps else 1.0
    trajectory_ids = list(range(len(extra_fields))) if trajectory_ids is None else trajectory_ids
    stats = dict(
        events=0,
        applied=0,
        abstained=0,
        cost=0.0,
        active=int(active.sum()),
        generated_turns=0,
        flagged_turns=0,
        alignment_abstentions=0,
        no_thinking=0,
        unflagged=0,
        penalized_tokens=0,
        flagged_rows=0,
        flagged_trajectories=0,
        active_tokens=int(response_mask[active.bool()].sum()),
    )
    sessions, generations = {}, {}
    for row, (ids, mask, extra) in enumerate(
        zip(responses.tolist(), response_mask.tolist(), extra_fields, strict=True)
    ):
        extra = getattr(extra, "data", extra)
        if not isinstance(extra, dict):
            continue
        for key, value in extra.items():
            if key.startswith("thinking_repeat_") and type(value) is int and value >= 0:
                name = key.removeprefix("thinking_repeat_")
                stats[name] = stats.get(name, 0) + value
        events = extra.get("thinking_repeat_events", [])
        if not isinstance(events, list):
            continue
        stats["events"] += len(events)
        if not bool(active[row]):
            continue
        turns = {}
        for event in events:
            spans = verified_event(event, ids, mask)
            identity = (
                (trajectory_ids[row], event.get("start"), event.get("end"), event.get("prefix_sha256"))
                if spans
                else None
            )
            if spans is None:
                stats["abstained"] += 1
                continue
            # Single- and cross-turn detectors can select the same generation.
            # Union token positions BEFORE per-generation cost/cap allocation.
            turns.setdefault(identity, set()).update(i for a, b in spans for i in range(a, b))
        for identity, positions in turns.items():
            if identity in generations:
                stats["abstained"] += 1  # Duplicate generation, not a second cost.
                generations[identity][1].update(positions)
            else:
                generations[identity] = (row, positions)
    for identity, (row, positions) in generations.items():
        ordered = sorted(positions)
        spans = []
        for position in ordered:
            if spans and spans[-1][1] == position:
                spans[-1][1] += 1
            else:
                spans.append([position, position + 1])
        sessions.setdefault(identity[0], []).append((row, spans, len(positions)))
    flagged_rows = set()
    for turns in sessions.values():
        per_turn = min(cost, cap / len(turns)) * ramp
        if per_turn == 0:
            continue
        for row, spans, count in turns:
            for a, b in spans:
                weights[row, a:b] = per_turn / count
            stats["penalized_tokens"] += count
            flagged_rows.add(row)
        stats["applied"] += len(turns)
        stats["cost"] += per_turn * len(turns)
        stats["flagged_trajectories"] += 1
    stats["flagged_rows"] = len(flagged_rows)
    return weights, stats

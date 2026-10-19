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


DEFAULT_CONFIG = Config()


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
    for (fragment, identity), occurrences in groups.items():
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


def capture_generation(tokenizer, agent_data, *, prompt_thinking=False):
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
        return None, "alignment_abstentions"
    if list(agent_data.prompt_ids[-len(ids) :]) != ids:
        return None, "alignment_abstentions"
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None:
        return None, "alignment_abstentions"
    text = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    encoded = backend.encode(text, add_special_tokens=False)
    if encoded.ids != ids:
        return None, "alignment_abstentions"
    offsets = encoded.offsets
    # Native offsets must reconstruct each token without ambiguous byte pieces.
    if any(not 0 <= a < b <= len(text) for a, b in offsets):
        return None, "alignment_abstentions"
    if any(a < pa or b < pb for (pa, pb), (a, b) in zip(offsets, offsets[1:], strict=False)):
        return None, "alignment_abstentions"
    # One leading thinking region only; tool arguments and final output are out.
    opening = re.match(r"\s*<think>", text)
    if opening:
        if prompt_thinking:
            return None, "alignment_abstentions"  # nested protocol ambiguity
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
        return None, "alignment_abstentions"
    # Shared-offset UTF-8 byte tokens must not straddle thinking boundaries.
    if (
        first
        and offsets[first - 1][1] > offsets[first][0]
        or last < len(offsets)
        and offsets[last][0] < offsets[last - 1][1]
    ):
        return None, "alignment_abstentions"
    chunk = text[start:end]
    native = SimpleNamespace(ids=ids[first:last], offsets=[(a - start, b - start) for a, b in offsets[first:last]])
    result = detect(chunk, backend, encoding=native)
    if not result["matched"]:
        return None, "unflagged"
    offset = len(mask) - len(ids)
    spans = [[offset + first + a, offset + first + b] for a, b in result["token_spans"]]
    response = list(agent_data.prompt_ids[-len(mask) :])
    stop = len(mask)
    event = {
        "version": 1,
        "detector": "exact_thinking_v2",
        "start": offset,
        "end": stop,
        "thinking_start": offset + first,
        "thinking_end": offset + last,
        "spans": spans,
        "token_sha256": token_digest(ids),
        "prefix_sha256": token_digest(response[:stop]),
        "rules": sorted({e["rule"] for e in result["evidence"]}),
    }
    if verified_event(event, response, mask) is None:
        return None, "alignment_abstentions"
    return event, "flagged_turns"


def verified_event(event, response, mask):
    """Reject shifted/truncated observations or stale events after template merge."""
    from verl.utils.swe_tool_failure import token_digest

    if not isinstance(event, dict) or type(event.get("version")) is not int or event["version"] != 1:
        return None
    if event.get("detector") != "exact_thinking_v2":
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
    sessions, seen = {}, set()
    for row, (ids, mask, extra) in enumerate(
        zip(responses.tolist(), response_mask.tolist(), extra_fields, strict=True)
    ):
        extra = getattr(extra, "data", extra)
        if not isinstance(extra, dict):
            continue
        for key in ("generated_turns", "flagged_turns", "alignment_abstentions", "no_thinking", "unflagged"):
            value = extra.get("thinking_repeat_" + key, 0)
            if type(value) is int and value >= 0:
                stats[key] += value
        events = extra.get("thinking_repeat_events", [])
        if not isinstance(events, list):
            continue
        stats["events"] += len(events)
        if not bool(active[row]):
            continue
        occupied = set()
        for event in events:
            spans = verified_event(event, ids, mask)
            identity = (
                (trajectory_ids[row], event.get("start"), event.get("end"), event.get("prefix_sha256"))
                if spans
                else None
            )
            positions = {i for a, b in spans for i in range(a, b)} if spans else set()
            if spans is None or identity in seen or occupied & positions:
                stats["abstained"] += 1
                continue
            seen.add(identity)
            occupied.update(positions)
            sessions.setdefault(trajectory_ids[row], []).append((row, spans, len(positions)))
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

"""Default-off tool loops and token-local PPO costs.

No shell purity inference, output-text error classification, or RM dependency.
The first two calls (two cycles for cyclic reads) are protected. A complete
five-call streak (ten calls by default for shell reads) or >=20-call/five-copy
read cycle is required before charging. The separately opted-in Python/test
observed-loop heuristic protects 20 calls; it does not establish program purity.
The independent edit_identical_calls rule charges each authoritative no-op,
including the first; it does not require a repetition episode.
"""

import hashlib
import json

from verl.utils.swe_tool_failure import token_digest, valid_proof

VERSION = 1
RULES = (
    "noop_edits",
    "unchanged_reads",
    "read_cycles",
    "contract_rejections",
    "varied_noop_edits",
    "noop_writes",
    "shell_reads",
    "python_tests",
    "edit_noop_episodes",
    "edit_rejection_episodes",
    "edit_identical_calls",
)


def _proof_digest(proof):
    return hashlib.sha256(json.dumps(proof, sort_keys=True).encode()).hexdigest()


def _keys(event):
    from recipe.swe_agent import repetition_reward as rr

    if (
        not isinstance(event, dict)
        or type(event.get("id")) is not int
        or event["id"] < 1
        or event.get("serial") is not True
        or event.get("feedback_delivered") is not True
        or event.get("infrastructure_failure") is True
    ):
        return {}
    # Historical or malformed metadata is a barrier, never evidence.
    if any(
        not isinstance(event.get(k, {}), dict)
        for k in ("evidence", "noop_edit_evidence", "observation_view")
        if event.get(k) is not None
    ):
        return {}
    read_proof = event.get("evidence") or {}
    if any(
        not isinstance(read_proof.get(k, ""), str)
        for k in ("before", "after", "arguments_sha256", "observation_sha256")
    ):
        return {}
    keys = {}
    from recipe.swe_agent.edit_recovery import RULES as EDIT_RULES, verified_edit_outcome

    edit = verified_edit_outcome(event)
    if edit and edit["family"] in {"identical_replacement", "old_string_missing", "multiple_matches", "file_missing"}:
        rule = EDIT_RULES[0] if edit["family"] == "identical_replacement" else EDIT_RULES[1]
        keys[rule] = (edit["target"], edit["before"], edit["family"], edit["arguments_sha256"],
                      edit["match_count"], edit["observation_sha256"])
        if edit["family"] == "identical_replacement":
            keys["edit_identical_calls"] = keys[rule]
    from recipe.swe_agent.python_test_repeat_evidence import observed_loop_key

    python_key = observed_loop_key(event)
    if python_key is not None:
        keys["python_tests"] = python_key
    from recipe.swe_agent.tool_repeat_evidence import proven_extension_key

    for rule in ("noop_writes", "shell_reads"):
        key = proven_extension_key(event, rule)
        if key is not None:
            keys[rule] = key
    if rr.explicit_noop_edit(event):
        keys["noop_edits"] = rr.digest(event["arguments"])
        proof = event.get("tool_repeat_noop_edit") or {}
        if (
            isinstance(proof, dict)
            and type(proof.get("version")) is int
            and proof["version"] == VERSION
            and isinstance(proof.get("target"), str)
            and proof["target"]
            and proof.get("rejected") is True
        ):
            keys["varied_noop_edits"] = proof["target"]
    read = rr.proven_read_key(event)
    if read is not None:
        keys["unchanged_reads"] = read
    proof = event.get("contract_rejection")
    # An authenticated rejection may be misclassified by the old generic
    # success heuristic. Delivery and executor attribution are checked separately.
    if valid_proof(proof) and event.get("contract_observation_complete") is True:
        keys["contract_rejections"] = (event.get("name"), rr.digest(event.get("arguments")), proof["code"])
    return keys


def detect(events, enabled_rules, *, shell_min_copies=10, python_test_protected_calls=20):
    """Return all eligible event IDs, not representative trajectory citations."""
    if type(shell_min_copies) is not int or shell_min_copies < 5:
        raise ValueError("shell_min_copies must be an integer >=5")
    if type(python_test_protected_calls) is not int or python_test_protected_calls < 20:
        raise ValueError("python_test_protected_calls must be an integer >=20")
    selected = {}
    keys = [_keys(e) for e in events]
    if "edit_identical_calls" in enabled_rules:
        for event, key in zip(events, keys, strict=True):
            if "edit_identical_calls" in key:
                selected[event["id"]] = {"edit_identical_calls"}
    from recipe.swe_agent.edit_recovery import RULES as EDIT_RULES, analyze as analyze_edits

    if set(EDIT_RULES).intersection(enabled_rules):
        edit_selected, _ = analyze_edits(events)
        for event_id, rules in edit_selected.items():
            selected.setdefault(event_id, set()).update(set(rules).intersection(enabled_rules))
        selected = {k: v for k, v in selected.items() if v}
    for rule in (r for r in RULES if r in enabled_rules and r not in ("read_cycles", "edit_identical_calls", *EDIT_RULES)):
        run, previous = [], None

        def flush(current_run, current_rule):
            if current_rule == "python_tests":
                for e in current_run[python_test_protected_calls:]:
                    selected.setdefault(e["id"], set()).add(current_rule)
                return
            if len(current_run) >= (shell_min_copies if current_rule == "shell_reads" else 5):
                for e in current_run[2:]:
                    selected.setdefault(e["id"], set()).add(current_rule)

        for e, k in zip(events, keys, strict=True):
            key = k.get(rule)
            if key is None or key != previous or run and e["id"] != run[-1]["id"] + 1:
                flush(run, rule)
                run = []
            if key is not None:
                run.append(e)
            previous = key
        flush(run, rule)
    if "read_cycles" in enabled_rules:
        for period in (2, 3, 4):
            i = 0
            while i + period <= len(events):
                pattern = [k.get("unchanged_reads") for k in keys[i : i + period]]
                if any(k is None for k in pattern) or len(set(pattern)) <= 1:
                    i += 1
                    continue
                j = i + 1
                while (
                    j < len(events)
                    and events[j]["id"] == events[j - 1]["id"] + 1
                    and keys[j].get("unchanged_reads") == pattern[(j - i) % period]
                ):
                    j += 1
                copies = (j - i) // period
                if copies >= 5 and copies * period >= 20:
                    for e in events[i + 2 * period : i + copies * period]:
                        selected.setdefault(e["id"], set()).add("read_cycles")
                    i = j
                else:
                    i += 1
    return {event_id: sorted(rules) for event_id, rules in selected.items()}


def capture_events(events, enabled_rules, *, shell_min_copies=10, python_test_protected_calls=20):
    """Attach exact generation alignment to every selected call, once."""
    selected = detect(events, enabled_rules, shell_min_copies=shell_min_copies,
                      python_test_protected_calls=python_test_protected_calls)
    captured = []
    for e in events:
        if e["id"] in selected and isinstance(e.get("token_span"), dict):
            proof = {
                "version": VERSION,
                "rules": selected[e["id"]],
                "event_id": e["id"],
                "evidence_sha256": hashlib.sha256(repr(_keys(e)).encode()).hexdigest(),
                "token_sha256": e["token_span"].get("token_sha256"),
                "prefix_sha256": e["token_span"].get("prefix_sha256"),
            }
            captured.append({**e["token_span"], "proof": proof, "proof_sha256": _proof_digest(proof)})
    return captured, len(selected)


def verified_span(event, response, mask):
    if not isinstance(event, dict) or type(event.get("version")) is not int or event["version"] != VERSION:
        return None
    proof = event.get("proof") or {}
    if (
        not isinstance(proof, dict)
        or type(proof.get("version")) is not int
        or proof["version"] != VERSION
        or event.get("proof_sha256") != _proof_digest(proof)
        or not proof.get("rules")
        or not isinstance(proof["rules"], list)
        or any(r not in RULES for r in proof["rules"])
        or type(proof.get("event_id")) is not int
        or proof["event_id"] < 1
        or not isinstance(proof.get("evidence_sha256"), str)
        or len(proof["evidence_sha256"]) != 64
    ):
        return None
    if any(proof.get(k) != event.get(k) for k in ("token_sha256", "prefix_sha256")):
        return None
    start, end = event.get("start"), event.get("end")
    if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(response):
        return None
    if end > len(mask) or not all(mask[start:end]):
        return None
    if token_digest(response[start:end]) != event.get("token_sha256") or token_digest(response[:end]) != event.get(
        "prefix_sha256"
    ):
        return None
    return start, end


def build_cost_weights(*args, enabled_rules=None, edit_cost=0.01, **kwargs):
    from verl.utils.swe_tool_failure import build_cost_weights as build

    enabled_rules = set(RULES if enabled_rules is None else enabled_rules)
    from recipe.swe_agent.edit_recovery import RULES as EDIT_RULES

    def cost_for_event(event):
        # A call matching an enabled new Edit rule is charged once at its new
        # cost, even when an older repetition label is also present. Existing
        # tool-failure weights retain precedence through existing_weights.
        active = enabled_rules.intersection(event["proof"]["rules"])
        return edit_cost if active.intersection((*EDIT_RULES, "edit_identical_calls")) else kwargs["cost"]

    def verify_enabled(event, response, mask):
        span = verified_span(event, response, mask)
        if span is not None and enabled_rules.intersection(event["proof"]["rules"]):
            return span
        return None

    return build(
        *args, event_key="tool_repeat_events", verifier=verify_enabled, diagnostic_prefix="tool_repeat_",
        event_cost=cost_for_event, **kwargs
    )


def _python_test_call(event):
    """Compatibility entry point for the independent diagnostic classifier."""
    from recipe.swe_agent.python_test_repeat_evidence import is_python_test_call

    return is_python_test_call(event)


def diagnostics(events, *, shadow=False, python_test_shadow=False, edit_shadow=False, command_shadow=False,
                search_shadow=False):
    """Numeric eligible/abstention counts; shadow candidates never create events."""
    from collections import Counter

    from recipe.swe_agent.repetition_reward import digest

    out = Counter()
    if search_shadow:
        from recipe.swe_agent.search_outcomes import diagnostics as search_diagnostics

        out.update({"tool_repeat_shadow_search_" + k: v for k, v in search_diagnostics(events).items()})
    if edit_shadow:
        from recipe.swe_agent.edit_recovery import analyze

        _, edit_counts = analyze(events)
        out.update({"tool_repeat_shadow_edit_" + k: v for k, v in edit_counts.items()})
    if command_shadow:
        from recipe.swe_agent.command_recovery import diagnostics as command_diagnostics

        out.update({"tool_repeat_shadow_command_" + k: v for k, v in command_diagnostics(events).items()})
    for event in events:
        keys = _keys(event)
        for rule in RULES:
            if rule != "read_cycles":
                out["tool_repeat_eligible_" + rule] += rule in keys
        reason = None
        if event.get("serial") is not True:
            reason = "nonserial"
        elif event.get("feedback_delivered") is not True:
            reason = "undelivered"
        elif event.get("infrastructure_failure") is True:
            reason = "infrastructure"
        elif not keys:
            field = (
                "repeat_python_test_evidence"
                if event.get("repeat_python_test_evidence") is not None
                else
                "repeat_shell_read_evidence"
                if event.get("name") in {"Bash", "run_shell"}
                else "tool_repeat_noop_write"
                if event.get("name") in {"Write", "write_file"}
                else "evidence"
            )
            proof = event.get(field)
            if not isinstance(proof, dict):
                reason = "proof_missing"
            elif proof.get("unavailable_reason") == "disabled_on_execution_service":
                reason = "proof_disabled"
            elif proof.get("before") and proof.get("before") != proof.get("after"):
                reason = "state_changed"
            elif proof.get("unavailable_reason") == "unsupported_or_incomplete":
                reason = "unsupported_or_incomplete_scope"
            elif proof.get("complete") is False:
                reason = "proof_incomplete"
            else:
                reason = "unsupported_or_unverified"
        if reason:
            out["tool_repeat_abstained_" + reason] += 1
    if shadow or python_test_shadow:
        if python_test_shadow:
            out["tool_repeat_shadow_python_test_streaks"] = 0
            out["tool_repeat_shadow_python_test_calls"] = 0
        previous, run = None, []

        def flush():
            if len(run) >= 20:
                if shadow:
                    out["tool_repeat_shadow_streaks"] += 1
                    out["tool_repeat_shadow_calls"] += len(run)
                if python_test_shadow and _python_test_call(run[0]):
                    out["tool_repeat_shadow_python_test_streaks"] += 1
                    out["tool_repeat_shadow_python_test_calls"] += len(run)

        for event in events:
            valid = (
                event.get("serial") is True
                and event.get("feedback_delivered") is True
                and event.get("infrastructure_failure") is not True
            )
            key = (event.get("name"), digest(event.get("arguments"))) if valid else None
            if key is None or key != previous or run and event["id"] != run[-1]["id"] + 1:
                flush()
                run = []
            if key is not None:
                run.append(event)
            previous = key
        flush()
    return dict(out)

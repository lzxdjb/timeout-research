"""Read-only Edit decisions and full-strength counterfactual cost accounting.

No reward, advantage, log probability, optimizer or model is accessed here.
Training application remains exclusively in the existing trainer loss path.
"""

import hashlib
import json

AUDIT_LIMIT = 2 * 1024 * 1024


def _alignment_diagnostics(event, ids, mask):
    from verl.utils.swe_tool_failure import token_digest

    span = event.get("token_span")
    if not isinstance(span, dict):
        return dict(reason="missing_generation_span")
    result = {key: span.get(key) for key in ("start", "end", "token_sha256", "prefix_sha256")}
    start, end = span.get("start"), span.get("end")
    if type(span.get("version")) is not int or span["version"] != 1:
        reason = "unsupported_span_version"
    elif type(start) is not int or type(end) is not int or not 0 <= start < end:
        reason = "invalid_span_bounds"
    elif end > len(ids) or end > len(mask):
        reason = "span_outside_retained_response"
    elif not all(mask[start:end]):
        reason = "span_contains_observation_tokens"
    elif token_digest(ids[start:end]) != span.get("token_sha256"):
        reason = "call_token_identity_mismatch"
    elif token_digest(ids[:end]) != span.get("prefix_sha256"):
        reason = "response_prefix_identity_mismatch"
    else:
        reason = "verified_generation_span"
    if type(start) is int and type(end) is int and start < end:
        result["candidate_token_count"] = end - start
    return dict(result, reason=reason)


def bounded_audit(header, rows, *, max_bytes=AUDIT_LIMIT):
    """Keep valid JSON and a useful prefix even when a long audit exceeds bounds."""
    encode = lambda value: json.dumps(value, separators=(",", ":"), ensure_ascii=False)
    full_hash = hashlib.sha256(encode(rows).encode()).hexdigest()
    base = dict(header, events_total=len(rows), events_sha256=full_hash)

    def pack(n):
        return encode(dict(base, events_exported=n, events_omitted=len(rows) - n, events=rows[:n]))

    encoded = pack(len(rows))
    if len(encoded.encode()) > max_bytes:
        lo, hi = 0, len(rows)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if len(pack(mid).encode()) <= max_bytes:
                lo = mid
            else:
                hi = mid - 1
        encoded = pack(lo)
        if len(encoded.encode()) > max_bytes:
            return {"edit_diagnostics_audit_error": "size_limit", "edit_diagnostics_events_omitted": len(rows)}
    omitted = len(rows) - len(json.loads(encoded)["events"])
    return dict(
        edit_diagnostics_audit_json=encoded,
        edit_diagnostics_audit_sha256=hashlib.sha256(encoded.encode()).hexdigest(),
        edit_diagnostics_events_omitted=omitted,
    )


def collect_edit_diagnostics(events, output, actor_config, *, validation):
    """Evaluate configured rules as if the sample were active, at full strength.

    Validation eligibility is not training eligibility. This projection does not
    pretend to reproduce completion-ratio filtering or incomplete sibling groups.
    Both master-enabled and diagnostic-only collection use the same rule flags.
    """
    import torch
    from recipe.swe_agent.edit_recovery import PENALTY_RULES, analyze_rejection_causes

    from verl.utils.swe_tool_failure import build_cost_weights as failure_weights
    from verl.utils.swe_tool_repeat import RULES, build_cost_weights, capture_events, verified_span

    _, counts, rows = analyze_rejection_causes(events)
    enabled = {r for r in RULES if actor_config.get("tool_repeat_penalty_" + r, False)}
    edit_rules = {*PENALTY_RULES, "edit_identical_calls"}
    captured, selected_count = capture_events(
        events,
        enabled,
        shell_min_copies=actor_config.get("tool_repeat_penalty_shell_min_copies", 10),
        python_test_protected_calls=actor_config.get("tool_repeat_penalty_python_test_protected_calls", 20),
        command_window=actor_config.get("tool_repeat_penalty_command_window", 16),
        command_protected_calls=actor_config.get("tool_repeat_penalty_command_protected_calls", 2),
    )
    ids, mask = output.response_ids, output.response_mask
    retained = [e for e in captured if verified_span(e, ids, mask) is not None]
    aligned = {e["proof"]["event_id"]: e for e in retained}
    responses, response_mask = torch.tensor([ids], dtype=torch.long), torch.tensor([mask], dtype=torch.bool)
    active = torch.tensor([True])
    failure_enabled = actor_config.get("tool_failure_penalty_enabled", False)
    repeat_cap = actor_config.get("tool_repeat_penalty_cap", 0.10)
    cap = min(repeat_cap, actor_config.get("tool_failure_penalty_cap", 0.10)) if failure_enabled else repeat_cap
    failure, _ = failure_weights(
        responses,
        response_mask,
        [{"tool_failure_events": output.extra_fields.get("tool_failure_events", []) if failure_enabled else []}],
        active,
        cost=actor_config.get("tool_failure_penalty_cost", 0.02),
        cap=cap,
        warmup_steps=0,
        step=1,
    )
    # Only this private dictionary receives the application-audit side effect.
    private = {"tool_repeat_events": retained}
    _, accounting = build_cost_weights(
        responses,
        response_mask,
        [private],
        active,
        enabled_rules=enabled,
        cost=actor_config.get("tool_repeat_penalty_cost", 0.02),
        edit_cost=actor_config.get("tool_repeat_penalty_edit_cost", 0.01),
        command_cost=actor_config.get("tool_repeat_penalty_command_cost", 0.01),
        cap=cap,
        existing_weights=failure,
        warmup_steps=0,
        step=1,
    )
    applications = {e["event_id"]: e for e in json.loads(private["tool_repeat_application_audit_json"])["events"]}
    metrics = {"edit_diagnostics_" + k: v for k, v in counts.items()}
    metrics.update(
        edit_diagnostics_enabled=1,
        edit_diagnostics_edit_calls=0,
        edit_diagnostics_selected_calls=0,
        edit_diagnostics_retained_calls=0,
        edit_diagnostics_would_apply_calls=0,
        edit_diagnostics_would_cost=0.0,
        edit_diagnostics_alignment_abstentions=0,
    )
    original = {event.get("id"): event for event in events if isinstance(event, dict)}
    for row in rows:
        if row["name"] not in {"Edit", "edit_file"}:
            continue
        metrics["edit_diagnostics_edit_calls"] += 1
        row["alignment"] = _alignment_diagnostics(original.get(row["id"], {}), ids, mask)
        labels = set(row["selected_rules"])
        if row.get("family") == "identical_replacement":
            labels.add("edit_identical_calls")
        row["selected_rules"] = sorted(labels)
        row["enabled_selected_rules"] = sorted(labels & enabled)
        projected = applications.get(row["id"])
        span = aligned.get(row["id"])
        if span:
            row.update(
                start=span["start"],
                end=span["end"],
                token_count=span["end"] - span["start"],
                token_sha256=span["token_sha256"],
                prefix_sha256=span["prefix_sha256"],
            )
        if row["enabled_selected_rules"]:
            metrics["edit_diagnostics_selected_calls"] += 1
            metrics["edit_diagnostics_retained_calls"] += span is not None
            metrics["edit_diagnostics_alignment_abstentions"] += span is None
        if projected:
            row["cost_projection"] = dict(
                would_apply=projected["applied"],
                would_cost=projected["cost"],
                requested_cost=projected["requested_cost"],
                reason=projected["reason"],
                cap_scale=projected["cap_scale"],
                charged_rule=projected["charged_rule"],
            )
            if projected["charged_rule"] in edit_rules:
                metrics["edit_diagnostics_would_apply_calls"] += projected["applied"]
                metrics["edit_diagnostics_would_cost"] += projected["cost"]
        else:
            row["cost_projection"] = dict(
                would_apply=False,
                would_cost=0.0,
                reason=(
                    "alignment_abstention"
                    if row["enabled_selected_rules"]
                    else "rule_disabled"
                    if labels
                    else "not_selected"
                ),
            )
    header = dict(
        version=1,
        stage="validation_counterfactual" if validation else "rollout_counterfactual",
        projection_assumes_active_sample=True,
        projection_warmup="full_strength",
        training_eligibility="not_evaluated",
        enabled_rules=sorted(enabled),
        repeat_master_enabled=bool(actor_config.get("tool_repeat_penalty_enabled", False)),
        effective_cap=cap,
        failure_reserved_cost=float(failure.sum()),
        total_repeat_would_cost=accounting["cost"],
        total_repeat_selected_calls=selected_count,
        total_repeat_retained_calls=len(retained),
        rule_costs={k: v for k, v in accounting.items() if k.startswith("rules/")},
        edit_cost=actor_config.get("tool_repeat_penalty_edit_cost", 0.01),
    )
    if validation:
        header["actual_optimization_cost"] = 0.0
        metrics["edit_diagnostics_actual_optimization_cost"] = 0.0
    metrics.update(bounded_audit(header, rows))
    return metrics

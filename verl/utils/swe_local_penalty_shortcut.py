"""Route verified local costs around repeat judging; never invent token evidence."""

import math


def verified_local_penalties(output, actor_config):
    """Return supported event counts, independent of warmup's current multiplier.

    Costs remain the trainer's responsibility, including exclusions and dedup.
    The experimental python_tests heuristic participates only with its separate
    rule flag enabled and a captured aligned event, never from shadow counters.
    A zero cost/cap, disabled rule or stale span must never suppress the judge.
    """
    from verl.utils.swe_thinking_repeat import verified_event
    from verl.utils.swe_tool_failure import verified_span as verify_failure
    from verl.utils.swe_tool_repeat import RULES
    from verl.utils.swe_tool_repeat import verified_span as verify_repeat

    response, mask = output.response_ids, output.response_mask
    extra = output.extra_fields
    found = {}
    for name, verifier in (("tool_failure", verify_failure), ("tool_repeat", verify_repeat),
                           ("thinking_repeat", verified_event)):
        if not actor_config.get(name + "_penalty_enabled", False):
            continue
        values = [actor_config.get(name + "_penalty_cap", .10)]
        if name != "tool_repeat":
            values.append(actor_config.get(name + "_penalty_cost", .02))
        if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in values):
            continue
        rules = {r for r in RULES if actor_config.get("tool_repeat_penalty_" + r, False)}
        events = extra.get(name + "_events", [])
        if not isinstance(events, list):
            continue
        count = 0
        for event in events:
            if verifier(event, response, mask) is None:
                continue
            if name == "tool_repeat" and not rules.intersection(event["proof"]["rules"]):
                continue
            if name == "tool_repeat":
                from recipe.swe_agent.edit_recovery import RULES as EDIT_RULES

                if rules.intersection(event["proof"]["rules"]).intersection((*EDIT_RULES, "edit_identical_calls")):
                    edit_cost = actor_config.get("tool_repeat_penalty_edit_cost", .01)
                else:
                    edit_cost = actor_config.get("tool_repeat_penalty_cost", .02)
                if not isinstance(edit_cost, (int, float)) or not math.isfinite(edit_cost) or edit_cost <= 0:
                    continue
            count += 1
        if count:
            found[name] = count
    return found

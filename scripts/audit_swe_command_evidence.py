#!/usr/bin/env python3
"""Read-only audit of saved command excerpts. Never executes trajectory commands.

The observations are classified for review; old logs lack v1 executor evidence
and therefore cannot authorize new penalties. CPU regression tests separately
reproduce the supported failures through the real executor.
"""

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from recipe.swe_agent.command_evidence import classify
from recipe.swe_agent.python_test_repeat_evidence import penalty_candidate


def verify_sources(cards, root):
    from scripts.make_swe_sft_data import _parse_output, TOOL_BLOCK_RE, PARAM_RE, TOOL_RESPONSE_RE

    grouped = defaultdict(lambda: defaultdict(list))
    for card in cards:
        grouped[card["source"]][card["line"]].append(card)
    verified = 0
    for source, wanted in grouped.items():
        with (root / source).open() as stream:
            for line_number, line in enumerate(stream, 1):
                if line_number not in wanted:
                    continue
                row = json.loads(line)
                calls, pending = [], []
                for message in _parse_output(row["output"]):
                    if message["role"] == "assistant":
                        pending.extend((name, {k: v.strip() for k, v in PARAM_RE.findall(body)})
                                       for name, body in TOOL_BLOCK_RE.findall(message["content"]))
                    else:
                        results = TOOL_RESPONSE_RE.findall(message["content"])
                        if len(results) == len(pending):
                            calls.extend((name, args, result) for (name, args), result in zip(pending, results))
                        pending = []
                for card in wanted[line_number]:
                    name, arguments, response = calls[card["turn"] - 1]
                    if (row["task_id"] != card["task"] or name != card["name"]
                            or arguments != card["arguments"] or response != card["response"]):
                        raise ValueError(f"Excerpt mismatch: {source}:{line_number} turn {card['turn']}")
                    verified += 1
                if line_number >= max(wanted):
                    break
    if verified != len(cards):
        raise ValueError("Missing source rows")
    return verified


def audit(cards):
    findings, counts = [], Counter()
    for card in cards:
        excerpt = json.dumps({k: card[k] for k in ("arguments", "response")}, sort_keys=True)
        if hashlib.sha256(excerpt.encode()).hexdigest() != card["excerpt_sha256"]:
            raise ValueError("Fixture content identity changed")
        match = re.match(r"\s*exit_code: (\d+)\n", card["response"])
        if match is None:
            raise ValueError("Expected saved executor status")
        output = card["response"][match.end():].split("\n[test outcome]")[0]
        result = classify(card["arguments"]["command"], output, int(match[1]))
        counts[result["reason"] or "diagnostic_only"] += 1
        findings.append({**{k: card[k] for k in ("source", "line", "task", "turn", "excerpt_sha256")},
                         "classification": result["reason"] or "diagnostic_only",
                         "test_outcome": result["test"]["test_outcome"],
                         "tests_executed": result["test"]["tests_executed"],
                         "unused_imports": result["unused_imports"],
                         "legacy_python_command_grammar": penalty_candidate(card["name"], card["arguments"]),
                         "new_penalty_authorized_from_history": False})
    return {"fixture_count": len(cards), "classification_counts": dict(counts), "findings": findings}


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path,
                        default=root / "tests/trainer/ppo/fixtures/swe_command_audit_20261027.json")
    parser.add_argument("--verify-sources", action="store_true", help="Compare excerpts with original rollout files")
    args = parser.parse_args()
    cards = json.loads(args.fixtures.read_text())
    report = audit(cards)
    report["source_excerpts_verified"] = verify_sources(cards, root) if args.verify_sources else None
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

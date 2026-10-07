"""Freeze evidence-backed SWE repeat fixtures and evaluate without training/GPU use.

The evidence rules deliberately do NOT import the detector. Unknown shell state,
ambiguous transcripts and unclassified repeats remain unknown, never clean labels.
Run ``build`` once into a new directory, then ``evaluate`` against frozen labels.
"""

from __future__ import annotations

import argparse
import collections
import functools
import gzip
import hashlib
import importlib
import itertools
import json
import re
import shlex
import sys
from pathlib import Path
from typing import Any

VERSION = "swe-repeat-evidence-v1"
ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    "grpo": "repete_tool_penalty_short_response_mutation_fix_bug",
    "ppo_labelled": "repete_tool_penalty_short_response_mutation_fix_bug_ppo",
    "sft": "sft_test_more_data",
}
ANCHORS = {
    ("grpo", "rollout_data", 1, 39): "confirmed_legitimate_pytest_reruns",
    ("grpo", "rollout_data", 1, 38): "failed_edits_are_not_progress",
    ("grpo", "rollout_data", 1, 252): "timeout_reward_mask_anomaly",
    ("grpo", "validation_data", 12, 40): "missing_rg_loop_raw_success",
    ("ppo_labelled", "rollout_data", 1, 11): "no_op_edit_is_not_progress",
    ("ppo_labelled", "validation_data", 10, 89): "repeated_vim_write",
    ("sft", "validation_data", 12, 31): "filename_search_loop",
}
FUNCTION = re.compile(r"<function=([^>\n]+)>(.*?)</function>", re.S)
PARAMETER = re.compile(r"<parameter=([^>\n]+)>\s*(.*?)\s*</parameter>", re.S)
RESPONSE = re.compile(r"<tool_response>(.*?)</tool_response>", re.S)
ROLE = re.compile(r"(?m)^(assistant|user)\n")
READS = {"Read", "read_file"}
SEARCHES = {"Grep", "Glob", "search_text", "search_files"}
EDITS = {"Edit", "edit_file", "edit"}
WRITES = {"Write", "write_file", "write"}
SHELLS = {"Bash", "bash", "run_shell"}
TRANSPORT = ("Remote execution error:", "Error executing tool", "Error when executing tool:")
NO_CHANGE = ("Edit failed:", "Edit made no changes:")


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint(event: dict) -> str:
    args = dict(event["arguments"])
    if event["name"] in SHELLS and isinstance(args.get("command"), str):
        args["command"] = args["command"].strip()
    return digest([event["name"], args])


def parse_transcript(output: str) -> tuple[list[dict], list[dict], list[str]]:
    """Pair dispatched prefixes only; preserve uncertain/truncated protocol status.

    Role-like text inside tool observations or function arguments is data. A
    terminal assistant call without a response is not assumed to have executed.
    """
    if not output.strip():
        return [], [], ["empty_output"]
    if output.count("<tool_response>") != output.count("</tool_response>"):
        return [], [], ["unbalanced_tool_response"]
    protected = sorted((m.start(), m.end()) for rx in (RESPONSE, FUNCTION) for m in rx.finditer(output))
    markers = []
    for match in ROLE.finditer(output):
        if not any(start <= match.start() < end for start, end in protected):
            markers.append(match)
    messages = []
    first = output[: markers[0].start() if markers else len(output)].strip()
    if first:
        messages.append({"role": "assistant", "content": first})
    for index, marker in enumerate(markers):
        end = markers[index + 1].start() if index + 1 < len(markers) else len(output)
        messages.append({"role": marker[1], "content": output[marker.end():end].strip()})
    if not messages or messages[0]["role"] != "assistant":
        return [], messages, ["missing_first_assistant"]
    if any(a["role"] == b["role"] for a, b in zip(messages, messages[1:])):
        return [], messages, ["nonalternating_roles"]
    events, issues = [], []
    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        calls = list(FUNCTION.finditer(message["content"]))
        responses = []
        if index + 1 < len(messages):
            responses = list(RESPONSE.finditer(messages[index + 1]["content"]))
        if len(responses) > len(calls):
            return [], messages, ["more_responses_than_calls"]
        if len(calls) > len(responses):
            issues.append("undispatched_or_unobserved_calls")
        for block, response in zip(calls, responses):
            pairs = PARAMETER.findall(block[2])
            if len({key for key, _ in pairs}) != len(pairs):
                return [], messages, ["duplicate_parameter"]
            events.append({
                "index": len(events), "batch": index, "name": block[1].strip(),
                "arguments": dict(pairs), "response": response[1].strip(),
            })
    if messages[-1]["role"] != "assistant" or not messages[-1]["content"]:
        issues.append("incomplete_final_turn")
    return events, messages, sorted(set(issues))


def target(event: dict) -> str | None:
    args = event["arguments"]
    return args.get("file_path") or args.get("path")


def shell_words(command: str) -> list[str]:
    try:
        return shlex.split(command)
    except ValueError:
        return []


def test_command(command: str) -> bool:
    # An allowlist of actual runner invocations, not a search for words inside
    # echo strings, comments, Python source, or arbitrary mutating shell code.
    text = re.sub(r"\s+2>&1\b", "", command.strip())
    if any(token in text for token in (";", "&&", "||", "`", "$(", "\n", ">")):
        return False
    parts = text.split("|")
    for part in parts[1:]:
        words = shell_words(part)
        if not words or words[0] not in {"head", "tail", "grep"}:
            return False
    words = shell_words(parts[0])
    if not words:
        return False
    executable = words[0].rsplit("/", 1)[-1]
    return (
        executable in {"pytest", "phpunit", "ctest"}
        or executable in {"python", "python3"} and words[1:3] in (["-m", "pytest"], ["-m", "unittest"])
        or executable in {"go", "cargo", "npm", "yarn", "mvn"} and words[1:2] == ["test"]
    )


# This predicate is part of the evidence annotator, not a pytest test function.
test_command.__test__ = False


def read_only_shell(command: str) -> bool:
    if any(token in command for token in (";", "&&", "||", "`", "$(", "\n", ">", "|")):
        return False
    words = shell_words(command)
    if not words:
        return False
    executable = words[0].rsplit("/", 1)[-1]
    return executable in {"cat", "head", "tail", "grep", "rg", "ls", "pwd", "wc", "stat"} or (
        executable == "sed" and "-n" in words and not any(w.startswith("-i") for w in words)
    )


def effect(event: dict) -> str:
    """Evidence classes, independent of the production success classifier."""
    name, response = event["name"], event["response"]
    if name in EDITS:
        if response.startswith(NO_CHANGE):
            return "no_change"
        match = re.fullmatch(r"Edited (.+) \(([1-9][0-9]*) replacements?\)\.", response)
        if match and match[1] == target(event) and event["arguments"].get("old_string") != event["arguments"].get("new_string"):
            return "changed"
        return "unknown"
    if name in WRITES:
        # A successful write does not establish that bytes changed.
        return "unknown"
    if name in READS | SEARCHES | {"repo_status"}:
        return "read_only"
    if name == "run_tests" or name in SHELLS and test_command(event["arguments"].get("command", "")):
        return "test"
    if name in SHELLS and read_only_shell(event["arguments"].get("command", "")):
        return "read_only"
    return "unknown"


def relevant_test_change(event: dict, changed: list[dict]) -> bool:
    if event["name"] == "run_tests":
        return False  # target semantics vary between harness versions
    command = event["arguments"].get("command", "")
    if not test_command(command):
        return False
    words = shell_words(command.split("|")[0])
    # Whole-project suites, or an explicitly named test for the changed module.
    bare = words in (["pytest"], ["npm", "test"], ["yarn", "test"], ["cargo", "test"], ["python", "-m", "pytest"], ["python3", "-m", "pytest"])
    if bare:
        return True
    return any(
        re.search(r"(?:test_" + re.escape(Path(target(e) or "").stem) + r"\b|\b" + re.escape(Path(target(e) or "").stem) + r"_test\b)", command)
        for e in changed if target(e)
    )


def annotate(events: list[dict]) -> tuple[list[dict], list[dict]]:
    """Produce partial, evidence-backed labels. Never treat unknown as allowed."""
    # Parsed transcripts already carry stable indexes/batches, while the
    # policy contracts intentionally describe only tool events. Normalize the
    # latter here so missing metadata means sequential dispatches, while an
    # explicit batch (including batch=0) still represents in-flight calls.
    events = [
        {
            **event,
            "index": event.get("index", index),
            "batch": event.get("batch", index),
        }
        for index, event in enumerate(events)
    ]
    history, labels, observations = {}, [], []
    for event in events:
        index = event["index"]
        kind = effect(event)
        if kind in {"changed", "no_change"}:
            observations.append({"event": index, "category": "real_edit" if kind == "changed" else "failed_or_noop_edit", "expected_progress": kind == "changed", "evidence": event["response"]})
        if event["name"] in SHELLS and kind == "unknown":
            observations.append({"event": index, "category": "unknown_shell_state", "expected_progress": None})
        key = fingerprint(event)
        prior = history.get(key)
        history[key] = event
        if prior is None:
            continue
        interval = events[prior["index"] + 1:index]
        changes = [e for e in interval if effect(e) == "changed"]
        unknown = any(effect(e) == "unknown" for e in interval)
        if event["name"] in READS | SEARCHES:
            unknown = unknown or any(effect(e) == "test" for e in interval)
        judgment, category = "unknown", "insufficient_state"
        if prior["batch"] == event["batch"]:
            judgment, category = "allow", "inflight_duplicate"
        elif prior["response"].startswith(TRANSPORT):
            older = [e for e in events[:prior["index"]] if fingerprint(e) == key]
            if older and older[-1]["response"].startswith(TRANSPORT):
                judgment, category = "penalize", "second_failed_retry"
            else:
                judgment, category = "allow", "first_transport_retry"
        elif prior["response"].startswith("Search unavailable: rg is not installed.") and not interval:
            # Two consecutive permanent capability failures cannot be cured by
            # invoking the same unavailable search again.
            judgment, category = "penalize", "unavailable_search_loop"
        elif prior["response"].startswith("Search unavailable:"):
            judgment, category = "unknown", "capability_state_unknown"
        elif event["name"] in READS and not unknown:
            same_file_changes = [e for e in changes if target(e) == target(event)]
            judgment = "allow" if same_file_changes else "penalize"
            category = "reread_after_edit" if same_file_changes else ("reread_after_unrelated_edit" if changes else "unchanged_read")
        elif event["name"] in SEARCHES and not unknown and not changes:
            judgment, category = "penalize", "unchanged_search"
        elif kind == "test" and not unknown:
            if relevant_test_change(event, changes):
                judgment, category = "allow", "retest_after_edit"
            elif not changes:
                judgment, category = "penalize", "unchanged_test"
        elif event["name"] in EDITS and event["arguments"].get("old_string") == event["arguments"].get("new_string"):
            judgment, category = "penalize", "repeated_noop_edit"
        if judgment == "penalize" and index - prior["index"] > 16:
            observations.append({"event": index, "category": "long_distance_repeat"})
        labels.append({
            "event": index, "prior": prior["index"], "expected": judgment,
            "category": category, "changed_events": [e["index"] for e in changes],
            "unknown_intervening_state": unknown, "label_source": VERSION,
        })
    return labels, observations


def identity(row: dict) -> tuple[str, str, str]:
    gts = row.get("gts") or {}
    if isinstance(gts, str):
        try:
            gts = json.loads(gts)
        except ValueError:
            gts = {}
    task = str(gts.get("instance_id") or gts.get("task_id") or row.get("task_id") or "unknown")
    repo = str(gts.get("repo") or gts.get("repository") or "unknown")
    benchmark = str(gts.get("benchmark") or "unknown")
    return benchmark, task, repo


def partition(task_key: str) -> str:
    return "holdout" if int(digest(task_key)[:8], 16) % 10 < 3 else "development"


def make_case(row: dict, provenance: dict) -> dict:
    events, messages, issues = parse_transcript(row.get("output") or "")
    labels, observations = annotate(events)
    benchmark, task, repo = identity(row)
    task_key = f"{repo}\x1f{task}"
    categories = {a["category"] for a in labels if a["expected"] != "unknown"}
    categories.update(o["category"] for o in observations)
    if issues:
        categories.add("transcript_issues")
    if row.get("train_sample_mask") is True and row.get("trajectory_penalized_repeated_tool_calls", 0) > 0 and row.get("shaped_score") != -0.1:
        categories.add("reward_mask_anomaly")
    if any(row.get(k) for k in ("trajectory_timeout", "trajectory_terminal_tool_failure", "completion_ratio_cutoff")):
        categories.add("exceptional_exit")
    unknowns = [a["event"] for a in labels if a["expected"] == "unknown"]
    penalties = [a["event"] for a in labels if a["expected"] == "penalize"]
    first = min(penalties) if penalties else None
    return {
        "id": digest([provenance["path"], provenance["line"], digest(row)])[:20],
        "source": {**provenance, "row_sha256": digest(row)},
        "task_key": task_key, "task": task, "repository": repo, "benchmark": benchmark,
        "partition": partition(task_key), "categories": sorted(categories),
        "labels": labels, "observations": observations, "events": events,
        "parse_issues": issues, "unknown_repeat_events": unknowns,
        "first_offense": first, "first_offense_exact": first is not None and not issues and not any(i < first for i in unknowns),
        "fully_labelled_repeats": not issues and not unknowns,
        "checkpoint_window": "early" if provenance["step"] <= 4 else "late" if provenance["step"] >= 10 else "middle",
        "raw": row,
    }


def select_cases(cases: list[dict], count: int) -> list[dict]:
    """Deterministic greedy coverage with task caps, then run/split/outcome balance."""
    selected, used, tasks = [], set(), collections.Counter()
    coverage = collections.Counter()

    def add(case: dict) -> None:
        selected.append(case)
        used.add(case["id"])
        tasks[case["task_key"]] += 1
        coverage.update(case["categories"])

    for case in cases:
        s = case["source"]
        if (s["run"], s["split"], s["step"], s["line"]) in ANCHORS:
            add(case)
    for _ in range(max(0, count - len(selected))):
        strata = collections.Counter((c["source"]["run"], c["source"]["split"], c["partition"], c["raw"].get("raw_score"), c["checkpoint_window"]) for c in selected)
        candidates = [c for c in cases if c["id"] not in used and tasks[c["task_key"]] < 2]
        if not candidates:
            break
        def score(c: dict) -> tuple:
            coverage_gain = sum(1 / (1 + coverage[k]) ** 2 for k in c["categories"])
            stratum = (c["source"]["run"], c["source"]["split"], c["partition"], c["raw"].get("raw_score"), c["checkpoint_window"])
            return coverage_gain + 1 / (1 + strata[stratum]), -tasks[c["task_key"]], c["id"]
        add(max(candidates, key=score))
    return sorted(selected, key=lambda c: c["id"])


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


@functools.lru_cache(maxsize=1)
def load_detector():
    sys.path.insert(0, str(ROOT.parent / "stock-rl-reflect"))
    return importlib.import_module("recipe.swe_agent.repeated_tool")


def replay_detector(events: list[dict], module=None) -> dict:
    module = module or load_detector()
    detector = module.RepeatedToolDetector()
    results = []
    for _, batch in itertools.groupby(events, key=lambda e: e["batch"]):
        pending = []
        for event in batch:
            f, entry, hit = detector.begin(event["name"], dict(event["arguments"]))
            results.append({"event": event["index"], "penalized": bool(hit), "progress_delta": 0})
            pending.append((event, f, entry, len(results) - 1))
        for event, f, entry, result_index in pending:
            before = detector.workspace_progress
            detector.finish(
                event["name"],
                dict(event["arguments"]),
                f,
                entry,
                module.classify_tool_outcome(event["name"], dict(event["arguments"]), event["response"]),
            )
            results[result_index]["progress_delta"] = detector.workspace_progress - before
    return {"count": detector.penalized_repeats, "events": results}


def read_fixtures(directory: Path) -> tuple[dict, list[dict]]:
    manifest = json.loads((directory / "manifest.json").read_text())
    fixture_path = directory / "fixtures.jsonl.gz"
    if file_digest(fixture_path) != manifest["fixtures_sha256"]:
        raise ValueError("Frozen fixture checksum mismatch; create a new version, do not relabel in place")
    if file_digest(directory / "contracts.json") != manifest["contracts_sha256"]:
        raise ValueError("Frozen contract checksum mismatch")
    with gzip.open(fixture_path, "rt", encoding="utf-8") as stream:
        cases = [json.loads(line) for line in stream]
    for case in cases:
        if digest(case["raw"]) != case["source"]["row_sha256"]:
            raise ValueError(f"Source row checksum mismatch: {case['id']}")
    return manifest, cases


def evaluate(directory: Path, output: Path) -> dict:
    manifest, cases = read_fixtures(directory)
    module = load_detector()
    counts, failures = collections.Counter(), []
    per_partition = {k: collections.Counter() for k in ("development", "holdout")}
    for case in cases:
        result = replay_detector(case["events"], module)
        actual = {x["event"]: x for x in result["events"]}
        counters = per_partition[case["partition"]]
        for label in case["labels"]:
            if label["expected"] == "unknown":
                counts["unknown_labels"] += 1
                continue
            key = "false_positive" if label["expected"] == "allow" else "false_negative"
            failed = actual[label["event"]]["penalized"] != (label["expected"] == "penalize")
            counts["label_checks"] += 1
            counters["label_checks"] += 1
            if failed:
                counts[key] += 1
                counters[key] += 1
                failures.append({"id": case["id"], "partition": case["partition"], "source": case["source"], "check": key, **label})
        for observation in case["observations"]:
            if observation.get("expected_progress") is None:
                continue
            counts["progress_checks"] += 1
            if (actual[observation["event"]]["progress_delta"] > 0) != observation["expected_progress"]:
                counts["progress_failures"] += 1
                failures.append({"id": case["id"], "partition": case["partition"], "check": "incorrect_progress", **observation})
        if case["first_offense_exact"]:
            counts["first_offense_checks"] += 1
            actual_first = next((e["event"] for e in result["events"] if e["penalized"]), None)
            if actual_first != case["first_offense"]:
                counts["first_offense_failures"] += 1
                failures.append({"id": case["id"], "check": "first_offense", "expected": case["first_offense"], "actual": actual_first})
    for contract in json.loads((directory / "contracts.json").read_text()):
        actual = replay_detector(contract["events"], module)
        for key, expected_key in (("penalized", "expected_hits"), ("progress_delta", "expected_progress")):
            observed = [e[key] for e in actual["events"]]
            counts["contract_checks"] += 1
            if observed != contract[expected_key]:
                counts["contract_failures"] += 1
                failures.append({"id": contract["id"], "check": key, "expected": contract[expected_key], "actual": observed})
    report = {
        "fixture_version": manifest["version"], "fixtures_sha256": manifest["fixtures_sha256"],
        "detector_sha256": file_digest(Path(module.__file__)), "cases": len(cases),
        "counts": dict(counts), "partitions": per_partition, "failures": failures,
        "regression_pass": not failures,
        "production_ready": False,
        "remaining_gates": ["unknown historical state adjudication", "structured live-event parity", "end-to-end exceptional reward/mask exits", "launcher and SFT artifact preflight"],
        "scope": "CPU detector event replay, not execution of historical commands or a GPU training run",
    }
    write_json(output, report)
    return report


def build(output: Path, count: int, max_step: int) -> dict:
    if output.exists():
        raise ValueError("Output already exists; frozen sets are immutable. Choose a new version directory.")
    output.mkdir(parents=True)
    candidates, corpus_counts, sources = [], collections.Counter(), []
    # Candidate summaries are small; full source rows are retained only for a
    # bounded reservoir per category and run. All rows still get audited.
    pools: dict[tuple, list[dict]] = collections.defaultdict(list)
    with (output / "corpus_audit.jsonl").open("w", encoding="utf-8") as audit:
        for run, folder in RUNS.items():
            for split in ("rollout_data", "validation_data"):
                for path in sorted((ROOT / split / folder).glob("*.jsonl"), key=lambda p: int(p.stem)):
                    if int(path.stem) > max_step:
                        continue
                    sources.append({"path": str(path.relative_to(ROOT)), "sha256": file_digest(path)})
                    with path.open() as stream:
                        for line_number, line in enumerate(stream, 1):
                            row = json.loads(line)
                            if row.get("is_padding"):
                                corpus_counts["padding"] += 1
                                continue
                            provenance = {"run": run, "split": split, "step": int(path.stem), "path": str(path.relative_to(ROOT)), "line": line_number}
                            case = make_case(row, provenance)
                            corpus_counts["rows"] += 1
                            corpus_counts.update("parse:" + issue for issue in case["parse_issues"])
                            corpus_counts.update("category:" + k for k in case["categories"])
                            replay = replay_detector(case["events"])
                            stored = row.get("trajectory_penalized_repeated_tool_calls")
                            mismatch = stored is not None and stored != replay["count"]
                            corpus_counts["stored_replay_mismatches"] += int(mismatch)
                            if mismatch:
                                case["categories"].append("stored_replay_disagreement")
                            audit.write(json.dumps({"id": case["id"], "source": provenance, "task": case["task"], "parse_issues": case["parse_issues"], "categories": case["categories"], "stored_count": stored, "replayed_count": replay["count"], "unknown_repeat_events": len(case["unknown_repeat_events"]), "row_sha256": case["source"]["row_sha256"]}) + "\n")
                            anchor = (run, split, int(path.stem), line_number) in ANCHORS
                            for category in case["categories"] or ["no_repeat_evidence"]:
                                key = (run, split, case["partition"], category)
                                pool = pools[key]
                                pool.append(case)
                                pool.sort(key=lambda c: c["id"])
                                if len(pool) > 12:
                                    pool.pop()
                            if anchor:
                                candidates.append(case)
                    print(f"audited {sources[-1]['path']}", flush=True)
    unique = {c["id"]: c for pool in pools.values() for c in pool}
    unique.update({c["id"]: c for c in candidates})
    selected = select_cases(list(unique.values()), count)
    with (output / "fixtures.jsonl.gz").open("wb") as binary:
        with gzip.GzipFile(fileobj=binary, mode="wb", mtime=0, filename="") as stream:
            for case in selected:
                stream.write((json.dumps(case, ensure_ascii=False) + "\n").encode())
    coverage = {category: {
        "trajectories": sum(category in c["categories"] for c in selected),
        "tasks": len({c["task_key"] for c in selected if category in c["categories"]}),
        "repositories": len({c["repository"] for c in selected if category in c["categories"]}),
    } for category in sorted({k for c in selected for k in c["categories"]})}
    contract_module = importlib.import_module(f"{__package__}.swe_repeat_contracts" if __package__ else "swe_repeat_contracts")
    write_json(output / "contracts.json", contract_module.contracts())
    write_json(output / "filesystem_evidence.json", contract_module.filesystem_evidence())
    manifest = {
        "version": VERSION, "builder_sha256": file_digest(Path(__file__)),
        "requested_count": count, "selected_count": len(selected), "max_step": max_step,
        "fixtures_sha256": file_digest(output / "fixtures.jsonl.gz"),
        "contracts_sha256": file_digest(output / "contracts.json"),
        "partition_counts": dict(collections.Counter(c["partition"] for c in selected)),
        "tasks": len({c["task_key"] for c in selected}), "repositories": len({c["repository"] for c in selected}),
        "sources": sources, "corpus_counts": dict(corpus_counts), "coverage": coverage,
        "labels": "Independent conservative evidence rules; not human adjudication. Unknown labels are excluded from accuracy denominators.",
        "holdout_policy": "SHA256 task split, approximately 30%; same repository/task is never in both partitions. Not repository-disjoint.",
        "first_offense_policy": "Exact only when the earlier repeat labels and transcript are fully known.",
    }
    write_json(output / "manifest.json", manifest)
    write_json(output / "review_index.json", [{k: v for k, v in c.items() if k not in {"raw", "events"}} for c in selected])
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("build")
    create.add_argument("--output-dir", type=Path, required=True)
    create.add_argument("--count", type=int, default=160)
    create.add_argument("--max-step", type=int, default=12)
    check = commands.add_parser("evaluate")
    check.add_argument("--fixtures", type=Path, required=True)
    check.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        if args.count < len(ANCHORS):
            parser.error("--count must accommodate the mandatory known-failure anchors")
        report = build(args.output_dir, args.count, args.max_step)
        print(json.dumps({k: report[k] for k in ("selected_count", "partition_counts", "tasks", "repositories")}))
        return 0
    report = evaluate(args.fixtures, args.report)
    print(json.dumps({k: report[k] for k in ("cases", "counts", "regression_pass", "production_ready")}))
    return 0 if report["regression_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

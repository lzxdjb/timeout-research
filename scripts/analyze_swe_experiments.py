#!/usr/bin/env python3
"""Audit SWE validation coverage, local W&B histories, and saved behavior.

Run with uv run --no-project python scripts/analyze_swe_experiments.py --help.
The fixed-roster interval is an identification bound, NOT a confidence interval.
Unobserved or invalid evaluations are never silently imputed or dropped from it.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import re
import shlex
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import numpy as np

try:
    import orjson

    loads = orjson.loads
except ImportError:
    loads = json.loads

CALL = re.compile(r"<tool_call>\s*<function=([^>]+)>(.*?)</function>\s*</tool_call>", re.S)
PARAM = re.compile(r"<parameter=([^>]+)>(.*?)</parameter>", re.S)
OBS = re.compile(r"<tool_response>.*?</tool_response>", re.S)
THOUGHT_SPLIT = re.compile(r"(?:^|\n)assistant\n(?:<think>\n?)?")
WORD = re.compile(r"\w+")
CORE_PREFIX = "val-core/swe_bench_verified/"
TRAIN = "training/swe_rebench_v2/success_ratio/excluding_completion_cutoff_and_infrastructure"


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def mean(values):
    values = [float(x) for x in values if finite(x)]
    return sum(values) / len(values) if values else None


def dump_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in r.items()} for r in rows)


def roster_from_parquet(path):
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=["reward_model", "extra_info"])
    ids, repos = [], {}
    for row in table.to_pylist():
        info = row.get("extra_info") or {}
        gt = (row.get("reward_model") or {}).get("ground_truth", {})
        gt = loads(gt) if isinstance(gt, str) else gt
        tid = info.get("task_id") or gt.get("instance_id")
        if not tid:
            raise ValueError("Validation row has no stable task ID")
        ids.append(tid)
        repos[tid] = gt.get("repo", info.get("repo", "unknown"))
    if len(ids) != len(set(ids)):
        raise ValueError("Roster contains repeated task IDs; provide a unique-task validation parquet")
    return ids, repos


def is_python_call(name, args):
    # Matches the production diagnostic classifier, not its narrower penalty rule.
    if name == "run_tests":
        return True
    if name not in {"Bash", "run_shell"} or not isinstance(args.get("command"), str):
        return False
    try:
        tokens = shlex.split(args["command"])
    except ValueError:
        return False
    if tokens[:1] == ["cd"]:
        if len(tokens) < 4 or tokens[2] != "&&":
            return False
        tokens = tokens[3:]
    while tokens and (tokens[0] in {"env", "command", "exec"} or re.match(r"^[A-Za-z_][A-Za-z_0-9]*=", tokens[0])):
        tokens = tokens[1:]
    if not tokens:
        return False
    exe = PurePosixPath(tokens[0]).name
    return bool(re.fullmatch(r"python(?:\d+(?:\.\d+)*)?", exe)) or exe in {"pytest", "py.test", "tox", "nox"} or tokens[0] in {"bin/test", "./bin/test"}


def text_features(output):
    observations = OBS.findall(output)
    assistant = OBS.sub("", output)
    calls, py_calls = [], []
    consecutive = py_consecutive = best = py_best = 0
    previous = None
    for name, body in CALL.findall(assistant):
        args = {k: v.strip() for k, v in PARAM.findall(body)}
        ident = (name, tuple(sorted(args.items())))
        consecutive = consecutive + 1 if ident == previous else 1
        best = max(best, consecutive)
        if is_python_call(name, args):
            py_calls.append(ident)
            py_consecutive = py_consecutive + 1 if ident == previous else 1
            py_best = max(py_best, py_consecutive)
        else:
            py_consecutive = 0
        calls.append(ident)
        previous = ident
    within, total, all_grams, exact = 0, 0, Counter(), Counter()
    thought_chars = []
    for piece in THOUGHT_SPLIT.split(assistant):
        if "</think>" not in piece:
            continue
        thought = piece.split("</think>", 1)[0].removeprefix("<think>").strip()
        words = WORD.findall(thought.lower())
        grams = Counter(tuple(words[i:i + 16]) for i in range(max(0, len(words) - 15)))
        total += sum(grams.values())
        within += sum(v - 1 for v in grams.values())
        all_grams.update(grams)
        if len(words) >= 32:
            exact[thought] += 1
        thought_chars.append(len(thought))
    cross = sum(v - 1 for v in all_grams.values()) - within
    counts = Counter(calls)
    return dict(
        parsed_calls=len(calls), exact_duplicate_calls=sum(v - 1 for v in counts.values()),
        identical_call_streak=best, python_calls=len(py_calls), identical_python_streak=py_best,
        thinking_16gram_total=total, thinking_16gram_within=within, thinking_16gram_cross=cross,
        thinking_within_fraction=within / max(1, total), thinking_cross_fraction=cross / max(1, total),
        thinking_chars=sum(thought_chars), thinking_exact_duplicate_turns=sum(v - 1 for v in exact.values()),
        thinking_exact_max_copies=max(exact.values(), default=0),
        observed_tool_responses=len(observations),
    )


def scan_file(job):
    label, kind, filename, behavior = job
    path = Path(filename)
    records, digest = [], hashlib.sha256()
    size_before = path.stat().st_size
    with path.open("rb") as stream:
        for line_number, line in enumerate(stream, 1):
            digest.update(line)
            d = loads(line)
            gt = d.get("gts") or {}
            if isinstance(gt, str):
                try:
                    gt = loads(gt)
                except ValueError:
                    gt = {}
            r = {k: v for k, v in d.items() if k not in {"input", "output", "gts"} and not isinstance(v, (dict, list))}
            r.update(experiment=label, kind=kind, step=int(path.stem), line=line_number,
                     task_id=d.get("task_id") or gt.get("instance_id"), repo=gt.get("repo", "unknown"), source=str(path))
            if behavior and not d.get("is_padding", False):
                r.update(text_features(d.get("output", "")))
            records.append(r)
    if path.stat().st_size != size_before:
        raise RuntimeError(f"Input changed while being read: {path}; retry after publication")
    return records, dict(path=str(path), bytes=size_before, sha256=digest.hexdigest(), rows=len(records))


def final_sessions(rows):
    """One terminal record per session; intermediate output chunks do not add trials."""
    selected = {}
    for r in rows:
        if r.get("is_padding"):
            continue
        uid = str(r.get("uid", f"line-{r['line']}"))
        parts = uid.rsplit("_", 2)
        if len(parts) == 3 and parts[-1].isdigit() and parts[-2].isdigit():
            key, index = tuple(parts[:2]), int(parts[-1])
        else:
            key, index = (uid,), 0
        if key in selected and index == selected[key][0]:
            raise ValueError(f"Duplicate terminal session: {uid}")
        if key not in selected or index > selected[key][0]:
            selected[key] = (index, r)
    return [r for _, r in selected.values()]


def confirmed(r):
    if r.get("evaluation_valid") not in (True, 1):
        return False
    if r.get("observed_infrastructure_failure") or r.get("infrastructure_failure") or r.get("evaluator_failure_excluded"):
        return False
    if r.get("completion_ratio_cutoff"):
        return False
    return r.get("hidden_pass") in (0, 1)


def validation_summary(rows, roster, samples):
    grouped = defaultdict(list)
    for r in final_sessions(rows):
        if r["task_id"] not in roster:
            raise ValueError(f"Task outside fixed roster: {r['task_id']}")
        grouped[r["task_id"]].append(r)
    if any(len(v) > samples for v in grouped.values()):
        raise ValueError("More sessions than --samples-per-task; check n or repeated evaluation publication")
    saved = [r for group in grouped.values() for r in group]
    valid = [r for r in saved if confirmed(r)]
    success = sum(r["hidden_pass"] for r in valid)
    expected = len(roster) * samples
    unknown = expected - len(valid)
    unknown_reasons = Counter((r.get("evaluator_invalid_reason") or r.get("infrastructure_failure_reason") or r.get("observed_infrastructure_failure_reason") or "unconfirmed") for r in saved if not confirmed(r))
    unknown_reasons["not_saved"] += expected - len(saved)
    complete_tasks = {tid: mean(r["hidden_pass"] for r in group) for tid, group in grouped.items() if len(group) == samples and all(confirmed(r) for r in group)}
    return dict(expected_tasks=len(roster), samples_per_task=samples, expected_trials=expected,
                saved_trials=len(saved), confirmed_trials=len(valid), successes=success,
                known_failures=len(valid) - success, unresolved=unknown, coverage=len(valid) / expected,
                lower=success / expected, upper=(success + unknown) / expected,
                observed_valid_only=success / len(valid) if valid else None,
                saved_success_fraction=sum(r.get("hidden_pass", 0) or 0 for r in saved) / len(saved) if saved else None,
                unknown_reasons=dict(unknown_reasons), complete_tasks=complete_tasks,
                unresolved_ids=sorted(set(roster) - set(complete_tasks)),
                missing_ids=sorted(set(roster) - set(grouped)))


def task_outcomes(rows, roster, samples):
    grouped = defaultdict(list)
    for r in final_sessions(rows):
        grouped[r["task_id"]].append(r)
    outcomes = []
    for tid in sorted(roster):
        group = grouped[tid]
        valid = [r for r in group if confirmed(r)]
        successes = sum(r["hidden_pass"] for r in valid)
        reasons = [r.get("evaluator_invalid_reason") or r.get("observed_infrastructure_failure_reason")
                   or r.get("infrastructure_failure_reason") or "unconfirmed" for r in group if not confirmed(r)]
        reasons += ["not_saved"] * (samples - len(group))
        outcomes.append(dict(task_id=tid, expected_trials=samples, saved_trials=len(group),
                             confirmed_trials=len(valid), successes=successes,
                             known_failures=len(valid) - successes, unresolved=samples - len(valid),
                             unresolved_reasons=dict(Counter(reasons)), uids=[r.get("uid") for r in group]))
    return outcomes


def reconstruct_core(rows, samples, threshold):
    """Mirror current core filtering for an audit, including its validity gap.

    Unlike confirmed(), this deliberately does not gate on evaluation_valid.
    mean@N aggregates only groups having N retained numeric responses, matching
    process_validation_metrics. The strict fixed-roster statistic is separate.
    """
    groups = defaultdict(list)
    for r in final_sessions(rows):
        groups[r["task_id"]].append(r)
    eligible = []
    unverified = lambda r: bool(r.get("unverified_trajectory", r.get("evaluator_group_excluded", False)))
    for group in groups.values():
        bad = sum(unverified(r) for r in group)
        valid = len(group) - bad
        ratio = (bad + max(0, samples - len(group))) / samples
        drop = (bool(bad) if threshold is None else
                ratio > threshold or valid < (2 if samples > 1 else 1) or len(group) > samples)
        infra = any(r.get("observed_infrastructure_failure") == 1 and not unverified(r) for r in group)
        if not drop and not infra:
            eligible.append([r for r in group if not unverified(r)])
    result = {}
    for key in ["reward", "acc"]:
        by_n = defaultdict(list)
        for group in eligible:
            values = [r[key] for r in group if finite(r.get(key))]
            if values:
                by_n[len(values)].append(mean(values))
        for n, values in by_n.items():
            result[f"{CORE_PREFIX}{key}/mean@{n}"] = mean(values)
    return result


def wandb_history(root, names):
    from wandb.sdk.internal.datastore import DataStore
    from wandb.proto.wandb_internal_pb2 import Record

    histories = {label: [] for label in names}
    provenance = []
    for path in sorted((root / "wandb").glob("run-*/run-*.wandb")):
        ds = DataStore()
        # Open read-only; the public scanning helper uses r+b unnecessarily.
        ds._fp = path.open("rb")
        ds._fname = str(path)
        ds._index = 0
        ds._size_bytes = path.stat().st_size
        ds._opened_for_scan = True
        ds._read_header()
        label, run_id, rows = None, None, []
        truncated = False
        try:
            while True:
                try:
                    payload = ds.scan_data()
                except (AssertionError, ValueError):
                    truncated = True
                    break
                if payload is None:
                    break
                rec = Record()
                rec.ParseFromString(payload)
                if rec.HasField("run"):
                    label = next((k for k, v in names.items() if v == rec.run.display_name), None)
                    run_id = rec.run.run_id
                    if label is None:
                        break
                if label and rec.HasField("history"):
                    row = {("/".join(item.nested_key) if item.nested_key else item.key): loads(item.value_json) for item in rec.history.item}
                    row["_local_run_id"] = run_id
                    rows.append(row)
        finally:
            ds.close()
        if label:
            histories[label].extend(rows)
            provenance.append(dict(experiment=label, path=str(path), run_id=run_id,
                                   records=len(rows), truncated_tail=truncated))
    # Later records in resumed runs replace only metrics actually present.
    for label, rows in histories.items():
        by_step = {}
        for row in sorted(rows, key=lambda r: r.get("_timestamp", 0)):
            step = row.get("training/global_step")
            if finite(step):
                by_step.setdefault(int(step), {}).update(row)
        histories[label] = [by_step[s] for s in sorted(by_step)]
    return histories, provenance


def aggregate_behavior(rows):
    out = dict(n=len(rows), unique_tasks=len({r.get("task_id") for r in rows}))
    keys = ["score", "raw_score", "hidden_pass", "trajectory_assistant_tokens", "trajectory_response_tokens",
            "trajectory_assistant_turns", "trajectory_tool_dispatches", "trajectory_tool_error_returns",
            "trajectory_repeated_tool_calls", "trajectory_penalized_repeated_tool_calls", "trajectory_max_turn",
            "trajectory_timeout", "trajectory_elapsed_seconds", "submission_signal_seen", "verification_passed",
            "trajectory_tool_Bash_dispatches", "trajectory_tool_Read_dispatches", "trajectory_tool_Edit_dispatches",
            "trajectory_tool_Write_dispatches", "public_test_attempt_count",
            "python_calls", "identical_call_streak", "identical_python_streak", "thinking_within_fraction",
            "thinking_cross_fraction", "thinking_exact_duplicate_turns", "thinking_exact_max_copies",
            "thinking_repeat_retained_events", "thinking_repeat_cross_turn_selected", "tool_repeat_retained_events",
            "tool_repeat_retained_python_tests", "tool_repeat_retained_shell_reads", "tool_repeat_retained_unchanged_reads"]
    for key in keys:
        values = [float(r[key]) for r in rows if finite(r.get(key)) and (r[key] >= 0 or key in {"score", "raw_score"})]
        if values:
            out[key + "_mean"] = mean(values)
            out[key + "_sum"] = sum(values)
            out[key + "_max"] = max(values)
            out[key + "_median"] = float(np.median(values))
            out[key + "_p90"] = float(np.quantile(values, .9))
    calls = sum(r.get("trajectory_tool_dispatches", 0) or 0 for r in rows)
    for key, name in [("trajectory_repeated_tool_calls", "repeat_per_100_calls"), ("trajectory_tool_error_returns", "errors_per_100_calls")]:
        out[name] = 100 * sum(max(0, r.get(key, 0) or 0) for r in rows) / calls if calls else None
    out["bash_percent_of_dispatches"] = 100 * sum(max(0, r.get("trajectory_tool_Bash_dispatches", 0) or 0) for r in rows) / calls if calls else None
    total = sum(r.get("thinking_16gram_total", 0) for r in rows)
    for suffix in ["within", "cross"]:
        out["thinking_" + suffix + "_pooled_percent"] = 100 * sum(r.get("thinking_16gram_" + suffix, 0) for r in rows) / total if total else None
        if rows and total:
            largest = max(rows, key=lambda r: r.get("thinking_16gram_" + suffix, 0))
            denominator = total - largest.get("thinking_16gram_total", 0)
            out["thinking_" + suffix + "_pooled_drop_largest_percent"] = (
                100 * (sum(r.get("thinking_16gram_" + suffix, 0) for r in rows)
                       - largest.get("thinking_16gram_" + suffix, 0)) / denominator if denominator else None
            )
    for key, threshold, name in [("identical_python_streak", 21, "python_streak_21plus"), ("identical_call_streak", 21, "call_streak_21plus"), ("thinking_exact_max_copies", 8, "thought_copies_8plus")]:
        out[name] = sum((r.get(key, 0) or 0) >= threshold for r in rows)
    return out


def paired_comparisons(validation, labels, shared_steps, seed):
    rng = np.random.default_rng(seed)
    comparisons = []
    for step in shared_steps:
        panel = set.intersection(*(set(validation[(label, step)]["complete_tasks"]) for label in labels))
        for i, left in enumerate(labels):
            for right in labels[i + 1:]:
                ids = sorted(panel)
                a = np.array([validation[(left, step)]["complete_tasks"][tid] for tid in ids])
                b = np.array([validation[(right, step)]["complete_tasks"][tid] for tid in ids])
                diff = b - a
                if len(ids):
                    bootstrap = diff[rng.integers(0, len(ids), size=(10000, len(ids)))].mean(1)
                    lo, hi = map(float, np.quantile(bootstrap, [.025, .975]))
                else:
                    lo = hi = None
                comparisons.append(dict(step=step, left=left, right=right, common_tasks=len(ids),
                                        left_rate=mean(a.tolist()), right_rate=mean(b.tolist()),
                                        right_minus_left=mean(diff.tolist()), paired_bootstrap_low=lo,
                                        paired_bootstrap_high=hi, right_wins=int((diff > 0).sum()),
                                        left_wins=int((diff < 0).sum())))
    return comparisons


def behavior_audit(out, sessions, validation, histories, labels, shared_steps):
    """Publish retained/all-session checks and task-matched sensitivities."""
    latest = max(shared_steps)
    windows = [("early", 1, 4), ("matched_late", max(1, latest - 3), latest)]
    window_rows, populations, matched, diagnostics = [], [], [], []
    latest_panel = set.intersection(*(set(validation[(label, latest)]["complete_tasks"]) for label in labels))
    sensitivity_steps = [step for step in shared_steps if step <= 4 or latest - 3 <= step <= latest]
    sensitivity_panel = set.intersection(*(set(validation[(label, step)]["complete_tasks"])
                                           for label in labels for step in sensitivity_steps))
    for label in labels:
        train = [r for r in sessions if r["experiment"] == label and r["kind"] == "rollout_data" and r["step"] <= latest]
        if train:
            for subset in ["all", "trainable"]:
                rr = [r for r in train if subset == "all" or r.get("train_sample_mask", True)]
                populations.append(dict(experiment=label, last_step=latest, subset=subset,
                                        masked=sum(not r.get("train_sample_mask", True) for r in rr),
                                        cutoff=sum(bool(r.get("completion_ratio_cutoff")) for r in rr),
                                        **aggregate_behavior(rr)))
        for kind in ["rollout_data", "validation_data"]:
            for window, lo, hi in windows:
                all_window = [r for r in sessions if r["experiment"] == label and r["kind"] == kind and lo <= r["step"] <= hi]
                for subset in (["all", "trainable"] if kind == "rollout_data" else ["all"]):
                    rr = [r for r in all_window if subset == "all" or r.get("train_sample_mask", True)]
                    if rr:
                        window_rows.append(dict(experiment=label, kind=kind, subset=subset, window=window,
                                                first_step=lo, last_step=hi, **aggregate_behavior(rr)))
                if kind == "validation_data":
                    rr = [r for r in all_window if r["task_id"] in sensitivity_panel]
                    if rr:
                        matched.append(dict(experiment=label, panel="fixed_across_early_and_late", window=window,
                                            panel_tasks=len(sensitivity_panel), **aggregate_behavior(rr)))
        rr = [r for r in sessions if r["experiment"] == label and r["kind"] == "validation_data"
              and r["step"] == latest and r["task_id"] in latest_panel]
        matched.append(dict(experiment=label, panel="latest_common_confirmed", window=f"step_{latest}",
                            panel_tasks=len(latest_panel), **aggregate_behavior(rr)))
        for window, lo, hi in windows:
            hh = [h for h in histories[label] if lo <= h.get("training/global_step", -1) <= hi]
            keys = sorted({k for h in hh for k in h if k.startswith(("tool_failure/", "tool_repeat/", "thinking_repeat/"))
                           or k in {TRAIN, "actor/entropy", "actor/grad_norm", "actor/pg_clipfrac"}})
            for key in keys:
                values = [h[key] for h in hh if finite(h.get(key))]
                if values:
                    diagnostics.append(dict(experiment=label, window=window, metric=key, observations=len(values),
                                            mean=mean(values), sum=sum(values)))
    write_csv(out / "behavior_windows.csv", window_rows)
    write_csv(out / "behavior_population_checks.csv", populations)
    write_csv(out / "behavior_matched_tasks.csv", matched)
    write_csv(out / "wandb_diagnostics_windows.csv", diagnostics)
    dump_json(out / "behavior_panels.json", dict(latest_common_step=latest, latest_panel_ids=sorted(latest_panel),
                                                sensitivity_steps=sensitivity_steps,
                                                sensitivity_panel_ids=sorted(sensitivity_panel)))


def plots(out, labels, validation, histories, behavior, stable, shared_steps):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    palette = ["#2c7fb8", "#d95f0e", "#4b9b59", "#9e6ab5"] + list(plt.get_cmap("tab20").colors)
    colors = {label: palette[i % len(palette)] for i, label in enumerate(labels)}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    height = math.ceil((len(labels) + 1) / 2)
    fig, axes = plt.subplots(height, 2, figsize=(13, 4.5 * height), constrained_layout=True, squeeze=False)
    for ax, label in zip(axes.flat, labels):
        vals = sorted((s, r) for (l, s), r in validation.items() if l == label)
        steps, rr = zip(*vals)
        color = colors[label]
        ax.fill_between(steps, [r["lower"] for r in rr], [r["upper"] for r in rr], color=color, alpha=.22, label="Fixed-roster unresolved-outcome bounds")
        ax.plot(steps, [r["lower"] for r in rr], "o-", color=color, label="Confirmed successes / all scheduled trials")
        ax.plot(steps, [r["logged_core"] for r in rr], "x--", color="#5c5c5c", label="Logged core metric (filtered denominator)")
        ax.set(title=label, xlabel="Training update", ylabel="Success fraction", ylim=(.35, .85))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(alpha=.18)
        ax.legend(fontsize=8, loc="lower left")
    ax = axes.flat[len(labels)]
    step = max(shared_steps)
    panel = set.intersection(*(set(validation[(label, step)]["complete_tasks"]) for label in labels))
    for i, label in enumerate(labels):
        r = validation[(label, step)]
        ax.plot([r["lower"], r["upper"]], [i, i], lw=7, alpha=.3, color=colors[label])
        ax.plot(r["lower"], i, "o", color=colors[label], label="Full-roster lower bound" if i == 0 else None)
        if finite(r["logged_core"]):
            ax.plot(r["logged_core"], i, "x", color="#444444", label="Logged core" if i == 0 else None)
        if panel:
            selected_rate = mean([r["complete_tasks"][tid] for tid in panel])
            ax.plot(selected_rate, i, "D", color=colors[label], ms=6, label=f"Same {len(panel)} confirmed tasks (selected)" if i == 0 else None)
        ax.annotate(f'{r["successes"]}/{r["expected_trials"]}; {r["unresolved"]} unresolved',
                    (r["lower"], i), xytext=(0, 12), textcoords="offset points", fontsize=8)
    ax.set(title=f"Matched training update {step}", xlabel="Success fraction", yticks=range(len(labels)),
           yticklabels=labels, ylim=(-.8, len(labels) - .2), xlim=(.35, .85))
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(alpha=.18)
    ax.legend(fontsize=8, loc="lower left")
    for unused in list(axes.flat)[len(labels) + 1:]:
        unused.set_visible(False)
    fig.suptitle("SWE-bench validation: fixed-roster bounds and matched-task sensitivity\nShading is missing-outcome uncertainty, not a confidence interval; matched panel is selected", fontsize=13)
    for ext in ["png", "pdf"]:
        fig.savefig(out / f"validation_comparison.{ext}", dpi=180)
    plt.close(fig)
    if not behavior:
        return
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), constrained_layout=True)
    metrics = [("repeat_per_100_calls", "Repeated calls per 100 dispatches"),
               ("errors_per_100_calls", "Tool error returns per 100 dispatches"),
               ("trajectory_repeated_tool_calls_mean", "Repeated calls / trajectory"),
               ("thinking_cross_pooled_percent", "Across-thought repeated 16-grams (%)"),
               ("trajectory_assistant_tokens_mean", "Generated assistant tokens / trajectory"),
               ("trajectory_assistant_turns_mean", "Assistant turns / trajectory"),
               ("bash_percent_of_dispatches", "Bash share of tool dispatches (%)")]
    for ax, (key, title) in zip(axes.flat, metrics):
        for label in labels:
            for kind, style in [("rollout_data", "-"), ("validation_data", "--")]:
                rows = sorted((r for r in behavior if r["experiment"] == label and r["kind"] == kind and r["subset"] == ("trainable" if kind == "rollout_data" else "all")), key=lambda r: r["step"])
                if rows:
                    ax.plot([r["step"] for r in rows], [r.get(key) for r in rows], style, color=colors[label], label=f"{label}: {'train' if kind == 'rollout_data' else 'val'}", alpha=.9)
        ax.set(title=title, xlabel="Training update")
        ax.grid(alpha=.18)
    ax = axes.flat[-1]
    for label in labels:
        hh = histories[label]
        ax.plot([r["training/global_step"] for r in hh], [r.get(TRAIN) for r in hh], color=colors[label], label=label)
    ax.set(title="Training success on changing batches", xlabel="Training update", ylabel="Reward-based success fraction")
    ax.grid(alpha=.18)
    ax.legend(fontsize=8)
    axes.flat[0].legend(fontsize=7)
    fig.suptitle("Behavior drift: solid = retained training, dashed = saved validation\nRepetition proxies are descriptive; tool errors and repeated calls are runtime counters", fontsize=13)
    for ext in ["png", "pdf"]:
        fig.savefig(out / f"behavior_comparison.{ext}", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--experiment", action="append", required=True, metavar="LABEL=DIRECTORY_NAME", help="Repeat for each run; directory name must also equal its W&B display name")
    parser.add_argument("--val-file", type=Path, required=True)
    parser.add_argument("--samples-per-task", type=int, default=1, help="Validation val_kwargs.n, not training rollout.n")
    parser.add_argument("--unverified-threshold", type=float, default=.25, help="Recorded run's validation unverified ratio threshold, for logged-core reconstruction only")
    parser.add_argument("--core-metric", help="Optional exact W&B metric key; otherwise select the largest available SWE mean@N")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--behavior", action="store_true", help="Also scan training dumps and extract tool/thinking text features")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261024)
    args = parser.parse_args()
    if args.samples_per_task <= 0:
        parser.error("--samples-per-task must be positive")
    names = dict(spec.split("=", 1) for spec in args.experiment)
    if len(names) != len(args.experiment) or not names or any(not k or not v for k, v in names.items()):
        parser.error("Use distinct, nonempty experiment labels and directory names")
    if not 0 <= args.unverified_threshold <= 1:
        parser.error("--unverified-threshold must be between zero and one")
    root, out = args.root.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    ids, repos = roster_from_parquet(args.val_file)
    labels = list(names)
    histories, wandb_provenance = wandb_history(root, names)
    dump_json(out / "wandb_history.json", histories)
    jobs = []
    for label, name in names.items():
        for kind in (["validation_data", "rollout_data"] if args.behavior else ["validation_data"]):
            paths = sorted((root / kind / name).glob("[0-9]*.jsonl"), key=lambda p: int(p.stem))
            if kind == "validation_data" and not paths:
                raise ValueError(f"No validation dumps for {name}")
            jobs.extend((label, kind, str(path), args.behavior) for path in paths)
    all_rows, provenance = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        tasks = [pool.submit(scan_file, job) for job in jobs]
        for i, task in enumerate(as_completed(tasks), 1):
            rows, info = task.result()
            all_rows.extend(rows)
            provenance.append(info)
            if i % 10 == 0 or i == len(tasks):
                print(f"Scanned {i}/{len(tasks)} files", flush=True)
    all_rows.sort(key=lambda r: (r["experiment"], r["kind"], r["step"], r["line"]))
    # Publish extracted features before aggregation so they remain available if
    # plotting/aggregation encounters a new schema in an ongoing experiment.
    with gzip.open(out / "trajectory_features.jsonl.gz", "wt") as stream:
        for r in all_rows:
            stream.write(json.dumps(r, allow_nan=False) + "\n")
    groups = defaultdict(list)
    for r in all_rows:
        groups[(r["experiment"], r["kind"], r["step"])].append(r)
    sessions = [r for rows in groups.values() for r in final_sessions(rows)]
    validation, behavior, task_rows = {}, [], []
    for (label, kind, step), rows in groups.items():
        if kind == "validation_data":
            result = validation_summary(rows, set(ids), args.samples_per_task)
            history = next((h for h in histories[label] if h.get("training/global_step") == step), {})
            candidates = [k for k in history if re.fullmatch(re.escape(CORE_PREFIX) + r"(?:reward|acc)/mean@\d+", k)]
            core_key = args.core_metric or max(candidates, key=lambda k: int(k.rsplit("@", 1)[1]), default=None)
            reconstructed = reconstruct_core(rows, args.samples_per_task, args.unverified_threshold)
            result["logged_core_key"] = core_key
            result["logged_core"] = history.get(core_key)
            result["reconstructed_logged_core"] = reconstructed.get(core_key)
            a, b = result["logged_core"], result["reconstructed_logged_core"]
            result["logged_core_matches_raw"] = abs(a - b) < 1e-10 if finite(a) and finite(b) else None
            result["experiment"], result["step"] = label, step
            validation[(label, step)] = result
            task_rows.extend(dict(experiment=label, step=step, **r) for r in task_outcomes(rows, ids, args.samples_per_task))
        if args.behavior:
            subsets = {"all": final_sessions(rows)}
            if kind == "rollout_data":
                subsets["trainable"] = [r for r in subsets["all"] if r.get("train_sample_mask", True)]
            for subset, rr in subsets.items():
                behavior.append(dict(experiment=label, kind=kind, step=step, subset=subset, **aggregate_behavior(rr)))
    shared_steps = sorted(set.intersection(*({s for l, s in validation if l == label} for label in labels)))
    if not shared_steps:
        raise ValueError("No training steps have validation dumps in every experiment")
    stable = set.intersection(*(set(validation[(label, step)]["complete_tasks"]) for label in labels for step in shared_steps))
    comparisons = paired_comparisons(validation, labels, shared_steps, args.seed)
    summaries = [{k: v for k, v in r.items() if k != "complete_tasks"} for r in validation.values()]
    write_csv(out / "validation_summary.csv", summaries)
    write_csv(out / "validation_task_outcomes.csv", task_rows)
    write_csv(out / "paired_comparisons.csv", comparisons)
    write_csv(out / "behavior_by_step.csv", behavior)
    dump_json(out / "validation_details.json", list(validation.values()))
    latest_common = max(shared_steps)
    if args.behavior:
        behavior_audit(out, sessions, validation, histories, labels, shared_steps)
    extremes = sorted(sessions, key=lambda r: max(r.get("thinking_exact_max_copies", 0), r.get("identical_python_streak", 0), r.get("identical_call_streak", 0)), reverse=True)[:100]
    write_csv(out / "extreme_trajectories.csv", extremes)
    summary = dict(created_utc=datetime.now(timezone.utc).isoformat(), roster_tasks=len(ids),
                   roster_repo_counts=dict(Counter(repos.values())), samples_per_task=args.samples_per_task,
                   experiments=names, shared_steps=shared_steps, stable_panel_ids=sorted(stable),
                   rows=len(all_rows), files=len(provenance), common_final_step=latest_common,
                   logged_core_checks=dict(matched=sum(r["logged_core_matches_raw"] is True for r in validation.values()),
                                           mismatched=sum(r["logged_core_matches_raw"] is False for r in validation.values()),
                                           unavailable=sum(r["logged_core_matches_raw"] is None for r in validation.values())),
                   latest={label: {k: v for k, v in validation[(label, max(s for l, s in validation if l == label))].items() if k != "complete_tasks"} for label in labels},
                   stable_panel_rates={label: {step: mean([validation[(label, step)]["complete_tasks"][tid] for tid in stable]) for step in shared_steps} for label in labels})
    dump_json(out / "summary.json", summary)
    dump_json(out / "provenance.json", dict(val_file=str(args.val_file.resolve()), val_file_sha256=hashlib.sha256(args.val_file.read_bytes()).hexdigest(),
                                            analysis_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                            inputs=provenance, wandb=wandb_provenance,
                                            notes=["No training configuration was modified.", "Bounds include all missing and invalid trials.",
                                                   "Observed-only panels are selected subsets, not corrected full-benchmark scores.",
                                                   "Task bootstrap assumes exchangeable tasks; no correction for checkpoint selection or repository clusters.",
                                                   "Training file s uses the policy before update s; validation s uses the policy after update s."]))
    plots(out, labels, validation, histories, behavior, stable, shared_steps)
    print(json.dumps({"output": str(out), "roster": len(ids), "files": len(provenance), "rows": len(all_rows), "stable_panel": len(stable), "common_final_step": latest_common}), flush=True)


if __name__ == "__main__":
    main()

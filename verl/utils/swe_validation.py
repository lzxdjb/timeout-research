"""Fixed-roster SWE validation accounting, independent of rollout scheduling.

Unresolved outcomes are identification bounds, never confidence intervals or
silently imputed model failures. Strict mode is opt-in for SWE V1 launchers.
"""

import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path


def strict_validation_enabled():
    return os.getenv("SWE_AGENT_VALIDATION_STRICT_ACCOUNTING", "0").lower() in {"1", "true", "yes", "on"}


def positive_seconds(name, default):
    value = float(os.getenv(name, str(default)))
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def evaluation_reason(info):
    """None means a confirmed binary evaluation, including known model failures."""
    if info.get("completion_ratio_cutoff"):
        return "completion_ratio_cutoff"
    if info.get("validation_attempt_timeout"):
        return "validation_attempt_timeout"
    if any(
        info.get(k)
        for k in (
            "observed_infrastructure_failure",
            "infrastructure_failure",
            "evaluator_failure_excluded",
            "evaluator_group_excluded",
            "unverified_trajectory",
        )
    ):
        return (
            info.get("evaluator_invalid_reason")
            or info.get("observed_infrastructure_failure_reason")
            or info.get("infrastructure_failure_reason")
            or "infrastructure_or_evaluator_failure"
        )
    if info.get("evaluator_invalid_reason"):
        return info["evaluator_invalid_reason"]
    # Partial-reward count parsing is independent of the official binary
    # evaluator. Unsupported count formats do not invalidate a confirmed pass.
    if info.get("hidden_pass") not in (0, 1):
        return "missing_binary_hidden_pass"
    # A confirmed model rejection is already a benchmark failure. Requiring
    # tests to start in this case would retry bad patches and bias the score.
    model_rejection = info["hidden_pass"] == 0 and (
        info.get("rebench_failure_kind") == "model_failure"
        or (
            info.get("hidden_failure_phase") in {"patch_check", "patch_apply"}
            and info.get("hidden_verifier_staging_ok") in (True, 1)
            and info.get("hidden_verifier_setup_ok") in (True, 1)
        )
    )
    if info.get("evaluation_valid") not in (True, 1) and not model_rejection:
        return "evaluation_not_confirmed"
    return None


def task_identity(row):
    info = row.get("extra_info") or {}
    ground_truth = (row.get("reward_model") or {}).get("ground_truth") or {}
    if isinstance(ground_truth, str):
        ground_truth = json.loads(ground_truth)
    tid = info.get("task_id") or ground_truth.get("instance_id") or ground_truth.get("task_id")
    if not tid:
        raise ValueError("Strict SWE validation requires a stable task ID")
    source = row.get("metric_data_source") or row.get("data_source")
    if not isinstance(source, str) or not source:
        raise ValueError(f"Missing validation metric source for {tid}")
    return dict(task_id=str(tid), source=source)


def load_roster(files):
    import pyarrow.parquet as pq

    if isinstance(files, (str, Path)):
        files = [files]
    roster, provenance = [], []
    for filename in files:
        path = Path(filename)
        table = pq.read_table(path)
        roster.extend(
            task_identity(row)
            for row in table.select(
                [
                    k
                    for k in ("extra_info", "reward_model", "metric_data_source", "data_source")
                    if k in table.column_names
                ]
            ).to_pylist()
        )
        provenance.append(
            dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), rows=len(table))
        )
    if not roster or len({r["task_id"] for r in roster}) != len(roster):
        raise ValueError("Strict validation roster must contain nonempty, unique task IDs")
    expected = os.getenv("SWE_AGENT_VALIDATION_EXPECTED_TASKS")
    if expected is not None and len(roster) != int(expected):
        raise ValueError(f"Expected {expected} validation tasks, found {len(roster)}")
    return roster, provenance


def fixed_roster_report(roster, observations, samples, attempts=()):
    if samples <= 0:
        raise ValueError("Validation samples must be positive")
    tasks = {
        r["task_id"]: dict(
            r,
            expected_trials=samples,
            confirmed_trials=0,
            successes=0,
            known_failures=0,
            unresolved=samples,
            unresolved_reasons={},
        )
        for r in roster
    }
    seen = set()
    reasons = defaultdict(Counter)
    for row in attempts:
        if row["task_id"] not in tasks:
            raise ValueError(f"Attempt outside roster: {row['task_id']}")
        if row.get("reason"):
            reasons[row["task_id"]][row["reason"]] += 1
    for row in observations:
        tid, session = row["task_id"], row["session_id"]
        if tid not in tasks or (tid, session) in seen:
            raise ValueError(f"Unknown or duplicate validation result: {tid}/{session}")
        seen.add((tid, session))
        if sum(t == tid for t, _ in seen) > samples:
            raise ValueError(f"Too many validation sessions for {tid}")
        reason = evaluation_reason(row)
        if reason:
            reasons[tid][reason] += 1
            continue
        task = tasks[tid]
        task["confirmed_trials"] += 1
        task["successes"] += int(row["hidden_pass"])
        task["known_failures"] += 1 - int(row["hidden_pass"])
        task["unresolved"] -= 1
    by_source = defaultdict(list)
    for tid, task in tasks.items():
        if task["unresolved"]:
            task["unresolved_reasons"] = dict(reasons[tid]) or {"not_published": task["unresolved"]}
        by_source[task["source"]].append(task)
    metrics = {}
    for source, rows in by_source.items():
        expected = len(rows) * samples
        success = sum(r["successes"] for r in rows)
        confirmed = sum(r["confirmed_trials"] for r in rows)
        unknown = expected - confirmed
        prefix = f"val-core/{source}/fixed_roster"
        metrics.update(
            {
                f"{prefix}/success_lower@{samples}": success / expected,
                f"{prefix}/success_upper@{samples}": (success + unknown) / expected,
                f"{prefix}/coverage": confirmed / expected,
                f"{prefix}/expected_tasks": len(rows),
                f"{prefix}/expected_trials": expected,
                f"{prefix}/confirmed_trials": confirmed,
                f"{prefix}/successes": success,
                f"{prefix}/known_failures": confirmed - success,
                f"{prefix}/unresolved": unknown,
                f"{prefix}/complete": int(not unknown),
            }
        )
        if confirmed:
            metrics[f"val-aux/{source}/observed_only/hidden_pass_mean"] = success / confirmed
        if not unknown:
            metrics[f"val-core/{source}/reward/mean@{samples}"] = success / expected
    return metrics, list(tasks.values())


def write_json_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)

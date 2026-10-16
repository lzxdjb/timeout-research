#!/usr/bin/env bash
# Run in the foreground; use nohup at the call site to detach.
set -euo pipefail
repeat_repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repeat_repo_dir"
export PYTHONPATH="$PWD:$PWD/../stock-rl-reflect${PYTHONPATH:+:$PYTHONPATH}"
export SWE_AGENT_REPEAT_JUDGE_URL="${SWE_AGENT_REPEAT_JUDGE_URL:-http://10.248.103.79:18090}"
export SWE_AGENT_REPEAT_JUDGE_MODEL="${SWE_AGENT_REPEAT_JUDGE_MODEL:-swe-repeat-judge}"
export SWE_AGENT_REPEAT_JUDGE_MAX_CONTEXT="${SWE_AGENT_REPEAT_JUDGE_MAX_CONTEXT:-253952}"
export SWE_AGENT_REPEAT_JUDGE_MAX_OUTPUT="${SWE_AGENT_REPEAT_JUDGE_MAX_OUTPUT:-1024}"
export SWE_AGENT_REPEAT_JUDGE_TIMEOUT="${SWE_AGENT_REPEAT_JUDGE_TIMEOUT:-360}"
export SWE_AGENT_REPEAT_JUDGE_QUEUE_TIMEOUT="${SWE_AGENT_REPEAT_JUDGE_QUEUE_TIMEOUT:-3600}"
export SWE_AGENT_REPEAT_JUDGE_CONCURRENCY="${SWE_AGENT_REPEAT_JUDGE_CONCURRENCY:-4}"
export SWE_AGENT_REPEAT_JUDGE_TEMPERATURE="${SWE_AGENT_REPEAT_JUDGE_TEMPERATURE:-0.2}"
export SWE_AGENT_REPEAT_JUDGE_VOTE_COUNT="${SWE_AGENT_REPEAT_JUDGE_VOTE_COUNT:-8}"
export SWE_AGENT_REPEAT_JUDGE_VOTE_QUORUM="${SWE_AGENT_REPEAT_JUDGE_VOTE_QUORUM:-7}"
export SWE_AGENT_REPEAT_JUDGE_ENABLE_THINKING="${SWE_AGENT_REPEAT_JUDGE_ENABLE_THINKING:-0}"
export SWE_AGENT_REPEAT_REVIEWER_ENABLE_THINKING="${SWE_AGENT_REPEAT_REVIEWER_ENABLE_THINKING:-0}"
export SWE_AGENT_REPEAT_REVIEWER_ENABLED="${SWE_AGENT_REPEAT_REVIEWER_ENABLED:-1}"
export SWE_AGENT_REPEAT_REVIEWER_TEMPERATURE="${SWE_AGENT_REPEAT_REVIEWER_TEMPERATURE:-$SWE_AGENT_REPEAT_JUDGE_TEMPERATURE}"
export SWE_AGENT_REPEAT_REVIEWER_MAX_OUTPUT="${SWE_AGENT_REPEAT_REVIEWER_MAX_OUTPUT:-$SWE_AGENT_REPEAT_JUDGE_MAX_OUTPUT}"
export SWE_AGENT_REPEAT_JUDGE_ERROR_RETRIES="${SWE_AGENT_REPEAT_JUDGE_ERROR_RETRIES:-0}"
export SWE_AGENT_REPEAT_RULE_SET="${SWE_AGENT_REPEAT_RULE_SET:-expanded}"
repeat_suite="${SWE_REPEAT_SUITE:-$PWD/analysis/reward_model_trajectory_v13_20261016/frozen_task_disjoint}"
repeat_case_concurrency="${SWE_REPEAT_CASE_CONCURRENCY:-4}"
repeat_run_dir="${SWE_REPEAT_RUN_DIR:-$PWD/analysis/reward_vote_sweep/v13_n${SWE_AGENT_REPEAT_JUDGE_VOTE_COUNT}_q${SWE_AGENT_REPEAT_JUDGE_VOTE_QUORUM}_jt${SWE_AGENT_REPEAT_JUDGE_ENABLE_THINKING}_rt${SWE_AGENT_REPEAT_REVIEWER_ENABLE_THINKING}_rv${SWE_AGENT_REPEAT_REVIEWER_ENABLED}_t${SWE_AGENT_REPEAT_JUDGE_TEMPERATURE}_${SWE_AGENT_REPEAT_RULE_SET}_$(date -u +%Y%m%dT%H%M%S)_$$}"
mkdir -p "$repeat_run_dir"
if [[ -e "$repeat_run_dir/report.json" || -e "$repeat_run_dir/run.log" ]]; then
  echo "Refusing to overwrite an existing run: $repeat_run_dir" >&2
  exit 1
fi
export SWE_AGENT_REPEAT_AUDIT_DIR="$repeat_run_dir/judge_artifacts"
# Python's model preflight and configuration validation run before evaluation.
printf 'Run directory: %s\n' "$repeat_run_dir"
python -u -m scripts.swe_repeat_regression two-level-evaluate \
  --fixtures "$repeat_suite/fixtures.jsonl.gz" \
  --labels "$repeat_suite/labels.json" \
  --report "$repeat_run_dir/report.json" \
  --online --audit-level2 --case-concurrency "$repeat_case_concurrency" \
  2>&1 | tee "$repeat_run_dir/run.log"

#!/usr/bin/env bash
# Rendering evaluation; no training or service restarts. Use nohup to detach.
set -euo pipefail
repeat_repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repeat_repo_dir"
export SWE_AGENT_REPEAT_JUDGE_URL="${SWE_AGENT_REPEAT_JUDGE_URL:-http://10.248.100.12:18090}"
export SWE_AGENT_REPEAT_JUDGE_TIMEOUT="${SWE_AGENT_REPEAT_JUDGE_TIMEOUT:-1800}"
export SWE_AGENT_REPEAT_JUDGE_QUEUE_TIMEOUT="${SWE_AGENT_REPEAT_JUDGE_QUEUE_TIMEOUT:-14400}"
export SWE_REPEAT_SUITE="${SWE_REPEAT_SUITE:-$PWD/analysis/reward_model_rendering_v14_20261019/frozen_task_disjoint}"
export SWE_AGENT_REPEAT_JUDGE_INPUT_VIEW="${SWE_AGENT_REPEAT_JUDGE_INPUT_VIEW:-full}"
export SWE_AGENT_REPEAT_JUDGE_EARLY_EXIT_ENABLED="${SWE_AGENT_REPEAT_JUDGE_EARLY_EXIT_ENABLED:-1}"
export SWE_REPEAT_RUN_DIR="${SWE_REPEAT_RUN_DIR:-$PWD/analysis/reward_vote_sweep/v14_${SWE_AGENT_REPEAT_JUDGE_INPUT_VIEW}_early${SWE_AGENT_REPEAT_JUDGE_EARLY_EXIT_ENABLED}_$(date -u +%Y%m%dT%H%M%S)_$$}"
exec bash scripts/run_swe_repeat_v13.sh

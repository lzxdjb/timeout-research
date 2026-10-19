#!/usr/bin/env bash
# Curated semantic evaluation. Native-token shortcut coverage lives in CPU tests.
set -euo pipefail
repeat_repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repeat_repo_dir"
export SWE_REPEAT_SUITE="${SWE_REPEAT_SUITE:-$PWD/analysis/reward_model_curated_v15_20261019/frozen_task_disjoint}"
export SWE_AGENT_REPEAT_JUDGE_INPUT_FORMAT="${SWE_AGENT_REPEAT_JUDGE_INPUT_FORMAT:-blocks}"
export SWE_AGENT_REPEAT_JUDGE_PROMPT_POLICY="${SWE_AGENT_REPEAT_JUDGE_PROMPT_POLICY:-precision}"
export SWE_AGENT_REPEAT_JUDGE_LOCAL_PENALTY_SHORTCUT_ENABLED=0
export SWE_REPEAT_RUN_DIR="${SWE_REPEAT_RUN_DIR:-$PWD/analysis/reward_vote_sweep/v15_${SWE_AGENT_REPEAT_JUDGE_INPUT_FORMAT}_${SWE_AGENT_REPEAT_JUDGE_PROMPT_POLICY}_$(date -u +%Y%m%dT%H%M%S)_$$}"
exec bash scripts/run_swe_repeat_v14.sh

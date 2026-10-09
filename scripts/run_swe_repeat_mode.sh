#!/usr/bin/env bash
# Usage: bash scripts/run_swe_repeat_mode.sh shadow|apply <existing training command...>
set -euo pipefail
repeat_mode="${1:-}"
[[ "$repeat_mode" == shadow || "$repeat_mode" == apply ]] || { echo 'Usage: run_swe_repeat_mode.sh shadow|apply <training command...>' >&2; exit 2; }
shift
[[ $# -gt 0 ]] || { echo 'An existing training command is required' >&2; exit 2; }
repeat_verl_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$repeat_verl_dir:${SWE_SOURCE:-$repeat_verl_dir/../stock-rl-reflect}${PYTHONPATH:+:$PYTHONPATH}"
export SWE_AGENT_REPEAT_REWARD_MODE="$repeat_mode"
export SWE_AGENT_REPEAT_REWARD_MULTIPLIER="${SWE_AGENT_REPEAT_REWARD_MULTIPLIER:-0.1}"
: "${SWE_AGENT_REPEAT_JUDGE_URL:?Set SWE_AGENT_REPEAT_JUDGE_URL to http://judge-host:18090}"
export SWE_AGENT_REPEAT_JUDGE_URL
export SWE_AGENT_TRAINING_PROTOCOL_REWARD_SHAPING=1
export SWE_AGENT_TRAINING_PROTOCOL_SUBMISSION_ONLY=1
export SWE_AGENT_TRAINING_SUBMISSION_PENALTY=0.1
# Preserve explicit conflicting settings so validation reports them, never silently override them.
export SWE_AGENT_TRAINING_REPEATED_TOOL_REWARD_SHAPING="${SWE_AGENT_TRAINING_REPEATED_TOOL_REWARD_SHAPING:-0}"
python3 "$repeat_verl_dir/scripts/swe_repeat_regression.py" judge-preflight
exec "$@"

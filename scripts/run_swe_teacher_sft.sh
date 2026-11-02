#!/usr/bin/env bash
set -euo pipefail

VERL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$VERL_ROOT"
export PYTHONPATH="$VERL_ROOT:$VERL_ROOT/../stock-rl-reflect${PYTHONPATH:+:$PYTHONPATH}"
exec "${SWE_SFT_PYTHON:-python3}" -m scripts.swe_teacher_sft "$@"

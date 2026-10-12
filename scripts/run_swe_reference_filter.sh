#!/usr/bin/env bash
# Full or pilot reference audit, independent candidate repeats, verified export.
set -euo pipefail

if (( $# < 3 )); then
  echo "Usage: bash $0 SOURCE.parquet AUDIT_DIRECTORY OUTPUT.parquet [--pilot-size N] [--seed N] [--resume] [--resume-cleanup-upgrade]" >&2
  exit 2
fi
filter_source=$1
filter_root=$2
filter_output=$3
shift 3
filter_resume=()
filter_upgrade=()
filter_pilot_size=0
filter_seed=20261010
while (( $# )); do
  case $1 in
    --resume) filter_resume=(--resume); shift ;;
    --resume-cleanup-upgrade) filter_upgrade=(--resume-cleanup-upgrade); shift ;;
    --pilot-size|--seed)
      (( $# >= 2 )) && [[ $2 =~ ^[0-9]+$ ]] || {
        echo "$1 requires a nonnegative integer" >&2; exit 2;
      }
      if [[ $1 == --pilot-size ]]; then
        filter_pilot_size=$2
      else
        filter_seed=$2
      fi
      shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
filter_repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$filter_repo"
export PYTHONPATH="$filter_repo:$filter_repo/../stock-rl-reflect${PYTHONPATH:+:$PYTHONPATH}"
: "${SWE_AGENT_EXECUTION_URLS:?Set SWE_AGENT_EXECUTION_URLS before starting the filter}"
: "${SWE_REFERENCE_EVALUATOR_ROOT:?Set SERVICE_ROOT=LOCAL_ROOT for private evaluator evidence}"

# Every worker owns a task through first audit, eligible repeat/retries and
# terminal image cleanup. Preserve the established concurrency settings.
exec python3 scripts/swe_rebench_reference_filter.py \
  --dataset "$filter_source" \
  --output-dir "$filter_root" \
  --filtered-output "$filter_output" \
  --evaluator-root "$SWE_REFERENCE_EVALUATOR_ROOT" \
  --pilot-size "$filter_pilot_size" --seed "$filter_seed" \
  --image-prepare-deadline 1800 --image-prepare-parallelism 8 \
  --workers-per-endpoint 24 --scenario-deadline 900 \
  --image-cleanup task-images \
  "${filter_resume[@]}" "${filter_upgrade[@]}"

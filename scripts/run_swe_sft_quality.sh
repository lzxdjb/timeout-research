#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python_bin="${SWE_SFT_PYTHON:-python3}"
output_root="${SWE_SFT_OUTPUT_ROOT:-analysis/swe_sft_quality/v2}"
source_manifest="${SWE_SFT_SOURCE_MANIFEST:-scripts/swe_sft_sources.json}"
rl_reference="${SWE_SFT_RL_REFERENCE:-../stock-rl-reflect/data/swe_benchmarks/verl/swe_rebench_v2_reference_verified_snapshot_2000.parquet}"
export PYTHONPATH=".:../stock-rl-reflect${PYTHONPATH:+:$PYTHONPATH}"
common=(--quality-filter sft-judge --source-manifest "$source_manifest"
  --val-fraction 0 --max-response-tokens 95536
  --max-tool-dispatches 120 --relaxed-max-tool-dispatches 120
  --judge-vote-count 8
  --judge-cache-dir "$output_root/judge_cache" --rl-reference-parquet "$rl_reference")
action="${1:-}"
if [[ $# -gt 0 ]]; then shift; fi
case "$action" in
  prepare)
    exec "$python_bin" scripts/make_swe_sft_data.py "${common[@]}" \
      --output-dir "$output_root/pilot" --prepare-only "$@" ;;
  evidence)
    exec "$python_bin" -m scripts.swe_sft_filter_audit evidence \
      --pool-dir "$output_root/pilot" --output-dir "$output_root/offline_evidence" --cases 60 "$@" ;;
  pilot)
    exec "$python_bin" scripts/make_swe_sft_data.py "${common[@]}" \
      --output-dir "$output_root/pilot" --pilot-cases 60 "$@" ;;
  check)
    if [[ $# -ne 1 ]]; then echo 'Usage: run_swe_sft_quality.sh check LABELS.jsonl' >&2; exit 2; fi
    exec "$python_bin" -m scripts.swe_sft_filter_audit check --pilot-dir "$output_root/pilot" --labels "$1" ;;
  full)
    exec "$python_bin" scripts/make_swe_sft_data.py "${common[@]}" \
      --output-dir "$output_root/full" --audited-pilot-dir "$output_root/pilot" "$@" ;;
  *) echo 'Usage: run_swe_sft_quality.sh {prepare|evidence|pilot|check LABELS.jsonl|full} [options]' >&2; exit 2 ;;
esac

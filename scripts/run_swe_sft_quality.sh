#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python_bin="${SWE_SFT_PYTHON:-python3}"
output_root="${SWE_SFT_OUTPUT_ROOT:-analysis/swe_sft_quality/v4}"
source_manifest="${SWE_SFT_SOURCE_MANIFEST:-scripts/swe_sft_sources.json}"
rl_reference="${SWE_SFT_RL_REFERENCE:-../stock-rl-reflect/data/swe_benchmarks/verl/swe_rebench_v2_reference_verified_snapshot_2000.parquet}"
export PYTHONPATH=".:../stock-rl-reflect${PYTHONPATH:+:$PYTHONPATH}"
common=(--quality-filter sft-judge --source-manifest "$source_manifest"
  --val-fraction 0 --max-response-tokens 95536
  --max-tool-dispatches 120 --relaxed-max-tool-dispatches 120
  --judge-vote-count 8
  --judge-cache-dir "$output_root/judge_cache" --rl-reference-parquet "$rl_reference")
if [[ "${SWE_SFT_ALLOW_UNAUDITED_PILOT:-0}" == "1" ]]; then
  common+=(--allow-unaudited-pilot)
fi
# Pilot and full must use the same prepared file list even while RL adds files.
if [[ -f "$output_root/pilot/sources.json" ]]; then
  common+=(--source-snapshot "$output_root/pilot/sources.json")
fi
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
    audit_dir="$output_root/pilot"
    allow_unaudited="${SWE_SFT_ALLOW_UNAUDITED_PILOT:-0}"
    next_is_audit_dir=false
    for option in "$@"; do
      if $next_is_audit_dir; then
        audit_dir="$option"
        next_is_audit_dir=false
      else
        case "$option" in
          --allow-unaudited-pilot) allow_unaudited=1 ;;
          --audited-pilot-dir) next_is_audit_dir=true ;;
          --audited-pilot-dir=*) audit_dir="${option#*=}" ;;
        esac
      fi
    done
    if [[ ! -f "$audit_dir/audit_report.json" && "$allow_unaudited" != "1" ]]; then
      cat >&2 <<EOF
Cannot start full filtering: missing $audit_dir/audit_report.json
Effective output root: $output_root
Log redirection does not select the output root; set SWE_SFT_OUTPUT_ROOT explicitly.
Run pilot with this root, independently label every pilot case (at least 50),
then run: bash scripts/run_swe_sft_quality.sh check /absolute/path/pilot_labels.jsonl
If using --audited-pilot-dir, validate that pilot with scripts.swe_sft_filter_audit check.
The small calibration set cannot satisfy the full-run audit requirement.
To explicitly proceed without the audit, pass --allow-unaudited-pilot.
No source scan or judge request was started by this invocation.
EOF
      exit 2
    fi
    full_args=("${common[@]}" --output-dir "$output_root/full")
    if [[ "$allow_unaudited" != "1" ]]; then
      full_args+=(--audited-pilot-dir "$audit_dir")
    fi
    exec "$python_bin" scripts/make_swe_sft_data.py "${full_args[@]}" "$@" ;;
  *) echo 'Usage: run_swe_sft_quality.sh {prepare|evidence|pilot|check LABELS.jsonl|full} [options]' >&2; exit 2 ;;
esac

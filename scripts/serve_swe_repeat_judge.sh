#!/usr/bin/env bash
# Standalone vLLM service. No installation, downloads, or training side effects.
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${SWE_REPEAT_JUDGE_GPUS:-0,1,2,3}"
judge_model_path="${SWE_REPEAT_JUDGE_CHECKPOINT:-/cpfs01/thscc/sharestorage/iwc/HithinkGPT/models/Qwen3.5-122}"
judge_tp="${SWE_REPEAT_JUDGE_TP:-4}"
judge_context="${SWE_AGENT_REPEAT_JUDGE_MAX_CONTEXT:-131072}"
judge_host="${SWE_REPEAT_JUDGE_HOST:-0.0.0.0}"
judge_port="${SWE_REPEAT_JUDGE_PORT:-18090}"
# Set to 0 to load the vision encoder and accept image/video inputs.
judge_language_model_only="${SWE_REPEAT_JUDGE_LANGUAGE_MODEL_ONLY:-1}"
[[ "$judge_language_model_only" == 0 || "$judge_language_model_only" == 1 ]] || {
  echo "Invalid SWE_REPEAT_JUDGE_LANGUAGE_MODEL_ONLY: $judge_language_model_only (expected 0 or 1)" >&2
  exit 2
}
[[ -f "$judge_model_path/config.json" ]] || { echo "Checkpoint config not found: $judge_model_path" >&2; exit 2; }
IFS=',' read -ra judge_gpus <<< "$CUDA_VISIBLE_DEVICES"
judge_gpu_count="${#judge_gpus[@]}"
[[ "$judge_tp" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid judge TP: $judge_tp (expected a positive integer)" >&2; exit 2; }
# Check the number of digits before arithmetic so oversized TP values cannot overflow.
if (( ${#judge_tp} > ${#judge_gpu_count} )) || (( judge_gpu_count % judge_tp != 0 )); then
  echo "GPU list length ($judge_gpu_count) must be a multiple of TP ($judge_tp)" >&2
  exit 2
fi
# Use all listed GPUs locally; equal GPU count and TP preserves the single replica.
judge_inferred_dp="$((judge_gpu_count / judge_tp))"
judge_dp="${SWE_REPEAT_JUDGE_DP:-$judge_inferred_dp}"
[[ "$judge_dp" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid judge DP: $judge_dp (expected a positive integer)" >&2; exit 2; }
[[ "$judge_dp" == "$judge_inferred_dp" ]] || { echo "GPU list length ($judge_gpu_count) must equal TP ($judge_tp) * DP ($judge_dp)" >&2; exit 2; }
[[ "$judge_port" =~ ^[1-9][0-9]{0,4}$ && "$judge_port" -le 65535 ]] || { echo "Invalid judge port: $judge_port" >&2; exit 2; }

# 0.0.0.0 is a bind address only. Resolve a separate client endpoint.
judge_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
judge_endpoint_helper="$judge_script_dir/swe_repeat_judge_endpoint.py"
judge_advertise_url="$(python3 "$judge_endpoint_helper" resolve "$judge_host" "$judge_port")"
judge_command=(vllm serve "$judge_model_path"
  --served-model-name "${SWE_AGENT_REPEAT_JUDGE_MODEL:-swe-repeat-judge}"
  --host "$judge_host" --port "$judge_port"
  --tensor-parallel-size "$judge_tp" --dtype bfloat16
  --max-model-len "$judge_context" --max-num-seqs "${SWE_REPEAT_JUDGE_MAX_SEQS:-4}"
  --gpu-memory-utilization "${SWE_REPEAT_JUDGE_MEMORY_FRACTION:-0.90}"
  --generation-config vllm
  --default-chat-template-kwargs '{"enable_thinking":false}')
if [[ "$judge_language_model_only" == 1 ]]; then
  judge_command+=(--language-model-only)
fi
# Do not add a new vLLM flag to existing single-replica launches.
if (( judge_dp > 1 )); then
  judge_command+=(--data-parallel-size "$judge_dp")
fi
if [[ "${1:-}" == "--dry-run" ]]; then
  printf 'Judge bind address: http://%s:%s\n' "$judge_host" "$judge_port" >&2
  printf 'Judge client endpoint: %s\n' "$judge_advertise_url" >&2
  printf 'CUDA_VISIBLE_DEVICES=%q ' "$CUDA_VISIBLE_DEVICES"
  printf '%q ' "${judge_command[@]}"
  printf '\n'
  exit 0
fi
printf 'Judge bind address: http://%s:%s\n' "$judge_host" "$judge_port" >&2
printf 'Judge client endpoint: %s\n' "$judge_advertise_url" >&2
# The watcher exits if this process exits. exec keeps vLLM as the foreground
# process so signals and exit status retain their original behavior.
python3 "$judge_endpoint_helper" watch "$$" "$judge_host" "$judge_port" "$judge_advertise_url" &
exec "${judge_command[@]}" "$@"

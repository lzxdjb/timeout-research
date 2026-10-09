#!/usr/bin/env bash
# Standalone vLLM service. No installation, downloads, or training side effects.
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${SWE_REPEAT_JUDGE_GPUS:-0,1,2,3}"
judge_model_path="${SWE_REPEAT_JUDGE_CHECKPOINT:-/cpfs01/thscc/sharestorage/iwc/HithinkGPT/models/Qwen3.5-122}"
judge_tp="${SWE_REPEAT_JUDGE_TP:-4}"
judge_context="${SWE_AGENT_REPEAT_JUDGE_MAX_CONTEXT:-131072}"
judge_host="${SWE_REPEAT_JUDGE_HOST:-0.0.0.0}"
judge_port="${SWE_REPEAT_JUDGE_PORT:-18090}"
[[ -f "$judge_model_path/config.json" ]] || { echo "Checkpoint config not found: $judge_model_path" >&2; exit 2; }
IFS=',' read -ra judge_gpus <<< "$CUDA_VISIBLE_DEVICES"
[[ "$judge_tp" =~ ^[1-9][0-9]*$ && "${#judge_gpus[@]}" -eq "$judge_tp" ]] || { echo "GPU list length must equal TP" >&2; exit 2; }
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
  --language-model-only --generation-config vllm
  --default-chat-template-kwargs '{"enable_thinking":false}')
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

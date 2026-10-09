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

# 0.0.0.0 is a bind address only. A client in another pod must use this
# explicitly configured pod IP, service DNS name, or externally published URL.
detect_advertise_host() {
  local ip
  for ip in $(hostname -i 2>/dev/null || true); do
    if [[ "$ip" != 127.* && "$ip" != "::1" && "$ip" != 0.0.0.0 ]]; then
      printf '%s' "$ip"
      return 0
    fi
  done
  printf '%s' "127.0.0.1"
}
judge_advertise_host="${SWE_REPEAT_JUDGE_ADVERTISE_HOST:-$(detect_advertise_host)}"
if [[ "$judge_advertise_host" == *:* && "$judge_advertise_host" != \[*\] ]]; then
  judge_advertise_host="[$judge_advertise_host]"
fi
judge_advertise_url="${SWE_REPEAT_JUDGE_ADVERTISE_URL:-http://${judge_advertise_host}:${judge_port}}"
if ! python3 - "$judge_advertise_url" <<'PY'
import ipaddress
import sys
from urllib.parse import urlsplit

try:
    url = urlsplit(sys.argv[1])
    if (url.scheme not in {"http", "https"} or not url.hostname or
            url.username is not None or url.password is not None or
            url.query or url.fragment or any(c.isspace() for c in sys.argv[1])):
        raise ValueError("expected an HTTP(S) endpoint without credentials/query/fragment")
    if url.port is not None and not 1 <= url.port <= 65535:
        raise ValueError("invalid port")
    try:
        ip = ipaddress.ip_address(url.hostname)
    except ValueError:
        ip = None  # A DNS name is allowed; routing is verified from each client.
    if ip is not None and ip.is_unspecified:
        raise ValueError("0.0.0.0/:: are listen addresses, not client endpoints")
except ValueError as exc:
    print("Invalid judge advertised URL: " + str(exc), file=sys.stderr)
    sys.exit(2)
PY
then
  exit 2
fi
judge_command=(vllm serve "$judge_model_path"
  --served-model-name "${SWE_AGENT_REPEAT_JUDGE_MODEL:-swe-repeat-judge}"
  --host "$judge_host" --port "$judge_port"
  --tensor-parallel-size "$judge_tp" --dtype bfloat16
  --max-model-len "$judge_context" --max-num-seqs "${SWE_REPEAT_JUDGE_MAX_SEQS:-4}"
  --gpu-memory-utilization "${SWE_REPEAT_JUDGE_MEMORY_FRACTION:-0.90}"
  --language-model-only --generation-config vllm
  --default-chat-template-kwargs '{"enable_thinking":false}')
printf 'Judge bind address: http://%s:%s\n' "$judge_host" "$judge_port" >&2
printf 'Judge client endpoint: %s\n' "$judge_advertise_url" >&2
printf 'Set SWE_AGENT_REPEAT_JUDGE_URL to that endpoint on every client; routing is not verified by startup.\n' >&2
if [[ "$judge_host" == 127.* || "$judge_host" == "::1" || "$judge_host" == localhost ]]; then
  printf 'Warning: loopback bind prevents direct cross-pod connections; use SWE_REPEAT_JUDGE_HOST=0.0.0.0.\n' >&2
fi
if [[ -z "${SWE_REPEAT_JUDGE_ADVERTISE_HOST:-}" && -z "${SWE_REPEAT_JUDGE_ADVERTISE_URL:-}" ]]; then
  printf 'Advertised address was auto-detected; set SWE_REPEAT_JUDGE_ADVERTISE_URL explicitly for NAT or cross-cluster routing.\n' >&2
fi
if [[ "${1:-}" == "--dry-run" ]]; then
  printf 'CUDA_VISIBLE_DEVICES=%q ' "$CUDA_VISIBLE_DEVICES"
  printf '%q ' "${judge_command[@]}"
  printf '\n'
  exit 0
fi
exec "${judge_command[@]}" "$@"

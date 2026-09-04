#!/usr/bin/env bash
set -euo pipefail

TARGET_VERL="${TARGET_VERL:-/cpfs01/nlp/leizhengxing/swe/verl}"
SWE_SOURCE="${SWE_SOURCE:-/cpfs01/nlp/leizhengxing/swe/stock-rl-reflect}"
MODEL_PATH="${MODEL_PATH:-/cpfs01/nlp/leizhengxing/stock-rl-reflect/data/Qwen3.5-35-A3B}"

# Use quoted Hydra lists, for example:
#   TRAIN_FILES='["/path/train.parquet"]'
#   VAL_FILES='["/path/val.parquet"]'
: "${TRAIN_FILES:?Set TRAIN_FILES to a Hydra list of training parquet paths}"
: "${VAL_FILES:?Set VAL_FILES to a Hydra list of validation parquet paths}"
: "${SWE_AGENT_EXECUTION_URLS:?Set SWE_AGENT_EXECUTION_URLS to the execution-service URL list}"

[[ -d "$TARGET_VERL/verl" ]] || { echo "Target verl checkout not found: $TARGET_VERL" >&2; exit 2; }
[[ -f "$SWE_SOURCE/recipe/swe_agent/agent_loop.py" ]] || {
  echo "Source SWE agent not found: $SWE_SOURCE/recipe/swe_agent" >&2
  exit 2
}
[[ -e "$MODEL_PATH" ]] || { echo "Model path not found: $MODEL_PATH" >&2; exit 2; }

SWE_SOURCE_REVISION="unknown"
SWE_SOURCE_AGENT_STATE="unknown"
if git -C "$SWE_SOURCE" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  SWE_SOURCE_REVISION="$(git -C "$SWE_SOURCE" rev-parse HEAD)"
  if [[ -n "$(git -C "$SWE_SOURCE" status --porcelain -- recipe/swe_agent)" ]]; then
    SWE_SOURCE_AGENT_STATE="dirty"
  else
    SWE_SOURCE_AGENT_STATE="clean"
  fi
fi
if [[ -n "${SWE_SOURCE_EXPECTED_COMMIT:-}" ]]; then
  expected_revision="$(git -C "$SWE_SOURCE" rev-parse "${SWE_SOURCE_EXPECTED_COMMIT}^{commit}" 2>/dev/null)" || {
    echo "Cannot resolve SWE_SOURCE_EXPECTED_COMMIT=${SWE_SOURCE_EXPECTED_COMMIT}" >&2
    exit 2
  }
  if [[ "$SWE_SOURCE_REVISION" != "$expected_revision" ]]; then
    echo "SWE source revision mismatch: expected $expected_revision, found $SWE_SOURCE_REVISION" >&2
    exit 2
  fi
  if [[ "$SWE_SOURCE_AGENT_STATE" != "clean" ]]; then
    echo "SWE source recipe has uncommitted changes while SWE_SOURCE_EXPECTED_COMMIT is set." >&2
    exit 2
  fi
fi

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export CUDA_PATH="${CUDA_PATH:-$CUDA_HOME}"
export PATH="$CUDA_HOME/bin:$PATH"

# The target checkout must precede the source checkout: verl comes from the
# context-parallel target, while recipe.swe_agent comes from stock-rl-reflect.
export PYTHONPATH="$TARGET_VERL:$SWE_SOURCE${PYTHONPATH:+:$PYTHONPATH}"
export HF_MODEL_PATH="$MODEL_PATH"
export train_path="$TRAIN_FILES"
export test_path="$VAL_FILES"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export VLLM_ALLREDUCE_USE_SYMM_MEM="${VLLM_ALLREDUCE_USE_SYMM_MEM:-0}"

# Keep GRPO normalization stable and expose compact per-step invariants.
export VERL_GRPO_INVARIANT_CHECK="${VERL_GRPO_INVARIANT_CHECK:-1}"
export VERL_GRPO_INVALID_GROUP_POLICY="${VERL_GRPO_INVALID_GROUP_POLICY:-zero}"
export VERL_GRPO_DIAGNOSTICS="${VERL_GRPO_DIAGNOSTICS:-1}"

export DEVICE="${DEVICE:-gpu}"
export TP="${TP:-2}"
export PP="${PP:-1}"
export CP="${CP:-2}"
export EP="${EP:-8}"
export ETP="${ETP:-1}"
export GEN_TP="${GEN_TP:-2}"
export ALL_OFFLOAD="${ALL_OFFLOAD:-True}"
export NDEVICES_PER_NODE="${NDEVICES_PER_NODE:-8}"

export SWE_AGENT_EXECUTION_BACKEND="${SWE_AGENT_EXECUTION_BACKEND:-remote}"
export SWE_AGENT_EXECUTION_BYPASS_PROXY="${SWE_AGENT_EXECUTION_BYPASS_PROXY:-1}"
export SWE_AGENT_EXECUTION_HTTP_TIMEOUT="${SWE_AGENT_EXECUTION_HTTP_TIMEOUT:-2100}"
export SWE_AGENT_EXECUTION_CLAIM_HTTP_TIMEOUT_SECONDS="${SWE_AGENT_EXECUTION_CLAIM_HTTP_TIMEOUT_SECONDS:-900}"
export SWE_AGENT_EXECUTION_EXECUTE_HTTP_TIMEOUT_SECONDS="${SWE_AGENT_EXECUTION_EXECUTE_HTTP_TIMEOUT_SECONDS:-750}"
export SWE_AGENT_EXECUTION_REWARD_HTTP_TIMEOUT_SECONDS="${SWE_AGENT_EXECUTION_REWARD_HTTP_TIMEOUT_SECONDS:-750}"
export SWE_AGENT_EXECUTION_RELEASE_HTTP_TIMEOUT_SECONDS="${SWE_AGENT_EXECUTION_RELEASE_HTTP_TIMEOUT_SECONDS:-60}"
export SWE_AGENT_EXECUTION_PREFETCH_HTTP_TIMEOUT_SECONDS="${SWE_AGENT_EXECUTION_PREFETCH_HTTP_TIMEOUT_SECONDS:-60}"
export SWE_AGENT_EXECUTION_OPERATION_POLL_HTTP_TIMEOUT_SECONDS="${SWE_AGENT_EXECUTION_OPERATION_POLL_HTTP_TIMEOUT_SECONDS:-30}"
export SWE_AGENT_EXECUTION_OPERATION_POLL_INTERVAL_SECONDS="${SWE_AGENT_EXECUTION_OPERATION_POLL_INTERVAL_SECONDS:-1}"
export SWE_AGENT_EXECUTION_TRAINING_HARD_TIMEOUT_SECONDS="${SWE_AGENT_EXECUTION_TRAINING_HARD_TIMEOUT_SECONDS:-600}"
export SWE_AGENT_EXECUTION_VALIDATION_HARD_TIMEOUT_SECONDS="${SWE_AGENT_EXECUTION_VALIDATION_HARD_TIMEOUT_SECONDS:-900}"
export SWE_AGENT_ROLLOUT_TRAINING_TRAJECTORY_TIMEOUT_SECONDS="${SWE_AGENT_ROLLOUT_TRAINING_TRAJECTORY_TIMEOUT_SECONDS:-2100}"
export SWE_AGENT_ROLLOUT_VALIDATION_TRAJECTORY_TIMEOUT_SECONDS="${SWE_AGENT_ROLLOUT_VALIDATION_TRAJECTORY_TIMEOUT_SECONDS:-400}"

for completion_ratio_name in \
  SWE_AGENT_ROLLOUT_TRAINING_COMPLETION_RATIO_THRESHOLD \
  SWE_AGENT_ROLLOUT_VALIDATION_COMPLETION_RATIO_THRESHOLD; do
  if [[ -v "$completion_ratio_name" ]]; then
    python - "$completion_ratio_name" "${!completion_ratio_name}" <<'PY'
import math
import sys

name, raw_value = sys.argv[1:]
try:
    value = float(raw_value)
except ValueError as exc:
    raise SystemExit(f"{name} must be numeric: {raw_value!r}") from exc
if not math.isfinite(value) or not 0.0 < value <= 1.0:
    raise SystemExit(f"{name} must be greater than 0 and at most 1: {raw_value!r}")
PY
  fi
done
unset completion_ratio_name
export SWE_AGENT_TRAINING_VERIFICATION_REWARD_SHAPING="${SWE_AGENT_TRAINING_VERIFICATION_REWARD_SHAPING:-1}"
export SWE_AGENT_TRAINING_VERIFICATION_PENALTY="${SWE_AGENT_TRAINING_VERIFICATION_PENALTY:-0.1}"
export SWE_AGENT_TRAINING_VERIFICATION_WINDOW_TURNS="${SWE_AGENT_TRAINING_VERIFICATION_WINDOW_TURNS:-2}"
export SWE_AGENT_EXECUTION_IMAGE_AFFINITY="${SWE_AGENT_EXECUTION_IMAGE_AFFINITY:-1}"
export SWE_AGENT_IMAGE_PREFETCH_REPLICATION_FACTOR="${SWE_AGENT_IMAGE_PREFETCH_REPLICATION_FACTOR:-1}"
export SWE_AGENT_REMOTE_MAX_OUTPUT_CHARS="${SWE_AGENT_REMOTE_MAX_OUTPUT_CHARS:-8000}"

# Image-only prefetch is nonblocking. Lazy claim remains the fallback.
export SWE_AGENT_TRAINER_INTEGRATED_PREWARM=0
export SWE_AGENT_TRAINING_IMAGE_PREFETCH="${SWE_AGENT_TRAINING_IMAGE_PREFETCH:-1}"
export SWE_AGENT_TRAINING_IMAGE_PREFETCH_PARALLELISM="${SWE_AGENT_TRAINING_IMAGE_PREFETCH_PARALLELISM:-2}"
export SWE_AGENT_TRAINING_IMAGE_PREFETCH_CHUNK_SIZE="${SWE_AGENT_TRAINING_IMAGE_PREFETCH_CHUNK_SIZE:-24}"
export SWE_AGENT_TRAINING_IMAGE_PREFETCH_LOOKAHEAD_BATCHES="${SWE_AGENT_TRAINING_IMAGE_PREFETCH_LOOKAHEAD_BATCHES:-1}"
export SWE_AGENT_TRAINING_IMAGE_PREFETCH_STRICT="${SWE_AGENT_TRAINING_IMAGE_PREFETCH_STRICT:-0}"
export SWE_AGENT_TRAINING_IMAGE_PREFETCH_TIMEOUT="${SWE_AGENT_TRAINING_IMAGE_PREFETCH_TIMEOUT:-60}"
export SWE_AGENT_TRAINING_IMAGE_PREFETCH_RETRY_TIMEOUT_SECONDS="${SWE_AGENT_TRAINING_IMAGE_PREFETCH_RETRY_TIMEOUT_SECONDS:-600}"
export SWE_AGENT_TRAINING_IMAGE_PREFETCH_RETRY_INITIAL_SECONDS="${SWE_AGENT_TRAINING_IMAGE_PREFETCH_RETRY_INITIAL_SECONDS:-1}"
export SWE_AGENT_TRAINING_IMAGE_PREFETCH_RETRY_MAX_SECONDS="${SWE_AGENT_TRAINING_IMAGE_PREFETCH_RETRY_MAX_SECONDS:-30}"
export SWE_AGENT_VALIDATION_IMAGE_PREFETCH="${SWE_AGENT_VALIDATION_IMAGE_PREFETCH:-1}"
export SWE_AGENT_VALIDATION_IMAGE_PREFETCH_PARALLELISM="${SWE_AGENT_VALIDATION_IMAGE_PREFETCH_PARALLELISM:-2}"
export SWE_AGENT_VALIDATION_IMAGE_PREFETCH_CHUNK_SIZE="${SWE_AGENT_VALIDATION_IMAGE_PREFETCH_CHUNK_SIZE:-24}"
export SWE_AGENT_VALIDATION_IMAGE_PREFETCH_LOOKAHEAD_BATCHES="${SWE_AGENT_VALIDATION_IMAGE_PREFETCH_LOOKAHEAD_BATCHES:-1}"
export SWE_AGENT_VALIDATION_IMAGE_PREFETCH_STRICT="${SWE_AGENT_VALIDATION_IMAGE_PREFETCH_STRICT:-0}"
export SWE_AGENT_VALIDATION_IMAGE_PREFETCH_TIMEOUT="${SWE_AGENT_VALIDATION_IMAGE_PREFETCH_TIMEOUT:-60}"
export SWE_AGENT_VALIDATION_IMAGE_PREFETCH_RETRY_TIMEOUT_SECONDS="${SWE_AGENT_VALIDATION_IMAGE_PREFETCH_RETRY_TIMEOUT_SECONDS:-600}"
export SWE_AGENT_VALIDATION_IMAGE_PREFETCH_RETRY_INITIAL_SECONDS="${SWE_AGENT_VALIDATION_IMAGE_PREFETCH_RETRY_INITIAL_SECONDS:-1}"
export SWE_AGENT_VALIDATION_IMAGE_PREFETCH_RETRY_MAX_SECONDS="${SWE_AGENT_VALIDATION_IMAGE_PREFETCH_RETRY_MAX_SECONDS:-30}"

# Preserve the source SWE admission and lifecycle retry behavior.
export SWE_AGENT_ROLLOUT_CLAIM_RETRY="${SWE_AGENT_ROLLOUT_CLAIM_RETRY:-1}"
export SWE_AGENT_ROLLOUT_CLAIM_RETRY_TIMEOUT_SECONDS="${SWE_AGENT_ROLLOUT_CLAIM_RETRY_TIMEOUT_SECONDS:-600}"
export SWE_AGENT_ROLLOUT_CLAIM_RETRY_INITIAL_SECONDS="${SWE_AGENT_ROLLOUT_CLAIM_RETRY_INITIAL_SECONDS:-0.5}"
export SWE_AGENT_ROLLOUT_CLAIM_RETRY_MAX_SECONDS="${SWE_AGENT_ROLLOUT_CLAIM_RETRY_MAX_SECONDS:-10}"
export SWE_AGENT_ROLLOUT_CLAIM_RETRY_JITTER="${SWE_AGENT_ROLLOUT_CLAIM_RETRY_JITTER:-0.25}"
export SWE_AGENT_ROLLOUT_SERVICE_FAILOVER_RETRIES="${SWE_AGENT_ROLLOUT_SERVICE_FAILOVER_RETRIES:-2}"
export SWE_AGENT_ROLLOUT_SERVICE_FAILOVER_BACKOFF_SECONDS="${SWE_AGENT_ROLLOUT_SERVICE_FAILOVER_BACKOFF_SECONDS:-1}"
export SWE_AGENT_ROLLOUT_SERVICE_FAILOVER_BACKOFF_MAX_SECONDS="${SWE_AGENT_ROLLOUT_SERVICE_FAILOVER_BACKOFF_MAX_SECONDS:-10}"
export SWE_AGENT_ROLLOUT_MAX_CONCURRENT_CLAIM_HTTP_REQUESTS="${SWE_AGENT_ROLLOUT_MAX_CONCURRENT_CLAIM_HTTP_REQUESTS:-8}"
export SWE_AGENT_ROLLOUT_EXECUTE_CAPACITY_RETRY="${SWE_AGENT_ROLLOUT_EXECUTE_CAPACITY_RETRY:-1}"
export SWE_AGENT_ROLLOUT_EXECUTE_CAPACITY_RETRY_TIMEOUT_SECONDS="${SWE_AGENT_ROLLOUT_EXECUTE_CAPACITY_RETRY_TIMEOUT_SECONDS:-900}"
export SWE_AGENT_ROLLOUT_EXECUTE_CAPACITY_RETRY_INITIAL_SECONDS="${SWE_AGENT_ROLLOUT_EXECUTE_CAPACITY_RETRY_INITIAL_SECONDS:-0.5}"
export SWE_AGENT_ROLLOUT_EXECUTE_CAPACITY_RETRY_MAX_SECONDS="${SWE_AGENT_ROLLOUT_EXECUTE_CAPACITY_RETRY_MAX_SECONDS:-10}"
export SWE_AGENT_ROLLOUT_EXECUTE_CAPACITY_RETRY_JITTER="${SWE_AGENT_ROLLOUT_EXECUTE_CAPACITY_RETRY_JITTER:-0.25}"
export SWE_AGENT_ROLLOUT_REWARD_RETRY="${SWE_AGENT_ROLLOUT_REWARD_RETRY:-1}"
export SWE_AGENT_ROLLOUT_REWARD_RETRY_TIMEOUT_SECONDS="${SWE_AGENT_ROLLOUT_REWARD_RETRY_TIMEOUT_SECONDS:-900}"
export SWE_AGENT_ROLLOUT_REWARD_RETRY_INITIAL_SECONDS="${SWE_AGENT_ROLLOUT_REWARD_RETRY_INITIAL_SECONDS:-0.5}"
export SWE_AGENT_ROLLOUT_REWARD_RETRY_MAX_SECONDS="${SWE_AGENT_ROLLOUT_REWARD_RETRY_MAX_SECONDS:-10}"
export SWE_AGENT_ROLLOUT_REWARD_RETRY_JITTER="${SWE_AGENT_ROLLOUT_REWARD_RETRY_JITTER:-0.25}"
export SWE_AGENT_ROLLOUT_RELEASE_RETRY="${SWE_AGENT_ROLLOUT_RELEASE_RETRY:-1}"
export SWE_AGENT_ROLLOUT_RELEASE_RETRY_TIMEOUT_SECONDS="${SWE_AGENT_ROLLOUT_RELEASE_RETRY_TIMEOUT_SECONDS:-300}"
export SWE_AGENT_ROLLOUT_RELEASE_RETRY_INITIAL_SECONDS="${SWE_AGENT_ROLLOUT_RELEASE_RETRY_INITIAL_SECONDS:-0.5}"
export SWE_AGENT_ROLLOUT_RELEASE_RETRY_MAX_SECONDS="${SWE_AGENT_ROLLOUT_RELEASE_RETRY_MAX_SECONDS:-10}"
export SWE_AGENT_ROLLOUT_RELEASE_RETRY_JITTER="${SWE_AGENT_ROLLOUT_RELEASE_RETRY_JITTER:-0.25}"
export SWE_AGENT_ROLLOUT_CLAIM_STRICT="${SWE_AGENT_ROLLOUT_CLAIM_STRICT:-0}"
export SWE_AGENT_ROLLOUT_REWARD_STRICT="${SWE_AGENT_ROLLOUT_REWARD_STRICT:-0}"
export SWE_AGENT_ROLLOUT_RELEASE_STRICT="${SWE_AGENT_ROLLOUT_RELEASE_STRICT:-0}"

PROJECT_NAME="${PROJECT_NAME:-35A3B-SWE-RL}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3_5_35b_a3b_swe_cp2}"
OUTPUT_DIR="${OUTPUT_DIR:-$SWE_SOURCE/output/checkpoints/$EXPERIMENT_NAME}"
SWE_AGENT_MAX_TURNS="${SWE_AGENT_MAX_TURNS:-70}"
MAX_TOOL_RESPONSE_LENGTH="${MAX_TOOL_RESPONSE_LENGTH:-8000}"

# Prompt/response lengths define the maximum sequence accepted by the model.
# PPO/log-prob token budgets are independent knobs because lowering them makes
# the same batch run as more, smaller dynamic micro-batches. Lowercase variables
# are supported for consistency with the other launch scripts; uppercase aliases
# are accepted for environment-based configuration.
max_prompt_length="${max_prompt_length:-${MAX_PROMPT_LENGTH:-8192}}"
max_response_length="${max_response_length:-${MAX_RESPONSE_LENGTH:-40000}}"
ppo_max_token_len_per_gpu="${ppo_max_token_len_per_gpu:-${PPO_MAX_TOKEN_LEN_PER_GPU:-32768}}"
rollout_log_prob_max_token_len_per_gpu="${rollout_log_prob_max_token_len_per_gpu:-${ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-$ppo_max_token_len_per_gpu}}"
ref_log_prob_max_token_len_per_gpu="${ref_log_prob_max_token_len_per_gpu:-${REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-$ppo_max_token_len_per_gpu}}"
if ! [[ "$max_prompt_length" =~ ^[0-9]+$ ]] || (( 10#$max_prompt_length <= 0 )); then
  echo "max_prompt_length must be a positive integer: $max_prompt_length" >&2
  exit 2
fi
if ! [[ "$max_response_length" =~ ^[0-9]+$ ]] || (( 10#$max_response_length <= 0 )); then
  echo "max_response_length must be a positive integer: $max_response_length" >&2
  exit 2
fi
if ! [[ "$ppo_max_token_len_per_gpu" =~ ^[0-9]+$ ]] || (( 10#$ppo_max_token_len_per_gpu <= 0 )); then
  echo "ppo_max_token_len_per_gpu must be a positive integer: $ppo_max_token_len_per_gpu" >&2
  exit 2
fi
if ! [[ "$rollout_log_prob_max_token_len_per_gpu" =~ ^[0-9]+$ ]] || (( 10#$rollout_log_prob_max_token_len_per_gpu <= 0 )); then
  echo "rollout_log_prob_max_token_len_per_gpu must be a positive integer: $rollout_log_prob_max_token_len_per_gpu" >&2
  exit 2
fi
if ! [[ "$ref_log_prob_max_token_len_per_gpu" =~ ^[0-9]+$ ]] || (( 10#$ref_log_prob_max_token_len_per_gpu <= 0 )); then
  echo "ref_log_prob_max_token_len_per_gpu must be a positive integer: $ref_log_prob_max_token_len_per_gpu" >&2
  exit 2
fi
max_sequence_length=$((10#$max_prompt_length + 10#$max_response_length))

SWE_AGENT_HARNESS_PROFILE="${SWE_AGENT_HARNESS_PROFILE:-baseline}"
export SWE_AGENT_HARNESS_PROFILE
if [[ -z "${SWE_AGENT_TOOL_CONFIG_PATH:-}" ]]; then
  if [[ "$SWE_AGENT_HARNESS_PROFILE" == "claude_like" ]]; then
    SWE_AGENT_TOOL_CONFIG_PATH="$SWE_SOURCE/recipe/swe_agent/config/tool_config_claude_like.yaml"
  else
    SWE_AGENT_TOOL_CONFIG_PATH="$SWE_SOURCE/recipe/swe_agent/config/tool_config.yaml"
  fi
fi
[[ -f "$SWE_AGENT_TOOL_CONFIG_PATH" ]] || {
  echo "Tool config not found: $SWE_AGENT_TOOL_CONFIG_PATH" >&2
  exit 2
}
python - "$SWE_AGENT_TOOL_CONFIG_PATH" <<'PY'
import sys
from pathlib import Path

import yaml

path = Path(sys.argv[1])
try:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
except Exception as exc:
    raise SystemExit(f"Invalid tool config YAML: {path}: {exc}")
if not isinstance(config, dict) or not isinstance(config.get("tools"), list):
    raise SystemExit(f"Invalid tool config shape: {path} (expected a top-level tools list)")
print(f"Tool config valid: {path} ({len(config['tools'])} tools)")
PY

overrides=(
  data.train_files="$TRAIN_FILES"
  data.val_files="$VAL_FILES"
  data.train_batch_size=64
  data.max_prompt_length="$max_prompt_length"
  data.max_response_length="$max_response_length"
  data.return_raw_chat=True
  actor_rollout_ref.model.enable_gradient_checkpointing=True
  actor_rollout_ref.model.use_remove_padding=True
  actor_rollout_ref.actor.optim.lr=3e-6
  actor_rollout_ref.actor.ppo_mini_batch_size=64
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$ppo_max_token_len_per_gpu"
  actor_rollout_ref.actor.use_dynamic_bsz=True
  actor_rollout_ref.actor.megatron.pad_bshd_to_minibatch_max=False
  actor_rollout_ref.actor.megatron.use_remove_padding=True
  actor_rollout_ref.ref.megatron.use_mbridge=True
  actor_rollout_ref.ref.megatron.vanilla_mbridge=True
  actor_rollout_ref.ref.megatron.use_remove_padding=True
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
  actor_rollout_ref.rollout.n=8
  actor_rollout_ref.rollout.gpu_memory_utilization=0.80
  actor_rollout_ref.rollout.max_num_seqs=2048
  actor_rollout_ref.rollout.max_num_batched_tokens="$max_sequence_length"
  actor_rollout_ref.rollout.max_model_len="$max_sequence_length"
  actor_rollout_ref.rollout.enforce_eager=True
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="$rollout_log_prob_max_token_len_per_gpu"
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="$ref_log_prob_max_token_len_per_gpu"
  actor_rollout_ref.rollout.multi_turn.enable=True
  actor_rollout_ref.rollout.multi_turn.format=qwen3_coder
  actor_rollout_ref.rollout.multi_turn.tool_config_path="$SWE_AGENT_TOOL_CONFIG_PATH"
  actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns="$SWE_AGENT_MAX_TURNS"
  actor_rollout_ref.rollout.multi_turn.max_user_turns="$SWE_AGENT_MAX_TURNS"
  actor_rollout_ref.rollout.multi_turn.max_tool_response_length="$MAX_TOOL_RESPONSE_LENGTH"
  actor_rollout_ref.rollout.agent.default_agent_loop=swe_tool_agent
  actor_rollout_ref.rollout.agent.agent_loop_config_path="$SWE_SOURCE/recipe/swe_agent/config/agent_loop_config.yaml"
  reward.custom_reward_function.path="$SWE_SOURCE/recipe/swe_agent/reward_function.py"
  reward.custom_reward_function.name=compute_score
  +reward.custom_reward_function.reward_kwargs.return_dict=True
  trainer.use_v1=False
  trainer.project_name="$PROJECT_NAME"
  trainer.experiment_name="$EXPERIMENT_NAME"
  trainer.default_local_dir="$OUTPUT_DIR"
  ray_kwargs.ray_init.num_cpus="${RAY_NUM_CPUS:-128}"
  algorithm.filter_groups.enable=False
  +actor_rollout_ref.rollout.engine_kwargs.vllm.gdn_prefill_backend=triton
)

echo "Target verl: $TARGET_VERL"
echo "SWE source:  $SWE_SOURCE ($SWE_SOURCE_REVISION, recipe=$SWE_SOURCE_AGENT_STATE)"
echo "Model:       $MODEL_PATH"
echo "Harness:     $SWE_AGENT_HARNESS_PROFILE"
echo "Sequence lengths: prompt=$max_prompt_length response=$max_response_length total=$max_sequence_length"
echo "Token budgets per GPU: ppo=$ppo_max_token_len_per_gpu rollout_log_prob=$rollout_log_prob_max_token_len_per_gpu ref_log_prob=$ref_log_prob_max_token_len_per_gpu"
echo "Parallelism: TP=$TP PP=$PP CP=$CP EP=$EP ETP=$ETP GEN_TP=$GEN_TP"
echo "Prefetch:    training=$SWE_AGENT_TRAINING_IMAGE_PREFETCH validation=$SWE_AGENT_VALIDATION_IMAGE_PREFETCH"
echo "GRPO checks: invariant=$VERL_GRPO_INVARIANT_CHECK policy=$VERL_GRPO_INVALID_GROUP_POLICY diagnostics=$VERL_GRPO_DIAGNOSTICS"
echo "Timeouts:    trajectory(train/val)=$SWE_AGENT_ROLLOUT_TRAINING_TRAJECTORY_TIMEOUT_SECONDS/$SWE_AGENT_ROLLOUT_VALIDATION_TRAJECTORY_TIMEOUT_SECONDS hard(train/val)=$SWE_AGENT_EXECUTION_TRAINING_HARD_TIMEOUT_SECONDS/$SWE_AGENT_EXECUTION_VALIDATION_HARD_TIMEOUT_SECONDS http(claim/execute/reward/release/prefetch)=$SWE_AGENT_EXECUTION_CLAIM_HTTP_TIMEOUT_SECONDS/$SWE_AGENT_EXECUTION_EXECUTE_HTTP_TIMEOUT_SECONDS/$SWE_AGENT_EXECUTION_REWARD_HTTP_TIMEOUT_SECONDS/$SWE_AGENT_EXECUTION_RELEASE_HTTP_TIMEOUT_SECONDS/$SWE_AGENT_EXECUTION_PREFETCH_HTTP_TIMEOUT_SECONDS"
echo "Completion ratios: training=${SWE_AGENT_ROLLOUT_TRAINING_COMPLETION_RATIO_THRESHOLD:-disabled} validation=${SWE_AGENT_ROLLOUT_VALIDATION_COMPLETION_RATIO_THRESHOLD:-disabled}"
echo "Verification: shaping=$SWE_AGENT_TRAINING_VERIFICATION_REWARD_SHAPING penalty=$SWE_AGENT_TRAINING_VERIFICATION_PENALTY window_turns=$SWE_AGENT_TRAINING_VERIFICATION_WINDOW_TURNS (training only)"

cd "$TARGET_VERL"
if [[ "${CONFIG_ONLY:-0}" == "1" ]]; then
  exec bash examples/grpo_trainer/run_qwen3_5_35b_megatron.sh "${overrides[@]}" "$@" --cfg job --resolve
fi

exec bash examples/grpo_trainer/run_qwen3_5_35b_megatron.sh "${overrides[@]}" "$@"

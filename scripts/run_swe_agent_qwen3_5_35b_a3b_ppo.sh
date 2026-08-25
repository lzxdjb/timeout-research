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

# Keep the existing rollout diagnostics available while PPO uses GAE.
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
export SWE_AGENT_ROLLOUT_VALIDATION_TRAJECTORY_TIMEOUT_SECONDS="${SWE_AGENT_ROLLOUT_VALIDATION_TRAJECTORY_TIMEOUT_SECONDS:-2100}"
export SWE_AGENT_TRAINING_VERIFICATION_REWARD_SHAPING="${SWE_AGENT_TRAINING_VERIFICATION_REWARD_SHAPING:-1}"
export SWE_AGENT_TRAINING_VERIFICATION_PENALTY="${SWE_AGENT_TRAINING_VERIFICATION_PENALTY:-0.1}"
export SWE_AGENT_TRAINING_VERIFICATION_WINDOW_TURNS="${SWE_AGENT_TRAINING_VERIFICATION_WINDOW_TURNS:-2}"
export SWE_AGENT_TASK_FILTER_TRAINING_ENABLED="${SWE_AGENT_TASK_FILTER_TRAINING_ENABLED:-1}"
export SWE_AGENT_TASK_FILTER_INCLUDE_QUEUE_FAILURES="${SWE_AGENT_TASK_FILTER_INCLUDE_QUEUE_FAILURES:-1}"
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

PROJECT_NAME="${PROJECT_NAME:-35A3B-SWE-PPO}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3_5_35b_a3b_swe_ppo}"
OUTPUT_DIR="${OUTPUT_DIR:-$SWE_SOURCE/output/checkpoints/$EXPERIMENT_NAME}"
SWE_AGENT_MAX_TURNS="${SWE_AGENT_MAX_TURNS:-70}"
MAX_TOOL_RESPONSE_LENGTH="${MAX_TOOL_RESPONSE_LENGTH:-8000}"

overrides=(
  data.train_files="$TRAIN_FILES"
  data.val_files="$VAL_FILES"
  data.train_batch_size=512
  data.max_prompt_length=8192
  data.max_response_length=40000
  data.return_raw_chat=True
  actor_rollout_ref.model.enable_gradient_checkpointing=True
  actor_rollout_ref.model.use_remove_padding=True
  actor_rollout_ref.actor.optim.lr=3e-6
  actor_rollout_ref.actor.use_kl_loss=False
  actor_rollout_ref.actor.ppo_mini_batch_size=512
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=48192
  actor_rollout_ref.actor.use_dynamic_bsz=True
  actor_rollout_ref.actor.megatron.pad_bshd_to_minibatch_max=False
  actor_rollout_ref.actor.megatron.use_remove_padding=True
  actor_rollout_ref.ref.megatron.use_mbridge=True
  actor_rollout_ref.ref.megatron.vanilla_mbridge=True
  actor_rollout_ref.ref.megatron.use_remove_padding=True
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
  actor_rollout_ref.rollout.n=1
  actor_rollout_ref.rollout.gpu_memory_utilization=0.80
  actor_rollout_ref.rollout.max_num_seqs=2048
  actor_rollout_ref.rollout.max_model_len=48192
  actor_rollout_ref.rollout.enforce_eager=True
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=48192
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=48192
  actor_rollout_ref.rollout.multi_turn.enable=True
  actor_rollout_ref.rollout.multi_turn.format=qwen3_coder
  actor_rollout_ref.rollout.multi_turn.tool_config_path="$SWE_SOURCE/recipe/swe_agent/config/tool_config.yaml"
  actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns="$SWE_AGENT_MAX_TURNS"
  actor_rollout_ref.rollout.multi_turn.max_user_turns="$SWE_AGENT_MAX_TURNS"
  actor_rollout_ref.rollout.multi_turn.max_tool_response_length="$MAX_TOOL_RESPONSE_LENGTH"
  actor_rollout_ref.rollout.agent.default_agent_loop=swe_tool_agent
  actor_rollout_ref.rollout.agent.agent_loop_config_path="$SWE_SOURCE/recipe/swe_agent/config/agent_loop_config.yaml"
  reward.custom_reward_function.path="$SWE_SOURCE/recipe/swe_agent/reward_function.py"
  reward.custom_reward_function.name=compute_score
  +reward.custom_reward_function.reward_kwargs.return_dict=True
  algorithm.adv_estimator=gae
  algorithm.gamma=1.0
  algorithm.lam=1.0
  algorithm.use_kl_in_reward=False
  algorithm.filter_groups.enable=False
  critic.model.path="$MODEL_PATH"
  critic.enable=True
  critic.model.trust_remote_code=True
  critic.model.enable_gradient_checkpointing=True
  critic.model.use_remove_padding=True
  critic.optim.lr=1e-5
  +critic.optim.override_optimizer_config.optimizer_offload_fraction=1
  +critic.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True
  +critic.optim.override_optimizer_config.use_precision_aware_optimizer=True
  +critic.optim.override_optimizer_config.optimizer_cpu_offload=True
  critic.ppo_mini_batch_size=64
  critic.ppo_micro_batch_size_per_gpu=1
  critic.ppo_max_token_len_per_gpu=48192
  critic.forward_max_token_len_per_gpu=48192
  critic.use_dynamic_bsz=True
  critic.megatron.use_mbridge=True
  critic.megatron.vanilla_mbridge=True
  critic.megatron.use_remove_padding=True
  critic.megatron.tensor_model_parallel_size="$TP"
  critic.megatron.pipeline_model_parallel_size="$PP"
  critic.megatron.context_parallel_size="$CP"
  critic.megatron.expert_model_parallel_size="$EP"
  critic.megatron.expert_tensor_parallel_size="$ETP"
  critic.megatron.param_offload="$ALL_OFFLOAD"
  critic.megatron.optimizer_offload="$ALL_OFFLOAD"
  critic.megatron.grad_offload="$ALL_OFFLOAD"
  trainer.balance_batch=True
  trainer.critic_warmup=0
  trainer.use_v1=False
  trainer.project_name="$PROJECT_NAME"
  trainer.experiment_name="$EXPERIMENT_NAME"
  trainer.default_local_dir="$OUTPUT_DIR"
  ray_kwargs.ray_init.num_cpus="${RAY_NUM_CPUS:-128}"
  +actor_rollout_ref.rollout.engine_kwargs.vllm.gdn_prefill_backend=triton
)

echo "Target verl: $TARGET_VERL"
echo "SWE source:  $SWE_SOURCE ($SWE_SOURCE_REVISION, recipe=$SWE_SOURCE_AGENT_STATE)"
echo "Model:       $MODEL_PATH"
echo "Parallelism: TP=$TP PP=$PP CP=$CP EP=$EP ETP=$ETP GEN_TP=$GEN_TP"
echo "Prefetch:    training=$SWE_AGENT_TRAINING_IMAGE_PREFETCH validation=$SWE_AGENT_VALIDATION_IMAGE_PREFETCH"
echo "PPO: adv_estimator=gae rollout.n=1 critic_model=$MODEL_PATH infra_filter=$SWE_AGENT_TASK_FILTER_TRAINING_ENABLED"
echo "Timeouts:    trajectory(train/val)=$SWE_AGENT_ROLLOUT_TRAINING_TRAJECTORY_TIMEOUT_SECONDS/$SWE_AGENT_ROLLOUT_VALIDATION_TRAJECTORY_TIMEOUT_SECONDS hard(train/val)=$SWE_AGENT_EXECUTION_TRAINING_HARD_TIMEOUT_SECONDS/$SWE_AGENT_EXECUTION_VALIDATION_HARD_TIMEOUT_SECONDS http(claim/execute/reward/release/prefetch)=$SWE_AGENT_EXECUTION_CLAIM_HTTP_TIMEOUT_SECONDS/$SWE_AGENT_EXECUTION_EXECUTE_HTTP_TIMEOUT_SECONDS/$SWE_AGENT_EXECUTION_REWARD_HTTP_TIMEOUT_SECONDS/$SWE_AGENT_EXECUTION_RELEASE_HTTP_TIMEOUT_SECONDS/$SWE_AGENT_EXECUTION_PREFETCH_HTTP_TIMEOUT_SECONDS"
echo "Verification: shaping=$SWE_AGENT_TRAINING_VERIFICATION_REWARD_SHAPING penalty=$SWE_AGENT_TRAINING_VERIFICATION_PENALTY window_turns=$SWE_AGENT_TRAINING_VERIFICATION_WINDOW_TURNS (training only)"

cd "$TARGET_VERL"
if [[ "${CONFIG_ONLY:-0}" == "1" ]]; then
  exec bash examples/ppo_trainer/run_qwen3_5_35b_megatron.sh "${overrides[@]}" "$@" --cfg job --resolve
fi

exec bash examples/ppo_trainer/run_qwen3_5_35b_megatron.sh "${overrides[@]}" "$@"

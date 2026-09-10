#!/usr/bin/env bash
set -euo pipefail

TARGET_VERL="${TARGET_VERL:-/cpfs01/nlp/leizhengxing/swe/verl}"
SWE_SOURCE="${SWE_SOURCE:-/cpfs01/nlp/leizhengxing/swe/stock-rl-reflect}"
ASYNC_DEPS="${ASYNC_DEPS:-/cpfs01/nlp/leizhengxing/verl-async-deps}"
MODEL_PATH="${MODEL_PATH:-/cpfs01/nlp/leizhengxing/stock-rl-reflect/data/Qwen3.5-35-A3B}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# Use quoted Hydra lists, for example:
#   TRAIN_FILES='["/path/train.parquet"]'
#   VAL_FILES='["/path/val.parquet"]'
: "${TRAIN_FILES:?Set TRAIN_FILES to a Hydra list of training parquet paths}"
: "${VAL_FILES:?Set VAL_FILES to a Hydra list of validation parquet paths}"
: "${SWE_AGENT_EXECUTION_URLS:?Set SWE_AGENT_EXECUTION_URLS to the execution-service URL list}"

[[ -d "$TARGET_VERL/verl" ]] || { echo "Target verl checkout not found: $TARGET_VERL" >&2; exit 2; }
[[ -d "$ASYNC_DEPS" ]] || { echo "Async dependency path not found: $ASYNC_DEPS" >&2; exit 2; }
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

# Async-only dependencies come first, followed by the target verl and the
# external SWE recipe checkout.
export PYTHONPATH="$ASYNC_DEPS:$TARGET_VERL:$SWE_SOURCE${PYTHONPATH:+:$PYTHONPATH}"
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

export TP="${TP:-2}"
export PP="${PP:-1}"
export CP="${CP:-2}"
export EP="${EP:-4}"
export ETP="${ETP:-1}"
export GEN_TP="${GEN_TP:-2}"
export TRAIN_NNODES="${TRAIN_NNODES:-1}"
export ROLLOUT_NNODES="${ROLLOUT_NNODES:-1}"
export NGPUS_PER_NODE="${NGPUS_PER_NODE:-4}"

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
export SWE_AGENT_ROLLOUT_TRAINING_TRAJECTORY_TIMEOUT_SECONDS="${SWE_AGENT_ROLLOUT_TRAINING_TRAJECTORY_TIMEOUT_SECONDS:-2400}"
export SWE_AGENT_ROLLOUT_VALIDATION_TRAJECTORY_TIMEOUT_SECONDS="${SWE_AGENT_ROLLOUT_VALIDATION_TRAJECTORY_TIMEOUT_SECONDS:-3600}"
export SWE_AGENT_TRAINING_VERIFICATION_REWARD_SHAPING="${SWE_AGENT_TRAINING_VERIFICATION_REWARD_SHAPING:-1}"
export SWE_AGENT_TRAINING_VERIFICATION_PENALTY="${SWE_AGENT_TRAINING_VERIFICATION_PENALTY:-0.1}"
export SWE_AGENT_TRAINING_VERIFICATION_WINDOW_TURNS="${SWE_AGENT_TRAINING_VERIFICATION_WINDOW_TURNS:-2}"
export SWE_AGENT_TRAINING_VERIFICATION_AUX_V1_ENABLED="${SWE_AGENT_TRAINING_VERIFICATION_AUX_V1_ENABLED:-0}"
export SWE_AGENT_TRAINING_VERIFICATION_LOOP_AUX_REWARD_ENABLED="${SWE_AGENT_TRAINING_VERIFICATION_LOOP_AUX_REWARD_ENABLED:-0}"
export SWE_AGENT_TRAINING_VERIFICATION_LOOP_AUX_REWARD_MAX="${SWE_AGENT_TRAINING_VERIFICATION_LOOP_AUX_REWARD_MAX:-0.01}"
export SWE_AGENT_TRAINING_VERIFICATION_DIVERSITY_AUX_REWARD_ENABLED="${SWE_AGENT_TRAINING_VERIFICATION_DIVERSITY_AUX_REWARD_ENABLED:-0}"
export SWE_AGENT_TRAINING_VERIFICATION_DIVERSITY_AUX_REWARD_MAX="${SWE_AGENT_TRAINING_VERIFICATION_DIVERSITY_AUX_REWARD_MAX:-0.005}"
export SWE_AGENT_EXECUTION_IMAGE_AFFINITY="${SWE_AGENT_EXECUTION_IMAGE_AFFINITY:-1}"
export SWE_AGENT_IMAGE_PREFETCH_REPLICATION_FACTOR="${SWE_AGENT_IMAGE_PREFETCH_REPLICATION_FACTOR:-1}"
export SWE_AGENT_REMOTE_MAX_OUTPUT_CHARS="${SWE_AGENT_REMOTE_MAX_OUTPUT_CHARS:-8000}"

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

export SWE_AGENT_ROLLOUT_CLAIM_RETRY="${SWE_AGENT_ROLLOUT_CLAIM_RETRY:-1}"
export SWE_AGENT_ROLLOUT_CLAIM_RETRY_TIMEOUT_SECONDS="${SWE_AGENT_ROLLOUT_CLAIM_RETRY_TIMEOUT_SECONDS:-600}"
export SWE_AGENT_ROLLOUT_CLAIM_RETRY_INITIAL_SECONDS="${SWE_AGENT_ROLLOUT_CLAIM_RETRY_INITIAL_SECONDS:-0.5}"
export SWE_AGENT_ROLLOUT_CLAIM_RETRY_MAX_SECONDS="${SWE_AGENT_ROLLOUT_CLAIM_RETRY_MAX_SECONDS:-10}"
export SWE_AGENT_ROLLOUT_CLAIM_RETRY_JITTER="${SWE_AGENT_ROLLOUT_CLAIM_RETRY_JITTER:-0.25}"
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

PROJECT_NAME="${PROJECT_NAME:-35A3B-SWE-RL-ASYNC}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3_5_35b_a3b_swe_async_cp2}"
OUTPUT_DIR="${OUTPUT_DIR:-$SWE_SOURCE/output/checkpoints/$EXPERIMENT_NAME}"
SWE_AGENT_MAX_TURNS="${SWE_AGENT_MAX_TURNS:-40}"
MAX_TOOL_RESPONSE_LENGTH="${MAX_TOOL_RESPONSE_LENGTH:-8000}"
ASYNC_CONCURRENT_SAMPLES_PER_REPLICA="${ASYNC_CONCURRENT_SAMPLES_PER_REPLICA:-2}"

overrides=(
  algorithm.adv_estimator=grpo
  algorithm.use_kl_in_reward=False
  +reward.custom_reward_function.reward_kwargs.return_dict=True
  algorithm.filter_groups.enable=True
  algorithm.filter_groups.metric=acc
  algorithm.filter_groups.max_inflight_gen_batches=1
  data.train_files="$TRAIN_FILES"
  data.val_files="$VAL_FILES"
  data.train_batch_size=0
  data.gen_batch_size=1
  data.val_batch_size=200
  data.max_prompt_length=8192
  data.max_response_length=40000
  data.truncation=left
  data.return_raw_chat=True
  actor_rollout_ref.model.path="$MODEL_PATH"
  actor_rollout_ref.model.trust_remote_code=True
  actor_rollout_ref.model.enable_gradient_checkpointing=True
  actor_rollout_ref.model.use_remove_padding=True
  actor_rollout_ref.model.use_fused_kernels=True
  actor_rollout_ref.actor.optim.lr=3e-6
  actor_rollout_ref.actor.optim.lr_decay_steps=3000
  actor_rollout_ref.actor.optim.weight_decay=0.1
  +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1
  +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True
  +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True
  +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True
  actor_rollout_ref.actor.ppo_mini_batch_size=64
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=48192
  actor_rollout_ref.actor.use_dynamic_bsz=True
  actor_rollout_ref.actor.use_kl_loss=True
  actor_rollout_ref.actor.kl_loss_coef=0.01
  actor_rollout_ref.actor.kl_loss_type=low_var_kl
  actor_rollout_ref.actor.entropy_coeff=0
  actor_rollout_ref.actor.use_rollout_log_probs=True
  actor_rollout_ref.actor.megatron.use_mbridge=True
  actor_rollout_ref.actor.megatron.vanilla_mbridge=True
  actor_rollout_ref.actor.megatron.use_remove_padding=True
  actor_rollout_ref.actor.megatron.pad_bshd_to_minibatch_max=False
  actor_rollout_ref.actor.megatron.tensor_model_parallel_size="$TP"
  actor_rollout_ref.actor.megatron.pipeline_model_parallel_size="$PP"
  actor_rollout_ref.actor.megatron.context_parallel_size="$CP"
  actor_rollout_ref.actor.megatron.expert_model_parallel_size="$EP"
  actor_rollout_ref.actor.megatron.expert_tensor_parallel_size="$ETP"
  actor_rollout_ref.actor.megatron.param_offload=True
  actor_rollout_ref.actor.megatron.grad_offload=True
  actor_rollout_ref.actor.megatron.optimizer_offload=True
  actor_rollout_ref.actor.megatron.dtype=bfloat16
  actor_rollout_ref.actor.megatron.virtual_pipeline_model_parallel_size=null
  actor_rollout_ref.actor.megatron.override_transformer_config.attention_backend=auto
  actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
  actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
  actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
  +actor_rollout_ref.actor.megatron.override_transformer_config.moe_aux_loss_coeff=0.01
  +actor_rollout_ref.actor.megatron.override_transformer_config.moe_z_loss_coeff=0.001
  +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=True
  +actor_rollout_ref.actor.megatron.override_transformer_config.moe_grouped_gemm=True
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=48192
  actor_rollout_ref.ref.megatron.use_mbridge=True
  actor_rollout_ref.ref.megatron.vanilla_mbridge=True
  actor_rollout_ref.ref.megatron.use_remove_padding=True
  actor_rollout_ref.ref.megatron.tensor_model_parallel_size="$TP"
  actor_rollout_ref.ref.megatron.pipeline_model_parallel_size="$PP"
  actor_rollout_ref.ref.megatron.context_parallel_size="$CP"
  actor_rollout_ref.ref.megatron.expert_model_parallel_size="$EP"
  actor_rollout_ref.ref.megatron.expert_tensor_parallel_size="$ETP"
  actor_rollout_ref.ref.megatron.param_offload=True
  actor_rollout_ref.hybrid_engine=False
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.mode=async
  actor_rollout_ref.rollout.n=8
  actor_rollout_ref.rollout.tensor_model_parallel_size="$GEN_TP"
  actor_rollout_ref.rollout.gpu_memory_utilization=0.80
  actor_rollout_ref.rollout.max_num_seqs=2048
  actor_rollout_ref.rollout.max_model_len=48192
  actor_rollout_ref.rollout.enforce_eager=True
  actor_rollout_ref.rollout.free_cache_engine=True
  actor_rollout_ref.rollout.calculate_log_probs=True
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=48192
  actor_rollout_ref.rollout.val_kwargs.n=1
  actor_rollout_ref.rollout.val_kwargs.do_sample=True
  actor_rollout_ref.rollout.disable_log_stats=False
  actor_rollout_ref.rollout.multi_turn.enable=True
  actor_rollout_ref.rollout.multi_turn.format=qwen3_coder
  actor_rollout_ref.rollout.multi_turn.tool_config_path="$SWE_SOURCE/recipe/swe_agent/config/tool_config.yaml"
  actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns="$SWE_AGENT_MAX_TURNS"
  actor_rollout_ref.rollout.multi_turn.max_user_turns="$SWE_AGENT_MAX_TURNS"
  actor_rollout_ref.rollout.multi_turn.max_tool_response_length="$MAX_TOOL_RESPONSE_LENGTH"
  actor_rollout_ref.rollout.agent.default_agent_loop=swe_tool_agent
  actor_rollout_ref.rollout.agent.agent_loop_config_path="$SWE_SOURCE/recipe/swe_agent/config/agent_loop_config.yaml"
  actor_rollout_ref.nccl_timeout=9600
  reward.custom_reward_function.path="$SWE_SOURCE/recipe/swe_agent/reward_function.py"
  reward.custom_reward_function.name=compute_score
  trainer.logger='[console,wandb]'
  trainer.project_name="$PROJECT_NAME"
  trainer.experiment_name="$EXPERIMENT_NAME"
  trainer.default_local_dir="$OUTPUT_DIR"
  trainer.nnodes="$TRAIN_NNODES"
  trainer.n_gpus_per_node="$NGPUS_PER_NODE"
  trainer.total_epochs=15
  trainer.val_before_train=False
  trainer.test_freq=5
  trainer.save_freq=20
  trainer.max_actor_ckpt_to_keep=1
  trainer.resume_mode=auto
  rollout.nnodes="$ROLLOUT_NNODES"
  rollout.n_gpus_per_node="$NGPUS_PER_NODE"
  rollout.total_rollout_steps=3000
  async_training.use_dynamic_resource_scheduling=True
  async_training.staleness_threshold=0.5
  async_training.trigger_parameter_sync_step=1
  async_training.require_batches=1
  async_training.partial_rollout=True
  async_training.concurrent_samples_per_replica="$ASYNC_CONCURRENT_SAMPLES_PER_REPLICA"
  ray_kwargs.ray_init.num_cpus="${RAY_NUM_CPUS:-128}"
  +actor_rollout_ref.rollout.engine_kwargs.vllm.gdn_prefill_backend=triton
)

echo "Target verl:       $TARGET_VERL"
echo "Async deps:        $ASYNC_DEPS"
echo "SWE source:        $SWE_SOURCE ($SWE_SOURCE_REVISION, recipe=$SWE_SOURCE_AGENT_STATE)"
echo "Model:             $MODEL_PATH"
echo "Parallelism:       TP=$TP PP=$PP CP=$CP EP=$EP ETP=$ETP GEN_TP=$GEN_TP"
echo "Resources:         train=${TRAIN_NNODES}x${NGPUS_PER_NODE} rollout=${ROLLOUT_NNODES}x${NGPUS_PER_NODE}"
echo "Async concurrency: $ASYNC_CONCURRENT_SAMPLES_PER_REPLICA samples/replica, rollout.n=8"
echo "Prefetch:          training=$SWE_AGENT_TRAINING_IMAGE_PREFETCH validation=$SWE_AGENT_VALIDATION_IMAGE_PREFETCH"
echo "GRPO checks:       invariant=$VERL_GRPO_INVARIANT_CHECK policy=$VERL_GRPO_INVALID_GROUP_POLICY diagnostics=$VERL_GRPO_DIAGNOSTICS"
echo "Async claims:      max_http=$SWE_AGENT_ROLLOUT_MAX_CONCURRENT_CLAIM_HTTP_REQUESTS retry_timeout=$SWE_AGENT_ROLLOUT_CLAIM_RETRY_TIMEOUT_SECONDS"
echo "Timeouts:          trajectory(train/val)=$SWE_AGENT_ROLLOUT_TRAINING_TRAJECTORY_TIMEOUT_SECONDS/$SWE_AGENT_ROLLOUT_VALIDATION_TRAJECTORY_TIMEOUT_SECONDS hard(train/val)=$SWE_AGENT_EXECUTION_TRAINING_HARD_TIMEOUT_SECONDS/$SWE_AGENT_EXECUTION_VALIDATION_HARD_TIMEOUT_SECONDS http(claim/execute/reward/release/prefetch)=$SWE_AGENT_EXECUTION_CLAIM_HTTP_TIMEOUT_SECONDS/$SWE_AGENT_EXECUTION_EXECUTE_HTTP_TIMEOUT_SECONDS/$SWE_AGENT_EXECUTION_REWARD_HTTP_TIMEOUT_SECONDS/$SWE_AGENT_EXECUTION_RELEASE_HTTP_TIMEOUT_SECONDS/$SWE_AGENT_EXECUTION_PREFETCH_HTTP_TIMEOUT_SECONDS"
echo "Verification:      shaping=$SWE_AGENT_TRAINING_VERIFICATION_REWARD_SHAPING penalty=$SWE_AGENT_TRAINING_VERIFICATION_PENALTY window_turns=$SWE_AGENT_TRAINING_VERIFICATION_WINDOW_TURNS (training only)"
echo "Verification auxiliary rewards: v1=$SWE_AGENT_TRAINING_VERIFICATION_AUX_V1_ENABLED loop=$SWE_AGENT_TRAINING_VERIFICATION_LOOP_AUX_REWARD_ENABLED/$SWE_AGENT_TRAINING_VERIFICATION_LOOP_AUX_REWARD_MAX diversity=$SWE_AGENT_TRAINING_VERIFICATION_DIVERSITY_AUX_REWARD_ENABLED/$SWE_AGENT_TRAINING_VERIFICATION_DIVERSITY_AUX_REWARD_MAX"

command=(
  "$PYTHON_BIN"
  -X faulthandler
  -m verl.experimental.fully_async_policy.fully_async_main
  --config-path=config
  --config-name=fully_async_ppo_megatron_trainer.yaml
)
if [[ "${CONFIG_ONLY:-0}" == "1" ]]; then
  command+=(--cfg job --resolve)
fi

cd "$TARGET_VERL"
exec "${command[@]}" "${overrides[@]}" "$@"

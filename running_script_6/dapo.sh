cd /cpfs01/thscc/sharestorage/iwc/HithinkOmni/user_workspace/leizhengxing/leizhengxing/swe/verl



WANDB_API_KEY=${WANDB_API_KEY:-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}
wandb login "$WANDB_API_KEY"



export SWE_AGENT_TASK_FILTER_TRAINING_ENABLED=1
export SWE_AGENT_TASK_FILTER_INFRA_RATIO_THRESHOLD=0.25
export SWE_AGENT_TASK_FILTER_MIN_GROUP_ATTEMPTS=8
export SWE_AGENT_TASK_FILTER_INCLUDE_QUEUE_FAILURES=1 ##### should set like that !!!!! this will also filter the data which is failed due to the queue

export SWE_AGENT_TRAINING_VERIFICATION_REWARD_SHAPING=1
export SWE_AGENT_TRAINING_VERIFICATION_PENALTY=0.1
export SWE_AGENT_TRAINING_VERIFICATION_WINDOW_TURNS=2

# Verification output parsing and workspace fingerprinting are diagnostics by
# default; only an observed mutation or failed check invalidates freshness.
export SWE_AGENT_VERIFICATION_REQUIRE_MEANINGFUL_TESTS=0
export SWE_AGENT_VERIFICATION_UNKNOWN_STATE_POLICY=allow
export SWE_AGENT_TRAINING_PROTOCOL_REWARD_SHAPING=1
export SWE_AGENT_TRAINING_PROTOCOL_SUBMISSION_ONLY=1
export SWE_AGENT_TRAINING_SUBMISSION_PENALTY=0.1

# export SWE_AGENT_TRAINING_REPEATED_TOOL_REWARD_SHAPING=1
# export SWE_AGENT_TRAINING_REPEATED_TOOL_PENALTY=0.1

export TRAIN_FILES='["/cpfs01/thscc/sharestorage/iwc/HithinkOmni/user_workspace/leizhengxing/leizhengxing/swe/stock-rl-reflect/data/swe_benchmarks/verl/swe_rebench_v2_reference_verified_snapshot_2000.parquet"]'
export VAL_FILES='["/cpfs01/thscc/sharestorage/iwc/HithinkOmni/user_workspace/leizhengxing/leizhengxing/swe/stock-rl-reflect/data/swe_benchmarks/verl/swe_bench_verified_reference_verified_repeat2_500_images_v1.parquet"]'

# export VAL_FILES='["/cpfs01/thscc/sharestorage/iwc/HithinkOmni/user_workspace/leizhengxing/leizhengxing/swe/stock-rl-reflect/data/5_terminate/terminal_bench_2_0.parquet"]'

export SWE_AGENT_EXECUTION_URLS="http://10.248.103.91:18080,http://10.248.101.232:18080,http://10.248.100.13:18080,http://10.248.103.97:18080,http://10.248.102.171:18080,http://10.248.103.96:18080,http://10.248.103.92:18080" #### 1-7
export SWE_AGENT_ROLLOUT_CLAIM_EXECUTOR_WORKERS=3
export SWE_AGENT_HEALTH_CHECK_TIMEOUT_SECONDS="${SWE_AGENT_HEALTH_CHECK_TIMEOUT_SECONDS:-5}"



export GEN_TP=2

# export SWE_AGENT_ROLLOUT_TRAINING_COMPLETION_RATIO_THRESHOLD=0.85
unset SWE_AGENT_ROLLOUT_TRAINING_COMPLETION_RATIO_THRESHOLD

# Explicitly disable the whole-trajectory deadline.
export SWE_AGENT_ROLLOUT_TRAINING_TRAJECTORY_TIMEOUT_SECONDS=0

export SWE_AGENT_ROLLOUT_TRAINING_INFRA_FAILURE_GROUP_RATIO_THRESHOLD=0.25
export SWE_AGENT_ROLLOUT_TRAINING_UNVERIFIED_TRAJECTORY_RATIO_THRESHOLD=0.25

##### validation
export SWE_AGENT_ROLLOUT_VALIDATION_MAX_GROUP_RETRIES=10
export SWE_AGENT_ROLLOUT_VALIDATION_COMPLETION_RATIO_THRESHOLD=0.98
export SWE_AGENT_ROLLOUT_VALIDATION_INFRA_FAILURE_GROUP_RATIO_THRESHOLD=0.25
export SWE_AGENT_VALIDATION_STRICT_ACCOUNTING=1
export SWE_AGENT_VALIDATION_EXPECTED_TASKS=357

# export SWE_AGENT_ROLLOUT_VALIDATION_COMPLETION_RATIO_THRESHOLD=0.95
# unset SWE_AGENT_ROLLOUT_VALIDATION_INFRA_FAILURE_GROUP_RATIO_THRESHOLD
# unset SWE_AGENT_ROLLOUT_VALIDATION_COMPLETION_RATIO_THRESHOLD

export SWE_AGENT_ROLLOUT_VALIDATION_UNVERIFIED_TRAJECTORY_RATIO_THRESHOLD=0.25
export SWE_AGENT_HARNESS_PROFILE=claude_like
export SWE_AGENT_MAX_TURNS=300


export EXPERIMENT_NAME=dapo_token_loss



export ROLLOUT_DATA_DIR="/cpfs01/thscc/sharestorage/iwc/HithinkOmni/user_workspace/leizhengxing/leizhengxing/swe/verl/rollout_data/${EXPERIMENT_NAME}"
export VALIDATION_DATA_DIR="/cpfs01/thscc/sharestorage/iwc/HithinkOmni/user_workspace/leizhengxing/leizhengxing/swe/verl/validation_data/${EXPERIMENT_NAME}"
export AUDIO_BENCHMARK_OUTPUT_DIR="/cpfs01/thscc/sharestorage/iwc/HithinkOmni/user_workspace/leizhengxing/leizhengxing/swe/verl/validation_data/audio/${EXPERIMENT_NAME}"
mkdir -p "$ROLLOUT_DATA_DIR" "$VALIDATION_DATA_DIR"

export TP=2
export CP=4
export EP=8
export LD_LIBRARY_PATH="/usr/local/cuda/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PPO_MAX_TOKEN_LEN_PER_GPU=65536
export ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU=65536
export REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU=65536
export MAX_RESPONSE_LENGTH=253952
# export MAX_RESPONSE_LENGTH=98304

export MODEL_PATH=/cpfs01/thscc/sharestorage/iwc/HithinkOmni/user_workspace/leizhengxing/leizhengxing/swe/stock-rl-reflect/data/3552-checkpoint
export AUDIO_BENCHMARK_DATA_ROOT=/cpfs01/thscc/sharestorage/iwc/HithinkOmni/user_workspace/leizhengxing/leizhengxing/swe/stock-rl-reflect/data/restored/audio_benchmark
export VLLM_PLUGINS=verl_qwen35_omni


export TARGET_VERL=/cpfs01/thscc/sharestorage/iwc/HithinkOmni/user_workspace/leizhengxing/leizhengxing/swe/verl
# python3 -m pip install --no-deps -e "$TARGET_VERL" --break-system-packages

# export SWE_AGENT_TRAINING_PARTIAL_HIDDEN_TEST_REWARD=1

# Recommended starting point: increase only the affected benchmarks.
# export AUDIO_BENCHMARK_MAX_NEW_TOKENS=512
export AUDIO_BENCHMARK_CHARTQA_MAX_NEW_TOKENS=512
export AUDIO_BENCHMARK_MMSTAR_MAX_NEW_TOKENS=512

export SWE_AGENT_ROLLOUT_CLAIM_RETRY=1
export SWE_AGENT_ROLLOUT_CLAIM_RETRY_TIMEOUT_SECONDS=7200

############
# Enable conservative tool-failure penalties.
export SWE_AGENT_TRAINING_TOOL_FAILURE_PENALTY_ENABLED=1



# Optional: these values are already the defaults.
export SWE_AGENT_TRAINING_TOOL_FAILURE_PENALTY_COST=0.02
export SWE_AGENT_TRAINING_TOOL_FAILURE_PENALTY_CAP=0.10
export SWE_AGENT_TRAINING_TOOL_FAILURE_PENALTY_WARMUP_STEPS=0

export SWE_AGENT_TRAINING_THINKING_REPEAT_PENALTY_ENABLED=1
export SWE_AGENT_TRAINING_THINKING_REPEAT_PENALTY_COST=0.02
export SWE_AGENT_TRAINING_THINKING_REPEAT_PENALTY_CAP=0.10
export SWE_AGENT_TRAINING_THINKING_REPEAT_PENALTY_WARMUP_STEPS=0

export SWE_AGENT_TRAINING_THINKING_REPEAT_AGGREGATE_ENABLED=1
export SWE_AGENT_TRAINING_THINKING_REPEAT_CROSS_TURN_ENABLED=1

export SWE_AGENT_TRAINING_TOOL_REPEAT_PENALTY_ENABLED=1
export SWE_AGENT_TRAINING_TOOL_REPEAT_PENALTY_NOOP_WRITES=1
export SWE_AGENT_TRAINING_TOOL_REPEAT_PENALTY_SHELL_READS=1
export SWE_AGENT_TRAINING_TOOL_REPEAT_PENALTY_SHELL_MIN_COPIES=20
export SWE_AGENT_TOOL_REPEAT_SHADOW_ENABLED=1

##### new python tests detect
export SWE_AGENT_TRAINING_TOOL_REPEAT_PENALTY_PYTHON_TESTS=1
export SWE_AGENT_TRAINING_TOOL_REPEAT_PENALTY_PYTHON_TEST_PROTECTED_CALLS=20
export SWE_AGENT_TOOL_REPEAT_PYTHON_TEST_SHADOW_ENABLED=1
#####

export SWE_AGENT_TRAINING_TOOL_REPEAT_PENALTY_COST=0.02
export SWE_AGENT_TRAINING_TOOL_REPEAT_PENALTY_CAP=0.10
export SWE_AGENT_TRAINING_TOOL_REPEAT_PENALTY_WARMUP_STEPS=0

export SWE_AGENT_TRAINING_TOOL_REPEAT_PENALTY_NOOP_EDITS=1
export SWE_AGENT_TRAINING_TOOL_REPEAT_PENALTY_UNCHANGED_READS=1
export SWE_AGENT_TRAINING_TOOL_REPEAT_PENALTY_READ_CYCLES=1
export SWE_AGENT_TRAINING_TOOL_REPEAT_PENALTY_CONTRACT_REJECTIONS=1
export SWE_AGENT_TRAINING_TOOL_REPEAT_PENALTY_VARIED_NOOP_EDITS=1


####### new timeout command 
export SWE_AGENT_EXECUTION_TRAINING_HARD_TIMEOUT_SECONDS=1200

bash scripts/run_swe_agent_qwen3_5_35b_a3b.sh \
  trainer.rollout_data_dir="$ROLLOUT_DATA_DIR" \
  trainer.validation_data_dir="$VALIDATION_DATA_DIR" \
  trainer.val_before_train=False \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.70 \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.max_num_seqs=512 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
  actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
  actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
  trainer.use_v1=True \
  actor_rollout_ref.actor.megatron.moe_loss_respects_train_sample_mask=True \
  actor_rollout_ref.rollout.max_model_len=262144 \
  actor_rollout_ref.rollout.max_num_batched_tokens=32678 \
  actor_rollout_ref.rollout.max_num_seqs=512 \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.kl_loss_coef=0.0 \
  algorithm.use_kl_in_reward=False \
  algorithm.kl_ctrl.kl_coef=0.0 \
  actor_rollout_ref.model.use_fused_kernels=True \
  trainer.save_freq=1 \
  trainer.test_freq=2 \
  actor_rollout_ref.rollout.enable_audio=True \
  actor_rollout_ref.rollout.limit_audio=4 \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.allowed_local_media_path="$AUDIO_BENCHMARK_DATA_ROOT" \
  +trainer.audio_benchmark.enabled=True \
  +trainer.audio_benchmark.gage_root=/cpfs01/thscc/sharestorage/iwc/HithinkOmni/user_workspace/leizhengxing/leizhengxing/gage-leizhengxing \
  +trainer.audio_benchmark.data_root="$AUDIO_BENCHMARK_DATA_ROOT" \
  +trainer.audio_benchmark.benchmarks='[aishell2,chartqa,clothoaqa,mmbench,mmstar,ocrbench,realworldqa,rul_muchomusic,voicebench_bbh]' \
  +trainer.audio_benchmark.max_samples=100 \
  +trainer.audio_benchmark.concurrency=32 \
  +trainer.audio_benchmark.check_mode=light \
  +trainer.audio_benchmark.failure_policy=warn \
  +data.apply_chat_template_kwargs.enable_thinking=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  algorithm.use_kl_in_reward=False \
  algorithm.kl_ctrl.kl_coef=0.0 \
  actor_rollout_ref.actor.loss_agg_mode=token-mean \
  +trainer.audio_benchmark.output_dir="$AUDIO_BENCHMARK_OUTPUT_DIR" \
  algorithm.filter_groups.cutoff_when_full=True \
algorithm.filter_groups.requeue_unselected_prompts=True \
algorithm.filter_groups.cutoff_cleanup_timeout_seconds=300 \
  algorithm.filter_groups.enable=True \
algorithm.filter_groups.metric=score \
algorithm.filter_groups.max_inflight_gen_batches=2 \
actor_rollout_ref.actor.clip_ratio_low=0.2 \
actor_rollout_ref.actor.clip_ratio_high=0.28
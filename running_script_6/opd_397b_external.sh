#!/usr/bin/env bash
# Text SWE on-policy distillation from the existing FP8 Qwen3.5-397B server.
# Start the same execution-service Docker fleet used by three_penalty_baseline.sh.
# This script launches training in the foreground and never starts a teacher.
#
# Launch:
#   nohup bash running_script_6/opd_397b_external.sh > opd_397b_external.log 2>&1 &
# Inspect resolved configuration without starting workers:
#   CONFIG_ONLY=1 bash running_script_6/opd_397b_external.sh
# Mix hidden-reward GRPO with OPD (unset an inherited hidden-reward skip flag):
#   SWE_AGENT_OPD_USE_TASK_REWARDS=1 SWE_AGENT_TRAINING_SKIP_HIDDEN_REWARD=0 \
#     SWE_AGENT_OPD_DISTILLATION_LOSS_COEF=1.0 bash running_script_6/opd_397b_external.sh
# Teacher serving must reserve one generated token for prompt-logprob scoring;
# the current 262143-token student budget fits a 262144-token teacher.
# Resume an OPD checkpoint explicitly (the default is a fresh optimizer):
#   bash running_script_6/opd_397b_external.sh \
#     trainer.resume_mode=resume_path trainer.resume_from_path=/path/global_step_10
# Additional Hydra arguments at the end override defaults below.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export TARGET_VERL="${TARGET_VERL:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
export SWE_SOURCE="${SWE_SOURCE:-$(cd -- "$TARGET_VERL/.." && pwd)/stock-rl-reflect}"
WORKSPACE_ROOT="$(cd -- "$TARGET_VERL/../.." && pwd)"

# Start from the audio SFT checkpoint, independently of the previous RL run.
export MODEL_PATH="${MODEL_PATH:-$SWE_SOURCE/data/3552-checkpoint}"
export TRAIN_FILES="${TRAIN_FILES:-[\"$SWE_SOURCE/data/swe_benchmarks/verl/swe_rebench_v2_reference_verified_snapshot_2000.parquet\"]}"
export VAL_FILES="${VAL_FILES:-[\"$SWE_SOURCE/data/swe_benchmarks/verl/swe_bench_verified_reference_verified_repeat2_500_images_v1.parquet\"]}"
export SWE_AGENT_EXECUTION_URLS="${SWE_AGENT_EXECUTION_URLS:-http://10.248.102.222:18080,http://10.248.103.15:18080,http://10.248.102.67:18080,http://10.248.103.68:18080,http://10.248.100.59:18080,http://10.248.100.43:18080}"

TEACHER_BASE_URL="${TEACHER_BASE_URL:-http://10.248.102.141:18090}"
TEACHER_MODEL_NAME="${TEACHER_MODEL_NAME:-swe-repeat-judge}"
TEACHER_MAX_MODEL_LEN="${TEACHER_MAX_MODEL_LEN:-262144}"
# This limit applies globally across rollout workers through the shared scorer.
TEACHER_MAX_CONCURRENT_REQUESTS="${TEACHER_MAX_CONCURRENT_REQUESTS:-4}"
TEACHER_REQUEST_TIMEOUT_S="${TEACHER_REQUEST_TIMEOUT_S:-1800}"
TEACHER_MAX_RETRIES="${TEACHER_MAX_RETRIES:-2}"
TEACHER_RETRY_BACKOFF_S="${TEACHER_RETRY_BACKOFF_S:-2}"

export EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen35_audio_swe_opd_397b_fp8}"
export PROJECT_NAME="${PROJECT_NAME:-35A3B-SWE-RL}"
export OUTPUT_DIR="${OUTPUT_DIR:-$SWE_SOURCE/output/checkpoints/$EXPERIMENT_NAME}"
export ROLLOUT_DATA_DIR="${ROLLOUT_DATA_DIR:-/cpfs01/thscc/sharestorage/iwc/HithinkOmni/user_workspace/leizhengxing/leizhengxing/swe/verl/rollout_data/$EXPERIMENT_NAME}"
export VALIDATION_DATA_DIR="${VALIDATION_DATA_DIR:-/cpfs01/thscc/sharestorage/iwc/HithinkOmni/user_workspace/leizhengxing/leizhengxing/swe/verl/validation_data/$EXPERIMENT_NAME}"
export AUDIO_BENCHMARK_OUTPUT_DIR="${AUDIO_BENCHMARK_OUTPUT_DIR:-/cpfs01/thscc/sharestorage/iwc/HithinkOmni/user_workspace/leizhengxing/leizhengxing/swe/verl/validation_data/audio/$EXPERIMENT_NAME}"

# Preserve the known working student parallelism and audio-enabled rollout.
export TP="${TP:-2}"
export CP="${CP:-4}"
export EP="${EP:-8}"
export GEN_TP="${GEN_TP:-2}"
export LD_LIBRARY_PATH="/usr/local/cuda/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export VLLM_PLUGINS="${VLLM_PLUGINS:-verl_qwen35_omni}"
export PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-65536}"
export ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU="${ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-65536}"
export REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU="${REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-65536}"
# Keep the current prompt/response limits for training and validation.
# Lowercase aliases are accepted by the shared wrapper; honor them here too.
export MAX_PROMPT_LENGTH="${max_prompt_length:-${MAX_PROMPT_LENGTH:-8192}}"
export MAX_RESPONSE_LENGTH="${max_response_length:-${MAX_RESPONSE_LENGTH:-253951}}"
max_sequence_length="$(python3 - "$MAX_PROMPT_LENGTH" "$MAX_RESPONSE_LENGTH" "$TEACHER_MAX_MODEL_LEN" <<'PY'
import sys

names = ("MAX_PROMPT_LENGTH", "MAX_RESPONSE_LENGTH", "TEACHER_MAX_MODEL_LEN")
values = []
for name, raw in zip(names, sys.argv[1:], strict=True):
    try:
        value = int(raw)
    except ValueError as exc:
        raise SystemExit(f"{name} must be a positive integer: {raw!r}") from exc
    if value <= 0:
        raise SystemExit(f"{name} must be a positive integer: {raw!r}")
    values.append(value)
prompt, response, teacher_limit = values
if prompt + response + 1 > teacher_limit:
    raise SystemExit(
        f"Student prompt ({prompt}) + response ({response}) + 1 scoring token "
        f"exceeds the teacher context ({teacher_limit})."
    )
print(prompt + response)
PY
)"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-64}"
ROLLOUT_N="${ROLLOUT_N:-8}"
ACTOR_LR="${ACTOR_LR:-1e-6}"
# Match the baseline's 15 epochs; a step limit is an explicit optional override.
TRAINER_STOP_OVERRIDES=()
if [[ -n "${TOTAL_TRAINING_STEPS:-}" ]]; then
  TRAINER_STOP_OVERRIDES+=("trainer.total_training_steps=$TOTAL_TRAINING_STEPS")
fi

# Retain infrastructure filtering; valid incorrect solutions remain training data.
export SWE_AGENT_TASK_FILTER_TRAINING_ENABLED=1
export SWE_AGENT_TASK_FILTER_MIN_GROUP_ATTEMPTS=8
export SWE_AGENT_TASK_FILTER_INCLUDE_QUEUE_FAILURES=1
export SWE_AGENT_TASK_FILTER_INFRA_RATIO_THRESHOLD=0.25
export SWE_AGENT_ROLLOUT_CLAIM_EXECUTOR_WORKERS="${SWE_AGENT_ROLLOUT_CLAIM_EXECUTOR_WORKERS:-3}"
export SWE_AGENT_HEALTH_CHECK_TIMEOUT_SECONDS="${SWE_AGENT_HEALTH_CHECK_TIMEOUT_SECONDS:-5}"
# Match the historical baseline's completion cutoff; callers can retain all groups.
export SWE_AGENT_ROLLOUT_TRAINING_COMPLETION_RATIO_THRESHOLD="${SWE_AGENT_ROLLOUT_TRAINING_COMPLETION_RATIO_THRESHOLD:-0.95}"
export SWE_AGENT_ROLLOUT_TRAINING_INFRA_FAILURE_GROUP_RATIO_THRESHOLD=0.25
export SWE_AGENT_ROLLOUT_TRAINING_UNVERIFIED_TRAJECTORY_RATIO_THRESHOLD=0.25
export SWE_AGENT_ROLLOUT_VALIDATION_MAX_GROUP_RETRIES="${SWE_AGENT_ROLLOUT_VALIDATION_MAX_GROUP_RETRIES:-10}"
export SWE_AGENT_ROLLOUT_VALIDATION_COMPLETION_RATIO_THRESHOLD="${SWE_AGENT_ROLLOUT_VALIDATION_COMPLETION_RATIO_THRESHOLD:-0.98}"
export SWE_AGENT_ROLLOUT_VALIDATION_INFRA_FAILURE_GROUP_RATIO_THRESHOLD="${SWE_AGENT_ROLLOUT_VALIDATION_INFRA_FAILURE_GROUP_RATIO_THRESHOLD:-0.25}"
export SWE_AGENT_ROLLOUT_VALIDATION_UNVERIFIED_TRAJECTORY_RATIO_THRESHOLD="${SWE_AGENT_ROLLOUT_VALIDATION_UNVERIFIED_TRAJECTORY_RATIO_THRESHOLD:-0.25}"
export SWE_AGENT_VALIDATION_STRICT_ACCOUNTING=1
export SWE_AGENT_VALIDATION_EXPECTED_TASKS="${SWE_AGENT_VALIDATION_EXPECTED_TASKS:-357}"
export SWE_AGENT_HARNESS_PROFILE="${SWE_AGENT_HARNESS_PROFILE:-claude_like}"
export SWE_AGENT_MAX_TURNS="${SWE_AGENT_MAX_TURNS:-300}"
export SWE_AGENT_EXECUTION_TRAINING_HARD_TIMEOUT_SECONDS="${SWE_AGENT_EXECUTION_TRAINING_HARD_TIMEOUT_SECONDS:-1200}"
export SWE_AGENT_ROLLOUT_CLAIM_RETRY=1
export SWE_AGENT_ROLLOUT_CLAIM_RETRY_TIMEOUT_SECONDS="${SWE_AGENT_ROLLOUT_CLAIM_RETRY_TIMEOUT_SECONDS:-7200}"

# Keep the baseline tool prompt/observations and conservative tool-failure
# penalty. Disable other auxiliary objectives, including inherited shell flags.
export SWE_AGENT_TOOL_CONFIG_PATH="$SWE_SOURCE/recipe/swe_agent/config/tool_config_claude_like.yaml"
export SWE_AGENT_EFFICIENCY_PROFILE=0
export SWE_AGENT_TOOL_COMMAND_FEEDBACK_ENABLED=0
export SWE_AGENT_TOOL_COMMAND_DIAGNOSTICS_ENABLED=1
# Pure OPD skips hidden evaluation by default; opt-in task rewards reuse GRPO.
# Validate before launching workers, including inherited skip flags.
opd_loss_settings="$(python3 - \
  "${SWE_AGENT_OPD_USE_TASK_REWARDS:-0}" \
  "${SWE_AGENT_TRAINING_SKIP_HIDDEN_REWARD:-}" \
  "${SWE_AGENT_OPD_DISTILLATION_LOSS_COEF:-1.0}" <<'PY'
import math
import sys


def boolean(name, raw):
    normalized = raw.strip().lower()
    if normalized not in {"0", "1", "false", "true"}:
        raise SystemExit(f"{name} must be 0, 1, false, or true: {raw!r}")
    return normalized in {"1", "true"}


task_rewards = boolean("SWE_AGENT_OPD_USE_TASK_REWARDS", sys.argv[1])
skip_hidden = (
    boolean("SWE_AGENT_TRAINING_SKIP_HIDDEN_REWARD", sys.argv[2]) if sys.argv[2] else not task_rewards
)
if task_rewards and skip_hidden:
    raise SystemExit(
        "SWE_AGENT_OPD_USE_TASK_REWARDS=1 requires hidden evaluation; "
        "unset SWE_AGENT_TRAINING_SKIP_HIDDEN_REWARD or set it to 0."
    )
try:
    coefficient = float(sys.argv[3])
except ValueError as exc:
    raise SystemExit("SWE_AGENT_OPD_DISTILLATION_LOSS_COEF must be finite and nonnegative.") from exc
if not math.isfinite(coefficient) or coefficient < 0:
    raise SystemExit("SWE_AGENT_OPD_DISTILLATION_LOSS_COEF must be finite and nonnegative.")
print(int(task_rewards), int(skip_hidden), coefficient, str(task_rewards))
PY
)"
read -r SWE_AGENT_OPD_USE_TASK_REWARDS SWE_AGENT_TRAINING_SKIP_HIDDEN_REWARD \
  SWE_AGENT_OPD_DISTILLATION_LOSS_COEF opd_use_task_rewards <<< "$opd_loss_settings"
export SWE_AGENT_OPD_USE_TASK_REWARDS SWE_AGENT_TRAINING_SKIP_HIDDEN_REWARD SWE_AGENT_OPD_DISTILLATION_LOSS_COEF
unset opd_loss_settings
# Cancel unfinished group RPCs at cutoff and account for cancelled work in timings.
export SWE_AGENT_OPD_CANCEL_ON_CUTOFF="${SWE_AGENT_OPD_CANCEL_ON_CUTOFF:-1}"
# Opt in to generation-only cutoff and best-effort SWE release cleanup.
# Completed generations drain hidden evaluation and queued/running teacher scoring.
export SWE_AGENT_OPD_CUTOFF_WAIT_FOR_SCORING="${SWE_AGENT_OPD_CUTOFF_WAIT_FOR_SCORING:-0}"
export SWE_AGENT_OPD_PHASE_TIMING="${SWE_AGENT_OPD_PHASE_TIMING:-1}"
export SWE_AGENT_OPD_CUTOFF_CLEANUP_TIMEOUT_SECONDS="${SWE_AGENT_OPD_CUTOFF_CLEANUP_TIMEOUT_SECONDS:-120}"
export SWE_AGENT_TRAINING_TOOL_FAILURE_PENALTY_ENABLED=1
export SWE_AGENT_TRAINING_TOOL_FAILURE_PENALTY_COST="${SWE_AGENT_TRAINING_TOOL_FAILURE_PENALTY_COST:-0.02}"
export SWE_AGENT_TRAINING_TOOL_FAILURE_PENALTY_CAP="${SWE_AGENT_TRAINING_TOOL_FAILURE_PENALTY_CAP:-0.30}"
export SWE_AGENT_TRAINING_TOOL_FAILURE_PENALTY_WARMUP_STEPS="${SWE_AGENT_TRAINING_TOOL_FAILURE_PENALTY_WARMUP_STEPS:-0}"
export SWE_AGENT_TOOL_COMMAND_SNAPSHOT_TIMEOUT_SECONDS=5.0
for opd_disabled_flag in \
  SWE_AGENT_TRAINING_PARTIAL_HIDDEN_TEST_REWARD \
  SWE_AGENT_TRAINING_VERIFICATION_REWARD_SHAPING \
  SWE_AGENT_TRAINING_PROTOCOL_REWARD_SHAPING \
  SWE_AGENT_TRAINING_PROTOCOL_SUBMISSION_ONLY \
  SWE_AGENT_TRAINING_VERIFICATION_AUX_V1_ENABLED \
  SWE_AGENT_TRAINING_VERIFICATION_LOOP_AUX_REWARD_ENABLED \
  SWE_AGENT_TRAINING_VERIFICATION_DIVERSITY_AUX_REWARD_ENABLED \
  SWE_AGENT_TRAINING_REPEATED_TOOL_REWARD_SHAPING \
  SWE_AGENT_TRAINING_TOOL_REPEAT_PENALTY_ENABLED \
  SWE_AGENT_TRAINING_THINKING_REPEAT_PENALTY_ENABLED \
  SWE_AGENT_TRAINING_THINKING_REPEAT_AGGREGATE_ENABLED \
  SWE_AGENT_TRAINING_THINKING_REPEAT_CROSS_TURN_ENABLED \
  SWE_AGENT_REPEAT_JUDGE_VALIDATION_ENABLED \
  SWE_AGENT_REPEAT_JUDGE_LOCAL_PENALTY_SHORTCUT_ENABLED \
  SWE_AGENT_TOOL_REPEAT_SHADOW_ENABLED \
  SWE_AGENT_TOOL_REPEAT_PYTHON_TEST_SHADOW_ENABLED \
  SWE_AGENT_TOOL_REPEAT_EDIT_SHADOW_ENABLED \
  SWE_AGENT_TOOL_REPEAT_COMMAND_SHADOW_ENABLED \
  SWE_AGENT_TOOL_REPEAT_SEARCH_SHADOW_ENABLED; do
  export "$opd_disabled_flag=0"
done
unset opd_disabled_flag
export SWE_AGENT_REPEAT_REWARD_MODE=off
export VERL_GRPO_GOLDEN_REPLAY_ENABLED=0

export AUDIO_BENCHMARK_DATA_ROOT="${AUDIO_BENCHMARK_DATA_ROOT:-$SWE_SOURCE/data/restored/audio_benchmark}"
export AUDIO_BENCHMARK_CHARTQA_MAX_NEW_TOKENS="${AUDIO_BENCHMARK_CHARTQA_MAX_NEW_TOKENS:-512}"
export AUDIO_BENCHMARK_MMSTAR_MAX_NEW_TOKENS="${AUDIO_BENCHMARK_MMSTAR_MAX_NEW_TOKENS:-512}"
AUDIO_BENCHMARK_ENABLED="${AUDIO_BENCHMARK_ENABLED:-True}"

if [[ "${CONFIG_ONLY:-0}" != "1" ]]; then
  mkdir -p "$ROLLOUT_DATA_DIR" "$VALIDATION_DATA_DIR" "$AUDIO_BENCHMARK_OUTPUT_DIR"
fi

cd "$TARGET_VERL"
exec bash scripts/run_swe_agent_qwen3_5_35b_a3b.sh \
  trainer.use_v1=True \
  trainer.v1.trainer_mode=sync \
  trainer.resume_mode=disable \
  trainer.val_before_train=False \
  trainer.rollout_data_dir="$ROLLOUT_DATA_DIR" \
  trainer.validation_data_dir="$VALIDATION_DATA_DIR" \
  trainer.save_freq=1 \
  trainer.test_freq=2 \
  trainer.total_epochs=15 \
  data.train_batch_size="$TRAIN_BATCH_SIZE" \
  actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.optim.lr="$ACTOR_LR" \
  actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
  actor_rollout_ref.actor.megatron.moe_loss_respects_train_sample_mask=True \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.entropy_coeff=0.0 \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  algorithm.kl_ctrl.kl_coef=0.0 \
  algorithm.filter_groups.enable=False \
  actor_rollout_ref.model.use_fused_kernels=True \
  actor_rollout_ref.rollout.n="$ROLLOUT_N" \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.70 \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.max_num_seqs=512 \
  actor_rollout_ref.rollout.max_model_len="$max_sequence_length" \
  actor_rollout_ref.rollout.max_num_batched_tokens=32678 \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.top_k=-1 \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
  actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
  actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
  actor_rollout_ref.rollout.enable_audio=True \
  actor_rollout_ref.rollout.limit_audio=4 \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.allowed_local_media_path="$AUDIO_BENCHMARK_DATA_ROOT" \
  +data.apply_chat_template_kwargs.enable_thinking=True \
  distillation.enabled=True \
  distillation.n_gpus_per_node=0 \
  distillation.nnodes=0 \
  distillation.teacher_models.teacher_model.backend=external \
  distillation.teacher_models.teacher_model.external.base_url="$TEACHER_BASE_URL" \
  distillation.teacher_models.teacher_model.external.model_name="$TEACHER_MODEL_NAME" \
  distillation.teacher_models.teacher_model.external.max_model_len="$TEACHER_MAX_MODEL_LEN" \
  distillation.teacher_models.teacher_model.external.max_concurrent_requests="$TEACHER_MAX_CONCURRENT_REQUESTS" \
  distillation.teacher_models.teacher_model.external.request_timeout_s="$TEACHER_REQUEST_TIMEOUT_S" \
  distillation.teacher_models.teacher_model.external.max_retries="$TEACHER_MAX_RETRIES" \
  distillation.teacher_models.teacher_model.external.retry_backoff_s="$TEACHER_RETRY_BACKOFF_S" \
  distillation.teacher_models.teacher_model.external.unsupported_token_ids='[248070,248071,248072,248073,248074,248075,248076]' \
  distillation.teacher_models.teacher_model.inference.max_model_len="$TEACHER_MAX_MODEL_LEN" \
  distillation.distillation_loss.loss_mode=k1 \
  distillation.distillation_loss.use_policy_gradient=True \
  distillation.distillation_loss.use_task_rewards="$opd_use_task_rewards" \
  distillation.distillation_loss.distillation_loss_coef="$SWE_AGENT_OPD_DISTILLATION_LOSS_COEF" \
  distillation.distillation_loss.loss_max_clamp=null \
  distillation.distillation_loss.log_prob_min_clamp=null \
  +trainer.audio_benchmark.enabled="$AUDIO_BENCHMARK_ENABLED" \
  +trainer.audio_benchmark.gage_root="$WORKSPACE_ROOT/gage-leizhengxing" \
  +trainer.audio_benchmark.data_root="$AUDIO_BENCHMARK_DATA_ROOT" \
  +trainer.audio_benchmark.benchmarks='[aishell2,chartqa,clothoaqa,mmbench,mmstar,ocrbench,realworldqa,rul_muchomusic,voicebench_bbh]' \
  +trainer.audio_benchmark.max_samples=100 \
  +trainer.audio_benchmark.concurrency=32 \
  +trainer.audio_benchmark.check_mode=light \
  +trainer.audio_benchmark.failure_policy=warn \
  +trainer.audio_benchmark.output_dir="$AUDIO_BENCHMARK_OUTPUT_DIR" \
  "${TRAINER_STOP_OVERRIDES[@]}" \
  "$@"

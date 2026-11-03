#!/usr/bin/env bash
# Opt-in SWE + audio OPD; audio encoder AND projector are frozen by default.
# Stage 2 opts in through opd_397b_multi_teacher_projector.sh.
# Start the same execution-service Docker fleet used by three_penalty_baseline.sh.
# This script launches training in the foreground and never starts a teacher.
#
# Launch:
#   nohup bash running_script_6/opd_397b_multi_teacher.sh > multi_teacher.log 2>&1 &
# Inspect resolved configuration without starting workers:
#   CONFIG_ONLY=1 bash running_script_6/opd_397b_multi_teacher.sh
# Run the one-step gate before beginning the curriculum:
#   bash running_script_6/opd_397b_multi_teacher_smoke.sh
# Include SWE and audio validation after that update with SMOKE_VALIDATE=1.
# Audio suite settings accept the same AUDIO_BENCHMARK_* overrides as audio_test.sh;
# see docs/advance/swe_audio_multi_teacher_opd.md for the six mounted audio suites.
# Both teacher protocols reserve one generated token for prompt-logprob scoring.
# Resume this experiment explicitly (the default is a fresh optimizer):
#   bash running_script_6/opd_397b_multi_teacher.sh \
#     trainer.resume_mode=resume_path trainer.resume_from_path=/path/global_step_10
# Additional Hydra arguments at the end override defaults below.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export TARGET_VERL="${TARGET_VERL:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
export SWE_SOURCE="${SWE_SOURCE:-$(cd -- "$TARGET_VERL/.." && pwd)/stock-rl-reflect}"
WORKSPACE_ROOT="$(cd -- "$TARGET_VERL/../.." && pwd)"

# A new mode is explicit so existing frozen launches keep their defaults.
case "${AUDIO_TRAIN_PROJECTOR:-0}" in
  0)
    audio_stage='Stage 1'
    actor_frozen_audio=True
    actor_train_audio_projector=False
    default_experiment=qwen35_swe_audio_multi_teacher_frozen_v1
    default_model="$SWE_SOURCE/output/checkpoints/qwen35_audio_swe_opd_397b_fp8/global_step_18/actor/model/huggingface"
    audio_rollout_overrides=()
    ;;
  1)
    audio_stage='Stage 2'
    actor_frozen_audio=False
    actor_train_audio_projector=True
    default_experiment=qwen35_swe_audio_multi_teacher_projector_v1
    default_model="$SWE_SOURCE/output/checkpoints/qwen35_multi_teacher_frozen_step1_audio_20261102T153523Z/global_step_1/actor/model/huggingface"
    audio_rollout_overrides=(+actor_rollout_ref.rollout.engine_kwargs.vllm.hf_overrides.verl_audio_projector_trainable=True)
    ;;
  *) echo 'AUDIO_TRAIN_PROJECTOR must be 0 or 1.' >&2; exit 2 ;;
esac
# Every stage starts with a fresh optimizer unless explicitly resumed below.
export MODEL_PATH="${MODEL_PATH:-$default_model}"
export AUDIO_ROOT="${AUDIO_ROOT:-$SWE_SOURCE/data/audio_opd_pilot_v1}"
export TRAIN_FILES="${TRAIN_FILES:-[\"$SWE_SOURCE/data/swe_benchmarks/verl/swe_rebench_v2_reference_verified_snapshot_2000.parquet\",\"$AUDIO_ROOT/train.opd.parquet\"]}"
export VAL_FILES="${VAL_FILES:-[\"$SWE_SOURCE/data/swe_benchmarks/verl/swe_bench_verified_reference_verified_repeat2_500_images_v1.parquet\"]}"
export SWE_AGENT_EXECUTION_URLS="${SWE_AGENT_EXECUTION_URLS:-http://10.248.102.222:18080,http://10.248.103.15:18080,http://10.248.102.67:18080,http://10.248.103.68:18080,http://10.248.100.59:18080,http://10.248.100.43:18080}"

export TEACHER_BASE_URL="${TEACHER_BASE_URL:-http://10.248.102.141:18090}"
export TEACHER_MODEL_NAME="${TEACHER_MODEL_NAME:-swe-repeat-judge}"
export TEACHER_MAX_MODEL_LEN="${TEACHER_MAX_MODEL_LEN:-262144}"
# This limit applies globally across rollout workers through the shared scorer.
export TEACHER_MAX_CONCURRENT_REQUESTS="${TEACHER_MAX_CONCURRENT_REQUESTS:-4}"
export TEACHER_REQUEST_TIMEOUT_S="${TEACHER_REQUEST_TIMEOUT_S:-1800}"
export TEACHER_MAX_RETRIES="${TEACHER_MAX_RETRIES:-2}"
export TEACHER_RETRY_BACKOFF_S="${TEACHER_RETRY_BACKOFF_S:-2}"

export AUDIO_TEACHER_BASE_URL="${AUDIO_TEACHER_BASE_URL:-http://10.248.101.72:18090}"
export AUDIO_TEACHER_MODEL_NAME="${AUDIO_TEACHER_MODEL_NAME:-swe-repeat-judge}"
export AUDIO_TEACHER_TOKENIZER_PATH="${AUDIO_TEACHER_TOKENIZER_PATH:-$SWE_SOURCE/data/Qwen3-Omni-30B-A3B-Instruct}"
export AUDIO_TEACHER_MAX_MODEL_LEN="${AUDIO_TEACHER_MAX_MODEL_LEN:-65536}"
export AUDIO_TEACHER_MAX_CONCURRENT_REQUESTS="${AUDIO_TEACHER_MAX_CONCURRENT_REQUESTS:-2}"
export AUDIO_MAX_PROMPT_LENGTH="${AUDIO_MAX_PROMPT_LENGTH:-8192}"
export AUDIO_MAX_RESPONSE_LENGTH="${AUDIO_MAX_RESPONSE_LENGTH:-2048}"
export AUDIO_ALLOWED_MEDIA_ROOT="${AUDIO_ALLOWED_MEDIA_ROOT:-$SWE_SOURCE/data}"
if [[ "$actor_train_audio_projector" == True ]]; then
  if [[ "${AUDIO_FREEZE_ENCODER:-1}" != 1 || "${AUDIO_FREEZE_PROJECTOR:-0}" != 0 ]]; then
    echo 'Stage 2 requires AUDIO_FREEZE_ENCODER=1 and AUDIO_FREEZE_PROJECTOR=0.' >&2
    exit 2
  fi
else
  if [[ "${AUDIO_FREEZE_ENCODER:-1}" != 1 || "${AUDIO_FREEZE_PROJECTOR:-1}" != 1 ]]; then
    echo 'Stage 1 requires AUDIO_FREEZE_ENCODER=1 and AUDIO_FREEZE_PROJECTOR=1.' >&2
    exit 2
  fi
fi
if [[ "${SWE_AGENT_OPD_USE_TASK_REWARDS:-0}" != 0 ]]; then
  echo "$audio_stage supports pure OPD only; SWE_AGENT_OPD_USE_TASK_REWARDS must be 0." >&2
  exit 2
fi

export EXPERIMENT_NAME="${EXPERIMENT_NAME:-$default_experiment}"
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

export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-64}"
ROLLOUT_N="${ROLLOUT_N:-2}"
ACTOR_LR="${ACTOR_LR:-5e-7}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-100}"
if [[ -z "${OPD_MIX_SCHEDULE:-}" ]]; then
  OPD_MIX_SCHEDULE='[{"steps":20,"quotas":{"swe":16,"audio_librispeech":18,"audio_aishell1":18,"audio_clotho_aqa":12}},{"steps":60,"quotas":{"swe":32,"audio_librispeech":12,"audio_aishell1":12,"audio_clotho_aqa":8}},{"steps":20,"quotas":{"swe":48,"audio_librispeech":6,"audio_aishell1":6,"audio_clotho_aqa":4}}]'
fi
export OPD_MIX_SCHEDULE
config_helper="$TARGET_VERL/scripts/audio_data/multi_teacher_config.py"
swe_teacher_config="$(python3 "$config_helper" swe)"
audio_teacher_config="$(python3 "$config_helper" audio)"
mix_config="$(python3 "$config_helper" mix)"
# The curriculum uses optimizer steps; the epoch ceiling remains a fallback.
TRAINER_STOP_OVERRIDES=()
if [[ -n "${TOTAL_TRAINING_STEPS:-}" ]]; then
  TRAINER_STOP_OVERRIDES+=("trainer.total_training_steps=$TOTAL_TRAINING_STEPS")
fi

# Retain infrastructure filtering; valid incorrect solutions remain training data.
export SWE_AGENT_TASK_FILTER_TRAINING_ENABLED=1
export SWE_AGENT_TASK_FILTER_MIN_GROUP_ATTEMPTS="${SWE_AGENT_TASK_FILTER_MIN_GROUP_ATTEMPTS:-$ROLLOUT_N}"
export SWE_AGENT_TASK_FILTER_INCLUDE_QUEUE_FAILURES=1
export SWE_AGENT_TASK_FILTER_INFRA_RATIO_THRESHOLD=0.25
export SWE_AGENT_ROLLOUT_CLAIM_EXECUTOR_WORKERS="${SWE_AGENT_ROLLOUT_CLAIM_EXECUTOR_WORKERS:-3}"
export SWE_AGENT_HEALTH_CHECK_TIMEOUT_SECONDS="${SWE_AGENT_HEALTH_CHECK_TIMEOUT_SECONDS:-5}"
# Wait for both routes so faster audio requests do not crowd out SWE at cutoff.
export SWE_AGENT_ROLLOUT_TRAINING_COMPLETION_RATIO_THRESHOLD="${SWE_AGENT_ROLLOUT_TRAINING_COMPLETION_RATIO_THRESHOLD:-1.0}"
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
# Pure OPD skips hidden evaluation; task rewards are unsupported in this pilot.
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
export SWE_AGENT_OPD_CUTOFF_WAIT_FOR_SCORING="${SWE_AGENT_OPD_CUTOFF_WAIT_FOR_SCORING:-1}"
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
# Keep the historical validation defaults while allowing the mounted audio suite.
AUDIO_BENCHMARK_BENCHMARKS="${AUDIO_BENCHMARK_BENCHMARKS:-[aishell2,chartqa,clothoaqa,mmbench,mmstar,ocrbench,realworldqa,rul_muchomusic,voicebench_bbh]}"
AUDIO_BENCHMARK_MAX_SAMPLES="${AUDIO_BENCHMARK_MAX_SAMPLES:-100}"
AUDIO_BENCHMARK_CONCURRENCY="${AUDIO_BENCHMARK_CONCURRENCY:-32}"
AUDIO_BENCHMARK_LOAD_BALANCE="${AUDIO_BENCHMARK_LOAD_BALANCE:-True}"
AUDIO_BENCHMARK_FAILURE_POLICY="${AUDIO_BENCHMARK_FAILURE_POLICY:-warn}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"
TEST_FREQ="${TEST_FREQ:-10}"
SAVE_FREQ="${SAVE_FREQ:-10}"

if [[ "${CONFIG_ONLY:-0}" != "1" ]]; then
  mkdir -p "$ROLLOUT_DATA_DIR" "$VALIDATION_DATA_DIR" "$AUDIO_BENCHMARK_OUTPUT_DIR"
fi

cd "$TARGET_VERL"
if [[ "$actor_train_audio_projector" == True ]]; then
  echo 'Stage 2: frozen audio encoder + trainable proj1/proj2; SWE token OPD + audio sequence surrogate'
else
  echo 'Stage 1: frozen audio encoder + projector; SWE token OPD + audio sequence surrogate'
fi
echo "Teachers: SWE=$TEACHER_BASE_URL (parallel=$TEACHER_MAX_CONCURRENT_REQUESTS) audio=$AUDIO_TEACHER_BASE_URL (parallel=$AUDIO_TEACHER_MAX_CONCURRENT_REQUESTS)"
echo "Audio response cap=$AUDIO_MAX_RESPONSE_LENGTH; prompt quota schedule=$OPD_MIX_SCHEDULE"
exec bash scripts/run_swe_agent_qwen3_5_35b_a3b.sh \
  trainer.use_v1=True \
  trainer.v1.trainer_mode=sync \
  trainer.resume_mode=disable \
  trainer.val_before_train="$VAL_BEFORE_TRAIN" \
  trainer.val_only=False \
  trainer.rollout_data_dir="$ROLLOUT_DATA_DIR" \
  trainer.validation_data_dir="$VALIDATION_DATA_DIR" \
  trainer.save_freq="$SAVE_FREQ" \
  trainer.test_freq="$TEST_FREQ" \
  trainer.total_epochs=15 \
  data.train_batch_size="$TRAIN_BATCH_SIZE" \
  data.gen_batch_size=1 \
  data.dataloader_num_workers=0 \
  data.seed=42 \
  data.custom_cls.path=pkg://verl.utils.dataset.audio_opd_dataset \
  data.custom_cls.name=AudioOPDMixedDataset \
  +data.audio_sampling_rate=16000 \
  "+data.opd_source_mix=$mix_config" \
  actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.optim.lr="$ACTOR_LR" \
  actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
  actor_rollout_ref.actor.megatron.moe_loss_respects_train_sample_mask=True \
  actor_rollout_ref.actor.megatron.frozen_audio="$actor_frozen_audio" \
  actor_rollout_ref.actor.megatron.train_audio_projector="$actor_train_audio_projector" \
  actor_rollout_ref.ref.megatron.frozen_audio=True \
  actor_rollout_ref.ref.megatron.train_audio_projector=False \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.entropy_coeff=0.0 \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  algorithm.kl_ctrl.kl_coef=0.0 \
  algorithm.filter_groups.enable=False \
  algorithm.rollout_correction.rollout_is=null \
  algorithm.rollout_correction.rollout_rs=null \
  actor_rollout_ref.model.use_fused_kernels=True \
  actor_rollout_ref.model.enable_audio_processor=True \
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
  +actor_rollout_ref.rollout.engine_kwargs.vllm.allowed_local_media_path="$AUDIO_ALLOWED_MEDIA_ROOT" \
  "${audio_rollout_overrides[@]}" \
  +data.apply_chat_template_kwargs.enable_thinking=True \
  distillation.enabled=True \
  distillation.n_gpus_per_node=0 \
  distillation.nnodes=0 \
  distillation.teacher_key=teacher_route \
  "+distillation.teacher_models.swe_teacher=$swe_teacher_config" \
  "+distillation.teacher_models.audio_teacher=$audio_teacher_config" \
  distillation.distillation_loss.loss_mode=k1 \
  distillation.distillation_loss.use_policy_gradient=True \
  distillation.distillation_loss.audio_sequence_surrogate=True \
  distillation.distillation_loss.use_task_rewards="$opd_use_task_rewards" \
  distillation.distillation_loss.distillation_loss_coef="$SWE_AGENT_OPD_DISTILLATION_LOSS_COEF" \
  distillation.distillation_loss.loss_max_clamp=null \
  distillation.distillation_loss.log_prob_min_clamp=null \
  reward.custom_reward_function.path="$TARGET_VERL/verl/utils/reward_score/swe_audio_opd.py" \
  +trainer.audio_benchmark.enabled="$AUDIO_BENCHMARK_ENABLED" \
  +trainer.audio_benchmark.only=False \
  +trainer.audio_benchmark.gage_root="$WORKSPACE_ROOT/gage-leizhengxing" \
  +trainer.audio_benchmark.data_root="$AUDIO_BENCHMARK_DATA_ROOT" \
  +trainer.audio_benchmark.benchmarks="$AUDIO_BENCHMARK_BENCHMARKS" \
  +trainer.audio_benchmark.max_samples="$AUDIO_BENCHMARK_MAX_SAMPLES" \
  +trainer.audio_benchmark.concurrency="$AUDIO_BENCHMARK_CONCURRENCY" \
  +trainer.audio_benchmark.load_balance="$AUDIO_BENCHMARK_LOAD_BALANCE" \
  +trainer.audio_benchmark.check_mode=light \
  +trainer.audio_benchmark.failure_policy="$AUDIO_BENCHMARK_FAILURE_POLICY" \
  +trainer.audio_benchmark.output_dir="$AUDIO_BENCHMARK_OUTPUT_DIR" \
  "${TRAINER_STOP_OVERRIDES[@]}" \
  "$@"

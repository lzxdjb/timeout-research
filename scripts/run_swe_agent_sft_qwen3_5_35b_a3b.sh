#!/usr/bin/env bash
set -euo pipefail

TARGET_VERL="${TARGET_VERL:-/cpfs01/thscc/sharestorage/iwc/HithinkOmni/user_workspace/leizhengxing/leizhengxing/swe/verl}"

SFT_TRAIN_FILE="${SFT_TRAIN_FILE:-$TARGET_VERL/analysis/swe_sft_data/core_train.parquet}"
SFT_TEACHER_VAL_FILE="${SFT_TEACHER_VAL_FILE:-}"
SFT_BATCH_SIZE="${SFT_BATCH_SIZE:-8}"
SFT_MAX_LENGTH="${SFT_MAX_LENGTH:-65536}"
SFT_LR="${SFT_LR:-5e-7}"
SFT_EPOCHS="${SFT_EPOCHS:-1}"
SFT_TEST_FREQ="${SFT_TEST_FREQ:-10}"
SFT_SAVE_FREQ="${SFT_SAVE_FREQ:-10}"
SFT_TEACHER_TEST_FREQ="${SFT_TEACHER_TEST_FREQ:--1}"

: "${VAL_FILES:?Set VAL_FILES to a Hydra list of RL validation parquet paths}"
: "${SWE_AGENT_EXECUTION_URLS:?Set SWE_AGENT_EXECUTION_URLS to the execution-service URL list}"
[[ -f "$SFT_TRAIN_FILE" ]] || { echo "SFT train file not found: $SFT_TRAIN_FILE" >&2; exit 2; }
if [[ -n "$SFT_TEACHER_VAL_FILE" && ! -f "$SFT_TEACHER_VAL_FILE" ]]; then
  echo "Teacher-forced validation file not found: $SFT_TEACHER_VAL_FILE" >&2
  exit 2
fi

for integer_setting in \
  SFT_BATCH_SIZE \
  SFT_MAX_LENGTH \
  SFT_EPOCHS; do
  if ! [[ "${!integer_setting}" =~ ^[0-9]+$ ]] || (( 10#${!integer_setting} <= 0 )); then
    echo "$integer_setting must be a positive integer: ${!integer_setting}" >&2
    exit 2
  fi
done
unset integer_setting

export TRAIN_FILES="[\"$SFT_TRAIN_FILE\"]"
export VERL_TRAINER_MODULE=verl.trainer.main_sft_rl
export PROJECT_NAME="${PROJECT_NAME:-35A3B-SWE-SFT}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-swe_core_sft_rl_validation}"
export PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-$SFT_MAX_LENGTH}"
export ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU="${ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-$SFT_MAX_LENGTH}"

if [[ -n "$SFT_TEACHER_VAL_FILE" ]]; then
  teacher_val_files="[\"$SFT_TEACHER_VAL_FILE\"]"
else
  teacher_val_files=null
fi

echo "SFT train data:       $SFT_TRAIN_FILE"
echo "RL validation data:   $VAL_FILES"
echo "Teacher-forced data:  ${SFT_TEACHER_VAL_FILE:-disabled}"
echo "SFT batch/lr/epochs:  $SFT_BATCH_SIZE / $SFT_LR / $SFT_EPOCHS"
echo "SFT max length:       $SFT_MAX_LENGTH"
echo "Validation frequency: RL=$SFT_TEST_FREQ teacher-forced=$SFT_TEACHER_TEST_FREQ"

exec bash "$TARGET_VERL/scripts/run_swe_agent_qwen3_5_35b_a3b.sh" \
  trainer.use_v1=True \
  trainer.v1.trainer_mode=sync \
  trainer.total_epochs="$SFT_EPOCHS" \
  trainer.test_freq="$SFT_TEST_FREQ" \
  trainer.save_freq="$SFT_SAVE_FREQ" \
  data.train_batch_size="$SFT_BATCH_SIZE" \
  +data.messages_key=messages \
  +data.tools_key=tools \
  +data.enable_thinking_key=enable_thinking \
  +data.enable_thinking_default=null \
  +data.pad_mode=no_padding \
  +data.max_length="$SFT_MAX_LENGTH" \
  +data.ignore_input_ids_mismatch=False \
  actor_rollout_ref.actor.optim.lr="$SFT_LR" \
  actor_rollout_ref.actor.ppo_mini_batch_size="$SFT_BATCH_SIZE" \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.kl_loss_coef=0.0 \
  actor_rollout_ref.rollout.n=1 \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  algorithm.use_kl_in_reward=False \
  algorithm.kl_ctrl.kl_coef=0.0 \
  algorithm.filter_groups.enable=False \
  critic.enable=False \
  reward.reward_model.enable=False \
  +sft.teacher_forced_val_files="$teacher_val_files" \
  +sft.teacher_forced_test_freq="$SFT_TEACHER_TEST_FREQ" \
  +sft.teacher_forced_val_before_train=False \
  +sft.teacher_forced_val_batch_size="$SFT_BATCH_SIZE" \
  "$@"

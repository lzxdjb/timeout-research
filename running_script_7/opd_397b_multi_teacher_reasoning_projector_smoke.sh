#!/usr/bin/env bash
# One-step projector gate; SMOKE_VALIDATE=1 adds SWE, audio and reasoning scores.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen35_multi_teacher_reasoning_projector_smoke_$(date -u +%Y%m%dT%H%M%SZ)}"
export TOTAL_TRAINING_STEPS=1
export ROLLOUT_N=1
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-$TRAIN_BATCH_SIZE}"
export GENERAL_REASONING_MAX_SAMPLES="${GENERAL_REASONING_MAX_SAMPLES:-5}"
case "${SMOKE_VALIDATE:-0}" in
  1|[Tt][Rr][Uu][Ee]) smoke_test_freq=1; smoke_audio_default=True ;;
  0|[Ff][Aa][Ll][Ss][Ee]) smoke_test_freq=0; smoke_audio_default=False ;;
  *) echo 'SMOKE_VALIDATE must be 0, 1, false, or true.' >&2; exit 2 ;;
esac
export AUDIO_BENCHMARK_ENABLED="${AUDIO_BENCHMARK_ENABLED:-$smoke_audio_default}"
export PYTHONUNBUFFERED=1
exec bash "$script_dir/opd_397b_multi_teacher_reasoning_projector.sh" \
  "$@" \
  trainer.resume_mode=disable \
  trainer.total_training_steps=1 \
  trainer.val_before_train=False \
  trainer.val_only=False \
  trainer.audio_benchmark.only=False \
  actor_rollout_ref.actor.megatron.frozen_audio=False \
  actor_rollout_ref.actor.megatron.train_audio_projector=True \
  actor_rollout_ref.ref.megatron.frozen_audio=True \
  actor_rollout_ref.ref.megatron.train_audio_projector=False \
  actor_rollout_ref.rollout.n=1 \
  actor_rollout_ref.rollout.engine_kwargs.vllm.hf_overrides.verl_audio_projector_trainable=True \
  trainer.test_freq="$smoke_test_freq" \
  trainer.save_freq=1

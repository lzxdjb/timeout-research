#!/usr/bin/env bash
# Evaluate all selected audio/reasoning examples without SWE or teacher scoring.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen35_audio_reasoning_eval_$(date -u +%Y%m%dT%H%M%SZ)}"
export TRAIN_FILES='[]' VAL_FILES='[]'
export TRAIN_BATCH_SIZE=64 PPO_MINI_BATCH_SIZE=64 ROLLOUT_N=1 TOTAL_TRAINING_STEPS=1
export SAVE_FREQ=0 TEST_FREQ=0 VAL_BEFORE_TRAIN=True
# Align the inherited startup settings and banner with teacher-free evaluation.
export SWE_AGENT_TRAINING_SKIP_HIDDEN_REWARD=0 SWE_AGENT_OPD_CANCEL_ON_CUTOFF=0
export SWE_AGENT_OPD_CUTOFF_WAIT_FOR_SCORING=0 SWE_AGENT_OPD_PHASE_TIMING=0
export MAX_PROMPT_LENGTH=8192 MAX_RESPONSE_LENGTH=252000
export max_prompt_length="$MAX_PROMPT_LENGTH" max_response_length="$MAX_RESPONSE_LENGTH"
export AUDIO_MAX_RESPONSE_LENGTH=4096
export AUDIO_BENCHMARK_ENABLED=True AUDIO_BENCHMARK_DATA_ROOT="${AUDIO_BENCHMARK_DATA_ROOT:-/mnt}"
export AUDIO_BENCHMARK_BENCHMARKS='[aishell1,librispeech,wenetspeech,clothoaqa,rul_muchomusic,voicebench_bbh]'
export AUDIO_BENCHMARK_MAX_SAMPLES=null GENERAL_REASONING_MAX_SAMPLES=-1
export AUDIO_BENCHMARK_CONCURRENCY="${AUDIO_BENCHMARK_CONCURRENCY:-128}"
export AUDIO_BENCHMARK_LOAD_BALANCE=True AUDIO_BENCHMARK_FAILURE_POLICY=error
unset GAGE_EVAL_MAX_SAMPLES GENERAL_REASONING_VAL_FILES OPD_MIX_SCHEDULE

exec bash "$script_dir/opd_397b_multi_teacher_reasoning_projector.sh" \
  trainer.val_only=True \
  trainer.audio_benchmark.only=True \
  trainer.resume_mode=disable \
  distillation.enabled=False \
  distillation.skip_hidden_reward=False \
  distillation.cancel_pending_on_cutoff=False \
  distillation.wait_for_scoring_on_cutoff=False \
  distillation.phase_timing=False \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
  actor_rollout_ref.rollout.val_kwargs.top_k=20 \
  actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
  trainer.general_reasoning.temperature=0.6 \
  trainer.general_reasoning.top_k=20 \
  trainer.general_reasoning.top_p=0.95 \
  +trainer.audio_benchmark.temperature=0.0 \
  +trainer.audio_benchmark.top_k=1 \
  +trainer.audio_benchmark.top_p=1.0 \
  +trainer.audio_benchmark.max_tokens=4096 \
  "$@"

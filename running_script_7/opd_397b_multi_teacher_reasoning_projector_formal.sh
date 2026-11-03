#!/usr/bin/env bash
# Formal 100-step profile: 64 prompts x 8 rollouts, all selected benchmarks.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen35_swe_audio_reasoning_formal_$(date -u +%Y%m%dT%H%M%SZ)}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-100}"
export TRAIN_BATCH_SIZE=64 PPO_MINI_BATCH_SIZE=64 ROLLOUT_N=8
export ACTOR_LR=1e-6 SAVE_FREQ=1 TEST_FREQ=2 VAL_BEFORE_TRAIN=False
# Decimal 252k leaves room for the prompt in the 262,144-token teacher context.
export MAX_PROMPT_LENGTH=8192 MAX_RESPONSE_LENGTH=252000
export max_prompt_length="$MAX_PROMPT_LENGTH" max_response_length="$MAX_RESPONSE_LENGTH"
export SWE_AGENT_ROLLOUT_TRAINING_RESPONSE_LENGTH=252000
export SWE_AGENT_ROLLOUT_VALIDATION_RESPONSE_LENGTH=252000
export SWE_AGENT_ROLLOUT_TRAINING_COMPLETION_RATIO_THRESHOLD=0.96
export SWE_AGENT_ROLLOUT_VALIDATION_COMPLETION_RATIO_THRESHOLD=0.98
export AUDIO_MAX_RESPONSE_LENGTH=4096
export AUDIO_BENCHMARK_ENABLED=True AUDIO_BENCHMARK_DATA_ROOT="${AUDIO_BENCHMARK_DATA_ROOT:-/mnt}"
export AUDIO_BENCHMARK_BENCHMARKS='[aishell1,librispeech,wenetspeech,clothoaqa,rul_muchomusic,voicebench_bbh]'
export AUDIO_BENCHMARK_MAX_SAMPLES=null GENERAL_REASONING_MAX_SAMPLES=-1
export AUDIO_BENCHMARK_CONCURRENCY="${AUDIO_BENCHMARK_CONCURRENCY:-128}"
export AUDIO_BENCHMARK_LOAD_BALANCE=True AUDIO_BENCHMARK_FAILURE_POLICY=error
unset TRAIN_FILES VAL_FILES GAGE_EVAL_MAX_SAMPLES GENERAL_REASONING_VAL_FILES OPD_MIX_SCHEDULE

exec bash "$script_dir/opd_397b_multi_teacher_reasoning_projector.sh" \
  +trainer.audio_benchmark.max_tokens=4096 \
  data.filter_overlong_prompts_workers=16 \
  "$@"

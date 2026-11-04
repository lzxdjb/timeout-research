#!/usr/bin/env bash
# SWE/audio-only formal profile: 64 prompts x 8 rollouts, no reasoning data/evaluation.
# Starts from explicit HF weights with a fresh optimizer unless resumed via Hydra.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

: "${MODEL_PATH:?Set MODEL_PATH to the Hugging Face checkpoint to train}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen35_swe_audio_formal_$(date -u +%Y%m%dT%H%M%SZ)}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-100}"
export TEACHER_BASE_URL="${TEACHER_BASE_URL:-http://10.248.102.92:18090}"
export AUDIO_TEACHER_BASE_URL="${AUDIO_TEACHER_BASE_URL:-http://10.248.102.8:18090}"
export SWE_AGENT_EXECUTION_URLS="${SWE_AGENT_EXECUTION_URLS:-http://10.248.103.90:18080,http://10.248.103.88:18080}"
export TRAIN_BATCH_SIZE=64 PPO_MINI_BATCH_SIZE=64 ROLLOUT_N=8
export ACTOR_LR=1e-6 SAVE_FREQ=1 TEST_FREQ=2 VAL_BEFORE_TRAIN=False
export MAX_PROMPT_LENGTH=8192 MAX_RESPONSE_LENGTH=252000
export max_prompt_length="$MAX_PROMPT_LENGTH" max_response_length="$MAX_RESPONSE_LENGTH"
export SWE_AGENT_ROLLOUT_TRAINING_RESPONSE_LENGTH=252000
export SWE_AGENT_ROLLOUT_VALIDATION_RESPONSE_LENGTH=252000
export SWE_AGENT_ROLLOUT_TRAINING_COMPLETION_RATIO_THRESHOLD=0.96
export SWE_AGENT_ROLLOUT_VALIDATION_COMPLETION_RATIO_THRESHOLD=0.98
export AUDIO_MAX_RESPONSE_LENGTH="${AUDIO_MAX_RESPONSE_LENGTH:-8192}"
export AUDIO_BENCHMARK_ENABLED=True AUDIO_BENCHMARK_DATA_ROOT="${AUDIO_BENCHMARK_DATA_ROOT:-/mnt}"
export AUDIO_BENCHMARK_BENCHMARKS='[aishell1,librispeech,wenetspeech,clothoaqa,rul_muchomusic,voicebench_bbh]'
export AUDIO_BENCHMARK_MAX_SAMPLES=null
export AUDIO_BENCHMARK_CONCURRENCY="${AUDIO_BENCHMARK_CONCURRENCY:-128}"
export AUDIO_BENCHMARK_LOAD_BALANCE=True AUDIO_BENCHMARK_FAILURE_POLICY=error
# Rebuild the two training files, SWE validation roster and original 64-prompt
# SWE/audio curriculum; inherited reasoning files or quotas must not leak in.
unset TRAIN_FILES VAL_FILES GAGE_EVAL_MAX_SAMPLES OPD_MIX_SCHEDULE

exec bash "$script_dir/opd_397b_multi_teacher_projector.sh" \
  +trainer.general_reasoning.enabled=False \
  +trainer.audio_opd.exclude_incomplete_sequences=True \
  +trainer.training_metrics_snapshot.enabled=True \
  +trainer.audio_benchmark.max_tokens=4096 \
  data.filter_overlong_prompts_workers=16 \
  "$@"

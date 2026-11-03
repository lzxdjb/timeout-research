#!/usr/bin/env bash
# Research pilot: local media checks + reference-blind Qwen ASR, without RL auditing.
set -euo pipefail

audio_repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd -- "$audio_repo_dir"

usage() {
  cat <<'EOF'
Usage: bash scripts/audio_data/prepare_opd.sh STAGE
Stages: plan download normalize offline asr asr-filter convert convert-offline all

Required: AUDIO_ROOT=/absolute/path/to/data
Optional environment:
  AUDIO_PYTHON=python
  AUDIO_DATASETS="librispeech aishell1 clotho_aqa"
  AUDIO_QWEN_BASE_URL=http://10.248.101.72:18090
  AUDIO_QWEN_MODEL=auto
  AUDIO_QWEN_MODEL_REVISION=immutable-checkpoint-id
  AUDIO_MAX_DOWNLOAD_BYTES=34359738368     # 32 GiB; all five need >=67.35 GB
  AUDIO_MAX_EXTRACTED_BYTES=137438953472   # 128 GiB
  AUDIO_DOWNLOAD_CONNECTIONS=8            # concurrent ranges within one archive
  AUDIO_DOWNLOAD_MIRROR=original          # original or official OpenSLR elda mirror
  AUDIO_ASR_SOURCE_SAMPLES=5000           # candidates per official split/source
  AUDIO_CLOTHO_SAMPLES=0                  # all QA; keep all strict-consensus cases
  AUDIO_FSD_SAMPLES=5000
  AUDIO_NSYNTH_SAMPLES=1000
  AUDIO_SAMPLE_SEED=42                   # uniform reservoir, not a sorted prefix
  AUDIO_MAX_DURATION=30                  # seconds, bounds this first pilot
  AUDIO_ASR_MAX_SAMPLES=10000             # attempted training ASR rows; 0=no requests
  AUDIO_ASR_MAX_RUNTIME_SECONDS=86400     # soft budget, in-flight requests finish
  AUDIO_QWEN_CONCURRENCY=2
  AUDIO_ASR_PROGRESS_INTERVAL=2          # seconds between ASR status updates
  AUDIO_QWEN_TIMEOUT=120
  AUDIO_QWEN_RETRIES=2
  AUDIO_MAX_WER=0.30
  AUDIO_MAX_CER=0.25
  AUDIO_PROGRESS=on                     # on=TTY bars or periodic stderr logs
  AUDIO_ASR_FILTER_CHECKPOINT_DIR=$AUDIO_ROOT/checkpoints/asr-filter
  AUDIO_CHECKPOINT_EVERY=1000            # commit every N completed records
  AUDIO_RESET_ASR_FILTER_CHECKPOINT=0    # 1 explicitly starts a fresh filter run
  AUDIO_HOLDOUT_MANIFEST=/absolute/path/to/benchmark.jsonl

plan makes no network calls. all explicitly executes download through convert.
convert uses ASR-checked rows; convert-offline creates a separate audio-only baseline.
Known noncommercial licenses are enabled throughout this research recipe.
EOF
}

audio_stage="${1:-plan}"
if [[ "$audio_stage" == --help || "$audio_stage" == -h ]]; then
  usage
  exit 0
fi
if [[ $# -gt 1 ]]; then
  usage >&2
  exit 2
fi
case "$audio_stage" in
  plan|download|normalize|offline|asr|asr-filter|convert|convert-offline|all) ;;
  *) usage >&2; exit 2 ;;
esac
: "${AUDIO_ROOT:?Set AUDIO_ROOT to an absolute data directory; use --help for the recipe.}"
if [[ "$AUDIO_ROOT" != /* ]]; then
  printf '%s\n' 'AUDIO_ROOT must be an absolute path.' >&2
  exit 2
fi
if [[ "$audio_stage" != plan ]]; then
  if ! command -v flock >/dev/null 2>&1; then
    printf '%s\n' 'The preparation wrapper requires flock (util-linux).' >&2
    exit 2
  fi
  mkdir -p -- "$AUDIO_ROOT"
  exec {audio_lock_fd}>"$AUDIO_ROOT/.prepare_opd.lock"
  if ! flock -n "$audio_lock_fd"; then
    printf '%s\n' 'Another preparation stage is running for this AUDIO_ROOT.' >&2
    exit 2
  fi
fi
AUDIO_PYTHON="${AUDIO_PYTHON:-python}"
export AUDIO_PROGRESS="${AUDIO_PROGRESS:-on}"
case "$AUDIO_PROGRESS" in
  auto|on|off) ;;
  *) printf '%s\n' 'AUDIO_PROGRESS must be auto, on, or off.' >&2; exit 2 ;;
esac
audio_checkpoint_reset_args=()
case "${AUDIO_RESET_ASR_FILTER_CHECKPOINT:-0}" in
  0) ;;
  1) audio_checkpoint_reset_args+=(--reset-checkpoint) ;;
  *) printf '%s\n' 'AUDIO_RESET_ASR_FILTER_CHECKPOINT must be 0 or 1.' >&2; exit 2 ;;
esac
read -r -a audio_datasets <<< "${AUDIO_DATASETS:-librispeech aishell1 clotho_aqa}"
if [[ ${#audio_datasets[@]} -eq 0 ]]; then
  printf '%s\n' 'Select at least one dataset in AUDIO_DATASETS.' >&2
  exit 2
fi
declare -A audio_seen_datasets=()
audio_download_args=()
for audio_dataset in "${audio_datasets[@]}"; do
  case "$audio_dataset" in
    librispeech|aishell1|clotho_aqa|fsd50k|nsynth) ;;
    *) printf 'Unsupported dataset: %s\n' "$audio_dataset" >&2; exit 2 ;;
  esac
  if [[ -n "${audio_seen_datasets[$audio_dataset]:-}" ]]; then
    printf 'Duplicate dataset: %s\n' "$audio_dataset" >&2
    exit 2
  fi
  audio_seen_datasets[$audio_dataset]=1
  audio_download_args+=(--dataset "$audio_dataset")
done
audio_download_args+=(--connections "${AUDIO_DOWNLOAD_CONNECTIONS:-8}" --mirror "${AUDIO_DOWNLOAD_MIRROR:-original}")
audio_holdout_args=()
if [[ -n "${AUDIO_HOLDOUT_MANIFEST:-}" ]]; then
  audio_holdout_args+=(--holdout-manifest "$AUDIO_HOLDOUT_MANIFEST")
fi
audio_revision_args=()
if [[ -n "${AUDIO_QWEN_MODEL_REVISION:-}" ]]; then
  audio_revision_args+=(--model-revision "$AUDIO_QWEN_MODEL_REVISION")
fi

run_stage() {
  local audio_current_stage="$1"
  printf 'Audio preparation stage: %s\n' "$audio_current_stage" >&2
  case "$audio_current_stage" in
    plan)
      "$AUDIO_PYTHON" -m scripts.audio_data.download "${audio_download_args[@]}" \
        --root "$AUDIO_ROOT/raw" --allow-noncommercial --progress "$AUDIO_PROGRESS"
      ;;
    download)
      "$AUDIO_PYTHON" -m scripts.audio_data.download "${audio_download_args[@]}" \
        --root "$AUDIO_ROOT/raw" --allow-noncommercial --execute --extract \
        --progress "$AUDIO_PROGRESS" \
        --max-download-bytes "${AUDIO_MAX_DOWNLOAD_BYTES:-34359738368}" \
        --max-extracted-bytes "${AUDIO_MAX_EXTRACTED_BYTES:-137438953472}"
      ;;
    normalize)
      local audio_sample_limit
      for audio_dataset in "${audio_datasets[@]}"; do
        case "$audio_dataset" in
          librispeech|aishell1) audio_sample_limit="${AUDIO_ASR_SOURCE_SAMPLES:-5000}" ;;
          clotho_aqa) audio_sample_limit="${AUDIO_CLOTHO_SAMPLES:-0}" ;;
          fsd50k) audio_sample_limit="${AUDIO_FSD_SAMPLES:-5000}" ;;
          nsynth) audio_sample_limit="${AUDIO_NSYNTH_SAMPLES:-1000}" ;;
        esac
        "$AUDIO_PYTHON" -m scripts.audio_data.normalize \
          --dataset "$audio_dataset" --source-root "$AUDIO_ROOT/raw/$audio_dataset/extracted" \
          --output "$AUDIO_ROOT/manifests/$audio_dataset.jsonl" --split all \
          --max-samples "$audio_sample_limit" --sample-seed "${AUDIO_SAMPLE_SEED:-42}" \
          --allow-noncommercial --progress "$AUDIO_PROGRESS"
      done
      # An atomic global manifest gives cross-source leakage checks the whole mixture.
      "$AUDIO_PYTHON" - "$AUDIO_ROOT" "${audio_datasets[@]}" <<'PY'
import os
import sys
from pathlib import Path
from scripts.audio_data.common import read_jsonl, write_jsonl
from scripts.audio_data.progress import Progress
root = Path(sys.argv[1])
sources = [root / 'manifests' / f'{name}.jsonl' for name in sys.argv[2:]]
with Progress('Combine dataset manifests', mode=os.environ['AUDIO_PROGRESS']) as bar:
    def records():
        for source in sources:
            for row in read_jsonl(source):
                yield row
                bar.update()
    write_jsonl(root / 'pilot.normalized.jsonl', records())
PY
      ;;
    offline)
      "$AUDIO_PYTHON" -m scripts.audio_data.filter_offline \
        --input "$AUDIO_ROOT/pilot.normalized.jsonl" --output "$AUDIO_ROOT/pilot.offline.jsonl" \
        --allow-noncommercial --max-duration "${AUDIO_MAX_DURATION:-30}" \
        --progress "$AUDIO_PROGRESS" "${audio_holdout_args[@]}"
      ;;
    asr)
      "$AUDIO_PYTHON" -m scripts.audio_data.transcribe_qwen \
        --input "$AUDIO_ROOT/pilot.offline.jsonl" --output "$AUDIO_ROOT/qwen_asr.jsonl" \
        --base-url "${AUDIO_QWEN_BASE_URL:-http://10.248.101.72:18090}" \
        --model "${AUDIO_QWEN_MODEL:-auto}" --cache-dir "$AUDIO_ROOT/qwen_asr_cache" \
        --split train --max-samples "${AUDIO_ASR_MAX_SAMPLES:-10000}" \
        --max-duration "${AUDIO_MAX_DURATION:-30}" \
        --max-runtime-seconds "${AUDIO_ASR_MAX_RUNTIME_SECONDS:-86400}" \
        --concurrency "${AUDIO_QWEN_CONCURRENCY:-2}" --timeout "${AUDIO_QWEN_TIMEOUT:-120}" \
        --progress "$AUDIO_PROGRESS" --progress-interval "${AUDIO_ASR_PROGRESS_INTERVAL:-2}" \
        --retries "${AUDIO_QWEN_RETRIES:-2}" "${audio_revision_args[@]}"
      "$AUDIO_PYTHON" - "$AUDIO_ROOT" <<'PY'
import sys
from pathlib import Path
from scripts.audio_data.common import read_jsonl
root = Path(sys.argv[1])
has_asr = any(row.get('task') == 'asr' and row.get('split') == 'train'
              and row.get('qc', {}).get('offline', {}).get('decision') == 'accept'
              for row in read_jsonl(root / 'pilot.offline.jsonl'))
has_hypotheses = next(read_jsonl(root / 'qwen_asr.jsonl'), None) is not None
if has_asr and not has_hypotheses:
    raise SystemExit('No ASR hypotheses were produced. Inspect qwen_asr.errors.jsonl and the endpoint before asr-filter.')
PY
      ;;
    asr-filter)
      "$AUDIO_PYTHON" -m scripts.audio_data.filter_offline \
        --input "$AUDIO_ROOT/pilot.offline.jsonl" --output "$AUDIO_ROOT/pilot.asr_checked.jsonl" \
        --allow-noncommercial --max-duration "${AUDIO_MAX_DURATION:-30}" \
        --asr-hypotheses "$AUDIO_ROOT/qwen_asr.jsonl" --reference-check-split train \
        --max-wer "${AUDIO_MAX_WER:-0.30}" --max-cer "${AUDIO_MAX_CER:-0.25}" \
        --progress "$AUDIO_PROGRESS" \
        --checkpoint-dir "${AUDIO_ASR_FILTER_CHECKPOINT_DIR:-$AUDIO_ROOT/checkpoints/asr-filter}" \
        --checkpoint-interval "${AUDIO_CHECKPOINT_EVERY:-1000}" "${audio_checkpoint_reset_args[@]}" \
        --require-reference-check "${audio_holdout_args[@]}"
      ;;
    convert)
      "$AUDIO_PYTHON" -m scripts.audio_data.convert_verl \
        --input "$AUDIO_ROOT/pilot.asr_checked.jsonl" --output "$AUDIO_ROOT/train.opd.parquet" \
        --mode opd --split train --allow-noncommercial
      ;;
    convert-offline)
      "$AUDIO_PYTHON" -m scripts.audio_data.convert_verl \
        --input "$AUDIO_ROOT/pilot.offline.jsonl" --output "$AUDIO_ROOT/train.audio_only.opd.parquet" \
        --mode opd --split train --allow-noncommercial
      ;;
  esac
}

if [[ "$audio_stage" == all ]]; then
  for audio_next_stage in download normalize offline asr asr-filter convert; do
    run_stage "$audio_next_stage"
  done
else
  run_stage "$audio_stage"
fi

#!/usr/bin/env bash
# Evaluate all saved checkpoints on the six mounted audio suites, without SWE.
#
# From the verl checkout:
#   bash running_script_6/full_audio_benchmark.sh
# Preview the resolved configuration without starting workers or creating logs:
#   CONFIG_ONLY=1 bash running_script_6/full_audio_benchmark.sh
# Select one saved step instead of all checkpoints:
#   CHECKPOINT_STEP=16 bash running_script_6/full_audio_benchmark.sh
#
# Optional environment: CHECKPOINT_ROOT, CHECKPOINT_STEP, AUDIO_BENCHMARK_DATA_ROOT,
# AUDIO_BENCHMARK_CONCURRENCY, AUDIO_BENCHMARK_LOAD_BALANCE (defaults to True),
# EXPERIMENT_NAME, AUDIO_FULL_RUN_DIR, CONFIG_ONLY,
# AUDIO_PROCESSOR_SOURCE (defaults to the original data/3552-checkpoint).
# The existing audio_test.sh parallelism/model-generation overrides still apply.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export TARGET_VERL="${TARGET_VERL:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
export SWE_SOURCE="${SWE_SOURCE:-$(cd -- "$TARGET_VERL/.." && pwd)/stock-rl-reflect}"
export CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$SWE_SOURCE/output/checkpoints/qwen35_audio_swe_opd_397b_fp8}"

# Keep the single-checkpoint path unchanged for explicit selection. Otherwise,
# snapshot all completed exports and launch this script sequentially per step.
if [[ -z "${CHECKPOINT_STEP:-}" ]]; then
  exec python3 "$SCRIPT_DIR/full_audio_benchmark_checkpoints.py" "$SCRIPT_DIR/full_audio_benchmark.sh" "$@"
fi

# Load the selected trained HF export directly; no optimizer resume is needed.
checkpoint_info="$(python3 - "$CHECKPOINT_ROOT" "${CHECKPOINT_STEP:-}" "${AUDIO_PROCESSOR_SOURCE:-$SWE_SOURCE/data/3552-checkpoint}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).expanduser().resolve()
step = sys.argv[2].strip()
if not step:
    marker = root / "latest_checkpointed_iteration.txt"
    if not marker.is_file():
        raise SystemExit(f"Completed checkpoint marker not found: {marker}; set CHECKPOINT_STEP explicitly.")
    step = marker.read_text().strip()
if not step.isascii() or not step.isdigit():
    raise SystemExit(f"CHECKPOINT_STEP must be a nonnegative integer, found {step!r}")
step = str(int(step))
model = root / f"global_step_{step}" / "actor" / "model" / "huggingface"
index_path = model / "model.safetensors.index.json"
if not (model / "config.json").is_file() or not index_path.is_file():
    raise SystemExit(f"Complete Hugging Face checkpoint export not found: {model}")
weights = json.loads(index_path.read_text())["weight_map"]
if not any(name.startswith("model.audio_tower.") for name in weights):
    raise SystemExit(f"Checkpoint index has no frozen audio tower: {index_path}")
missing = sorted(shard for shard in set(weights.values()) if not (model / shard).is_file())
if missing:
    raise SystemExit(f"Checkpoint export is missing weight shards: {missing}")
preprocessor = model / "preprocessor_config.json"
if not preprocessor.is_file():
    source = Path(sys.argv[3]).expanduser().resolve()
    preprocessor = source / "preprocessor_config.json"
    if not preprocessor.is_file():
        raise SystemExit(f"Audio preprocessing metadata missing: {preprocessor}; set AUDIO_PROCESSOR_SOURCE.")
    source_config = json.loads((source / "config.json").read_text())
    model_config = json.loads((model / "config.json").read_text())
    if source_config.get("audio_config") != model_config.get("audio_config"):
        raise SystemExit("AUDIO_PROCESSOR_SOURCE audio_config does not match the selected checkpoint")
metadata = json.loads(preprocessor.read_text())
if metadata.get("feature_extractor_type") != "WhisperFeatureExtractor":
    raise SystemExit(f"Expected WhisperFeatureExtractor metadata in {preprocessor}")
mel_bins = json.loads((model / "config.json").read_text()).get("audio_config", {}).get("num_mel_bins")
if mel_bins is not None and metadata.get("feature_size") != mel_bins:
    raise SystemExit(f"Audio feature_size does not match checkpoint num_mel_bins: {preprocessor}")
print(step)
print(model)
print(preprocessor)
PY
)"
mapfile -t checkpoint_fields <<< "$checkpoint_info"
checkpoint_step="${checkpoint_fields[0]}"
checkpoint_model_path="${checkpoint_fields[1]}"
audio_preprocessor_config="${checkpoint_fields[2]}"
export MODEL_PATH="$checkpoint_model_path"

export AUDIO_BENCHMARK_ONLY=True
export AUDIO_BENCHMARK_ENABLED=True
export TRAIN_FILES='[]'
export VAL_FILES='[]'
export AUDIO_BENCHMARK_DATA_ROOT="${AUDIO_BENCHMARK_DATA_ROOT:-/mnt}"
export AUDIO_BENCHMARK_BENCHMARKS='[aishell1,librispeech,wenetspeech,clothoaqa,rul_muchomusic,voicebench_bbh]'
export AUDIO_BENCHMARK_MAX_SAMPLES=null
export AUDIO_BENCHMARK_CONCURRENCY="${AUDIO_BENCHMARK_CONCURRENCY:-8}"
export AUDIO_BENCHMARK_FAILURE_POLICY=error
# GAGE otherwise uses this inherited limit when max_samples is null.
unset GAGE_EVAL_MAX_SAMPLES

for dataset in aishell1 librispeech wenetspeech clothoaqa muchomusic voicebench_bbh; do
  [[ -d "$AUDIO_BENCHMARK_DATA_ROOT/$dataset" ]] || {
    echo "Audio dataset directory not found: $AUDIO_BENCHMARK_DATA_ROOT/$dataset" >&2
    exit 2
  }
done

export EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen35_audio_full_step${checkpoint_step}}"
AUDIO_FULL_RUN_DIR="${AUDIO_FULL_RUN_DIR:-$TARGET_VERL/validation_data/audio_full/$EXPERIMENT_NAME}"
AUDIO_FULL_RUN_DIR="$(python3 -c 'import sys; from pathlib import Path; print(Path(sys.argv[1]).expanduser().resolve())' "$AUDIO_FULL_RUN_DIR")"
# Isolate evaluation artifacts from the training checkpoint and previous logs.
export OUTPUT_DIR="$AUDIO_FULL_RUN_DIR/trainer"
export ROLLOUT_DATA_DIR="$AUDIO_FULL_RUN_DIR/rollouts"
export VALIDATION_DATA_DIR="$AUDIO_FULL_RUN_DIR/validation"
export AUDIO_BENCHMARK_OUTPUT_DIR="$AUDIO_FULL_RUN_DIR/benchmarks"

if [[ "${CONFIG_ONLY:-0}" != "1" ]]; then
  mkdir -p -- "$(dirname -- "$AUDIO_FULL_RUN_DIR")"
  mkdir -- "$AUDIO_FULL_RUN_DIR" || {
    echo "Refusing to reuse an existing evaluation run directory: $AUDIO_FULL_RUN_DIR" >&2
    exit 2
  }
  if [[ ! -f "$MODEL_PATH/preprocessor_config.json" ]]; then
    # Keep checkpoint weights/configuration read-only. Supply only the omitted
    # audio metadata in a separate directory, without copying weight shards.
    python3 - "$MODEL_PATH" "$audio_preprocessor_config" "$AUDIO_FULL_RUN_DIR/model" <<'PY'
import shutil
import sys
from pathlib import Path

source, metadata, destination = map(Path, sys.argv[1:])
destination.mkdir()
for child in source.iterdir():
    (destination / child.name).symlink_to(child)
shutil.copy2(metadata, destination / "preprocessor_config.json")
PY
    export MODEL_PATH="$AUDIO_FULL_RUN_DIR/model"
  fi
fi

overrides=(
  "$@"
  "actor_rollout_ref.model.path=$MODEL_PATH"
  trainer.use_v1=True
  trainer.v1.trainer_mode=sync
  trainer.resume_mode=disable
  trainer.val_before_train=True
  trainer.val_only=True
  trainer.save_freq=-1
  trainer.test_freq=-1
  distillation.enabled=False
  data.train_files='[]'
  data.val_files='[]'
  trainer.audio_benchmark.enabled=True
  trainer.audio_benchmark.only=True
  "trainer.audio_benchmark.benchmarks=$AUDIO_BENCHMARK_BENCHMARKS"
  trainer.audio_benchmark.max_samples=null
)

printf 'Evaluating checkpoint step %s: %s\n' "$checkpoint_step" "$checkpoint_model_path"
printf 'Audio preprocessing metadata: %s\n' "$audio_preprocessor_config"
printf 'Serving model directory: %s\n' "$MODEL_PATH"
printf 'Audio suites: %s (all samples)\n' "$AUDIO_BENCHMARK_BENCHMARKS"
printf 'Run directory: %s\n' "$AUDIO_FULL_RUN_DIR"

if [[ "${CONFIG_ONLY:-0}" == "1" ]]; then
  exec bash "$SCRIPT_DIR/audio_test.sh" "${overrides[@]}"
fi

bash "$SCRIPT_DIR/audio_test.sh" "${overrides[@]}" 2>&1 | tee "$AUDIO_FULL_RUN_DIR/launcher.log"

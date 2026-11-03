#!/usr/bin/env bash
# Stage 2: train proj1/proj2 and the language model; keep the audio encoder frozen.
# Uses the successful frozen step-1 HF checkpoint and a fresh optimizer by default.
# MODEL_PATH and the shared launcher's existing overrides remain available.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export AUDIO_TRAIN_PROJECTOR=1
export AUDIO_FREEZE_ENCODER=1
export AUDIO_FREEZE_PROJECTOR=0
exec bash "$script_dir/opd_397b_multi_teacher.sh" "$@"

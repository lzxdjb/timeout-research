#!/usr/bin/env bash
# Stage 2 reasoning profile: train audio projector and language model.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export AUDIO_TRAIN_PROJECTOR=1
export AUDIO_FREEZE_ENCODER=1
export AUDIO_FREEZE_PROJECTOR=0
exec bash "$script_dir/opd_397b_multi_teacher_reasoning.sh" "$@"

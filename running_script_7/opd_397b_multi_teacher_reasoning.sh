#!/usr/bin/env bash
# Opt-in general reasoning profile; the existing SWE/audio launcher stays intact.
# Additional Hydra arguments override these profile defaults.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export TARGET_VERL="${TARGET_VERL:-$(cd -- "$script_dir/.." && pwd)}"
export SWE_SOURCE="${SWE_SOURCE:-$(cd -- "$TARGET_VERL/.." && pwd)/stock-rl-reflect}"
workspace_root="$(cd -- "$TARGET_VERL/../.." && pwd)"
export GENERAL_REASONING_DATA_ROOT="${GENERAL_REASONING_DATA_ROOT:-$workspace_root/stock-rl-reflect/data/restored}"
export TEACHER_BASE_URL="${TEACHER_BASE_URL:-http://10.248.102.92:18090}"
export AUDIO_TEACHER_BASE_URL="${AUDIO_TEACHER_BASE_URL:-http://10.248.102.8:18090}"
export GENERAL_TEACHER_TOKENIZER_PATH="${GENERAL_TEACHER_TOKENIZER_PATH:-/cpfs01/thscc/sharestorage/iwc/HithinkOmni/user_workspace/models/Qwen3.5-397B-A17B-FP8}"
if [[ "${AUDIO_TRAIN_PROJECTOR:-0}" == 1 ]]; then
  default_experiment=qwen35_swe_audio_reasoning_multi_teacher_projector_v1
else
  default_experiment=qwen35_swe_audio_reasoning_multi_teacher_frozen_v1
fi
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-$default_experiment}"
export GENERAL_REASONING_OUTPUT_DIR="${GENERAL_REASONING_OUTPUT_DIR:-$TARGET_VERL/validation_data/reasoning/$EXPERIMENT_NAME}"

config_helper="$TARGET_VERL/scripts/audio_data/reasoning_multi_teacher_config.py"
export TRAIN_FILES="${TRAIN_FILES:-$(python3 "$config_helper" train)}"
export OPD_MIX_SCHEDULE="${OPD_MIX_SCHEDULE:-$(python3 "$config_helper" mix)}"
reasoning_config="$(python3 "$config_helper" validation)"

# Inherit the base SWE token budget, audio caps/objective, tool penalty and
# validation roster. New routing, image scoring and benchmark hooks are opt-in.
exec bash "$script_dir/opd_397b_multi_teacher.sh" \
  data.custom_cls.path=pkg://verl.utils.dataset.reasoning_opd_dataset \
  data.custom_cls.name=ReasoningOPDMixedDataset \
  actor_rollout_ref.rollout.agent.agent_loop_config_path="$script_dir/reasoning_agent_loop_config.yaml" \
  reward.custom_reward_function.path="$TARGET_VERL/verl/utils/reward_score/swe_audio_reasoning_opd.py" \
  +distillation.teacher_models.swe_teacher.external.enable_image_candidate_scoring=True \
  +distillation.teacher_models.swe_teacher.external.tokenizer_path="$GENERAL_TEACHER_TOKENIZER_PATH" \
  "+trainer.general_reasoning=$reasoning_config" \
  "$@"

# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Read-only predicates for both unresolved Hydra and dataclass configurations."""


def requires_teacher_resource_pool(distillation_config) -> bool:
    """Whether enabled distillation includes a trainer-managed GPU teacher.

    Do not instantiate DistillationConfig here: its preparation rewrites teacher
    inference lengths. Resource allocation must only inspect the configuration.
    """
    if not distillation_config or not distillation_config.get("enabled", False):
        return False
    teachers = dict(distillation_config.get("teacher_models", {}))
    if len(teachers) > 1:
        teachers.pop("teacher_model", None)  # Same default-entry resolution as DistillationConfig.
    # Preserve legacy validation for an enabled but incomplete configuration.
    return not teachers or any(teacher.get("backend", "managed") != "external" for teacher in teachers.values())


def is_pure_distillation(distillation_config) -> bool:
    """True only when task rewards do not participate in the actor objective."""
    return bool(
        distillation_config
        and distillation_config.get("enabled", False)
        and not distillation_config.get("distillation_loss", {}).get("use_task_rewards", True)
    )

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

"""Entry point for SFT with V1 RL generation validation."""

from pprint import pprint

import hydra
import ray
from omegaconf import DictConfig, OmegaConf

from verl.trainer.main_ppo import run_ppo
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.trainer.sft_rl_trainer import SFTRLTrainer
from verl.utils.config import validate_config
from verl.utils.device import auto_set_device
from verl.utils.import_utils import load_class_from_fqn
from verl.utils.logging_utils import configure_verl_logging


@ray.remote
class TaskRunnerSFTRL:
    """Initialize the colocated actor/rollout stack and run hybrid SFT."""

    def __init__(self):
        self.config = None
        self.trainer = None
        self.agent_loop_manager = None

    def _init_agent_loop_manager(self):
        from verl.trainer.ppo.v1 import AgentLoopManagerTQ

        manager_class_fqn = self.config.actor_rollout_ref.rollout.get("agent", {}).get(
            "agent_loop_manager_class"
        )
        if manager_class_fqn:
            manager_class = load_class_from_fqn(manager_class_fqn, "AgentLoopManager")
        else:
            manager_class = AgentLoopManagerTQ
        self.agent_loop_manager = manager_class.create(
            config=self.config,
            llm_client=self.trainer.get_llm_client(),
            teacher_client=self.trainer.get_teacher_client(),
            reward_loop_worker_handles=self.trainer.get_reward_handles(),
        )

    def run(self, config: DictConfig):
        configure_verl_logging()

        import transfer_queue as tq

        config.transfer_queue.enable = True
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)
        self.config = config

        tq.init(config.transfer_queue)
        succeeded = False
        try:
            self.trainer = SFTRLTrainer(config=config)
            self.trainer.init()
            self._init_agent_loop_manager()
            self.trainer.fit(self.agent_loop_manager)
            succeeded = True
        finally:
            try:
                tracking = getattr(self.trainer, "logger", None)
                if tracking is not None:
                    tracking.finish(exit_code=0 if succeeded else 1)
            finally:
                tq.close()


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    auto_set_device(config)
    validate_config(
        config=config,
        use_reference_policy=need_reference_policy(config),
        use_critic=need_critic(config),
    )
    run_ppo(config, task_runner_class=TaskRunnerSFTRL)


if __name__ == "__main__":
    main()

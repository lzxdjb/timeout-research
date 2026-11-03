"""A separate single-turn loop for the opt-in reasoning profile."""

from verl.experimental.agent_loop.agent_loop import register
from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop


@register("general_reasoning_agent")
class GeneralReasoningAgentLoop(SingleTurnAgentLoop):
    """Use reasoning chat formatting and the existing rollout response budget."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.apply_chat_template_kwargs = {**self.apply_chat_template_kwargs, "enable_thinking": True}

from ShaLab.models import Generator
from ShaLab.data import prepare_profile_memory

from .base_agent import BaseAgent


class ModelBasedAgent(BaseAgent):
    def __init__(
        self,
        model,
        max_length=20,
        policy='sample',
        num_candidates=1,
        context=None,
        length_normalization_factor=0,
        length_normalization_const=0,
    ):
        super(ModelBasedAgent, self).__init__(
            model.get_device_id(),
            num_candidates,
            context,
        )
        self.max_length = max_length
        self.policy = policy
        self.length_normalization_factor = length_normalization_factor
        self.length_normalization_const = length_normalization_const
        self.model = model
        self.generator = Generator(self.model)

    def set_context(self, context):
        self.context = context
        if isinstance(context, list):
            context = prepare_profile_memory(
                context,
                self.model.pad_token_id,
            )
        self.context = self.model.to_device(context)
        if len(self.context.shape) == 2:
            self.context = self.context.unsqueeze(1)

    def generate(self, input_source):
        return self.generator.generate(
            input_source=input_source,
            max_length=self.max_length,
            policy=self.policy,
            num_candidates=self.num_candidates,
            context=self.context,
            length_normalization_factor=self.length_normalization_factor,
            length_normalization_const=self.length_normalization_const,
        )

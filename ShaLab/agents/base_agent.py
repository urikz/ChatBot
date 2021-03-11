
class BaseAgent(object):
    def __init__(self, device_id=None, num_candidates=1, context=None):
        self.device_id = device_id
        self.num_candidates = num_candidates
        self.context = context

    def set_context(self, context):
        self.context = context
        if len(self.context.shape) == 2:
            self.context = self.context.unsqueeze(1)

    def generate(self, input_source):
        raise NotImplementedError()

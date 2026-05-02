from .registry import register
from .abstract import EvalFunc

@register("base")
class BaseEvalFunc(EvalFunc):

    def __init__(self, llm, **kwargs):
        super().__init__(llm)

    def evaluate(self, examples):
        """
        Batch inference function:
        - Input: examples containing "prompt"
        - Output: updated prompts with assistant responses appended
        """
        batched_prompts = examples["prompt"]
        batched_responses = self.llm.generate(batched_prompts)

        return {"response": batched_responses}
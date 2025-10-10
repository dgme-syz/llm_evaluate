from .registry import register
from .abstract import EvalFunc

@register("no_steps")
class NoStepsEvalFunc(EvalFunc):

    def __init__(self, llm):
        super().__init__(llm)

    def evaluate(self, examples):
        """
        Batch inference function:
        - Input: examples containing "prompt"
        - Output: updated prompts with assistant responses appended
        """
        batched_prompts = examples["prompt"]

        for i in range(len(batched_prompts)):
            batched_prompts[i] += "The answer is"

        batched_responses = self.llm.generate(batched_prompts)
        for i in range(len(batched_responses)):
            for j in range(len(batched_responses[i])):
                batched_responses[i][j] = batched_responses[i][j].split(".")[0].strip()

        return {"response": batched_responses}
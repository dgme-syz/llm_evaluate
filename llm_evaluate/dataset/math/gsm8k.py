from llm_evaluate.dataset import EvalDataset
from llm_evaluate.dataset.translation import LANG_DICT
from llm_evaluate.dataset import register

_SOLUTION_CLIP_CHARS = 300

def extract_solution(solution_str, method="strict"):
    import re
    assert method in ["strict", "flexible"]

    # Optimization: Regular expression matching on very long strings can be slow.
    # For math problems, the final answer is usually at the end.
    # We only match on the last 300 characters, which is a safe approximation for 300 tokens.
    if len(solution_str) > _SOLUTION_CLIP_CHARS:
        solution_str = solution_str[-_SOLUTION_CLIP_CHARS:]

    if method == "strict":
        # this also tests the formatting of the model
        solutions = re.findall("#### (\\-?[0-9\\.\\,]+)", solution_str)
        if len(solutions) == 0:
            final_answer = None
        else:
            # take the last solution
            final_answer = solutions[-1].replace(",", "").replace("$", "")
    elif method == "flexible":
        answer = re.findall("(\\-?[0-9\\.\\,]+)", solution_str)
        final_answer = None
        if len(answer) == 0:
            # no reward is there is no answer
            pass
        else:
            invalid_str = ["", "."]
            # find the last number that is not '.'
            for final_answer in reversed(answer):
                if final_answer not in invalid_str:
                    break
    return final_answer



@register("gsm8k")
class gsm8k(EvalDataset):
    system_prompt = "You are a helpful assistant."
    prompt = "{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}."

    def __init__(self, data_dir, subset_name=None, split="train", builder=None):
        super().__init__(data_dir, subset_name, split, builder)

    def convert_item(self, examples, **kwargs):

        message_list = []
        if hasattr(self, "system_prompt") and self.system_prompt is not None:
            message_list.append({"role": "system", "content": self.system_prompt})
        if hasattr(self, "prompt") and self.prompt is not None:
            message_list.append(
                {"role": "user", "content": self.prompt.format(question=examples["question"])}
            )

        item = {
            "data_source": "gsm8k",
            "prompt": message_list,
            "ability": "math",
            "reward_model": {
                "ground_truth": extract_solution(examples["answer"]),
                "style": "rule",
            },
            "extra_info": {}
        }
        return item




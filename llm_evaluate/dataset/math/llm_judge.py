import re

from llm_evaluate.dataset import EvalDataset, register

@register("llm_judge")
class llm_judge(EvalDataset):

    system_prompt: str = "You are a helpful assistant.You are given a problem and a solution; the solution's final answer is correct, but the steps may not be. Ignore any code parts in the solution—only consider whether the calculation steps exist and are correct (disregarding the code). Strictly Check whether the solution's steps both exist and are correct. If they exist and are correct, output \\boxed{1} at the end; otherwise output \\boxed{0}."
    user_prompt: str = "Problem: {question}\nSolution: {solution}\n"

    def __init__(self, data_dir: str, subset_name=None, split: str = "train", builder=None, **kwargs):
        """
        Initialize the GSM8K dataset wrapper.

        Args:
            data_dir (str): Directory containing the dataset.
            subset_name (optional): Not used for GSM8K.
            split (str, optional): Dataset split ('train', 'test', etc.).
            builder (callable, optional): Custom dataset builder.
        """
        super().__init__(data_dir, subset_name, split, builder)

    def convert_item(self, examples: dict, **kwargs) -> dict:
        """
        Convert a single dataset example into structured format for LLM evaluation.

        Args:
            examples (dict): A single example with keys 'question' and 'answer'.

        Returns:
            dict: Structured item with system/user prompt, ability, reward model, and extra info.
        """
        ret = examples
        raw_prompt = ret.pop("prompt")
        pattern = r'Question:(.*?)\nPlease reason step by step, and put your final answer within'
        match = re.search(pattern, raw_prompt, re.DOTALL)
        if match:
            question = match.group(1).strip()
        else:
            raise ValueError(f"Prompt format is incorrect: {raw_prompt}")
        
        ret.update({"question": question})
        ret.update({"prompt": []})
        gt = ret["reward_model"].pop("ground_truth")
        ret["reward_model"].update({"solution": gt, "ground_truth": 0})

        if match:
            question = match.group(1).strip()

        num_responses = len(examples["response"])
        assert num_responses > 0
        for i in range(num_responses):
            if examples.get("union_math_accuracy_detail_results", examples.get("math_train_union_math_accuracy_detail_results"))[i] > 0.0:
                ret["prompt"].append([
                    {"role": "system", "content": self.system_prompt},
                    {
                        "role": "user",
                        "content": self.user_prompt.format(
                            question=ret.get("problem", ret.get("question")),
                            solution=ret["response"][i],
                        ),
                    },
                ])
        return ret

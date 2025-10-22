from llm_evaluate.dataset import EvalDataset, register

import re

# Number of characters to clip from the end for solution extraction
_SOLUTION_CLIP_CHARS = 300


def extract_solution(solution_str: str, method: str = "strict") -> str | None:
    """
    Extract the numeric solution from a string, typically a math problem answer.

    Args:
        solution_str (str): The string containing the solution.
        method (str, optional): Extraction mode. 'strict' checks for formatted solution,
                                'flexible' extracts the last numeric value. Defaults to 'strict'.

    Returns:
        str | None: Extracted numeric solution, or None if not found.
    """
    assert method in ["strict", "flexible"], "method must be either 'strict' or 'flexible'"

    # Clip the string to the last N characters for performance
    if len(solution_str) > _SOLUTION_CLIP_CHARS:
        solution_str = solution_str[-_SOLUTION_CLIP_CHARS:]

    final_answer: str | None = None

    if method == "strict":
        # Look for specifically formatted solution: "#### <number>"
        solutions = re.findall(r"#### (\-?[0-9\.,]+)", solution_str)
        if solutions:
            # Take the last solution, remove commas and dollar signs
            final_answer = solutions[-1].replace(",", "").replace("$", "")
    elif method == "flexible":
        # Extract all numeric patterns
        answers = re.findall(r"(\-?[0-9\.,]+)", solution_str)
        invalid_values = {"", "."}

        # Take the last valid numeric value
        for ans in reversed(answers):
            if ans not in invalid_values:
                final_answer = ans
                break

    return final_answer


@register("gsm8k")
class gsm8k(EvalDataset):

    # name: gsm8k        
    # data_path: /home/nfs06/shenyz/data/gsm8k_hf
    # subset_name: main
    # split: test
    # num_samples: null


    """
    Dataset class for GSM8K math problem evaluation.

    Constructs a system + user prompt and extracts the ground-truth numeric answer.
    """
    system_prompt: str = "You are a helpful assistant."
    prompt: str = "{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
    simple_prompt: str = "Question:{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.\n\nAnswer:\n"

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
        # message_list = []

        # if getattr(self, "system_prompt", None):
        #     message_list.append({"role": "system", "content": self.system_prompt})

        # if getattr(self, "prompt", None):
        #     message_list.append({
        #         "role": "user",
        #         "content": self.prompt.format(question=examples["question"])
        #     })

        prompt = self.simple_prompt.format(question=examples["question"])
        return {
            "data_source": "gsm8k",
            "prompt": prompt,
            "ability": "math",
            "reward_model": {
                "ground_truth": extract_solution(examples["answer"]),
                "style": "rule",
            },
            "extra_info": {}
        }

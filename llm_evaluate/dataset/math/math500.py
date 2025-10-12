import re

from llm_evaluate.dataset import EvalDataset, register

@register("math_train")
class mathtrain(EvalDataset):

#   name: math_train      
#   data_path: /home/nfs06/shenyz/data/SimpleRL/hard/train_processed.parquet
#   subset_name: null
#   split: train
#   num_samples: 512
#   builder: parquet

    """
    Dataset class for GSM8K math problem evaluation.

    Constructs a system + user prompt and extracts the ground-truth numeric answer.
    """
    system_prompt: str = "You are a helpful assistant."
    prompt: str = "{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
    simple_prompt: str = "Question:{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.\n\nAnswer:\n"

    def __init__(self, data_dir: str, subset_name=None, split: str = "train", builder=None):
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
        prompt = self.simple_prompt.format(question=examples.get("problem", examples.get("question")))
        return {
            "data_source": "math_train",
            "prompt": prompt,
            "ability": "math",
            "reward_model": {
                "ground_truth": examples["answer"],
                "style": "rule",
            },
            "extra_info": {}
        }

@register("math500")
class math500(EvalDataset):

    # name: math500        
    # data_path: /home/nfs06/shenyz/data/math500
    # subset_name: null
    # split: test
    # num_samples: null

    """
    Dataset class for GSM8K math problem evaluation.

    Constructs a system + user prompt and extracts the ground-truth numeric answer.
    """
    system_prompt: str = "You are a helpful assistant."
    prompt: str = "{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
    simple_prompt: str = "Question:{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.\n\nAnswer:\n"

    def __init__(self, data_dir: str, subset_name=None, split: str = "train", builder=None):
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
        prompt = self.simple_prompt.format(question=examples.get("problem", examples.get("question")))
        return {
            "data_source": "math500",
            "prompt": prompt,
            "ability": "math",
            "reward_model": {
                "ground_truth": examples["answer"],
                "style": "rule",
            },
            "extra_info": {}
        }

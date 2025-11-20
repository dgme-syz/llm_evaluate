from llm_evaluate.dataset import EvalDataset, register
from llm_evaluate.dataset.translation import (
    LANGUAGE_BY_CODE,
    get_prompt_template
)


@register("challenge_set")
class challenge_set(EvalDataset):
# name: challenge_set        
# data_path: /home/nfs06/shenyz/data/seed_x_challenge_set
# subset_name: en-zh_CN
# split: train

    # https://github.com/ByteDance-Seed/Seed-X-7B/blob/main/challenge_set/Challenge_set_en2zh.json
    def __init__(self, data_dir, subset_name=None, split="train", builder=None, extra_args=None):
        """
        Initialize the WMT24 dataset wrapper.

        Args:
            data_dir (str): Directory containing the dataset.
            subset_name (str, optional): Language pair, e.g., 'en-zh_CN'.
            split (str, optional): Dataset split to load ('train', 'test', etc.).
            builder (callable, optional): Custom dataset builder.
            extra_args (dict, optional): Additional arguments for dataset processing.
        """
        super().__init__(data_dir, subset_name, split, builder)

        # Extract source and target languages from subset_name
        src_code, tgt_code = subset_name.strip().split("-")
        self.src_lang = LANGUAGE_BY_CODE.get(src_code, src_code)
        self.tgt_lang = LANGUAGE_BY_CODE.get(tgt_code, tgt_code)
        self.src_code = src_code
        self.tgt_code = tgt_code.split("_")[0]
        self.extra_args = extra_args or {}
        if "prompt_template" not in self.extra_args:
            raise ValueError("This dataset must need a standart prompt template.")
        self.template_func = get_prompt_template(self.extra_args.get("prompt_template"))
        self.src_key_name = extra_args.get("src_key")
        self.tgt_key_name = extra_args.get("tgt_key")

    def convert_item(self, examples, **kwargs):
        """
        Convert a single dataset example into structured format for LLM evaluation.

        Args:
            examples (dict): A single example from the dataset with 'source' and 'target' keys.

        Returns:
            dict: Structured item with prompt, ability, reward model, and extra info.
        """
        src_text = examples[self.src_key_name]
        tgt_text = examples[self.tgt_key_name]
        prompt = self.template_func(self.src_lang, self.tgt_lang, src_text)
        
        return {
            "data_source": "challenge_set",
            "prompt": prompt,
            "ability": "translation",
            "reward_model": {
                "ground_truth": tgt_text,
                "style": "rule",
            },
            "extra_info": {
                "src_lang": self.src_code,
                "tgt_lang": self.tgt_code,
                "src": src_text,
            }
        }

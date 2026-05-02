from llm_evaluate.dataset import EvalDataset, register
from llm_evaluate.dataset.translation import (
    get_prompt_template,
    resolve_lang_name,
)


@register("train_mixed")
class train_mixed_dataset(EvalDataset):
    """Dataset wrapper for the WMT 2017–2020 mixed training set used in MT-R1-Zero.

    Reference:
        https://github.com/fzp0424/MT-R1-Zero/blob/main/data/train/json/train_enzh_6565.jsonl
    """

    def __init__(self, data_dir, subset_name=None, split="train", builder=None, extra_args=None):
        """Initialize the train_mixed dataset wrapper.

        Args:
            data_dir: Directory containing the dataset.
            subset_name: Language pair, e.g. ``"en-zh_CN"``.
            split: Dataset split to load.
            builder: Optional custom dataset builder.
            extra_args: Additional dataset arguments. Must contain ``prompt_template``,
                ``src_key`` and ``tgt_key``.
        """
        super().__init__(data_dir, subset_name, split, builder)

        self.extra_args = extra_args or {}
        if "prompt_template" not in self.extra_args:
            raise ValueError(
                "train_mixed dataset requires `prompt_template` in extra_args."
            )
        if not isinstance(subset_name, str) or "-" not in subset_name:
            raise ValueError(
                f"train_mixed expects subset_name like 'en-zh_CN', got {subset_name!r}."
            )

        src_code, tgt_code = subset_name.strip().split("-", 1)
        self.src_code = src_code
        self.tgt_code = tgt_code.split("_")[0]
        self.src_lang = resolve_lang_name(src_code)
        self.tgt_lang = resolve_lang_name(tgt_code)
        self.template_func = get_prompt_template(self.extra_args["prompt_template"])
        self.src_key_name = self.extra_args.get("src_key")
        self.tgt_key_name = self.extra_args.get("tgt_key")
        if self.src_key_name is None or self.tgt_key_name is None:
            raise ValueError(
                "train_mixed requires `src_key` and `tgt_key` in extra_args / dataset_args."
            )

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
            "data_source": "wmt17-20",
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

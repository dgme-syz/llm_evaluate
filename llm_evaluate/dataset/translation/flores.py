from llm_evaluate.dataset import EvalDataset, register
from llm_evaluate.dataset.translation import (
    LANG_DICT,
    get_prompt_template,
    resolve_lang_name,
)


@register("flores")
class flores(EvalDataset):
    """Dataset wrapper for the FLORES parallel translation evaluation set.

    Expects two parallel sources (one per language). Builds a prompt via the
    shared template registry and emits items compatible with the LLM evaluation
    pipeline.
    """

    def __init__(self, data_dir, subset_name=None, split="train", builder=None, extra_args=None):
        """Initialize the FLORES dataset wrapper.

        Args:
            data_dir: Directory or list of directories containing the parallel data.
            subset_name: Tuple/list of two subset names ``(source_subset, target_subset)``,
                e.g. ``("flores_en", "flores_zh")``.
            split: Dataset split to load (``"train"``, ``"test"``, ...).
            builder: Optional custom dataset builder.
            extra_args: Additional dataset arguments. Must contain ``prompt_template``.
        """
        super().__init__(data_dir, subset_name, split, builder)

        self.extra_args = extra_args or {}
        if "prompt_template" not in self.extra_args:
            raise ValueError(
                "flores dataset requires `prompt_template` in extra_args "
                "(set llm.prompt_template / server.prompt_template in the master config)."
            )

        if subset_name is None or len(subset_name) != 2:
            raise ValueError(
                f"flores expects subset_name of length 2 (source, target), got {subset_name!r}."
            )
        src_code, tgt_code = subset_name
        self.src_code = src_code.split("_")[-1]
        self.tgt_code = tgt_code.split("_")[-1]
        self.src_lang = LANG_DICT.get(self.src_code, src_code)
        self.tgt_lang = LANG_DICT.get(self.tgt_code, tgt_code)
        self.template_func = get_prompt_template(self.extra_args["prompt_template"])

    def convert_item(self, examples: list, **kwargs) -> dict:
        """
        Convert a single dataset example into structured format for LLM evaluation.

        Args:
            examples (list[dict]): A single example from the dataset. 
            Expecting [source_dict, target_dict] with 'text' keys.

        Returns:
            dict: Structured item with prompt, ability, reward model, and extra info.
        """
        src_text = examples[0]["text"]
        tgt_text = examples[1]["text"]

        prompt = self.template_func(self.src_lang, self.tgt_lang, src_text)
        return {
            "data_source": "flores",
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

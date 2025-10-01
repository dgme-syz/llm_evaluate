from llm_evaluate.dataset import EvalDataset, register
from llm_evaluate.dataset.translation import LANG_DICT  # Mapping from language codes to language names


@register("flores")
class flores(EvalDataset):
    """
    Dataset class for FLORES translation evaluation.

    It constructs a prompt for translation tasks and returns structured items
    compatible with LLM evaluation pipelines.
    """
    prompt = (
        "Translate the following text from {src_lang} to {tgt_lang}, "
        "without additional explanation.\n\n"
        "{src_lang} source:\n{src}\n\n"
        "{tgt_lang} translation:"
    )

    def __init__(self, data_dir, subset_name=None, split="train", builder=None):
        """
        Initialize the FLORES dataset wrapper.

        Args:
            data_dir (str): Directory containing the dataset.
            subset_name (tuple[str, str], optional): Source and target language codes, e.g., ('en_XX', 'zh_CN').
            split (str, optional): Dataset split to load ('train', 'test', etc.).
            builder (callable, optional): Custom dataset builder.
        """
        super().__init__(data_dir, subset_name, split, builder)

        # Extract source and target languages safely
        src_code, tgt_code = subset_name
        self.src_lang = LANG_DICT.get(src_code.split("_")[-1], src_code)
        self.tgt_lang = LANG_DICT.get(tgt_code.split("_")[-1], tgt_code)

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

        return {
            "data_source": "flores",
            "prompt": [{
                "content": self.prompt.format(
                    src_lang=self.src_lang,
                    src=src_text,
                    tgt_lang=self.tgt_lang,
                ),
                "role": "user"
            }],
            "ability": "translation",
            "reward_model": {
                "ground_truth": tgt_text,
                "style": "rule",
            },
            "extra_info": {
                "src_lang": self.src_lang,
                "tgt_lang": self.tgt_lang,
                "src": src_text,
            }
        }

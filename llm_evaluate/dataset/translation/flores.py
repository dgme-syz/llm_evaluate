from llm_evaluate.dataset import EvalDataset, register
from llm_evaluate.dataset.translation import LANG_DICT  # Mapping from language codes to language names
from llm_evaluate.dataset.translation import (
    qwen_chat_input, 
    hunyuan_chat_input,
    seed_x_ppo_input,
    qwen_think_input
)
#   name: flores        
#   data_path: 
#     - /home/nfs06/shenyz/data/BenchMAX_General_Translation
#     - /home/nfs06/shenyz/data/BenchMAX_General_Translation
#   subset_name:
#     - flores_en
#     - flores_ja
#   split:
#     - train
#     - train
#   num_samples: null


@register("flores")
class flores(EvalDataset):
    """
    Dataset class for FLORES translation evaluation.

    It constructs a prompt for translation tasks and returns structured items
    compatible with LLM evaluation pipelines.
    """

    def __init__(self, data_dir, subset_name=None, split="train", builder=None, extra_args=None):
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
        self.src_code = src_code.split("_")[-1]
        self.tgt_code = tgt_code.split("_")[-1]
        self.extra_args = extra_args or {}

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

        m = self.extra_args.get("model").lower()
        if "qwen" in m:
            if "think" not in m:
                prompt = qwen_chat_input(self.src_lang, self.tgt_lang, src_text)
            else:
                prompt = qwen_think_input(self.src_lang, self.tgt_lang, src_text)
        elif "hunyuan" in m:
            prompt = hunyuan_chat_input(self.src_lang, self.tgt_lang, src_text)
        elif "seed-x" in m:
            prompt = seed_x_ppo_input(self.src_lang, self.tgt_lang, src_text)
        else:
            raise ValueError(f"Unsupported model {m} for challenge_set dataset.")

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

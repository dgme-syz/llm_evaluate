from llm_evaluate.dataset import EvalDataset
from llm_evaluate.dataset.translation import LANG_DICT
from llm_evaluate.dataset import register

@register("flores")
class flores(EvalDataset):
    prompt = "Translate the following text from {src_lang} to {tgt_lang}, without additional explanation.\n\n{src_lang} source:\n{src}\n\n{tgt_lang} translation:"

    def __init__(self, data_dir, subset_name=None, split="train", builder=None):
        super().__init__(data_dir, subset_name, split, builder)
        self.src_lang = LANG_DICT[subset_name[0].split("_")[-1]]
        self.tgt_lang = LANG_DICT[subset_name[1].split("_")[-1]]

    def convert_item(self, examples, **kwargs):
        src_text = examples[0]["text"]
        tgt_text = examples[1]["text"]
        item = {
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
            "extra_info": {"src_lang": self.src_lang, "tgt_lang": self.tgt_lang}
        }
        return item




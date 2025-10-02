import json
import subprocess
import tempfile
from pathlib import Path

import sacrebleu
from llm_evaluate.utils.metric.registry import register
from llm_evaluate.utils.metric.abstract import Metric


@register("BLEU")
class BLEU(Metric):
    """SacreBLEU metric with language-specific tokenization."""

    def __call__(self, responses, references, extra_infos=None) -> float:
        assert len(responses) == len(references), (
            "The number of translations should be equal to the number of references"
        )
        tgt_lang = "en"
        if extra_infos and len(extra_infos) > 0:
            tgt_lang = extra_infos[0].get("tgt_lang", "en")

        # Choose tokenizer based on target language
        if tgt_lang == "zh":
            tokenizer = "zh"
        elif tgt_lang == "ja":
            tokenizer = "ja-mecab"
        elif tgt_lang == "ko":
            tokenizer = "ko-mecab"
        else:
            tokenizer = "13a"

        result = sacrebleu.corpus_bleu(
            responses,
            [references],
            tokenize=tokenizer,
            force=True,
        )
        return result.score


@register("spBLEU")
class spBLEU(Metric):
    """SacreBLEU metric with flores200 tokenizer."""

    def __call__(self, responses, references, extra_infos=None) -> float:
        assert len(responses) == len(references), (
            "The number of translations should be equal to the number of references"
        )
        result = sacrebleu.corpus_bleu(
            responses,
            [[x] for x in references],
            tokenize="flores200",
            force=True,
        )
        return result.score


def build_Comet_cls(model_name: str):
    class DynamicComet(Metric):
        def __init__(self):
            self.model_name = model_name

        def __call__(self, responses, references, extra_infos=None) -> dict:
            if extra_infos is None:
                raise ValueError("extra_infos must be provided for Comet evaluation.")

            sources = [x["src"] for x in extra_infos]
            assert len(responses) == len(references) == len(sources), "Mismatch lengths"

            data = [{"src": src, "mt": hyp, "ref": ref} 
                    for src, hyp, ref in zip(sources, responses, references)]

            with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".json") as f:
                json.dump(data, f)
                input_file = f.name

            output_file = Path(tempfile.gettempdir()) / f"comet_result_{Path(input_file).stem}.json"

            subprocess.run([
                "python",
                "./llm_evaluate/utils/metric/tools/comet_worker.py",
                input_file,
                self.model_name,
                str(output_file)
            ], check=True)

            with open(output_file, "r") as f:
                prediction = json.load(f)
            print(prediction)
            return prediction

    return DynamicComet

for cls_name, model_name in [
    ("xComet-xxl", "Unbabel/XCOMET-XXL"), 
    ("cometkiwi", "Unbabel/wmt22-cometkiwi-da"),
    ("comet-22", "Unbabel/wmt22-comet-da")
]:
    cls = build_Comet_cls(model_name)
    register(cls_name)(cls)


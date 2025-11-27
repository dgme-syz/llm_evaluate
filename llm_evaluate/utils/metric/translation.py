import re
import gc

import sacrebleu
import torch

from llm_evaluate.utils.metric.registry import register
from llm_evaluate.utils.metric.abstract import Metric


def preprocess(text: str) -> str:
    if not isinstance(text, str):
        raise ValueError(f"{text} is not a str.")
    parts = re.split(r'</think\s*>', text, flags=re.IGNORECASE)
    text_after_think = parts[-1] if len(parts) > 1 else text

    match = re.search(r'<text\s*>(.*?)</text\s*>', text_after_think, flags=re.IGNORECASE | re.DOTALL)
    if match:
        extracted = match.group(1)
    else:
        extracted = text_after_think  

    return extracted.strip()


@register("BLEU")
class BLEU(Metric):
    """SacreBLEU metric with language-specific tokenization."""

    def __call__(self, responses, references, extra_infos=None) -> float:

        responses = [preprocess(x) for x in responses]
        assert len(responses) == len(references), (
            "The number of translations should be equal to the number of references"
        )
        tgt_lang = "en"
        if extra_infos and len(extra_infos) > 0:
            tgt_lang = extra_infos[0].get("tgt_lang", "en")
            print(f"Detected target language: {tgt_lang}")

        # Choose tokenizer based on target language
        if tgt_lang == "zh":
            tokenizer = "zh"
        elif tgt_lang == "ja":
            tokenizer = "ja-mecab"
        elif tgt_lang == "ko":
            tokenizer = "ko-mecab"
        else:
            tokenizer = "13a"
        print(
            f"Preview of responses and references:\nResponse: {responses[0]}\nReference: {references[0]}"
        )
        result = sacrebleu.corpus_bleu(
            responses,
            [references],
            tokenize=tokenizer,
            force=True,
        )
        return {"score": result.score}

@register("sentenceBLEU")
class sentenceBLEU(Metric):
    """Sentence-level BLEU metric."""

    def __call__(self, responses, references, extra_infos=None) -> float:
        responses = [preprocess(x) for x in responses]
        assert len(responses) == len(references), (
            "The number of translations should be equal to the number of references"
        )
        scores = []
        tgt_lang = "en"
        if extra_infos and len(extra_infos) > 0:
            tgt_lang = extra_infos[0].get("tgt_lang", "en")
            print(f"Detected target language: {tgt_lang}")

        # Choose tokenizer based on target language
        if tgt_lang == "zh":
            tokenizer = "zh"
        elif tgt_lang == "ja":
            tokenizer = "ja-mecab"
        elif tgt_lang == "ko":
            tokenizer = "ko-mecab"
        else:
            tokenizer = "13a"
        
        for resp, ref in zip(responses, references):
            result = sacrebleu.sentence_bleu(
                resp,
                [ref],
                tokenize=tokenizer,
            )
            scores.append(result.score)
        average_score = sum(scores) / len(scores)
        return {"score": average_score, "extra_dict": {"score_per_example": scores}}

@register("BLEURT")
class BLEURT(Metric):
    """BLEURT metric."""

    def __init__(self, model_path: str = "/home/nfs05/shenyz/bleurt/bleurt/BLEURT-20") -> None:
        try:
            from bleurt import score
        except ImportError:
            raise ImportError("Please install BLEURT: pip install bleurt")
        self.scorer = score.BleurtScorer(model_path)

    def __call__(self, responses, references, extra_infos=None) -> float:
        responses = [preprocess(x) for x in responses]
        assert len(responses) == len(references), (
            "The number of translations should be equal to the number of references"
        )
        scores = self.scorer.score(references=references, candidates=responses)
        average_score = sum(scores) / len(scores)
        return {"score": average_score, "extra_dict": {"score_per_example": scores}}

    def __del__(self):
        del self.scorer
        
        gc.collect()
        torch.cuda.empty_cache()


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
        return {"score": result.score}



from comet import download_model, load_from_checkpoint
from comet.models.utils import Prediction

def build_Comet_cls(model_name: str):
    class DynamicComet(Metric):
        def __init__(self):
            

            torch.serialization.add_safe_globals([Prediction])

            self.model_name = model_name
            self.model = None
            self.bsz = 64
            # 48GB GPU memory, self.bsz = 64

        def __call__(self, responses, references, extra_infos=None) -> dict:
            if self.model is None:
                model_path = download_model(self.model_name)
                self.model = load_from_checkpoint(model_path)

            responses = [preprocess(x) for x in responses]
            if extra_infos is None:
                raise ValueError("extra_infos must be provided for Comet evaluation.")

            sources = [x["src"] for x in extra_infos]
            assert len(responses) == len(references) == len(sources), "Mismatch lengths"

            data = [{"src": src, "mt": hyp, "ref": ref} 
                    for src, hyp, ref in zip(sources, responses, references)]

            preds = self.model.predict(data, batch_size=self.bsz, gpus=1) # gpus must be 1
            outputs: dict = {"extra_dict": {}}
            if hasattr(preds, "system_score"):
                outputs["score"] = float(preds.system_score)
            if hasattr(preds, "scores"):
                outputs["extra_dict"]["score_per_example"] = list(preds.scores)

            return outputs
        def __del__(self):
            if self.model is not None:
                del self.model
                torch.cuda.empty_cache()
    return DynamicComet

for cls_name, model_name in [
    ("xcomet-xxl", "Unbabel/XCOMET-XXL"), 
    ("cometkiwi", "Unbabel/wmt23-cometkiwi-da-xl"),
    ("comet-22", "Unbabel/wmt22-comet-da")
]:
    cls = build_Comet_cls(model_name)
    register(cls_name)(cls)


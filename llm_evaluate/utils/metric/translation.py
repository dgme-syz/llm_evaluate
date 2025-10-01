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
            self.comet_model = None
            self.saved_directory = None

        def _load_model(self):
            """Lazy-load the COMET model when first needed."""
            if self.comet_model is None:
                from comet import load_from_checkpoint, download_model

                model_path = None
                if self.saved_directory == None:
                    print("In this setting, you will download the model from hf")
                else:
                    model_path = self.saved_directory

                if model_path == None:
                    model_path = download_model(
                        self.model_name, 
                        saving_directory=self.saving_directory
                    )
                self.comet_model = load_from_checkpoint(model_path, local_files_only=False)

        def __call__(self, responses, references, extra_infos=None) -> dict:
            if extra_infos is None:
                raise ValueError("extra_infos must be provided for Comet evaluation.")

            sources = [x["src"] for x in extra_infos]
            assert len(responses) == len(references) == len(sources), (
                "The number of translations, references, and sources must match."
            )

            self._load_model()

            # Prepare data for COMET
            data = [
                {"src": src, "mt": hyp, "ref": ref}
                for src, hyp, ref in zip(sources, responses, references)
            ]

            # Detect device for COMET
            import torch
            gpus = 1 if torch.cuda.is_available() else 0

            # Run prediction
            prediction = self.comet_model.predict(data, batch_size=16, gpus=gpus)

            # Build output dict
            outputs: dict = {"extra_dict": {}}

            if hasattr(prediction, "system_score"):
                outputs["score"] = float(prediction.system_score)

            if hasattr(prediction, "scores"):
                outputs["extra_dict"]["score_per_example"] = list(prediction.scores)

            return outputs
    return DynamicComet

for cls_name, model_name in [
    ("xComet-xxl", "Unbabel/XCOMET-XXL"), 
    ("cometkiwi", "Unbabel/wmt22-cometkiwi-da"),
    ("comet-22", "Unbabel/wmt22-comet-da")
]:
    cls = build_Comet_cls(model_name)
    register(cls_name)(cls)


import sacrebleu

from llm_evaluate.utils.metric.registry import register

@register("BLEU")
def get_BLEU(responses, answers, extra_info):
    assert len(responses) == len(answers), "The number of translations should be equal to the number of references"
    tgt_lang = extra_info.get("tgt_lang", "en")
    tokenizer = "13a"
    if tgt_lang == "zh":
        tokenizer = "zh"
    elif tgt_lang == "ja":
        tokenizer = "ja-mecab"
    elif tgt_lang == "ko":
        tokenizer = "ko-mecab"
    result = sacrebleu.corpus_bleu(
        responses, 
        [[x] for x in answers], 
        tokenize=tokenizer, 
        force=True
    )
    return result.score


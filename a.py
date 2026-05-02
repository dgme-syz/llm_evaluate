import textwrap
from typing import List, Dict
from datasets import load_dataset


LANG_DICT = {
    "zh": "Chinese (Simplified)",
    "en": "English",
    "fi": "Finnish",
}


def build_recheck_prompt(
    target_lang: str,
    src_text: str,
    pred_text: str,
    use_test_prompt: bool = True,
) -> List[Dict[str, str]]:
    """Build the prompt for the recheck/refinement step."""

    prompts = {
        "zh": textwrap.dedent(
            f"""
            给定源文：'{src_text}'，和它的翻译草稿：'{pred_text}'
            必须先理解源文，然后参考以下标准对草稿进行进一步修改润色

            1. 草稿翻译可能漏翻，请不要遗漏原文的含义
            2. 保证翻译文段读起来流畅，通顺，符合人类表达，可以调整句子的顺序
            3. 请注意仔细理解语境，选择书面语还是口语化表达
            4. 请再检查每个词翻译的意思，是否符合整个语境和现实社会
            5. 请再检查每个句翻译的意思，是否符合整个语境和现实社会
            6. 注意你的润色对象是翻译后的草稿，不是源文
            7. 当你觉得语句读起来困惑的时候，尝试从源文本重新思考
            8. 如果翻译草稿的语言并非中文，请确保你的润色文本为中文
            9. 注意检查，不要遗漏源文的含义，也不要添加补充，也不要尝试在翻译中使用过分的比喻
            10. 可以有思考过程，不要无限思考下去，最终回复中仅输出你润色后的内容

            请返回你最后的润色翻译文本，不要输出多余内容。
            """
        ).strip(),

        "en": textwrap.dedent(
            f"""
            Given the source text: '{src_text}' and its draft translation: '{pred_text}', please refine and polish the draft translation according to the following guidelines:

            1. The draft translation may have omissions — do not leave out any meaning from the source text.
            2. Ensure the translation reads smoothly and naturally, consistent with human expression; you may adjust sentence order if needed.
            3. Carefully understand the context, and decide whether to use formal or colloquial language.
            4. Recheck the meaning of each word to ensure it fits the overall context and real-world usage.
            5. Recheck the meaning of each sentence to ensure it fits the overall context and real-world usage.
            6. Note that your task is to polish the translated draft, not the source text.
            7. When a sentence feels confusing, reconsider it from the perspective of the source text.
            8. If the draft translation is not in English, make sure your refined version is in English.
            9. Do not include any extra reasoning or commentary — only output your polished translation.

            Please return only the final refined translation text, without any additional content.
            """
        ).strip(),

        "fi": textwrap.dedent(
            f"""
            Kun sinulle annetaan lähdeteksti: '{src_text}' ja sen käännösluonnos: '{pred_text}',
            sinun tulee ensin ymmärtää lähdeteksti ja sen jälkeen muokata ja viimeistellä luonnosta
            seuraavien periaatteiden mukaisesti.

            1. Käännösluonnoksessa saattaa olla puutteita; älä jätä pois mitään alkutekstin merkitystä.
            2. Varmista, että käännetty teksti on sujuvaa, luonnollista ja ihmisen tapaista.
            3. Kiinnitä huomiota kontekstiin ja valitse sopiva tyyli.
            4. Tarkista jokaisen sanan merkitys.
            5. Tarkista jokaisen virkkeen merkitys.
            6. Muista, että kohde on käännösluonnos.
            7. Jos jokin kohta tuntuu epäselvältä, palaa lähdetekstiin.
            8. Jos luonnos ei ole kiinaksi, varmista että lopputulos on kiinaksi.
            9. Älä lisää ylimääräistä sisältöä.
            10. Lopullisessa vastauksessa tulee olla vain viimeistelty teksti.

            Palauta vain viimeistelty käännös.
            """
        ).strip(),

        "test": textwrap.dedent(
            f"""
            Given the source text:

            {src_text}

            Improve the following draft {LANG_DICT.get(target_lang, target_lang)} translation into a high-quality {LANG_DICT.get(target_lang, target_lang)} version, without explanations:

            {pred_text}
            """
        ).strip(),
    }

    lang_key = next((k for k in prompts if k in target_lang), None)
    if lang_key is None:
        raise ValueError(f"Unsupported target language: {target_lang}")

    if use_test_prompt:
        lang_key = "test"

    return [{"role": "user", "content": prompts[lang_key]}]


def response_len_ok(example, max_words: int = 400):
    resp = example["response"][0]
    return len(resp.split()) <= max_words


def rebuild_prompt(example):
    src_text = example["extra_info"]["src"]
    pred_text = example["response"][0]
    tgt_lang = example["extra_info"]["tgt_lang"]

    example.pop("response")

    example["prompt"] = build_recheck_prompt(
        target_lang=tgt_lang,
        src_text=src_text,
        pred_text=pred_text,
    )
    example["data_source"] = "wmt17-20"
    example["last_response"] = pred_text
    return example


if __name__ == "__main__":
    dataset = load_dataset(
        "json",
        data_files="train_en2fi.jsonl",
        split="train",
    )
    dataset = dataset.filter(
        response_len_ok,
        desc="Filtering examples with long responses (>4000 words)",
    )

    print(len(dataset))
    dataset = dataset.map(
        rebuild_prompt,
        desc="Rebuilding recheck prompts",
    )

    dataset.to_parquet("Qwen3-4B_en-fi_FI.parquet")
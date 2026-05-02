import re
import textwrap

from .registry import register
from .abstract import EvalFunc

LANG_DICT = {
    "en": "English", "zh": "Chinese (Simplified)", "hu": "Hungarian", "es": "Spanish", "fr": "French", "de": "German", "ru": "Russian", "ja": "Japanese", "th": "Thai", "sw": "Swahili", "bn": "Bengali", "te": "Telugu", "ar": "Arabic", "ko": "Korean", "vi": "Vietnamese", "cs": "Czech", "sr": "Cyrillic Serbian"
}

LANG_DICT.update({
    "ha": "Hausa",
    "om": "Oromo",
    "so": "Somali",
    "am": "Amharic",
    "he": "Hebrew",
    "mt": "Maltese",
    "km": "Khmer",
    "jv": "Javanese",
    "id": "Indonesian",
    "ms": "Malay",
    "mi": "Maori",
    "ceb": "Cebuano",
    "tl": "Tagalog",
    "kn": "Kannada",
    "ml": "Malayalam",
    "ta": "Tamil",
    "hy": "Armenian",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "be": "Belarusian",
    "bg": "Bulgarian",
    "bs": "Bosnian",
    "hr": "Croatian",
    "mk": "Macedonian",
    "pl": "Polish",
    "sk": "Slovak",
    "sl": "Slovenian",
    "uk": "Ukrainian",
    "cy": "Welsh",
    "ga": "Irish",
    "is": "Icelandic",
    "sv": "Swedish",
    "da": "Danish",
    "no": "Norwegian",
    "af": "Afrikaans",
    "lb": "Luxembourgish",
    "nl": "Dutch",
    "el": "Greek",
    "as": "Assamese",
    "gu": "Gujarati",
    "hi": "Hindi",
    "mr": "Marathi",
    "ne": "Nepali",
    "or": "Odia",
    "pa": "Punjabi",
    "sd": "Sindhi",
    "ur": "Urdu",
    "fa": "Persian",
    "ku": "Kurdish",
    "ps": "Pashto",
    "tg": "Tajik",
    "ast": "Asturian",
    "ca": "Catalan",
    "gl": "Galician",
    "it": "Italian",
    "oc": "Occitan",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ka": "Georgian",
    "lo": "Lao",
    "mn": "Mongolian",
    "wo": "Wolof",
    "ln": "Lingala",
    "ns": "Northern Sotho",
    "lg": "Luganda",
    "ny": "Nyanja",
    "sn": "Shona",
    "umb": "Umbundu",
    "xh": "Xhosa",
    "yo": "Yoruba",
    "zu": "Zulu",
    "ig": "Igbo",
    "kam": "Kamba",
    "ff": "Fulani",
    "luo": "Dholuo",
    "kea": "Kabuverdianu",
    "zhtrad": "Traditional Chinese",
    "my": "Burmese",
    "uz": "Uzbek",
    "kk": "Kazakh",
    "ky": "Kyrgyz",
    "az": "Azerbaijani",
    "tr": "Turkish",
    "et": "Estonian",
    "fi": "Finnish",
})

def recheck_prompt(target_lang: str, src_text: str, pred_text: str, simple: bool = True):
    """Builds the prompt for the recheck/refinement step."""
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
            2. Varmista, että käännetty teksti on sujuvaa, luonnollista ja ihmisen tapaista; voit tarvittaessa
                muuttaa virkkeen sanajärjestystä.
            3. Kiinnitä erityistä huomiota kontekstiin ja valitse tilanteeseen sopiva tyyli, olipa se sitten
                kirjakielinen tai puhekielisempi.
            4. Tarkista jokaisen sanan merkitys ja varmista, että se sopii sekä lauseyhteyteen että todelliseen maailmaan.
            5. Tarkista jokaisen virkkeen merkitys ja varmista, että se vastaa koko kontekstia ja todellisuutta.
            6. Muista, että viimeisteltävä kohde on käännösluonnos, ei alkuperäinen lähdeteksti.
            7. Jos jokin kohta tuntuu epäselvältä, palaa lähdetekstiin ja mieti sitä uudelleen.
            8. Jos käännösluonnos ei ole kiinaksi, varmista että viimeistelty tekstisi on kiinankielinen.
            9. Huolehdi siitä, ettet jätä pois alkutekstin merkityksiä, älä lisää ylimääräistä sisältöä
                äläkä käytä ylenpalttisia vertauksia käännöksessä.
            10. Voit näyttää ajatteluprosessin, mutta älä jatka sitä loputtomiin; lopullisessa vastauksessa
                tulee olla vain viimeistelemäsi teksti.

            Palauta lopuksi vain viimeistelty käännöksesi, älä mitään ylimääräistä.
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

    lang_key = next((key for key in prompts if key in target_lang), None)
    if lang_key is None:
        raise ValueError(f"Unsupported target language: {target_lang}. Supported: {list(prompts.keys())}")

    if simple:
        lang_key = "test"

    return [{"role": "user", "content": prompts[lang_key]}]


def process_recheck_output(output: str, raw: str) -> str:
    if not (isinstance(output, str) and isinstance(raw, str)):
        raise ValueError(f"{output} is not a string, or {raw} is not a string") 
    start_tag = "<text>"
    end_tag = "</text>"

    start_idx = output.rfind(start_tag)
    end_idx = output.rfind(end_tag)

    if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
        return output[start_idx + len(start_tag):end_idx].strip()
    else:
        think_pattern = r"<think>.*?</think>"
        output = re.sub(think_pattern, "", output, flags=re.DOTALL).strip()
        return output if len(output) else raw

@register("recheck")
class ReCheckEvalFunc(EvalFunc):
    def __init__(self, llm, **kwargs):
        super().__init__(llm)
        self.check_num = kwargs.get("check_num", 5)
        self.use_thinking = kwargs.get("use_thinking", False)
        self.use_thinking_prompts = kwargs.get("use_thinking_prompts", False)

        print(
            "ReCheckEvalFunc initialized with:\n"
            f"  check_num = {self.check_num}\n"
            f"  use_thinking = {self.use_thinking}\n"
            f"  use_thinking_prompts = {self.use_thinking_prompts}\n"
        )

    def evaluate(self, examples):
        """
        Run batched evaluation with optional rechecking.

        Args:
            examples (dict): Contains:
                - "prompt": list[str], input prompts.
                - "response": (optional) list[list[str]], model outputs.
                - "extra_info": list[dict], must include 'tgt_lang' and 'src'.

        Returns:
            dict: {
                "response": list[list[str]],       # all rechecked results
                "raw_response": list[list[str]],   # raw outputs before processing
            }
        """
        batched_prompts = examples["prompt"]
        current_pred_list = examples.get("response", None)
        batched_responses_raw = None
        if current_pred_list is None:
            current_pred_list = self.llm.generate(batched_prompts)
            batched_responses_raw = [pred.copy() for pred in current_pred_list]
            current_pred_list = [[process_recheck_output(x[0], x[0])] for x in current_pred_list]
        else:
            current_pred_list = [[x[0]] for x in current_pred_list]

        if not isinstance(examples.get("extra_info"), list):
            raise ValueError("Expected 'extra_info' to be a list for batched evaluation.")

        lang = examples["extra_info"][0].get("tgt_lang")
        assert isinstance(lang, str), "'tgt_lang' must be a string."
        src_list = [x["src"] for x in examples["extra_info"]]

        batched_responses = [pred.copy() for pred in current_pred_list]
        if batched_responses_raw is None:
            batched_responses_raw = [pred.copy() for pred in current_pred_list]

        if self.use_thinking:
            original_thinking = self.llm.get_tokenize_args().get("enable_thinking", False)
            self.llm.update_tokenize_args({"enable_thinking": True})

        for _ in range(self.check_num):
            new_prompts = [
                recheck_prompt(lang, src, preds[0])
                for src, preds in zip(src_list, current_pred_list)
            ]
            new_predictions = self.llm.generate(new_prompts)
            processed_predictions = [
                [process_recheck_output(new, old) for new, old in zip(new_batch, old_batch)]
                for new_batch, old_batch in zip(new_predictions, current_pred_list)
            ]

            assert len(processed_predictions) == len(batched_responses)
            for i, (proc, raw) in enumerate(zip(processed_predictions, new_predictions)):
                batched_responses[i].extend(proc)
                batched_responses_raw[i].extend(raw)

            current_pred_list = processed_predictions

        if self.use_thinking:
            self.llm.update_tokenize_args({"enable_thinking": original_thinking})

        return {
            "response": batched_responses,
            "raw_response": batched_responses_raw,
        }
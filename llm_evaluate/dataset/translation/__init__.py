from typing import Callable, Any
from collections import OrderedDict

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

# Supported language pairs for evaluation
LANGUAGE_PAIRS = (
    "en-ar_EG", "en-ar_SA", "en-bg_BG", "en-bn_IN", "en-ca_ES", "en-cs_CZ",
    "en-da_DK", "en-de_DE", "en-el_GR", "en-es_MX", "en-et_EE", "en-fa_IR",
    "en-fi_FI", "en-fil_PH", "en-fr_CA", "en-fr_FR", "en-gu_IN", "en-he_IL",
    "en-hi_IN", "en-hr_HR", "en-hu_HU", "en-id_ID", "en-is_IS", "en-it_IT",
    "en-ja_JP", "en-kn_IN", "en-ko_KR", "en-lt_LT", "en-lv_LV", "en-ml_IN",
    "en-mr_IN", "en-nl_NL", "en-no_NO", "en-pa_IN", "en-pl_PL", "en-pt_BR",
    "en-pt_PT", "en-ro_RO", "en-ru_RU", "en-sk_SK", "en-sl_SI", "en-sr_RS",
    "en-sv_SE", "en-sw_KE", "en-sw_TZ", "en-ta_IN", "en-te_IN", "en-th_TH",
    "en-tr_TR", "en-uk_UA", "en-ur_PK", "en-vi_VN", "en-zh_CN", "en-zh_TW",
    "en-zu_ZA",
)

# Mapping from language codes to readable language names
LANGUAGE_BY_CODE = {
    "en": "English",
    "ar_EG": "Arabic (Egypt)",
    "ar_SA": "Arabic (Saudi Arabia)",
    "bg_BG": "Bulgarian",
    "bn_BD": "Bengali (Bangladesh)",
    "bn_IN": "Bengali (India)",
    "ca_ES": "Catalan",
    "cs_CZ": "Czech",
    "da_DK": "Danish",
    "de_DE": "German",
    "el_GR": "Greek",
    "es_MX": "Spanish (Mexico)",
    "et_EE": "Estonian",
    "fa_IR": "Farsi",
    "fi_FI": "Finnish",
    "fil_PH": "Filipino",
    "fr_CA": "French (Canada)",
    "fr_FR": "French (France)",
    "gu_IN": "Gujarati",
    "he_IL": "Hebrew",
    "hi_IN": "Hindi",
    "hr_HR": "Croatian",
    "hu_HU": "Hungarian",
    "id_ID": "Indonesian",
    "is_IS": "Icelandic",
    "it_IT": "Italian",
    "ja_JP": "Japanese",
    "kn_IN": "Kannada",
    "ko_KR": "Korean",
    "lt_LT": "Lithuanian",
    "lv_LV": "Latvian",
    "ml_IN": "Malayalam",
    "mr_IN": "Marathi",
    "nl_NL": "Dutch",
    "no_NO": "Norwegian",
    "pa_IN": "Punjabi",
    "pl_PL": "Polish",
    "pt_BR": "Portuguese (Brazil)",
    "pt_PT": "Portuguese (Portugal)",
    "ro_RO": "Romanian",
    "ru_RU": "Russian",
    "sk_SK": "Slovak",
    "sl_SI": "Slovenian",
    "sr_RS": "Serbian",
    "sv_SE": "Swedish",
    "sw_KE": "Swahili (Kenya)",
    "sw_TZ": "Swahili (Tanzania)",
    "ta_IN": "Tamil",
    "te_IN": "Telugu",
    "th_TH": "Thai",
    "tr_TR": "Turkish",
    "uk_UA": "Ukrainian",
    "ur_PK": "Urdu",
    "vi_VN": "Vietnamese",
    "zh_CN": "Chinese (Simplified)",
    "zh_TW": "Chinese (Traditional)",
    "zu_ZA": "Zulu",
}

ppo_lang_dict = {
    "Arabic": "ar",
    "Czech": "cs",
    "Danish": "da",
    "German": "de",
    "English": "en",
    "Spanish": "es",
    "Finnish": "fi",
    "French": "fr",
    "Croatian": "hr",
    "Hungarian": "hu",
    "Indonesian": "id",
    "Italian": "it",
    "Japanese": "ja",
    "Korean": "ko",
    "Malay": "ms",
    "Norwegian Bokmal": "nb",
    "Norwegian": "no",
    "Polish": "pl",
    "Portuguese": "pt",
    "Romanian": "ro",
    "Russian": "ru",
    "Swedish": "sv",
    "Thai": "th",
    "Turkish": "tr",
    "Ukrainian": "uk",
    "Vietnamese": "vi",
    "Chinese (Simplified)": "zh",
}



def qwen_chat_input(src_lang_name, tgt_lang_name, src_text):
    user_input = (
        f"Translate the following text into {tgt_lang_name} "
        f"without additional explanations:\n\n"
        f"{src_text}\n\n"
    )

    return [{"role": "user", "content": user_input}]

def qwen_think_input(src_lang_name, tgt_lang_name, src_text):
    user_input = (
        f"This is a math problem, we define a function f: {src_lang_name} -> {tgt_lang_name}, "
        f"f converts a English text to Chinese text and works as a expert translator, "
        f"use 'rethinking' and 'wait' to help you check the structure of the whole sentence, "
        f"keep in mind that some word can have other meanings, "
        f"you should consider all of them, consider all the cases, this is a complex math problem, "
        f"not an easy QA.\n\n"
        f"now x = {src_text}.\n\n"
        f"please give me f(x), think step by step and ensure your answer fit the real world, "
        f"translating directly is not always a good idea.Make your answer is a fluent senetece, "
        f"you can slightly change its structure, return your f(x)'s translation, without any extra contents."
    )

    return [
        {"role": "user", "content": user_input},
    ]

def qwen_mt_input(src_lang_name, tgt_lang_name, src_text):
    return [{"role": "user", "content": src_text}]

def hunyuan_chat_input(src_lang_name, tgt_lang_name, src_text):
    # reference: https://arxiv.org/pdf/2509.05209
    if src_lang_name != "Chinese (Simplified)":
        # XX -> XX
        user_input = (
            f"Translate the following segment into {tgt_lang_name}, without additional explanation\n\n"
            f"{src_text}"
        )
    else:
        # ZH -> XX
        user_input = (
            f"把下面的文本翻译成{tgt_lang_name}，不要额外解释。\n\n"
            f"{src_text}"
        )
    return [
        {"role": "user", "content": user_input}
    ]

def seed_x_ppo_input(src_lang_name, tgt_lang_name, src_text):
    # reference: https://arxiv.org/pdf/2507.13618
    
    if tgt_lang_name in ppo_lang_dict:
        tag_name = ppo_lang_dict[tgt_lang_name]
    else:
        raise ValueError(f"Unsupported target language {tgt_lang_name} for Seed-X PPO input. We cannot find it in {ppo_lang_dict}.")
    user_input = (
        f"Translate the following {src_lang_name} sentence into {tgt_lang_name}:\n{src_text} <{tag_name}>"
    )
    return user_input


PROMPT_TEMPLATE = OrderedDict[str, Callable](
    [
        ("qwen_chat_mt", qwen_chat_input),
        ("qwen_think_mt", qwen_think_input),
        ("qwen_mt_turbo", qwen_mt_input),
        ("hunyuan_mt_chat", hunyuan_chat_input),
        ("seed_x_mt", seed_x_ppo_input)
    ]
)

def get_prompt_template(template_name: str) -> Callable[[str, str, str], Any]:
    """
    Retrieve the corresponding prompt template function by name.

    Args:
        template_name (str): The name of the template.

    Returns:
        Callable: The function corresponding to the specified template name.
            Parameters:
            src_lang_name (str): The source language name.
            tgt_lang_name (str): The target language name.
            src_text (str): The source text to be processed.

    Raises:
        ValueError: If the template name does not exist in the `PROMPT_TEMPLATE` dictionary.
    """
    if template_name not in PROMPT_TEMPLATE:
        raise ValueError(f"Template {template_name} not found in prompt templates. Please check the name and try again.")

    return PROMPT_TEMPLATE[template_name]

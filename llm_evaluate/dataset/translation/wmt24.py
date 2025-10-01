from llm_evaluate.dataset import EvalDataset, register

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


@register("wmt24")
class wmt24(EvalDataset):
    """
    Dataset class for WMT24 translation evaluation.

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
        Initialize the WMT24 dataset wrapper.

        Args:
            data_dir (str): Directory containing the dataset.
            subset_name (str, optional): Language pair, e.g., 'en-zh_CN'.
            split (str, optional): Dataset split to load ('train', 'test', etc.).
            builder (callable, optional): Custom dataset builder.
        """
        super().__init__(data_dir, subset_name, split, builder)

        # Extract source and target languages from subset_name
        src_code, tgt_code = subset_name.strip().split("-")
        self.src_lang = LANGUAGE_BY_CODE.get(src_code, src_code)
        self.tgt_lang = LANGUAGE_BY_CODE.get(tgt_code, tgt_code)

    def convert_item(self, examples, **kwargs):
        """
        Convert a single dataset example into structured format for LLM evaluation.

        Args:
            examples (dict): A single example from the dataset with 'source' and 'target' keys.

        Returns:
            dict: Structured item with prompt, ability, reward model, and extra info.
        """
        src_text = examples["source"]
        tgt_text = examples["target"]

        return {
            "data_source": "wmt24",
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

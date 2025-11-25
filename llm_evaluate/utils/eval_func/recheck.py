import re

from .registry import register
from .abstract import EvalFunc

def recheck_prompt(target_lang: str, src_text: str, pred_text: str, thinking: bool = False):

    if "zh" in target_lang:
        user_input = (
            f"给定源文：'{src_text}'，和它的翻译草稿：'{pred_text}'"
            f"必须先理解源文，然后参考以下标准对草稿进行进一步修改润色\n\n"
            f"1. 草稿翻译可能漏翻，请不要遗漏原文的含义\n\n"
            f"2. 保证翻译文段读起来流畅，通顺，符合人类表达，可以调整句子的顺序\n\n"
            f"3. 请注意仔细理解语境，选择书面语还是口语化表达\n\n"
            f"4. 请再检查每个词翻译的意思，是否符合整个语境和现实社会\n\n"
            f"5. 请再检查每个句翻译的意思，是否符合整个语境和现实社会\n\n"
            f"6. 注意你的润色对象是翻译后的草稿，不是源文\n\n"
            f"7. 当你觉得语句读起来困惑的时候，尝试从源文本重新思考\n\n"
            f"8. 如果翻译草稿的语言并非中文，请确保你的润色文本为中文\n\n"
            f"9. 注意检查，不要遗漏源文的含义，也不要添加补充，也不要尝试在翻译中使用过分的比喻\n\n"
        )

        if not thinking:
            user_input += (
                f"10. 可以有思考过程，但是最终回复中仅输出你润色后的内容\n\n"
                f"请返回你最后的润色翻译文本，不要输出多余内容。"
            )
        else:
            user_input += (
                f"10. 可以有思考过程，然后输出你润色后的内容\n\n"
                f"请将你最后的润色翻译文本放置在 <text> 和 </text> 之间，请不要输出多余内容。"
            )
    elif "en" in target_lang:
        user_input = (
            f"Given the source text: '{src_text}' and its draft translation: '{pred_text}', "
            f"please refine and polish the draft translation according to the following guidelines:\n\n"
            f"1. The draft translation may have omissions — do not leave out any meaning from the source text.\n\n"
            f"2. Ensure the translation reads smoothly and naturally, consistent with human expression; you may adjust sentence order if needed.\n\n"
            f"3. Carefully understand the context, and decide whether to use formal or colloquial language.\n\n"
            f"4. Recheck the meaning of each word to ensure it fits the overall context and real-world usage.\n\n"
            f"5. Recheck the meaning of each sentence to ensure it fits the overall context and real-world usage.\n\n"
            f"6. Note that your task is to polish the translated draft, not the source text.\n\n"
            f"7. When a sentence feels confusing, reconsider it from the perspective of the source text.\n\n"
            f"8. If the draft translation is not in English, make sure your refined version is in English.\n\n"
            f"9. Do not include any extra reasoning or commentary — only output your polished translation.\n\n"
            f"Please return only the final refined translation text, without any additional content."
        )
    else:
        raise ValueError(
            f"Now we just support lang=['Chinese', 'English'], we get {target_lang}"
        )

    return [{"role": "user", "content": user_input}]


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
                recheck_prompt(lang, src, preds[0], thinking=self.use_thinking_prompts)
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
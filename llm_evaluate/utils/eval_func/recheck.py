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
                f"10. 不要输出多余思考内容，仅输出你润色后的内容\n\n"
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
        if "think>" not in output:
            return output
        return raw

@register("recheck")
class ReCheckEvalFunc(EvalFunc):

    def __init__(self, llm, **kwargs):
        super().__init__(llm)
        self.check_num = kwargs.get("check_num", 5)
        self.use_thinking = kwargs.get("use_thinking", False)

    def evaluate(self, examples):
        """
        Batch inference function:
        - Input: examples containing "prompt"
        - Output: updated prompts with assistant responses appended
        """
        batched_prompts = examples["prompt"]
        current_pred_list = self.llm.generate(batched_prompts)

        if isinstance(examples["extra_info"], list):
            lang = examples["extra_info"][0]["tgt_lang"]
            assert isinstance(lang, str)
            src_list = [ x["src"] for x in examples["extra_info"] ]
        else:
            raise ValueError("should not reach here. We just support batched evaluate here.")

        assert len(current_pred_list[0]) == 1, "If you want to use ReCheckEvalFunc, num_generate should be 1."
        current_pred_list = [x[0] for x in current_pred_list]
        batched_responses = [ [x] for x in current_pred_list]

        if self.use_thinking:
            bk = self.llm.tokenize_args["enable_thinking"]
            self.llm.tokenize_args["enable_thinking"] = True

        for _ in range(self.check_num):
            new_batched_prompts = [
                recheck_prompt(lang, src, pred, thinking=self.use_thinking) for src, pred in zip(src_list, current_pred_list)
            ]
            _current_pred_list = self.llm.generate(new_batched_prompts)
            current_pred_list = [process_recheck_output(x[0], y) for x, y in zip(_current_pred_list, current_pred_list)]
            assert len(current_pred_list) == len(batched_responses)
            for i in range(len(batched_responses)):
                batched_responses[i].append(current_pred_list[i])

        if self.use_thinking:
            self.llm.tokenize_args["enable_thinking"] = bk

        return {"response": batched_responses}
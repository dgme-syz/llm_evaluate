import os

from vllm import SamplingParams
from transformers import AutoTokenizer

from .registry import register
from .abstract import EvalFunc
from llm_evaluate.vllm.utils import build_model

MODEL_HOME = os.environ["MODEL"]
MODEL_NAME = "Qwen3-8B"
MODEL_PATH = os.path.join(MODEL_HOME, MODEL_NAME)

def split_think(response: str, n: int = 4) -> list[str]:
    parts = response.split("\n\n")
    chunk_size = max(1, len(parts) // n)
    return [
        "\n\n".join(parts[: min(i + chunk_size, len(parts))]) + "\n\n"
        for i in range(0, len(parts), chunk_size)
    ]

def control_final_answer(sa: str) -> str:
    import re

    backup = sa
    pattern = re.compile(
        r"(?:^|\n|\r)(.*?)(?=(?:\bthe\s+answer\b|\\boxed|final\s+answer|答案|output\s*:|result\s*:))",
        flags=re.IGNORECASE | re.DOTALL,
    )

    match = pattern.search(sa)
    if match:
        sa = match.group(1).strip()
    else:
        if "\n\n" in sa:
            sa = sa[: sa.rfind("\n\n")]
        elif "\n" in sa:
            sa = sa[: sa.rfind("\n")]

    if not sa.strip():
        sa = backup
    return sa

@register("split")
class SplitEvalFunc(EvalFunc):
    continue_prompt: str = "\nI acknowledge that the previous text is right and cannot be changed. Now continuing the steps from where it left off\n\n"
    def __init__(self, llm):
        super().__init__(llm)
        config = {
            "use_server": False,
            "llm": {
                "model": MODEL_PATH,
                "tokenizer": MODEL_PATH,
                "trust_remote_code": True,
                "dtype": "bfloat16", 
                "gpu_memory_utilization": 0.9,
                "tensor_parallel_size": 1,
                "seed": 42,
            },
            "sample_params": {
                "offline": {
                    "temperature": 0.7, 
                    "top_p": 0.8, 
                    "top_k": 20,
                    "repetition_penalty": 1.05,
                    "max_tokens": 8192
                }
            },
            "server": {}
        }

        os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
        self.thick_llm = build_model(config)
        self.n_split = 6
        self.tokenizer = AutoTokenizer.from_pretrained(config.get("llm").get("tokenizer"))

    def evaluate(self, examples):
        """
        Batch inference function:
        - Input: examples containing "prompt"
        - Output: updated prompts with assistant responses appended
        """
        batched_prompts = examples["prompt"]
        batched_responses = self.llm.generate(batched_prompts)
        self.llm.llm.sleep(level=1)
        self.thick_llm.llm.wake_up()
        new_prompts = []
        re_generation_nums = []
        history = []
        for qur, ans in zip(batched_prompts, batched_responses):
            qur = qur[:qur.rfind("\n\n")]
            ans = ans[0]
            split_ans = split_think(ans, n=self.n_split)
            re_generation_nums.append(len(split_ans[::-1]))

            for i, sa in enumerate(split_ans[::-1]):
                if i == 0:
                    sa = control_final_answer(sa)
                history += [qur + "\n\nAnswer:\n\n" + sa + self.continue_prompt]
                new_prompts.append(
                    self.tokenizer.apply_chat_template(
                        [
                            {"role": "system", "content": "Continue from the previous reasoning and keep the previous content before fixed. Do not redo any calculation."},
                            {"role": "user", "content": qur}, 
                            {"role": "assistant", "content": sa + self.continue_prompt}
                        ],
                        add_generation_prompt=False,
                        continue_final_message=True,
                        tokenize=False,
                    )
                )

        outputs = self.thick_llm.generate(new_prompts)
        self.thick_llm.llm.sleep(level=1)
        self.llm.llm.wake_up()

        resps = []
        whole_mes = []
        offset = 0
        for num in re_generation_nums:
            group = []
            mes_group = []
            for j in range(num):
                output = outputs[offset + j]
                mes = history[offset + j]
                resp = output[0]
                group.append(resp)
                mes_group.append(mes + resp)
            resps.append(group)
            whole_mes.append(mes_group)
            offset += num

        return {"main_response": batched_responses, "response": resps, "whole_message": whole_mes}
        
    



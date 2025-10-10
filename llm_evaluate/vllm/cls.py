from abc import ABC, abstractmethod
import asyncio
from typing import Any

import httpx
from vllm import LLM, SamplingParams
from openai import AsyncOpenAI
from transformers import AutoTokenizer

class CausalLLM(ABC):
    llm: Any = None  # type: ignore

    @abstractmethod
    def generate(self, prompts, **kwargs):
        pass


class syncCausalLLM(CausalLLM):

    def __init__(self, config) -> None:
        self.llm = LLM(enable_sleep_mode=True, **config["llm"])
        self.tokenizer = AutoTokenizer.from_pretrained(config["llm"]["tokenizer"])
        self.config = config

    def generate(self, prompts) -> list[list[str]]:
        sampling_params = SamplingParams(**self.config["sample_params"]["offline"])

        vllm_inputs = []
        for prompt in prompts:
            if isinstance(prompt, list):
                prompt = self.tokenizer.apply_chat_template(
                    prompt, tokenize=False, add_generation_prompt=True
                )
            elif not isinstance(prompt, str):
                raise ValueError("Prompt must be a string or a list of dict.")
            vllm_inputs.append(prompt)

        resps = self.llm.generate(prompts, sampling_params=sampling_params)

        ret = []
        for r in resps:
            _ret = []
            for i in range(len(r.outputs)):
                _ret += [r.outputs[i].text.strip()]
            ret += [_ret]
        
        return ret


class asyncCausalLLM(CausalLLM):

    def __init__(self, config) -> None:
        ignore_proxy = config["server"].pop("ignore_proxy", False)

        if ignore_proxy:
            self.llm = AsyncOpenAI(
                **config["server"],
                http_client=httpx.AsyncClient(
                    transport=httpx.AsyncHTTPTransport(proxy=None)
                )
            )
        else:
            self.llm = AsyncOpenAI(**config["server"])
        self.config = config

    async def _generate(self, prompts) -> list[str]:
        # [TODO] update to list[list[str]]
        model = self.config["llm"].get("model", None)

        models_list = await self.llm.models.list()
        if not any(model == x.id for x in models_list.data):
            model = models_list.data[0].id
            print(f"Warning: model {self.config['llm'].get('model', None)} not found, "
                  f"using {model} instead.")

        tasks = []
        for prompt in prompts:
            tasks.append(
                self.llm.completions.create(
                    model=model,
                    prompt=prompt,
                    **self.config["sample_params"]["online"]
                )
            )

        results = await asyncio.gather(*tasks)
        return [r.choices[0].text.strip() for r in results]

    def generate(self, prompts) -> list[str]:
        return asyncio.run(self._generate(prompts))
from abc import ABC, abstractmethod
import asyncio
from typing import Any

from vllm import LLM, SamplingParams
from openai import AsyncOpenAI
import httpx


class CausalLLM(ABC):
    llm: Any = None  # type: ignore

    @abstractmethod
    def generate(self, prompts, **kwargs):
        pass


class syncCausalLLM(CausalLLM):

    def __init__(self, config) -> None:
        self.llm = LLM(**config["llm"])
        self.config = config

    def generate(self, prompts) -> list[str]:
        sampling_params = SamplingParams(**self.config["sample_params"]["offline"])
        resps = self.llm.chat(prompts, sampling_params=sampling_params)
        return [r.outputs[0].text.strip() for r in resps]


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
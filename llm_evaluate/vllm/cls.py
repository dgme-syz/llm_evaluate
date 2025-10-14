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


def flatten_with_paths(data, path=()):
    if isinstance(data, list):
        if not data:
            return [(path, [])]
        if all(isinstance(x, (str, dict)) for x in data):
            return [(path, data)]
        res = []
        for i, x in enumerate(data):
            res.extend(flatten_with_paths(x, path + (i,)))
        return res
    else:
        return [(path, data)]


def reconstruct_from_paths(paths, values):
    res = {}
    for (path, value) in zip(paths, values):
        d = res
        for p in path[:-1]:
            d = d.setdefault(p, {})
        d[path[-1]] = value

    def dict_to_list(d):
        if not isinstance(d, dict):
            return d
        if not d:
            return [] 
        max_idx = max(d.keys())
        return [dict_to_list(d.get(i, [])) for i in range(max_idx + 1)]

    return dict_to_list(res)





class syncCausalLLM(CausalLLM):

    def __init__(self, config) -> None:
        self.llm = LLM(enable_sleep_mode=True, **config["llm"])
        self.tokenizer = AutoTokenizer.from_pretrained(config["llm"]["tokenizer"])
        self.config = config

    def generate(self, prompts) -> list[list[str]]:
        sampling_params = SamplingParams(**self.config["sample_params"]["offline"])

        paths, flat = zip(*flatten_with_paths(prompts))
        vllm_inputs = []
        for prompt in flat:
            if isinstance(prompt, list):
                prompt = self.tokenizer.apply_chat_template(
                    prompt, tokenize=False, add_generation_prompt=True
                )
            elif not isinstance(prompt, str):
                raise ValueError("Prompt must be a string or a list of dict.")
            vllm_inputs.append(prompt)

        resps = self.llm.generate(vllm_inputs, sampling_params=sampling_params)

        ret = []
        for r in resps:
            _ret = []
            for i in range(len(r.outputs)):
                _ret += [r.outputs[i].text.strip()]
            ret += [_ret]
        
        return reconstruct_from_paths(paths, ret)


class asyncCausalLLM(CausalLLM):

    def __init__(self, config) -> None:
        server_cfg = dict(config["server"])
        ignore_proxy = server_cfg.pop("ignore_proxy", False)

        print("connect to server with config:", server_cfg)
        if ignore_proxy:
            self.llm = AsyncOpenAI(
                **server_cfg,
                http_client=httpx.AsyncClient(
                    transport=httpx.AsyncHTTPTransport(proxy=None)
                )
            )
        else:
            self.llm = AsyncOpenAI(**server_cfg)
        self.config = config
        self.sampling_params = config["sample_params"]["online"]
        self.model = self.config["llm"].get("model", None)

        if self.model is None:

            models_list = self.llm.models.list()
            if not any(self.model == x.id for x in models_list.data):
                self.model = models_list.data[0].id
                print(f"Warning: model {self.config['llm'].get('model', None)} not found, "
                    f"using {self.model} instead.")

    async def _generate(self, prompts) -> list[str]:
        # [TODO] update to list[list[str]]

        tasks = []
        params = dict(self.sampling_params).copy()
        n = params.pop("n", 1)

        # print(prompts)
        paths, flat = zip(*flatten_with_paths(prompts))
        empty_prompts_idx = [] 

        for i, prompt in enumerate(flat):
            if not isinstance(prompt, (str, list)):
                raise ValueError("Prompt must be a string or a list of dict.")

            if isinstance(prompt, list) and len(prompt) == 0:
                empty_prompts_idx.append(i)
                continue

            for _ in range(n):
                tasks.append(
                    self.llm.chat.completions.create(
                        model=self.model,
                        messages=prompt,
                        **self.config["sample_params"]["online"]
                    )
                )

        results = await asyncio.gather(*tasks)

        for idx in empty_prompts_idx:
            for _ in range(n):
                results.insert(idx * n, [])

        ret = []
        for i in range(0, len(results), n):
            ret += [r.choices[0].message.content.strip() if hasattr(r, "choices") else r for r in results[i:i+n]]
        ret = reconstruct_from_paths(paths, ret)
        print(ret)
        return ret

    def generate(self, prompts) -> list[str]:
        return asyncio.run(self._generate(prompts))
    


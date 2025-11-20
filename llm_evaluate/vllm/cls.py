from __future__ import annotations
from abc import ABC, abstractmethod
import asyncio
from typing import Any, Iterable
import httpx
from vllm import LLM, SamplingParams
from vllm.sampling_params import BeamSearchParams

from openai import AsyncOpenAI
from transformers import AutoTokenizer


# ===== Base Class =====

class CausalLLM(ABC):
    """Abstract base class for all causal LLM implementations."""
    llm: Any = None  # type: ignore

    @abstractmethod
    def generate(self, prompts, **kwargs):
        """Generate responses from the language model."""
        raise NotImplementedError


# ===== Utilities =====

def flatten_with_paths(data, path=()) -> list[tuple[tuple[int, ...], Any]]:
    """Flatten nested lists while keeping index paths."""
    if isinstance(data, list):
        if not data:
            return [(path, [])]
        if all(isinstance(x, dict) for x in data):
            return [(path, data)]
        return [p for i, x in enumerate(data) for p in flatten_with_paths(x, path + (i,))]
    return [(path, data)]


def reconstruct_from_paths(paths, values):
    """Reconstruct nested list structure from flattened paths."""
    res = {}
    for path, value in zip(paths, values):
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


def check_item_valid(item: Any) -> bool:
    """Check if item is valid: a string or a list of dicts."""
    return isinstance(item, str) or (
        isinstance(item, list) and all(isinstance(x, dict) for x in item)
    )


def validate_prompts(prompts: Iterable[Any]) -> tuple[bool, int]:
    """Validate prompt structure and determine list flattening behavior."""
    if all(check_item_valid(x) for x in prompts):
        return True, 1  # Normal case
    if all(isinstance(x, list) for x in prompts):
        if not all(all(check_item_valid(y) for y in x) for x in prompts):
            raise ValueError(
                "Each inner list must contain only valid items "
                "(string or list of dicts)."
            )
        return False, 0  # Multi-generation, no num_generations
    raise ValueError(
        "Each prompt must be a string or a list of dicts. "
        "Alternatively, provide a list of prompts for multi-generation."
    )


# ===== Sync Implementation =====

class SyncCausalLLM(CausalLLM):
    """Synchronous causal LLM using vLLM backend."""

    def __init__(self, config: dict) -> None:
        self.config = config
        print(f"Initializing SyncCausalLLM...\n Got config: {config.llm.vllm}")
        self.llm = LLM(enable_sleep_mode=True, **config.llm.vllm)
        tokenizer_path = config.llm.vllm.get("tokenizer", config.llm.vllm.model)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.tokenize_args = dict(config.get("tokenize_args", {}))

    def generate(self, prompts) -> list[list[str]]:
        beam_serach = False

        if self.config.generate_params.offline.get("use_beam_search", False):
            beam_serach = True
            sampling_params = BeamSearchParams(**self.config.generate_params.offline.beam_search_params)
        else:
            sampling_params = SamplingParams(**self.config.generate_params.offline.sampling_params)

        _, n = validate_prompts(prompts)
        list_remove = n == 0
        if list_remove:
            if hasattr(sampling_params, "n") and sampling_params.n != 1:
                print("Note: Generating multiple responses per prompt.So n must be set to 1.")
            if hasattr(sampling_params, "n"):
                sampling_params.n = 1

        paths, flat = zip(*flatten_with_paths(prompts))
        vllm_inputs, empty_idx = [], []

        for i, prompt in enumerate(flat):
            if isinstance(prompt, list) and not prompt:
                empty_idx.append(i)
                continue
            if isinstance(prompt, list):
                prompt = self.tokenizer.apply_chat_template(
                    prompt, tokenize=False, add_generation_prompt=True, **self.tokenize_args
                )
            elif not isinstance(prompt, str):
                raise ValueError("Prompt must be a string or list of dicts.")
            vllm_inputs.append(prompt)
        if beam_serach == False:
            responses = self.llm.generate(vllm_inputs, sampling_params)
        else:
            responses = self.llm.beam_search(vllm_inputs, sampling_params)

        results = []
        for r in responses:
            texts = [out.text.strip() for out in r.outputs]
            results += texts if list_remove else [texts]

        for idx in empty_idx:
            assert list_remove
            results.insert(idx, [])

        final = reconstruct_from_paths(paths, results)
        return final


# ===== Async Implementation =====

class AsyncCausalLLM(CausalLLM):
    """Asynchronous causal LLM using OpenAI-compatible API."""

    def __init__(self, config: dict) -> None:
        self.config = config
        server_cfg = config["server"]

        print(f"Connecting to server with config: {server_cfg}")
        client_args = {"transport": httpx.AsyncHTTPTransport(proxy=None)} if server_cfg.ignore_proxy else {}
        self.llm = AsyncOpenAI(**server_cfg.openai_args, http_client=httpx.AsyncClient(**client_args))

        self.sampling_params = dict(config.generate_params.online)
        self.n = self.sampling_params.pop("n", 1)
        self.model = server_cfg.get("model")
        self.tokenize_args = dict(config.get("tokenize_args", {})) # dummy

    def merge_data_args(self, new_args: dict) -> None:
        # Sometimes, we need to use args about dataset.
        # Now, our main script hope to use one model to test more datasets.
        # new_args need to be the content of "generate_args"

        self.sampling_params.setdefault("extra_body", {}).update(new_args)
    def update_sampling_params(self, backup: dict) -> None:
        self.sampling_params = backup

    async def _generate(self, prompts) -> list[str]:
        valid_type, _ = validate_prompts(prompts)
        list_remove = not valid_type
        if list_remove:
            if self.n != 1:
                print("Note: Generating multiple responses per prompt.So n must be set to 1.")
            self.n = 1

        paths, flat = zip(*flatten_with_paths(prompts))
        empty_idx, tasks = [], []

        for i, prompt in enumerate(flat):
            if isinstance(prompt, list) and not prompt:
                empty_idx.append(i)
                continue
            if not isinstance(prompt, (str, list)):
                raise ValueError("Prompt must be a string or list of dicts.")
            for _ in range(self.n):
                tasks.append(
                    self.llm.chat.completions.create(
                        model=self.model,
                        messages=prompt,
                        **self.sampling_params,
                        timeout=360,
                    )
                )

        responses = await asyncio.gather(*tasks)

        for idx in empty_idx:
            assert list_remove
            responses.insert(idx, [])

        def warraper_think(rep: str) -> str:
            if not rep:
                return ""
            t = []
            if not rep.startswith("<think>"):
                t.append("<think>")
            t.append(rep)
            if not rep.endswith("</think>"):
                t.append("</think>")
            return "\n".join(t)

        results = []
        # print(dict(responses[0].choices[0]))
        for i in range(0, len(responses), self.n):
            segment = []
            for r in responses[i : i + self.n]:
                if hasattr(r, "choices") and r.choices:
                    msg = r.choices[0].message
                    content = getattr(msg, "reasoning_content",  "").strip()
                    segment.append(warraper_think(content) + "\n\n" + getattr(msg, "content", "").strip())
                else:
                    segment.append(str(r))  # fallback
            results += segment if list_remove else [segment]

        final = reconstruct_from_paths(paths, results)
        return final

    def generate(self, prompts) -> list[str]:
        return asyncio.run(self._generate(prompts))

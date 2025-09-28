from typing import Any

def build_model(config: dict[str, Any]):
    """
    Build and return a CausalLLM instance based on the given configuration.

    Args:
        config (dict): Configuration dictionary with the following keys:
            - use_server (bool): Whether to use online mode. Defaults to True.
            - server (dict): Config for online mode.
            - offline (dict): Config for offline mode.

    Returns:
        CausalLLM: An initialized causal language model instance.

    Raises:
        ValueError: If `use_server` contains an invalid value.
    """
    if config.get("use_server", True):
        from llm_evaluate.vllm.cls import asyncCausalLLM as CausalLLM
    else:
        from llm_evaluate.vllm.cls import syncCausalLLM as CausalLLM

    return CausalLLM(config)

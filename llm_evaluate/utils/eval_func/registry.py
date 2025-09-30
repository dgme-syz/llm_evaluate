from typing import Callable

from .abstract import EvalFunc 

EVAL_REGISTRY: dict[str, EvalFunc] = {}

# register a metric func
def register(name: str) -> Callable[[EvalFunc], EvalFunc]:
    """Decorator to register a metric function with a given name.

    Args:
        name (str): The name to register the eval function under.

    Returns:
        Callable[[Callable], Callable]: A decorator that registers the eval function.
    """
    def decorator(func: Callable) -> Callable:
        if name in EVAL_REGISTRY:
            raise ValueError(f"Eval function '{name}' is already registered.")
        EVAL_REGISTRY[name] = func
        return func
    return decorator

from typing import Callable, Union, List, Dict

def get_eval(name: str) -> EvalFunc:
    """Retrieve one or more registered eval functions by name.

    Args:
        names (str or list of str): The name(s) of the eval function(s) to retrieve.

    Returns:
        dict: A dictionary mapping eval function name to registered eval function.
    """
    if name not in EVAL_REGISTRY:
        raise ValueError(f"Eval function '{name}' is not registered.")
    return EVAL_REGISTRY[name]

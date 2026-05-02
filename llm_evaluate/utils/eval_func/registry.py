from __future__ import annotations

from typing import Callable

from .abstract import EvalFunc

EVAL_REGISTRY: dict[str, EvalFunc] = {}


def register(
    name: str,
    overwrite: bool = False,
) -> Callable[[EvalFunc], EvalFunc]:
    """Decorator to register an eval function under ``name``.

    Args:
        name: Lookup key referenced by ``evaluate.eval_func`` in the master config.
        overwrite: If False (default), raise when ``name`` is already registered.
    """

    def decorator(func: EvalFunc) -> EvalFunc:
        if not overwrite and name in EVAL_REGISTRY:
            raise ValueError(f"Eval function '{name}' is already registered.")
        EVAL_REGISTRY[name] = func
        return func

    return decorator


def get_eval(name: str) -> EvalFunc:
    """Retrieve a registered eval function by name.

    Args:
        name: Name of the eval function to retrieve.

    Returns:
        The registered eval function.

    Raises:
        ValueError: If ``name`` is not registered. The error lists every
            currently registered eval function to aid debugging.
    """
    if name not in EVAL_REGISTRY:
        available = ", ".join(sorted(EVAL_REGISTRY.keys()))
        raise ValueError(
            f"Eval function '{name}' is not registered. "
            f"Available eval functions: [{available}]."
        )
    return EVAL_REGISTRY[name]

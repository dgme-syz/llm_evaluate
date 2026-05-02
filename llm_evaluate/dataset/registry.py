from __future__ import annotations

from typing import Callable

from llm_evaluate.dataset.abstract import EvalDataset

DATASET_REGISTRY: dict[str, type[EvalDataset]] = {}


def register(
    name: str,
    overwrite: bool = False,
) -> Callable[[type[EvalDataset]], type[EvalDataset]]:
    """Decorator to register a dataset class under ``name``.

    Args:
        name: The lookup key used by ``get_dataset`` and the ``name`` field of
            ``config/data/.../*.yaml``.
        overwrite: If False (default), raise when ``name`` already exists. Pass
            True to allow re-registration during interactive development.

    Returns:
        A decorator that registers and returns the dataset class.
    """

    def decorator(cls: type[EvalDataset]) -> type[EvalDataset]:
        if not overwrite and name in DATASET_REGISTRY:
            raise ValueError(f"Dataset '{name}' is already registered.")
        DATASET_REGISTRY[name] = cls
        return cls

    return decorator


def get_dataset(name: str) -> type[EvalDataset]:
    """Retrieve a registered dataset class by name.

    Args:
        name: The name of the dataset class to retrieve.

    Returns:
        The registered dataset class.

    Raises:
        ValueError: If ``name`` is not registered. The error lists every
            currently registered dataset to aid debugging.
    """
    if name not in DATASET_REGISTRY:
        available = ", ".join(sorted(DATASET_REGISTRY.keys()))
        raise ValueError(
            f"Dataset '{name}' is not registered. Available datasets: [{available}]."
        )
    return DATASET_REGISTRY[name]


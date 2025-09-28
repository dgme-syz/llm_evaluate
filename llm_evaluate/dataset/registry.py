from typing import Callable

from llm_evaluate.dataset.abstract import EvalDataset

DATASET_REGISTRY: dict[str, type[EvalDataset]] = {}

def register(name: str) -> Callable[[type[EvalDataset]], type[EvalDataset]]:
    """Decorator to register a dataset class with a given name.

    Args:
        name (str): The name to register the dataset class under.

    Returns:
        Callable[[type[EvalDataset]], type[EvalDataset]]: A decorator that registers the dataset class.
    """
    def decorator(cls: type[EvalDataset]) -> type[EvalDataset]:
        if name in DATASET_REGISTRY:
            raise ValueError(f"Dataset '{name}' is already registered.")
        DATASET_REGISTRY[name] = cls
        return cls
    return decorator


def get_dataset(name: str) -> type[EvalDataset]:
    """Retrieve a registered dataset class by name.

    Args:
        name (str): The name of the dataset class to retrieve

    Returns:
        type[EvalDataset]: The registered dataset class.
    """

    if name not in DATASET_REGISTRY:
        raise ValueError(f"Dataset '{name}' is not registered.")
    return DATASET_REGISTRY[name]


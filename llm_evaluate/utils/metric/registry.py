from typing import Callable, Union, List, Dict, Type
from .abstract import Metric

# Registry to hold metric functions or Metric subclasses
METRIC_REGISTRY: dict[str, Callable | Type[Metric]] = {}


def register(name: str) -> Callable[[Callable | Type[Metric]], Callable | Type[Metric]]:
    """Decorator to register a metric function or Metric subclass with a given name.

    Args:
        name (str): The name to register the metric under.

    Returns:
        Callable: A decorator that registers the metric.
    """
    def decorator(obj: Callable | Type[Metric]) -> Callable | Type[Metric]:
        if name in METRIC_REGISTRY:
            raise ValueError(f"Metric '{name}' is already registered.")
        METRIC_REGISTRY[name] = obj
        return obj
    return decorator


def get_metrics(names: Union[str, List[str]], instantiate: bool = True) -> Dict[str, Metric | Callable]:
    """Retrieve one or more registered metrics by name.

    Args:
        names (str or list of str): The name(s) of the metric(s) to retrieve.
        instantiate (bool, optional): If True, instantiate Metric subclasses before returning.
            Defaults to True.

    Returns:
        dict: A dictionary mapping metric name to metric instance or function.
    """
    if isinstance(names, str):
        names = [names]

    metrics = {}
    for name in names:
        if name not in METRIC_REGISTRY:
            raise ValueError(f"Metric '{name}' is not registered.")
        
        obj = METRIC_REGISTRY[name]
        if instantiate and isinstance(obj, type) and issubclass(obj, Metric):
            metrics[name] = obj()  # instantiate the class
        else:
            metrics[name] = obj

    return metrics

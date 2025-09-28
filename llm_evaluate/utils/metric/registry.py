from typing import Callable

METRIC_REGISTRY: dict[str, Callable] = {}

# register a metric func
def register(name: str) -> Callable[[Callable], Callable]:
    """Decorator to register a metric function with a given name.

    Args:
        name (str): The name to register the metric function under.

    Returns:
        Callable[[Callable], Callable]: A decorator that registers the metric function.
    """
    def decorator(func: Callable) -> Callable:
        if name in METRIC_REGISTRY:
            raise ValueError(f"Metric '{name}' is already registered.")
        METRIC_REGISTRY[name] = func
        return func
    return decorator

from typing import Callable, Union, List, Dict

def get_metrics(names: Union[str, List[str]]) -> Dict[str, Callable]:
    """Retrieve one or more registered metric functions by name.

    Args:
        names (str or list of str): The name(s) of the metric function(s) to retrieve.

    Returns:
        dict: A dictionary mapping metric name to registered metric function.
    """
    if isinstance(names, str):
        names = [names]

    metrics_func = {}
    for name in names:
        if name not in METRIC_REGISTRY:
            raise ValueError(f"Metric '{name}' is not registered.")
        metrics_func[name] = METRIC_REGISTRY[name]

    return metrics_func
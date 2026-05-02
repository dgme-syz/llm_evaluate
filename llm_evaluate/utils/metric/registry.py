from __future__ import annotations

from typing import Callable, Dict, List, Type, Union

from .abstract import Metric

# Registry of metric functions or Metric subclasses keyed by name.
METRIC_REGISTRY: dict[str, Callable | Type[Metric]] = {}


def register(
    name: str,
    overwrite: bool = False,
) -> Callable[[Callable | Type[Metric]], Callable | Type[Metric]]:
    """Decorator to register a metric function or ``Metric`` subclass under ``name``.

    Args:
        name: Lookup key referenced by ``evaluate.metrics`` in the master config.
        overwrite: If False (default), raise when ``name`` is already registered.
    """

    def decorator(obj: Callable | Type[Metric]) -> Callable | Type[Metric]:
        if not overwrite and name in METRIC_REGISTRY:
            raise ValueError(f"Metric '{name}' is already registered.")
        METRIC_REGISTRY[name] = obj
        return obj

    return decorator


def get_metrics(
    names: Union[str, List[str]],
    instantiate: bool = True,
) -> Dict[str, Metric | Callable]:
    """Retrieve one or more registered metrics by name.

    Args:
        names: The metric name (or list of names) to look up.
        instantiate: If True, ``Metric`` subclasses are instantiated before
            being returned. Plain callables are returned as-is.

    Returns:
        Mapping ``{name: metric_obj}`` preserving the requested order.

    Raises:
        ValueError: If any name is not registered. The error lists every
            currently registered metric to aid debugging.
    """
    if isinstance(names, str):
        names = [names]

    metrics: Dict[str, Metric | Callable] = {}
    for name in names:
        if name not in METRIC_REGISTRY:
            available = ", ".join(sorted(METRIC_REGISTRY.keys()))
            raise ValueError(
                f"Metric '{name}' is not registered. Available metrics: [{available}]."
            )

        obj = METRIC_REGISTRY[name]
        if instantiate and isinstance(obj, type) and issubclass(obj, Metric):
            metrics[name] = obj()
        else:
            metrics[name] = obj

    return metrics

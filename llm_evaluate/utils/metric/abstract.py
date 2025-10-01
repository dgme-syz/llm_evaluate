from abc import ABC, abstractmethod
from typing import Sequence, Union


class Metric(ABC):
    """Abstract base class for evaluation metrics."""

    @abstractmethod
    def __call__(
        self,
        responses: Sequence[str],
        references: Sequence[str],
        extra_infos: Sequence[dict] | None = None,
    ) -> Union[float, dict]:
        """
        Compute the evaluation metric.

        Args:
            responses (Sequence[str]): Model output responses.
            references (Sequence[str]): Reference answers.
            extra_infos (Sequence[dict] | None, optional): Additional information
                such as context or metadata. Defaults to None.

        Returns:
            float | dict:
                - float: a single scalar score.
                - dict: a structured output with at least the key "score", e.g.:
                  {
                      "score": float | list[float],
                      "extra_dict": dict
                  }
        """
        raise NotImplementedError

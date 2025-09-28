from __future__ import annotations

from abc import ABC
from typing import Any

from datasets import Dataset, load_dataset
from omegaconf import DictConfig, ListConfig


class EvalDataset(ABC):
    def __init__(
        self,
        data_path: str | list[str] | DictConfig | ListConfig,
        subset_name: str | list[str] | DictConfig | ListConfig | None = None,
        split: str | list[str] | DictConfig | ListConfig = "test",
        builder: str | list[str] | DictConfig | ListConfig | None = None,
    ) -> None:
        """
        Args:
            data_path: Dataset name or local path. Can be a string, list, or Hydra config.
            subset_name: Subset name(s), corresponding to HuggingFace datasets subsets.
            split: Split(s) such as "train", "test", etc.
            builder: Optional dataset builder(s).
        """

        def is_list_like(x):
            return isinstance(x, (list, ListConfig))

        def is_dict_like(x):
            return isinstance(x, (dict, DictConfig))

        params = [data_path, subset_name, split, builder]

        if any(is_list_like(x) for x in params if x is not None):
            length = max(len(x) for x in params if is_list_like(x))

            def ensure_list(x):
                if x is None:
                    return [None] * length
                if is_list_like(x):
                    return list(x)  # ListConfig -> list
                if is_dict_like(x):
                    raise TypeError(
                        f"Dict-like config found where list or str expected: {x}\n"
                        f"Please extract the proper field before passing it."
                    )
                return [x] * length

            data_path = ensure_list(data_path)
            subset_name = ensure_list(subset_name)
            split = ensure_list(split)
            builder = ensure_list(builder)

            if not (len(data_path) == len(subset_name) == len(split) == len(builder)):
                raise ValueError(
                    f"Length mismatch:\n"
                    f"  data_path={len(data_path)}\n"
                    f"  subset_name={len(subset_name)}\n"
                    f"  split={len(split)}\n"
                    f"  builder={len(builder)}"
                )

        print(data_path, subset_name, split, builder)

        # Single-dataset mode
        if isinstance(data_path, str):
            if builder is None:
                self.datasets: Dataset = load_dataset(data_path, subset_name, split=split)
            else:
                self.datasets: Dataset = load_dataset(
                    builder, subset_name, data_files=data_path, split=split
                )
        else:
            # Multi-dataset mode
            if not (isinstance(subset_name, list) and isinstance(split, list)):
                raise ValueError(
                    f"When data_path is a list, subset_name and split must also be lists.\n"
                    f"types: {type(subset_name)}, {type(split)}"
                )

            if all(b is None for b in builder):
                self.datasets: list[Dataset] = [
                    load_dataset(da, su, split=sp)
                    for da, su, sp in zip(data_path, subset_name, split)
                ]
            else:
                self.datasets: list[Dataset] = [
                    load_dataset(bu, su, data_files=da, split=sp)
                    for da, su, sp, bu in zip(data_path, subset_name, split, builder)
                ]

    def convert_item(
        self, examples: dict[str, Any] | list[dict[str, Any]], **kwargs
    ) -> dict[str, Any]:
        """
        Single-sample version.
        Should be overridden by subclasses. If not, raises NotImplementedError.
        """
        d = (
            next(iter(self.datasets))
            if not isinstance(self.datasets, list)
            else [next(iter(ds)) for ds in self.datasets]
        )
        raise NotImplementedError(
            "convert_item() was not implemented in subclass.\n"
            f"Example data to design your function:\n{repr(d)}"
        )

    def convert_items(
        self, examples: list[dict[str, Any]], **kwargs
    ) -> dict[str, Any]:
        """
        Batched version.
        Should be overridden by subclasses if batched=True is used.
        """
        d = (
            self.datasets[:2]
            if not isinstance(self.datasets, list)
            else [x[:2] for x in self.datasets]
        )
        raise NotImplementedError(
            "convert_items() was not implemented in subclass.\n"
            f"Example data to design your function:\n{repr(d)}"
        )

    def _convert_item(
        self, example: dict[str, Any] | dict[str, list], idx: int | list[int]
    ) -> dict[str, Any]:
        """
        Internal helper for multi-dataset mode.

        - If `idx` is an int (single sample), it merges the sample from all datasets
          at the same index and applies `convert_item`.
        - If `idx` is a list (batched mode), it merges the batch of samples from all datasets
          and applies `convert_items`.
        """
        examples = [example]
        if isinstance(self.datasets, list):
            for data in self.datasets[1:]:
                examples.append(data[idx])

        if isinstance(idx, int):
            return self.convert_item(examples if len(examples) > 1 else examples[0])
        elif isinstance(idx, list):
            return self.convert_items(examples)
        else:
            raise ValueError("idx must be int or list[int]")

    def map(
        self,
        num_proc: int | None = None,
        batched: bool = False,
        batch_size: int = 100,
        num_examines: int = 1,
    ) -> Dataset:
        """
        Apply `convert_item` (single mode) or `convert_items` (batched mode)
        to the dataset(s) and return a new Dataset.

        Args:
            num_proc: Number of processes for multiprocessing.
            batched: Whether to process in batch mode.
            batch_size: Batch size used when `batched=True`.
            num_examines: Number of samples to preview after transformation.

        Returns:
            A new Dataset with transformed items.
        """
        if isinstance(self.datasets, Dataset):
            remove_columns = self.datasets.column_names
            obj = self.datasets
        else:
            remove_columns = self.datasets[0].column_names
            obj = self.datasets[0]

        new_obj = obj.map(
            self._convert_item,
            with_indices=True,
            drop_last_batch=False,
            remove_columns=remove_columns,
            num_proc=num_proc,
            batched=batched,
            batch_size=batch_size,
        )

        print("[Done] Map Process.")
        if num_examines > 0:
            print(
                f"Preview {num_examines} samples:\n{repr(new_obj[:num_examines])}"
            )
        return new_obj

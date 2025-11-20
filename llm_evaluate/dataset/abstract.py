from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
import os
import re
import json

from datasets import Dataset, load_dataset
from omegaconf import DictConfig, ListConfig

class EvalDataset(ABC):
    @staticmethod
    def normalize_jsonl(input_file: str, output_file: str):
        """Convert multi-line JSONL to single-line JSONL format"""

        dirname = os.path.dirname(output_file)
        filename = os.path.basename(output_file)

        filename_safe = re.sub(r'[^a-zA-Z0-9._-]', "_", filename)
        filename_safe = re.sub(r"_+", "_", filename_safe).strip("_")

        output_file = os.path.join(dirname, filename_safe)

        with open(input_file, "r", encoding="utf-8") as fin, \
            open(output_file, "w", encoding="utf-8") as fout:

            buffer = []
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                buffer.append(line)
                try:
                    obj = json.loads("".join(buffer))
                    fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    buffer = []
                except json.JSONDecodeError:
                    continue

            if buffer:
                try:
                    obj = json.loads("".join(buffer))
                    fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                except json.JSONDecodeError:
                    print(f"[Warning] Incomplete JSON discarded: {''.join(buffer)[:100]}...")

        return output_file

    def __init__(
        self,
        data_path: str | list[str] | DictConfig | ListConfig,
        subset_name: str | list[str] | DictConfig | ListConfig | None = None,
        split: str | list[str] | DictConfig | ListConfig = "test",
        builder: str | list[str] | DictConfig | ListConfig | None = None,
        **kwargs
    ) -> None:
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
                    return list(x)
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
            # JSON file processing
            if builder == "json" and os.path.isfile(data_path):
                normalized_file = data_path + ".singleline"
                data_path = self.normalize_jsonl(data_path, normalized_file)

            if builder is None:
                self.datasets: Dataset = load_dataset(data_path, subset_name, split=split)
            else:
                if not os.path.isfile(data_path):
                    raise ValueError(f"Data path must be a file when builder is specified: {data_path}")
                print(f"Open {data_path}")
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

            # JSON file processing for each dataset
            for i, (dp, bu) in enumerate(zip(data_path, builder)):
                if bu == "json" and isinstance(dp, str) and os.path.isfile(dp):
                    normalized_file = dp + ".singleline"
                    self.normalize_jsonl(dp, normalized_file)
                    data_path[i] = normalized_file

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

    @abstractmethod
    def convert_item(
        self, examples: dict[str, Any] | list[dict[str, Any]], **kwargs
    ) -> dict[str, Any]:
        """Subclasses must implement this method to convert single data items"""
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
        """Subclasses must implement this method to convert batched data items"""
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
        """Internal method to handle both single and batched conversion"""
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
        batch_size: int | None = None,
        num_examines: int = 1,
        save_columns: bool = False,
    ) -> Dataset:
        """Apply conversion to the entire dataset"""
        if isinstance(self.datasets, Dataset):
            remove_columns = self.datasets.column_names
            obj = self.datasets
        else:
            remove_columns = self.datasets[0].column_names
            obj = self.datasets[0]

        if save_columns:
            remove_columns = None

        print("[Start] Map Process...")
        new_obj = obj.map(
            self._convert_item,
            with_indices=True,
            drop_last_batch=False,
            remove_columns=remove_columns,
            num_proc=num_proc,
            batched=batched,
            batch_size=batch_size,
            load_from_cache_file=False
        )
        print("[Done] Map Process.")
        if num_examines > 0:
            print(
                f"Preview {num_examines} samples:\n{repr(new_obj[:num_examines])}"
            )
        return new_obj
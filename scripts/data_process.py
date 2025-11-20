#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocess generation results into RL training datasets.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Callable, Any
from functools import partial
from collections import OrderedDict

from datasets import Dataset

from llm_evaluate.utils.eval_func.recheck import recheck_prompt

def recheck_template(example: dict[str, Any], **kwargs) -> dict[str, Any]:
    """Construct recheck prompt from example."""

    pred_text = example.get("response")
    if isinstance(pred_text, list):
        pred_text = pred_text[0]
    if not isinstance(pred_text, str):
        raise ValueError(
            f"[ERROR] Expected 'response' to be str or list[str], but got {type(pred_text)}"
        )

    extra = example.get("extra_info", {})
    example["prompt"] = recheck_prompt(
        pred_text=pred_text,
        src_text=extra.get("src"),
        target_lang=extra.get("tgt_lang"),
        thinking=kwargs.get("thinking", False),
    )

    if kwargs.get("sft", False):
        if not ((resp := example.get("raw_response", None)) and len(resp) >= 2):
            raise ValueError(
                "[ERROR] For SFT recheck template, 'raw_response' must be a list with at least two elements."
                "The second element is used as the response."
            )
        example["response"] = resp[1]

    if "idx" in kwargs:
        example.setdefault("extra_info", {})["index"] = kwargs["idx"]

    return example


TEMPLATE: dict[str, Callable[[dict, Any], dict]] = OrderedDict(
    [
        ("recheck", recheck_template),
    ]
)


def read_jsonl(file_path: str | Path) -> list[dict[str, Any]]:
    """Load a JSONL file safely with buffering for multiline records."""
    records: list[dict[str, Any]] = []
    buffer = ""

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            buffer += line
            if line.endswith("}"):
                try:
                    record = json.loads(buffer)
                    records.append(record)
                    buffer = ""
                except json.JSONDecodeError:
                    continue

    return records


def parse_template_args(arg_value: str | None) -> dict[str, Any]:
    """Parse template arguments from a JSON string."""
    if not arg_value:
        return {}
    try:
        return json.loads(arg_value)
    except json.JSONDecodeError:
        raise ValueError(
            f"[ERROR] --template_args must be a valid JSON string, got: {arg_value}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate RL training dataset from generation results."
    )
    parser.add_argument("--file", required=True, help="Path to the input JSONL file.")
    parser.add_argument("--template_type", required=True, help="Template key to use.")
    parser.add_argument(
        "--template_args",
        help='Optional JSON string for template args, e.g. \'{"thinking": true}\'',
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Number of samples to keep (randomly sampled).",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="",
        help="Output path (Parquet file) for the processed dataset.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="",
        help="the name of dataset.",
    )
    args = parser.parse_args()

    print(f"[INFO] Loading data from {args.file}")
    data = read_jsonl(args.file)
    ds = Dataset.from_list(data)

    if args.template_type not in TEMPLATE:
        valid_keys = ", ".join(TEMPLATE.keys())
        raise ValueError(
            f"[ERROR] Unknown template '{args.template_type}'. "
            f"Valid options: {valid_keys}"
        )

    template = TEMPLATE[args.template_type]
    template_args = parse_template_args(args.template_args)
    if template_args:
        template = partial(template, **template_args)

    print("[INFO] Processing dataset ...")
    ds = ds.map(lambda x, idx: template(example=x, idx=idx), with_indices=True)

    if args.sample_size and args.sample_size and len(ds) > args.sample_size:
        ds = ds.shuffle(seed=42).select(range(args.sample_size))
        print(f"[INFO] Sampled {args.sample_size} examples.")

    if args.save_dir:
        save_path = Path(args.save_dir)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Saving dataset to {save_path}")
        ds.to_parquet(os.path.join(str(save_path), f"{args.name}.parquet"))
        print("[INFO] Done.")

    print("\n[DEBUG] Example sample:")
    print(ds[0])


if __name__ == "__main__":
    main()

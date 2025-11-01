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


def recheck_prompt(target_lang: str, src_text: str, pred_text: str, thinking: bool = False):

    if "zh" in target_lang:
        user_input = (
            f"给定源文：'{src_text}'，和它的翻译草稿：'{pred_text}'"
            f"必须先理解源文，然后参考以下标准对草稿进行进一步修改润色\n\n"
            f"1. 草稿翻译可能漏翻，请不要遗漏原文的含义\n\n"
            f"2. 保证翻译文段读起来流畅，通顺，符合人类表达，可以调整句子的顺序\n\n"
            f"3. 请注意仔细理解语境，选择书面语还是口语化表达\n\n"
            f"4. 请再检查每个词翻译的意思，是否符合整个语境和现实社会\n\n"
            f"5. 请再检查每个句翻译的意思，是否符合整个语境和现实社会\n\n"
            f"6. 注意你的润色对象是翻译后的草稿，不是源文\n\n"
            f"7. 当你觉得语句读起来困惑的时候，尝试从源文本重新思考\n\n"
            f"8. 如果翻译草稿的语言并非中文，请确保你的润色文本为中文\n\n"
            f"9. 注意检查，不要遗漏源文的含义，也不要添加补充，也不要尝试在翻译中使用过分的比喻\n\n"
        )

        if not thinking:
            user_input += (
                f"10. 不要输出多余思考内容，仅输出你润色后的内容\n\n"
                f"请返回你最后的润色翻译文本，不要输出多余内容。"
            )
        else:
            user_input += (
                f"10. 可以有思考过程，然后输出你润色后的内容\n\n"
                f"请将你最后的润色翻译文本放置在 <text> 和 </text> 之间，请不要输出多余内容。"
            )
    elif "en" in target_lang:
        user_input = (
            f"Given the source text: '{src_text}' and its draft translation: '{pred_text}', "
            f"please refine and polish the draft translation according to the following guidelines:\n\n"
            f"1. The draft translation may have omissions — do not leave out any meaning from the source text.\n\n"
            f"2. Ensure the translation reads smoothly and naturally, consistent with human expression; you may adjust sentence order if needed.\n\n"
            f"3. Carefully understand the context, and decide whether to use formal or colloquial language.\n\n"
            f"4. Recheck the meaning of each word to ensure it fits the overall context and real-world usage.\n\n"
            f"5. Recheck the meaning of each sentence to ensure it fits the overall context and real-world usage.\n\n"
            f"6. Note that your task is to polish the translated draft, not the source text.\n\n"
            f"7. When a sentence feels confusing, reconsider it from the perspective of the source text.\n\n"
            f"8. If the draft translation is not in English, make sure your refined version is in English.\n\n"
            f"9. Do not include any extra reasoning or commentary — only output your polished translation.\n\n"
            f"Please return only the final refined translation text, without any additional content."
        )
    else:
        raise ValueError(
            f"Now we just support lang=['Chinese', 'English'], we get {target_lang}"
        )

    return [{"role": "user", "content": user_input}]


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

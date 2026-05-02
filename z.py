from llm_evaluate.dataset.translation.wmt24 import wmt24
from llm_evaluate.dataset.translation.flores import flores
from datasets import Dataset, concatenate_datasets
import random


wmt_name="en-th_TH"
flores_name="en2th"

x = wmt24(
    "/home/nfs06/shenyz/data/wmt24",
    subset_name=wmt_name,
    split="train",
    extra_args={
        "reverse": False,
        "prompt_template": "qwen_chat_mt",
        "src_key": "source",
        "tgt_key": "target",
        "data_tag": "wmt24"
    }
)


a, b = flores_name.split("2")
y = flores(
    data_dir=[
        "/home/nfs06/shenyz/data/BenchMAX_General_Translation",
        "/home/nfs06/shenyz/data/BenchMAX_General_Translation"
    ],
    subset_name=[
        f"flores_{a}",
        f"flores_{b}"
    ],
    split=["train", "train"],
    extra_args={"prompt_template": "qwen_chat_mt"}
)

x = x.map()
y = y.map()

x.to_parquet(f"/home/nfs06/shenyz/data/recheck/test/wmt24_{wmt_name}.parquet")
y.to_parquet(f"/home/nfs06/shenyz/data/recheck/test/flores_{flores_name}.parquet")

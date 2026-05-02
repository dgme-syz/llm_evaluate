from llm_evaluate.dataset.translation.wmt24 import wmt24
from datasets import Dataset, concatenate_datasets
import random


x = wmt24(
    "wmt/wmt17",
    subset_name="fi-en",
    split="test",
    extra_args={
        "reverse": True,
        "prompt_template": "qwen_chat_mt",
        "src_key": "en",
        "tgt_key": "fi",
        "data_tag": "wmt17"
    }
)

y = wmt24(
    "wmt/wmt18",
    subset_name="fi-en",
    split="test",
    extra_args={
        "reverse": True,
        "prompt_template": "qwen_chat_mt",
        "src_key": "en",
        "tgt_key": "fi",
        "data_tag": "wmt18"
    }
)

z = wmt24(
    "wmt/wmt19",
    subset_name="fi-en",
    split="validation",
    extra_args={
        "reverse": True,
        "prompt_template": "qwen_chat_mt",
        "src_key": "en",
        "tgt_key": "fi",
        "data_tag": "wmt19"
    }
)

x = x.map()
y = y.map()
z = z.map()

combined = concatenate_datasets([x, y, z])

combined = combined.shuffle(seed=42)

sample_size = min(10000000000, len(combined))
sampled = combined.select(range(sample_size))

save_path = "/home/nfs06/shenyz/data/recheck/train/wmt_en2fi_nolim.parquet"
sampled.to_parquet(save_path)

print("Saved to:", save_path)
print("Final size:", len(sampled))
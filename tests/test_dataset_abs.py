import pytest
import json
import os
from datasets import load_dataset, Dataset
from llm_evaluate.dataset import EvalDataset

# ---- 测试数据文件 ----
TEST_JSONL = "test_data.json"

def generate_test_json():
    """生成本地 JSONL 测试数据"""
    data = [
        {"text": "Hello world", "label": 0},
        {"text": "Bonjour le monde", "label": 1},
        {"text": "こんにちは世界", "label": 2},
    ]
    with open(TEST_JSONL, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

# ---- Mock EvalDataset ----
class MockEvalDataset(EvalDataset):
    def __init__(self, data_dir, subset_name=None, split="train", builder=None):
        super().__init__(data_dir, subset_name, split, builder)

    def convert_item(self, example):
        if isinstance(example, dict):
            return {"t": example.get("text", "")}
        elif isinstance(example, list):
            return {"t": "\n\n".join([e.get("text", "") for e in example])}
        return {"t": ""}

    def convert_items(self, examples):
        if isinstance(examples, dict):
            return {"t": examples.get("text", [])}
        elif isinstance(examples, list):
            x = []
            for e in examples:
                x += e.get("text", [])
            return {"t": x}
        return {"t": []}


# ---- Pytest Fixture ----
@pytest.fixture(scope="module", autouse=True)
def setup_and_teardown():
    generate_test_json()
    yield
    if os.path.exists(TEST_JSONL):
        os.remove(TEST_JSONL)

# ---- Tests ----
def test_single_dataset_init():
    ds = MockEvalDataset(TEST_JSONL, subset_name=None, split="train", builder="json")

def test_multi_dataset_init():
    ds = MockEvalDataset([TEST_JSONL, TEST_JSONL], subset_name=[None, None], split=["train", "train"], builder=["json", "json"])

def test__convert_item_single():
    ds = MockEvalDataset(TEST_JSONL, subset_name=None, split="train", builder="json")
    ex = ds.datasets[0]
    out = ds.convert_items(ex)
    assert "t" in out

def test__convert_item_batch():
    ds = MockEvalDataset(TEST_JSONL, subset_name=None, split="train", builder="json")
    indices = [0, 1, 2]
    ex = ds.datasets[indices]
    out = ds.convert_items(ex)
    assert "t" in out
    assert isinstance(out["t"], str)

def test_map_single_mode():
    ds = MockEvalDataset(TEST_JSONL, subset_name=None, split="train", builder="json")
    mapped = ds.map(batched=False, num_examines=2)
    assert "t" in mapped.column_names
    assert "label" not in mapped.column_names
    assert isinstance(mapped[0]["t"], str)

def test_map_batched_mode():
    ds = MockEvalDataset(TEST_JSONL, subset_name=None, split="train", builder="json")
    mapped = ds.map(batched=True, batch_size=2, num_examines=2)
    assert "t" in mapped.column_names
    assert isinstance(mapped[0]["t"], str)

def test_map_single_multi_datasets():
    ds = MockEvalDataset([TEST_JSONL, TEST_JSONL], subset_name=[None, None], split=["train", "train"], builder=["json", "json"])
    mapped = ds.map(batched=False, num_examines=2)
    assert "t" in mapped.column_names
    assert isinstance(mapped[0]["t"], str)

def test_map_batched_multi_datasets():
    ds = MockEvalDataset([TEST_JSONL, TEST_JSONL], subset_name=[None, None], split=["train", "train"], builder=["json", "json"])
    mapped = ds.map(batched=True, batch_size=2, num_examines=2)
    assert "t" in mapped.column_names
    assert isinstance(mapped[0]["t"], str)

from .abstract import EvalDataset
from .registry import get_dataset, register
from .translation.flores import flores
from .translation.wmt24 import wmt24
from .math.gsm8k import gsm8k
__all__ = [
    "EvalDataset",
    "get_dataset",
    "register",
    "flores",
    "gsm8k",
]
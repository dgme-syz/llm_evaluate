from .abstract import EvalDataset
from .registry import get_dataset, register
from .translation.flores import flores
from .translation.wmt24 import wmt24
from .math.gsm8k import gsm8k
from .math.math500 import math500
from .math.math500 import mathtrain
from .math.aime2024 import aime2024
from .math.aime2025 import aime2025

__all__ = [
    "EvalDataset",
    "get_dataset",
    "register",
    "flores",
    "gsm8k",
    "math500", 
    "aime2024",
    "aime2025",
    "mathtrain",
]
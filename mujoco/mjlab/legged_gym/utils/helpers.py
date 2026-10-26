from __future__ import annotations

import random
from typing import Any

import numpy as np


def class_to_dict(obj: Any) -> Any:
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        value = getattr(obj, key)
        if callable(value):
            continue
        if isinstance(value, list):
            result[key] = [class_to_dict(item) for item in value]
        else:
            result[key] = class_to_dict(value)
    return result


def set_seed(seed: int) -> int:
    if seed == -1:
        seed = int(np.random.randint(0, 10000))
    random.seed(seed)
    np.random.seed(seed)
    return seed


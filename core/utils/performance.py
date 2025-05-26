from __future__ import annotations

import time
from functools import wraps
from typing import Any, Callable, Tuple


def measure_time(func: Callable[..., Any]) -> Callable[..., Tuple[Any, float]]:
    """Decorator to measure execution time of a function."""

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Tuple[Any, float]:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        return result, end - start

    return wrapper


__all__ = ["measure_time"]

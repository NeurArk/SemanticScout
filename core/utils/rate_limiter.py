from __future__ import annotations

# pragma: no cover

import time
import threading


class RateLimiter:
    """Simple rate limiter enforcing a delay between calls."""

    def __init__(self, delay: int = 1) -> None:
        self.delay = delay
        self._lock = threading.Lock()
        self._last_call = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait_time = self.delay - (now - self._last_call)
            if wait_time > 0:
                time.sleep(wait_time)
            self._last_call = time.monotonic()


__all__ = ["RateLimiter"]

"""
Thread-safe sliding-window rate limiter for remote oracle backends.

Used by both DartmouthOracle and GeminiOracle to enforce per-minute
request quotas without coupling the backends to each other.
"""

import threading
import time
from collections import deque


class RateLimiter:
    """Thread-safe sliding-window rate limiter.

    Enforces a maximum number of requests within a rolling 60-second window.
    Threads call acquire() before each API request; if the window is full,
    the caller sleeps until a slot opens.
    """

    def __init__(self, requests_per_minute: int):
        self.rpm = requests_per_minute
        self._timestamps: deque = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        """Block until a request slot is available."""
        while True:
            with self._lock:
                now = time.monotonic()
                while self._timestamps and now - self._timestamps[0] >= 60.0:
                    self._timestamps.popleft()
                if len(self._timestamps) < self.rpm:
                    self._timestamps.append(now)
                    return
                wait = 60.0 - (now - self._timestamps[0]) + 0.05
            time.sleep(wait)

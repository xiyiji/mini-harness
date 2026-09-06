"""Retry an API call with exponential backoff.

Two budgets: a short one for transient network/server errors, a longer,
capped one for rate limits (429), which usually need tens of seconds to clear.
Anything else (bad request, auth error) is not retried; it would just fail
the same way again.
"""

import random
import time
from typing import Callable

from httpx import RemoteProtocolError
from openai import APIConnectionError, InternalServerError, RateLimitError

from mini_harness.config import CONFIG

RETRYABLE = (APIConnectionError, InternalServerError, RateLimitError, RemoteProtocolError)


def retry_call(request: Callable, cfg=CONFIG):
    attempts = max(cfg.max_retry, cfg.rate_retry)
    for attempt in range(attempts):
        try:
            return request()
        except RETRYABLE as e:
            limited = isinstance(e, RateLimitError)
            allowed = cfg.rate_retry if limited else cfg.max_retry
            if attempt >= allowed - 1:
                raise
            base = cfg.rate_base if limited else cfg.retry_base
            wait = base * (2**attempt) + random.uniform(0, 1)   # jitter avoids thundering herd
            if limited:
                wait = min(wait, cfg.rate_cap)
            print(f"[retry]: {type(e).__name__}, attempt {attempt + 1}/{allowed}, sleeping {wait:.1f}s")
            time.sleep(wait)
    raise ValueError(f"[retry]: retry budget must be positive, got {attempts}")

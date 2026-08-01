"""Concurrency + rate-limit helpers shared by every provider.

ComfyUI runs `async def` nodes **concurrently**, so several API nodes in one graph fire
their requests at the same moment. That is great for throughput and fatal for providers
with a small burst allowance — Replicate drops to "6 requests per minute with a burst of
1" while an account holds less than $5 credit, so parallel calls reliably 429.

Two independent mitigations live here:

* :func:`serial_lock` — an asyncio lock so nodes sharing a key run one at a time.
* :func:`with_rate_limit_retry` — retries a 429 with exponential backoff, honouring the
  provider's ``Retry-After`` when it sends one.
"""

import asyncio
import contextlib
import random
import re
import time

_LOCKS = {}
_SEMAPHORES = {}


def serial_lock(key="replicate"):
    """A per-event-loop asyncio.Lock, created on first use.

    Locks must belong to the running loop, and ComfyUI may restart it, so they are keyed
    by loop identity rather than created at import time.
    """
    loop = asyncio.get_running_loop()
    k = (id(loop), key)
    lock = _LOCKS.get(k)
    if lock is None:
        lock = _LOCKS[k] = asyncio.Lock()
    return lock


def concurrency_gate(limit, key="replicate"):
    """An async context manager capping concurrent calls to ``limit``.

    ``limit <= 0`` means no cap. Like :func:`serial_lock` the semaphore is keyed by the
    running loop, and it is rebuilt if the limit changes so turning the dial takes effect
    on the next run instead of needing a restart.
    """
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = 0
    if limit <= 0:
        return contextlib.nullcontext()
    if limit == 1:
        return serial_lock(key)

    loop = asyncio.get_running_loop()
    k = (id(loop), key)
    sem, current = _SEMAPHORES.get(k, (None, None))
    if sem is None or current != limit:
        sem = asyncio.Semaphore(limit)
        _SEMAPHORES[k] = (sem, limit)
    return sem


def is_rate_limited(exc):
    status = getattr(exc, "status", None) or getattr(getattr(exc, "response", None), "status_code", None)
    if status == 429:
        return True
    return "429" in str(exc) or "throttled" in str(exc).lower()


def _retry_after_seconds(exc):
    """Seconds the provider asked us to wait, if it said so."""
    resp = getattr(exc, "response", None)
    try:
        v = resp.headers.get("retry-after")
        if v:
            return float(v)
    except Exception:
        pass
    m = re.search(r"resets? in ~?(\d+(?:\.\d+)?)\s*s", str(exc), re.I)
    return float(m.group(1)) if m else None


def with_rate_limit_retry(fn, attempts=6, base_delay=5.0, max_delay=90.0, log=None):
    """Call ``fn()``, retrying only on rate-limit errors. Other errors propagate."""
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:
            if not is_rate_limited(exc) or attempt == attempts - 1:
                raise
            delay = _retry_after_seconds(exc)
            if delay is None:
                delay = min(base_delay * (2 ** attempt), max_delay)
            delay += random.uniform(0.5, 1.5)      # jitter so parallel nodes desynchronise
            if log:
                log(f"rate limited, retrying in {delay:.0f}s "
                    f"(attempt {attempt + 1}/{attempts - 1})")
            time.sleep(delay)

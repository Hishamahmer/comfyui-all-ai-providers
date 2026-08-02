"""Concurrency + retry helpers shared by every provider.

ComfyUI runs `async def` nodes **concurrently**, so several API nodes in one graph fire
their requests at the same moment. That is great for throughput and fatal for providers
with a small burst allowance — Replicate drops to "6 requests per minute with a burst of
1" while an account holds less than $5 credit, so parallel calls reliably 429.

Three independent mitigations live here:

* :func:`serial_lock` — an asyncio lock so nodes sharing a key run one at a time.
* :func:`concurrency_gate` — the same idea with a dial instead of a hard 1.
* :func:`with_retry` — retries a call that never got a complete answer: a 429, a 5xx, or
  a connection that died mid-response. Honours ``Retry-After`` when the provider sends
  one. Anything else (a refusal, a bad request, an auth failure) propagates immediately —
  retrying those just burns time and money.

Why the transport cases matter: one 24-shot batch died at shot ~20 on
``httpx.RemoteProtocolError: peer closed connection without sending complete message body``
from the Codex backend. A node exception aborts the whole ComfyUI prompt, so a single
network blip threw away the rest of the run.
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


def _status_of(exc):
    return (getattr(exc, "status", None)
            or getattr(getattr(exc, "response", None), "status_code", None))


def is_rate_limited(exc):
    if _status_of(exc) == 429:
        return True
    return "429" in str(exc) or "throttled" in str(exc).lower()


# Matched by class NAME so httpx/httpcore/requests stay un-imported here — this module is
# loaded by every provider and must not care which HTTP client they chose.
_TRANSIENT_EXCEPTIONS = {
    "RemoteProtocolError",      # what killed the 24-shot Codex run
    "LocalProtocolError",
    "ProtocolError",
    "ReadError", "ReadTimeout",
    "WriteError", "WriteTimeout",
    "ConnectError", "ConnectTimeout",
    "PoolTimeout", "NetworkError",
    "IncompleteRead", "ChunkedEncodingError",
    "ConnectionResetError", "ConnectionAbortedError", "ConnectionError",
}

_TRANSIENT_TEXT = (
    "incomplete chunked read",
    "peer closed connection",
    "server disconnected",
    "connection reset",
    "connection aborted",
    "connection closed",
    "timed out",
    "bad gateway",
    "service unavailable",
    "gateway time-out", "gateway timeout",
)


def is_transient(exc):
    """True for a call that never got a complete answer, so retrying is worth it.

    Deliberately narrow: a moderation refusal, a malformed request or an expired login
    are all permanent, and retrying them wastes the user's time and quota.
    """
    if getattr(exc, "retryable", False):     # providers can mark their own exceptions
        return True
    if _status_of(exc) in (500, 502, 503, 504):
        return True
    for cls in type(exc).__mro__:
        if cls.__name__ in _TRANSIENT_EXCEPTIONS:
            return True
    text = str(exc).lower()
    return any(fragment in text for fragment in _TRANSIENT_TEXT)


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


def with_retry(fn, attempts=6, base_delay=5.0, max_delay=90.0, log=None):
    """Call ``fn()``, retrying rate limits and dropped connections. Else propagate.

    A rate limit means "come back later", so it backs off from ``base_delay``. A dropped
    connection means "that one got lost", so it retries almost immediately — waiting a
    minute does not make the socket healthier.
    """
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:
            limited = is_rate_limited(exc)
            if attempt == attempts - 1 or not (limited or is_transient(exc)):
                raise
            delay = _retry_after_seconds(exc)
            if delay is None:
                delay = (min(base_delay * (2 ** attempt), max_delay) if limited
                         else min(2.0 * (attempt + 1), 15.0))
            delay += random.uniform(0.5, 1.5)      # jitter so parallel nodes desynchronise
            if log:
                log("%s, retrying in %.0fs (attempt %d/%d): %s"
                    % ("rate limited" if limited else "connection lost",
                       delay, attempt + 1, attempts - 1, str(exc)[:160]))
            time.sleep(delay)

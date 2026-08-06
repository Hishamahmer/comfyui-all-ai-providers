"""Shared stream consumption for the Codex nodes: watchdogs, terminal events, timings.

WHY THIS EXISTS
    Both Codex nodes opened a stream, looped to exhaustion and hoped. Four things went
    wrong with that, and all four look identical from outside — the node just sits there:

    1. HTTP 200 proves only that headers arrived. The read timeout was set to the TOTAL
       timeout (600 s by default), so a response that never sent a body byte blocked for
       ten minutes; and because httpx's read timeout measures the gap BETWEEN bytes, a
       stream that heartbeats every few seconds and never says anything blocked forever.
       Fixed by timing the first event and the idle gap separately from the total, with
       the socket read timeout pinned to the idle budget.

    2. The loop had no `break` on the terminal event, so after `response.completed` it
       kept reading until EOF. A server that holds the connection open turns a finished
       request into a hang. Fixed by stopping the moment a terminal event lands.

    3. `response.failed` and `response.incomplete` were treated as success. That is how
       a truncated answer reached `json.loads` and died at "column 2911" after thirteen
       minutes of work, and how a half-rendered image could be returned as final. They
       are now failures, and retryable ones.

    4. Validation lived outside the retry, so a truncated answer killed the run instead
       of causing another attempt.

NOT A TIMEOUT INCREASE
    Nothing here waits longer than before. It waits for the RIGHT thing and gives up on
    silence quickly, which is the opposite trade: healthy requests are untouched, dead
    ones are detected in ~45 s instead of ~600 s or never.
"""

import time

# Only this one means "the answer is complete and good".
TERMINAL_OK = ("response.completed",)

# These are terminal too, but they are FAILURES. Returning their partial payload is what
# produced truncated JSON and half-finished images.
TERMINAL_BAD = ("response.failed", "response.incomplete", "response.cancelled", "error")

# Watchdog budgets, seconds. Deliberately much shorter than the total timeout: they
# measure SILENCE, not work. A request that is streaming stays alive indefinitely under
# these; a request that has gone quiet is killed and retried.
FIRST_EVENT_TIMEOUT = 60.0     # HTTP 200 -> first parsed event
IDLE_TIMEOUT = 120.0           # gap between two events once streaming has begun


class CodexStalled(RuntimeError):
    """The stream went silent. Always worth a fresh attempt with a new connection."""
    retryable = True


class CodexCancelled(RuntimeError):
    """The user pressed Cancel. NOT retryable — retrying a cancellation is absurd, and
    marking it transient made a cancelled run burn its whole retry budget on attempts
    that each died in under a second."""
    retryable = False


class CodexRemoteFailure(RuntimeError):
    """The server ended the response with a failure/incomplete terminal event."""

    def __init__(self, message, retryable=True):
        RuntimeError.__init__(self, message)
        self.retryable = retryable


class StreamStats(object):
    """Timings for one attempt. Printed as one line so logs stay readable."""

    def __init__(self, label):
        self.label = label
        self.started = time.time()
        self.headers_at = None
        self.first_event_at = None
        self.terminal_at = None
        self.events = 0
        self.deltas = 0
        self.chars = 0
        self.last_event_type = None

    def _since(self, stamp):
        return None if stamp is None else stamp - self.started

    def line(self, outcome):
        def fmt(value):
            return "-" if value is None else "%.1fs" % value
        return ("[%s] %s | headers %s | first event %s | terminal %s | total %.1fs | "
                "events=%d deltas=%d chars=%d | last=%s"
                % (self.label, outcome, fmt(self._since(self.headers_at)),
                   fmt(self._since(self.first_event_at)),
                   fmt(self._since(self.terminal_at)), time.time() - self.started,
                   self.events, self.deltas, self.chars, self.last_event_type or "-"))


def consume(events, stats, on_event, first_event_timeout=FIRST_EVENT_TIMEOUT,
            idle_timeout=IDLE_TIMEOUT, total_timeout=None, should_stop=None):
    """Drive an SSE event iterator under watchdogs, stopping at the terminal event.

    `on_event(event)` is called for every non-terminal event and may accumulate whatever
    the caller needs. Returns the terminal event.

    Raises `CodexStalled` if the stream goes quiet, `CodexRemoteFailure` if the server
    ends it badly, and `CodexStalled` if it ends with no terminal event at all — every
    one of those retryable, because each means "this attempt is dead", not "the request
    was wrong".

    The socket-level read timeout still does the heavy lifting for true silence; these
    checks catch the case where events keep arriving but the response never terminates.
    """
    stats.headers_at = stats.headers_at or time.time()
    last_seen = time.time()

    for event in events:
        now = time.time()
        if should_stop is not None and should_stop():
            raise CodexCancelled("cancelled by ComfyUI")
        if not isinstance(event, dict):
            continue

        # Measure the gap BEFORE moving the marker, or it is always zero.
        gap, last_seen = now - last_seen, now
        stats.events += 1
        stats.last_event_type = event.get("type")
        if stats.first_event_at is None:
            stats.first_event_at = now
            if gap > first_event_timeout:
                raise CodexStalled("first event took %.0fs (budget %.0fs)"
                                   % (gap, first_event_timeout))
        elif gap > idle_timeout:
            raise CodexStalled("stream idle for %.0fs between events (budget %.0fs)"
                               % (gap, idle_timeout))
        if total_timeout and (now - stats.started) > total_timeout:
            raise CodexStalled("exceeded the total budget of %.0fs" % total_timeout)

        kind = event.get("type")
        if kind in TERMINAL_OK:
            stats.terminal_at = now
            return event
        if kind in TERMINAL_BAD:
            stats.terminal_at = now
            detail = _describe_failure(event)
            raise CodexRemoteFailure(
                "Codex ended the response as '%s'%s. The answer is incomplete, so it is "
                "being discarded rather than passed on." % (kind, detail))

        on_event(event)

    # Ran out of events without a terminal one: the connection was cut short.
    raise CodexStalled("the stream ended without a completion event")


def _describe_failure(event):
    response = event.get("response")
    if isinstance(response, dict):
        for key in ("incomplete_details", "error"):
            value = response.get(key)
            if isinstance(value, dict) and value.get("reason"):
                return " (%s)" % value["reason"]
            if isinstance(value, dict) and value.get("message"):
                return " (%s)" % str(value["message"])[:160]
    error = event.get("error")
    if isinstance(error, dict) and error.get("message"):
        return " (%s)" % str(error["message"])[:160]
    return ""


def interrupted():
    """True when the user has pressed Cancel in ComfyUI.

    Checked between events so a long stream stops promptly instead of running to
    completion after the queue has already been cleared.
    """
    try:
        import comfy.model_management as mm
        check = getattr(mm, "processing_interrupted", None)
        if callable(check):
            return bool(check())
        return bool(getattr(mm, "interrupt_processing", False))
    except Exception:
        return False


def timeouts(idle=IDLE_TIMEOUT, connect=20.0):
    """httpx timeouts that treat SILENCE as the failure, not elapsed work.

    `read` is the gap allowed between bytes, so pinning it to the idle budget is what
    stops a dead connection from being held for the whole total timeout. It does NOT cap
    how long a healthy streaming request may run.
    """
    import httpx
    return httpx.Timeout(idle, connect=connect, read=idle, write=30.0, pool=connect)

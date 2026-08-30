"""Image generation through a ChatGPT/Codex login — no API key.

Routes `gpt-image-2` through the ChatGPT Codex Responses API's ``image_generation``
tool, authenticated with the OAuth credentials `codex login` already wrote. Same model
as the Replicate Image Gen node; different billing path (your ChatGPT plan instead of a
Replicate balance).

Multiple accounts: give each its own ``CODEX_HOME`` folder, log into each with the CLI,
and point ``codex_home`` at the one you want. Blank = ``~/.codex``.

Availability is account-dependent: not every ChatGPT plan can call the hosted image
tool. When it cannot, the backend answers with a specific 400 that we translate into a
readable error rather than a raw HTTP dump.
"""

import json
import time

from ..common.image_utils import (
    collect_images_to_data_uris,
    bytes_list_to_image_tensor,
    output_to_bytes_list,
)
from ..common.throttle import (
    concurrency_gate, serial_lock, with_retry,
)
from ..replicate_provider.settings import SETTINGS_TYPE
from . import auth as codex_auth
from . import stream as codex_stream

DEFAULT = "default"
RUN_ONE_AT_A_TIME = "one at a time"
RUN_ALL_AT_ONCE = "all at once"

# The Codex CLI's Speed setting, which travels as `service_tier` on the request. Its own
# UI calls these "Standard - default speed" and "Fast - 1.5x speed, increased usage".
# Probed against the live backend 2026-08-09: "priority" and "default" are accepted;
# "fast" and "flex" both come back 400 `Unsupported service_tier`. So the wire value is
# "priority" — do not "correct" it to match the label. Shared with the LLM node, which
# imports these from here.
SPEED_STANDARD = "standard"
SPEED_FAST = "fast (1.5x, uses more quota)"
SERVICE_TIER = {SPEED_STANDARD: "default", SPEED_FAST: "priority"}

BASE_URL = "https://chatgpt.com/backend-api/codex"
IMAGE_MODEL = "gpt-image-2"

# Preview frames to ask for. **0 — none.** Nothing consumes them: the node returns one
# finished image, so a preview was only ever a fallback that could not be used.
#
# Measured 2026-08-21, and this is why it is 0 rather than 1. On a run that failed six
# times, the stream census showed the backend emitting
# `response.image_generation_call.partial_image` carrying a 2.5 MB payload at
# `quality: medium`, then `response.completed` with the final `result` never populated —
# so the node had a medium-quality preview it must not return and no image it could.
# A second call in the same run, which emitted no partial at all, returned its final on
# the first attempt. Asking for previews is at best free and at worst the thing that
# stops the real image arriving; nothing here wants them either way.
PARTIAL_IMAGES = 0
# The chat model is only the host that invokes the tool; it makes no pixels itself.
HOST_MODEL = "gpt-5.5"
INSTRUCTIONS = ("You are an assistant that must fulfill image generation and image "
                "editing requests by using the image_generation tool when provided.")

# gpt-image-2's supported canvases, keyed by the same vocabulary the Replicate node
# uses so one shared Settings node can drive both.
# Pixel sizes for the named ratios. Best-effort only — this backend ignores the tool's
# `size` field, so the prompt line below is what actually shapes the output.
SIZES = {
    "1:1": "1024x1024",
    "3:2": "1536x1024",
    "2:3": "1024x1536",
    "4:3": "1536x1152",
    "3:4": "1152x1536",
    "16:9": "1792x1024",
    "9:16": "1024x1792",
}


def _tool_size(aspect_ratio):
    """A WxH string for the tool, or '' when there is nothing sensible to send.

    Must never raise: the aspect list includes ratios and explicit sizes, and a value we
    have no mapping for is not a reason to fail a paid call.
    """
    if aspect_ratio in (DEFAULT, "auto", "", None):
        return ""
    mapped = SIZES.get(aspect_ratio)
    if mapped:
        return mapped
    if "x" in str(aspect_ratio).lower():          # already an explicit size
        return str(aspect_ratio).lower()
    return ""

# The Codex backend IGNORES the tool's `size` field — verified 2026-08-01: asking for
# 1024x1536 and 1536x1024 both returned 1254x1254. Stating the shape in the prompt DOES
# work (same test returned exactly 1024x1536 and 1536x1024), so the orientation is
# appended to the prompt. `size` is still sent in case the backend starts honouring it.
ASPECT_TEXT = {
    "1:1": "a SQUARE image, 1:1 aspect ratio, 1024x1024 pixels, equal width and height",
    "3:2": "a LANDSCAPE image, 3:2 aspect ratio, 1536x1024 pixels, wider than it is tall",
    "2:3": "a PORTRAIT image, 2:3 aspect ratio, 1024x1536 pixels, taller than it is wide",
    "4:3": "a LANDSCAPE image, 4:3 aspect ratio, 1536x1152 pixels, wider than it is tall",
    "3:4": "a PORTRAIT image, 3:4 aspect ratio, 1152x1536 pixels, taller than it is wide",
    "16:9": "a WIDESCREEN image, 16:9 aspect ratio, 1792x1024 pixels, much wider than tall",
    "9:16": "a VERTICAL image, 9:16 aspect ratio, 1024x1792 pixels, much taller than wide",
    "auto": "",
}


def _aspect_line(aspect_ratio):
    """The orientation sentence appended to the prompt, or '' when there is nothing to say.

    Handles both named ratios and explicit WxH values from the shared Settings node.
    """
    if aspect_ratio in ("default", "auto"):
        return ""
    described = ASPECT_TEXT.get(aspect_ratio)
    if described:
        return "Output format: %s." % described
    if "x" in aspect_ratio:                       # an explicit size like 2048x1152
        try:
            w, h = (int(v) for v in aspect_ratio.lower().split("x", 1))
        except ValueError:
            return ""
        shape = ("a SQUARE image" if w == h else
                 "a LANDSCAPE image, wider than it is tall" if w > h else
                 "a PORTRAIT image, taller than it is wide")
        return "Output format: %s, %dx%d pixels." % (shape, w, h)
    return ""

_UNSUPPORTED = "Tool choice 'image_generation' not found in 'tools' parameter."
_UNSUPPORTED_HELP = (
    "This ChatGPT account cannot use the hosted image tool. Try another account via "
    "codex_home, or use the Replicate Image Gen node instead.")


class CodexImageUnavailable(RuntimeError):
    """The signed-in ChatGPT account has no access to the hosted image tool."""


def _is_unsupported(status, body):
    if status != 400:
        return False
    try:
        payload = json.loads(body)
        message = (payload.get("error") or {}).get("message")
    except Exception:
        message = body
    return isinstance(message, str) and message.strip() == _UNSUPPORTED


def _iter_sse(response):
    """Yield JSON events from an SSE stream.

    Parsed by hand rather than with an SDK: the backend emits image events newer than
    any pinned SDK understands, and an unknown event shape should not abort the run.
    """
    event_name, data_lines = None, []

    def flush():
        nonlocal event_name, data_lines
        if not data_lines:
            event_name = None
            return None
        raw = "\n".join(data_lines).strip()
        name, event_name, data_lines = event_name, None, []
        if not raw or raw == "[DONE]":
            return None
        try:
            payload = json.loads(raw)
        except ValueError:
            return None
        if isinstance(payload, dict) and name and "type" not in payload:
            payload["type"] = name
        return payload

    for line in response.iter_lines():
        if isinstance(line, bytes):
            line = line.decode("utf-8", errors="replace")
        line = str(line)
        if line == "":
            got = flush()
            if got is not None:
                yield got
        elif line.startswith(":"):
            continue
        elif line.startswith("event:"):
            event_name = line[6:].strip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].lstrip())

    got = flush()
    if got is not None:
        yield got


# A 1024x1024 PNG is hundreds of kilobytes of base64. Anything short is a message, an
# id or a status string, not a picture.
_MIN_IMAGE_B64 = 2048


def _shape_of(value, depth=0):
    """Key names and value KINDS of a payload — never the values themselves.

    Enough to answer "where did the image go" from a log line, without printing a
    megabyte of base64 or anything the user would not want on disk.
    """
    if isinstance(value, dict):
        if depth >= 3:
            return "{%s}" % ", ".join(sorted(value)[:8])
        return "{%s}" % ", ".join(
            "%s: %s" % (k, _shape_of(v, depth + 1)) for k, v in list(value.items())[:12])
    if isinstance(value, list):
        return "[%d x %s]" % (len(value),
                              _shape_of(value[0], depth + 1) if value else "-")
    if isinstance(value, str):
        return "str(%d)" % len(value)
    return type(value).__name__


def _looks_like_image_b64(value):
    """Is this string plausibly a base64 image payload rather than ordinary text?"""
    if not isinstance(value, str) or len(value) < _MIN_IMAGE_B64:
        return False
    head = value[:64]
    return not (" " in head or "\n" in head)


def _find_image_b64(value):
    """``(base64, is_final)`` — the image in an event payload (shapes vary by version).

    ``is_final`` separates a finished render from a `partial_images` preview. They used to
    be indistinguishable, so a stream cut short mid-render could silently hand back a
    half-drawn image as if it were the result.

    **A FINAL ALWAYS WINS.** The previous version walked the payload and let the last hit
    replace whatever it had already found — so an event carrying both the completed
    `image_generation_call` and a `partial_image_b64` (or any nesting that visited the
    partial second) returned the PREVIEW and reported "no final image". The caller then
    retried six times, found the same thing every time, and killed the run with a
    `CodexStreamTruncated` on a request the backend had actually completed.
    """
    final = None
    partial = None

    def walk(node):
        nonlocal final, partial
        if isinstance(node, dict):
            # `type` has been both "image_generation_call" and, in later shapes, a
            # dotted variant of it. Substring rather than equality so a rename of the
            # kind this undocumented backend does without notice degrades to "still
            # works" instead of "silently returns previews forever".
            kind = node.get("type")
            if isinstance(kind, str) and "image_generation_call" in kind:
                result = node.get("result")
                if isinstance(result, str) and result and final is None:
                    final = result
            # Fallback shapes, for when the item type gets renamed again. Guarded by
            # `_looks_like_image_b64` because `result` is a common key for ordinary text
            # too, and treating a sentence as a PNG fails much later and much worse.
            for key in ("result", "b64_json", "image_b64"):
                candidate = node.get(key)
                if final is None and _looks_like_image_b64(candidate):
                    final = candidate
            preview = node.get("partial_image_b64")
            if isinstance(preview, str) and preview:
                partial = preview
            for child in node.values():
                walk(child)
        elif isinstance(node, list):
            for child in node:
                walk(child)

    walk(value)
    if final:
        return (final, True)
    return (partial, False) if partial else (None, False)


# Any of these means the backend finished talking. Without one, the stream was cut off.
_TERMINAL_EVENTS = ("response.completed", "response.failed", "response.incomplete",
                    "response.error", "error")


class CodexStreamTruncated(RuntimeError):
    """The backend closed the SSE stream before the image finished."""

    retryable = True          # read by common.throttle.is_transient


class ArkCodexImageGen:
    CATEGORY = "arkennemasis/Image Gen"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "account")
    DESCRIPTION = ("Generate or edit an image with gpt-image-2 using your ChatGPT/Codex "
                   "login instead of an API key. Run `codex login` first. Multiple "
                   "accounts: set codex_home per account.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "image_1": ("IMAGE", {"tooltip": "Optional reference / image to edit. "
                                                 "Can be a batch."}),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "aspect_ratio": ([DEFAULT, "1:1", "3:2", "2:3", "4:3", "3:4", "16:9", "9:16", "auto", "1024x1024", "1536x1024", "1024x1536", "1536x1152", "1152x1536", "2048x2048", "2048x1152", "1152x2048", "3840x2160", "2160x3840"], {
                    "tooltip": "Stated in the prompt, because this backend ignores "
                               "the size field. Wide/tall targets land close but not "
                               "exact (16:9 came back as 7:4). Use the Replicate node "
                               "when the ratio must be precise.",
                }),
                "quality": ([DEFAULT, "low", "medium", "high"],),
                "background": ([DEFAULT, "opaque", "transparent", "auto"],),
                "codex_home": ("STRING", {
                    "default": "",
                    "tooltip": "Folder holding auth.json. Blank = CODEX_HOME, else "
                               "~/.codex. Give each ChatGPT account its own folder and "
                               "point here to switch account.",
                }),
                "allow_refresh": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Refresh an expired token and save it back. Off = fail "
                               "instead of writing to auth.json.",
                }),
                "timeout_seconds": ("INT", {
                    "default": 300, "min": 0, "max": 86400,
                    "tooltip": "0 = wait indefinitely.",
                }),
                "force_rerun": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Call again even if inputs are unchanged.",
                }),
                "run_mode": ([RUN_ONE_AT_A_TIME, RUN_ALL_AT_ONCE], {
                    "tooltip": "ComfyUI runs async nodes concurrently. 'one at a time' "
                               "serialises every arkennemasis API node in the graph.",
                }),
                "max_concurrent": ("INT", {
                    "default": 2, "min": 0, "max": 32,
                    "tooltip": "Only used when run_mode is 'all at once': how many "
                               "arkennemasis API calls may be in flight together. "
                               "0 = no cap. 2 is a safe default for a 24-shot batch.",
                }),
                "on_failure": (["fail the run", "use the reference image"], {
                    "tooltip": "What to do when the backend definitively returns no "
                               "image after every retry - usually a moderation refusal. "
                               "'fail the run' is right for a single image. Inside a "
                               "multi-scene loop it is catastrophic: one refusal on the "
                               "23rd image threw away 22 finished images and 36 minutes "
                               "of work. 'use the reference image' substitutes the input "
                               "image for that scene so the run completes, and says so "
                               "loudly in the log so you can re-render just that scene.",
                }),
                "settings": (SETTINGS_TYPE, {
                    "tooltip": "Optional: the shared 'Image Gen Settings' node, so this "
                               "and the Replicate nodes follow one place. Codex uses "
                               "aspect_ratio / quality / background / run_mode / "
                               "timeout_seconds; output_format, moderation and api_token "
                               "do not apply and are ignored.",
                }),
                # APPENDED last: widgets_values is positional, so anywhere else would
                # shift every later value in workflows already saved.
                "speed": ([SPEED_STANDARD, SPEED_FAST], {
                    "tooltip": "The Codex CLI's Speed setting. 'fast' sends "
                               "service_tier=priority (its '1.5x speed, increased "
                               "usage' option). Worth far less here than on the text "
                               "step: an image is dominated by generation time, not by "
                               "queueing, so this mostly just spends plan quota faster. "
                               "Leave it on standard unless you are waiting on a single "
                               "hero image.",
                }),
            },
        }

    @classmethod
    def IS_CHANGED(cls, force_rerun=False, **kwargs):
        import time
        return time.time() if force_rerun else ""

    async def run(self, prompt, image_1=None, image_2=None, image_3=None, image_4=None,
                  aspect_ratio=DEFAULT, quality=DEFAULT, background=DEFAULT,
                  codex_home="", allow_refresh=True, timeout_seconds=300,
                  force_rerun=False, run_mode=RUN_ONE_AT_A_TIME, settings=None,
                  max_concurrent=2, on_failure="fail the run",
                  speed=SPEED_STANDARD):
        import asyncio

        if settings:                       # a wired Settings node overrides the widgets
            aspect_ratio = settings.get("aspect_ratio", aspect_ratio)
            quality = settings.get("quality", quality)
            background = settings.get("background", background)
            run_mode = settings.get("run_mode", run_mode)
            max_concurrent = settings.get("max_concurrent", max_concurrent)
            timeout_seconds = settings.get("timeout_seconds", timeout_seconds)

        async def go():
            return await asyncio.to_thread(
                self._blocking, prompt, image_1, image_2, image_3, image_4,
                aspect_ratio, quality, background, codex_home, allow_refresh,
                timeout_seconds, on_failure, speed,
            )

        # 'all at once' still respects max_concurrent; 'one at a time' is always 1.
        gate = (concurrency_gate(max_concurrent) if run_mode == RUN_ALL_AT_ONCE
                else serial_lock())
        async with gate:
            return await go()

    def _blocking(self, prompt, image_1, image_2, image_3, image_4, aspect_ratio,
                  quality, background, codex_home, allow_refresh, timeout_seconds,
                  on_failure="fail the run", speed=SPEED_STANDARD):
        import httpx

        token = codex_auth.get_access_token(codex_home, allow_refresh=allow_refresh)
        who = codex_auth.describe(codex_home)
        account = "%s (%s)" % (who.get("email", "unknown"), who.get("plan", "?"))

        # `size` is ignored by this backend, so the shape has to be said in the prompt
        text = prompt
        line = _aspect_line(aspect_ratio)
        if line:
            text = prompt.rstrip() + "\n\n" + line
        content = [{"type": "input_text", "text": text}]
        for uri in collect_images_to_data_uris(image_1, image_2, image_3, image_4):
            content.append({"type": "input_image", "image_url": uri})

        tool = {"type": "image_generation", "model": IMAGE_MODEL,
                "output_format": "png", "partial_images": PARTIAL_IMAGES}
        # Best-effort only: this backend ignores `size` (the prompt line above is what
        # actually works). Never raise on a ratio we have no pixel mapping for.
        size = _tool_size(aspect_ratio)
        if size:
            tool["size"] = size
        if quality not in (DEFAULT, "auto"):     # Codex has no "auto" quality
            tool["quality"] = quality
        if background != DEFAULT:
            tool["background"] = background

        payload = {
            "model": HOST_MODEL,
            "store": False,
            "instructions": INSTRUCTIONS,
            "input": [{"type": "message", "role": "user", "content": content}],
            "tools": [tool],
            "tool_choice": {"type": "allowed_tools", "mode": "required",
                            "tools": [{"type": "image_generation"}]},
            "stream": True,
        }
        tier = SERVICE_TIER.get(speed)
        if tier:
            payload["service_tier"] = tier
        if speed == SPEED_FAST:
            print("[arkennemasis] codex-image speed: fast (service_tier=priority)")

        # Idle budget, not total — see codex_provider/stream.py. The old setting held a
        # dead connection for the whole timeout_seconds.
        timeout = codex_stream.timeouts()
        headers = codex_auth.request_headers(token)
        attempt = {"n": 0}

        def call():
            attempt["n"] += 1
            stats = codex_stream.StreamStats("codex-image attempt %d" % attempt["n"])
            images = {}
            seen_types = []
            image_shapes = {}
            with httpx.Client(timeout=timeout, headers=headers) as http:
                with http.stream("POST", BASE_URL + "/responses", json=payload) as resp:
                    stats.headers_at = time.time()
                    if resp.status_code >= 400:
                        resp.read()
                        body = resp.text
                        if _is_unsupported(resp.status_code, body):
                            raise CodexImageUnavailable(_UNSUPPORTED_HELP)
                        if resp.status_code in (401, 403):
                            raise RuntimeError(
                                "Codex rejected the login (HTTP %s). Run `codex login` "
                                "(or check codex_home). Body: %s"
                                % (resp.status_code, body[:300]))
                        err = RuntimeError("Codex Responses API returned HTTP %s: %s"
                                           % (resp.status_code, body[:400]))
                        # 429/5xx are "try again"; 4xx is our request being wrong.
                        err.retryable = resp.status_code in (429, 500, 502, 503, 504)
                        raise err
                    def collect(event):
                        # Keep a shape-only census of the stream. When no final image
                        # arrives this is the only evidence of WHY, and without it the
                        # answer is another billed run. Types and key names only —
                        # never payloads, so nothing large or sensitive is logged.
                        if isinstance(event, dict):
                            kind = event.get("type")
                            if isinstance(kind, str):
                                seen_types.append(kind)
                                # `output_item.done` and `completed` are where a finished
                                # image_generation_call would carry its `result`, and
                                # neither has "image" in its type — the first census
                                # missed exactly the two events that answer the question.
                                if ("image" in kind or kind.endswith("output_item.done")
                                        or kind.endswith("response.completed")):
                                    image_shapes.setdefault(kind, _shape_of(event))
                        found, is_final = _find_image_b64(event)
                        if found:
                            if is_final:
                                images["final"] = found
                            else:
                                images["partial"] = found

                    try:
                        codex_stream.consume(
                            _iter_sse(resp), stats, collect,
                            should_stop=codex_stream.interrupted)
                    except Exception:
                        print("[arkennemasis] %s" % stats.line("FAILED"))
                        raise
                    # Stops at the terminal event instead of reading to EOF.

            final_b64 = images.get("final")
            print("[arkennemasis] %s" % stats.line(
                "ok" if final_b64 else "no final image"))
            if final_b64:
                return final_b64

            # No image. Print the stream's shape ONCE (first attempt only — six
            # identical censuses is noise), so the next question is answerable from the
            # log instead of from another billed run.
            if attempt["n"] == 1:
                from collections import Counter
                census = ", ".join("%s x%d" % (name, n)
                                   for name, n in Counter(seen_types).most_common())
                print("[arkennemasis] codex-image stream census: %s" % (census or "(none)"))
                for kind, shape in image_shapes.items():
                    print("[arkennemasis]   %s -> %s" % (kind, shape))
                if not image_shapes:
                    print("[arkennemasis]   no event carried 'image' in its type — the "
                          "tool may not have been invoked at all")
            # A partial_images preview is a HALF-RENDERED frame. It used to be returned
            # as the result, which is how a low-detail render reached the video stage
            # looking like a finished still. Retry for the real one instead.
            if images.get("partial"):
                raise CodexStreamTruncated(
                    "Codex completed the response but only sent a partial_images "
                    "preview, never the final image. Previews are no longer requested "
                    "(PARTIAL_IMAGES=0), so receiving one means the backend sent it "
                    "unasked — and its preview is rendered at a lower quality than the "
                    "one requested, which is why it is not returned. Retrying.")
            raise CodexStreamTruncated(
                "Codex completed the response without sending an image. The host model "
                "answered in text instead of finishing the image tool call — check the "
                "stream census above for `output_text` events. Retrying.")

        print("[arkennemasis] %s via Codex login as %s" % (IMAGE_MODEL, account))

        def _substitute(reason):
            """Hand back the input image so a long run survives one bad scene."""
            print("[arkennemasis] *** NO IMAGE RETURNED for this scene (%s). "
                  "Substituting the REFERENCE IMAGE so the run survives - re-render "
                  "this scene on its own afterwards. ***" % reason)
            return (image_1, account)

        try:
            image_b64 = with_retry(call, log=lambda m: print("[arkennemasis] %s" % m))
        except Exception as exc:
            # `with_retry` RAISES on its final attempt rather than returning None, so the
            # `if not image_b64` branch below was unreachable for every exception-based
            # failure — which is most of them, `CodexStreamTruncated` included. That made
            # `on_failure` dead for the commonest failure there is, and in a batch a
            # single truncated stream discarded every option already rendered. Measured
            # 2026-08-21: six attempts at ~60 s each, then the whole 110-option prompt
            # died on option one.
            #
            # `fail the run` still fails the run — the raise is re-raised untouched — so
            # nothing changes for a workflow that never set this widget.
            if on_failure == "use the reference image" and image_1 is not None:
                return _substitute(type(exc).__name__)
            raise
        if not image_b64:
            # Definitive, not transient: with_retry has already exhausted its attempts.
            if on_failure == "use the reference image" and image_1 is not None:
                return _substitute("likely a moderation refusal")
            raise RuntimeError(
                "Codex returned no image. The account may not have the image tool, or "
                "the prompt was refused by moderation. Set on_failure to 'use the "
                "reference image' to keep a multi-scene run alive.")
        return (bytes_list_to_image_tensor(output_to_bytes_list(image_b64)), account)


class ArkCodexLoginStatus:
    CATEGORY = "arkennemasis/Utility"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    DESCRIPTION = ("Report the Codex login this machine will use, without generating "
                   "anything. Never outputs the token itself.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "codex_home": ("STRING", {
                    "default": "",
                    "tooltip": "Blank = CODEX_HOME, else ~/.codex.",
                }),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")          # always re-check; a token expires on its own

    def run(self, codex_home=""):
        info = codex_auth.describe(codex_home)
        if not info.get("ok"):
            return ("NOT LOGGED IN\n%s\n\nRun `codex login`."
                    % info.get("detail", ""),)
        lines = [
            "Codex login OK",
            "account          : %s" % info["email"],
            "auth file        : %s" % info["path"],
            "plan             : %s" % info["plan"],
            "account id       : %s" % ("present" if info["account_id_present"] else "MISSING"),
            "refresh token    : %s" % ("present" if info["has_refresh_token"] else "MISSING"),
            "expires in       : %s h" % info["expires_in_hours"],
            "expired          : %s" % info["expired"],
        ]
        return ("\n".join(lines),)


NODE_CLASS_MAPPINGS = {
    "ArkCodexImageGen": ArkCodexImageGen,
    "ArkCodexLoginStatus": ArkCodexLoginStatus,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArkCodexImageGen": "arkennemasis Codex Image Gen (ChatGPT login)",
    "ArkCodexLoginStatus": "arkennemasis Codex Login Status",
}

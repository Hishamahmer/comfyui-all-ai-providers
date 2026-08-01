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

from ..common.image_utils import (
    collect_images_to_data_uris,
    bytes_list_to_image_tensor,
    output_to_bytes_list,
)
from ..common.throttle import serial_lock, with_rate_limit_retry
from ..replicate_provider.settings import SETTINGS_TYPE
from . import auth as codex_auth

DEFAULT = "default"
RUN_ONE_AT_A_TIME = "one at a time"
RUN_ALL_AT_ONCE = "all at once"

BASE_URL = "https://chatgpt.com/backend-api/codex"
IMAGE_MODEL = "gpt-image-2"
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


def _find_image_b64(value):
    """Newest base64 image anywhere in an event payload (shapes vary by version)."""
    found = None
    if isinstance(value, dict):
        if value.get("type") == "image_generation_call":
            result = value.get("result")
            if isinstance(result, str) and result:
                found = result
        partial = value.get("partial_image_b64")
        if isinstance(partial, str) and partial:
            found = partial
        for child in value.values():
            nested = _find_image_b64(child)
            if nested:
                found = nested
    elif isinstance(value, list):
        for child in value:
            nested = _find_image_b64(child)
            if nested:
                found = nested
    return found


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
                "settings": (SETTINGS_TYPE, {
                    "tooltip": "Optional: the shared 'Image Gen Settings' node, so this "
                               "and the Replicate nodes follow one place. Codex uses "
                               "aspect_ratio / quality / background / run_mode / "
                               "timeout_seconds; output_format, moderation and api_token "
                               "do not apply and are ignored.",
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
                  force_rerun=False, run_mode=RUN_ONE_AT_A_TIME, settings=None):
        import asyncio

        if settings:                       # a wired Settings node overrides the widgets
            aspect_ratio = settings.get("aspect_ratio", aspect_ratio)
            quality = settings.get("quality", quality)
            background = settings.get("background", background)
            run_mode = settings.get("run_mode", run_mode)
            timeout_seconds = settings.get("timeout_seconds", timeout_seconds)

        async def go():
            return await asyncio.to_thread(
                self._blocking, prompt, image_1, image_2, image_3, image_4,
                aspect_ratio, quality, background, codex_home, allow_refresh,
                timeout_seconds,
            )

        if run_mode == RUN_ALL_AT_ONCE:
            return await go()
        async with serial_lock():
            return await go()

    def _blocking(self, prompt, image_1, image_2, image_3, image_4, aspect_ratio,
                  quality, background, codex_home, allow_refresh, timeout_seconds):
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
                "output_format": "png", "partial_images": 1}
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

        read = float(timeout_seconds) if timeout_seconds else None
        timeout = httpx.Timeout(read, connect=30.0, read=read, write=30.0, pool=30.0)
        headers = codex_auth.request_headers(token)

        def call():
            image_b64 = None
            with httpx.Client(timeout=timeout, headers=headers) as http:
                with http.stream("POST", BASE_URL + "/responses", json=payload) as resp:
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
                        raise RuntimeError("Codex Responses API returned HTTP %s: %s"
                                           % (resp.status_code, body[:400]))
                    for event in _iter_sse(resp):
                        found = _find_image_b64(event)
                        if found:
                            image_b64 = found
            return image_b64

        print("[arkennemasis] %s via Codex login as %s" % (IMAGE_MODEL, account))
        image_b64 = with_rate_limit_retry(
            call, log=lambda m: print("[arkennemasis] %s" % m))
        if not image_b64:
            raise RuntimeError(
                "Codex returned no image. The account may not have the image tool, or "
                "the prompt was refused by moderation.")
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

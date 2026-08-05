"""Text + vision through a ChatGPT/Codex login — no API key.

The counterpart to `ArkCodexImageGen`: same Responses endpoint, same OAuth credentials
`codex login` already wrote, same retry/throttle plumbing — but it returns TEXT instead
of pixels. Built for the storyboard/scene-planning step of a video pipeline, where one
call has to turn an idea into a structured JSON array of scenes.

Why this exists: the pack's only LLM was `ReplicateOpenAILLM`, which bills a Replicate
balance per call. This routes the same class of model through a ChatGPT plan instead,
so a workflow can plan AND render without an API key anywhere in it.

Model availability is account-dependent and undocumented — the backend is the Codex
CLI's, not a published API. `model` is a combo of what this machine's Codex CLI has
actually been observed using, and `model_override` takes a raw string for anything
newer than this file.
"""

import json

from ..common.image_utils import collect_images_to_data_uris
from ..common.throttle import concurrency_gate, serial_lock, with_retry
from . import auth as codex_auth
from .nodes import (
    BASE_URL,
    CodexStreamTruncated,
    _iter_sse,
    _TERMINAL_EVENTS,
    RUN_ALL_AT_ONCE,
    RUN_ONE_AT_A_TIME,
)

DEFAULT = "default"

# Observed in this machine's Codex CLI state, smartest first. `gpt-5.6-sol` is the
# model the CLI itself defaults to, so it is the safest "smartest" choice. The list is
# a convenience, not a contract — the backend is undocumented and can change under us,
# which is what `model_override` is for.
MODELS = [
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-5.6-luna",
    "gpt-5.5",
    "gpt-5.1",
    "gpt-5",
]

# Appended when json_only is on. Belt and braces: the instruction steers the model, and
# _strip_fences cleans up the ```json wrapper it sometimes adds anyway.
JSON_RULE = (
    "Return ONLY valid JSON. No prose before or after it, no explanation, and no "
    "markdown code fences."
)


def _strip_fences(text):
    """Drop a ```json ... ``` wrapper if the model added one.

    Worth doing even with the instruction above: a fenced answer is still a correct
    answer, and failing the whole graph over three backticks would be absurd.
    """
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    body = stripped[3:]
    newline = body.find("\n")
    if newline != -1 and body[:newline].strip().isalpha():
        body = body[newline + 1:]          # drop the language tag line ("json")
    if body.rstrip().endswith("```"):
        body = body.rstrip()[:-3]
    return body.strip()


def _find_output_text(value):
    """Concatenate every output_text found in a completed-response payload.

    Walks the structure rather than indexing a fixed path, for the same reason
    `_find_image_b64` does: the event shapes drift between backend versions and an
    unexpected nesting should not turn a finished answer into an empty string.
    """
    found = []
    if isinstance(value, dict):
        if value.get("type") == "output_text":
            text = value.get("text")
            if isinstance(text, str) and text:
                found.append(text)
                return found                # a leaf; do not also walk its children
        for child in value.values():
            found.extend(_find_output_text(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(_find_output_text(child))
    return found


class ArkCodexLLM:
    CATEGORY = "arkennemasis/LLM"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("text", "account")
    DESCRIPTION = ("Text + vision from the GPT-5 family using your ChatGPT/Codex login "
                   "instead of an API key. Run `codex login` first. Multiple accounts: "
                   "set codex_home per account.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "system_instructions": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "The role/rules for the model. Wire the 'System "
                               "Instructions' node here, or type it directly.",
                }),
                "image_1": ("IMAGE", {"tooltip": "Optional image to look at. Can be a "
                                                 "batch."}),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "model": (MODELS, {
                    "tooltip": "Smartest first. Availability depends on your ChatGPT "
                               "plan; if one is refused, try the next.",
                }),
                "model_override": ("STRING", {
                    "default": "",
                    "tooltip": "Raw model id, used INSTEAD of the dropdown when not "
                               "blank. For models newer than this node.",
                }),
                "reasoning_effort": ([DEFAULT, "low", "medium", "high", "xhigh",
                                     "ultra", "max"], {
                    "tooltip": "How hard the model thinks before answering. Levels above "
                               "'high' are what the Codex CLI itself uses on this "
                               "machine - 'ultra' is its workhorse, 'xhigh' and 'max' "
                               "are rarer. Use 'ultra' for story planning, where one "
                               "call decides the quality of everything downstream. "
                               "'default' omits the field entirely.",
                }),
                "json_only": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Demand bare JSON and strip any ``` fences from the "
                               "answer. Turn on when a downstream node parses the text.",
                }),
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
                    "default": 600, "min": 0, "max": 86400,
                    "tooltip": "0 = wait indefinitely. Planning a long scene list with "
                               "high effort can take minutes.",
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
                               "0 = no cap.",
                }),
            },
        }

    @classmethod
    def IS_CHANGED(cls, force_rerun=False, **kwargs):
        import time
        return time.time() if force_rerun else ""

    async def run(self, prompt, system_instructions="", image_1=None, image_2=None,
                  image_3=None, image_4=None, model=MODELS[0], model_override="",
                  reasoning_effort=DEFAULT, json_only=False, codex_home="",
                  allow_refresh=True, timeout_seconds=600, force_rerun=False,
                  run_mode=RUN_ONE_AT_A_TIME, max_concurrent=2):
        import asyncio

        async def go():
            return await asyncio.to_thread(
                self._blocking, prompt, system_instructions, image_1, image_2, image_3,
                image_4, model, model_override, reasoning_effort, json_only, codex_home,
                allow_refresh, timeout_seconds,
            )

        # 'all at once' still respects max_concurrent; 'one at a time' is always 1.
        gate = (concurrency_gate(max_concurrent) if run_mode == RUN_ALL_AT_ONCE
                else serial_lock())
        async with gate:
            return await go()

    def _blocking(self, prompt, system_instructions, image_1, image_2, image_3, image_4,
                  model, model_override, reasoning_effort, json_only, codex_home,
                  allow_refresh, timeout_seconds):
        import httpx

        token = codex_auth.get_access_token(codex_home, allow_refresh=allow_refresh)
        who = codex_auth.describe(codex_home)
        account = "%s (%s)" % (who.get("email", "unknown"), who.get("plan", "?"))
        chosen = (model_override or "").strip() or model

        instructions = (system_instructions or "").strip()
        if json_only:
            instructions = (instructions + "\n\n" + JSON_RULE).strip()

        content = [{"type": "input_text", "text": prompt}]
        for uri in collect_images_to_data_uris(image_1, image_2, image_3, image_4):
            content.append({"type": "input_image", "image_url": uri})

        payload = {
            "model": chosen,
            "store": False,
            "input": [{"type": "message", "role": "user", "content": content}],
            "stream": True,
        }
        if instructions:
            payload["instructions"] = instructions
        if reasoning_effort != DEFAULT:
            payload["reasoning"] = {"effort": reasoning_effort}

        read = float(timeout_seconds) if timeout_seconds else None
        timeout = httpx.Timeout(read, connect=30.0, read=read, write=30.0, pool=30.0)
        headers = codex_auth.request_headers(token)

        def call():
            deltas = []
            final = []
            finished = False
            with httpx.Client(timeout=timeout, headers=headers) as http:
                with http.stream("POST", BASE_URL + "/responses", json=payload) as resp:
                    if resp.status_code >= 400:
                        resp.read()
                        body = resp.text
                        if resp.status_code in (401, 403):
                            raise RuntimeError(
                                "Codex rejected the login (HTTP %s). Run `codex login` "
                                "(or check codex_home). Body: %s"
                                % (resp.status_code, body[:300]))
                        if resp.status_code == 400 and chosen in body:
                            raise RuntimeError(
                                "Codex refused the model '%s' for this account. Pick "
                                "another from the dropdown, or set model_override. "
                                "Body: %s" % (chosen, body[:300]))
                        err = RuntimeError("Codex Responses API returned HTTP %s: %s"
                                           % (resp.status_code, body[:400]))
                        # 429/5xx are "try again"; other 4xx is our request being wrong.
                        err.retryable = resp.status_code in (429, 500, 502, 503, 504)
                        raise err
                    for event in _iter_sse(resp):
                        if not isinstance(event, dict):
                            continue
                        kind = event.get("type")
                        if kind in _TERMINAL_EVENTS:
                            finished = True
                            final.extend(_find_output_text(event))
                        elif kind == "response.output_text.delta":
                            piece = event.get("delta")
                            if isinstance(piece, str):
                                deltas.append(piece)
                        elif kind == "response.output_text.done":
                            piece = event.get("text")
                            if isinstance(piece, str) and piece:
                                final.append(piece)

            # A stream cut short must never look like a short answer — same trap the
            # image node hit, where a truncated render passed as a finished image.
            if not finished:
                raise CodexStreamTruncated(
                    "Codex closed the stream before the answer finished (no completion "
                    "event).")
            # Prefer the completed payload; deltas are the fallback for event shapes
            # this file has not seen.
            return "".join(final) if final else "".join(deltas)

        print("[arkennemasis] %s via Codex login as %s" % (chosen, account))
        text = with_retry(call, log=lambda m: print("[arkennemasis] %s" % m))
        if not text or not text.strip():
            raise RuntimeError(
                "Codex returned no text. The account may not have access to '%s', or "
                "the prompt was refused." % chosen)

        if json_only:
            text = _strip_fences(text)
            try:                              # fail here, not three nodes downstream
                json.loads(text)
            except ValueError as exc:
                raise RuntimeError(
                    "json_only is on but the answer is not valid JSON (%s). First 300 "
                    "chars: %s" % (exc, text[:300]))
        return (text, account)


NODE_CLASS_MAPPINGS = {
    "ArkCodexLLM": ArkCodexLLM,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArkCodexLLM": "arkennemasis Codex LLM (ChatGPT login)",
}

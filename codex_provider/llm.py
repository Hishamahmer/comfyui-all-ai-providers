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

import time

from ..common.image_utils import collect_images_to_data_uris
from ..common.throttle import concurrency_gate, serial_lock, with_retry
from . import auth as codex_auth
from . import stream as codex_stream
from .nodes import (
    BASE_URL,
    CodexStreamTruncated,
    _iter_sse,
    _TERMINAL_EVENTS,
    RUN_ALL_AT_ONCE,
    RUN_ONE_AT_A_TIME,
    SERVICE_TIER,
    SPEED_FAST,
    SPEED_STANDARD,
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
                "batch_keys": ("STRING", {
                    "multiline": True, "default": "", "forceInput": True,
                    "tooltip": "Optional JSON array of the item keys being asked for. "
                               "When supplied, batching slices by these keys instead of "
                               "by scene number, each batch names the exact keys it must "
                               "return, and the numbering is NOT re-stamped — so the "
                               "caller can match results by key rather than by position.",
                }),
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
                # APPENDED at the end on purpose: ComfyUI maps widgets_values
                # positionally, so a new widget anywhere else would shift every later
                # value in workflows already saved.
                "batch_total": ("INT", {
                    "default": 0, "min": 0, "max": 500,
                    "tooltip": "How many items the answer must contain in total (scene "
                               "count). 0 = ask for everything in one call, the old "
                               "behaviour.\n\nSet this WITH batch_size to split a long "
                               "JSON answer across several short calls. A 16-scene plan "
                               "is ~12,000 output tokens in one response, which is slow, "
                               "goes silent long enough for the connection to be cut, "
                               "and can hit the output cap and come back truncated.",
                }),
                "batch_size": ("INT", {
                    "default": 5, "min": 1, "max": 100,
                    "tooltip": "Items per call when batch_total is set. 5 keeps each "
                               "call to a few thousand tokens, so it answers in well "
                               "under a minute and cannot be truncated. Earlier items "
                               "are passed forward so the story still joins up.",
                }),
                # APPENDED at the end, same reason as batch_total above.
                "speed": ([SPEED_STANDARD, SPEED_FAST], {
                    "tooltip": "The Codex CLI's own Speed setting. 'fast' is its "
                               "'1.5x speed, increased usage' option and sends "
                               "service_tier=priority; 'standard' sends the default "
                               "tier. Measured on this account: first token 0.75 s vs "
                               "1.00 s, whole short answer 1.7 s vs 3.2 s. It spends "
                               "your plan's quota faster, so leave it on standard for "
                               "long batch runs.",
                }),
            },
        }

    @classmethod
    def IS_CHANGED(cls, force_rerun=False, **kwargs):
        import time
        return time.time() if force_rerun else ""

    async def run(self, prompt, system_instructions="", batch_keys="",
                  image_1=None, image_2=None,
                  image_3=None, image_4=None, model=MODELS[0], model_override="",
                  reasoning_effort=DEFAULT, json_only=False, codex_home="",
                  allow_refresh=True, timeout_seconds=600, force_rerun=False,
                  run_mode=RUN_ONE_AT_A_TIME, max_concurrent=2,
                  batch_total=0, batch_size=5, speed=SPEED_STANDARD):
        import asyncio

        async def go():
            if batch_total and batch_size and json_only:
                return await asyncio.to_thread(
                    self._batched, prompt, system_instructions, batch_keys,
                    image_1, image_2,
                    image_3, image_4, model, model_override, reasoning_effort,
                    codex_home, allow_refresh, timeout_seconds,
                    int(batch_total), int(batch_size), speed,
                )
            return await asyncio.to_thread(
                self._blocking, prompt, system_instructions, image_1, image_2, image_3,
                image_4, model, model_override, reasoning_effort, json_only, codex_home,
                allow_refresh, timeout_seconds, speed,
            )

        # 'all at once' still respects max_concurrent; 'one at a time' is always 1.
        gate = (concurrency_gate(max_concurrent) if run_mode == RUN_ALL_AT_ONCE
                else serial_lock())
        async with gate:
            return await go()

    def _batched(self, prompt, system_instructions, batch_keys, image_1, image_2,
                 image_3, image_4,
                 model, model_override, reasoning_effort, codex_home, allow_refresh,
                 timeout_seconds, batch_total, batch_size, speed=SPEED_STANDARD):
        """Build a long JSON array over several short calls instead of one huge one.

        A 16-scene plan is roughly 12,000 output tokens in a single response. That is
        slow, it stays silent long enough for the connection to be cut upstream, and it
        can hit the output cap and come back as `response.incomplete` — a truncated
        array. Asking for five scenes at a time makes each call a few thousand tokens:
        fast, and structurally unable to be truncated.

        This does NOT reduce what is produced. Every scene is still written in full, to
        the same instructions. Earlier scenes are passed forward so continuity holds,
        and the scene numbers are re-stamped at the end so the array is always 1..N in
        order however the model numbered its own batch.
        """
        # Keys turn this from "write scenes 9-16 of a story" into "return an object for
        # each of THESE named items" — the difference between the model choosing which
        # items a slice covers and being told.
        keys = []
        if str(batch_keys or "").strip():
            try:
                parsed = json.loads(batch_keys)
                if isinstance(parsed, list):
                    keys = [str(k) for k in parsed if str(k).strip()]
            except ValueError:
                keys = [line.strip() for line in str(batch_keys).splitlines()
                        if line.strip()]
            if keys and len(keys) != batch_total:
                print("[arkennemasis] codex-llm: %d batch_keys but batch_total is %d — "
                      "using the keys" % (len(keys), batch_total))
                batch_total = len(keys)

        collected = []
        answered = {}                  # key -> object; the keyed path's real state
        batch = 0
        account = ""
        while len(collected) < batch_total:
            if keys:
                # Advance by WHICH keys are still outstanding, not by HOW MANY objects
                # came back. Counting assumes a short batch dropped its trailing items;
                # a batch that omits a middle one then shifts the window, so that key is
                # never asked for again AND an already-answered one is re-requested. The
                # array still ends up the right length, so nothing downstream notices —
                # it just has a hole and a duplicate, and the cell with the hole gets
                # another cell's prompt.
                mine = [k for k in keys if k not in answered][:batch_size]
                done = [k for k in keys if k in answered]
                first = keys.index(mine[0]) + 1
                last = keys.index(mine[-1]) + 1
            else:
                first = len(collected) + 1
                last = min(first + batch_size - 1, batch_total)
            batch += 1
            wanted = last - first + 1

            story_so_far = ""
            if keys:
                slice_rule = (
                    "\n\nYou are answering this request in parts. Return entries for "
                    "EXACTLY these %d items, in this order, and nothing else:\n%s\n\n"
                    "Return a JSON array of EXACTLY %d objects. Every object MUST carry "
                    "its own \"key\" field set to the item key it answers, copied "
                    "verbatim from the list above — the caller matches on it, and an "
                    "object whose key is missing or altered is applied to the wrong "
                    "item. Keep every rule above."
                    % (len(mine), "\n".join("  - %s" % k for k in mine), len(mine)))
                if done:
                    slice_rule += ("\n\nAlready answered, do not repeat: %s"
                                   % ", ".join(done[-12:]))
                print("[arkennemasis] codex-llm batch %d: items %d-%d of %d (by key)"
                      % (batch, first, last, batch_total))
                text, account = self._blocking(
                    prompt + slice_rule, system_instructions, image_1, image_2, image_3,
                    image_4, model, model_override, reasoning_effort, True, codex_home,
                    allow_refresh, timeout_seconds, speed)
                try:
                    part = json.loads(text)
                except ValueError as exc:
                    raise RuntimeError("batch %d returned invalid JSON: %s"
                                       % (batch, exc))
                if isinstance(part, dict):
                    for name in ("scenes", "output", "items", "prompts"):
                        if isinstance(part.get(name), list):
                            part = part[name]
                            break
                if not isinstance(part, list) or not part:
                    raise RuntimeError("batch %d did not return a JSON array." % batch)
                # Merge by the key each object carries, so an object that arrives out
                # of order, twice, or not at all cannot shift anything else. Keys not
                # answered simply stay outstanding and are asked for again.
                before = len(answered)
                for item in part:
                    if not isinstance(item, dict):
                        continue
                    returned = str(item.get("key") or "").strip()
                    if returned in mine and returned not in answered:
                        answered[returned] = item
                if len(answered) == before:
                    # The count-based loop used to guarantee termination; merging by key
                    # does not, so a batch that answers none of its keys is fatal rather
                    # than an infinite retry against a model that cannot comply.
                    raise RuntimeError(
                        "batch %d answered none of the %d keys it was given (first: "
                        "%s). The model is not echoing the key field."
                        % (batch, len(mine), mine[0]))
                collected = [answered[k] for k in keys if k in answered]
                if len(answered) - before < len(mine):
                    print("[arkennemasis] batch %d answered %d of %d; the rest stay "
                          "queued by key" % (batch, len(answered) - before, len(mine)))
                continue

            if collected:
                # Enough for continuity, not so much that it bloats every later call.
                previous = "\n".join(
                    "%s. %s" % (item.get("scene", index + 1),
                                str(item.get("voiceText", ""))[:200])
                    for index, item in enumerate(collected))
                story_so_far = (
                    "\n\nSCENES ALREADY WRITTEN (do not repeat them, continue straight "
                    "on from the last one):\n" + previous)

            slice_rule = (
                "\n\nYou are writing this story in parts. Produce ONLY scenes %d to %d "
                "of %d. Return a JSON array of EXACTLY %d objects and nothing else. "
                "Keep every rule above, and keep the character, wardrobe, style and "
                "colour identical to the scenes already written.%s"
                % (first, last, batch_total, wanted, story_so_far))

            print("[arkennemasis] codex-llm batch %d: scenes %d-%d of %d"
                  % (batch, first, last, batch_total))
            text, account = self._blocking(
                prompt + slice_rule, system_instructions, image_1, image_2, image_3,
                image_4, model, model_override, reasoning_effort, True, codex_home,
                allow_refresh, timeout_seconds, speed)

            try:
                part = json.loads(text)
            except ValueError as exc:
                raise RuntimeError("batch %d returned invalid JSON: %s" % (batch, exc))
            if isinstance(part, dict):                 # tolerate {"scenes": [...]}
                for key in ("scenes", "output", "items"):
                    if isinstance(part.get(key), list):
                        part = part[key]
                        break
            if not isinstance(part, list) or not part:
                raise RuntimeError(
                    "batch %d did not return a JSON array of scenes." % batch)

            collected.extend(part[:wanted])            # never overshoot the total
            if len(part) < wanted:
                print("[arkennemasis] batch %d returned %d of %d asked for; continuing"
                      % (batch, len(part), wanted))

        # Re-stamp the numbering: each batch numbers from its own viewpoint, and
        # ArkSceneList needs a clean 1..N.
        # Re-stamping is for the story path, whose consumer wants a clean 1..N. With
        # keys it would overwrite the only thing the caller can match on.
        if not keys:
            for index, item in enumerate(collected[:batch_total], start=1):
                if isinstance(item, dict):
                    item["scene"] = index
        result = json.dumps(collected[:batch_total], ensure_ascii=False)
        print("[arkennemasis] codex-llm assembled %d scenes from %d batches (%d chars)"
              % (len(collected[:batch_total]), batch, len(result)))
        return (result, account)

    def _blocking(self, prompt, system_instructions, image_1, image_2, image_3, image_4,
                  model, model_override, reasoning_effort, json_only, codex_home,
                  allow_refresh, timeout_seconds, speed=SPEED_STANDARD):
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
        tier = SERVICE_TIER.get(speed)
        if tier:
            payload["service_tier"] = tier
        if speed == SPEED_FAST:
            print("[arkennemasis] codex-llm speed: fast (service_tier=priority)")

        # The read timeout is the IDLE budget, not the total. It used to be the total
        # (600 s), which is why a response that never sent a body byte held the node for
        # ten minutes — and why one that heartbeated held it forever. A healthy stream is
        # unaffected; only silence is punished.
        total_budget = float(timeout_seconds) if timeout_seconds else None
        timeout = codex_stream.timeouts()
        headers = codex_auth.request_headers(token)
        attempt = {"n": 0}

        def call():
            attempt["n"] += 1
            stats = codex_stream.StreamStats("codex-llm attempt %d" % attempt["n"])
            deltas = []
            final = []
            with httpx.Client(timeout=timeout, headers=headers) as http:
                with http.stream("POST", BASE_URL + "/responses", json=payload) as resp:
                    stats.headers_at = time.time()
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
                    def collect(event):
                        kind = event.get("type")
                        if kind == "response.output_text.delta":
                            piece = event.get("delta")
                            if isinstance(piece, str):
                                deltas.append(piece)
                                stats.deltas += 1
                                stats.chars += len(piece)
                        elif kind == "response.output_text.done":
                            piece = event.get("text")
                            if isinstance(piece, str) and piece:
                                final.append(piece)

                    try:
                        done = codex_stream.consume(
                            _iter_sse(resp), stats, collect,
                            total_timeout=total_budget,
                            should_stop=codex_stream.interrupted)
                    except Exception:
                        print("[arkennemasis] %s" % stats.line("FAILED"))
                        raise
                    # Stop reading here. The old loop ran to EOF, so a server that kept
                    # the connection open after `response.completed` hung the node.
                    final.extend(_find_output_text(done))

            print("[arkennemasis] %s" % stats.line("ok"))
            # Prefer the completed payload; deltas are the fallback for event shapes
            # this file has not seen.
            text = "".join(final) if final else "".join(deltas)
            if not text or not text.strip():
                raise CodexStreamTruncated(
                    "Codex completed the response but sent no text.")

            # Validation belongs INSIDE the retry. It used to sit after `with_retry`, so
            # a truncated answer ended the whole run instead of causing a fresh attempt —
            # that is what turned a bad stream into "invalid JSON at column 2911" after
            # thirteen minutes.
            if json_only:
                text = _strip_fences(text)
                try:
                    json.loads(text)
                except ValueError as exc:
                    raise CodexStreamTruncated(
                        "json_only is on but the answer is not valid JSON (%s). "
                        "Retrying with a fresh request. First 300 chars: %s"
                        % (exc, text[:300]))
            return text

        print("[arkennemasis] %s via Codex login as %s" % (chosen, account))
        started = time.time()
        text = with_retry(call, log=lambda m: print("[arkennemasis] %s" % m))
        print("[arkennemasis] codex-llm done in %.1fs over %d attempt(s), %d chars"
              % (time.time() - started, attempt["n"], len(text)))
        if not text or not text.strip():
            raise RuntimeError(
                "Codex returned no text. The account may not have access to '%s', or "
                "the prompt was refused." % chosen)
        return (text, account)


NODE_CLASS_MAPPINGS = {
    "ArkCodexLLM": ArkCodexLLM,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArkCodexLLM": "arkennemasis Codex LLM (ChatGPT login)",
}

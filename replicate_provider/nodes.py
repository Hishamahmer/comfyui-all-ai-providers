"""Replicate provider nodes.

  - OpenAI LLM (Replicate)        -> text out (STRING)   [GPT-5 family]
  - OpenAI GPT-Image-2 (Replicate) -> image out (IMAGE)  [openai/gpt-image-2, generate + edit]

Token: the node's `api_token` field (optional) -> REPLICATE_API_TOKEN env var -> .env file.
Dependency: `replicate`.
"""

import time
import asyncio

from ..common.image_utils import (
    collect_images_to_data_uris,
    output_to_text,
    output_to_bytes_list,
    bytes_list_to_image_tensor,
)
from ..common.keys import resolve_key
from ..common.throttle import serial_lock, with_rate_limit_retry
from .settings import SETTINGS_TYPE

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------

GPT5_MODELS = [
    "openai/gpt-5",
    "openai/gpt-5-mini",
    "openai/gpt-5-nano",
    "openai/gpt-5-pro",
    "openai/gpt-5-structured",
    "openai/gpt-5.1",
    "openai/gpt-5.2",
    "openai/gpt-5.4",
    "openai/gpt-5.6-luna",
    "openai/gpt-5.6-terra",
    "openai/gpt-5.6-sol",
]

IMAGE_MODEL = "openai/gpt-image-2"
USER_AGENT = "comfyui-arkennemasis/1.0.0"

# sentinel meaning "don't send this parameter — let the model use its default"
DEFAULT = "default"
RUN_ONE_AT_A_TIME = "one at a time"
RUN_ALL_AT_ONCE = "all at once"

TOKEN_HELP = ("No Replicate API token. Options: paste it into the node's `api_token` field, "
              "set the REPLICATE_API_TOKEN environment variable, or add a line "
              "`REPLICATE_API_TOKEN=...` to a .env file in the custom-node folder or ComfyUI root.")

TOKEN_TOOLTIP = ("Replicate API token (optional). Blank = use REPLICATE_API_TOKEN from the "
                 "environment or a .env file.")


# ----------------------------------------------------------------------------
# Client + runner
# ----------------------------------------------------------------------------

def _get_client(api_token):
    from replicate.client import Client
    import httpx

    token = resolve_key(api_token, "REPLICATE_API_TOKEN")
    if not token:
        raise RuntimeError(TOKEN_HELP)
    # Generous per-request timeout. Overall run length is handled by polling (see
    # _run_model), so a long model run never trips a read timeout.
    timeout = httpx.Timeout(connect=15.0, read=120.0, write=120.0, pool=60.0)
    return Client(api_token=token, timeout=timeout, headers={"User-Agent": USER_AGENT})


def _run_model(client, model_ref, inputs, timeout_seconds=0, poll_interval=2.0):
    """Create a prediction and poll it to completion.

    We do NOT use replicate.run()'s blocking "Prefer: wait" mode (caps ~60s and causes
    httpx.ReadTimeout on longer models). Instead we create the prediction and poll short
    GET requests, so a run of ANY duration is fine. Supports ComfyUI's Cancel button and
    an optional overall timeout (0 = wait indefinitely).
    """
    try:
        import comfy.model_management as _mm
    except Exception:
        _mm = None

    owner, _, name = model_ref.partition("/")
    # Creating the prediction is the call providers throttle; polling is not.
    prediction = with_rate_limit_retry(
        lambda: client.models.predictions.create(model=(owner, name), input=inputs),
        log=lambda m: print(f"[arkennemasis] {m}"),
    )

    start = time.time()
    fails = 0
    terminal = ("succeeded", "failed", "canceled")
    while prediction.status not in terminal:
        if _mm is not None and _mm.processing_interrupted():
            try:
                prediction.cancel()
            except Exception:
                pass
            raise RuntimeError("Replicate run interrupted by user.")
        if timeout_seconds and (time.time() - start) > timeout_seconds:
            try:
                prediction.cancel()
            except Exception:
                pass
            raise RuntimeError(
                f"Replicate run exceeded the {timeout_seconds}s timeout (last status: "
                f"'{prediction.status}'). Raise timeout_seconds, or set it to 0 for no limit."
            )
        time.sleep(poll_interval)
        try:
            prediction.reload()
            fails = 0
        except Exception as e:  # transient network hiccup — keep polling
            fails += 1
            print(f"[arkennemasis] poll error (retrying {fails}/10): {e}")
            if fails >= 10:
                raise RuntimeError(f"Replicate polling failed repeatedly: {e}")

    if prediction.status != "succeeded":
        err = getattr(prediction, "error", None) or prediction.status
        raise RuntimeError(f"Replicate prediction {prediction.status}: {err}")

    return prediction.output


# ----------------------------------------------------------------------------
# Node: OpenAI LLM (Replicate) — GPT-5 family, text out
# ----------------------------------------------------------------------------

class ReplicateOpenAILLM:
    CATEGORY = "arkennemasis/LLM"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    DESCRIPTION = "Call an OpenAI GPT-5 family model hosted on Replicate and return its text output."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (GPT5_MODELS,),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "system_prompt": ("STRING", {"multiline": True, "default": "",
                                             "tooltip": "System instructions. Type here, or wire in a 'System Instructions' node."}),
                "image_1": ("IMAGE", {"tooltip": "Optional image for vision. Can be a batch."}),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "reasoning_effort": ([DEFAULT, "minimal", "low", "medium", "high"],),
                "verbosity": ([DEFAULT, "low", "medium", "high"],),
                "max_completion_tokens": ("INT", {"default": 0, "min": 0, "max": 128000,
                                                  "tooltip": "0 = model default."}),
                "api_token": ("STRING", {"default": "", "tooltip": TOKEN_TOOLTIP}),
                "timeout_seconds": ("INT", {"default": 0, "min": 0, "max": 86400,
                                            "tooltip": "Max seconds to wait for the model. 0 = wait indefinitely."}),
                "force_rerun": ("BOOLEAN", {"default": False,
                                            "tooltip": "Re-call the API even if inputs are unchanged."}),
                "run_mode": ([RUN_ONE_AT_A_TIME, RUN_ALL_AT_ONCE], {
                    "tooltip": "ComfyUI runs async nodes concurrently. 'one at a time' "
                               "serialises every arkennemasis API node in the graph — use "
                               "it when the provider throttles bursts (Replicate allows a "
                               "burst of 1 under $5 credit). 'all at once' is faster when "
                               "your rate limit allows it.",
                }),
            },
        }

    @classmethod
    def IS_CHANGED(cls, force_rerun=False, **kwargs):
        return time.time() if force_rerun else ""

    async def run(self, model, prompt, system_prompt="", image_1=None, image_2=None,
                  image_3=None, image_4=None, reasoning_effort=DEFAULT, verbosity=DEFAULT,
                  max_completion_tokens=0, api_token="", timeout_seconds=0, force_rerun=False,
                  run_mode=RUN_ONE_AT_A_TIME):
        # Offload the (long, blocking) network + poll to a worker thread so ComfyUI's
        # event loop stays responsive during the whole run.
        async def _go():
            return await asyncio.to_thread(
                self._blocking, model, prompt, system_prompt, image_1, image_2, image_3,
                image_4, reasoning_effort, verbosity, max_completion_tokens, api_token,
                timeout_seconds,
            )
        if run_mode == RUN_ALL_AT_ONCE:
            return await _go()
        async with serial_lock():          # shares the queue with the image node
            return await _go()

    def _blocking(self, model, prompt, system_prompt, image_1, image_2, image_3, image_4,
                  reasoning_effort, verbosity, max_completion_tokens, api_token, timeout_seconds):
        client = _get_client(api_token)

        inputs = {"prompt": prompt}
        if system_prompt.strip():
            inputs["system_prompt"] = system_prompt
        image_uris = collect_images_to_data_uris(image_1, image_2, image_3, image_4)
        if image_uris:
            inputs["image_input"] = image_uris
        if reasoning_effort != DEFAULT:
            inputs["reasoning_effort"] = reasoning_effort
        if verbosity != DEFAULT:
            inputs["verbosity"] = verbosity
        if max_completion_tokens and max_completion_tokens > 0:
            inputs["max_completion_tokens"] = int(max_completion_tokens)

        print(f"[arkennemasis] running {model}")
        output = _run_model(client, model, inputs, timeout_seconds)
        return (output_to_text(output),)


# ----------------------------------------------------------------------------
# Node: OpenAI GPT-Image-2 (Replicate) — generate + edit, image out
# ----------------------------------------------------------------------------

class ReplicateOpenAIGPTImage2:
    CATEGORY = "arkennemasis/Image Gen"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    DESCRIPTION = "Generate or edit images with openai/gpt-image-2 on Replicate. Optional input images enable editing."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "image_1": ("IMAGE", {"tooltip": "Optional image to edit/transform. Omit for text-to-image. Can be a batch."}),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "number_of_images": ("INT", {"default": 1, "min": 1, "max": 10}),
                # The full set the live Replicate schema accepts - ratios AND explicit sizes.
                # DEFAULT is our own sentinel for "do not send"; "auto" is a real value.
                "aspect_ratio": ([DEFAULT, "1:1", "3:2", "2:3", "4:3", "3:4", "16:9", "9:16", "auto", "1024x1024", "1536x1024", "1024x1536", "1536x1152", "1152x1536", "2048x2048", "2048x1152", "1152x2048", "3840x2160", "2160x3840"],),
                "quality": ([DEFAULT, "low", "medium", "high", "auto"],),
                "background": ([DEFAULT, "auto", "transparent", "opaque"],),
                "output_format": ([DEFAULT, "png", "jpeg", "webp"],),
                "moderation": ([DEFAULT, "auto", "low"],),
                "api_token": ("STRING", {"default": "", "tooltip": TOKEN_TOOLTIP}),
                "timeout_seconds": ("INT", {"default": 0, "min": 0, "max": 86400,
                                            "tooltip": "Max seconds to wait for the model. 0 = wait indefinitely."}),
                "force_rerun": ("BOOLEAN", {"default": False,
                                            "tooltip": "Re-call the API even if inputs are unchanged."}),
                "run_mode": ([RUN_ONE_AT_A_TIME, RUN_ALL_AT_ONCE], {
                    "tooltip": "ComfyUI runs async nodes concurrently. 'one at a time' "
                               "serialises every arkennemasis API node in the graph — use "
                               "it when the provider throttles bursts (Replicate allows a "
                               "burst of 1 under $5 credit). 'all at once' is faster when "
                               "your rate limit allows it.",
                }),
                "settings": (SETTINGS_TYPE, {
                    "tooltip": "Optional: wire one 'Image Gen Settings (shared)' node in "
                               "here to drive this and every other Image Gen node from a "
                               "single place. Any field it leaves on \"use node's own\" "
                               "falls back to the widgets above.",
                }),
            },
        }

    @classmethod
    def IS_CHANGED(cls, force_rerun=False, **kwargs):
        return time.time() if force_rerun else ""

    async def run(self, prompt, image_1=None, image_2=None, image_3=None, image_4=None,
                  number_of_images=1, aspect_ratio=DEFAULT, quality=DEFAULT, background=DEFAULT,
                  output_format=DEFAULT, moderation=DEFAULT, api_token="", timeout_seconds=0,
                  force_rerun=False, run_mode=RUN_ONE_AT_A_TIME, settings=None):
        # A wired Settings node overrides the widgets above, field by field.
        if settings:
            number_of_images = settings.get("number_of_images", number_of_images)
            aspect_ratio = settings.get("aspect_ratio", aspect_ratio)
            quality = settings.get("quality", quality)
            background = settings.get("background", background)
            output_format = settings.get("output_format", output_format)
            moderation = settings.get("moderation", moderation)
            timeout_seconds = settings.get("timeout_seconds", timeout_seconds)
            api_token = settings.get("api_token", api_token)
            run_mode = settings.get("run_mode", run_mode)

        # Offload blocking network + poll + image download to a worker thread.
        async def _go():
            return await asyncio.to_thread(
                self._blocking, prompt, image_1, image_2, image_3, image_4, number_of_images,
                aspect_ratio, quality, background, output_format, moderation, api_token,
                timeout_seconds,
            )
        if run_mode == RUN_ALL_AT_ONCE:
            return await _go()
        async with serial_lock():          # one arkennemasis API call at a time
            return await _go()

    def _blocking(self, prompt, image_1, image_2, image_3, image_4, number_of_images,
                  aspect_ratio, quality, background, output_format, moderation, api_token,
                  timeout_seconds):
        client = _get_client(api_token)

        inputs = {"prompt": prompt, "number_of_images": int(number_of_images)}
        image_uris = collect_images_to_data_uris(image_1, image_2, image_3, image_4)
        if image_uris:
            inputs["input_images"] = image_uris
        if aspect_ratio != DEFAULT:
            inputs["aspect_ratio"] = aspect_ratio
        if quality != DEFAULT:
            inputs["quality"] = quality
        if background != DEFAULT:
            inputs["background"] = background
        if output_format != DEFAULT:
            inputs["output_format"] = output_format
        if moderation != DEFAULT:
            inputs["moderation"] = moderation

        print(f"[arkennemasis] running {IMAGE_MODEL}")
        output = _run_model(client, IMAGE_MODEL, inputs, timeout_seconds)
        image = bytes_list_to_image_tensor(output_to_bytes_list(output))
        return (image,)


from .settings import (
    NODE_CLASS_MAPPINGS as _SET_CLS, NODE_DISPLAY_NAME_MAPPINGS as _SET_DISP,
)

NODE_CLASS_MAPPINGS = {
    "ReplicateOpenAILLM": ReplicateOpenAILLM,
    "ReplicateOpenAIGPTImage2": ReplicateOpenAIGPTImage2,
    **_SET_CLS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ReplicateOpenAILLM": "arkennemasis Replicate LLM (OpenAI GPT-5)",
    "ReplicateOpenAIGPTImage2": "arkennemasis Replicate Image Gen (GPT-Image-2)",
    **_SET_DISP,
}

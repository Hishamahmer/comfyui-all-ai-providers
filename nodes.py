"""
ComfyUI custom nodes: call OpenAI models hosted on Replicate.

Two nodes only (by design — nothing else is exposed here):
  1. OpenAI LLM (Replicate)        -> text out (STRING)   [GPT-5 family]
  2. OpenAI GPT-Image-2 (Replicate) -> image out (IMAGE)  [openai/gpt-image-2, generate + edit]

Auth: each node has an `api_token` field. If left blank it falls back to the
REPLICATE_API_TOKEN environment variable.
"""

import io
import os
import time
import base64
import asyncio

import numpy as np
import torch
from PIL import Image

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------

# GPT-5 family text models on Replicate (ids under the `openai` org).
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

USER_AGENT = "replicate-openai-comfyui/1.0.0"

# sentinel meaning "don't send this parameter — let the model use its default"
DEFAULT = "default"


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _get_client(api_token):
    """Build a Replicate client. Prefer the node's api_token, else env var."""
    from replicate.client import Client
    import httpx

    token = (api_token or "").strip() or os.environ.get("REPLICATE_API_TOKEN", "").strip()
    if not token:
        raise RuntimeError(
            "No Replicate API token. Paste one into the node's `api_token` field, "
            "or set the REPLICATE_API_TOKEN environment variable."
        )
    # Generous per-request timeout. The OVERALL run can take much longer than any single
    # request — we don't block on one request; we poll (see _run_model), so a long model
    # run never trips a read timeout.
    timeout = httpx.Timeout(connect=15.0, read=120.0, write=120.0, pool=60.0)
    return Client(api_token=token, timeout=timeout, headers={"User-Agent": USER_AGENT})


def _run_model(client, model_ref, inputs, timeout_seconds=0, poll_interval=2.0):
    """Create a prediction and poll it to completion.

    Unlike replicate.run()'s default, this does NOT ask the server to hold the HTTP
    connection open until the model finishes (that "Prefer: wait" mode caps at ~60s and
    causes httpx.ReadTimeout on longer models). Instead we create the prediction and poll
    short GET requests, so a run of ANY duration is fine.

    Also supports ComfyUI's Cancel button and an optional overall timeout.
    `timeout_seconds` = 0 means wait indefinitely.
    """
    try:
        import comfy.model_management as _mm
    except Exception:
        _mm = None

    owner, _, name = model_ref.partition("/")
    prediction = client.models.predictions.create(model=(owner, name), input=inputs)

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
            print(f"[replicate_openai] poll error (retrying {fails}/10): {e}")
            if fails >= 10:
                raise RuntimeError(f"Replicate polling failed repeatedly: {e}")

    if prediction.status != "succeeded":
        err = getattr(prediction, "error", None) or prediction.status
        raise RuntimeError(f"Replicate prediction {prediction.status}: {err}")

    return prediction.output


def _tensor_to_pil(image):
    """A single ComfyUI IMAGE tensor (H,W,C) float 0..1 -> PIL.Image."""
    arr = (image.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[2] == 1:
        return Image.fromarray(arr[:, :, 0], mode="L")
    return Image.fromarray(arr)


def _image_batch_to_data_uris(images):
    """A batched ComfyUI IMAGE tensor (B,H,W,C) -> list of PNG data URIs."""
    uris = []
    for i in range(images.shape[0]):
        pil = _tensor_to_pil(images[i]).convert("RGB")
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        uris.append(f"data:image/png;base64,{b64}")
    return uris


def _collect_images_to_data_uris(*images):
    """Several optional IMAGE sockets (each may be a batch) -> flat list of data URIs."""
    uris = []
    for img in images:
        if img is not None:
            uris.extend(_image_batch_to_data_uris(img))
    return uris


def _output_to_bytes_list(output):
    """Normalise a Replicate image output into a list of raw image bytes.

    Handles: FileOutput objects (.read()), http(s) URLs, data: URIs, raw bytes,
    and lists of any of those.
    """
    items = output if isinstance(output, list) else [output]
    out = []
    for item in items:
        if item is None:
            continue
        if hasattr(item, "read"):  # replicate FileOutput / file-like
            out.append(item.read())
        elif isinstance(item, (bytes, bytearray)):
            out.append(bytes(item))
        elif isinstance(item, str):
            if item.startswith("data:"):
                out.append(base64.b64decode(item.split(",", 1)[1]))
            elif item.startswith("http://") or item.startswith("https://"):
                import urllib.request  # stdlib — no extra dependency

                with urllib.request.urlopen(item, timeout=300) as resp:
                    out.append(resp.read())
            else:  # assume raw base64
                out.append(base64.b64decode(item))
        else:
            raise RuntimeError(f"Unexpected image output element: {type(item)}")
    return out


def _bytes_list_to_image_tensor(bytes_list):
    """List of image bytes -> a single batched ComfyUI IMAGE tensor (B,H,W,C)."""
    tensors = []
    for data in bytes_list:
        img = Image.open(io.BytesIO(data))
        if img.mode != "RGB":
            img = img.convert("RGB")
        arr = np.asarray(img).astype(np.float32) / 255.0  # (H,W,3)
        tensors.append(torch.from_numpy(arr)[None, ...])   # (1,H,W,3)
    if not tensors:
        raise RuntimeError("Model returned no image output.")
    if len(tensors) == 1:
        return tensors[0]
    # pad-free concat only works if sizes match; gpt-image returns uniform sizes
    return torch.cat(tensors, dim=0)


def _output_to_text(output):
    """Normalise a Replicate text output (str / list / streaming iterator) -> str."""
    if isinstance(output, str):
        return output.strip()
    if isinstance(output, (list, tuple)):
        return "".join(str(x) for x in output).strip()
    try:
        return "".join(str(x) for x in output).strip()  # iterator/stream
    except TypeError:
        return str(output).strip()


# ----------------------------------------------------------------------------
# Node 1: OpenAI LLM (Replicate) — GPT-5 family, text out
# ----------------------------------------------------------------------------

class ReplicateOpenAILLM:
    CATEGORY = "Replicate/OpenAI"
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
                "api_token": ("STRING", {"default": "",
                                         "tooltip": "Replicate API token. Blank = use REPLICATE_API_TOKEN env var."}),
                "timeout_seconds": ("INT", {"default": 0, "min": 0, "max": 86400,
                                            "tooltip": "Max seconds to wait for the model. 0 = wait indefinitely."}),
                "force_rerun": ("BOOLEAN", {"default": False,
                                            "tooltip": "Re-call the API even if inputs are unchanged."}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, force_rerun=False, **kwargs):
        return time.time() if force_rerun else ""

    async def run(self, model, prompt, system_prompt="", image_1=None, image_2=None,
                  image_3=None, image_4=None, reasoning_effort=DEFAULT, verbosity=DEFAULT,
                  max_completion_tokens=0, api_token="", timeout_seconds=0, force_rerun=False):
        # Offload the (long, blocking) network + poll to a worker thread so ComfyUI's
        # event loop stays responsive during the whole run.
        return await asyncio.to_thread(
            self._blocking, model, prompt, system_prompt, image_1, image_2, image_3,
            image_4, reasoning_effort, verbosity, max_completion_tokens, api_token,
            timeout_seconds,
        )

    def _blocking(self, model, prompt, system_prompt, image_1, image_2, image_3, image_4,
                  reasoning_effort, verbosity, max_completion_tokens, api_token, timeout_seconds):
        client = _get_client(api_token)

        inputs = {"prompt": prompt}
        if system_prompt.strip():
            inputs["system_prompt"] = system_prompt
        image_uris = _collect_images_to_data_uris(image_1, image_2, image_3, image_4)
        if image_uris:
            inputs["image_input"] = image_uris
        if reasoning_effort != DEFAULT:
            inputs["reasoning_effort"] = reasoning_effort
        if verbosity != DEFAULT:
            inputs["verbosity"] = verbosity
        if max_completion_tokens and max_completion_tokens > 0:
            inputs["max_completion_tokens"] = int(max_completion_tokens)

        print(f"[replicate_openai] running {model}")
        output = _run_model(client, model, inputs, timeout_seconds)
        return (_output_to_text(output),)


# ----------------------------------------------------------------------------
# Node 2: OpenAI GPT-Image-2 (Replicate) — generate + edit, image out
# ----------------------------------------------------------------------------

class ReplicateOpenAIGPTImage2:
    CATEGORY = "Replicate/OpenAI"
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
                "aspect_ratio": ([DEFAULT, "1:1", "3:2", "2:3"],),
                "quality": ([DEFAULT, "low", "medium", "high", "auto"],),
                "background": ([DEFAULT, "auto", "transparent", "opaque"],),
                "output_format": ([DEFAULT, "png", "jpeg", "webp"],),
                "moderation": ([DEFAULT, "auto", "low"],),
                "api_token": ("STRING", {"default": "",
                                         "tooltip": "Replicate API token. Blank = use REPLICATE_API_TOKEN env var."}),
                "timeout_seconds": ("INT", {"default": 0, "min": 0, "max": 86400,
                                            "tooltip": "Max seconds to wait for the model. 0 = wait indefinitely."}),
                "force_rerun": ("BOOLEAN", {"default": False,
                                            "tooltip": "Re-call the API even if inputs are unchanged."}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, force_rerun=False, **kwargs):
        return time.time() if force_rerun else ""

    async def run(self, prompt, image_1=None, image_2=None, image_3=None, image_4=None,
                  number_of_images=1, aspect_ratio=DEFAULT, quality=DEFAULT, background=DEFAULT,
                  output_format=DEFAULT, moderation=DEFAULT, api_token="", timeout_seconds=0,
                  force_rerun=False):
        # Offload blocking network + poll + image download to a worker thread.
        return await asyncio.to_thread(
            self._blocking, prompt, image_1, image_2, image_3, image_4, number_of_images,
            aspect_ratio, quality, background, output_format, moderation, api_token,
            timeout_seconds,
        )

    def _blocking(self, prompt, image_1, image_2, image_3, image_4, number_of_images,
                  aspect_ratio, quality, background, output_format, moderation, api_token,
                  timeout_seconds):
        client = _get_client(api_token)

        inputs = {"prompt": prompt, "number_of_images": int(number_of_images)}
        image_uris = _collect_images_to_data_uris(image_1, image_2, image_3, image_4)
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

        print(f"[replicate_openai] running {IMAGE_MODEL}")
        output = _run_model(client, IMAGE_MODEL, inputs, timeout_seconds)
        image = _bytes_list_to_image_tensor(_output_to_bytes_list(output))
        return (image,)


# ----------------------------------------------------------------------------
# Node 3: System Instructions — reusable system prompt, text out
# ----------------------------------------------------------------------------

class SystemInstructions:
    CATEGORY = "Replicate/OpenAI"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("system_instructions",)
    DESCRIPTION = ("Hold reusable system instructions and output them as text. "
                   "Wire the output into the LLM node's `system_prompt`.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "system_instructions": ("STRING", {
                    "multiline": True,
                    "default": "You are a helpful assistant.",
                }),
            },
        }

    def run(self, system_instructions):
        return (system_instructions,)


# ----------------------------------------------------------------------------
# Registration
# ----------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "ReplicateOpenAILLM": ReplicateOpenAILLM,
    "ReplicateOpenAIGPTImage2": ReplicateOpenAIGPTImage2,
    "ReplicateOpenAISystemInstructions": SystemInstructions,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ReplicateOpenAILLM": "OpenAI LLM (Replicate)",
    "ReplicateOpenAIGPTImage2": "OpenAI GPT-Image-2 (Replicate)",
    "ReplicateOpenAISystemInstructions": "System Instructions (Replicate/OpenAI)",
}

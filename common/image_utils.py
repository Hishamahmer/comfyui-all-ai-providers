"""Provider-neutral image/text conversion helpers.

ComfyUI IMAGE tensors <-> data URIs / raw bytes, and model output normalisation.
Reused by every provider that sends/receives images or text.
"""

import io
import base64

import numpy as np
import torch
from PIL import Image


def tensor_to_pil(image):
    """A single ComfyUI IMAGE tensor (H,W,C) float 0..1 -> PIL.Image."""
    arr = (image.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[2] == 1:
        return Image.fromarray(arr[:, :, 0], mode="L")
    return Image.fromarray(arr)


def image_batch_to_data_uris(images):
    """A batched ComfyUI IMAGE tensor (B,H,W,C) -> list of PNG data URIs."""
    uris = []
    for i in range(images.shape[0]):
        pil = tensor_to_pil(images[i]).convert("RGB")
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        uris.append(f"data:image/png;base64,{b64}")
    return uris


def collect_images_to_data_uris(*images):
    """Several optional IMAGE sockets (each may be a batch) -> flat list of data URIs."""
    uris = []
    for img in images:
        if img is not None:
            uris.extend(image_batch_to_data_uris(img))
    return uris


def output_to_bytes_list(output):
    """Normalise an image output into a list of raw image bytes.

    Handles FileOutput objects (.read()), http(s) URLs, data: URIs, raw bytes,
    and lists of any of those.
    """
    items = output if isinstance(output, list) else [output]
    out = []
    for item in items:
        if item is None:
            continue
        if hasattr(item, "read"):  # file-like / SDK FileOutput
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


def bytes_list_to_image_tensor(bytes_list):
    """List of image bytes -> a single batched ComfyUI IMAGE tensor (B,H,W,C)."""
    tensors = []
    for data in bytes_list:
        img = Image.open(io.BytesIO(data))
        if img.mode != "RGB":
            img = img.convert("RGB")
        arr = np.asarray(img).astype(np.float32) / 255.0   # (H,W,3)
        tensors.append(torch.from_numpy(arr)[None, ...])    # (1,H,W,3)
    if not tensors:
        raise RuntimeError("Model returned no image output.")
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim=0)


def output_to_text(output):
    """Normalise a text output (str / list / streaming iterator) -> str."""
    if isinstance(output, str):
        return output.strip()
    if isinstance(output, (list, tuple)):
        return "".join(str(x) for x in output).strip()
    try:
        return "".join(str(x) for x in output).strip()  # iterator/stream
    except TypeError:
        return str(output).strip()

"""Color tools — provider-neutral nodes.

  - Color Picker:      pick one color (pin on the image / screen eyedropper / manual hex)
                       -> hex, rgb, prompt-ready text, and a solid swatch IMAGE.
  - Palette Analyzer:  top-N dominant colors of an image (tiny built-in k-means, no sklearn)
                       -> numbered text list, compact hex list, and a palette-strip IMAGE.

Both are pure local compute (fast, deterministic) — they feed any LLM node (text outputs)
and any image node (IMAGE outputs).
"""

import os
import numpy as np
import torch


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def _parse_hex(s):
    v = (s or "").strip().lstrip("#")
    if len(v) == 3:
        v = "".join(c * 2 for c in v)
    if len(v) != 6:
        raise RuntimeError(f"Invalid hex color {s!r} — use #RGB or #RRGGBB.")
    try:
        return tuple(int(v[i:i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        raise RuntimeError(f"Invalid hex color {s!r} — use #RGB or #RRGGBB.")


def _fmt(r, g, b):
    hexs = "#{:02X}{:02X}{:02X}".format(r, g, b)
    rgbs = f"{r}, {g}, {b}"
    return hexs, rgbs, f"HEX {hexs} (RGB {rgbs})"


def _solid_swatch(r, g, b, size):
    arr = np.empty((size, size, 3), dtype=np.float32)
    arr[..., 0] = r / 255.0
    arr[..., 1] = g / 255.0
    arr[..., 2] = b / 255.0
    return torch.from_numpy(arr)[None, ...]


def _save_temp_preview(arr01):
    """Save a downscaled copy of the image to ComfyUI's temp dir for the front-end
    preview. Returns ui info, or None outside a ComfyUI runtime."""
    try:
        import folder_paths
        import random
        import string
        from PIL import Image as PILImage

        pil = PILImage.fromarray((arr01 * 255.0).clip(0, 255).astype(np.uint8))
        pil.thumbnail((1024, 1024))
        d = folder_paths.get_temp_directory()
        os.makedirs(d, exist_ok=True)
        fname = "aap_pick_" + "".join(random.choices(string.ascii_lowercase + string.digits, k=8)) + ".png"
        pil.save(os.path.join(d, fname))
        return [{"filename": fname, "subfolder": "", "type": "temp"}]
    except Exception:
        return None


# ----------------------------------------------------------------------------
# Color Picker
# ----------------------------------------------------------------------------

class AIColorPicker:
    CATEGORY = "arkennemasis/Utility"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("hex", "rgb", "prompt_text", "swatch")
    DESCRIPTION = ("Pick a color: drag the pin on the image preview, use the screen "
                   "eyedropper button, or type a hex. Outputs hex/RGB text for LLM "
                   "prompts and a solid swatch IMAGE for image nodes.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source": (["pin_on_image", "manual_hex"],
                           {"tooltip": "pin_on_image samples the connected image at the pin; "
                                       "manual_hex uses the hex field (set by typing or the eyedropper)."}),
                "manual_hex": ("STRING", {"default": "#FFFFFF",
                                          "tooltip": "Used when source = manual_hex. The screen eyedropper writes here."}),
                "pick_x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.001,
                                     "tooltip": "Pin position (0..1 of image width). Drag the pin instead of typing."}),
                "pick_y": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.001,
                                     "tooltip": "Pin position (0..1 of image height)."}),
                "sample_size": (["1x1", "3x3", "5x5"], {"default": "3x3",
                                "tooltip": "Average over a small window — steadier color on photos."}),
                "swatch_size": ("INT", {"default": 512, "min": 8, "max": 2048,
                                        "tooltip": "Size of the solid-color swatch IMAGE output."}),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "Image to pick from (needed for pin_on_image)."}),
            },
        }

    def run(self, source, manual_hex, pick_x, pick_y, sample_size, swatch_size, image=None):
        if source == "manual_hex":
            r, g, b = _parse_hex(manual_hex)
            arr = None
        else:
            if image is None:
                raise RuntimeError("source = pin_on_image needs an image connected "
                                   "(or switch source to manual_hex).")
            arr = image[0].cpu().numpy()[..., :3]  # (H,W,3) 0..1
            h, w = arr.shape[:2]
            px = int(round(float(pick_x) * (w - 1)))
            py = int(round(float(pick_y) * (h - 1)))
            k = {"1x1": 0, "3x3": 1, "5x5": 2}[sample_size]
            region = arr[max(0, py - k):min(h, py + k + 1),
                         max(0, px - k):min(w, px + k + 1)]
            mean = region.reshape(-1, 3).mean(axis=0)
            r, g, b = [int(round(float(c) * 255.0)) for c in mean]

        hexs, rgbs, text = _fmt(r, g, b)
        swatch = _solid_swatch(r, g, b, int(swatch_size))
        print(f"[arkennemasis] color picked: {text}")

        ui = {}
        if image is not None:
            if arr is None:
                arr = image[0].cpu().numpy()[..., :3]
            prev = _save_temp_preview(arr)
            if prev:
                ui["aap_preview"] = prev
        return {"ui": ui, "result": (hexs, rgbs, text, swatch)}


# ----------------------------------------------------------------------------
# Palette Analyzer
# ----------------------------------------------------------------------------

def _kmeans(pixels, k, iters=12, seed=0):
    """Tiny deterministic k-means over (N,3) float pixels in 0..255."""
    rng = np.random.default_rng(seed)
    n = pixels.shape[0]
    k = min(k, n)
    centers = pixels[rng.choice(n, size=k, replace=False)].astype(np.float32)
    labels = np.zeros(n, dtype=np.int64)
    for _ in range(iters):
        d2 = ((pixels[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
        labels = d2.argmin(1)
        for j in range(k):
            m = labels == j
            if m.any():
                centers[j] = pixels[m].mean(0)
    counts = np.bincount(labels, minlength=k)
    order = np.argsort(-counts)
    keep = counts[order] > 0
    return centers[order][keep], counts[order][keep] / counts.sum()


class AIPaletteAnalyzer:
    CATEGORY = "arkennemasis/Utility"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("palette_text", "hex_list", "palette_image")
    DESCRIPTION = ("Extract the top-N dominant colors of an image. Outputs a numbered "
                   "text list (for LLM prompts), a compact hex list, and a palette-strip IMAGE.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "num_colors": ("INT", {"default": 5, "min": 2, "max": 12}),
            },
        }

    def run(self, image, num_colors):
        arr = image[0].cpu().numpy()[..., :3].reshape(-1, 3) * 255.0
        step = max(1, arr.shape[0] // 4096)          # subsample for speed, deterministic
        centers, shares = _kmeans(arr[::step][:4096], int(num_colors))

        lines, hexes, blocks = [], [], []
        for i, (c, s) in enumerate(zip(centers, shares)):
            r, g, b = [int(round(float(v))) for v in np.clip(c, 0, 255)]
            hexs, rgbs, _ = _fmt(r, g, b)
            lines.append(f"{i + 1}. {hexs} (RGB {rgbs}) — {s * 100:.0f}%")
            hexes.append(hexs)
            block = np.empty((256, 256, 3), dtype=np.float32)
            block[..., 0], block[..., 1], block[..., 2] = r / 255.0, g / 255.0, b / 255.0
            blocks.append(block)

        strip = torch.from_numpy(np.concatenate(blocks, axis=1))[None, ...]
        palette_text = "\n".join(lines)
        print(f"[arkennemasis] palette: {', '.join(hexes)}")
        return (palette_text, ", ".join(hexes), strip)


NODE_CLASS_MAPPINGS = {
    "AIColorPicker": AIColorPicker,
    "AIPaletteAnalyzer": AIPaletteAnalyzer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AIColorPicker": "arkennemasis Color Picker",
    "AIPaletteAnalyzer": "arkennemasis Palette Analyzer",
}

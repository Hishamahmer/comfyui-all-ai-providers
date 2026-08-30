"""Pin an image model's output shape to the shape of the image it is editing.

`aspect_ratio` on the Codex node is a dropdown of fixed ratios plus `auto`, and `auto`
sends **nothing**: no `size` on the tool and no aspect sentence in the prompt. For a
generate-from-nothing workflow that is fine. For an EDIT — a thumbnail, a portrait — it
means the one thing that must not change is the one thing nothing is asking for. The
result comes back reshaped and the whole preservation promise is broken by omission.

This node measures the actual image and emits a settings bundle naming its ratio, which
the image node reads in place of its own widget. So the shape follows the upload instead
of being guessed or hard-coded, and it is FIXED rather than `auto`.

**Why a settings bundle and not a wire into the dropdown**: ComfyUI rejects `STRING` ->
`COMBO` links. `ARK_IMAGE_SETTINGS` is the typed dict that exists precisely to get around
that, and every arkennemasis Image Gen node already has a `settings` socket that overrides
its own widgets. Nothing about the image node changes.

Chain a shared `ArkImageGenSettings` in through `base` to set quality and the rest in one
place; this node only ever replaces `aspect_ratio`.
"""

from __future__ import annotations

from ..replicate_provider.settings import SETTINGS_TYPE

# The ratios gpt-image-2 actually renders, as the same vocabulary the Codex node's SIZES
# table uses. Kept here as (label, width/height) so the nearest match is a number
# comparison rather than a lookup that can miss.
RATIOS = (
    ("1:1", 1.0),
    ("3:2", 3 / 2),
    ("2:3", 2 / 3),
    ("4:3", 4 / 3),
    ("3:4", 3 / 4),
    ("16:9", 16 / 9),
    ("9:16", 9 / 16),
)

NEAREST = "nearest supported ratio"
EXACT = "exact pixels of the input"
MODES = [NEAREST, EXACT]


def nearest_ratio(width, height):
    """The supported ratio closest to this image, and how far off it is.

    Compared in LOG space. Linearly, 16:9 (1.778) and 3:2 (1.5) sit 0.278 apart while
    9:16 (0.5625) and 2:3 (0.667) sit 0.104 apart — so a linear metric quietly treats
    portrait mistakes as less serious than landscape ones. A ratio and its reciprocal
    should be equally distant from a square, which is exactly what a log does.
    """
    import math

    if not width or not height:
        return "1:1", 0.0
    target = math.log(width / height)
    best, best_gap = RATIOS[0][0], None
    for label, value in RATIOS:
        gap = abs(math.log(value) - target)
        if best_gap is None or gap < best_gap:
            best, best_gap = label, gap
    return best, best_gap or 0.0


class ArkMatchAspect:
    CATEGORY = "arkennemasis/Image Gen"
    FUNCTION = "run"
    RETURN_TYPES = (SETTINGS_TYPE, "STRING", "INT", "INT")
    RETURN_NAMES = ("settings", "aspect_ratio", "width", "height")
    DESCRIPTION = (
        "Measure an image and emit a settings bundle pinning the generator's output to "
        "that shape. Wire `settings` into the image node's `settings` socket — its own "
        "aspect_ratio widget is then ignored, so the output follows the upload instead "
        "of a fixed guess or `auto`."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "The image being edited. Its dimensions decide the "
                               "output shape.",
                }),
                "mode": (MODES, {
                    "tooltip": "'nearest supported ratio' names one of the model's own "
                               "canvases and is what actually renders. 'exact pixels' "
                               "asks for the input's literal size — more precise in the "
                               "prompt, but the model may not offer that canvas.",
                }),
            },
            "optional": {
                "base": (SETTINGS_TYPE, {
                    "tooltip": "An upstream ArkImageGenSettings to extend. Everything in "
                               "it is passed through; only aspect_ratio is replaced.",
                }),
            },
        }

    def run(self, image, mode=NEAREST, base=None):
        # IMAGE is (B, H, W, C), so height comes before width — getting this backwards
        # turns every landscape thumbnail into a portrait request.
        if image is None or len(image.shape) < 3:
            raise RuntimeError("ArkMatchAspect needs an IMAGE to measure.")
        height, width = int(image.shape[1]), int(image.shape[2])

        label, gap = nearest_ratio(width, height)
        value = ("%dx%d" % (width, height)) if mode == EXACT else label

        settings = dict(base) if isinstance(base, dict) else {}
        settings["aspect_ratio"] = value

        note = ""
        if mode == NEAREST and gap > 0.06:
            # ~6% in log space. The input is not close to anything the model renders, so
            # SOMETHING will be reshaped — say so here rather than letting it look like
            # the generator ignored the instruction.
            note = "  (no close match — the output will be reshaped)"
        print("[arkennemasis] match aspect: %dx%d -> %s%s" % (width, height, value, note))
        return (settings, value, width, height)


NODE_CLASS_MAPPINGS = {
    "ArkMatchAspect": ArkMatchAspect,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArkMatchAspect": "arkennemasis Match Aspect (output follows the input)",
}

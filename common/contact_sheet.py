"""Tile every scene's still into one contact sheet, so a storyboard reads at a glance.

The other aggregate end of the loop. ``ArkSceneList`` fans the plan out so the image node
runs once per scene; this declares ``INPUT_IS_LIST`` so ComfyUI hands it every still in
one call, in order, however many there are.

It exists because the single most important check in a character pipeline — did the same
person survive all N scenes — is impossible to make from N separate preview nodes.

Because INPUT_IS_LIST makes *every* input arrive as a list, the scalar settings come in
wrapped too and are unwrapped with `_one`.
"""

import math


def _one(value, default=None):
    """First element of an INPUT_IS_LIST-wrapped scalar."""
    if isinstance(value, list):
        return value[0] if value else default
    return default if value is None else value


class ArkContactSheet:
    CATEGORY = "arkennemasis/Utility"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("sheet",)
    INPUT_IS_LIST = True          # receive every still of the run in one call
    DESCRIPTION = ("Tile every scene's still into one numbered contact sheet — the "
                   "quickest way to check a character held across a whole storyboard.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Wire the scene chain's image output here. The loop "
                               "delivers every scene to this one socket, in order.",
                }),
                "columns": ("INT", {
                    "default": 5, "min": 1, "max": 10,
                    "tooltip": "Tiles per row.",
                }),
                "cell_width": ("INT", {
                    "default": 420, "min": 64, "max": 2048, "step": 16,
                    "tooltip": "Width of each tile. Height follows the first image's "
                               "aspect ratio.",
                }),
                "label": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Number each tile, so a drifting scene is easy to name.",
                }),
            },
        }

    def run(self, images, columns=5, cell_width=420, label=True):
        import numpy as np
        import torch
        from PIL import Image, ImageDraw

        cols_n = int(_one(columns, 5))
        width = int(_one(cell_width, 420))
        do_label = bool(_one(label, True))

        tiles = []
        for batch in (images or []):
            if batch is None:
                continue
            # IMAGE is a batched float32 tensor (B,H,W,C) in 0..1; take the first frame.
            arr = (batch[0].cpu().numpy() * 255.0).clip(0, 255).astype("uint8")
            tiles.append(Image.fromarray(arr))
        if not tiles:
            raise ValueError("ArkContactSheet: no images arrived. Is the scene chain "
                             "wired into `images`?")

        first = tiles[0]
        cell_h = max(1, round(width * first.height / first.width))
        cols = min(cols_n, len(tiles))
        rows = math.ceil(len(tiles) / cols)
        pad = 8
        sheet = Image.new("RGB",
                          (cols * width + pad * (cols + 1),
                           rows * cell_h + pad * (rows + 1)),
                          (24, 24, 28))
        draw = ImageDraw.Draw(sheet)

        for index, im in enumerate(tiles):
            x = pad + (index % cols) * (width + pad)
            y = pad + (index // cols) * (cell_h + pad)
            sheet.paste(im.resize((width, cell_h), Image.LANCZOS), (x, y))
            if do_label:
                # Plain rectangle + default font: no font file to find, so this cannot
                # fail on a machine that happens to lack a particular typeface.
                draw.rectangle([x, y, x + 34, y + 26], fill=(0, 0, 0))
                draw.text((x + 12, y + 8), str(index + 1), fill=(255, 255, 255))

        out = np.asarray(sheet).astype("float32") / 255.0
        print("[arkennemasis] contact sheet: %d images, %dx%d grid, %dx%d px"
              % (len(tiles), cols, rows, sheet.width, sheet.height))
        return (torch.from_numpy(out)[None, ...],)


NODE_CLASS_MAPPINGS = {
    "ArkContactSheet": ArkContactSheet,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArkContactSheet": "arkennemasis Contact Sheet (all stills in one image)",
}

"""Render an .excalidraw board to an image, so it can be previewed inside ComfyUI.

The board is written as JSON — elements on an infinite canvas plus embedded images —
which is perfect for opening in Excalidraw and useless for checking your work without
leaving ComfyUI. This draws it.

Deliberately a RENDERER, not an embedded editor. Excalidraw's own viewer is a React
bundle that would have to be vendored into `web/` and kept in step with upstream, and it
cannot run at all when the page has no network. Drawing the elements with PIL needs
nothing, works offline, and produces an IMAGE — which means the board can be previewed,
saved, dropped into a contact sheet, or sent to a client through any node that takes an
image.

Only the element types this pack's boards actually emit are supported: images,
rectangles and text. Anything else is skipped with a note rather than drawn wrong.
"""

from __future__ import annotations

import base64
import io
import json
import os

import numpy as np
import torch

from .schema import ValidationError

FONT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "fonts")


def _load_font(size, bold=False):
    """A real font from the pack's own `fonts/`, falling back to PIL's default.

    The bundled families are there for subtitle burn-in, and reusing them keeps the
    board's labels legible at the sizes these grids use — PIL's built-in bitmap font
    does not scale and turns a 46px title into a smear.
    """
    from PIL import ImageFont
    candidates = []
    if os.path.isdir(FONT_DIR):
        names = sorted(os.listdir(FONT_DIR))
        want = ("bold",) if bold else ("regular", "medium")
        for token in want:
            candidates += [n for n in names
                           if n.lower().endswith((".ttf", ".otf")) and token in n.lower()]
        candidates += [n for n in names if n.lower().endswith((".ttf", ".otf"))]
    for name in candidates:
        try:
            return ImageFont.truetype(os.path.join(FONT_DIR, name), int(size))
        except Exception:
            continue
    try:
        return ImageFont.truetype("arial.ttf", int(size))
    except Exception:
        return ImageFont.load_default()


def render_board(board, scale=0.5, background="#FFFFFF", margin=40):
    """Draw an Excalidraw document to a PIL image."""
    from PIL import Image, ImageDraw

    elements = [e for e in (board.get("elements") or []) if not e.get("isDeleted")]
    if not elements:
        raise ValidationError("The board has no elements to draw.")

    left = min(e.get("x", 0) for e in elements)
    top = min(e.get("y", 0) for e in elements)
    right = max(e.get("x", 0) + e.get("width", 0) for e in elements)
    bottom = max(e.get("y", 0) + e.get("height", 0) for e in elements)

    width = int((right - left) * scale) + margin * 2
    height = int((bottom - top) * scale) + margin * 2
    if width < 8 or height < 8:
        raise ValidationError("The board's bounding box is empty.")
    if width * height > 400_000_000:
        raise ValidationError(
            "The board would render at %dx%d, which is too large. Lower `scale`."
            % (width, height))

    canvas = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(canvas)
    files = board.get("files") or {}
    drawn = {"image": 0, "rectangle": 0, "text": 0, "skipped": 0}

    def place(value, axis):
        origin = left if axis == "x" else top
        return int((value - origin) * scale) + margin

    # Painter's order: Excalidraw draws in array order, so anything meant to sit on top
    # is later in the list. Preserving that keeps labels above their swatches.
    for element in elements:
        kind = element.get("type")
        x, y = place(element.get("x", 0), "x"), place(element.get("y", 0), "y")
        w = max(1, int(element.get("width", 0) * scale))
        h = max(1, int(element.get("height", 0) * scale))

        if kind == "image":
            record = files.get(element.get("fileId")) or {}
            data_url = record.get("dataURL") or ""
            if "," not in data_url:
                drawn["skipped"] += 1
                continue
            try:
                raw = base64.b64decode(data_url.split(",", 1)[1])
                with Image.open(io.BytesIO(raw)) as picture:
                    picture = picture.convert("RGB").resize((w, h), Image.LANCZOS)
                    canvas.paste(picture, (x, y))
                drawn["image"] += 1
            except Exception:
                drawn["skipped"] += 1

        elif kind in ("rectangle", "ellipse", "diamond"):
            fill = element.get("backgroundColor")
            fill = None if fill in (None, "", "transparent") else fill
            stroke = element.get("strokeColor")
            stroke = None if stroke in (None, "", "transparent") else stroke
            shape = draw.ellipse if kind == "ellipse" else draw.rectangle
            try:
                shape([x, y, x + w, y + h], fill=fill, outline=stroke,
                      width=max(1, int(element.get("strokeWidth", 1))))
                drawn["rectangle"] += 1
            except Exception:
                drawn["skipped"] += 1

        elif kind == "text":
            text = str(element.get("text") or "")
            if not text:
                continue
            size = max(6, int(element.get("fontSize", 20) * scale))
            font = _load_font(size, bold=element.get("fontFamily") == 5)
            colour = element.get("strokeColor") or "#1e1e1e"
            try:
                draw.text((x, y), text, font=font, fill=colour)
                drawn["text"] += 1
            except Exception:
                drawn["skipped"] += 1
        else:
            drawn["skipped"] += 1

    return canvas, drawn


class ArkBoardPreview:
    CATEGORY = "arkennemasis/Variation"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", "STRING", "INT", "INT")
    RETURN_NAMES = ("image", "report", "width", "height")
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Draw an .excalidraw board as an image so the whole variation matrix can be "
        "checked without leaving ComfyUI. Wire the output into a Preview Image, or into "
        "any node that takes an image to save or send it."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "board_path": ("STRING", {
                    "default": "", "forceInput": True,
                    "tooltip": "The .excalidraw file, from Review Board.",
                }),
            },
            "optional": {
                "scale": ("FLOAT", {
                    "default": 1.0, "min": 0.05, "max": 4.0, "step": 0.05,
                    "tooltip": "Board units to pixels. 1.0 renders every image at its "
                               "full embedded resolution - the right default, since this "
                               "picture is the whole matrix in one file. Lower it only "
                               "for a quick look; raise it past 1.0 and you are "
                               "upscaling thumbnails, not gaining detail.",
                }),
                "background": ("STRING", {"default": "#FFFFFF"}),
                "margin": ("INT", {"default": 40, "min": 0, "max": 400}),
                "save_png": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Also write matrix.png beside the .excalidraw file — the "
                               "version you can email to a client without them needing "
                               "Excalidraw at all.",
                }),
            },
        }

    @classmethod
    def IS_CHANGED(cls, board_path="", **kwargs):
        try:
            return os.path.getmtime(str(board_path).strip().strip('"'))
        except OSError:
            return float("nan")

    def run(self, board_path, scale=0.5, background="#FFFFFF", margin=40,
            save_png=True):
        path = str(board_path or "").strip().strip('"')
        if not path:
            raise ValidationError(
                "No board path. Wire Review Board's excalidraw_path output here — it is "
                "empty when the run produced no delivered masters.")
        if not os.path.isfile(path):
            raise ValidationError("No board at %s." % path)

        with open(path, "r", encoding="utf-8") as handle:
            board = json.load(handle)

        canvas, drawn = render_board(board, float(scale), background, int(margin))

        written = ""
        if save_png:
            written = os.path.splitext(path)[0] + ".png"
            canvas.save(written)

        array = np.asarray(canvas).astype(np.float32) / 255.0
        report = "\n".join([
            "BOARD PREVIEW",
            "  source   : %s" % path,
            "  rendered : %d x %d px at scale %.2f" % (canvas.width, canvas.height, scale),
            "  images   : %d" % drawn["image"],
            "  shapes   : %d" % drawn["rectangle"],
            "  labels   : %d" % drawn["text"],
            "  skipped  : %d element(s) this renderer does not draw" % drawn["skipped"],
        ] + (["  png      : %s" % written] if written else []))

        print("[arkennemasis] board preview: %dx%d, %d image(s), %d label(s)"
              % (canvas.width, canvas.height, drawn["image"], drawn["text"]))
        return (torch.from_numpy(array)[None, ...], report, canvas.width, canvas.height)


NODE_CLASS_MAPPINGS = {
    "ArkBoardPreview": ArkBoardPreview,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArkBoardPreview": "arkennemasis Board Preview (render the excalidraw matrix)",
}

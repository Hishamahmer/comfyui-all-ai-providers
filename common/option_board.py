"""Collect a fanned-out run back into one labelled board — .excalidraw plus a preview.

A batch node declares ``OUTPUT_IS_LIST``, so ComfyUI runs the chain once per option. This
node declares ``INPUT_IS_LIST`` and therefore receives **every** result of the run in a
single call, which is what lets it lay them out together. That pairing — fan out, run,
aggregate — is the same one ``ArkSceneList`` and ``ArkVideoAssemble`` use for video.

Two outputs from ONE layout pass:

* an **.excalidraw** file, which is the point. A board opens in excalidraw.com or the
  desktop app, where the tiles can be moved, circled, annotated and commented on. A PNG
  contact sheet is a thing you look at; a board is a thing you work on, and picking
  winners out of forty variants is work.
* a rendered **IMAGE** of the same geometry, so the result is visible inside ComfyUI
  without leaving the tab. Same positions, two renderers — not a second layout that can
  disagree with the first.

Generic on purpose: it knows about images and captions, not about hairstyles, markets or
products. Both the hairstyle and thumbnail canvases wire their own batch node into it, and
so can anything else that fans out.

**Every input arrives as a list** because INPUT_IS_LIST is all-or-nothing, so the scalar
settings come in wrapped and have to be unwrapped with ``_one``. Forgetting that is how a
widget value ends up as ``['board']`` in a filename.
"""

from __future__ import annotations

import os
import re

from .excalidraw import Board, INK, MUTED

TITLE_SIZE = 34
CAPTION_SIZE = 15
GAP = 28
CAPTION_H = 46
TITLE_H = 110
PAD = 40
BG = "#ffffff"


def _one(value, default=None):
    """First element of an INPUT_IS_LIST-wrapped scalar."""
    if isinstance(value, list):
        return value[0] if value else default
    return value if value is not None else default


def _slug(text):
    return re.sub(r"[^a-z0-9]+", "-", str(text or "").lower()).strip("-") or "board"


def _frames(images):
    """Every option's picture, as PIL, from whatever shape the input arrived in.

    INPUT_IS_LIST hands this node a list whose elements are the per-execution IMAGE
    values — each one a `(B, H, W, C)` tensor, normally with B=1. Nesting happens easily
    though (an upstream node returning a list, or a hand-written call passing the list
    twice), and the failure it produced was a PIL `Cannot handle this data type` several
    frames deep, which says nothing about the actual mistake. So flatten first and fail
    with a sentence instead.

    Only frame 0 of a batch is taken: one option, one picture. A multi-frame batch on one
    option would have no label of its own and would silently push every later caption out
    of step with its tile.
    """
    import numpy as np
    from PIL import Image

    flat = []

    def walk(value, depth=0):
        if value is None:
            return
        if isinstance(value, (list, tuple)):
            if depth > 4:
                raise RuntimeError("ArkOptionBoard: `images` is nested too deeply to "
                                   "unpack. Wire the batch chain's image node straight "
                                   "into it.")
            for item in value:
                walk(item, depth + 1)
            return
        if not hasattr(value, "cpu"):
            raise RuntimeError(
                "ArkOptionBoard: `images` contains a %s, not an IMAGE. Wire the image "
                "node of the batch chain into it." % type(value).__name__)
        flat.append(value)

    walk(images)

    out = []
    for tensor in flat:
        array = tensor
        # A bare (H, W, C) tensor has no batch dimension; a (B, H, W, C) one does.
        if getattr(array, "ndim", 3) == 4:
            array = array[0]
        array = (array.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        out.append(Image.fromarray(array))
    return out


def _to_pil(tensor):
    """One IMAGE tensor -> PIL. Kept for the single-image callers."""
    found = _frames(tensor)
    return found[0] if found else None


def _output_dir(sub):
    """Resolve a relative path under ComfyUI's output dir, or honour an absolute one."""
    sub = (sub or "").strip().replace("\\", "/")
    if os.path.isabs(sub):
        return sub
    try:
        import folder_paths
        root = folder_paths.get_output_directory()
    except Exception:                       # importable outside ComfyUI, for tests
        root = os.path.abspath("output")
    return os.path.join(root, *[p for p in sub.split("/") if p])


class ArkOptionBoard:
    CATEGORY = "arkennemasis/Utility"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING", "IMAGE", "STRING")
    RETURN_NAMES = ("excalidraw_path", "board", "report")
    OUTPUT_NODE = True
    INPUT_IS_LIST = True          # receive every result of the run in one call
    DESCRIPTION = (
        "Lay every result of a fanned-out run on one labelled board: an .excalidraw "
        "file you can open and annotate, plus a rendered preview of the same layout. "
        "Wire a batch node's images and labels in."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Every generated option. Comes from the image node in the "
                               "batch chain — the list arrives here in one call.",
                }),
                "out_dir": ("STRING", {
                    "default": "boards",
                    "tooltip": "Relative to ComfyUI's output folder, or an absolute "
                               "path. The board and its preview are written here.",
                }),
                "filename": ("STRING", {
                    "default": "options",
                    "tooltip": "Base name. An existing board is never overwritten — a "
                               "counter is appended, so a second run sits beside the "
                               "first instead of replacing the thing you were reviewing.",
                }),
                "columns": ("INT", {
                    "default": 0, "min": 0, "max": 24,
                    "tooltip": "Tiles per row. 0 picks a roughly square grid from the "
                               "number of options, which is what you want almost always.",
                }),
                "tile_size": ("INT", {
                    "default": 420, "min": 96, "max": 2048, "step": 4,
                    "tooltip": "Longest edge of each tile on the board. Bigger reads "
                               "better and costs file size — 100 tiles at 420 is a few MB.",
                }),
                "write_preview_png": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Also write the rendered board as a PNG beside the "
                               ".excalidraw, for pasting into a message.",
                }),
            },
            "optional": {
                "labels": ("STRING", {
                    "forceInput": True,
                    "tooltip": "One caption per image, in the same order. Wire the batch "
                               "node's `label` output.",
                }),
                "title": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Heading printed at the top of the board.",
                }),
            },
        }

    def run(self, images, out_dir="boards", filename="options", columns=0,
            tile_size=420, write_preview_png=True, labels=None, title=None):
        import numpy as np
        import torch
        from PIL import Image, ImageDraw, ImageFont

        # INPUT_IS_LIST wraps everything, scalars included.
        out_dir = _one(out_dir, "boards")
        filename = _one(filename, "options")
        columns = int(_one(columns, 0) or 0)
        tile_size = int(_one(tile_size, 420) or 420)
        write_preview_png = bool(_one(write_preview_png, True))
        heading = _one(title, "") or "Options"

        # `images` is every result of the run. Flattened and unbatched in one place, so
        # a nested delivery shape is handled rather than dying several frames deep.
        frames = _frames(images)
        if not frames:
            raise RuntimeError(
                "ArkOptionBoard got no images. Wire the batch chain's image node into "
                "`images` — and check the batch node's `limit` is not 0.")

        captions = labels if isinstance(labels, list) else ([labels] if labels else [])
        captions = [str(c) for c in captions]
        # A missing caption is numbered rather than blank: on a board of forty tiles,
        # "the third one in row two" is not a thing anyone can say back to you.
        while len(captions) < len(frames):
            captions.append("Option %d" % (len(captions) + 1))

        count = len(frames)
        if columns <= 0:
            # Roughly square, but never so wide that a row does not fit on a screen.
            columns = min(6, max(1, int(round(count ** 0.5))))
        rows = (count + columns - 1) // columns

        # Tiles are uniform so the grid reads as a grid; each image is letterboxed into
        # its tile rather than stretched, because a squashed variant is a variant you
        # judge wrongly.
        widest = max(f.width / f.height for f in frames)
        tile_w = tile_size if widest >= 1 else int(tile_size * widest)
        tile_h = int(tile_w / widest) if widest >= 1 else tile_size
        tile_w, tile_h = max(48, tile_w), max(48, tile_h)

        cell_w = tile_w + GAP
        cell_h = tile_h + CAPTION_H + GAP
        board_w = PAD * 2 + columns * cell_w - GAP
        board_h = PAD * 2 + TITLE_H + rows * cell_h - GAP

        # ── one layout pass, used by both renderers ─────────────────────────
        placements = []
        for index, frame in enumerate(frames):
            column, row = index % columns, index // columns
            x = PAD + column * cell_w
            y = PAD + TITLE_H + row * cell_h
            scale = min(tile_w / frame.width, tile_h / frame.height)
            w, h = max(1, int(frame.width * scale)), max(1, int(frame.height * scale))
            placements.append({
                "frame": frame, "caption": captions[index],
                "x": x + (tile_w - w) // 2, "y": y + (tile_h - h) // 2,
                "w": w, "h": h,
                "cap_x": x, "cap_y": y + tile_h + 10, "cap_w": tile_w,
            })

        destination = _output_dir(out_dir)
        os.makedirs(destination, exist_ok=True)
        stem = _slug(filename)
        # Never overwrite. A board is the thing being reviewed; silently replacing it on
        # the next run is how a comparison disappears mid-review.
        base, counter = stem, 1
        while os.path.exists(os.path.join(destination, "%s.excalidraw" % stem)):
            counter += 1
            stem = "%s_%03d" % (base, counter)

        # ── the .excalidraw ─────────────────────────────────────────────────
        board = Board(source="arkennemasis/option-board")
        board.text(PAD, PAD, heading, size=TITLE_SIZE, bold=True, width=board_w - PAD * 2)
        board.text(PAD, PAD + TITLE_SIZE * 1.5,
                   "%d option%s" % (count, "" if count == 1 else "s"),
                   size=CAPTION_SIZE + 2, colour=MUTED, width=400)
        for place in placements:
            board.image(place["x"], place["y"], place["w"], place["h"], place["frame"])
            board.text(place["cap_x"], place["cap_y"], place["caption"],
                       size=CAPTION_SIZE, colour=INK, width=place["cap_w"])
        excalidraw_path = os.path.join(destination, "%s.excalidraw" % stem)
        size = board.write(excalidraw_path, background=BG)

        # ── the same geometry, rendered ─────────────────────────────────────
        sheet = Image.new("RGB", (board_w, board_h), (255, 255, 255))
        draw = ImageDraw.Draw(sheet)
        title_font = _font(TITLE_SIZE)
        caption_font = _font(CAPTION_SIZE + 2)
        draw.text((PAD, PAD), heading, font=title_font, fill=(30, 30, 30))
        draw.text((PAD, PAD + int(TITLE_SIZE * 1.5)),
                  "%d option%s" % (count, "" if count == 1 else "s"),
                  font=caption_font, fill=(134, 142, 150))
        for place in placements:
            sheet.paste(place["frame"].resize((place["w"], place["h"]), Image.LANCZOS),
                        (place["x"], place["y"]))
            draw.text((place["cap_x"], place["cap_y"]),
                      _ellipsise(draw, place["caption"], caption_font, place["cap_w"]),
                      font=caption_font, fill=(30, 30, 30))

        png_path = ""
        if write_preview_png:
            png_path = os.path.join(destination, "%s.png" % stem)
            tmp = png_path + ".part"
            # `format` is explicit because PIL infers it from the extension, and the
            # atomic-write suffix makes that ".part" — which it does not recognise.
            sheet.save(tmp, format="PNG")
            os.replace(tmp, png_path)

        report = "\n".join([
            "board       : %s" % excalidraw_path,
            "preview     : %s" % (png_path or "(not written)"),
            "options     : %d in a %dx%d grid" % (count, columns, rows),
            "size        : %.1f MB" % (size / 1048576.0),
        ])
        print("[arkennemasis] option board: %d option(s), %dx%d, %.1f MB -> %s"
              % (count, columns, rows, size / 1048576.0, excalidraw_path))

        out = torch.from_numpy(np.array(sheet).astype("float32") / 255.0)[None, ...]
        return (excalidraw_path, out, report)


def _font(size, bold=False):
    from PIL import ImageFont

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [
        os.path.join(here, "fonts", "Roboto-Bold.ttf" if bold else "Roboto-Regular.ttf"),
        os.path.join(here, "fonts", "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"),
        r"C:\Windows\Fonts\segoeuib.ttf" if bold else r"C:\Windows\Fonts\segoeui.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, ValueError):
            continue
    return ImageFont.load_default()


def _ellipsise(draw, text, font, width):
    """Trim a caption to the tile width in the PNG render.

    The .excalidraw keeps the full text — it has `autoResize` and an infinite canvas, so
    nothing is lost there. Only the fixed-width PNG needs the trim.
    """
    if draw.textlength(text, font=font) <= width:
        return text
    while text and draw.textlength(text + "…", font=font) > width:
        text = text[:-1]
    return text + "…"


NODE_CLASS_MAPPINGS = {
    "ArkOptionBoard": ArkOptionBoard,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArkOptionBoard": "arkennemasis Option Board (excalidraw + preview)",
}

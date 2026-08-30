"""Primitives for writing an .excalidraw board.

An .excalidraw file is JSON: elements positioned on an infinite canvas, plus a `files`
map of embedded images. So a board is a **layout problem, not a rendering one** — no image
tooling and no canvas library is needed to produce one, and the result opens in
excalidraw.com, the desktop app or an Obsidian vault, where the client can move things,
annotate them and comment without any of it being baked into a picture.

That is the whole reason a review surface is an .excalidraw rather than a PNG contact
sheet: a PNG is a thing you look at, and a board is a thing you work on.

These helpers are the element shapes only. What goes where is the caller's business —
`ArkOptionBoard` lays out a labelled grid, and a future board can lay out something else
from the same primitives.

**Known duplication, deliberately left alone:** `variation/deliver.py` carries its own
copy of `_el` / `_text` / `_swatch`, written before this module existed. It is working,
shipped code driving a delivered client pipeline, and folding it into this module is a
refactor worth doing on purpose rather than as a side effect of adding a board somewhere
else. If you touch either, reconcile both.
"""

from __future__ import annotations

import base64
import io
import json
import os

# Excalidraw's own font ids. 2 is the normal hand-drawn face, 5 reads as bolder at board
# scale; 3 is the monospace one. Named because the integers are meaningless in isolation.
FONT_NORMAL = 2
FONT_BOLD = 5
FONT_MONO = 3

INK = "#1e1e1e"
MUTED = "#868e96"
TRANSPARENT = "transparent"


class Board:
    """An accumulating list of excalidraw elements plus the images they reference.

    Stateful on purpose: every element needs a unique id and a seed, and threading a
    counter through a dozen call sites is how two elements end up sharing an id and one
    of them silently disappears in the editor.
    """

    def __init__(self, source="arkennemasis"):
        self.elements = []
        self.files = {}
        self.source = source
        self._seq = 0

    # ── element shapes ───────────────────────────────────────────────────────

    def el(self, kind, x, y, w, h, **extra):
        element = {
            "type": kind, "id": "e%d" % self._seq, "x": round(x), "y": round(y),
            "width": round(w), "height": round(h), "angle": 0,
            "strokeColor": INK, "backgroundColor": TRANSPARENT,
            "fillStyle": "solid", "strokeWidth": 1, "strokeStyle": "solid",
            "roughness": 0, "opacity": 100, "groupIds": [], "frameId": None,
            "roundness": None, "seed": self._seq + 1, "version": 1, "versionNonce": 1,
            "isDeleted": False, "boundElements": None, "updated": 1, "link": None,
            "locked": False,
        }
        element.update(extra)
        self._seq += 1
        self.elements.append(element)
        return element

    def text(self, x, y, content, size=20, colour=INK, width=None, bold=False,
             align="left"):
        content = str(content)
        # 0.56em is the measured average advance of excalidraw's default face. It only
        # has to be close: `autoResize` lets the editor correct it on open, and the
        # estimate is what keeps the layout honest before anyone opens it.
        width = width or max(60, int(len(content) * size * 0.56))
        return self.el(
            "text", x, y, width, size * 1.25,
            strokeColor=colour, text=content, fontSize=size,
            fontFamily=(FONT_BOLD if bold else FONT_NORMAL),
            textAlign=align, verticalAlign="top", baseline=size,
            containerId=None, originalText=content, lineHeight=1.25, autoResize=True)

    def rect(self, x, y, w, h, fill=TRANSPARENT, stroke=MUTED, radius=True):
        return self.el("rectangle", x, y, w, h, strokeColor=stroke,
                       backgroundColor=fill, fillStyle="solid",
                       roundness={"type": 3} if radius else None)

    def image(self, x, y, w, h, pil_image, quality=80):
        """Embed a PIL image as a WEBP data URL and place it.

        WEBP rather than PNG because a board is often a hundred tiles: the same grid came
        to 2.4 MB as WEBP and 31 MB as PNG, and a 31 MB board is one nobody can email.
        Lossy at 80 is invisible at tile size and this is a review surface, not a
        deliverable — the deliverables are the PNGs the save node already wrote.
        """
        buffer = io.BytesIO()
        pil_image.convert("RGB").save(buffer, format="WEBP", quality=quality, method=4)
        payload = base64.b64encode(buffer.getvalue()).decode("ascii")
        file_id = "f%d" % len(self.files)
        self.files[file_id] = {
            "mimeType": "image/webp", "id": file_id,
            "dataURL": "data:image/webp;base64,%s" % payload,
            "created": 1, "lastRetrieved": 1,
        }
        return self.el("image", x, y, w, h, strokeColor=TRANSPARENT,
                       backgroundColor=TRANSPARENT, status="saved", fileId=file_id,
                       scale=[1, 1])

    # ── writing ──────────────────────────────────────────────────────────────

    def document(self, background="#ffffff"):
        return {
            "type": "excalidraw", "version": 2, "source": self.source,
            "elements": self.elements, "files": self.files,
            "appState": {"gridSize": None, "viewBackgroundColor": background},
        }

    def write(self, path, background="#ffffff"):
        """Atomic write — a half-written board is a file that opens to an error."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".part"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.document(background), fh, ensure_ascii=False)
        os.replace(tmp, path)
        return os.path.getsize(path)

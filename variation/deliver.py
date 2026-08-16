"""Stages 9 and 10 — the format ladder, the review surface, and the store import.

The client's real objective is "my variations are live on the product page", not "I have
a folder of PNG files". A pipeline that ends at a folder leaves the hardest and most
tedious step — mapping images to store variations — with the client. So delivery here
means four things, not one:

    ArkDeliver       one approved master -> the whole format ladder, filed by axis
    ArkReviewBoard   an HTML contact sheet with real approve/reject, plus an
                     .excalidraw board for the client-facing summary
    ArkStoreExport   a completed WooCommerce variation CSV carrying each row's image

Derivation is a pure function: the ladder can be regenerated at any time from the
masters without re-running any model, which is why nothing here is allowed to be
expensive or clever.

On review tooling: Excalidraw is a whiteboard. It has no state, no approve/reject
affordance and no write-back, so it makes an excellent client-facing visual summary and
a poor review surface. The HTML sheet is the review surface, and it writes decisions
straight back into the durable job records through a registered route.
"""

from __future__ import annotations

import base64
import csv
import datetime
import html
import json
import os

import numpy as np

from .job_store import load_job, resolve_dir, safe_key, save_job
from .schema import ValidationError, canonical

LADDER_DEFAULT = "png"
THUMB_DEFAULT = 512


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _pil(image):
    from PIL import Image
    tensor = image[0] if len(image.shape) == 4 else image
    array = tensor.detach().cpu().numpy() if hasattr(tensor, "detach") else np.asarray(tensor)
    array = (np.clip(array.astype(np.float32), 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(array[:, :, :3])


def cell_folder(root, cell, recipe=None, structure="flat"):
    """Where one cell's files land.

    `flat` puts every finished image for a product in ONE directory. That is the
    default because it is what a delivery actually looks like: the filename already
    encodes every axis value, so a directory per axis adds depth without adding
    information, and a two-axis product buries 40 images 5 levels down where neither
    the operator nor the client can see them side by side.

    `by-axis` keeps the nested `{product}/{axis1}/{axis2}/…` layout — one level per
    axis, for any N — for clients whose asset system expects that shape.
    """
    if str(structure or "flat").lower().startswith("flat"):
        return root                       # everything for this run in ONE directory

    parts = [canonical(cell.get("product")) or "product"]
    order = []
    if recipe:
        order = [a.get("name") for a in
                 sorted(recipe.get("axes") or [], key=lambda a: a.get("order") or 99)]
    axes = cell.get("axes") or {}
    for name in (order or sorted(axes)):
        value = axes.get(name)
        if value:
            parts.append(canonical(value))
    return os.path.join(root, *parts)


class ArkDeliver:
    CATEGORY = "arkennemasis/Variation"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("master_path", "delivered_json", "report")
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Write one accepted master and derive its whole delivery ladder — PNG, JPG, "
        "WebP and thumbnail — into the per-axis folder structure, using the client's "
        "own filename. Re-running overwrites in place: no _v2, no timestamps, no "
        "ambiguity about which file is live."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The verified, accepted candidate."}),
                "cell_json": ("STRING", {
                    "multiline": True, "default": "{}", "forceInput": True,
                }),
                "output_dir": ("STRING", {
                    "default": "variation/delivered",
                    "tooltip": "Delivery root. Relative paths resolve inside ComfyUI's "
                               "output directory.",
                }),
            },
            "optional": {
                "recipe_json": ("STRING", {
                    "multiline": True, "default": "", "forceInput": True,
                }),
                "formats": ("STRING", {
                    "default": LADDER_DEFAULT,
                    "tooltip": "Comma-separated ladder: png, jpg, webp, thumb. Derived "
                               "deterministically, so the ladder can always be rebuilt "
                               "from the masters without touching a model.",
                }),
                "jpg_background": ("STRING", {
                    "default": "#FFFFFF",
                    "tooltip": "Matte colour for the white-background JPG.",
                }),
                "thumb_size": ("INT", {
                    "default": THUMB_DEFAULT, "min": 32, "max": 4096,
                }),
                "quality": ("INT", {
                    "default": 92, "min": 40, "max": 100,
                }),
                "colour_profile": ("STRING", {
                    "default": "sRGB",
                    "tooltip": "Embedded into every derived file. Generating in one "
                               "colour space and delivering in another is the usual "
                               "cause of 'the colour looks wrong' after delivery.",
                }),
                "jobs_dir": ("STRING", {
                    "default": "variation/jobs",
                    "tooltip": "Delivered paths are recorded on the job record here.",
                }),
                # APPENDED LAST on purpose: widgets_values is positional, so a new
                # widget anywhere else would shift every later value in graphs already
                # saved against this node.
                "structure": (["flat", "by-axis"], {
                    "tooltip": "'flat' puts every image for a product in ONE folder — "
                               "the filename already carries every axis value, so a "
                               "directory per axis adds depth without adding "
                               "information and buries the set. 'by-axis' keeps the "
                               "nested layout for clients whose asset system wants it.",
                }),
            },
        }

    def run(self, image, cell_json, output_dir="variation/delivered", recipe_json="",
            formats=LADDER_DEFAULT, jpg_background="#FFFFFF", thumb_size=THUMB_DEFAULT,
            quality=92, colour_profile="sRGB", jobs_dir="variation/jobs",
            structure="flat"):
        from PIL import Image

        cell = json.loads(cell_json or "{}")
        recipe = json.loads(recipe_json) if str(recipe_json or "").strip() else None
        if not cell.get("filename"):
            raise ValidationError("ArkDeliver: the cell has no filename (the primary key).")

        root = resolve_dir(output_dir, "variation/delivered")
        folder = cell_folder(root, cell, recipe, structure)
        stem = os.path.splitext(os.path.basename(cell["filename"]))[0]

        picture = _pil(image)
        wanted = [f.strip().lower() for f in str(formats or "").split(",") if f.strip()]
        delivered = {}

        profile = None
        try:
            from PIL import ImageCms
            profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
        except Exception:
            profile = None

        # Flat means flat: no product folder, no per-format folder. The filename
        # already carries the product and every axis value, so the directories were
        # depth without information.
        flat = str(structure or "flat").lower().startswith("flat")

        def target(fmt, extension):
            if flat:
                # png/jpg/webp differ by extension, but the thumbnail is ALSO webp, so
                # in one directory it would overwrite the full-size webp. Suffix it.
                name = "%s_thumb%s" % (stem, extension) if fmt == "thumb"                     else "%s%s" % (stem, extension)
                return os.path.join(folder, name)
            sub = os.path.join(folder, fmt)
            os.makedirs(sub, exist_ok=True)
            return os.path.join(sub, "%s%s" % (stem, extension))

        # The master is the PNG, written first and hashed by everything downstream.
        os.makedirs(folder, exist_ok=True)
        master_path = target("png", ".png")
        picture.save(master_path, format="PNG",
                     **({"icc_profile": profile} if profile else {}))
        delivered["png"] = master_path

        for fmt in wanted:
            if fmt in ("png", "master"):
                continue
            if fmt == "jpg" or fmt == "jpeg":
                flat = Image.new("RGB", picture.size, jpg_background.strip() or "#FFFFFF")
                flat.paste(picture.convert("RGB"), (0, 0))
                path = target("jpg", ".jpg")
                flat.save(path, format="JPEG", quality=int(quality), subsampling=0,
                          **({"icc_profile": profile} if profile else {}))
                delivered["jpg"] = path
            elif fmt == "webp":
                path = target("webp", ".webp")
                picture.save(path, format="WEBP", quality=int(quality), method=6,
                             **({"icc_profile": profile} if profile else {}))
                delivered["webp"] = path
            elif fmt in ("thumb", "thumbnail"):
                thumb = picture.copy()
                thumb.thumbnail((int(thumb_size), int(thumb_size)), Image.LANCZOS)
                path = target("thumb", ".webp")
                thumb.save(path, format="WEBP", quality=int(quality), method=6)
                delivered["thumb"] = path
            else:
                print("[arkennemasis] deliver: unknown format '%s', skipped." % fmt)

        folder_for_jobs = resolve_dir(jobs_dir)
        job = load_job(folder_for_jobs, cell.get("key"))
        if job:
            job["master"] = master_path
            job["delivered"] = delivered
            job["colour_profile"] = colour_profile
            job["updated_at"] = _now()
            # "generating" is the state ArkJobSkip leaves a live cell in, so it is the
            # status seen when this node runs BEFORE ArkJobRecord. Promoting from it
            # too is what makes the two orders converge; ArkJobRecord then reads the
            # `delivered` payload written just above and agrees. A cell that later
            # fails is still correctly moved to flagged by ArkJobRecord.
            if job.get("status") in ("passed", "generating"):
                job["status"] = "delivered"
            save_job(folder_for_jobs, job)

        report = "\n".join(
            ["DELIVERED %s" % cell["filename"], "  folder : %s" % folder] +
            ["  %-6s %s" % (k, v) for k, v in sorted(delivered.items())])
        print("[arkennemasis] delivered %s -> %d format(s)"
              % (cell["filename"], len(delivered)))
        return (master_path, json.dumps(delivered, ensure_ascii=False), report)


# ── The review surface ───────────────────────────────────────────────────────
# Registering a route lets the contact sheet write approve/reject decisions straight
# into the durable job records. Without it the sheet would be another read-only board,
# and the review step would still be living in someone's inbox.

def _register_route():
    from server import PromptServer
    from aiohttp import web

    @PromptServer.instance.routes.post("/arkennemasis/variation/decide")
    async def _decide(request):
        payload = await request.json()
        folder = resolve_dir(payload.get("jobs_dir") or "variation/jobs")
        key = payload.get("key")
        decision = str(payload.get("decision") or "").lower()
        if decision not in ("approved", "flagged"):
            return web.json_response({"ok": False, "error": "bad decision"}, status=400)
        job = load_job(folder, key)
        if job is None:
            return web.json_response({"ok": False, "error": "no such job"}, status=404)
        job["status"] = decision
        job["reviewed_at"] = _now()
        job["reviewed_by"] = str(payload.get("by") or "operator")
        save_job(folder, job)
        return web.json_response({"ok": True, "key": key, "status": decision})


try:
    _register_route()
except Exception as exc:
    print("[arkennemasis] variation review route not registered: %s" % exc)


_SHEET_CSS = """
:root{--bg:#fbfbfc;--fg:#16171a;--line:#e3e4e8;--ok:#1a7f45;--bad:#b3261e;--muted:#6b6d76}
@media(prefers-color-scheme:dark){:root{--bg:#131417;--fg:#eceef2;--line:#2b2d33;--muted:#9a9da6}}
*{box-sizing:border-box}
body{margin:0;padding:28px;background:var(--bg);color:var(--fg);
 font:14px/1.5 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
h1{font-size:20px;margin:0 0 4px}
.sub{color:var(--muted);margin-bottom:22px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:16px}
.card{border:1px solid var(--line);border-radius:10px;overflow:hidden;background:transparent}
.card img{width:100%;aspect-ratio:1;object-fit:contain;display:block;background:#fff}
.meta{padding:9px 11px;font-size:12px;border-top:1px solid var(--line)}
.key{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11px;
 word-break:break-all;color:var(--muted)}
.row{display:flex;gap:6px;padding:9px 11px;border-top:1px solid var(--line)}
button{flex:1;padding:6px 0;border:1px solid var(--line);border-radius:6px;
 background:transparent;color:inherit;cursor:pointer;font-size:12px}
button.on{color:#fff}
.ok.on{background:var(--ok);border-color:var(--ok)}
.bad.on{background:var(--bad);border-color:var(--bad)}
.tag{display:inline-block;padding:1px 7px;border-radius:99px;font-size:11px;
 border:1px solid var(--line)}
table{border-collapse:collapse;margin:0 0 22px}
td,th{border:1px solid var(--line);padding:5px 10px;font-size:12px;text-align:left}
"""


class ArkReviewBoard:
    CATEGORY = "arkennemasis/Variation"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("html_path", "excalidraw_path", "report")
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Build the review surface: an HTML contact sheet whose approve/reject buttons "
        "write straight into the job records, and an .excalidraw board laid out as the "
        "axis matrix for the client. Review is organised by axis value, so the number "
        "of decisions scales with the axes rather than with the outputs."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "jobs_dir": ("STRING", {"default": "variation/jobs"}),
                "out_dir": ("STRING", {"default": "variation/review"}),
            },
            "optional": {
                "recipe_json": ("STRING", {
                    "multiline": True, "default": "", "forceInput": True,
                    "tooltip": "Supplies the axis order and each value's label and hex "
                               "for the board's row and column headers.",
                }),
                "product": ("STRING", {"default": ""}),
                "embed_images": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Inline the thumbnails as data URIs so the HTML file is "
                               "self-contained and can be emailed. Off links to files "
                               "on disk instead, which keeps it small.",
                }),
                "thumb_size": ("INT", {
                    "default": 0, "min": 0, "max": 2048,
                    "tooltip": "Pixels per image on the board, which also sets the tile "
                               "size and therefore how big the finished matrix picture "
                               "is. Leave it at 0 for FULL SIZE - it matches whatever "
                               "resolution your images actually are, so nothing is lost. "
                               "Set 640 or 1024 only if the picture is too heavy to open.",
                }),
                "server_url": ("STRING", {
                    "default": "http://127.0.0.1:8188",
                    "tooltip": "Where the approve/reject buttons post their decisions.",
                }),
                "write_excalidraw": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Write the .excalidraw matrix - the client-facing board.",
                }),
                "write_html": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Write the HTML contact sheet - the approve/reject "
                               "surface that writes back into the job records.",
                }),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def run(self, jobs_dir, out_dir, recipe_json="", product="", embed_images=True,
            thumb_size=320, server_url="http://127.0.0.1:8188",
            write_excalidraw=True, write_html=True):
        from PIL import Image

        folder = resolve_dir(jobs_dir)
        destination = resolve_dir(out_dir, "variation/review")
        recipe = json.loads(recipe_json) if str(recipe_json or "").strip() else {}
        wanted = str(product or "").strip()

        jobs = []
        for entry in sorted(os.listdir(folder)):
            if not entry.endswith(".json"):
                continue
            try:
                with open(os.path.join(folder, entry), "r", encoding="utf-8") as handle:
                    job = json.load(handle)
            except (OSError, ValueError):
                continue
            if wanted and job.get("product") != wanted:
                continue
            if job.get("master") and os.path.isfile(job["master"]):
                jobs.append(job)

        if not jobs:
            report = ("No delivered masters found in %s%s — nothing to review yet."
                      % (folder, " for product '%s'" % wanted if wanted else ""))
            print("[arkennemasis] review board: %s" % report)
            return ("", "", report)

        axes = [a.get("name") for a in
                sorted(recipe.get("axes") or [], key=lambda a: a.get("order") or 99)]
        if not axes:
            axes = sorted({k for job in jobs for k in (job.get("axes") or {})})

        # thumb_size 0 means "as large as the images actually are". The board is the
        # whole matrix in one picture, so it should lose nothing to a fixed default that
        # happened to be smaller than this product's renders — and a t-shirt shot at
        # 2048 and a pendant at 1450 must both come out 1:1 without anyone tuning a
        # widget. PIL's thumbnail() only ever shrinks, so asking for the exact native
        # size is a no-op resize rather than an upscale.
        if int(thumb_size) <= 0:
            largest = 0
            for job in jobs:
                try:
                    with Image.open(job["master"]) as probe:
                        largest = max(largest, probe.width, probe.height)
                except Exception:
                    continue
            thumb_size = min(2048, largest) or 640
            print("[arkennemasis] review board: matching source resolution, %d px"
                  % thumb_size)

        def thumb_data(path, with_size=False):
            """Base64 WebP, and optionally the thumbnail's real pixel dimensions.

            The board needs the dimensions: a product photograph is rarely square, and
            drawing every tile into a fixed square box stretches it. Excalidraw scales
            an image element to whatever width/height it is given, so the aspect has to
            be right at layout time — there is no 'contain' mode to fall back on.
            """
            with Image.open(path) as picture:
                picture = picture.convert("RGB")
                picture.thumbnail((int(thumb_size), int(thumb_size)), Image.LANCZOS)
                width, height = picture.size
                import io
                buffer = io.BytesIO()
                picture.save(buffer, format="WEBP", quality=90, method=6)
            payload = base64.b64encode(buffer.getvalue()).decode()
            return (payload, width, height) if with_size else payload

        counts = {}
        for job in jobs:
            counts[job.get("status", "?")] = counts.get(job.get("status", "?"), 0) + 1

        cards = []
        for job in sorted(jobs, key=lambda j: j.get("key") or ""):
            if embed_images:
                try:
                    source = "data:image/webp;base64,%s" % thumb_data(job["master"])
                except Exception:
                    source = "file:///" + job["master"].replace("\\", "/")
            else:
                source = "file:///" + job["master"].replace("\\", "/")

            attempts = job.get("attempts") or []
            last = attempts[-1] if attempts else {}
            checks = last.get("checks") or {}
            bits = []
            for label, key in (("dE", "colour_de"), ("IoU", "silhouette_iou"),
                               ("SSIM", "ssim_untargeted"), ("shift", "bbox_centre_shift")):
                if key in checks:
                    bits.append("%s %.3f" % (label, float(checks[key])))
            axis_line = " · ".join("%s: %s" % (a, (job.get("axes") or {}).get(a, "?"))
                                   for a in axes)
            status = job.get("status", "?")
            cards.append(
                '<div class="card" data-key="{key}">'
                '<img loading="lazy" src="{src}" alt="{alt}">'
                '<div class="meta"><span class="tag">{status}</span> {axis}<br>'
                '<span class="key">{alt}</span><br><span class="key">{checks}</span></div>'
                '<div class="row">'
                '<button class="ok{okon}" onclick="decide(this,\'approved\')">Approve</button>'
                '<button class="bad{badon}" onclick="decide(this,\'flagged\')">Reject</button>'
                '</div></div>'.format(
                    key=html.escape(job.get("key") or ""),
                    src=source,
                    alt=html.escape(job.get("filename") or job.get("key") or ""),
                    status=html.escape(status),
                    axis=html.escape(axis_line),
                    checks=html.escape(", ".join(bits) or "no measurements recorded"),
                    okon=" on" if status == "approved" else "",
                    badon=" on" if status == "flagged" else ""))

        summary_rows = "".join(
            "<tr><td>%s</td><td>%d</td></tr>" % (html.escape(k), v)
            for k, v in sorted(counts.items()))

        page = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Variation review — {title}</title><style>{css}</style></head><body>
<h1>Variation review — {title}</h1>
<div class="sub">{count} delivered masters · axes: {axes} · decisions write straight into
the job records</div>
<table><tr><th>status</th><th>count</th></tr>{rows}</table>
<div class="grid">{cards}</div>
<script>
const ENDPOINT = {endpoint} + "/arkennemasis/variation/decide";
const JOBS_DIR = {jobsdir};
async function decide(button, decision) {{
  const card = button.closest(".card");
  const key = card.dataset.key;
  const [ok, bad] = card.querySelectorAll("button");
  try {{
    const response = await fetch(ENDPOINT, {{
      method: "POST", headers: {{"Content-Type": "application/json"}},
      body: JSON.stringify({{key: key, decision: decision, jobs_dir: JOBS_DIR}})
    }});
    if (!response.ok) throw new Error(await response.text());
    ok.classList.toggle("on", decision === "approved");
    bad.classList.toggle("on", decision === "flagged");
    card.querySelector(".tag").textContent = decision;
  }} catch (error) {{
    alert("Could not save: " + error + "\\n\\nIs ComfyUI running at " + {endpoint} + "?");
  }}
}}
</script></body></html>""".format(
            title=html.escape(recipe.get("product_display") or wanted or "product"),
            css=_SHEET_CSS,
            count=len(jobs),
            axes=html.escape(", ".join(map(str, axes))),
            rows=summary_rows,
            cards="".join(cards),
            endpoint=json.dumps(server_url.rstrip("/")),
            jobsdir=json.dumps(jobs_dir))

        html_path = ""
        if write_html:
            html_path = os.path.join(destination, "review.html")
            with open(html_path, "w", encoding="utf-8") as handle:
                handle.write(page)

        excalidraw_path = ""
        if write_excalidraw:
            excalidraw_path = self._write_board(destination, jobs, axes, recipe,
                                                thumb_data, tile_size=int(thumb_size))

        report = "\n".join([
            "REVIEW BOARD",
            "  masters      : %d" % len(jobs),
            "  contact sheet: %s" % html_path,
            "  excalidraw   : %s" % (excalidraw_path or "(needs 2 axes)"),
            "  decisions post to %s/arkennemasis/variation/decide" % server_url.rstrip("/"),
            "",
            "STATUS",
        ] + ["  %-12s %d" % (k, v) for k, v in sorted(counts.items())])
        print("[arkennemasis] review board: %d masters -> %s" % (len(jobs), html_path))
        return (html_path, excalidraw_path, report)

    # ── The Excalidraw matrix ────────────────────────────────────────────────
    # An .excalidraw file is JSON: elements positioned on an infinite canvas, plus a
    # `files` map of embedded images. So the board is a layout problem, not a rendering
    # one — no image tooling and no ComfyUI node needed to produce it.
    #
    # The layout is driven entirely by the recipe: axis 1 becomes the columns, axis 2
    # the rows, and any further axes repeat the whole grid as stacked sections. Nothing
    # about the product is assumed, so a 5x8 pendant and a 12x4x5 sneaker both come out
    # organised without the template changing.

    TILE = 320              # default; overridden per run from thumb_size
    GAP = 28                # between tiles
    ROW_LABEL_W = 300       # left gutter for the row axis's labels
    HEADER_H = 96           # strip above the grid for the column axis's labels
    SWATCH = 46             # colour chip beside a label that has a hex
    _k = 1.0                # type scale; _write_board resets it from the real tile
    TITLE_H = 150
    SECTION_GAP = 190

    def _el(self, kind, x, y, w, h, **extra):
        element = {
            "type": kind, "id": "e%d" % self._seq, "x": round(x), "y": round(y),
            "width": round(w), "height": round(h), "angle": 0,
            "strokeColor": "#1e1e1e", "backgroundColor": "transparent",
            "fillStyle": "solid", "strokeWidth": 1, "strokeStyle": "solid",
            "roughness": 0, "opacity": 100, "groupIds": [], "frameId": None,
            "roundness": None, "seed": self._seq + 1, "version": 1, "versionNonce": 1,
            "isDeleted": False, "boundElements": None, "updated": 1, "link": None,
            "locked": False,
        }
        element.update(extra)
        self._seq += 1
        return element

    def _badge(self, x, y, label, colour):
        """A small coloured pill naming the input format.

        Text on a tinted rectangle rather than a coloured word: at board scale a
        colour-coded word is easy to miss, and the whole point is that the format is
        visible at a glance beside the colour it produced.
        """
        return self._text(x, y, label, self._f(13), colour, width=self.TILE)

    def _f(self, size):
        """A font size from the 320-tile design, scaled to the tile actually in use."""
        return max(6, int(round(size * self._k)))

    def _text(self, x, y, content, size=20, colour="#1e1e1e", width=None, bold=False):
        content = str(content)
        width = width or max(60, int(len(content) * size * 0.56))
        return self._el(
            "text", x, y, width, size * 1.25,
            strokeColor=colour, text=content, fontSize=size,
            fontFamily=(5 if bold else 2),      # 5 = Excalifont bold-ish, 2 = normal
            textAlign="left", verticalAlign="top", baseline=size,
            containerId=None, originalText=content, lineHeight=1.25, autoResize=True)

    def _swatch(self, x, y, size, colour):
        return self._el("rectangle", x, y, size, size,
                        strokeColor="#868e96", backgroundColor=colour,
                        fillStyle="solid", roundness={"type": 3})

    def _write_board(self, destination, jobs, axes, recipe, thumb_data,
                     tile_size=320):
        """Lay every delivered image out as a labelled matrix and write the .excalidraw.

        Columns are axis 1, rows are axis 2, and a third or fourth axis repeats the
        whole grid as a titled section below. Row labels carry a colour chip and the
        hex where the axis has one, which is what makes the board reviewable at a
        glance rather than merely a wall of pictures.
        """
        if not axes:
            return ""

        # One tile = one thumbnail, so asking for bigger thumbnails genuinely produces a
        # bigger board. Previously the tile was pinned at 320 units whatever the
        # thumbnail resolution, which silently capped how large the rendered matrix
        # could ever be — for a 25-image sheet that is the difference between a
        # thumbnail strip and something a client can actually inspect.
        self.TILE = max(160, int(tile_size))
        self.ROW_LABEL_W = max(280, int(self.TILE * 0.85))
        self.HEADER_H = max(96, int(self.TILE * 0.30))
        self.SWATCH = max(28, int(self.TILE * 0.14))
        # Type scale: 1.0 at the original 320 tile, so an unchanged board renders
        # byte-identically and a larger one stays proportionate.
        self._k = self.TILE / 320.0

        self.TITLE_H = max(140, int(140 * self._k))

        self._seq = 0
        elements, files = [], {}

        def input_kind(value):
            """Which of the four specification formats this value arrived as.

            Shown on the board because "why does this one look different?" is usually
            answered by how it was specified, not by what it was called — a value given
            only as a photograph has no colour target at all, and a reviewer comparing
            it against a hex they assume exists is comparing against nothing.
            """
            has_ref = bool(value.get("refs") or value.get("ref"))
            has_hex = bool(value.get("hex"))
            has_text = bool(str(value.get("description") or "").strip())
            if has_ref and has_hex:
                return "IMAGE + HEX", "#7048e8"
            if has_ref:
                return "IMAGE", "#1098ad"
            if has_hex and has_text:
                return "HEX + TEXT", "#2f9e44"
            if has_hex:
                return "HEX", "#868e96"
            return "TEXT ONLY", "#e03131"

        def values_of(name):
            """(id, display, hex, kind, kind_colour) for an axis, in the recipe's order."""
            for axis in recipe.get("axes") or []:
                if axis.get("name") == name:
                    out = []
                    for v in axis.get("values") or []:
                        label, colour = input_kind(v)
                        out.append((v.get("id"), v.get("display") or v.get("id"),
                                    v.get("hex"), label, colour))
                    return out
            seen, out = set(), []
            for job in jobs:
                value = (job.get("axes") or {}).get(name)
                if value and value not in seen:
                    seen.add(value)
                    out.append((value, value, None, "", ""))
            return out

        col_axis = axes[0]
        row_axis = axes[1] if len(axes) > 1 else None

        # A ONE-AXIS product has no second axis to become rows, so every value landed in
        # a single row and a 30-value set rendered 45,881 pixels wide and 2,703 tall —
        # technically a matrix, useless as a sheet. With nothing to put down the side,
        # the axis wraps instead: values continue on the next line, roughly square, so
        # the set can actually be looked at.
        wrap_at = 0
        if row_axis is None:
            count = len(values_of(col_axis))
            if count > 8:
                # Slightly wider than tall reads better on a screen than a true square.
                wrap_at = max(4, int(round((count * 1.45) ** 0.5)))
        rest = axes[2:]

        all_values = values_of(col_axis)
        if wrap_at:
            # Each band is a slice of the same axis. Rows carry no label of their own —
            # the value's name sits under its own tile, where the eye already is.
            bands = [all_values[i:i + wrap_at]
                     for i in range(0, len(all_values), wrap_at)]
            columns = bands[0]
        else:
            bands = None
            columns = all_values
        rows = values_of(row_axis) if row_axis else [(None, "", None, "", "")]

        by_key = {}
        for job in jobs:
            axis_map = job.get("axes") or {}
            key = (axis_map.get(col_axis),
                   axis_map.get(row_axis) if row_axis else None,
                   tuple(axis_map.get(a) for a in rest))
            by_key[key] = job

        # Every combination of the axes beyond the first two becomes its own section.
        sections = [()]
        for name in rest:
            sections = [combo + (value[0],)
                        for combo in sections for value in values_of(name)]

        widest = (max((len(b) for b in bands), default=1) if bands
                  else len(columns))
        label_w = 0 if bands else self.ROW_LABEL_W
        grid_w = label_w + widest * (self.TILE + self.GAP)
        title = recipe.get("product_display") or recipe.get("product") or "Variations"

        elements.append(self._text(0, 0, title, self._f(46), "#1e1e1e", bold=True))
        subtitle = ("%d variations   ·   %s   ·   %d across, wrapped"
                    % (len(jobs), col_axis, widest) if bands else
                    "%d variations   ·   %s across   ·   %s down"
                    % (len(jobs), col_axis, row_axis or "—"))
        elements.append(self._text(0, 62 * self._k, subtitle, self._f(20),
                                   "#6b6d76", width=grid_w))

        y = self.TITLE_H
        for section in sections:
            if section:
                label = "   \u00b7   ".join(
                    "%s: %s" % (rest[i], section[i]) for i in range(len(section)))
                elements.append(self._text(0, y, label, self._f(28), "#1e1e1e",
                                           bold=True))
                y += 54 * self._k

            # A two-axis product draws one band: columns across, rows down. A ONE-axis
            # product has nothing to put down the side, so its single axis is wrapped
            # into several bands of `wrap_at` values instead — otherwise 30 values
            # render as one row 45,881 pixels wide, which is a matrix in name only.
            for band in (bands if bands else [columns]):
                for index, (_cid, display, chex, kind, kcolour) in enumerate(band):
                    x = label_w + index * (self.TILE + self.GAP)
                    if chex:
                        elements.append(self._swatch(x, y + 6, self.SWATCH, chex))
                        elements.append(self._text(
                            x + self.SWATCH + 12 * self._k, y + 14 * self._k, display,
                            self._f(20), "#1e1e1e", width=self.TILE))
                        elements.append(self._text(
                            x + self.SWATCH + 12 * self._k, y + 40 * self._k, chex,
                            self._f(15), "#6b6d76", width=self.TILE))
                    else:
                        elements.append(self._text(x, y + 18 * self._k, display,
                                                   self._f(20), "#1e1e1e",
                                                   width=self.TILE))
                    if kind:
                        elements.append(self._badge(x, y + 66 * self._k, kind, kcolour))
                y += self.HEADER_H

                for row_index, row in enumerate(rows):
                    rid, rdisplay, rhex, rkind, rkcolour = row
                    top = y + row_index * (self.TILE + self.GAP)

                    if row_axis:
                        label_y = top + self.TILE / 2 - 34 * self._k
                        if rhex:
                            elements.append(self._swatch(0, label_y, self.SWATCH, rhex))
                            elements.append(self._text(
                                self.SWATCH + 14 * self._k, label_y + 6 * self._k,
                                rdisplay, self._f(22), "#1e1e1e",
                                width=self.ROW_LABEL_W - self.SWATCH - 20))
                            elements.append(self._text(
                                self.SWATCH + 14 * self._k, label_y + 36 * self._k,
                                rhex, self._f(16), "#6b6d76",
                                width=self.ROW_LABEL_W - self.SWATCH - 20))
                        else:
                            elements.append(self._text(
                                0, label_y + 10 * self._k, rdisplay, self._f(22),
                                "#1e1e1e", width=self.ROW_LABEL_W - 20))
                        if rkind:
                            elements.append(self._badge(0, label_y + 64 * self._k,
                                                        rkind, rkcolour))

                    for col_index, (cid, _cd, _ch, _ck, _kc) in enumerate(band):
                        job = by_key.get((cid, rid, section))
                        x = label_w + col_index * (self.TILE + self.GAP)
                        if not job:
                            # An absent combination is stated, not silently skipped — a
                            # gap you cannot see is a gap you cannot chase.
                            elements.append(self._el(
                                "rectangle", x, top, self.TILE, self.TILE,
                                strokeColor="#d0d0d0", backgroundColor="transparent",
                                strokeStyle="dashed", roundness={"type": 3}))
                            elements.append(self._text(
                                x + 16 * self._k, top + self.TILE / 2 - 10 * self._k,
                                "not generated", self._f(16), "#adb5bd",
                                width=self.TILE - 32))
                            continue
                        try:
                            payload, pw, ph = thumb_data(job["master"], with_size=True)
                        except Exception:
                            continue
                        file_id = "f%04d" % len(files)
                        files[file_id] = {
                            "mimeType": "image/webp", "id": file_id,
                            "dataURL": "data:image/webp;base64,%s" % payload,
                            "created": 1, "lastRetrieved": 1,
                        }
                        # Fit inside the cell WITHOUT distorting, then centre what is
                        # left over. A portrait product in a square tile was being
                        # stretched sideways, which makes every image on the board look
                        # wrong in exactly the way this pipeline exists to prevent.
                        ratio = min(self.TILE / max(1, pw), self.TILE / max(1, ph))
                        draw_w, draw_h = pw * ratio, ph * ratio
                        ox = x + (self.TILE - draw_w) / 2.0
                        oy = top + (self.TILE - draw_h) / 2.0
                        elements.append(self._el(
                            "image", ox, oy, draw_w, draw_h,
                            strokeColor="transparent", status="saved", fileId=file_id,
                            scale=[1, 1]))
                        # The client's own filename under each tile: it is the primary
                        # key everywhere else, so a board that shows it can be talked
                        # about.
                        elements.append(self._text(
                            x, top + self.TILE + 8 * self._k,
                            os.path.splitext(job.get("filename")
                                             or job.get("key") or "")[0],
                            self._f(12), "#868e96", width=self.TILE))

                y += len(rows) * (self.TILE + self.GAP) + self.SECTION_GAP

        board = {
            "type": "excalidraw", "version": 2,
            "source": "arkennemasis-variation",
            "elements": elements, "files": files,
            "appState": {"gridSize": None, "viewBackgroundColor": "#ffffff"},
        }
        path = os.path.join(destination, "matrix.excalidraw")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(board, handle, ensure_ascii=False)
        print("[arkennemasis] excalidraw board: %d columns x %d rows x %d section(s), "
              "%d images" % (len(columns), len(rows), len(sections), len(files)))
        return path



class ArkStoreExport:
    CATEGORY = "arkennemasis/Variation"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("csv_path", "report", "row_count")
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Emit a completed store-import CSV in which every variation row carries its "
        "attribute values and its image. This is what converts the output from 'a "
        "folder of images' into 'the variations are live', which is the client's actual "
        "objective."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "intake_json": ("STRING", {
                    "multiline": True, "default": "{}", "forceInput": True,
                }),
                "jobs_dir": ("STRING", {"default": "variation/jobs"}),
                "out_dir": ("STRING", {"default": "variation/review"}),
            },
            "optional": {
                "url_prefix": ("STRING", {
                    "default": "",
                    "tooltip": "Public base URL the delivered images will live at. The "
                               "image column becomes <prefix>/<filename>. Blank writes "
                               "the local path, which is fine for a dry run but is not "
                               "importable.",
                }),
                "image_column": ("STRING", {
                    "default": "Images",
                    "tooltip": "WooCommerce's variation image column. Verify this "
                               "against the Product CSV Import Schema for the target "
                               "store before a real import — the exact header is "
                               "importer-version specific.",
                }),
                "attribute_prefix": ("STRING", {
                    "default": "meta:attribute_pa_",
                    "tooltip": "WooCommerce maps meta:-prefixed columns to post meta, "
                               "and a variation stores its selection under "
                               "attribute_pa_{taxonomy}. Change it for a different "
                               "platform.",
                }),
                "enabled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Off: write no store import file.",
                }),
                "only_status": ("STRING", {
                    "default": "delivered,approved",
                    "tooltip": "Comma-separated statuses to include. Shipping a flagged "
                               "cell to a live product page is the failure this guards.",
                }),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def run(self, intake_json, jobs_dir, out_dir, url_prefix="", image_column="Images",
            attribute_prefix="meta:attribute_pa_", enabled=True,
            only_status="delivered,approved"):
        if not enabled:
            print("[arkennemasis] store export: disabled")
            return ("", "STORE EXPORT DISABLED.", 0)
        intake = json.loads(intake_json or "{}")
        folder = resolve_dir(jobs_dir)
        destination = resolve_dir(out_dir, "variation/review")

        allowed = {s.strip().lower() for s in str(only_status or "").split(",") if s.strip()}
        product = (intake.get("product") or {})
        variants = intake.get("variants") or []
        axes = intake.get("axes") or []

        prefix = str(url_prefix or "").rstrip("/")
        rows, missing = [], []
        for variant in variants:
            filename = str(variant.get("filename") or "").strip()
            if not filename:
                continue
            key = filename
            job = load_job(folder, key)
            if job is None or (allowed and str(job.get("status", "")).lower() not in allowed):
                missing.append((filename, job.get("status") if job else "no job record"))
                continue

            delivered = job.get("delivered") or {}
            local = delivered.get("jpg") or delivered.get("png") or job.get("master") or ""
            image_value = ("%s/%s" % (prefix, os.path.basename(local)) if prefix and local
                           else local)

            row = {
                "Parent": product.get("display_name") or product.get("product"),
                "Type": "variation",
                "SKU": os.path.splitext(filename)[0],
                image_column: image_value,
                "status": job.get("status"),
            }
            for axis in axes:
                row["%s%s" % (attribute_prefix, axis)] = (job.get("axes") or {}).get(axis, "")
            rows.append(row)

        if not rows:
            report = ("No rows to export — no job reached a status in {%s}."
                      % ", ".join(sorted(allowed)))
            print("[arkennemasis] store export: %s" % report)
            return ("", report, 0)

        columns = ["Parent", "Type", "SKU"] + \
                  ["%s%s" % (attribute_prefix, a) for a in axes] + \
                  [image_column, "status"]
        path = os.path.join(destination, "%s-variations.csv"
                            % (product.get("product") or "product"))
        with open(path, "w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

        lines = [
            "STORE IMPORT — %s" % (product.get("display_name") or product.get("product")),
            "  file    : %s" % path,
            "  rows    : %d of %d variant rows" % (len(rows), len(variants)),
            "  axes    : %s" % ", ".join(axes),
            "  images  : %s" % ("%s/<filename>" % prefix if prefix
                                else "LOCAL PATHS — set url_prefix before a real import"),
        ]
        if missing:
            lines += ["", "NOT EXPORTED (%d)" % len(missing)]
            for filename, status in missing[:25]:
                lines.append("  %-64s %s" % (filename[:64], status))
            if len(missing) > 25:
                lines.append("  ... and %d more" % (len(missing) - 25))
        lines += ["", "Verify the image column header against the target store's CSV "
                      "import schema before a live import."]

        print("[arkennemasis] store export: %d rows -> %s" % (len(rows), path))
        return (path, "\n".join(lines), len(rows))


NODE_CLASS_MAPPINGS = {
    "ArkDeliver": ArkDeliver,
    "ArkReviewBoard": ArkReviewBoard,
    "ArkStoreExport": ArkStoreExport,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArkDeliver": "arkennemasis Deliver (format ladder + folders)",
    "ArkReviewBoard": "arkennemasis Review Board (contact sheet + excalidraw)",
    "ArkStoreExport": "arkennemasis Store Export (WooCommerce variation CSV)",
}

# -*- coding: utf-8 -*-
"""CSV-driven catalogue generation — the direct path.

The ten-stage pipeline in this package earns its complexity when a client hands over a
messy spreadsheet and the prompts must be DERIVED: intake maps the columns, the library
resolves every reference, a vision model compiles a recipe, a human approves it, and the
prompts are assembled from templates plus locks.

This module is for the case where all of that has already happened OUTSIDE the graph and
the answers are sitting in a CSV. One row is one output image, and it carries everything:

    input_1_model_ref   the identity image      -> image_1
    input_2_cloth_ref   the garment image       -> image_2
    full_prompt         the finished prompt     -> prompt
    output_filename     what to call the result

Nothing is inferred, nothing is generated, nothing is validated against a recipe. The CSV
IS the instruction set. That makes this path cheap and predictable, and it deliberately
gives up what the long path buys: no colour verification, no identity measurement, no
review board, no bounded retry. Use `ArkVerifyCandidate` downstream if any of that is
wanted back.

Paths in the CSV may be absolute, or relative to the CSV's own directory — which is what
makes a fixture folder portable.
"""

from __future__ import annotations

import csv
import io
import json
import os

import numpy as np
import torch

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# The columns a row must carry to be generatable. Named here rather than inline so the
# error can list exactly what was missing rather than failing on the first absent key.
REQUIRED = ("output_filename", "input_1_model_ref", "input_2_cloth_ref", "full_prompt")


class CatalogueError(RuntimeError):
    pass


def _load_image(path):
    """Path -> the (1, H, W, 3) float tensor ComfyUI calls an IMAGE."""
    from PIL import Image
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(array)[None, ...]


def _resolve(base_dir, path):
    """Absolute paths win; everything else is relative to the CSV that named it."""
    path = str(path or "").strip().strip('"')
    if not path:
        return ""
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(base_dir, path))


def _read_rows(csv_path):
    if not os.path.isfile(csv_path):
        raise CatalogueError("No CSV at %s" % csv_path)
    base = os.path.dirname(os.path.abspath(csv_path))
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise CatalogueError("%s has a header but no rows." % csv_path)

    missing = [c for c in REQUIRED if c not in rows[0]]
    if missing:
        raise CatalogueError(
            "%s is missing required column(s): %s. Present: %s"
            % (csv_path, ", ".join(missing), ", ".join(sorted(rows[0]))))

    out = []
    for number, row in enumerate(rows, start=2):        # 2 = first data line in the file
        record = {k: str(v or "").strip() for k, v in row.items() if k}
        record["_line"] = number
        record["_model_path"] = _resolve(base, record.get("input_1_model_ref"))
        record["_cloth_path"] = _resolve(base, record.get("input_2_cloth_ref"))
        out.append(record)
    return out


class ArkCatalogueLoad:
    """Read the catalogue, filter it, and report before a single image is paid for."""

    CATEGORY = "arkennemasis/Variation"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING", "INT", "STRING")
    RETURN_NAMES = ("rows_json", "count", "report")
    DESCRIPTION = (
        "Read a catalogue CSV where one row is one output image. Validates that every "
        "referenced file exists BEFORE anything is generated, because a missing "
        "reference discovered at row 60 has already cost 59 images."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "csv_path": ("STRING", {
                    "default": "",
                    "tooltip": "The catalogue CSV. Needs the columns output_filename, "
                               "input_1_model_ref, input_2_cloth_ref, full_prompt.",
                }),
            },
            "optional": {
                "only": ("STRING", {
                    "default": "",
                    "tooltip": "Filter, one 'column=value' per line. e.g. 'shot=front' "
                               "to render every front shot, or 'garment_code=m01' to "
                               "render one product. Blank runs everything.",
                }),
                "limit": ("INT", {
                    "default": 0, "min": 0, "max": 100000,
                    "tooltip": "Render at most this many rows. 0 = all. Set 1 for the "
                               "first end-to-end test.",
                }),
                "offset": ("INT", {
                    "default": 0, "min": 0, "max": 100000,
                    "tooltip": "Skip this many matching rows first. With limit, renders "
                               "a deliberate slice.",
                }),
                "output_dir": ("STRING", {
                    "default": "catalogue",
                    "tooltip": "Where results are written, under ComfyUI/output. Also "
                               "what skip_existing looks in.",
                }),
                "skip_existing": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Leave out rows whose output file already exists. This is "
                               "how a part-finished run resumes without paying twice.",
                }),
                "strict": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Fail the run if any row's model or cloth reference is "
                               "missing from disk. Off = skip those rows and carry on.",
                }),
            },
        }

    @classmethod
    def IS_CHANGED(cls, csv_path="", **kwargs):
        try:
            return "%s|%s" % (csv_path, os.path.getmtime(csv_path))
        except Exception:
            return csv_path

    def run(self, csv_path, only="", limit=0, offset=0, output_dir="catalogue",
            skip_existing=True, strict=True):
        rows = _read_rows(csv_path)
        total = len(rows)
        lines = ["CATALOGUE: %s" % csv_path, "  rows in file      : %d" % total]

        filters = []
        for raw in str(only or "").splitlines():
            raw = raw.strip()
            if not raw or "=" not in raw:
                continue
            column, _, value = raw.partition("=")
            filters.append((column.strip(), value.strip()))
        if filters:
            for column, value in filters:
                if column not in rows[0]:
                    raise CatalogueError(
                        "Filter column '%s' is not in the CSV. Columns: %s"
                        % (column, ", ".join(sorted(k for k in rows[0] if not k.startswith("_")))))
                rows = [r for r in rows if r.get(column) == value]
            lines.append("  after filter      : %d  (%s)"
                         % (len(rows), ", ".join("%s=%s" % f for f in filters)))

        # Existence is checked across the WHOLE filtered set before any slicing, so a
        # broken reference is reported even when this run would not have reached it.
        broken = []
        for row in rows:
            for label, key in (("model", "_model_path"), ("cloth", "_cloth_path")):
                if not os.path.isfile(row[key]):
                    broken.append("line %s (%s): %s ref not found -> %s"
                                  % (row["_line"], row.get("output_filename", "?"),
                                     label, row[key] or "(blank)"))
        if broken:
            head = "\n".join("    " + b for b in broken[:12])
            more = "" if len(broken) <= 12 else "\n    ... and %d more" % (len(broken) - 12)
            if strict:
                raise CatalogueError(
                    "%d row(s) reference a file that is not on disk:\n%s%s\n"
                    "Fix the paths, or switch strict off to skip these rows."
                    % (len(broken), head, more))
            keep = [r for r in rows
                    if os.path.isfile(r["_model_path"]) and os.path.isfile(r["_cloth_path"])]
            lines.append("  UNREADABLE, skipped: %d" % (len(rows) - len(keep)))
            rows = keep

        if skip_existing:
            try:
                import folder_paths
                root = folder_paths.get_output_directory()
            except Exception:
                root = os.path.join(os.getcwd(), "output")
            target = output_dir if os.path.isabs(output_dir) else os.path.join(root, output_dir)
            before = len(rows)
            rows = [r for r in rows
                    if not os.path.isfile(os.path.join(target, r["output_filename"]))]
            if before != len(rows):
                lines.append("  already delivered : %d  (skipped)" % (before - len(rows)))

        if offset:
            rows = rows[offset:]
        if limit:
            rows = rows[:limit]

        lines.append("  TO GENERATE       : %d" % len(rows))
        if rows:
            lines += ["", "First 8:"]
            for r in rows[:8]:
                lines.append("   %-46s  %s + %s"
                             % (r.get("output_filename", "")[:46],
                                os.path.basename(r["_model_path"]),
                                os.path.basename(r["_cloth_path"])))
        else:
            lines += ["", "Nothing to do — everything matching is already delivered."]

        report = "\n".join(lines)
        print("[arkennemasis] catalogue: %d/%d rows to generate" % (len(rows), total))
        return (json.dumps(rows, ensure_ascii=False), len(rows), report)


class ArkCatalogueFanOut:
    """One chain of nodes, every row, concurrently."""

    CATEGORY = "arkennemasis/Variation"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING", "STRING", "STRING", "INT", "INT")
    RETURN_NAMES = ("model_image", "cloth_image", "prompt", "filename", "sku",
                    "index", "count")
    # Everything but `count` fans out — ComfyUI runs the downstream chain once per row,
    # which is what lets the generator's own concurrency gate overlap calls. Same
    # mechanism as ArkCellFanOut; see the note there.
    OUTPUT_IS_LIST = (True, True, True, True, True, True, False)
    DESCRIPTION = (
        "Fan the catalogue out so one chain renders every row concurrently. "
        "model_image -> image_1, cloth_image -> image_2, prompt -> prompt."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "rows_json": ("STRING", {
                    "multiline": True, "default": "[]", "forceInput": True,
                }),
            },
        }

    def run(self, rows_json):
        rows = json.loads(rows_json or "[]")
        if not rows:
            raise CatalogueError(
                "ArkCatalogueFanOut: nothing to generate. Either every row is already "
                "delivered, or the filter on Catalogue Load matched no rows.")

        out = {name: [] for name in self.RETURN_NAMES}
        for index, row in enumerate(rows):
            out["model_image"].append(_load_image(row["_model_path"]))
            out["cloth_image"].append(_load_image(row["_cloth_path"]))
            out["prompt"].append(row.get("full_prompt", ""))
            out["filename"].append(row.get("output_filename", "%04d.png" % (index + 1)))
            out["sku"].append(row.get("sku", ""))
            out["index"].append(index)

        print("[arkennemasis] catalogue fan-out: %d rows — the chain below runs once "
              "per row, concurrently" % len(rows))
        return tuple(out[n] for n in self.RETURN_NAMES[:-1]) + (len(rows),)


class ArkCatalogueSave:
    """Save under the CSV's own filename, overwriting in place."""

    CATEGORY = "arkennemasis/Variation"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("path",)
    OUTPUT_NODE = True
    INPUT_IS_LIST = True
    DESCRIPTION = (
        "Write each generated image to the filename its catalogue row specified. "
        "Overwrites in place — no _v2, no timestamps, because several plausible files "
        "with no way to tell which is live is worse than one that is simply current."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "filename": ("STRING", {"forceInput": True}),
            },
            "optional": {
                "output_dir": ("STRING", {"default": "catalogue"}),
            },
        }

    def run(self, image, filename, output_dir=("catalogue",)):
        from PIL import Image

        directory = output_dir[0] if isinstance(output_dir, (list, tuple)) else output_dir
        directory = str(directory or "catalogue")
        if not os.path.isabs(directory):
            try:
                import folder_paths
                root = folder_paths.get_output_directory()
            except Exception:
                root = os.path.join(os.getcwd(), "output")
            directory = os.path.join(root, directory)
        os.makedirs(directory, exist_ok=True)

        names = filename if isinstance(filename, (list, tuple)) else [filename]
        written = []
        for index, batch in enumerate(image):
            name = str(names[index] if index < len(names) else names[-1]).strip()
            if not name:
                name = "%04d.png" % (index + 1)
            if not os.path.splitext(name)[1]:
                name += ".png"
            # A filename is a leaf, never a path — a stray separator in a CSV cell must
            # not be able to write outside the output directory.
            name = os.path.basename(name)

            array = batch[0] if batch.ndim == 4 else batch
            pixels = (array.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            path = os.path.join(directory, name)
            Image.fromarray(pixels).save(path)
            written.append(path)
            print("[arkennemasis] catalogue saved %s" % path)

        return ("\n".join(written),)


class ArkCatalogueBoard:
    """The delivered set as a board: one row per product, one column per shot.

    `ArkReviewBoard` builds its grid from durable job records, which the CSV path never
    writes. This reads the CSV and the output directory instead, so the board is a
    picture of what is actually on disk — including the gaps. A missing cell is drawn as
    an empty outline rather than silently closing up, because "which four am I still
    waiting on" is the question a review board exists to answer.
    """

    CATEGORY = "arkennemasis/Variation"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("excalidraw_path", "html_path", "report")
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Lay the delivered catalogue out as an .excalidraw board — a row per product, a "
        "column per shot — plus an HTML contact sheet. Missing images are drawn as gaps, "
        "so the board doubles as the run's to-do list."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "csv_path": ("STRING", {"default": ""}),
            },
            "optional": {
                "output_dir": ("STRING", {
                    "default": "catalogue",
                    "tooltip": "Where the generated images landed — the same value the "
                               "Save node used.",
                }),
                "board_dir": ("STRING", {
                    "default": "catalogue/review",
                    "tooltip": "Where the board and contact sheet are written.",
                }),
                "row_by": ("STRING", {
                    "default": "garment",
                    "tooltip": "CSV column that becomes the rows. 'garment' gives one "
                               "product per row.",
                }),
                "column_by": ("STRING", {
                    "default": "shot",
                    "tooltip": "CSV column that becomes the columns. 'shot' gives "
                               "front / detail / full / back across the page.",
                }),
                "thumb_size": ("INT", {
                    "default": 0, "min": 0, "max": 4096,
                    "tooltip": "Tile size. 0 = match the source, so nothing is "
                               "downsampled. Set 640 or 1024 if the board is too heavy "
                               "to open.",
                }),
                "write_excalidraw": ("BOOLEAN", {"default": True}),
                "write_html": ("BOOLEAN", {"default": True}),
                "after": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Wire Catalogue Save's `path` here. The value is unused - "
                               "it exists purely to make this node run AFTER the saves. "
                               "Without the edge the board is free to read the output "
                               "folder before the images land in it, and reports a set "
                               "as half-missing that was actually complete.",
                }),
            },
        }

    def run(self, csv_path, output_dir="catalogue", board_dir="catalogue/review",
            row_by="garment", column_by="shot", thumb_size=0,
            write_excalidraw=True, write_html=True, after=None):
        import base64
        import html as html_mod
        from PIL import Image

        from ..common.excalidraw import Board, INK, MUTED

        rows = _read_rows(csv_path)
        for column in (row_by, column_by):
            if column not in rows[0]:
                raise CatalogueError(
                    "No column '%s' in the CSV. Columns: %s"
                    % (column, ", ".join(sorted(k for k in rows[0] if not k.startswith("_")))))

        try:
            import folder_paths
            root = folder_paths.get_output_directory()
        except Exception:
            root = os.path.join(os.getcwd(), "output")
        images_dir = output_dir if os.path.isabs(output_dir) else os.path.join(root, output_dir)
        out_dir = board_dir if os.path.isabs(board_dir) else os.path.join(root, board_dir)
        os.makedirs(out_dir, exist_ok=True)

        # Preserve first-seen order rather than sorting: the CSV's order is the
        # operator's intended order, and alphabetising it scrambles the catalogue.
        row_keys, col_keys = [], []
        for r in rows:
            if r[row_by] not in row_keys:
                row_keys.append(r[row_by])
            if r[column_by] not in col_keys:
                col_keys.append(r[column_by])

        grid = {}
        for r in rows:
            grid[(r[row_by], r[column_by])] = os.path.join(images_dir, r["output_filename"])

        present = sum(1 for p in grid.values() if os.path.isfile(p))
        lines = ["CATALOGUE BOARD",
                 "  grid        : %d %s x %d %s" % (len(row_keys), row_by, len(col_keys), column_by),
                 "  delivered   : %d of %d" % (present, len(grid))]

        # tile size from the largest delivered image, so nothing is downsampled
        tile = thumb_size
        if not tile:
            widest = 0
            for path in grid.values():
                if os.path.isfile(path):
                    try:
                        with Image.open(path) as im:
                            widest = max(widest, im.width, im.height)
                    except Exception:
                        pass
            tile = widest or 512
        lines.append("  tile        : %d px" % tile)

        gap, label_w, header_h = int(tile * 0.06), int(tile * 1.1), int(tile * 0.22)
        excalidraw_path = html_path = ""

        if write_excalidraw:
            board = Board(source="arkennemasis-catalogue")
            board.text(0, 0, os.path.basename(csv_path), size=max(20, int(tile * 0.06)), bold=True)
            for c, col in enumerate(col_keys):
                board.text(label_w + c * (tile + gap), header_h * 0.5, str(col),
                           size=max(16, int(tile * 0.045)), bold=True)
            for r_i, rk in enumerate(row_keys):
                y = header_h + r_i * (tile + gap)
                board.text(0, y + tile * 0.4, str(rk), size=max(14, int(tile * 0.035)),
                           width=label_w - gap)
                for c_i, ck in enumerate(col_keys):
                    x = label_w + c_i * (tile + gap)
                    path = grid.get((rk, ck))
                    if path and os.path.isfile(path):
                        try:
                            with Image.open(path) as im:
                                im = im.copy()
                            im.thumbnail((tile, tile))
                            board.image(x, y, im.width, im.height, im)
                            continue
                        except Exception as exc:
                            print("[arkennemasis] board: unreadable %s (%s)" % (path, exc))
                    board.rect(x, y, tile, tile, stroke=MUTED)
                    board.text(x + tile * 0.3, y + tile * 0.45, "not yet",
                               size=max(12, int(tile * 0.03)), colour=MUTED)
            excalidraw_path = os.path.join(out_dir, "catalogue.excalidraw")
            with open(excalidraw_path, "w", encoding="utf-8") as handle:
                json.dump(board.document(), handle)
            lines.append("  board       : %s (%.1f MB)"
                         % (excalidraw_path, os.path.getsize(excalidraw_path) / 1e6))

        if write_html:
            parts = ["<!doctype html><meta charset='utf-8'>",
                     "<title>Catalogue review</title>",
                     "<style>body{font:14px system-ui;margin:24px;background:#fff;color:#111}",
                     "table{border-collapse:collapse}td,th{padding:6px;text-align:center;",
                     "vertical-align:bottom;border:1px solid #eee}img{max-width:240px;height:auto;display:block}",
                     ".miss{color:#999;font-style:italic;padding:40px 20px}</style>",
                     "<h1>Catalogue review</h1>",
                     "<p>%d of %d delivered.</p><table><tr><th></th>"
                     % (present, len(grid))]
            parts += ["<th>%s</th>" % html_mod.escape(str(c)) for c in col_keys]
            parts.append("</tr>")
            for rk in row_keys:
                parts.append("<tr><th style='text-align:left'>%s</th>" % html_mod.escape(str(rk)))
                for ck in col_keys:
                    path = grid.get((rk, ck))
                    if path and os.path.isfile(path):
                        try:
                            with Image.open(path) as im:
                                im = im.copy()
                            im.thumbnail((480, 480))
                            buf = io.BytesIO()
                            im.convert("RGB").save(buf, format="WEBP", quality=80)
                            uri = base64.b64encode(buf.getvalue()).decode("ascii")
                            parts.append("<td><img src='data:image/webp;base64,%s'>"
                                         "<div>%s</div></td>"
                                         % (uri, html_mod.escape(os.path.basename(path))))
                            continue
                        except Exception:
                            pass
                    parts.append("<td class='miss'>not yet</td>")
                parts.append("</tr>")
            parts.append("</table>")
            html_path = os.path.join(out_dir, "catalogue.html")
            with open(html_path, "w", encoding="utf-8") as handle:
                handle.write("".join(parts))
            lines.append("  contact sheet: %s (%.1f MB)"
                         % (html_path, os.path.getsize(html_path) / 1e6))

        missing = [("%s / %s" % (rk, ck)) for rk in row_keys for ck in col_keys
                   if not os.path.isfile(grid.get((rk, ck), ""))]
        if missing:
            lines += ["", "STILL MISSING — %d:" % len(missing)]
            lines += ["   - %s" % m for m in missing[:20]]
            if len(missing) > 20:
                lines.append("   ... and %d more" % (len(missing) - 20))

        report = "\n".join(lines)
        print("[arkennemasis] catalogue board: %d/%d delivered" % (present, len(grid)))
        return (excalidraw_path, html_path, report)


NODE_CLASS_MAPPINGS.update({
    "ArkCatalogueLoad": ArkCatalogueLoad,
    "ArkCatalogueFanOut": ArkCatalogueFanOut,
    "ArkCatalogueSave": ArkCatalogueSave,
    "ArkCatalogueBoard": ArkCatalogueBoard,
})

NODE_DISPLAY_NAME_MAPPINGS.update({
    "ArkCatalogueLoad": "arkennemasis Catalogue Load (CSV in, one row per image)",
    "ArkCatalogueFanOut": "arkennemasis Catalogue Fan-Out (one chain, N rows, concurrent)",
    "ArkCatalogueSave": "arkennemasis Catalogue Save (the CSV's own filename)",
    "ArkCatalogueBoard": "arkennemasis Catalogue Board (product x shot, Excalidraw + HTML)",
})

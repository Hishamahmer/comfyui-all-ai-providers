"""The cell matrix — the cartesian product of N axes, and one cell at a time.

A cell is one combination of one value from every axis, for one plate, and it maps 1:1
to one output image. Cell count is the product of all axis lengths times the plate count.

**N is read from the recipe.** Two nested loops over exactly two axes is a defect, not a
simplification: real products arrive with one axis or six, and a pipeline that assumes
two is a single-product pipeline wearing a configuration file. `itertools.product` over
whatever axes the recipe declares is the whole implementation, and it is correct for
every N without a special case.

Two nodes, mirroring the loop shape the rest of this pack already uses: one builds the
list, the other indexes into it, so a `for` loop on the canvas drives the run and the
per-cell chain is written once rather than duplicated per axis.
"""

from __future__ import annotations

import itertools
import json
import os

import numpy as np
import torch

from .schema import (
    ValidationError,
    axis_by_name,
    canonical,
    render_filename,
    value_by_id,
)


def _load_image(path):
    from PIL import Image
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(array)[None, ...]


def build_cells(recipe, variants=None, plates=None):
    """Every cell for this recipe, in a stable order.

    When the client declared explicit VARIANTS rows those win — a client who omitted
    combinations meant to omit them, and generating the full product would deliver
    images they never asked for. With no declared variants the full cartesian product
    is generated instead.
    """
    axes = sorted(recipe.get("axes") or [], key=lambda a: a.get("order") or 99)
    if not axes:
        raise ValidationError("Recipe declares no axes.")

    plate_list = plates or recipe.get("plates") or [{"id": "front"}]
    plate_ids = [str(p.get("id") or "front") for p in plate_list]

    cells = []

    if variants:
        for row in variants:
            values = {}
            complete = True
            for axis in axes:
                name = axis["name"]
                raw = row.get("axis:" + name)
                value_id = canonical(raw)
                if not value_id or value_by_id(axis, value_id) is None:
                    complete = False
                    break
                values[name] = value_id
            if not complete:
                continue
            filename = str(row.get("filename") or "").strip()
            for plate_id in plate_ids:
                cells.append(_make_cell(recipe, axes, values, plate_id,
                                        len(plate_ids) > 1, filename))
    else:
        # The general case: N axes, whatever N is.
        value_lists = [[v["id"] for v in (axis.get("values") or [])] for axis in axes]
        for combination in itertools.product(*value_lists):
            values = {axes[i]["name"]: combination[i] for i in range(len(axes))}
            for plate_id in plate_ids:
                cells.append(_make_cell(recipe, axes, values, plate_id,
                                        len(plate_ids) > 1, ""))
    return cells


def _make_cell(recipe, axes, values, plate_id, multi_plate, filename=""):
    """One cell record. `key` is the primary key for every artefact this cell produces."""
    if not filename:
        filename = render_filename(recipe, values)

    key = filename
    if multi_plate:
        stem, ext = os.path.splitext(filename)
        key = "%s__%s%s" % (stem, plate_id, ext)

    # The target axis is the one that paints — with several axes every one of them
    # paints something, so each axis contributes its own instruction slot. `order` 1 is
    # the axis changing the largest area and leads the prompt.
    slots = []
    for axis in axes:
        value = value_by_id(axis, values[axis["name"]])
        slots.append({
            "axis": axis["name"],
            "value": value["id"],
            "display": value.get("display") or value["id"],
            "region": axis.get("paints"),
            # The VALUE's type, not the axis's. One axis routinely mixes formats, and
            # this slot is what selects the prompt template downstream — taking the
            # axis's summary type would describe a hex-only value as if a reference
            # photograph had been supplied for it.
            "spec_type": value.get("spec_type") or axis.get("spec_type"),
            "hex": value.get("hex"),
            "description": value.get("description") or "",
            "ref": value.get("ref"),
            "refs": value.get("refs") or [],
            "swatch": value.get("swatch"),
            "filename_token": value.get("filename_token") or value["id"],
            "order": axis.get("order") or 99,
        })

    return {
        "key": key,
        "filename": filename,
        "product": recipe.get("product"),
        "plate": plate_id,
        "axes": values,
        "slots": slots,
        "regions": [s["region"] for s in slots if s["region"]],
    }


def _take(cells, limit, pick):
    """Keep `limit` cells — either the first ones, or a spread across the matrix.

    Taking the first N is the wrong sample for a test run, and expensively so. Cells
    come out in cartesian order, so the first five of a 6x5 matrix are five variations
    of the SAME leading value: one colour, and whichever specification format that
    value happens to use. The operator pays for five images and learns about one case.

    A spread walks the full list at even intervals, so the same five images cover
    different values on both axes and — because formats are distributed across the
    values — different specification formats too. The first and last cells are always
    included, since those are the corners of the matrix a reviewer looks at first.
    """
    limit = max(0, int(limit))
    if not limit or limit >= len(cells):
        return cells
    if str(pick or "").startswith("first"):
        return cells[:limit]
    if limit == 1:
        return [cells[0]]
    step = (len(cells) - 1) / float(limit - 1)
    seen, out = set(), []
    for i in range(limit):
        index = int(round(i * step))
        if index not in seen:
            seen.add(index)
            out.append(cells[index])
    # Rounding can collide on a short list; top up in order so the count is honoured.
    for i, cell in enumerate(cells):
        if len(out) >= limit:
            break
        if i not in seen:
            seen.add(i)
            out.append(cell)
    return out


class ArkCellMatrix:
    CATEGORY = "arkennemasis/Variation"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING", "INT", "STRING")
    RETURN_NAMES = ("cells_json", "count", "report")
    DESCRIPTION = (
        "Expand the recipe into every cell: the cartesian product of all N axes times "
        "the plates. Honours the client's declared variant rows when intake supplied "
        "them, so combinations they deliberately omitted are not generated."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "recipe_json": ("STRING", {
                    "multiline": True, "default": "{}", "forceInput": True,
                }),
            },
            "optional": {
                "intake_json": ("STRING", {
                    "multiline": True, "default": "", "forceInput": True,
                    "tooltip": "Optional. When present, only the variant rows the "
                               "client actually declared are generated.",
                }),
                "limit": ("INT", {
                    "default": 0, "min": 0, "max": 100000,
                    "tooltip": "Generate at most this many cells. 0 = all of them. Use a "
                               "small number for the first end-to-end test — three cells "
                               "on one axis is the spec's own step 2.",
                }),
                "offset": ("INT", {
                    "default": 0, "min": 0, "max": 100000,
                    "tooltip": "Skip this many cells first. With limit, lets you run a "
                               "specific slice without touching the rest.",
                }),
                "only_axis_value": ("STRING", {
                    "default": "",
                    "tooltip": "Optional filter, 'axis=value' per line. Pin one axis to "
                               "one value to sweep another axis on its own — which is "
                               "exactly what axis-level review needs.",
                }),
                # APPENDED LAST: widgets_values is positional, so a new widget anywhere
                # else would shift every later value in graphs already saved.
                "pick": (["spread across the matrix", "first ones"], {
                    "tooltip": "WHICH cells the limit keeps. 'spread' walks the whole "
                               "matrix at even intervals, so a 5-image test covers "
                               "different colours AND different specification formats "
                               "instead of five variations of the first one. 'first "
                               "ones' takes them in order — use it to resume a "
                               "deliberate slice with offset.",
                }),
            },
        }

    def run(self, recipe_json, intake_json="", limit=0, offset=0, only_axis_value="",
            pick="spread across the matrix"):
        recipe = json.loads(recipe_json or "{}")
        if not recipe.get("axes"):
            raise ValidationError("ArkCellMatrix: the recipe has no axes.")

        variants = None
        if str(intake_json or "").strip():
            intake = json.loads(intake_json)
            variants = intake.get("variants") or None

        cells = build_cells(recipe, variants)
        total = len(cells)

        pins = {}
        for line in str(only_axis_value or "").splitlines():
            if "=" in line:
                axis_name, _, value = line.partition("=")
                pins[canonical(axis_name)] = canonical(value)
        if pins:
            cells = [c for c in cells
                     if all(canonical(c["axes"].get(a)) == v for a, v in pins.items())]

        if offset:
            cells = cells[int(offset):]
        if limit:
            cells = _take(cells, int(limit), pick)

        axes = sorted(recipe.get("axes") or [], key=lambda a: a.get("order") or 99)
        lines = [
            "CELL MATRIX — %s" % (recipe.get("product_display") or recipe.get("product")),
            "  axes    : %d" % len(axes),
        ]
        for axis in axes:
            lines.append("    %-18s %3d values  paints %-14s (%s)"
                         % (axis.get("name"), len(axis.get("values") or []),
                            axis.get("paints"), axis.get("spec_type")))
        lines += [
            "  plates  : %d" % len(recipe.get("plates") or [{}]),
            "  source  : %s" % ("client's declared variant rows" if variants
                                else "full cartesian product"),
            "  total   : %d cells" % total,
        ]
        if pins:
            lines.append("  pinned  : %s" % ", ".join("%s=%s" % kv for kv in pins.items()))
        if offset or limit:
            lines.append("  slice   : offset %d, limit %s" % (offset, limit or "none"))
        lines.append("  selected: %d cells" % len(cells))
        if len(cells) != total:
            lines.append("  NOTE: %d of %d cells are NOT in this run."
                         % (total - len(cells), total))
        lines += ["", "FIRST CELLS"]
        for cell in cells[:8]:
            lines.append("  %s" % cell["key"])
        if len(cells) > 8:
            lines.append("  ... and %d more" % (len(cells) - 8))

        print("[arkennemasis] cell matrix: %d of %d cells over %d axes"
              % (len(cells), total, len(axes)))
        return (json.dumps(cells, ensure_ascii=False), len(cells), "\n".join(lines))


class ArkCellAt:
    CATEGORY = "arkennemasis/Variation"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "IMAGE", "IMAGE", "IMAGE", "STRING",
                    "STRING", "STRING", "STRING", "INT")
    RETURN_NAMES = ("cell_json", "key", "target_region", "reference", "reference_2",
                    "reference_3", "target_hex", "target_rgb", "spec_type", "summary",
                    "index")
    DESCRIPTION = (
        "One cell out of the matrix, by index. Wire a for-loop's index in and the "
        "per-cell chain is written once no matter how many axes the product has."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cells_json": ("STRING", {
                    "multiline": True, "default": "[]", "forceInput": True,
                }),
                "index": ("INT", {
                    "default": 0, "min": 0, "max": 1000000,
                    "tooltip": "0-based. Wire a for-loop's index here.",
                }),
            },
        }

    @classmethod
    def IS_CHANGED(cls, cells_json="", index=0, **kwargs):
        return "%s|%s" % (hash(str(cells_json)), index)

    def run(self, cells_json, index=0):
        cells = json.loads(cells_json or "[]")
        if not cells:
            raise ValidationError("ArkCellAt: the cell list is empty.")
        position = max(0, min(int(index), len(cells) - 1))
        cell = cells[position]

        # The leading slot is the axis with order 1 — the one changing the largest area.
        # It supplies the reference image and the colour target for verification.
        slots = sorted(cell.get("slots") or [], key=lambda s: s.get("order") or 99)
        lead = slots[0] if slots else {}

        # ONLY real material photographs are handed on as reference images, and EVERY
        # one of them is, across every axis of the cell — up to the three sockets here.
        #
        # A hex value also has a rendered swatch in the library, and falling back to it
        # was actively harmful: a flat colour chip attached to an image-editing request
        # reads as "make this region look like THIS" — a flat, textureless fill — when
        # the hex only ever specified a tint for a material that keeps its own
        # structure. Colour goes in the prompt text; the swatch is for the operator to
        # eyeball in the library preview and nothing else.
        #
        # Multiple references are kept SEPARATE, never merged. A flat material sample
        # and a lit in-situ example say different things, and averaging them produces a
        # third material matching neither — so each keeps its own socket and its own
        # declared role.
        gathered = []
        for slot in slots:
            for reference_record in (slot.get("refs")
                                     or ([{"path": slot["ref"], "role": "material"}]
                                         if slot.get("ref") else [])):
                path = reference_record.get("path") or reference_record.get("url")
                if path and os.path.isfile(path):
                    gathered.append((slot.get("axis"), reference_record.get("role")
                                     or "material", path))

        images, roles = [], []
        for axis_name, role, path in gathered[:3]:
            try:
                images.append(_load_image(path))
                roles.append("%s/%s" % (axis_name, role))
            except Exception as exc:
                print("[arkennemasis] cell %s: reference unreadable (%s)"
                      % (cell.get("key"), exc))
        if len(gathered) > 3:
            print("[arkennemasis] cell %s: %d references available, 3 sockets — %s not "
                  "attached" % (cell.get("key"), len(gathered),
                                ", ".join(g[1] for g in gathered[3:])))

        hex_only = [s.get("axis") for s in slots if s.get("hex") and not s.get("refs")
                    and not s.get("ref")]
        if hex_only and not images:
            print("[arkennemasis] cell %s: axes %s are hex specs — sending NO reference "
                  "image (colour is stated in the prompt; a flat swatch would ask the "
                  "model for a flat fill)" % (cell.get("key"), ", ".join(hex_only)))

        # An 8x8 black tile is the pack's "nothing here" IMAGE. It must never reach the
        # generator, so leave the matching socket unwired when there is no reference.
        blank = torch.zeros((1, 8, 8, 3), dtype=torch.float32)
        while len(images) < 3:
            images.append(blank)
        reference, reference_2, reference_3 = images[0], images[1], images[2]

        target_rgb = ""
        if lead.get("hex"):
            try:
                from .colour import hex_to_rgb01
                rgb = hex_to_rgb01(lead["hex"])
                target_rgb = "%d, %d, %d" % (round(rgb[0] * 255), round(rgb[1] * 255),
                                             round(rgb[2] * 255))
            except ValueError:
                pass

        summary = "\n".join(
            ["cell %d/%d  %s" % (position + 1, len(cells), cell.get("key")),
             "  plate  : %s" % cell.get("plate")] +
            ["  %-18s = %-26s -> %-14s %s"
             % (s.get("axis"), s.get("value"), s.get("region") or "?",
                s.get("hex") or (os.path.basename(s["ref"]) if s.get("ref") else ""))
             for s in slots])

        if roles:
            summary += "\n  references attached: %s" % ", ".join(roles)

        print("[arkennemasis] cell %d/%d: %s%s"
              % (position + 1, len(cells), cell.get("key"),
                 (" (%d reference image(s))" % len(roles)) if roles else ""))
        return (json.dumps(cell, ensure_ascii=False), cell.get("key", ""),
                lead.get("region") or "", reference, reference_2, reference_3,
                lead.get("hex") or "", target_rgb, lead.get("spec_type") or "",
                summary, position)


NODE_CLASS_MAPPINGS = {
    "ArkCellMatrix": ArkCellMatrix,
    "ArkCellAt": ArkCellAt,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArkCellMatrix": "arkennemasis Cell Matrix (N-axis cartesian product)",
    "ArkCellAt": "arkennemasis Cell At (one cell by index)",
}


class ArkCellFanOut:
    CATEGORY = "arkennemasis/Variation"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "IMAGE", "IMAGE", "IMAGE", "STRING",
                    "STRING", "STRING", "INT", "INT")
    RETURN_NAMES = ("cell_json", "key", "target_region", "reference", "reference_2",
                    "reference_3", "target_hex", "target_rgb", "spec_type", "index",
                    "count")
    # Everything but `count` fans out: ComfyUI runs each downstream node once per cell.
    # This is what makes concurrency possible at all — `easy forLoop` is strictly
    # sequential, so an async generator inside one can only ever have a single call in
    # flight however high its concurrency widget is set. With a list there are N chains,
    # and the generator's own `concurrency_gate` decides how many overlap.
    OUTPUT_IS_LIST = (True, True, True, True, True, True, True, True, True, True, False)
    DESCRIPTION = (
        "Fan the cell list out so ONE chain of nodes renders every cell, natively and "
        "CONCURRENTLY. A drop-in replacement for Cell At + a for-loop: set the "
        "generator's run_mode to 'all at once' and cap it with max_concurrent."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cells_json": ("STRING", {
                    "multiline": True, "default": "[]", "forceInput": True,
                }),
            },
        }

    def run(self, cells_json):
        cells = json.loads(cells_json or "[]")
        if not cells:
            raise ValidationError("ArkCellFanOut: the cell list is empty.")

        blank = torch.zeros((1, 8, 8, 3), dtype=torch.float32)
        out = {name: [] for name in self.RETURN_NAMES}
        at = ArkCellAt()

        for index in range(len(cells)):
            # Reuse ArkCellAt so reference resolution, the hex-only guard and the RGB
            # conversion exist in exactly one place rather than two that can drift.
            (cell_json, key, region, ref, ref2, ref3, target_hex, target_rgb,
             spec_type, _summary, _i) = at.run(cells_json=cells_json, index=index)
            out["cell_json"].append(cell_json)
            out["key"].append(key)
            out["target_region"].append(region)
            out["reference"].append(ref if ref is not None else blank)
            out["reference_2"].append(ref2 if ref2 is not None else blank)
            out["reference_3"].append(ref3 if ref3 is not None else blank)
            out["target_hex"].append(target_hex)
            out["target_rgb"].append(target_rgb)
            out["spec_type"].append(spec_type)
            out["index"].append(index)

        print("[arkennemasis] fan-out: %d cells — the chain below runs once per cell, "
              "concurrently" % len(cells))
        return tuple(out[name] for name in self.RETURN_NAMES[:-1]) + (len(cells),)


NODE_CLASS_MAPPINGS["ArkCellFanOut"] = ArkCellFanOut
NODE_DISPLAY_NAME_MAPPINGS["ArkCellFanOut"] =     "arkennemasis Cell Fan-Out (one chain, N cells, concurrent)"

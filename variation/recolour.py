"""The non-generative path, and the router that chooses when to use it.

A significant share of product-variation work is a masked colour transform that requires
no generative model at all. Where the spec is a hex code and the region is a reasonably
simple surface — painted, lacquered, dyed, anodised — the deterministic route is
simultaneously faster, cheaper, and more accurate, and it **cannot drift by
construction**: no pixel outside the mask is touched, so the camera, the framing, the
shadow and the product's geometry are mathematically unchanged rather than merely asked
to stay put.

    property        generative path              deterministic path
    frame drift     possible on every call       impossible by construction
    colour accuracy approximate, must be checked exact
    cost per image  a per-call API charge        negligible
    speed           seconds                      milliseconds

Generation is the expensive and risky path. Spend it only where the material genuinely
has to be invented — figured timber, veined marble, woven textile. Routing that decision
automatically, per axis, is a primary source of margin, and `ArkGenRoute` makes it a
recipe field rather than a human's memory.

The router's inputs are LAZY, exactly like the pack's image-engine switch: the branch it
does not choose is never evaluated, so an axis routed to the deterministic path costs no
API call at all rather than making one and discarding it.
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch

from .colour import recolour_region
from .schema import ValidationError, canonical, normalise_hex


def _load_mask(path, shape):
    from PIL import Image
    with Image.open(path) as image:
        image = image.convert("L")
        if image.size != (shape[1], shape[0]):
            image = image.resize((shape[1], shape[0]), Image.NEAREST)
        return np.asarray(image).astype(np.float32) / 255.0

try:
    from comfy_execution.graph_utils import ExecutionBlocker
except ImportError:
    from execution import ExecutionBlocker

GENERATIVE = "generative — the image model (works for any product)"
DETERMINISTIC = "deterministic — mask + colour transform (needs hand-authored masks)"
AUTO = "auto — from the recipe's route field, else generative"
# Generative leads because it is the only route that generalises. The deterministic
# transform is faster, free and exact, but it requires a hand-authored mask per region
# per plate — which is fine for one known product and impossible for an arbitrary
# stream of sofas, shoes, t-shirts and cars. It is an optimisation to opt INTO for a
# high-volume product whose masks are worth authoring, never the default.
ROUTES = [GENERATIVE, AUTO, DETERMINISTIC]


def _mask_array(mask, shape):
    array = mask.detach().cpu().numpy() if hasattr(mask, "detach") else np.asarray(mask)
    array = array.astype(np.float32)
    while array.ndim > 2:
        array = array[0] if array.shape[0] == 1 else array[..., 0]
    if array.max() > 1.0:
        array = array / 255.0
    if array.shape != shape:
        try:
            import cv2
            array = cv2.resize(array, (shape[1], shape[0]),
                               interpolation=cv2.INTER_LINEAR)
        except Exception:
            raise ValidationError(
                "Region mask is %s but the plate is %s, and cv2 is unavailable to "
                "resize it." % (array.shape, shape))
    return np.clip(array, 0.0, 1.0)


class ArkRegionRecolour:
    CATEGORY = "arkennemasis/Variation"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", "STRING", "INT")
    RETURN_NAMES = ("image", "report", "pixels_changed")
    DESCRIPTION = (
        "Retint one masked region to an exact hex while keeping the plate's own "
        "shading, highlights and shadow. No model, no API call, and no possibility of "
        "frame drift — every pixel outside the mask is bit-identical to the plate."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "plate": ("IMAGE", {"tooltip": "The locked base plate."}),
            },
            "optional": {
                "cell_json": ("STRING", {
                    "multiline": True, "default": "", "forceInput": True,
                    "tooltip": "Preferred. Wire the cell and this node paints EVERY one "
                               "of its hex axes, each into its own region, in one pass. "
                               "With a product whose axes all specify colours, wiring "
                               "only region_mask/target_hex would paint the leading axis "
                               "and silently leave every other one untouched.",
                }),
                "plate_lock_json": ("STRING", {
                    "multiline": True, "default": "", "forceInput": True,
                    "tooltip": "Supplies each region's mask, so cell_json alone is "
                               "enough to resolve everything.",
                }),
                "region_mask": ("MASK", {
                    "tooltip": "Single-region fallback, used only when cell_json is not "
                               "wired.",
                }),
                "target_hex": ("STRING", {
                    "default": "", "forceInput": True,
                    "tooltip": "Single-region fallback, used only when cell_json is not "
                               "wired.",
                }),
                "preserve_luma": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Keep the plate's lightness structure and only re-level "
                               "the region's average to the target. Off replaces "
                               "lightness too, which flattens the surface — almost "
                               "never what you want on a photograph.",
                }),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Blend towards the new colour. 1.0 is exact; lower only "
                               "for a tinted rather than repainted look.",
                }),
                "feather": ("FLOAT", {
                    "default": 1.5, "min": 0.0, "max": 32.0, "step": 0.5,
                    "tooltip": "Soften the mask edge by this many pixels so the "
                               "transition reads photographic rather than cut out. "
                               "Applied to the alpha only, never to the colour, so it "
                               "cannot bleed the new colour outside the region.",
                }),
            },
        }

    def _jobs(self, cell_json, plate_lock_json, region_mask, target_hex, shape):
        """`[(label, hex, mask)]` — every region this cell repaints, in axis order."""
        if not str(cell_json or "").strip():
            wanted = normalise_hex(target_hex)
            if not wanted:
                raise ValidationError(
                    "ArkRegionRecolour needs either a cell (preferred) or a hex colour; "
                    "got %r. This node is the deterministic path and only serves hex "
                    "specs — route reference-image axes to the generator." % (target_hex,))
            if region_mask is None:
                raise ValidationError("ArkRegionRecolour: no region_mask wired.")
            return [("(single)", wanted, _mask_array(region_mask, shape))]

        cell = json.loads(cell_json)
        lock = json.loads(plate_lock_json) if str(plate_lock_json or "").strip() else {}
        regions = lock.get("regions") or {}

        out = []
        missing = []
        for slot in sorted(cell.get("slots") or [], key=lambda s: s.get("order") or 99):
            wanted = normalise_hex(slot.get("hex"))
            if not wanted:
                continue                    # a reference-image axis; not ours to paint
            name = canonical(slot.get("region"))
            record = regions.get(name)
            path = record.get("mask") if isinstance(record, dict) else None
            if not path or not os.path.isfile(path):
                missing.append((slot.get("axis"), name))
                continue
            out.append((("%s=%s" % (slot.get("axis"), slot.get("value"))), wanted,
                        _load_mask(path, shape)))

        if missing:
            raise ValidationError(
                "The locked plate has no mask for %s. It has: %s. Those axes would "
                "paint nothing, so the run is stopped here rather than delivering "
                "images that were never changed."
                % ("; ".join("axis '%s' -> region '%s'" % m for m in missing),
                   ", ".join(sorted(regions)) or "no regions at all"))
        if not out:
            raise ValidationError(
                "Cell '%s' has no hex axis for this node to paint. Route it to the "
                "generator instead." % cell.get("key"))
        return out

    def run(self, plate, cell_json="", plate_lock_json="", region_mask=None,
            target_hex="", preserve_luma=True, strength=1.0, feather=1.5):
        array = plate[0].detach().cpu().numpy().astype(np.float32)[:, :, :3]
        shape = array.shape[:2]
        original = array.copy()

        jobs = self._jobs(cell_json, plate_lock_json, region_mask, target_hex, shape)

        result = array
        touched = np.zeros(shape, dtype=np.float64)
        lines = ["DETERMINISTIC RECOLOUR — %d region(s)" % len(jobs)]
        total = 0
        for label, wanted, mask in jobs:
            result, changed, applied = recolour_region(
                result, mask, wanted, strength=float(strength),
                preserve_luma=bool(preserve_luma), feather=float(feather))
            touched = np.maximum(touched, applied)
            total += changed
            lines.append("  %-34s %s  %6d px" % (label, wanted, changed))
            print("[arkennemasis] recolour %s -> %s (%d px)" % (label, wanted, changed))

        # Measured against the union of the FEATHERED supports — the pixels this node
        # actually claims to touch. Anything non-zero here would be a real defect.
        outside = touched <= 0.0
        untouched = (float(np.abs(result - original)[outside].max())
                     if outside.any() else 0.0)
        lines += [
            "  preserve luma   : %s" % preserve_luma,
            "  feather         : %.1f px" % float(feather),
            "  total changed   : %d px (%.2f%% of frame)"
            % (total, 100.0 * total / max(1, shape[0] * shape[1])),
            "  max change outside the touched area: %.6f  (0.000000 means the rest of "
            "the frame is provably identical)" % untouched,
            "  cost            : none — no model was called",
        ]
        if total == 0:
            raise ValidationError(
                "Every mask selected zero pixels, so this cell would be delivered "
                "unchanged. Check that the plate's region masks are not empty.")
        return (torch.from_numpy(result)[None, ...], "\n".join(lines), int(total))


class ArkGenRoute:
    CATEGORY = "arkennemasis/Variation"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "route_taken", "report")
    DESCRIPTION = (
        "Choose the deterministic or the generative path for this cell. Both inputs are "
        "lazy, so the branch not taken is never evaluated — an axis routed to the "
        "colour transform makes no API call at all."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "route": (ROUTES, {
                    "tooltip": "Auto reads the axis's 'route' field from the recipe, "
                               "falling back to: hex spec -> deterministic, anything "
                               "else -> generative.",
                }),
            },
            "optional": {
                "deterministic": ("IMAGE", {
                    "lazy": True,
                    "tooltip": "Wire Region Recolour here.",
                }),
                "generative": ("IMAGE", {
                    "lazy": True,
                    "tooltip": "Wire the image-model branch here.",
                }),
                "cell_json": ("STRING", {
                    "multiline": True, "default": "", "forceInput": True,
                }),
                "recipe_json": ("STRING", {
                    "multiline": True, "default": "", "forceInput": True,
                }),
            },
        }

    def _decide(self, route, cell_json="", recipe_json=""):
        if route == DETERMINISTIC:
            return "deterministic", "set explicitly on the node"
        if route == GENERATIVE:
            return "generative", "set explicitly on the node"

        try:
            cell = json.loads(cell_json or "{}")
        except ValueError:
            cell = {}
        try:
            recipe = json.loads(recipe_json or "{}")
        except ValueError:
            recipe = {}

        slots = sorted(cell.get("slots") or [], key=lambda s: s.get("order") or 99)
        if not slots:
            return "generative", "no cell wired — defaulting to the safe path"
        lead = slots[0]

        axis_name = canonical(lead.get("axis"))
        for axis in recipe.get("axes") or []:
            if canonical(axis.get("name")) == axis_name:
                declared = str(axis.get("route") or "").strip().lower()
                if declared in ("deterministic", "generative"):
                    return declared, "recipe declares route=%s for axis '%s'" % (
                        declared, axis_name)
                break

        # A hex spec no longer implies the deterministic route. It only becomes
        # available when the recipe explicitly opts in, because taking it silently
        # requires masks that exist for exactly one hand-prepared product — and the
        # failure mode when they are missing is an image that ships back unchanged.
        return "generative", ("axis '%s' declares no route; generative is the default "
                              "because it needs no per-product masks" % axis_name)

    def check_lazy_status(self, route, deterministic=None, generative=None,
                          cell_json="", recipe_json=""):
        taken, _why = self._decide(route, cell_json, recipe_json)
        return ["deterministic"] if taken == "deterministic" else ["generative"]

    def run(self, route, deterministic=None, generative=None, cell_json="",
            recipe_json=""):
        taken, why = self._decide(route, cell_json, recipe_json)
        image = deterministic if taken == "deterministic" else generative

        report = "\n".join([
            "ROUTE: %s" % taken.upper(),
            "  reason : %s" % why,
            "  cost   : %s" % ("none — no model call" if taken == "deterministic"
                               else "one generation call"),
        ])

        if image is None:
            return (ExecutionBlocker(
                "Route resolved to '%s' (%s) but nothing is wired into that input."
                % (taken, why)), taken, report)

        print("[arkennemasis] route: %s — %s" % (taken, why))
        return (image, taken, report)


NODE_CLASS_MAPPINGS = {
    "ArkRegionRecolour": ArkRegionRecolour,
    "ArkGenRoute": ArkGenRoute,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArkRegionRecolour": "arkennemasis Region Recolour (deterministic, exact)",
    "ArkGenRoute": "arkennemasis Gen Route (deterministic or generative)",
}

"""Stage 3 — choose one base photograph per camera angle, then freeze it.

This is the only point in the pipeline where manual retouching is economically
justified: one hour spent here propagates to every output. After this node runs, the
plate is never regenerated, re-rolled or re-selected mid-run, and every cell in the run
is generated from the identical file.

What gets frozen is not just the pixels. The lock also records the measurements that
verification will later compare candidates against — the subject's bounding box, each
named region's box, and the scale-invariant proportion ratios between regions. Deriving
those here, once, from a known-good photograph is what makes the identity check in
Stage 7 possible at all: without a reference measurement there is nothing to fail
against.

Region masks are per plate, not per cell. A hand-authored mask is entirely acceptable
here — it is authored once and reused by every one of the product's cells. Masks may
arrive from a folder of PNGs, from MASK sockets (so any segmentation node in ComfyUI can
feed this), or be omitted, in which case only whole-subject checks are available.
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch

from .schema import ValidationError, canonical, sha256_bytes, stable_json

BACKGROUND_NAMES = ("background", "backdrop", "surface", "scene")


def _resolve_dir(path, fallback="variation"):
    path = str(path or "").strip()
    if not path:
        path = fallback
    if not os.path.isabs(path):
        try:
            import folder_paths
            path = os.path.join(folder_paths.get_output_directory(), path)
        except Exception:
            path = os.path.abspath(path)
    os.makedirs(path, exist_ok=True)
    return path


def _mask_to_array(mask):
    """A ComfyUI MASK (B,H,W) or (H,W) -> a float (H,W) array in 0..1."""
    array = mask.detach().cpu().numpy() if hasattr(mask, "detach") else np.asarray(mask)
    array = array.astype(np.float32)
    while array.ndim > 2:
        array = array[0] if array.shape[0] == 1 else array[..., 0]
    if array.max() > 1.0:
        array = array / 255.0
    return np.clip(array, 0.0, 1.0)


def _load_mask_file(path, shape):
    from PIL import Image
    with Image.open(path) as image:
        image = image.convert("L")
        if image.size != (shape[1], shape[0]):
            image = image.resize((shape[1], shape[0]), Image.NEAREST)
        return np.asarray(image).astype(np.float32) / 255.0


def bbox_of(mask, threshold=0.5):
    """`(x, y, w, h)` of the truthy area, or None when the mask is empty."""
    ys, xs = np.where(np.asarray(mask) > threshold)
    if len(xs) == 0:
        return None
    return (int(xs.min()), int(ys.min()),
            int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))


def background_colour(rgb_image):
    """The plate's background, sampled from its border. Returns an RGB triple in 0..1."""
    image = np.asarray(rgb_image, dtype=np.float32)[:, :, :3]
    border = np.concatenate([
        image[:4, :, :].reshape(-1, 3), image[-4:, :, :].reshape(-1, 3),
        image[:, :4, :].reshape(-1, 3), image[:, -4:, :].reshape(-1, 3),
    ], axis=0)
    return np.median(border, axis=0)


def auto_subject_mask(rgb_image, background=None, threshold=None):
    """A subject mask derived from the image itself, for products on a plain ground.

    Ecommerce plates are overwhelmingly shot on a seamless pale surface, so the border
    pixels are a reliable sample of the background and anything far from that colour is
    subject. This is a convenience for getting started — a hand-authored mask is better
    and is authored only once per plate.

    `background` and `threshold` may be supplied so a candidate is segmented with the
    PLATE's parameters rather than its own. That matters: scene continuity means the
    background is unchanged, so the plate's values are the correct ones, and re-deriving
    them per candidate lets a variant whose colour drifts towards the background quietly
    move the threshold.
    """
    image = np.asarray(rgb_image, dtype=np.float32)[:, :, :3]
    if background is None:
        background = background_colour(image)
    distance = np.linalg.norm(image - np.asarray(background,
                                                 dtype=np.float32)[None, None, :], axis=2)

    if threshold is None:
        threshold = max(0.08, float(np.percentile(distance, 70)) * 0.5)
    mask = (distance > float(threshold)).astype(np.float32)

    try:
        import cv2
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        # Drop specks, but keep every substantial component. Taking only the LARGEST is
        # tempting and wrong: a variation whose colour approaches the background can
        # sever the product into two pieces, and keeping one of them makes the subject
        # appear to have moved and shrunk when nothing about it changed.
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8)
        if count > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            keep = 1 + np.flatnonzero(areas >= max(1, int(areas.max() * 0.10)))
            mask = np.isin(labels, keep).astype(np.float32)
    except Exception:
        pass
    return mask


def proportion_ratios(boxes):
    """Scale-invariant ratios between every pair of named region boxes.

    Ratios rather than absolute sizes, because a candidate that is uniformly 2% larger
    is a framing problem while one whose shade grew 8% wider relative to its base is an
    identity problem — and only the second is what Lock A prohibits. Ratios separate the
    two; absolute measurements confuse them.
    """
    ratios = {}
    names = sorted(k for k, v in boxes.items() if v)
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            a, b = boxes[first], boxes[second]
            if not a or not b or b[2] <= 0 or b[3] <= 0:
                continue
            ratios["%s.w/%s.w" % (first, second)] = round(a[2] / b[2], 6)
            ratios["%s.h/%s.h" % (first, second)] = round(a[3] / b[3], 6)
    for name in names:
        box = boxes[name]
        if box and box[3] > 0:
            ratios["%s.aspect" % name] = round(box[2] / box[3], 6)
    return ratios


class ArkPlateLock:
    CATEGORY = "arkennemasis/Variation"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("plate", "subject_mask", "plate_lock_json", "report", "plate_path")
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Freeze one base photograph and measure it. Records the file hash, dimensions, "
        "colour profile, per-region boxes and the scale-invariant proportion ratios "
        "that Stage 7 compares every candidate against. After this, the plate never "
        "changes for the rest of the run."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "plate": ("IMAGE", {
                    "tooltip": "The chosen, retouched base photograph. One per camera "
                               "angle.",
                }),
                "plate_id": ("STRING", {
                    "default": "front",
                    "tooltip": "Camera-angle id. Must match the plate id used in the "
                               "recipe and the job keys.",
                }),
                "lock_dir": ("STRING", {
                    "default": "variation/plates",
                    "tooltip": "Where the frozen plate and its lock file are written. "
                               "Relative paths resolve inside ComfyUI's output folder.",
                }),
            },
            "optional": {
                "regions": ("STRING", {
                    "default": "",
                    "tooltip": "Comma-separated region names, in the same order the mask "
                               "sockets below are wired. The names are the product's own "
                               "and come from the recipe — this node never invents them, "
                               "and nothing downstream assumes any particular set.",
                }),
                "region_mask_dir": ("STRING", {
                    "default": "",
                    "tooltip": "Optional folder of <region>.png masks (white = the "
                               "region). Authored once per plate and reused by every "
                               "cell, so hand-painting them is entirely reasonable.",
                }),
                "mask_1": ("MASK", {"tooltip": "Region mask 1 — pairs with the first "
                                               "name in 'regions'."}),
                "mask_2": ("MASK",),
                "mask_3": ("MASK",),
                "mask_4": ("MASK",),
                "subject_mask_in": ("MASK", {
                    "tooltip": "Optional explicit whole-subject mask. Omitted = derived "
                               "from the plate's own border colour.",
                }),
                "colour_profile": ("STRING", {
                    "default": "sRGB",
                    "tooltip": "Decided once, here, and enforced at verification and "
                               "embedded at delivery. Generating in one space and "
                               "delivering in another is the usual cause of 'the colour "
                               "looks wrong' after delivery.",
                }),
                "overwrite": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Off: refuse to replace an existing locked plate. That "
                               "refusal is the lock doing its job — re-locking mid-run "
                               "silently changes what every later cell is compared to.",
                }),
            },
        }

    def run(self, plate, plate_id="front", lock_dir="variation/plates", regions="",
            region_mask_dir="", mask_1=None, mask_2=None, mask_3=None, mask_4=None,
            subject_mask_in=None, colour_profile="sRGB", overwrite=False):
        from PIL import Image

        if plate is None or plate.shape[0] < 1:
            raise ValidationError("ArkPlateLock: no plate image supplied.")
        if plate.shape[0] > 1:
            print("[arkennemasis] plate lock: batch of %d, locking the first only."
                  % plate.shape[0])

        array = plate[0].detach().cpu().numpy().astype(np.float32)[:, :, :3]
        height, width = array.shape[:2]
        plate_key = canonical(plate_id) or "front"

        root = _resolve_dir(lock_dir)
        plate_path = os.path.join(root, "%s.png" % plate_key)
        lock_path = os.path.join(root, "%s.lock.json" % plate_key)

        if os.path.isfile(lock_path) and not overwrite:
            with open(lock_path, "r", encoding="utf-8") as handle:
                existing = json.load(handle)
            print("[arkennemasis] plate '%s' already locked — reusing %s"
                  % (plate_key, lock_path))
            existing_mask = np.zeros((height, width), dtype=np.float32)
            mask_file = existing.get("subject_mask")
            if mask_file and os.path.isfile(mask_file):
                existing_mask = _load_mask_file(mask_file, (height, width))
            report = ("PLATE ALREADY LOCKED — %s\n  %s\n\nTurn 'overwrite' on only if "
                      "you intend every later comparison to change." % (plate_key, lock_path))
            return (plate, torch.from_numpy(existing_mask)[None, ...],
                    json.dumps(existing, ensure_ascii=False), report, plate_path)

        # Write the plate first: the hash must be of the exact bytes every cell reads,
        # not of the tensor in memory.
        Image.fromarray((np.clip(array, 0, 1) * 255).astype(np.uint8)).save(plate_path)
        with open(plate_path, "rb") as handle:
            digest = sha256_bytes(handle.read())

        names = [canonical(n) for n in str(regions or "").split(",") if canonical(n)]
        supplied = [mask_1, mask_2, mask_3, mask_4]

        masks = {}
        for index, name in enumerate(names):
            if index < len(supplied) and supplied[index] is not None:
                masks[name] = _mask_to_array(supplied[index])

        folder = str(region_mask_dir or "").strip()
        if folder:
            if not os.path.isabs(folder):
                folder = _resolve_dir(folder)
            if os.path.isdir(folder):
                for entry in sorted(os.listdir(folder)):
                    stem, ext = os.path.splitext(entry)
                    if ext.lower() not in (".png", ".jpg", ".jpeg", ".webp"):
                        continue
                    name = canonical(stem)
                    if name and name not in masks:
                        masks[name] = _load_mask_file(os.path.join(folder, entry),
                                                      (height, width))

        for name, mask in list(masks.items()):
            if mask.shape != (height, width):
                raise ValidationError(
                    "Region mask '%s' is %s but the plate is %s. Masks must match the "
                    "plate exactly." % (name, mask.shape, (height, width)))

        if subject_mask_in is not None:
            subject = _mask_to_array(subject_mask_in)
        else:
            foreground = [m for n, m in masks.items() if n not in BACKGROUND_NAMES]
            if foreground:
                subject = np.clip(np.maximum.reduce(foreground), 0.0, 1.0)
            else:
                subject = auto_subject_mask(array)

        mask_dir = os.path.join(root, "%s.masks" % plate_key)
        os.makedirs(mask_dir, exist_ok=True)
        subject_path = os.path.join(mask_dir, "_subject.png")
        Image.fromarray((subject * 255).astype(np.uint8)).save(subject_path)

        region_records = {}
        for name, mask in sorted(masks.items()):
            path = os.path.join(mask_dir, "%s.png" % name)
            Image.fromarray((mask * 255).astype(np.uint8)).save(path)
            box = bbox_of(mask)
            region_records[name] = {
                "mask": path,
                "bbox": box,
                "area_fraction": round(float((mask > 0.5).mean()), 6),
            }

        boxes = {n: r["bbox"] for n, r in region_records.items()}
        subject_box = bbox_of(subject)
        if subject_box:
            boxes["subject"] = subject_box

        lock = {
            "schema_version": "1.0",
            "plate_id": plate_key,
            "file": plate_path,
            "hash": digest,
            "width": int(width),
            "height": int(height),
            "aspect": round(width / height, 6) if height else 0.0,
            "colour_profile": str(colour_profile or "sRGB").strip(),
            "subject_mask": subject_path,
            "subject_bbox": subject_box,
            "subject_area_fraction": round(float((subject > 0.5).mean()), 6),
            "regions": region_records,
            "ratios": proportion_ratios(boxes),
        }
        lock["lock_hash"] = sha256_bytes(stable_json(lock).encode("utf-8"))

        with open(lock_path, "w", encoding="utf-8") as handle:
            json.dump(lock, handle, indent=2, ensure_ascii=False)

        lines = [
            "PLATE LOCKED — %s" % plate_key,
            "  file      : %s" % plate_path,
            "  hash      : %s" % digest,
            "  size      : %d x %d  (aspect %.4f)" % (width, height, lock["aspect"]),
            "  profile   : %s" % lock["colour_profile"],
            "  subject   : bbox %s, %.1f%% of frame"
            % (subject_box, lock["subject_area_fraction"] * 100),
            "  lock file : %s" % lock_path,
        ]
        if region_records:
            lines += ["", "REGIONS (%d)" % len(region_records)]
            for name, record in sorted(region_records.items()):
                lines.append("  %-16s bbox %-24s %.2f%% of frame"
                             % (name, record["bbox"], record["area_fraction"] * 100))
        else:
            lines += ["", "No region masks supplied — only whole-subject checks will be "
                          "available in Stage 7. Wire MASK sockets or point "
                          "region_mask_dir at a folder of <region>.png files to enable "
                          "per-region colour and bleed checks."]
        if lock["ratios"]:
            lines += ["", "PROPORTION RATIOS (the identity fingerprint)"]
            for key, value in sorted(lock["ratios"].items()):
                lines.append("  %-28s %.4f" % (key, value))

        print("[arkennemasis] plate locked: %s (%dx%d, %d regions)"
              % (plate_key, width, height, len(region_records)))
        return (plate, torch.from_numpy(subject)[None, ...],
                json.dumps(lock, ensure_ascii=False), "\n".join(lines), plate_path)


class ArkRegionMask:
    CATEGORY = "arkennemasis/Variation"
    FUNCTION = "run"
    RETURN_TYPES = ("MASK", "STRING", "BOOLEAN")
    RETURN_NAMES = ("mask", "region", "found")
    DESCRIPTION = (
        "Fetch one named region's mask out of the locked plate. Give it a cell and it "
        "resolves that cell's own target region by itself, so the per-cell chain needs "
        "no per-region wiring however many regions the product has."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "plate_lock_json": ("STRING", {
                    "multiline": True, "default": "{}", "forceInput": True,
                    "tooltip": "From Plate Lock.",
                }),
            },
            "optional": {
                "cell_json": ("STRING", {
                    "multiline": True, "default": "", "forceInput": True,
                    "tooltip": "The cell whose target region to fetch. Its leading axis "
                               "— the one painting the largest area — decides.",
                }),
                "region": ("STRING", {
                    "default": "", "forceInput": True,
                    "tooltip": "Overrides the cell's region.",
                }),
                "invert": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Return everything EXCEPT this region.",
                }),
                "on_missing": (["empty mask", "fail"], {
                    "tooltip": "What to do when the plate has no mask for that region. "
                               "'empty mask' lets a run continue with a whole-frame "
                               "colour reading, which is weaker but not wrong; 'fail' "
                               "stops so you notice the masks were never authored.",
                }),
            },
        }

    def run(self, plate_lock_json, cell_json="", region="", invert=False,
            on_missing="empty mask"):
        lock = json.loads(plate_lock_json or "{}")
        wanted = canonical(region)

        if not wanted and str(cell_json or "").strip():
            cell = json.loads(cell_json)
            slots = sorted(cell.get("slots") or [], key=lambda s: s.get("order") or 99)
            if slots:
                wanted = canonical(slots[0].get("region"))

        height = int(lock.get("height") or 0) or 8
        width = int(lock.get("width") or 0) or 8

        record = (lock.get("regions") or {}).get(wanted)
        path = record.get("mask") if isinstance(record, dict) else None

        if not wanted or not path or not os.path.isfile(path):
            message = ("The locked plate has no mask for region '%s'. Available: %s."
                       % (wanted or "(none resolved)",
                          ", ".join(sorted(lock.get("regions") or {})) or "none"))
            if on_missing == "fail":
                raise ValidationError(message)
            print("[arkennemasis] region mask: %s Continuing with an empty mask."
                  % message)
            empty = np.zeros((height, width), dtype=np.float32)
            return (torch.from_numpy(empty)[None, ...], wanted, False)

        mask = _load_mask_file(path, (height, width))
        if invert:
            mask = 1.0 - mask
        print("[arkennemasis] region mask '%s': %.2f%% of frame"
              % (wanted, 100.0 * float((mask > 0.5).mean())))
        return (torch.from_numpy(mask)[None, ...], wanted, True)


NODE_CLASS_MAPPINGS = {
    "ArkPlateLock": ArkPlateLock,
    "ArkRegionMask": ArkRegionMask,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArkPlateLock": "arkennemasis Plate Lock (freeze + measure the base photo)",
    "ArkRegionMask": "arkennemasis Region Mask (one region from the locked plate)",
}

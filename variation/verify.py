"""Stage 7 — verify every candidate before it is accepted.

This is the software reconstruction of the step that was never written down: the
operator looking at each output and re-running the bad ones. Across 81 images that is
perhaps an hour of judgement so continuous it does not register as work — and automating
everything except it would remove the human from the only part that was holding quality
together.

Four checks, in the fidelity order they matter:

1. **Product identity** — is it the same object? Colour checking cannot see this. A
   subtly redesigned lamp in exactly the right amber passes a colour test and is still a
   defect, and it is the failure a customer notices fastest. Measured by silhouette
   overlap, by a scale- and translation-invariant shape distance, and by the proportion
   ratios frozen at plate-lock time.
2. **Frame match** — did the product or the scene move? Bounding box, aspect, and
   structural similarity computed over the regions that were NOT supposed to change.
3. **Colour delta** — is it the specified colour, in ΔE2000 against the hex.
4. **Bleed and hygiene** — did the colour appear outside its region, and are the
   dimensions and file properties right.

Drift is not a prompting defect that better wording eliminates. These models regenerate
the entire frame on every call, so drift is a property of the tool and MUST be caught
outside the generator. The lock clause reduces its frequency; this stage catches what
remains.

Thresholds are arguments, never constants. They are calibrated from measurement against
a labelled set — that is what `ArkCalibrate` is for.
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch

from .colour import coverage_outside, delta_e, hex_to_rgb01, rgb01_to_hex, sample_region
from .plate_lock import (
    auto_subject_mask,
    background_colour,
    bbox_of,
    proportion_ratios,
)
from .schema import ValidationError, canonical

RESULT_PASS = "pass"
RESULT_SOFT = "soft"
RESULT_HARD = "hard"


def _to_array(image):
    if image is None:
        raise ValidationError("verify: an image input is missing.")
    tensor = image[0] if hasattr(image, "shape") and len(image.shape) == 4 else image
    array = tensor.detach().cpu().numpy() if hasattr(tensor, "detach") else np.asarray(tensor)
    return np.clip(array.astype(np.float32), 0.0, 1.0)[:, :, :3]


def _mask_array(mask, shape):
    if mask is None:
        return None
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
                               interpolation=cv2.INTER_NEAREST)
        except Exception:
            return None
    return np.clip(array, 0.0, 1.0)


def _load_mask_file(path, shape):
    from PIL import Image
    with Image.open(path) as image:
        image = image.convert("L")
        if image.size != (shape[1], shape[0]):
            image = image.resize((shape[1], shape[0]), Image.NEAREST)
        return np.asarray(image).astype(np.float32) / 255.0


def shape_distance(mask_a, mask_b):
    """Shape difference with position and scale removed. 0 = identical outline.

    Each silhouette is cropped to its own bounding box and resampled to a common grid,
    so a candidate that is merely 2% larger or slightly off-centre scores ~0 while one
    whose outline genuinely changed scores high. That separation is the point: growing a
    wider flare is an identity defect, sitting 3px left is a framing defect, and
    conflating them makes both unfixable.

    Deliberately NOT Hu-moment matching (`cv2.matchShapes`). Every one of its three
    metrics divides by the log-transformed moments, and for the near-symmetric blobs
    that product silhouettes usually are, the higher-order moments sit near zero — so
    the reciprocal explodes and two visually identical outlines score far apart. It
    reported 0.22 on a pair with an IoU of 0.991. Normalised overlap is bounded, stable,
    and means something you can explain to a client.
    """
    try:
        import cv2
    except ImportError:
        return None

    grids = []
    for mask in (mask_a, mask_b):
        binary = (np.asarray(mask) > 0.5).astype(np.uint8)
        box = bbox_of(binary)
        if not box or box[2] < 2 or box[3] < 2:
            return None
        x, y, w, h = box
        crop = binary[y:y + h, x:x + w]
        grids.append(cv2.resize(crop, (128, 128), interpolation=cv2.INTER_AREA) > 0.5)

    union = np.logical_or(grids[0], grids[1]).sum()
    if union == 0:
        return None
    overlap = float(np.logical_and(grids[0], grids[1]).sum() / union)
    return round(1.0 - overlap, 6)          # 0 = identical, 1 = nothing in common


def silhouette_iou(mask_a, mask_b):
    a = np.asarray(mask_a) > 0.5
    b = np.asarray(mask_b) > 0.5
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(a, b).sum() / union)


def ssim_masked(image_a, image_b, mask=None):
    """Structural similarity over the pixels `mask` selects (1 = compare here)."""
    try:
        from skimage.metrics import structural_similarity
    except ImportError:
        return None
    grey_a = image_a @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    grey_b = image_b @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    score, full = structural_similarity(grey_a, grey_b, data_range=1.0, full=True)
    if mask is None:
        return float(score)
    selector = np.asarray(mask) > 0.5
    if not selector.any():
        return float(score)
    return float(full[selector].mean())


def box_metrics(candidate_box, plate_box, width, height):
    """Positional and scale drift of the subject, as fractions of the frame."""
    if not candidate_box or not plate_box:
        return {}
    cx, cy, cw, ch = candidate_box
    px, py, pw, ph = plate_box
    centre_shift = float(np.hypot(((cx + cw / 2) - (px + pw / 2)) / max(1, width),
                                  ((cy + ch / 2) - (py + ph / 2)) / max(1, height)))
    scale_w = cw / pw if pw else 1.0
    scale_h = ch / ph if ph else 1.0
    return {
        "bbox_centre_shift": round(centre_shift, 5),
        "bbox_scale_w": round(float(scale_w), 5),
        "bbox_scale_h": round(float(scale_h), 5),
        "bbox_scale_drift": round(float(max(abs(scale_w - 1.0), abs(scale_h - 1.0))), 5),
    }


def ratio_drift(candidate_ratios, plate_ratios):
    """Largest relative deviation across the shared proportion ratios."""
    shared = [k for k in plate_ratios if k in candidate_ratios]
    if not shared:
        return None, {}
    per_key = {}
    worst = 0.0
    for key in shared:
        base = plate_ratios[key]
        if not base:
            continue
        drift = abs(candidate_ratios[key] - base) / abs(base)
        per_key[key] = round(float(drift), 5)
        worst = max(worst, drift)
    return round(float(worst), 5), per_key


def run_checks(candidate, plate, plate_lock, target_region="", target_hex="",
               candidate_mask=None, plate_mask=None, region_masks=None,
               exclude_regions=None, candidate_region_masks=None):
    """Every measurement for one candidate. Pure numbers — no pass/fail decision here."""
    checks = {}
    notes = []
    height, width = candidate.shape[:2]
    plate_height, plate_width = plate.shape[:2]

    checks["dims_match"] = bool(height == plate_height and width == plate_width)
    checks["aspect_candidate"] = round(width / height, 5) if height else 0.0
    checks["aspect_plate"] = round(plate_width / plate_height, 5) if plate_height else 0.0
    checks["aspect_match"] = bool(
        abs(checks["aspect_candidate"] - checks["aspect_plate"]) < 0.005)

    if not checks["dims_match"]:
        notes.append("Candidate is %dx%d but the plate is %dx%d."
                     % (width, height, plate_width, plate_height))
        # Everything below compares pixel to pixel, so bring the candidate to the
        # plate's grid rather than refusing to measure at all.
        try:
            import cv2
            candidate = cv2.resize(candidate, (plate_width, plate_height),
                                   interpolation=cv2.INTER_AREA)
            height, width = candidate.shape[:2]
        except Exception:
            return checks, notes

    # The candidate is segmented with the PLATE's background and threshold, not its own.
    # Scene continuity means the background is unchanged, so the plate's values are the
    # correct ones — and re-deriving them per candidate would let a variant whose colour
    # drifts towards the background quietly move its own goalposts.
    supplied_masks = candidate_mask is not None
    background = background_colour(plate)
    plate_distance = np.linalg.norm(
        plate - np.asarray(background, dtype=np.float32)[None, None, :], axis=2)
    threshold = max(0.08, float(np.percentile(plate_distance, 70)) * 0.5)

    if plate_mask is None:
        plate_mask = auto_subject_mask(plate, background, threshold)
    if candidate_mask is None:
        candidate_mask = auto_subject_mask(candidate, background, threshold)

    # ── 1. Identity ─────────────────────────────────────────────────────────
    # A variation whose colour genuinely approaches the background cannot be segmented
    # by colour, and silently scoring that as a shape change would fail a correct image.
    # Detect it, say so, and let `decide` treat identity as unmeasured rather than bad.
    plate_area = float((np.asarray(plate_mask) > 0.5).mean())
    candidate_area = float((np.asarray(candidate_mask) > 0.5).mean())
    checks["subject_area_plate"] = round(plate_area, 5)
    checks["subject_area_candidate"] = round(candidate_area, 5)
    reliable = supplied_masks or (
        plate_area > 0.001
        and 0.75 <= (candidate_area / plate_area if plate_area else 0.0) <= 1.35)
    checks["mask_reliable"] = bool(reliable)
    if not reliable:
        notes.append(
            "Subject segmentation is unreliable for this candidate: it covers %.1f%% of "
            "the frame against the plate's %.1f%%. That normally means this variation's "
            "colour approaches the background, not that the object changed. Silhouette "
            "and shape were measured but are NOT being used to fail the cell — supply an "
            "explicit candidate_mask (e.g. from a segmentation node) to check identity "
            "properly here." % (candidate_area * 100, plate_area * 100))

    checks["silhouette_iou"] = round(silhouette_iou(candidate_mask, plate_mask), 5)
    distance = shape_distance(candidate_mask, plate_mask)
    if distance is not None:
        checks["shape_distance"] = round(distance, 6)

    # Both boxes come from the SAME estimator on the two images. The plate's locked
    # bbox is deliberately NOT used here: it was measured from hand-authored region
    # masks, while the candidate's can only be derived from the image itself, and
    # comparing a mask-derived box against a colour-derived one measures the difference
    # between two estimators rather than any drift in the product.
    candidate_box = bbox_of(candidate_mask)
    plate_box = bbox_of(plate_mask)
    checks.update(box_metrics(candidate_box, plate_box, width, height))

    if candidate_box and plate_box and plate_box[3] and candidate_box[3]:
        plate_aspect = plate_box[2] / plate_box[3]
        candidate_aspect = candidate_box[2] / candidate_box[3]
        checks["subject_aspect_drift"] = round(
            abs(candidate_aspect - plate_aspect) / abs(plate_aspect), 5)

    # Region-to-region proportion drift needs the regions located in the CANDIDATE.
    # Plate-space masks cannot supply that — applying them to the candidate would
    # measure the plate and always score a perfect zero, which is worse than not
    # measuring at all. So this check runs only when candidate-space masks are supplied,
    # e.g. from a segmentation node run on the candidate.
    locked_ratios = (plate_lock.get("ratios")
                     or (plate_lock.get("plate_lock") or {}).get("ratios") or {})
    if candidate_region_masks and locked_ratios:
        boxes = {name: bbox_of(mask) for name, mask in candidate_region_masks.items()}
        if candidate_box:
            boxes["subject"] = candidate_box
        worst, per_key = ratio_drift(proportion_ratios(boxes), locked_ratios)
        if worst is not None:
            checks["ratio_drift"] = worst
            checks["ratio_detail"] = per_key
    elif locked_ratios:
        notes.append(
            "Region-to-region proportion drift was not measured: it needs region masks "
            "for the CANDIDATE, not the plate. Silhouette, shape distance and subject "
            "aspect still cover identity.")

    # ── 2. Frame match, over the regions that were NOT supposed to change ───
    # EVERY axis paints a region, so with N axes there are N legitimately-changed
    # regions. Excluding only the leading one would score the other axes' correct
    # changes as scene drift and fail every multi-axis cell.
    changed = list(exclude_regions or ([target_region] if target_region else []))
    exclude = np.zeros((height, width), dtype=np.float32)
    for name in changed:
        name = canonical(name)
        if not name:
            continue
        mask = None
        if region_masks and name in region_masks:
            mask = np.asarray(region_masks[name], dtype=np.float32)
            if mask.shape != (height, width):
                # The image is already paid for. A mask that arrived at the wrong
                # scale is a wiring problem, not a reason to throw the result away.
                try:
                    import cv2
                    mask = cv2.resize(mask, (width, height),
                                      interpolation=cv2.INTER_NEAREST)
                except Exception:
                    notes.append("Region '%s' mask is %s but the comparison grid is "
                                 "%s, and it could not be resized — that region was "
                                 "left out of the frame check."
                                 % (name, mask.shape, (height, width)))
                    mask = None
        else:
            locked = (plate_lock.get("regions") or {}).get(name)
            if isinstance(locked, dict) and locked.get("mask") \
                    and os.path.isfile(locked["mask"]):
                mask = _load_mask_file(locked["mask"], (height, width))
        if mask is not None:
            exclude = np.maximum(exclude, np.clip(mask, 0.0, 1.0))

    # A hard mask edge is a pixel-wide lie: the generator's transition and the mask's
    # transition never land on exactly the same pixel. Dilating the excluded area
    # slightly stops a one-pixel seam from reading as scene drift.
    try:
        import cv2
        if exclude.any():
            exclude = cv2.dilate(exclude, np.ones((5, 5), np.uint8), iterations=1)
    except Exception:
        pass

    compare_area = 1.0 - np.clip(exclude, 0.0, 1.0)
    checks["compare_fraction"] = round(float((compare_area > 0.5).mean()), 5)
    score = ssim_masked(candidate, plate, compare_area)
    if score is not None:
        checks["ssim_untargeted"] = round(score, 5)
    difference = np.abs(candidate - plate).mean(axis=2)
    checks["mean_abs_diff_untargeted"] = round(
        float(difference[compare_area > 0.5].mean()) if (compare_area > 0.5).any() else 0.0, 5)

    # SSIM over the untargeted regions is computed from raw pixels and needs no mask of
    # the subject, so it is the one identity-adjacent signal a segmentation failure
    # cannot fool.
    #
    # When the untargeted scene is essentially bit-identical to the plate, geometry is
    # unchanged BY CONSTRUCTION: the camera did not move, the frame did not move, and
    # every pixel that was not supposed to change did not. That is a stronger identity
    # guarantee than any silhouette comparison can offer, so silhouette and shape stop
    # being decisive here.
    #
    # This matters because the subject mask is derived from colour distance to the
    # background — so a variation that legitimately makes a region paler or greyer moves
    # the mask for reasons that have nothing to do with the object's geometry. Without
    # this, the deterministic path (which cannot drift at all) fails its own cells.
    scene_identical = (score is not None and score > 0.995
                       and checks.get("mean_abs_diff_untargeted", 1.0) < 0.002)
    checks["scene_identical"] = bool(scene_identical)
    if scene_identical:
        checks["mask_reliable"] = False
        notes.append(
            "Everything outside the painted regions is bit-identical to the plate "
            "(SSIM %.4f, %.6f mean difference per pixel). Geometry is therefore "
            "unchanged by construction, and the silhouette metrics — which read a "
            "colour-derived mask and so move when a region's colour legitimately "
            "changes — are reported but not used to fail this cell."
            % (score, checks["mean_abs_diff_untargeted"]))
    elif checks.get("mask_reliable", True) and score is not None and score > 0.99 \
            and checks.get("silhouette_iou", 1.0) < 0.75:
        checks["mask_reliable"] = False
        notes.append(
            "Silhouette overlap is %.3f but the untargeted scene is %.4f similar to the "
            "plate and differs by only %.5f per pixel. Those cannot both be true of a "
            "moved product, so the subject mask is at fault, not the image. Identity "
            "checks are reported but not used to fail this cell."
            % (checks.get("silhouette_iou", -1), score,
               checks["mean_abs_diff_untargeted"]))

    # ── 3. Colour ───────────────────────────────────────────────────────────
    if target_hex:
        region_mask = None
        if region_masks and target_region and target_region in region_masks:
            region_mask = region_masks[target_region]
        elif np.any(exclude > 0.5):
            region_mask = exclude
        sampled, count = sample_region(candidate, region_mask)
        if sampled is not None:
            checks["sampled_hex"] = rgb01_to_hex(sampled)
            checks["sampled_pixels"] = count
            try:
                checks["colour_de"] = round(delta_e(sampled, hex_to_rgb01(target_hex)), 4)
            except ValueError:
                notes.append("target_hex %r is not a hex colour." % target_hex)
            if region_mask is None:
                notes.append(
                    "No mask for region '%s' — the colour was sampled over the whole "
                    "frame, which is only meaningful for a full-frame change. Supply "
                    "region masks at plate-lock time for a real reading." % target_region)
        else:
            notes.append("Region '%s' selected no pixels." % target_region)

        # ── 4. Bleed ────────────────────────────────────────────────────────
        # Considered area = everything no axis was allowed to touch. Regions painted by
        # OTHER axes changed on purpose and counting them would flag every multi-axis
        # cell as bleeding into itself.
        if region_masks:
            painted = {canonical(n) for n in changed}
            others = [m for n, m in region_masks.items() if canonical(n) not in painted]
            if others:
                considered = np.clip(np.maximum.reduce(others), 0.0, 1.0)
                considered = considered * (1.0 - np.clip(exclude, 0.0, 1.0))
                fraction, counted = coverage_outside(
                    considered, candidate, plate, target_hex, 8.0)
                checks["bleed_fraction"] = round(fraction, 5)
                checks["bleed_pixels_considered"] = counted

    # ── 5. Hygiene ──────────────────────────────────────────────────────────
    checks["mean_luma"] = round(float((candidate @ [0.2126, 0.7152, 0.0722]).mean()), 5)
    flat = float(candidate.std())
    checks["std"] = round(flat, 5)
    if flat < 0.01:
        notes.append("The candidate is almost flat — a blank or failed render.")

    return checks, notes


def decide(checks, tolerances):
    """Turn measurements into pass / soft / hard, plus the reasons.

    Hard failure is not retried blindly: a wrong region changed or a redesigned product
    usually means a recipe error, and retrying spends money without addressing the
    cause. Soft failure — colour slightly out, minor movement — is what retries are for.
    """
    frame_tolerance = float(tolerances.get("frame_tolerance", 0.02))
    colour_tolerance = float(tolerances.get("colour_tolerance_de", 5.0))
    identity_tolerance = float(tolerances.get("identity_tolerance", 0.06))
    min_iou = float(tolerances.get("min_silhouette_iou", 0.90))
    max_shape = float(tolerances.get("max_shape_distance", 0.10))
    min_ssim = float(tolerances.get("min_ssim_untargeted", 0.80))
    max_bleed = float(tolerances.get("max_bleed_fraction", 0.05))

    hard, soft = [], []

    # A generative backend picks its own output size — gpt-image-2 answers a 720x961
    # plate with 1085x1450 — so differing dimensions alone are normal, not a defect.
    # What matters is the ASPECT: same aspect means the same crop at a different scale,
    # and the candidate is resampled onto the plate's grid before anything is measured.
    # A changed aspect means the framing genuinely changed, and that is a hard failure.
    if checks.get("aspect_match") is False:
        hard.append("Aspect ratio does not match the plate — the crop changed.")
    if checks.get("std", 1.0) < 0.01:
        hard.append("The candidate is blank or flat.")

    # Silhouette and shape are only decisive when the subject could actually be
    # segmented. When it could not, the measurements are still reported but must not
    # fail a cell — an unmeasurable check is not a failed check.
    measurable = checks.get("mask_reliable", True)

    iou = checks.get("silhouette_iou")
    if iou is not None and measurable:
        if iou < min_iou * 0.85:
            hard.append("Silhouette overlap %.3f is far below %.3f — the object moved or "
                        "changed shape grossly." % (iou, min_iou))
        elif iou < min_iou:
            soft.append("Silhouette overlap %.3f is below %.3f." % (iou, min_iou))

    distance = checks.get("shape_distance")
    if distance is not None and measurable:
        if distance > max_shape * 2:
            hard.append("Shape distance %.4f is far above %.4f — the product was "
                        "redesigned." % (distance, max_shape))
        elif distance > max_shape:
            soft.append("Shape distance %.4f is above %.4f." % (distance, max_shape))

    drift = checks.get("ratio_drift", checks.get("subject_aspect_drift"))
    if drift is not None and measurable:
        if drift > identity_tolerance * 2:
            hard.append("Proportion drift %.3f is far above %.3f — parts changed size "
                        "relative to each other." % (drift, identity_tolerance))
        elif drift > identity_tolerance:
            soft.append("Proportion drift %.3f is above %.3f." % (drift, identity_tolerance))

    shift = checks.get("bbox_centre_shift")
    if shift is not None and measurable:
        if shift > frame_tolerance * 3:
            hard.append("Subject moved %.3f of the frame — far above %.3f."
                        % (shift, frame_tolerance))
        elif shift > frame_tolerance:
            soft.append("Subject moved %.3f of the frame." % shift)

    scale = checks.get("bbox_scale_drift")
    if scale is not None and measurable:
        if scale > frame_tolerance * 4:
            hard.append("Subject scale changed by %.1f%% — far above tolerance."
                        % (scale * 100))
        elif scale > frame_tolerance * 2:
            soft.append("Subject scale changed by %.1f%%." % (scale * 100))

    ssim = checks.get("ssim_untargeted")
    if ssim is not None:
        if ssim < min_ssim * 0.8:
            hard.append("Untargeted regions changed heavily (SSIM %.3f) — the scene was "
                        "redrawn." % ssim)
        elif ssim < min_ssim:
            soft.append("Untargeted regions drifted (SSIM %.3f below %.3f)."
                        % (ssim, min_ssim))

    bleed = checks.get("bleed_fraction")
    if bleed is not None and bleed > max_bleed:
        hard.append("Target colour appeared on %.1f%% of the untargeted regions — it "
                    "bled outside its region." % (bleed * 100))

    difference = checks.get("colour_de")
    if difference is not None and difference > colour_tolerance:
        # Colour is the recoverable failure: the object is right, the tint is not.
        soft.append("Colour difference %.2f exceeds tolerance %.2f."
                    % (difference, colour_tolerance))

    if hard:
        return RESULT_HARD, hard + soft
    if soft:
        return RESULT_SOFT, soft
    return RESULT_PASS, []


class ArkVerifyCandidate:
    CATEGORY = "arkennemasis/Variation"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING", "STRING", "BOOLEAN", "STRING", "FLOAT")
    RETURN_NAMES = ("verdict_json", "report", "passed", "result", "colour_de")
    DESCRIPTION = (
        "Measure a candidate against the locked plate: product identity, frame match, "
        "colour delta, bleed and hygiene. Emits every measurement plus a pass / soft / "
        "hard verdict. This is the step that was living in the operator's head."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "candidate": ("IMAGE", {"tooltip": "The generated image."}),
                "plate": ("IMAGE", {"tooltip": "The locked base plate."}),
            },
            "optional": {
                "plate_lock_json": ("STRING", {
                    "multiline": True, "default": "", "forceInput": True,
                    "tooltip": "From Plate Lock — the frozen measurements to compare "
                               "against. Without it only same-image checks are possible.",
                }),
                "recipe_json": ("STRING", {
                    "multiline": True, "default": "", "forceInput": True,
                    "tooltip": "Supplies the calibrated tolerances.",
                }),
                "cell_json": ("STRING", {
                    "multiline": True, "default": "", "forceInput": True,
                    "tooltip": "Supplies the target region and the target hex for this "
                               "cell, so nothing has to be typed per cell.",
                }),
                "target_region": ("STRING", {
                    "default": "", "forceInput": True,
                    "tooltip": "Overrides the cell's region.",
                }),
                "target_hex": ("STRING", {
                    "default": "", "forceInput": True,
                    "tooltip": "Overrides the cell's hex. Blank = skip the colour check, "
                               "which is correct for a reference-image axis.",
                }),
                "candidate_mask": ("MASK", {
                    "tooltip": "Optional explicit subject mask for the candidate. "
                               "Omitted = derived from the image's own border colour.",
                }),
                "plate_mask": ("MASK",),
                "region_mask": ("MASK", {
                    "tooltip": "Optional mask of the region this axis paints. Supplying "
                               "it is what turns the colour reading from a whole-frame "
                               "average into a real measurement.",
                }),
                "frame_tolerance": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001,
                    "tooltip": "0 = take it from the recipe. Non-zero overrides.",
                }),
                "colour_tolerance_de": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1,
                    "tooltip": "0 = take it from the recipe.",
                }),
                "identity_tolerance": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001,
                    "tooltip": "0 = take it from the recipe.",
                }),
                # APPENDED LAST: widgets_values is positional, so a new widget anywhere
                # else would shift every later value in graphs already saved.
                "enabled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Off: skip every measurement and pass the candidate "
                               "through. Useful while you are still tuning prompts and "
                               "want images without verdicts - but nothing then stands "
                               "between a drifted render and delivery.",
                }),
            },
        }

    def run(self, candidate, plate, plate_lock_json="", recipe_json="", cell_json="",
            target_region="", target_hex="", candidate_mask=None, plate_mask=None,
            region_mask=None, frame_tolerance=0.0, colour_tolerance_de=0.0,
            identity_tolerance=0.0, enabled=True):
        if not enabled:
            cell = json.loads(cell_json) if str(cell_json or "").strip() else {}
            verdict = {"key": cell.get("key"), "result": RESULT_PASS, "passed": True,
                       "checks": {}, "failures": [],
                       "notes": ["Verification is DISABLED on this node."],
                       "tolerances": {}}
            print("[arkennemasis] verify %s: SKIPPED (disabled)"
                  % (cell.get("key") or "candidate"))
            return (json.dumps(verdict), "VERIFICATION DISABLED - the candidate was "
                    "passed through unmeasured.", True, RESULT_PASS, 0.0)

        candidate_array = _to_array(candidate)
        plate_array = _to_array(plate)
        # Masks belong to the PLATE, so they are loaded at the plate's dimensions —
        # never the candidate's. A generative backend returns whatever size it likes
        # (gpt-image-2 answered 1085x1450 to a 720x961 plate), and `run_checks`
        # resamples the candidate onto the plate's grid before comparing. Sizing the
        # masks off the candidate instead left them mismatched after that resample.
        shape = plate_array.shape[:2]

        plate_lock = json.loads(plate_lock_json) if str(plate_lock_json or "").strip() else {}
        recipe = json.loads(recipe_json) if str(recipe_json or "").strip() else {}
        cell = json.loads(cell_json) if str(cell_json or "").strip() else {}

        region = canonical(target_region)
        wanted_hex = str(target_hex or "").strip()
        painted = []
        if cell:
            slots = sorted(cell.get("slots") or [], key=lambda s: s.get("order") or 99)
            if slots:
                region = region or canonical(slots[0].get("region"))
                wanted_hex = wanted_hex or (slots[0].get("hex") or "")
            # Every axis's region is legitimately changed, not only the leading one.
            painted = [canonical(s.get("region")) for s in slots if s.get("region")]
        if region and region not in painted:
            painted.append(region)

        tolerances = dict(recipe.get("verification") or {})
        if frame_tolerance > 0:
            tolerances["frame_tolerance"] = frame_tolerance
        if colour_tolerance_de > 0:
            tolerances["colour_tolerance_de"] = colour_tolerance_de
        if identity_tolerance > 0:
            tolerances["identity_tolerance"] = identity_tolerance

        region_masks = {}
        for name, record in (plate_lock.get("regions") or {}).items():
            path = record.get("mask") if isinstance(record, dict) else None
            if path and os.path.isfile(path):
                try:
                    region_masks[canonical(name)] = _load_mask_file(path, shape)
                except Exception:
                    pass
        supplied_region = _mask_array(region_mask, shape)
        if supplied_region is not None and region:
            region_masks[region] = supplied_region

        checks, notes = run_checks(
            candidate_array, plate_array, plate_lock,
            target_region=region, target_hex=wanted_hex,
            candidate_mask=_mask_array(candidate_mask, shape),
            plate_mask=_mask_array(plate_mask, shape),
            region_masks=region_masks or None,
            exclude_regions=painted)

        result, failures = decide(checks, tolerances)

        verdict = {
            "key": cell.get("key"),
            "result": result,
            "passed": result == RESULT_PASS,
            "target_region": region,
            "target_hex": wanted_hex,
            "checks": checks,
            "failures": failures,
            "notes": notes,
            "tolerances": tolerances,
        }

        lines = ["VERIFY %s — %s" % (cell.get("key") or "(candidate)", result.upper())]
        if region or wanted_hex:
            lines.append("  target : region '%s'%s"
                         % (region, ", hex %s" % wanted_hex if wanted_hex else ""))
        lines.append("")
        lines.append("MEASUREMENTS")
        for key, value in sorted(checks.items()):
            if key == "ratio_detail":
                continue
            lines.append("  %-28s %s" % (key, value))
        if checks.get("ratio_detail"):
            lines.append("  proportion ratios vs the locked plate:")
            for key, value in sorted(checks["ratio_detail"].items(),
                                     key=lambda kv: -kv[1])[:8]:
                lines.append("      %-30s %.4f" % (key, value))
        if failures:
            lines += ["", "FAILURES"]
            lines += ["  - " + f for f in failures]
        if notes:
            lines += ["", "NOTES"]
            lines += ["  - " + n for n in notes]
        if result == RESULT_PASS:
            lines += ["", "Accepted."]
        elif result == RESULT_SOFT:
            lines += ["", "Soft failure — retry this cell."]
        else:
            lines += ["", "Hard failure — do NOT retry blindly. Repeated hard failure "
                          "usually means a recipe error."]

        print("[arkennemasis] verify %s: %s%s"
              % (cell.get("key") or "candidate", result.upper(),
                 " (dE %.2f)" % checks["colour_de"] if "colour_de" in checks else ""))

        return (json.dumps(verdict, ensure_ascii=False), "\n".join(lines),
                result == RESULT_PASS, result, float(checks.get("colour_de", 0.0)))


class ArkCalibrate:
    CATEGORY = "arkennemasis/Variation"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING", "STRING", "FLOAT", "FLOAT", "FLOAT")
    RETURN_NAMES = ("report", "tolerances_json", "frame_tolerance",
                    "colour_tolerance_de", "identity_tolerance")
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Run every check across an EXISTING set of images and derive thresholds from "
        "measurement instead of guesswork. No generation, no spend. With operator "
        "labels it picks thresholds that separate accepted from rejected and reports "
        "the true current pass rate — which is a number worth having before building "
        "anything."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_dir": ("STRING", {
                    "default": "",
                    "tooltip": "Folder of already-produced images for this product.",
                }),
                "plate": ("IMAGE", {"tooltip": "The locked base plate."}),
            },
            "optional": {
                "plate_lock_json": ("STRING", {
                    "multiline": True, "default": "", "forceInput": True,
                }),
                "labels_csv": ("STRING", {
                    "default": "",
                    "tooltip": "Optional CSV of 'filename,label' where label is accept "
                               "or reject. With it, thresholds are chosen to separate "
                               "the two populations; without it the report is "
                               "descriptive only.",
                }),
                "targets_csv": ("STRING", {
                    "default": "",
                    "tooltip": "Optional CSV of 'filename,region,hex' so the colour "
                               "check has a per-image target.",
                }),
                "percentile": ("FLOAT", {
                    "default": 95.0, "min": 50.0, "max": 100.0, "step": 0.5,
                    "tooltip": "With no labels, thresholds are set at this percentile of "
                               "the observed distribution.",
                }),
                "limit": ("INT", {
                    "default": 0, "min": 0, "max": 10000,
                    "tooltip": "Measure at most this many images. 0 = all.",
                }),
                "write_to": ("STRING", {
                    "default": "",
                    "tooltip": "Optional folder for calibration.json and the per-image "
                               "measurements CSV.",
                }),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def run(self, image_dir, plate, plate_lock_json="", labels_csv="", targets_csv="",
            percentile=95.0, limit=0, write_to=""):
        import csv as csv_module
        from PIL import Image

        folder = str(image_dir or "").strip().strip('"')
        if not os.path.isdir(folder):
            raise ValidationError("ArkCalibrate: %s is not a folder." % folder)

        plate_array = _to_array(plate)
        plate_lock = json.loads(plate_lock_json) if str(plate_lock_json or "").strip() else {}

        labels = {}
        if str(labels_csv or "").strip() and os.path.isfile(labels_csv.strip()):
            with open(labels_csv.strip(), "r", encoding="utf-8-sig", newline="") as handle:
                for row in csv_module.reader(handle):
                    if len(row) >= 2 and row[0].strip():
                        labels[os.path.basename(row[0].strip())] = row[1].strip().lower()

        targets = {}
        if str(targets_csv or "").strip() and os.path.isfile(targets_csv.strip()):
            with open(targets_csv.strip(), "r", encoding="utf-8-sig", newline="") as handle:
                for row in csv_module.reader(handle):
                    if len(row) >= 3 and row[0].strip():
                        targets[os.path.basename(row[0].strip())] = (row[1].strip(),
                                                                     row[2].strip())

        names = [n for n in sorted(os.listdir(folder))
                 if os.path.splitext(n)[1].lower() in (".png", ".jpg", ".jpeg", ".webp")]
        if limit:
            names = names[:int(limit)]
        if not names:
            raise ValidationError("ArkCalibrate: no images in %s." % folder)

        region_masks = {}
        for name, record in (plate_lock.get("regions") or {}).items():
            path = record.get("mask") if isinstance(record, dict) else None
            if path and os.path.isfile(path):
                try:
                    region_masks[canonical(name)] = _load_mask_file(
                        path, plate_array.shape[:2])
                except Exception:
                    pass

        rows = []
        for name in names:
            try:
                with Image.open(os.path.join(folder, name)) as image:
                    array = np.asarray(image.convert("RGB")).astype(np.float32) / 255.0
            except Exception as exc:
                print("[arkennemasis] calibrate: skipping %s (%s)" % (name, exc))
                continue
            region, wanted_hex = targets.get(name, ("", ""))
            checks, _notes = run_checks(
                array, plate_array, plate_lock,
                target_region=canonical(region), target_hex=wanted_hex,
                region_masks=region_masks or None)
            checks["_file"] = name
            checks["_label"] = labels.get(name, "")
            rows.append(checks)

        metrics = ("silhouette_iou", "shape_distance", "ratio_drift",
                   "subject_aspect_drift", "bbox_centre_shift", "bbox_scale_drift",
                   "ssim_untargeted", "colour_de", "bleed_fraction")
        higher_is_better = {"silhouette_iou", "ssim_untargeted"}

        def series(metric, label=None):
            out = []
            for row in rows:
                if label is not None and row.get("_label") != label:
                    continue
                value = row.get(metric)
                if isinstance(value, (int, float)):
                    out.append(float(value))
            return np.array(out, dtype=np.float64)

        lines = ["CALIBRATION — %d image(s) from %s" % (len(rows), folder)]
        accepted = [r for r in rows if r.get("_label") == "accept"]
        rejected = [r for r in rows if r.get("_label") == "reject"]
        if labels:
            lines.append("  labelled: %d accept, %d reject, %d unlabelled"
                         % (len(accepted), len(rejected),
                            len(rows) - len(accepted) - len(rejected)))
            if accepted or rejected:
                total = len(accepted) + len(rejected)
                lines.append("  operator's current pass rate: %.1f%%"
                             % (100.0 * len(accepted) / max(1, total)))
        else:
            lines.append("  no labels supplied — thresholds set at the %.0fth percentile "
                         "of the observed spread, which describes this set rather than "
                         "separating good from bad." % percentile)

        lines += ["", "MEASURED DISTRIBUTIONS"]
        lines.append("  %-24s %8s %8s %8s %8s %8s"
                     % ("metric", "n", "min", "median", "p95", "max"))
        chosen = {}
        for metric in metrics:
            values = series(metric)
            if values.size == 0:
                continue
            lines.append("  %-24s %8d %8.4f %8.4f %8.4f %8.4f"
                         % (metric, values.size, values.min(), np.median(values),
                            np.percentile(values, 95), values.max()))

            if accepted and rejected:
                good = series(metric, "accept")
                bad = series(metric, "reject")
                if good.size and bad.size:
                    if metric in higher_is_better:
                        candidate_threshold = float(np.percentile(good, 5))
                        separation = float(np.median(good) - np.median(bad))
                    else:
                        candidate_threshold = float(np.percentile(good, 95))
                        separation = float(np.median(bad) - np.median(good))
                    chosen[metric] = {
                        "threshold": round(candidate_threshold, 5),
                        "separation": round(separation, 5),
                        "accept_median": round(float(np.median(good)), 5),
                        "reject_median": round(float(np.median(bad)), 5),
                    }
            else:
                if metric in higher_is_better:
                    chosen[metric] = {"threshold": round(
                        float(np.percentile(values, 100 - percentile)), 5)}
                else:
                    chosen[metric] = {"threshold": round(
                        float(np.percentile(values, percentile)), 5)}

        if accepted and rejected:
            lines += ["", "SEPARATION (accept vs reject medians — a metric with no "
                          "separation is not measuring the failure you care about)"]
            for metric, record in sorted(chosen.items(),
                                         key=lambda kv: -abs(kv[1].get("separation", 0))):
                if "separation" in record:
                    lines.append("  %-24s threshold %8.4f   accept %8.4f  reject %8.4f  "
                                 "separation %8.4f"
                                 % (metric, record["threshold"], record["accept_median"],
                                    record["reject_median"], record["separation"]))

        frame = chosen.get("bbox_centre_shift", {}).get("threshold", 0.02)
        colour = chosen.get("colour_de", {}).get("threshold", 5.0)
        identity = chosen.get("ratio_drift", chosen.get("subject_aspect_drift", {})
                              ).get("threshold", 0.06)

        tolerances = {
            "frame_tolerance": round(float(frame), 5),
            "colour_tolerance_de": round(float(colour), 3),
            "identity_tolerance": round(float(identity), 5),
            "min_silhouette_iou": round(float(
                chosen.get("silhouette_iou", {}).get("threshold", 0.90)), 5),
            "max_shape_distance": round(float(
                chosen.get("shape_distance", {}).get("threshold", 0.10)), 6),
            "min_ssim_untargeted": round(float(
                chosen.get("ssim_untargeted", {}).get("threshold", 0.80)), 5),
            "max_bleed_fraction": round(float(
                chosen.get("bleed_fraction", {}).get("threshold", 0.05)), 5),
            "calibrated": True,
            "calibrated_from": folder,
            "sample_size": len(rows),
            "labelled": bool(labels),
        }

        lines += ["", "RECOMMENDED TOLERANCES — paste these into Recipe Compile"]
        for key, value in sorted(tolerances.items()):
            lines.append("  %-24s %s" % (key, value))
        if not labels:
            lines += ["", "These are descriptive, not validated. Label the set "
                          "accept/reject and re-run to get thresholds that actually "
                          "separate good from bad."]

        destination = str(write_to or "").strip()
        if destination:
            if not os.path.isabs(destination):
                try:
                    import folder_paths
                    destination = os.path.join(folder_paths.get_output_directory(),
                                               destination)
                except Exception:
                    destination = os.path.abspath(destination)
            os.makedirs(destination, exist_ok=True)
            with open(os.path.join(destination, "calibration.json"), "w",
                      encoding="utf-8") as handle:
                json.dump({"tolerances": tolerances, "chosen": chosen,
                           "rows": rows}, handle, indent=2, ensure_ascii=False, default=str)
            columns = ["_file", "_label"] + [m for m in metrics
                                             if any(m in r for r in rows)]
            with open(os.path.join(destination, "measurements.csv"), "w",
                      encoding="utf-8", newline="") as handle:
                writer = csv_module.writer(handle)
                writer.writerow(columns)
                for row in rows:
                    writer.writerow([row.get(c, "") for c in columns])
            lines.append("")
            lines.append("Written to %s" % destination)

        print("[arkennemasis] calibrate: %d images, frame %.4f, dE %.2f, identity %.4f"
              % (len(rows), tolerances["frame_tolerance"],
                 tolerances["colour_tolerance_de"], tolerances["identity_tolerance"]))

        return ("\n".join(lines), json.dumps(tolerances, ensure_ascii=False),
                float(tolerances["frame_tolerance"]),
                float(tolerances["colour_tolerance_de"]),
                float(tolerances["identity_tolerance"]))


NODE_CLASS_MAPPINGS = {
    "ArkVerifyCandidate": ArkVerifyCandidate,
    "ArkCalibrate": ArkCalibrate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArkVerifyCandidate": "arkennemasis Verify Candidate (identity/frame/colour)",
    "ArkCalibrate": "arkennemasis Calibrate (derive tolerances from a real set)",
}

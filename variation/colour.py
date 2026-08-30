"""Colour maths for specification targets: sampling, ΔE, swatches, and recolouring.

A hex code is the highest-value data the pipeline receives. It converts verification of
an axis from a subjective human judgement into arithmetic: sample the changed region,
compute the perceptual difference against the specified value, fail anything outside
tolerance. No model and no human is required to make that decision.

Three things have to be right for that to hold:

* **The sample must be robust.** A naive mean across a region is dragged by specular
  highlights and by whatever shows through a transmissive material. Median and trimmed
  mean are used instead, and the brightest and darkest tails are discarded.
* **The difference must be perceptual.** RGB distance is not; ΔE2000 in CIELAB is the
  standard that matches how a person judges "the amber looks wrong".
* **The recolour must preserve shading.** Replacing a region's pixels with a flat
  colour destroys the photograph. The deterministic path transplants hue and chroma
  while keeping the plate's own luminance, so the shading, shadow and highlight
  structure survive untouched.
"""

from __future__ import annotations

import warnings

import numpy as np

# skimage ships with this ComfyUI install and carries reference implementations of both
# the colour-space conversion and CIEDE2000. Reimplementing either by hand is a good way
# to introduce a subtle error into the one number the whole verification stage rests on.
from skimage import color as _skcolor


def hex_to_rgb01(value) -> np.ndarray:
    """`#E78820` -> float array [r, g, b] in 0..1."""
    text = str(value or "").strip().lstrip("#")
    if len(text) == 3:
        text = "".join(c * 2 for c in text)
    if len(text) != 6:
        raise ValueError("Not a hex colour: %r" % (value,))
    return np.array([int(text[i:i + 2], 16) / 255.0 for i in (0, 2, 4)], dtype=np.float64)


def rgb01_to_hex(rgb) -> str:
    arr = np.clip(np.asarray(rgb, dtype=np.float64), 0.0, 1.0)
    return "#" + "".join("%02X" % int(round(c * 255.0)) for c in arr[:3])


def rgb01_to_lab(rgb) -> np.ndarray:
    """One sRGB triple (0..1) -> one CIELAB triple."""
    arr = np.asarray(rgb, dtype=np.float64).reshape(1, 1, 3)
    return _skcolor.rgb2lab(arr).reshape(3)


def delta_e(rgb_a, rgb_b) -> float:
    """CIEDE2000 difference between two sRGB triples in 0..1.

    Rough reading of the scale: under 1 is imperceptible, 1-2 is visible only on a
    direct side-by-side, 2-5 is a noticeable difference, above 5 reads as a different
    colour. Commercially acceptable tolerance is a contractual matter as much as a
    technical one, so it lives in the recipe and is agreed with the client — it is
    never hard-coded here.
    """
    lab_a = rgb01_to_lab(rgb_a).reshape(1, 1, 3)
    lab_b = rgb01_to_lab(rgb_b).reshape(1, 1, 3)
    return float(_skcolor.deltaE_ciede2000(lab_a, lab_b).reshape(()))


def sample_region(rgb_image, mask=None, trim=0.20, statistic="median"):
    """A robust representative colour for a masked region.

    `rgb_image` is (H, W, 3) float 0..1. `mask` is (H, W) truthy where the region is.

    Specular highlights on a glossy surface and dark shadow at a region's edge are both
    real pixels and neither represents the material. Both tails are trimmed by
    luminance before the statistic is taken, which is what stops a gloss hotspot
    dragging a mid-brown reading towards white.

    Returns `(rgb, pixel_count)`; rgb is None when the region is empty.
    """
    image = np.asarray(rgb_image, dtype=np.float64)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("sample_region expects an (H, W, 3) image.")
    image = image[:, :, :3]

    if mask is None:
        pixels = image.reshape(-1, 3)
    else:
        selector = np.asarray(mask)
        if selector.ndim == 3:
            selector = selector[:, :, 0]
        pixels = image[selector > 0.5]

    if pixels.size == 0:
        return None, 0

    if 0.0 < trim < 0.5 and len(pixels) >= 10:
        # Rank by luminance, drop both tails. Rec.709 luma is close enough for ordering.
        luma = pixels @ np.array([0.2126, 0.7152, 0.0722])
        order = np.argsort(luma)
        cut = int(len(order) * trim)
        keep = order[cut:len(order) - cut] if len(order) - 2 * cut > 0 else order
        pixels = pixels[keep]

    if statistic == "mean":
        value = pixels.mean(axis=0)
    else:
        value = np.median(pixels, axis=0)
    return value, int(len(pixels))


def swatch_image(hex_value, size=256) -> np.ndarray:
    """A flat swatch for a hex spec, as an (H, W, 3) float image.

    Rendered so a hex axis has a visible artefact in the reference library exactly like
    an image axis does — the operator reviewing the library sees colours, not strings,
    and a transposed hex is obvious on sight.
    """
    rgb = hex_to_rgb01(hex_value)
    return np.tile(rgb.astype(np.float32), (int(size), int(size), 1))


def recolour_region(rgb_image, mask, target_hex, strength=1.0, preserve_luma=True,
                    feather=0.0):
    """Retint a masked region to a target colour while keeping the plate's shading.

    This is the deterministic path (spec §10.3). Where an axis specifies a hex and
    paints a reasonably simple surface, this route is faster, cheaper and exact, and it
    **cannot drift by construction** — no pixel outside the mask is touched, so the
    camera, the framing, the shadow and the product's geometry are mathematically
    unchanged rather than merely asked to stay put.

    The work happens in CIELAB. The region's `a*`/`b*` (its hue and chroma) are replaced
    with the target's, while `L*` (its lightness) is kept from the original and only
    re-levelled so the region's mean lightness matches the target's. That preserves
    every gradient, highlight and shadow the photograph already had.

    Not suitable for materials with structure — figured timber, veined marble, woven
    textile. Those genuinely have to be invented and belong on the generative path.
    """
    image = np.asarray(rgb_image, dtype=np.float64)[:, :, :3]
    height, width = image.shape[:2]

    alpha = np.asarray(mask, dtype=np.float64)
    if alpha.ndim == 3:
        alpha = alpha[:, :, 0]
    if alpha.shape != (height, width):
        raise ValueError("recolour_region: mask shape %s does not match image %s"
                         % (alpha.shape, (height, width)))
    if alpha.max() > 1.0:
        alpha = alpha / 255.0
    alpha = np.clip(alpha, 0.0, 1.0)

    if feather and feather > 0:
        # A hard mask edge reads as a cut-out. A small blur on the alpha only — never on
        # the colour — keeps the transition photographic without bleeding the new colour
        # outside the region.
        try:
            import cv2
            radius = int(max(1, round(float(feather))))
            kernel = radius * 2 + 1
            alpha = cv2.GaussianBlur(alpha.astype(np.float32), (kernel, kernel), 0)
            alpha = np.clip(alpha.astype(np.float64), 0.0, 1.0)
        except Exception:
            pass

    lab = _skcolor.rgb2lab(np.clip(image, 0.0, 1.0))
    target_lab = rgb01_to_lab(hex_to_rgb01(target_hex))

    selected = alpha > 0.01
    if not np.any(selected):
        return (np.clip(image, 0.0, 1.0).astype(np.float32), 0,
                np.zeros_like(alpha, dtype=np.float64))

    out = lab.copy()
    out[:, :, 1] = target_lab[1]
    out[:, :, 2] = target_lab[2]

    if preserve_luma:
        # Re-level lightness so the region's average matches the target's, without
        # flattening it: every pixel keeps its own deviation from the regional mean.
        current_mean = float(np.average(lab[:, :, 0], weights=alpha))
        out[:, :, 0] = np.clip(lab[:, :, 0] + (target_lab[0] - current_mean), 0.0, 100.0)
    else:
        out[:, :, 0] = target_lab[0]

    # Forcing a target chroma onto the plate's own lightness can land outside the sRGB
    # gamut in shadows and highlights, and skimage warns per call as it clips. That is
    # the intended behaviour — clipping to the nearest displayable colour is exactly
    # right here — and a warning per image would bury a 700-cell run's real log.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*negative Z values.*")
        recoloured = _skcolor.lab2rgb(out)
    blend = np.clip(alpha * float(strength), 0.0, 1.0)[:, :, None]
    result = image * (1.0 - blend) + recoloured * blend
    # The effective alpha is returned so a caller can state exactly which pixels were
    # touched. Feathering deliberately widens the mask by a pixel or two, so the honest
    # claim is "nothing outside the feathered support changed", not "nothing outside the
    # original mask changed" — and the difference is checkable rather than asserted.
    return (np.clip(result, 0.0, 1.0).astype(np.float32), int(selected.sum()),
            blend[:, :, 0])


def coverage_outside(mask_considered, candidate_rgb, plate_rgb, target_hex,
                     tolerance_de):
    """Fraction of untouched area that MOVED towards the target colour — bleed (§7.3).

    Bleed is a change, not a resemblance. Asking only "is this pixel now close to the
    target colour" flags every region that was already that colour: a white sole scores
    as 92% bled when the axis target is ivory, and the cell is rejected for being
    correct. So a pixel counts only when it is within tolerance of the target NOW and
    was NOT within tolerance in the plate.

    `mask_considered` must already exclude every region an axis legitimately paints —
    those changed on purpose and are not bleed.
    """
    candidate = np.asarray(candidate_rgb, dtype=np.float64)[:, :, :3]
    plate = np.asarray(plate_rgb, dtype=np.float64)[:, :, :3]
    region = np.asarray(mask_considered)
    if region.ndim == 3:
        region = region[:, :, 0]

    selected = region > 0.5
    if not selected.any():
        return 0.0, 0

    target_lab = rgb01_to_lab(hex_to_rgb01(target_hex)).reshape(1, 1, 3)

    def within(pixels):
        lab = _skcolor.rgb2lab(pixels.reshape(-1, 1, 3))
        diffs = _skcolor.deltaE_ciede2000(lab, np.tile(target_lab, (len(pixels), 1, 1)))
        return diffs.reshape(-1) <= float(tolerance_de)

    candidate_pixels = candidate[selected]
    plate_pixels = plate[selected]
    became = within(candidate_pixels) & ~within(plate_pixels)
    return float(became.mean()), int(selected.sum())

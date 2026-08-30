"""Turn a raw segmentation mask into one you can composite through.

A segmenter answers each frame on its own merits. That is correct per frame and wrong as
a video: the threshold lands a pixel differently on frame 41 than on frame 40, a hand
drops out for three frames, a hole opens in a dark jacket, and the cut-out boils along
every edge. None of that is visible in a still — it is only visible once the frames run.

So this is the step between "we have masks" and "we can composite":

  threshold   a soft mask has a halo of 0.3 pixels that composite as a grey fringe
  keep_largest  a stray blob of background scored as person is a floating artefact
  fill_holes    the inside of a jacket is not background just because it is dark
  grow          segmenters cut slightly INSIDE the subject; a pixel or two out recovers
                the hair edge, and a negative value pulls in when the cut is too generous
  temporal      the frame-to-frame boil, averaged away over a small window
  feather       a hard edge reads as a sticker; a soft one reads as a cut-out

Order matters and is fixed: shape first, then time, then softness. Feathering before the
temporal average would blur the softness itself into a wider, weaker edge, and growing
after feathering would push the soft ramp outward instead of the mask.

Generic. Any masked-subject composite wants this; nothing here knows what it is cutting.
"""

from __future__ import annotations

import numpy as np
import torch


def _binary(frame, threshold):
    return frame >= threshold


def _largest_component(binary, min_area_fraction):
    """Keep the biggest connected blob, plus anything comparable to it.

    NOT strictly the largest. A person whose arm is separated from the torso by an
    occluder is two components, and keeping only the bigger one deletes the arm — the
    same mistake `binary_mask_save` made when a stand clamp bisected a device. So
    components within `min_area_fraction` of the largest survive too.
    """
    try:
        from scipy import ndimage
    except ImportError:
        return binary
    labelled, count = ndimage.label(binary)
    if count <= 1:
        return binary
    sizes = ndimage.sum(binary, labelled, range(1, count + 1))
    biggest = float(sizes.max())
    if biggest <= 0:
        return binary
    keep = {i + 1 for i, size in enumerate(sizes)
            if size >= biggest * max(0.0, min_area_fraction)}
    return np.isin(labelled, list(keep))


def _grow(binary, pixels):
    """Dilate (positive) or erode (negative) by a square structuring element."""
    if not pixels:
        return binary
    try:
        from scipy import ndimage
    except ImportError:
        return binary
    size = abs(int(pixels)) * 2 + 1
    if pixels > 0:
        return ndimage.maximum_filter(binary, size=size)
    return ndimage.minimum_filter(binary, size=size)


def _feather(mask, radius):
    if radius <= 0:
        return mask
    try:
        from scipy import ndimage
    except ImportError:
        return mask
    # sigma ~= radius/2 puts the visible ramp at about `radius` pixels wide.
    return ndimage.gaussian_filter(mask, sigma=max(0.5, radius / 2.0))


class ArkMaskRefine:
    CATEGORY = "arkennemasis/Utility"
    FUNCTION = "run"
    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("masks", "report")
    DESCRIPTION = ("Clean a segmentation mask so it can be composited: threshold, drop "
                   "stray blobs, fill holes, grow or shrink the edge, smooth the "
                   "frame-to-frame boil, and feather. Built for video masks.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("MASK", {
                    "tooltip": "The raw masks, one per frame.",
                }),
                "threshold": ("FLOAT", {
                    "default": 0.5, "min": 0.01, "max": 0.99, "step": 0.01,
                    "tooltip": "Below this a pixel is background. A soft mask "
                               "composited straight through shows a grey fringe.",
                }),
                "keep_largest": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Drop blobs that are not part of the subject. Parts "
                               "comparable in size to the biggest are kept, so a limb "
                               "split off by an occluder does not vanish.",
                }),
                "fill_holes": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Close enclosed gaps. Turn OFF when a genuine hole "
                               "through the subject must stay transparent.",
                }),
                "grow": ("INT", {
                    "default": 1, "min": -20, "max": 20,
                    "tooltip": "Pixels to expand (+) or contract (-) the edge. "
                               "Segmenters usually cut just inside the subject, so a "
                               "small positive value recovers the hair line.",
                }),
                "temporal_smooth": ("INT", {
                    "default": 3, "min": 0, "max": 15,
                    "tooltip": "Frames to average the mask over. This is the setting "
                               "that stops the edge boiling. 0 disables it; keep it "
                               "small or a fast gesture smears.",
                }),
                "feather": ("INT", {
                    "default": 2, "min": 0, "max": 40,
                    "tooltip": "Softness of the final edge, in pixels. A hard edge "
                               "reads as a sticker pasted on the background.",
                }),
            },
            "optional": {
                "min_area_pct": ("FLOAT", {
                    "default": 10.0, "min": 0.0, "max": 100.0, "step": 1.0,
                    "tooltip": "A component this percentage of the largest is kept. "
                               "0 keeps every blob, 100 keeps strictly the largest.",
                }),
                "invert": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Flip the mask. Use when the segmenter returned the "
                               "background rather than the subject.",
                }),
            },
        }

    def run(self, masks, threshold=0.5, keep_largest=True, fill_holes=True, grow=1,
            temporal_smooth=3, feather=2, min_area_pct=10.0, invert=False):
        if masks is None or masks.numel() == 0:
            raise RuntimeError("ArkMaskRefine got an empty MASK.")
        # A MASK is (B, H, W). A single mask sometimes arrives as (H, W); treat it as a
        # one-frame batch rather than indexing rows as frames.
        data = masks.detach().cpu().float()
        if data.dim() == 2:
            data = data[None, ...]
        array = data.numpy()
        if invert:
            array = 1.0 - array

        frames, height, width = array.shape
        cleaned = np.empty_like(array)
        empty = 0
        for index in range(frames):
            binary = _binary(array[index], threshold)
            if keep_largest:
                binary = _largest_component(binary, min_area_pct / 100.0)
            if fill_holes:
                try:
                    from scipy import ndimage
                    binary = ndimage.binary_fill_holes(binary)
                except ImportError:
                    pass
            binary = _grow(binary, grow)
            if not binary.any():
                empty += 1
            cleaned[index] = binary.astype(np.float32)

        if temporal_smooth and frames > 1:
            # A centred moving average over an odd window. Edges of the clip use a
            # shorter window rather than repeating frames, so the first and last frames
            # are not dragged toward their neighbours.
            radius = int(temporal_smooth) // 2
            if radius > 0:
                smoothed = np.empty_like(cleaned)
                for index in range(frames):
                    lo, hi = max(0, index - radius), min(frames, index + radius + 1)
                    smoothed[index] = cleaned[lo:hi].mean(axis=0)
                cleaned = smoothed

        if feather:
            for index in range(frames):
                cleaned[index] = _feather(cleaned[index], feather)

        cleaned = np.clip(cleaned, 0.0, 1.0)
        coverage = float(cleaned.mean()) * 100.0
        report = ("%d frames %dx%d | subject covers %.1f%% | grow %+d | temporal %d | "
                  "feather %d%s" % (frames, width, height, coverage, grow,
                                    temporal_smooth, feather,
                                    "" if not empty else
                                    " | WARNING: %d empty frame(s)" % empty))
        if empty:
            # Loud, because an empty mask composites as a hole where the subject was and
            # is far easier to explain here than to spot in a finished video.
            print("[arkennemasis] mask refine: %d frame(s) came back EMPTY — the "
                  "segmenter found nothing there. Lower the detection threshold or "
                  "check the prompt." % empty)
        print("[arkennemasis] mask refine: %s" % report)
        return (torch.from_numpy(cleaned), report)


NODE_CLASS_MAPPINGS = {"ArkMaskRefine": ArkMaskRefine}
NODE_DISPLAY_NAME_MAPPINGS = {"ArkMaskRefine": "arkennemasis Mask Refine (clean a video mask)"}

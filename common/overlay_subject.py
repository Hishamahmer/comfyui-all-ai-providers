"""Stand a cut-out subject in front of a background, at a chosen size and position.

This is the piece a hosted avatar service does for you and never lets you see. HeyGen's
request body carries `character.scale`, `character.offset` and `background.fit`, and those
three numbers are the whole composition: how big the presenter is, where they stand, and
how the backdrop fills the frame. Doing it locally means owning them.

    background ──┐
                 ├─► canvas (W x H) ──► subject scaled to `subject_scale` of the height,
    subject ─────┤                      placed by `anchor` + `offset_x/offset_y`
    mask    ─────┘

**The subject is cropped to its mask before it is scaled**, and that is the difference
between this working and not. A generated talking-head frame is a whole picture with a
person somewhere inside it; scaling that by 0.7 scales the empty room too, and the
presenter ends up small and off to one side. Cropping to the mask first makes
`subject_scale` mean what it says: the fraction of the canvas height the PERSON occupies.

**The crop box is the union across every frame by default.** Per-frame boxes track the
subject more tightly, and they also breathe — the box grows a pixel when a hand lifts, so
the whole person shifts and scales slightly on that frame. Over a clip that reads as a
wobble. One box for the whole clip is a locked-off camera; that is what a presenter shot
should be.

**Frames are composited in chunks.** A minute of 720x1280 at 25 fps is 1500 frames, and
float32 RGB at that size is 11 MB each — 16 GB for one copy of the batch. The output is
preallocated once and filled a slice at a time, so peak memory is the output plus one
chunk rather than the output plus several full-size intermediates.

Nothing here is avatar-specific: a foreground, its mask, and a backdrop.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

COVER = "cover — fill the canvas, crop the overflow"
CONTAIN = "contain — fit inside, pad the rest"
STRETCH = "stretch — distort to fit exactly"
FITS = [COVER, CONTAIN, STRETCH]

BOTTOM, CENTRE, TOP = "bottom", "centre", "top"
ANCHORS = [BOTTOM, CENTRE, TOP]

UNION = "one box for the whole clip (locked off)"
PER_FRAME = "a box per frame (tighter, breathes)"
BOXES = [UNION, PER_FRAME]

# Frames per composite pass. Small enough that one chunk of working tensors is
# insignificant beside the output, large enough that the Python loop is not the cost.
CHUNK = 24


def _resize(nhwc, height, width):
    """Resize an (N,H,W,C) batch, with antialiasing on the way down."""
    chw = nhwc.movedim(-1, 1)
    out = F.interpolate(chw, size=(int(height), int(width)), mode="bilinear",
                        align_corners=False, antialias=True)
    return out.movedim(1, -1)


def fit_background(background, canvas_h, canvas_w, mode):
    """One background frame, resized to the canvas by the chosen rule."""
    _, src_h, src_w, _ = background.shape
    if mode == STRETCH:
        return _resize(background, canvas_h, canvas_w)

    scale_w, scale_h = canvas_w / src_w, canvas_h / src_h
    # cover takes the LARGER scale so neither axis is left short; contain takes the
    # smaller so nothing spills. That one choice is the whole difference between them.
    scale = max(scale_w, scale_h) if mode == COVER else min(scale_w, scale_h)
    new_h, new_w = max(1, round(src_h * scale)), max(1, round(src_w * scale))
    resized = _resize(background, new_h, new_w)

    canvas = torch.zeros((1, canvas_h, canvas_w, 3), dtype=torch.float32)
    if mode == COVER:
        top = max(0, (new_h - canvas_h) // 2)
        left = max(0, (new_w - canvas_w) // 2)
        canvas[:] = resized[:, top:top + canvas_h, left:left + canvas_w, :]
    else:
        top = max(0, (canvas_h - new_h) // 2)
        left = max(0, (canvas_w - new_w) // 2)
        canvas[:, top:top + new_h, left:left + new_w, :] = resized
    return canvas


def mask_box(mask, pad_fraction=0.0):
    """(top, left, bottom, right) of everything the mask touches, or None if empty."""
    rows = mask.amax(dim=-1)
    cols = mask.amax(dim=-2)
    row_hits = torch.nonzero(rows > 0.02).flatten()
    col_hits = torch.nonzero(cols > 0.02).flatten()
    if row_hits.numel() == 0 or col_hits.numel() == 0:
        return None
    top, bottom = int(row_hits[0]), int(row_hits[-1]) + 1
    left, right = int(col_hits[0]), int(col_hits[-1]) + 1
    if pad_fraction > 0:
        pad_y = int((bottom - top) * pad_fraction)
        pad_x = int((right - left) * pad_fraction)
        top, bottom = max(0, top - pad_y), min(mask.shape[-2], bottom + pad_y)
        left, right = max(0, left - pad_x), min(mask.shape[-1], right + pad_x)
    return top, left, bottom, right


class ArkOverlaySubject:
    CATEGORY = "arkennemasis/Utility"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("images", "placed_mask", "report")
    DESCRIPTION = ("Composite a masked subject over a background at a chosen canvas "
                   "size, scale and position. The subject is cropped to its mask first, "
                   "so `subject_scale` is the fraction of the frame the subject fills.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "background": ("IMAGE", {
                    "tooltip": "The backdrop. One frame is reused for every subject "
                               "frame; a batch is used frame for frame.",
                }),
                "subject": ("IMAGE", {
                    "tooltip": "The subject's frames — the whole picture it was "
                               "generated in. Cropping happens here.",
                }),
                "subject_mask": ("MASK", {
                    "tooltip": "White where the subject is. Refine it before this: a "
                               "raw segmentation boils along the edge.",
                }),
                "canvas_width": ("INT", {
                    "default": 720, "min": 64, "max": 8192,
                    "tooltip": "Output width. 720x1280 is the vertical format the "
                               "short-form platforms want.",
                }),
                "canvas_height": ("INT", {
                    "default": 1280, "min": 64, "max": 8192,
                    "tooltip": "Output height.",
                }),
                "subject_scale": ("FLOAT", {
                    "default": 0.70, "min": 0.05, "max": 3.0, "step": 0.01,
                    "tooltip": "The subject's height as a fraction of the canvas "
                               "height, measured on the cropped subject. Above 1.0 the "
                               "subject is taller than the frame and gets cropped by it.",
                }),
                "offset_x": ("FLOAT", {
                    "default": 0.25, "min": -1.5, "max": 1.5, "step": 0.01,
                    "tooltip": "Sideways shift as a fraction of canvas width, from the "
                               "centre. Positive moves right.",
                }),
                "offset_y": ("FLOAT", {
                    "default": 0.0, "min": -1.5, "max": 1.5, "step": 0.01,
                    "tooltip": "Vertical shift as a fraction of canvas height, from the "
                               "anchor. Positive moves down.",
                }),
                "anchor": (ANCHORS, {
                    "tooltip": "Where the subject sits before the offsets. `bottom` "
                               "stands them on the lower edge, which is how a presenter "
                               "is framed.",
                }),
                "background_fit": (FITS, {
                    "tooltip": "How the backdrop fills the canvas when its shape differs.",
                }),
            },
            "optional": {
                "crop_to_subject": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Crop to the mask before scaling. OFF makes "
                               "`subject_scale` mean the fraction of the frame the whole "
                               "SOURCE PICTURE fills, which is almost never what you want.",
                }),
                "crop_box": (BOXES, {
                    "tooltip": "One box for the clip holds the framing still. A box per "
                               "frame tracks tighter and makes the subject breathe.",
                }),
                "crop_padding": ("FLOAT", {
                    "default": 0.02, "min": 0.0, "max": 0.5, "step": 0.01,
                    "tooltip": "Extra margin around the mask box, as a fraction of its "
                               "size, so a feathered edge is not clipped by its own box.",
                }),
                "opacity": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Subject opacity. Below 1.0 the background shows through.",
                }),
            },
        }

    def run(self, background, subject, subject_mask, canvas_width=720,
            canvas_height=1280, subject_scale=0.70, offset_x=0.25, offset_y=0.0,
            anchor=BOTTOM, background_fit=COVER, crop_to_subject=True, crop_box=UNION,
            crop_padding=0.02, opacity=1.0):
        if subject is None or subject.numel() == 0:
            raise RuntimeError("ArkOverlaySubject got no subject frames.")
        if subject_mask is None or subject_mask.numel() == 0:
            raise RuntimeError("ArkOverlaySubject got no subject mask.")

        subject = subject.detach().cpu().float()[..., :3]
        masks = subject_mask.detach().cpu().float()
        if masks.dim() == 2:
            masks = masks[None, ...]
        background = background.detach().cpu().float()[..., :3]

        frames = subject.shape[0]
        if masks.shape[0] == 1 and frames > 1:
            masks = masks.expand(frames, -1, -1)
        if masks.shape[0] != frames:
            raise RuntimeError(
                "ArkOverlaySubject: %d subject frames but %d masks. They come from the "
                "same clip, so a mismatch means one side was resampled."
                % (frames, masks.shape[0]))
        if masks.shape[-2:] != subject.shape[1:3]:
            # The segmenter may work at its own resolution. Put the mask back on the
            # subject's grid rather than refusing — but do it here, once, where it is
            # visible, instead of letting it broadcast wrongly later.
            masks = _resize(masks[..., None], subject.shape[1],
                            subject.shape[2])[..., 0]

        canvas_h, canvas_w = int(canvas_height), int(canvas_width)
        backdrop_is_batch = background.shape[0] > 1
        if backdrop_is_batch and background.shape[0] != frames:
            raise RuntimeError(
                "ArkOverlaySubject: the background is a %d-frame batch but the subject "
                "has %d frames." % (background.shape[0], frames))
        if not backdrop_is_batch:
            backdrop = fit_background(background, canvas_h, canvas_w, background_fit)[0]

        # One box for the whole clip: measured on the union of every frame's mask, so
        # nothing any frame shows can fall outside it.
        shared_box = None
        if crop_to_subject and crop_box == UNION:
            shared_box = mask_box(masks.amax(dim=0), crop_padding)
            if shared_box is None:
                raise RuntimeError(
                    "ArkOverlaySubject: the mask is empty on every frame, so there is "
                    "no subject to place. Check the segmentation step.")

        out = torch.empty((frames, canvas_h, canvas_w, 3), dtype=torch.float32)
        placed = torch.zeros((frames, canvas_h, canvas_w), dtype=torch.float32)
        geometry = None

        for start in range(0, frames, CHUNK):
            stop = min(frames, start + CHUNK)
            chunk_rgb = subject[start:stop]
            chunk_mask = masks[start:stop]

            if crop_to_subject:
                box = shared_box or mask_box(chunk_mask.amax(dim=0), crop_padding)
                if box is not None:
                    top, left, bottom, right = box
                    chunk_rgb = chunk_rgb[:, top:bottom, left:right, :]
                    chunk_mask = chunk_mask[:, top:bottom, left:right]

            src_h, src_w = chunk_rgb.shape[1], chunk_rgb.shape[2]
            target_h = max(1, round(canvas_h * float(subject_scale)))
            target_w = max(1, round(src_w * (target_h / src_h)))
            chunk_rgb = _resize(chunk_rgb, target_h, target_w)
            chunk_mask = _resize(chunk_mask[..., None], target_h, target_w)[..., 0]
            chunk_mask = chunk_mask.clamp(0.0, 1.0) * float(opacity)

            # Placement. x is measured from the canvas centre; y from the anchor.
            centre_x = canvas_w / 2.0 + float(offset_x) * canvas_w
            left_px = round(centre_x - target_w / 2.0)
            if anchor == BOTTOM:
                top_px = canvas_h - target_h
            elif anchor == TOP:
                top_px = 0
            else:
                top_px = round((canvas_h - target_h) / 2.0)
            top_px = round(top_px + float(offset_y) * canvas_h)

            # Clip to the canvas. The subject is allowed to hang off any edge — a
            # presenter cropped at the waist by the bottom of the frame is normal — so
            # this trims the copy rather than refusing the placement.
            src_top = max(0, -top_px)
            src_left = max(0, -left_px)
            dst_top = max(0, top_px)
            dst_left = max(0, left_px)
            copy_h = min(target_h - src_top, canvas_h - dst_top)
            copy_w = min(target_w - src_left, canvas_w - dst_left)

            if backdrop_is_batch:
                base = fit_background(background[start:stop], canvas_h, canvas_w,
                                      background_fit)
                out[start:stop] = base
            else:
                out[start:stop] = backdrop

            if copy_h > 0 and copy_w > 0:
                fg = chunk_rgb[:, src_top:src_top + copy_h, src_left:src_left + copy_w, :]
                alpha = chunk_mask[:, src_top:src_top + copy_h,
                                   src_left:src_left + copy_w][..., None]
                window = out[start:stop, dst_top:dst_top + copy_h,
                             dst_left:dst_left + copy_w, :]
                out[start:stop, dst_top:dst_top + copy_h,
                    dst_left:dst_left + copy_w, :] = window * (1.0 - alpha) + fg * alpha
                placed[start:stop, dst_top:dst_top + copy_h,
                       dst_left:dst_left + copy_w] = alpha[..., 0]

            if geometry is None:
                geometry = (src_h, src_w, target_w, target_h, left_px, top_px,
                            copy_h > 0 and copy_w > 0)

        src_h, src_w, target_w, target_h, left_px, top_px, visible = geometry
        report = ("%d frames -> %dx%d | subject %dx%d cropped to %dx%d, placed at "
                  "(%d,%d) | %s%s"
                  % (frames, canvas_w, canvas_h, src_w, src_h, target_w, target_h,
                     left_px, top_px, background_fit.split(" ")[0],
                     "" if visible else " | WARNING: the subject fell outside the canvas"))
        if not visible:
            print("[arkennemasis] overlay: the offsets put the subject entirely off "
                  "the canvas — nothing of it is visible.")
        print("[arkennemasis] overlay: %s" % report)
        return (out, placed, report)


NODE_CLASS_MAPPINGS = {"ArkOverlaySubject": ArkOverlaySubject}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ArkOverlaySubject": "arkennemasis Overlay Subject (cut-out onto a background)",
}

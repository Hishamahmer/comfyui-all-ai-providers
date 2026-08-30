"""Stage 2 — resolve every specification into a cached, hashed, concrete artefact.

A reference that changes silently produces a batch that drifts silently. So every
reference image is downloaded once, stored locally, and content-hashed; every hex is
recorded and rendered as a swatch. From that point the run reads only local files, and
two runs a month apart against the same library produce the same inputs.

The library accumulates per client across products and never resets. Generation tooling
is rentable by anyone; a validated per-brand material library built over dozens of
products is not — it is the most defensible asset this pipeline creates, which is why it
is a first-class directory on disk rather than a temp folder.

Multi-reference values are kept as a list with explicit roles, never merged. Averaging a
flat material sample with a lit in-situ example produces a third material matching
neither.
"""

from __future__ import annotations

import json
import os
import shutil
import urllib.parse
import urllib.request

import numpy as np
import torch

from .colour import hex_to_rgb01, rgb01_to_hex, swatch_image
from .schema import ValidationError, canonical, normalise_hex, sha256_bytes, spec_type_of

_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")


def _resolve_dir(path, default_name="variation-library"):
    path = str(path or "").strip()
    if not path:
        path = default_name
    if not os.path.isabs(path):
        try:
            import folder_paths
            path = os.path.join(folder_paths.get_output_directory(), path)
        except Exception:
            path = os.path.abspath(path)
    os.makedirs(path, exist_ok=True)
    return path


def _fetch(url, timeout=120):
    """Bytes for a reference, from http(s), a file:// URL, or a local path."""
    text = str(url).strip()
    parsed = urllib.parse.urlparse(text)

    if parsed.scheme in ("http", "https"):
        request = urllib.request.Request(
            text, headers={"User-Agent": "arkennemasis-variation/1.0"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()

    if parsed.scheme == "file":
        local = urllib.request.url2pathname(parsed.path)
        with open(local, "rb") as handle:
            return handle.read()

    if os.path.isfile(text):
        with open(text, "rb") as handle:
            return handle.read()

    raise ValidationError("Cannot resolve reference: %s" % text)


def _extension_for(url, data):
    guess = os.path.splitext(urllib.parse.urlparse(str(url)).path)[1].lower()
    if guess in _IMAGE_EXTENSIONS:
        return guess
    if data[:8].startswith(b"\x89PNG"):
        return ".png"
    if data[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    return ".png"


def _load_image_array(path):
    from PIL import Image
    with Image.open(path) as image:
        image = image.convert("RGB")
        return np.asarray(image).astype(np.float32) / 255.0


def _to_batch(arrays, size=512):
    """Stack per-value preview images into one IMAGE batch for the canvas.

    Everything is resampled to a common square so the batch is one shape; this output is
    for looking at, not for generation, so the resize is not a fidelity concern.
    """
    from PIL import Image
    if not arrays:
        return torch.zeros((1, 8, 8, 3), dtype=torch.float32)
    frames = []
    for array in arrays:
        pil = Image.fromarray((np.clip(array, 0, 1) * 255).astype(np.uint8))
        pil = pil.convert("RGB").resize((int(size), int(size)), Image.LANCZOS)
        frames.append(torch.from_numpy(np.asarray(pil).astype(np.float32) / 255.0))
    return torch.stack(frames, dim=0)


class ArkSpecLibrary:
    CATEGORY = "arkennemasis/Variation"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING", "IMAGE", "STRING", "STRING", "INT")
    RETURN_NAMES = ("library_json", "previews", "report", "unresolved", "value_count")
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Download, cache and hash every reference image; record and render every hex. "
        "Produces the per-client material library the whole run reads from, so a "
        "reference that changes upstream cannot silently change a batch."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "intake_json": ("STRING", {
                    "multiline": True, "default": "{}", "forceInput": True,
                    "tooltip": "From Variation Intake.",
                }),
                "library_dir": ("STRING", {
                    "default": "variation/library",
                    "tooltip": "Where cached references live. Relative paths resolve "
                               "inside ComfyUI's output directory. Shared across "
                               "products for one client — do not make it per-run.",
                }),
            },
            "optional": {
                "refresh": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Re-download references that are already cached. Off is "
                               "correct almost always: the cache is the point.",
                }),
                "swatch_size": ("INT", {
                    "default": 512, "min": 64, "max": 2048,
                    "tooltip": "Pixel size of a rendered hex swatch.",
                }),
                "timeout_seconds": ("INT", {
                    "default": 120, "min": 5, "max": 3600,
                }),
                "strict": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "On: an unresolvable reference stops the run. A missing "
                               "material is the one thing that must never be guessed.",
                }),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")     # cheap when cached; always re-checks what is on disk

    def run(self, intake_json, library_dir, refresh=False, swatch_size=512,
            timeout_seconds=120, strict=True):
        try:
            intake = json.loads(intake_json or "{}")
        except ValueError as exc:
            raise ValidationError("intake_json is not valid JSON: %s" % exc)

        specs = intake.get("specs") or []
        if not specs:
            raise ValidationError("Intake carries no specifications to resolve.")

        product = (intake.get("product") or {}).get("product") or "product"
        root = _resolve_dir(library_dir)
        refs_root = os.path.join(root, "refs")
        swatch_root = os.path.join(root, "swatches")
        os.makedirs(refs_root, exist_ok=True)
        os.makedirs(swatch_root, exist_ok=True)

        library = {"product": product, "root": root, "values": {}}
        previews, lines, unresolved = [], [], []
        resolved_count = 0

        for entry in sorted(specs, key=lambda e: (e.get("axis", ""), e.get("value", ""))):
            axis = canonical(entry.get("axis"))
            value = canonical(entry.get("value"))
            if not axis or not value:
                continue

            record = {
                "axis": axis,
                "value": value,
                "display": entry.get("display") or value,
                "filename_token": entry.get("filename_token") or value,
                "description": entry.get("description") or "",
                "unchanged": bool(entry.get("unchanged")),
                "hex": normalise_hex(entry.get("hex")),
                "refs": [],
                "spec_type": spec_type_of(entry),
            }

            axis_dir = os.path.join(refs_root, axis)
            os.makedirs(axis_dir, exist_ok=True)

            for position, reference in enumerate(entry.get("refs") or []):
                url = reference.get("url")
                role = reference.get("role") or "material"
                try:
                    data = _fetch(url, timeout=timeout_seconds)
                except Exception as exc:
                    unresolved.append("%s / %s: %s (%s)" % (axis, value, url, exc))
                    lines.append("  FAIL  %-14s %-28s %s" % (axis, value, exc))
                    continue

                digest = sha256_bytes(data)
                suffix = "" if position == 0 else "_%d" % (position + 1)
                filename = "%s%s%s" % (value, suffix, _extension_for(url, data))
                path = os.path.join(axis_dir, filename)

                # Rewrite whenever the cached bytes are not the bytes just fetched.
                # Testing only for the file's EXISTENCE meant a reference that changed
                # at its source was never replaced, while the hash recorded alongside it
                # was the new bytes' hash — so library.json and the recipe described one
                # image and the generator was sent a different one, with the mismatch
                # invisible precisely because the hash looked correct.
                cached_digest = None
                if os.path.isfile(path):
                    try:
                        with open(path, "rb") as handle:
                            cached_digest = sha256_bytes(handle.read())
                    except OSError:
                        cached_digest = None
                if refresh or cached_digest != digest:
                    temporary = path + ".part"
                    with open(temporary, "wb") as handle:
                        handle.write(data)
                    os.replace(temporary, path)
                    if cached_digest is not None:
                        print("[arkennemasis] library: %s/%s reference changed at "
                              "source — cache replaced" % (axis, value))

                record["refs"].append({
                    "role": role,
                    "source": url,
                    "path": path,
                    "hash": digest,
                    "bytes": len(data),
                })
                resolved_count += 1
                try:
                    previews.append(_load_image_array(path))
                except Exception:
                    pass
                lines.append("  ok    %-14s %-28s %-10s %s  %s"
                             % (axis, value, role, digest[7:19], os.path.basename(path)))

            if record["hex"]:
                swatch_path = os.path.join(swatch_root, "%s__%s.png" % (axis, value))
                array = swatch_image(record["hex"], int(swatch_size))
                if refresh or not os.path.isfile(swatch_path):
                    from PIL import Image
                    Image.fromarray((array * 255).astype(np.uint8)).save(swatch_path)
                record["swatch"] = swatch_path
                previews.append(array)
                lines.append("  ok    %-14s %-28s %-10s %s"
                             % (axis, value, "hex", record["hex"]))
                resolved_count += 1

            if not record["refs"] and not record["hex"] \
                    and not entry.get("unchanged"):
                unresolved.append(
                    "%s / %s has neither a hex nor a reference — a bare word is not a "
                    "specification (spec §6.3)." % (axis, value))
                lines.append("  FAIL  %-14s %-28s no specification" % (axis, value))

            library["values"]["%s/%s" % (axis, value)] = record

        index_path = os.path.join(root, "library.json")
        with open(index_path, "w", encoding="utf-8") as handle:
            json.dump(library, handle, indent=2, ensure_ascii=False)

        header = [
            "REFERENCE LIBRARY — %s" % product,
            "  root      : %s" % root,
            "  values    : %d" % len(library["values"]),
            "  artefacts : %d resolved" % resolved_count,
            "  index     : %s" % index_path,
            "",
            "RESOLUTION LOG",
        ]
        if unresolved:
            header_tail = ["", "UNRESOLVED (%d)" % len(unresolved)]
            header_tail += ["  - " + u for u in unresolved]
        else:
            header_tail = ["", "Every specification resolved."]
        report = "\n".join(header + lines + header_tail)

        print("[arkennemasis] library: %d values, %d artefacts, %d unresolved"
              % (len(library["values"]), resolved_count, len(unresolved)))

        if unresolved and strict:
            raise ValidationError(
                "%d specification(s) could not be resolved:\n%s"
                % (len(unresolved), "\n".join("  - " + u for u in unresolved)))

        return (json.dumps(library, ensure_ascii=False), _to_batch(previews),
                report, "\n".join(unresolved), len(library["values"]))


NODE_CLASS_MAPPINGS = {
    "ArkSpecLibrary": ArkSpecLibrary,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArkSpecLibrary": "arkennemasis Spec Library (resolve + cache references)",
}

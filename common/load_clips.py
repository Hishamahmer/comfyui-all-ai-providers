"""Load a run's finished clips back off disk, as the list the assembler expects.

`ArkVideoAssemble` normally takes its clips straight from `ArkHailuoScene`: list
execution renders one clip per scene and hands the whole collection over in a single
call. That works only while the render and the assembly happen in the same run.

They often do not. A long film is rendered over hours and can stop half way — a power
cut, a crash, a cancelled queue — and once the clips are on disk there is no way to hand
them back to the assembler, because a ComfyUI input takes exactly one link and the
assembler wants many videos. Re-rendering hours of finished footage just to join it is
absurd.

So this reads them back: point it at the run folder and it yields one file-backed VIDEO
per clip, in scene order, as an `OUTPUT_IS_LIST` output that drops straight into
`ArkVideoAssemble.videos`.

File-backed on purpose. `VideoFromFile` holds a path rather than decoded frames, so
sixteen clips cost sixteen paths instead of every frame of the film in RAM.
"""

from __future__ import annotations

import os
import re

DEFAULT_PATTERN = r"scene_(\d+)_\.mp4"


def _natural_key(name):
    """Sort scene_00002_ before scene_00010_, whatever the padding."""
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", name)]


class ArkLoadClips:
    CATEGORY = "arkennemasis/Video"
    FUNCTION = "run"
    RETURN_TYPES = ("VIDEO", "STRING")
    RETURN_NAMES = ("videos", "report")
    # Only the videos are a list. `report` is a plain value: merge_result_data wraps a
    # non-list slot itself, and returning a list there would nest it.
    OUTPUT_IS_LIST = (True, False)
    DESCRIPTION = ("Read a run's finished clips back off disk as a VIDEO list, ready for "
                   "Video Assemble. Use it to join a film whose render was interrupted, "
                   "without re-rendering anything.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder": ("STRING", {
                    "default": "", "tooltip":
                        "The run folder, e.g. ComfyUI/output/nemasis/story_001. "
                        "Absolute, or relative to the ComfyUI output directory.",
                }),
                "pattern": ("STRING", {
                    "default": DEFAULT_PATTERN, "tooltip":
                        "Regex a filename must match. The first capture group, if there "
                        "is one, is the scene number and decides the order.",
                }),
                "expected": ("INT", {
                    "default": 0, "min": 0, "max": 999, "tooltip":
                        "How many clips there should be. 0 = accept whatever is there. "
                        "Set it and a short run fails HERE, with a message naming the "
                        "count, instead of quietly assembling an incomplete film.",
                }),
            },
        }

    @classmethod
    def IS_CHANGED(cls, folder, pattern, expected, **_):
        # The folder's contents are the real input. Without this a second assembly after
        # more clips landed would be served the first call's result from cache.
        try:
            base = _resolve(folder)
            stamp = sorted((n, os.path.getmtime(os.path.join(base, n)))
                           for n in os.listdir(base))
            return str(stamp)
        except OSError:
            return float("nan")

    def run(self, folder, pattern=DEFAULT_PATTERN, expected=0):
        # Same import ArkHailuoScene uses, so both produce the identical VIDEO type.
        from comfy_api.latest._input_impl.video_types import VideoFromFile

        base = _resolve(folder)
        if not os.path.isdir(base):
            raise RuntimeError(f"No such folder: {base}")

        try:
            matcher = re.compile(pattern)
        except re.error as exc:
            raise RuntimeError(f"Bad pattern {pattern!r}: {exc}")

        found = []
        for name in os.listdir(base):
            match = matcher.fullmatch(name)
            if not match:
                continue
            path = os.path.join(base, name)
            if os.path.getsize(path) == 0:          # a clip cut off mid-write
                print(f"[arkennemasis] load clips: skipping empty {name}")
                continue
            order = int(match.group(1)) if match.groups() else None
            found.append((order, name, path))

        if not found:
            raise RuntimeError(
                f"No clips matching {pattern!r} in {base}. "
                "Check the folder, or the pattern if your files are named differently.")

        # Scene number when the pattern provides one, natural filename order otherwise.
        if all(order is not None for order, _n, _p in found):
            found.sort(key=lambda item: item[0])
        else:
            found.sort(key=lambda item: _natural_key(item[1]))

        if expected and len(found) != expected:
            raise RuntimeError(
                f"Expected {expected} clips, found {len(found)} in {base}: "
                f"{[n for _o, n, _p in found]}. Render the missing scenes first — "
                "assembling now would silently produce a short film.")

        videos = [VideoFromFile(path) for _order, _name, path in found]
        names = [name for _order, name, _path in found]
        report = "%d clips from %s: %s" % (len(videos), base, ", ".join(names))
        print(f"[arkennemasis] load clips: {report}")
        return (videos, report)


def _resolve(folder):
    """Absolute path, or one relative to the ComfyUI output directory."""
    folder = (folder or "").strip().strip('"')
    if os.path.isabs(folder):
        return folder
    try:
        import folder_paths
        return os.path.join(folder_paths.get_output_directory(), folder)
    except Exception:
        return os.path.abspath(folder)


NODE_CLASS_MAPPINGS = {
    "ArkLoadClips": ArkLoadClips,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArkLoadClips": "arkennemasis Load Clips (finished clips from disk)",
}

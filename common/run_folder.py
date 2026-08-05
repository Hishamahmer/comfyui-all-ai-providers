"""Give every run its own numbered output folder.

You type a folder name; each queued run resolves to the next free
``<folder_name>_<NNN>`` inside ``parent_dir``, so nothing is ever overwritten and runs
stay comparable.

Why a node instead of a filename token: save nodes that support time tokens evaluate
them *at save time*. When a graph writes 24 images over an hour, a
``[time(%H-%M)]`` folder scatters them across several folders. This resolves the path
**once per run** and every save node reads the same string off one link.

``IS_CHANGED`` returns NaN so the path is recomputed on every queued run rather than
served from ComfyUI's cache — otherwise a second run would reuse the first run's folder.
Only the save nodes downstream re-execute; upstream generators stay cached, so re-running
does not re-bill a paid API node.
"""

import os
import re


def next_run_folder(parent_dir, folder_name, padding=3, start=1):
    """Absolute path of the next free ``<folder_name>_<NNN>`` inside ``parent_dir``."""
    parent = os.path.abspath(parent_dir)
    name = (folder_name or "run").strip().strip("/\\") or "run"
    highest = start - 1
    if os.path.isdir(parent):
        pattern = re.compile(r"^%s_(\d+)$" % re.escape(name))
        for entry in os.listdir(parent):
            if os.path.isdir(os.path.join(parent, entry)):
                m = pattern.match(entry)
                if m:
                    highest = max(highest, int(m.group(1)))
    number = highest + 1
    return os.path.join(parent, "%s_%0*d" % (name, padding, number)), number


class ArkRunFolder:
    CATEGORY = "arkennemasis/Utility"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING", "INT", "STRING")
    RETURN_NAMES = ("folder_path", "run_number", "save_prefix")
    DESCRIPTION = ("Resolve a fresh numbered output folder for this run: "
                   "<parent_dir>/<folder_name>_001, _002, ... Wire folder_path into every "
                   "save node so all of them write into the same run folder.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "parent_dir": ("STRING", {
                    "default": "",
                    "tooltip": "Where the run folders live. Relative paths resolve inside "
                               "ComfyUI's output directory. Blank = the output directory.",
                }),
                "folder_name": ("STRING", {
                    "default": "dataset",
                    "tooltip": "Base name you choose. Each run appends the next free "
                               "number, so the same name never overwrites a previous run.",
                }),
                "padding": ("INT", {
                    "default": 3, "min": 1, "max": 9,
                    "tooltip": "Digits in the run number: 3 -> _001.",
                }),
                "create_now": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Create the folder immediately. Off = only return the path "
                               "and let the save node create it.",
                }),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")          # never served from cache -> a new folder each run

    def run(self, parent_dir, folder_name, padding=3, create_now=True):
        parent = parent_dir.strip()
        if not parent or not os.path.isabs(parent):
            try:
                import folder_paths
                base = folder_paths.get_output_directory()
            except Exception:
                base = os.getcwd()
            parent = os.path.join(base, parent) if parent else base

        path, number = next_run_folder(parent, folder_name, padding)
        if create_now:
            os.makedirs(path, exist_ok=True)

        # `save_prefix` is the same folder expressed the way SaveImage/SaveVideo want it:
        # relative to ComfyUI's output dir, forward slashes. Those nodes build their own
        # path from filename_prefix and mangle an absolute one, so wiring folder_path
        # into them does not work — this output is what they take.
        # (Appending an OUTPUT is safe for saved workflows; appending a WIDGET is not.)
        prefix = ""
        try:
            import folder_paths
            rel = os.path.relpath(path, folder_paths.get_output_directory())
            if not rel.startswith(".."):
                prefix = rel.replace("\\", "/")
        except Exception:
            pass

        print("[arkennemasis] run folder: %s" % path)
        return (path, number, prefix)


NODE_CLASS_MAPPINGS = {
    "ArkRunFolder": ArkRunFolder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArkRunFolder": "arkennemasis Run Folder (auto-numbered)",
}

"""Write a sidecar text file next to a saved image.

Training toolkits (kohya, ai-toolkit, diffusers) pair an image with a caption by exact
basename: ``shot_001.png`` + ``shot_001.txt``. Give this node the same ``folder_path`` and
``filename`` the image save uses and the pair always matches.

Wire the generating image into ``images`` so the caption is bound to the same branch: if
that branch is skipped (a gate upstream returning ``ExecutionBlocker``), the caption is
skipped too and you never get an orphan ``.txt`` describing an image that was not made.

A note on captions for character LoRAs: **anything you caption is excluded from what the
LoRA learns.** Caption the variable parts - pose, expression, wardrobe, background - and
never the face, hair colour or eye colour, or those become separable instead of learned.
That is why a VLM description of the finished image is usually the wrong source here.
"""

import os


class ArkTextFileSave:
    CATEGORY = "arkennemasis/Utility"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("path",)
    OUTPUT_NODE = True
    DESCRIPTION = ("Write text to <folder_path>/<filename>.<extension> - a caption sidecar "
                   "matching an image saved with the same folder and filename.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "The caption. Trailing whitespace is stripped.",
                }),
                "folder_path": ("STRING", {
                    "default": "",
                    "tooltip": "Same folder the image save writes to. Relative paths "
                               "resolve inside ComfyUI's output directory.",
                }),
                "filename": ("STRING", {
                    "default": "",
                    "tooltip": "Same filename stem as the image, WITHOUT an extension.",
                }),
            },
            "optional": {
                "extension": ("STRING", {
                    "default": "txt",
                    "tooltip": "Sidecar extension, no dot.",
                }),
                "images": ("IMAGE", {
                    "tooltip": "Optional, but wire the generated image in: it binds this "
                               "caption to that branch, so a skipped branch writes no "
                               "orphan caption.",
                }),
                "overwrite": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Off = refuse to replace an existing file.",
                }),
            },
        }

    def run(self, text, folder_path, filename, extension="txt", images=None,
            overwrite=True):
        stem = os.path.basename(str(filename).strip())
        if not stem:
            raise ValueError("ArkTextFileSave: filename is empty.")
        ext = str(extension).strip().lstrip(".") or "txt"

        folder = str(folder_path).strip()
        if not folder or not os.path.isabs(folder):
            try:
                import folder_paths
                base = folder_paths.get_output_directory()
            except Exception:
                base = os.getcwd()
            folder = os.path.join(base, folder) if folder else base
        os.makedirs(folder, exist_ok=True)

        path = os.path.join(folder, "%s.%s" % (stem, ext))
        if os.path.exists(path) and not overwrite:
            raise FileExistsError(
                "ArkTextFileSave: %s exists and overwrite is off." % path)

        # atomic: write beside it, then replace, so a crash never leaves a half caption
        tmp = path + ".part"
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            f.write(str(text).strip() + "\n")
        os.replace(tmp, path)

        print("[arkennemasis] caption -> %s" % path)
        return (path,)


NODE_CLASS_MAPPINGS = {
    "ArkTextFileSave": ArkTextFileSave,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArkTextFileSave": "arkennemasis Text File Save (caption sidecar)",
}

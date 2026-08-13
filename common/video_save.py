"""Write a VIDEO to disk, free the models behind it, and hand back the path.

This exists for video models that arrive as a **subgraph** rather than as a node of ours.
`ArkHailuoScene` does its own saving and its own releasing, which is what lets a run of
twenty shots finish; a stock subgraph does neither, and in a scene loop that is fatal.

Two things go wrong without it:

*Memory.* A scene loop is driven by ``ArkSceneList``, which declares ``OUTPUT_IS_LIST``,
so ComfyUI runs each node across EVERY scene before moving to the next node. Nothing
gets a chance to run between shots except the nodes already in the chain. A 22B
transformer plus a 12B text encoder plus two VAEs stays resident across all of them, and
the run dies partway through staging — with the GPU nearly empty, because it is host RAM
that ran out. Releasing here, immediately after the clip is safely on disk, is the only
place in a subgraph-based chain where it can happen at all.

*The path.* ``ArkVideoDub`` wants the clip's file, not just the VIDEO object; given only
the object it falls back to ``save_to()`` into a temporary file, which costs an extra
encode of every shot and leaves nothing on disk to recover an interrupted run from.
Saving once, here, gives the dub a real path and leaves a numbered clip behind.
"""

from __future__ import annotations

import os


def _free_ram_gb():
    """Host RAM free, in GB. psutil is present in this install; never fail without it."""
    try:
        import psutil
        return psutil.virtual_memory().available / 1e9
    except Exception:
        return float("nan")


def release_everything(when):
    """Evict models and the pinned host buffers behind them, and say what it recovered.

    `unload_all_models` is the call that actually evicts — without it ComfyUI reports
    "0 models unloaded" and keeps the staged weights in host RAM, which is the side that
    actually runs out. Every call here is optional across ComfyUI versions, so none of
    them may fail hard.
    """
    import gc

    import comfy.model_management as mm

    before_vram = mm.get_free_memory(mm.get_torch_device()) / 1e9
    before_ram = _free_ram_gb()

    gc.collect()
    try:
        mm.unload_all_models()
    except Exception as exc:
        print("[arkennemasis] unload_all_models failed (%s)" % str(exc)[:80])
    for name in ("cleanup_models_gc", "cleanup_models", "reset_cast_buffers"):
        fn = getattr(mm, name, None)
        if callable(fn):
            try:
                fn()
            except Exception as exc:
                print("[arkennemasis] %s skipped (%s)" % (name, str(exc)[:70]))
    try:
        import comfy_aimdo.host_buffer as hb
        cleanup = getattr(hb, "cleanup_file_reader", None)
        if callable(cleanup):
            cleanup()
    except Exception:
        pass

    try:
        mm.soft_empty_cache(force=True)
    except TypeError:
        mm.soft_empty_cache()
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    after_vram = mm.get_free_memory(mm.get_torch_device()) / 1e9
    after_ram = _free_ram_gb()
    print("[arkennemasis] release (%s): VRAM %.1f -> %.1f GB | RAM %.1f -> %.1f GB"
          % (when, before_vram, after_vram, before_ram, after_ram))


class ArkVideoSave:
    CATEGORY = "arkennemasis/Video"
    FUNCTION = "run"
    RETURN_TYPES = ("VIDEO", "STRING")
    RETURN_NAMES = ("video", "path")
    DESCRIPTION = ("Save a clip under a numbered prefix and free the models behind it. "
                   "Put this straight after a video subgraph in a scene loop — the "
                   "subgraph cannot release memory between shots, and the dub "
                   "downstream wants the clip's path.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO", {"tooltip": "The clip to write."}),
                "filename_prefix": ("STRING", {
                    "default": "video/scene", "forceInput": True,
                    "tooltip": "Relative to ComfyUI's output dir. Each call appends the "
                               "next counter, so clips land in scene order.",
                }),
            },
            "optional": {
                "free_memory": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Evict every loaded model after writing. Leave ON in a "
                               "scene loop — it is what lets twenty shots finish. Turn "
                               "it off only for a single clip, where reloading the "
                               "model next run costs more than it saves.",
                }),
            },
        }

    def run(self, video, filename_prefix, free_memory=True):
        import folder_paths
        from comfy_api.latest._input_impl.video_types import VideoFromFile

        # get_save_image_path ends in os.makedirs(..., exist_ok=True), so a prefix
        # naming folders that do not exist yet creates them. That matters because this
        # workflow is shared: nobody should have to pre-make the output tree by hand.
        width, height = 0, 0
        try:
            size = video.get_dimensions()
            if size:
                width, height = int(size[0]), int(size[1])
        except Exception:
            pass
        folder, name, counter, _subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory(), width, height)
        path = os.path.join(folder, "%s_%05d_.mp4" % (name, counter))

        video.save_to(path)
        size_mb = os.path.getsize(path) / 1e6 if os.path.exists(path) else 0.0
        print("[arkennemasis] clip saved -> %s (%.1f MB)"
              % (os.path.basename(path), size_mb))

        if free_memory:
            release_everything("after saving")

        # A file-backed VIDEO holds a path, not frames, so the assembler downstream can
        # collect every scene of a run without keeping any of them in memory.
        return (VideoFromFile(path), path)


NODE_CLASS_MAPPINGS = {
    "ArkVideoSave": ArkVideoSave,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArkVideoSave": "arkennemasis Video Save (write a clip, free the models)",
}

"""Purge VRAM — unload every model, then hand the data straight on.

WHY THIS EXISTS
    A pipeline that draws stills with one big model and then animates them with another
    has to put the first one down before it picks the second one up. On this machine
    Flux.2-dev is 64 GB of weights plus a 36 GB text encoder, and MiniMax H3 is another
    61 GB; nothing like both fits in 48 GB of VRAM, and ComfyUI's dynamic staging only
    saves you if the host has the RAM to stage into. Without a purge in between, the
    video half starts by fighting the image half for memory and loses.

WHERE TO PUT IT
    Between the still and the video: wire the image generator's IMAGE into `images` and
    take `images` back out into whatever comes next. That data dependency is the whole
    trick — a node with no link has nothing to make ComfyUI run it at the right moment,
    or at all.

    It lands in the right place automatically for a list run. ComfyUI's executor is
    stage-wise: a node finishes EVERY item before the next node starts, so one purge
    node between the two stages runs after the last still and before the first frame of
    video. It will run once per item; the passes after the first are nearly free.

    `anything` is there for the same job when what you have to pass through is not an
    image — it takes any type and returns it untouched.
"""

import gc


def _free_gb():
    try:
        import torch
        if torch.cuda.is_available():
            free, _total = torch.cuda.mem_get_info()
            return free / 1e9
    except Exception:
        pass
    return None


class AnyType(str):
    """Matches any socket type. ComfyUI compares types with `!=`, so never be unequal."""

    def __ne__(self, other):
        return False


ANY = AnyType("*")


class ArkPurgeVRAM:
    CATEGORY = "arkennemasis/Utility"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", ANY, "STRING")
    RETURN_NAMES = ("images", "anything", "report")
    DESCRIPTION = ("Unload every model and empty the CUDA cache, then pass the input "
                   "straight through. Put it between two stages that use different "
                   "big models so the second one starts with the memory free.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "unload_models": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Evict every loaded model from VRAM. This is the one "
                               "that actually frees the gigabytes.",
                }),
                "empty_cache": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Return PyTorch's cached blocks to the driver. Without "
                               "this the memory stays reserved and the next model still "
                               "cannot see it.",
                }),
                "collect_garbage": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Run Python's collector first, so tensors waiting on a "
                               "reference cycle are actually released.",
                }),
            },
            "optional": {
                "images": ("IMAGE", {
                    "tooltip": "Wire the image generator in here and take `images` back "
                               "out into the next stage. That link is what makes "
                               "ComfyUI run the purge at this point in the graph.",
                }),
                "anything": (ANY, {
                    "tooltip": "Same idea for anything that is not an image — any type "
                               "in, the same value out.",
                }),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")          # a purge must never be served from cache

    def run(self, unload_models, empty_cache, collect_garbage, images=None,
            anything=None):
        before = _free_gb()

        if collect_garbage:
            gc.collect()
        if unload_models:
            try:
                import comfy.model_management as mm
                mm.unload_all_models()
            except Exception as exc:
                print("[arkennemasis] purge: unload_all_models failed (%s)" % exc)
        if empty_cache:
            try:
                import comfy.model_management as mm
                mm.soft_empty_cache(force=True)
            except Exception:
                pass
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            except Exception:
                pass

        after = _free_gb()
        if before is None or after is None:
            report = "purged (no CUDA device to measure)"
        else:
            report = ("VRAM free %.1f -> %.1f GB (+%.1f)"
                      % (before, after, after - before))
        print("[arkennemasis] purge: %s" % report)
        return (images, anything, report)


NODE_CLASS_MAPPINGS = {
    "ArkPurgeVRAM": ArkPurgeVRAM,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArkPurgeVRAM": "arkennemasis Purge VRAM (between stages)",
}

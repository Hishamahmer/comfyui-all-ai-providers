"""One settings node driving many Image Gen nodes.

ComfyUI rejects a STRING link into a COMBO widget ("received_type(STRING) mismatch
input_type(COMBO)"), so a shared aspect_ratio / quality / ... cannot be wired straight
into the widgets. This node bundles them into a single typed value instead: wire its
one output into the ``settings`` socket of every Image Gen node and change them all in
one place.

Anything left on ``use node's own`` falls through to that node's own widget, so you can
share most settings and still override one shot locally.

``number_of_images`` is deliberately **not** shared. Multiplying it across every wired node
is rarely intended, and in a graph that names files deterministically the extra images all
land on the same filename and overwrite each other. Set it per node if you really want it.
"""

INHERIT = "use node's own"

ASPECT = [INHERIT, "default", "1:1", "3:2", "2:3", "4:3", "3:4", "16:9", "9:16", "auto", "1024x1024", "1536x1024", "1024x1536", "1536x1152", "1152x1536", "2048x2048", "2048x1152", "1152x2048", "3840x2160", "2160x3840"]
QUALITY = [INHERIT, "default", "low", "medium", "high", "auto"]
BACKGROUND = [INHERIT, "default", "auto", "transparent", "opaque"]
FORMAT = [INHERIT, "default", "png", "jpeg", "webp"]
MODERATION = [INHERIT, "default", "auto", "low"]
RUN_MODE = [INHERIT, "one at a time", "all at once"]

SETTINGS_TYPE = "ARK_IMAGE_SETTINGS"


class ArkImageGenSettings:
    CATEGORY = "arkennemasis/Image Gen"
    FUNCTION = "run"
    RETURN_TYPES = (SETTINGS_TYPE,)
    RETURN_NAMES = ("settings",)
    DESCRIPTION = ("Shared settings for arkennemasis Image Gen nodes. Wire the output "
                   "into every node's `settings` socket to control them all from here. "
                   "Leave a field on \"use node's own\" to let that node keep its own "
                   "widget value.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "aspect_ratio": (ASPECT,),
                "quality": (QUALITY,),
                "run_mode": (RUN_MODE,),
                "background": (BACKGROUND,),
                "output_format": (FORMAT,),
                "moderation": (MODERATION,),
                "timeout_seconds": ("INT", {
                    "default": -1, "min": -1, "max": 86400,
                    "tooltip": "-1 = use each node's own value. 0 = wait indefinitely.",
                }),
                "api_token": ("STRING", {
                    "default": "",
                    "tooltip": "Blank = each node's own field, then REPLICATE_API_TOKEN "
                               "from the environment or a .env file.",
                }),
            },
        }

    def run(self, aspect_ratio, quality, run_mode, background,
            output_format, moderation, timeout_seconds, api_token):
        out = {}
        for key, val in (("aspect_ratio", aspect_ratio), ("quality", quality),
                         ("run_mode", run_mode), ("background", background),
                         ("output_format", output_format), ("moderation", moderation)):
            if val != INHERIT:
                out[key] = val
        if int(timeout_seconds) >= 0:
            out["timeout_seconds"] = int(timeout_seconds)
        if api_token.strip():
            out["api_token"] = api_token.strip()
        return (out,)


NODE_CLASS_MAPPINGS = {
    "ArkImageGenSettings": ArkImageGenSettings,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArkImageGenSettings": "arkennemasis Image Gen Settings (shared)",
}

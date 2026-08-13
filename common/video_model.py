"""Pick one video model and do not pay for the other.

Same problem as the image engine, one stage later. A canvas that offers two renderers —
MiniMax H3 and LTX-2.5 — cannot join both into a plain switch: ComfyUI evaluates the whole
graph an output depends on, so both would stage tens of gigabytes of weights and render
every shot twice before one result was thrown away.

The fix is laziness. Both branches are ``lazy`` inputs and ``check_lazy_status`` asks for
only the chosen one, so the branch you did not pick is never evaluated — no model load, no
render, no time. That is also why neither renderer needs muting: the dropdown IS the
switch, and a muted node would only duplicate what laziness already does.

The clip's PATH travels with it. ``ArkVideoDub`` wants the file rather than the VIDEO
object — given only the object it re-encodes through ``save_to()`` — and each renderer
knows its own path: ``ArkHailuoScene`` returns one directly, LTX gets one from
``ArkVideoSave``. Carrying both through this node keeps a single dub downstream instead of
one per renderer.
"""

from __future__ import annotations

try:                                             # ComfyUI >= the lazy/blocker rework
    from comfy_execution.graph_utils import ExecutionBlocker
except ImportError:                              # older layout
    from execution import ExecutionBlocker

H3 = "MiniMax H3 — local, omni-modal (its own audio)"
LTX = "LTX-2.5 — local, 22B distilled"
MODELS = [H3, LTX]

# Short, filesystem-safe tag per model, so the two films do not overwrite each other.
TAGS = {H3: "h3", LTX: "ltx"}


class ArkVideoModel:
    CATEGORY = "arkennemasis/Video"
    FUNCTION = "run"
    RETURN_TYPES = ("VIDEO", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("video", "path", "model", "film_name")
    DESCRIPTION = (
        "Choose which video model actually renders. The branch you did not pick is never "
        "evaluated, so it costs nothing — no weights staged and no shots rendered."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (MODELS, {
                    "tooltip": "Only this renderer runs. The other is skipped entirely, "
                               "so switching models costs nothing on the unused side.",
                }),
                "basename": ("STRING", {
                    "default": "walkthrough",
                    "tooltip": "The finished film's name. The chosen model's tag is "
                               "appended, so switching renderer does not overwrite the "
                               "film you already made.",
                }),
            },
            "optional": {
                "h3_video": ("VIDEO", {"lazy": True,
                                       "tooltip": "Wire ArkHailuoScene's video here."}),
                "h3_path": ("STRING", {"lazy": True, "forceInput": True,
                                       "tooltip": "ArkHailuoScene's path output."}),
                "ltx_video": ("VIDEO", {"lazy": True,
                                        "tooltip": "Wire the LTX branch's video here."}),
                "ltx_path": ("STRING", {"lazy": True, "forceInput": True,
                                        "tooltip": "ArkVideoSave's path output."}),
            },
        }

    @staticmethod
    def _wanted(model):
        return ["h3_video", "h3_path"] if model == H3 else ["ltx_video", "ltx_path"]

    def check_lazy_status(self, model, basename="walkthrough", **kwargs):
        # Naming only the chosen branch's inputs leaves the other unevaluated. Both of a
        # branch's inputs come from the same upstream node, so asking for the path costs
        # nothing beyond the video that was rendered anyway.
        return self._wanted(model)

    def run(self, model, basename="walkthrough", h3_video=None, h3_path=None,
            ltx_video=None, ltx_path=None):
        have = {"h3_video": h3_video, "h3_path": h3_path,
                "ltx_video": ltx_video, "ltx_path": ltx_path}
        wanted = self._wanted(model)
        video_key, path_key = wanted

        tag = TAGS.get(model, "video")
        film = "%s_%s" % (str(basename).strip() or "walkthrough", tag)

        if have[video_key] is None:
            # Block rather than raise: the message names the fix, and everything
            # downstream is skipped quietly instead of the run dying mid-way.
            blocked = ExecutionBlocker(
                "Video model is set to '%s' but nothing is wired into %s."
                % (model, video_key))
            return (blocked, blocked, model, film)

        path = have[path_key] or ""
        print("[arkennemasis] video model: %s -> %s" % (model, film))
        return (have[video_key], path, model, film)


NODE_CLASS_MAPPINGS = {
    "ArkVideoModel": ArkVideoModel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArkVideoModel": "arkennemasis Video Model (H3 or LTX — only one runs)",
}

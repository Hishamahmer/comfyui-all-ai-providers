"""Pull one scene out of a JSON array of scenes.

This is the node that lets a whole storyboard pipeline live in ONE ComfyUI workflow.

An LLM planning step returns a JSON array — one object per scene, each carrying the
prompts for that scene. A no-code tool would fan out with a split node feeding a
sub-workflow that runs once per item. ComfyUI has no loop and no per-item execution, so
the equivalent is N copies of the scene branch, each with one of these pulling out its
own index. Wire the SAME `scenes_json` into every copy and give each a different
`scene_index`; every branch then reads the slot it owns.

Out-of-range indices return ``ExecutionBlocker`` on every output rather than raising, so
a workflow built with 12 scene slots and fed a 6-scene plan simply runs 6 and leaves the
rest dark. Without that, asking for fewer scenes than the canvas has slots would fail the
whole prompt — which is exactly the case when you follow the advice to try 1-2 scenes
before committing to a long render.

Field names follow the storyboard schema
(``scene`` / ``voiceText`` / ``image_prompt`` / ``video_prompt``) but each is
configurable, so any array-of-objects plan works.
"""

import json

try:
    from comfy_execution.graph_utils import ExecutionBlocker
except ImportError:                                   # older ComfyUI
    from execution import ExecutionBlocker


class ArkSceneSplit:
    CATEGORY = "arkennemasis/Utility"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "INT", "INT")
    RETURN_NAMES = ("image_prompt", "video_prompt", "voice_text", "scene_number",
                    "scene_count")
    DESCRIPTION = ("Pull scene N out of a JSON array of scenes — the fan-out step that "
                   "lets one workflow drive many scene branches. Out-of-range indices "
                   "silently skip the branch instead of failing the run.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scenes_json": ("STRING", {
                    "multiline": True, "default": "[]", "forceInput": True,
                    "tooltip": "The whole JSON array. Wire the SAME source into every "
                               "scene branch.",
                }),
                "scene_index": ("INT", {
                    "default": 1, "min": 1, "max": 9999,
                    "tooltip": "Which scene this branch owns, 1-based. Unique per copy.",
                }),
            },
            "optional": {
                "image_prompt_key": ("STRING", {"default": "image_prompt"}),
                "video_prompt_key": ("STRING", {"default": "video_prompt"}),
                "voice_text_key": ("STRING", {"default": "voiceText"}),
                "scene_number_key": ("STRING", {"default": "scene"}),
                "strict": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "On = raise when this index is missing. Off (default) = "
                               "skip the branch quietly, so a short plan does not fail a "
                               "canvas built for more scenes.",
                }),
            },
        }

    def run(self, scenes_json, scene_index, image_prompt_key="image_prompt",
            video_prompt_key="video_prompt", voice_text_key="voiceText",
            scene_number_key="scene", strict=False):
        raw = str(scenes_json).strip()
        if not raw:
            scenes = []
        else:
            try:
                scenes = json.loads(raw)
            except ValueError as exc:
                raise ValueError("ArkSceneSplit: scenes_json is not valid JSON (%s). "
                                 "First 200 chars: %s" % (exc, raw[:200]))
        if isinstance(scenes, dict):          # tolerate {"scenes": [...]} wrappers
            for key in ("scenes", "output", "data", "items"):
                if isinstance(scenes.get(key), list):
                    scenes = scenes[key]
                    break
        if not isinstance(scenes, list):
            raise ValueError("ArkSceneSplit: expected a JSON array of scenes, got %s."
                             % type(scenes).__name__)

        count = len(scenes)
        # Prefer the scene's own declared number; fall back to array position, because a
        # model occasionally renumbers or omits the field.
        chosen = None
        for item in scenes:
            if isinstance(item, dict):
                try:
                    if int(item.get(scene_number_key)) == int(scene_index):
                        chosen = item
                        break
                except (TypeError, ValueError):
                    continue
        if chosen is None and 1 <= scene_index <= count:
            chosen = scenes[scene_index - 1]

        if not isinstance(chosen, dict):
            if strict:
                raise IndexError("ArkSceneSplit: scene %d not found (the plan has %d)."
                                 % (scene_index, count))
            print("[arkennemasis] scene %d not in the plan (%d scenes) — branch skipped"
                  % (scene_index, count))
            blocked = ExecutionBlocker(None)
            return (blocked, blocked, blocked, blocked, blocked)

        def text(key):
            value = chosen.get(key, "")
            return value if isinstance(value, str) else json.dumps(value,
                                                                   ensure_ascii=False)

        try:
            number = int(chosen.get(scene_number_key, scene_index))
        except (TypeError, ValueError):
            number = scene_index

        return (text(image_prompt_key), text(video_prompt_key), text(voice_text_key),
                number, count)


NODE_CLASS_MAPPINGS = {
    "ArkSceneSplit": ArkSceneSplit,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArkSceneSplit": "arkennemasis Scene Split (one scene from a JSON plan)",
}

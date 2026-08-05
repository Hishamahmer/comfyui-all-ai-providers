"""Fan a JSON scene plan out into a LIST, so one chain of nodes renders every scene.

This is the loop. ComfyUI has no loop construct, but its executor already iterates: a
node declaring ``OUTPUT_IS_LIST`` makes every downstream node run **once per item**,
and any scalar input alongside it is reused each time (``execution.py`` slices inputs
with ``v[i if len(v) > i else -1]``). One Codex node, one Hailuo chain and one save node
therefore cover N scenes without a single duplicated branch.

That is the same shape as the n8n original — Split Out, a sub-workflow per item, then
Aggregate — except the iteration is native instead of hand-copied. The scene count comes
from the plan at run time, so 5, 20 or 50 scenes all use the same canvas.

The counterpart is ``ArkVideoAssemble`` / ``ArkContactSheet``, which declare
``INPUT_IS_LIST`` and so receive the whole collection in one call to join it back up.

Per-scene duration: the plan may carry a ``seconds`` field. Whatever it says is floored
at ``min_seconds`` and snapped up to MiniMax H3's 17k+5 frame grid, so scenes can run
10s, 12s, 13s as the dialogue needs. If the field is missing the duration is derived
from the word count instead, at a natural speaking rate.
"""

import json

WORDS_PER_SECOND = 2.3          # unhurried narration; 2.0-2.6 is the usual band


def snap_length(seconds, fps=24):
    """Frames on MiniMax H3's 17k+5 grid. The node snaps internally too, but doing it
    here keeps the log, the canvas and the actual file quoting the same number."""
    length = max(5, int(round(seconds * fps)))
    return length + (5 - (length % 17)) % 17


class ArkSceneList:
    CATEGORY = "arkennemasis/Utility"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "INT", "INT", "STRING")
    RETURN_NAMES = ("image_prompt", "video_prompt", "voice_text", "scene_number",
                    "length", "scenes_json")
    # Everything except scenes_json fans out: downstream nodes run once per scene.
    OUTPUT_IS_LIST = (True, True, True, True, True, False)
    DESCRIPTION = ("Turn a JSON scene plan into per-scene LISTS so one chain of nodes "
                   "renders every scene. This is the loop — no duplicated branches, and "
                   "the scene count comes from the plan at run time.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scenes_json": ("STRING", {
                    "multiline": True, "default": "[]", "forceInput": True,
                    "tooltip": "The JSON array from the story agent.",
                }),
                "min_seconds": ("FLOAT", {
                    "default": 10.0, "min": 1.0, "max": 15.0, "step": 0.5,
                    "tooltip": "Floor for every clip. A scene asking for less is raised "
                               "to this; a scene asking for more keeps its own length.",
                }),
                "max_seconds": ("FLOAT", {
                    "default": 15.0, "min": 1.0, "max": 15.0, "step": 0.5,
                    "tooltip": "Ceiling. MiniMax H3 is trained to about 15s (362 "
                               "frames); beyond that quality falls off.",
                }),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 60.0}),
                "limit": ("INT", {
                    "default": 0, "min": 0, "max": 200,
                    "tooltip": "Render only the first N scenes. 0 = all of them. This "
                               "replaces the old per-branch gate — a shorter list simply "
                               "means fewer iterations, and nothing downstream is "
                               "blocked.",
                }),
            },
        }

    def run(self, scenes_json, min_seconds=10.0, max_seconds=15.0, fps=24.0, limit=0):
        raw = str(scenes_json).strip()
        try:
            scenes = json.loads(raw) if raw else []
        except ValueError as exc:
            raise ValueError("ArkSceneList: scenes_json is not valid JSON (%s). First "
                             "200 chars: %s" % (exc, raw[:200]))
        if isinstance(scenes, dict):          # tolerate {"scenes": [...]} wrappers
            for key in ("scenes", "output", "data", "items"):
                if isinstance(scenes.get(key), list):
                    scenes = scenes[key]
                    break
        if not isinstance(scenes, list) or not scenes:
            raise ValueError("ArkSceneList: expected a non-empty JSON array of scenes.")

        scenes = [s for s in scenes if isinstance(s, dict)]
        scenes.sort(key=lambda s: _as_int(s.get("scene"), 10 ** 6))
        if limit:
            scenes = scenes[:limit]

        images, videos, voices, numbers, lengths = [], [], [], [], []
        for position, scene in enumerate(scenes, start=1):
            voice = _as_text(scene.get("voiceText"))
            seconds = _as_float(scene.get("seconds"))
            if seconds is None:
                # No duration in the plan — infer it from how long the line takes to
                # say, rather than silently defaulting everything to the floor.
                seconds = len(voice.split()) / WORDS_PER_SECOND if voice else min_seconds
            seconds = min(max(seconds, min_seconds), max_seconds)

            images.append(_as_text(scene.get("image_prompt")))
            videos.append(_as_text(scene.get("video_prompt")))
            voices.append(voice)
            numbers.append(_as_int(scene.get("scene"), position))
            lengths.append(snap_length(seconds, fps))

        print("[arkennemasis] scene list: %d scenes, lengths %s frames (%s s)"
              % (len(scenes), lengths, [round(n / fps, 1) for n in lengths]))
        # The first five slots are OUTPUT_IS_LIST, so they must BE lists — the executor
        # does `value.extend(o[i])` on them. The sixth is NOT, so it must be a plain
        # string: `merge_result_data` wraps non-list slots itself with
        # `[o[i] for o in results]`, and returning a list here would nest it and hand
        # the assembler a list where a STRING belongs.
        return (images, videos, voices, numbers, lengths,
                json.dumps(scenes, ensure_ascii=False))


def _as_text(value):
    if isinstance(value, str):
        return value
    return "" if value is None else json.dumps(value, ensure_ascii=False)


def _as_int(value, fallback):
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


NODE_CLASS_MAPPINGS = {
    "ArkSceneList": ArkSceneList,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArkSceneList": "arkennemasis Scene List (the loop — one chain, N scenes)",
}

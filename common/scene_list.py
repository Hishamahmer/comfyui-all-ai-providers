"""Fan a JSON scene plan out into a LIST, so one chain of nodes renders every scene.

This is the loop. ComfyUI has no loop construct, but its executor already iterates: a
node declaring ``OUTPUT_IS_LIST`` makes every downstream node run **once per item**,
and any scalar input alongside it is reused each time (``execution.py`` slices inputs
with ``v[i if len(v) > i else -1]``). One Codex node, one Hailuo chain and one save node
therefore cover N scenes without a single duplicated branch.

Split the plan, run the chain once per item, then
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

# Measured on Qwen3-TTS output rather than assumed: 22 words -> 11.10 s, 21 -> 8.62 s,
# 17 -> 10.30 s, i.e. 1.65-2.44 and about 2.0 on average. The old 2.3 was optimistic and
# under-estimated every shot, which is how an 11.10 s line ended up over a 5.17 s clip.
# Only a fallback now — `ArkNarrationLength` measures the rendered speech instead.
WORDS_PER_SECOND = 2.0

# LOCKED, and must match ArkHailuoScene.FPS. Hailuo works in frames at 24 fps; this is
# what converts a scene's requested seconds into a frame count. Not a widget, because a
# value that must never change should not be a box that can be emptied.
FPS = 24.0


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
                "limit": ("INT", {
                    "default": 0, "min": 0, "max": 200,
                    "tooltip": "Render only the first N scenes. 0 = all of them. This "
                               "replaces the old per-branch gate — a shorter list simply "
                               "means fewer iterations, and nothing downstream is "
                               "blocked.",
                }),
            },
        }

    def run(self, scenes_json, min_seconds=10.0, max_seconds=15.0, limit=0):
        fps = FPS
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
            voice = _as_text(_pick(scene, "voiceText", "voice_text"))
            seconds = _as_float(scene.get("seconds"))
            if seconds is None:
                # No duration in the plan — infer it from how long the line takes to
                # say, rather than silently defaulting everything to the floor.
                seconds = len(voice.split()) / WORDS_PER_SECOND if voice else min_seconds
            seconds = min(max(seconds, min_seconds), max_seconds)

            images.append(_as_text(_pick(scene, "image_prompt", "imagePrompt")))
            videos.append(_as_text(_pick(scene, "video_prompt", "videoPrompt")))
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


def _pick(scene, *keys):
    """First key the plan actually carries.

    The pack spells this field ``voiceText`` while its neighbours are ``image_prompt``
    and ``video_prompt``, so a brief written in one consistent case is always wrong
    about half the keys. The model then obeys the brief, the key misses, and the shot
    gets an empty line — which surfaces much later as `ArkQwenTTS: \\`text\\` is empty`
    rather than as the naming mismatch it is. Accept both spellings instead.
    """
    for key in keys:
        value = scene.get(key)
        if value not in (None, ""):
            return value
    return None


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

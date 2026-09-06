"""One scene out of a shot plan, by index — the piece a real loop needs.

``ArkSceneList`` fans a plan out with ``OUTPUT_IS_LIST``, which makes ComfyUI run each
node across EVERY scene before moving to the next node. For a chain that only reads and
writes text that is fine. For one that renders video it is not: every shot is decoded and
held in memory before the first is written to disk, so memory grows with the number of
photos and a long enough folder cannot finish at all.

An explicit loop fixes that — shot rendered, shot saved, shot freed, next — but a loop
works in indexes, and a list output cannot be indexed. This node closes that gap: give it
the plan and an index and it returns that one scene's fields as plain values.

It clamps rather than raising. A loop driven by the photo count can ask for a scene the
model did not write (it returned fewer than there were photos), and stopping a
forty-minute render over an off-by-one is worse than rendering the last scene twice —
which is visible, and fixable in the plan.
"""

from __future__ import annotations

import json


def unfence(text):
    """Strip a markdown code fence a model wrapped its JSON in.

    Local models do this constantly no matter how firmly the brief says not to —
    ```json on the first line and ``` on the last. Cloud models mostly obey, which is
    why this went unnoticed until a local writer was wired in and every run died on
    `scenes_json is not valid JSON ... first 200 chars: ```json`. Asking more loudly in
    the prompt does not fix a thing the model does anyway; accepting what models
    actually emit does.
    """
    text = str(text or "").strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    lines = lines[1:]                       # drop the opening ``` or ```json
    while lines and not lines[-1].strip().startswith("```"):
        lines.pop()                         # trailing chatter after the closing fence
    if lines:
        lines.pop()                         # drop the closing ```
    return chr(10).join(lines).strip()


def parse_scenes(scenes_json):
    """The scene array, under any of the shapes a model tends to return."""
    if isinstance(scenes_json, list):
        return scenes_json
    text = unfence(scenes_json)
    if not text:
        return []
    try:
        data = json.loads(text)
    except ValueError:
        return []
    if isinstance(data, dict):
        data = data.get("scenes") or data.get("output") or data.get("shots") or []
    return data if isinstance(data, list) else []


def _text(scene, *names):
    """First non-empty field under any of `names`, camelCase or snake_case.

    Same tolerance as ArkSceneList and ArkVideoAssemble: a plan written consistently in
    snake_case still lands, instead of producing a silent empty string three nodes later.
    """
    if not isinstance(scene, dict):
        return ""
    for name in names:
        value = scene.get(name)
        if value:
            return str(value)
    return ""


class ArkSceneAt:
    CATEGORY = "arkennemasis/Video"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "INT", "INT")
    RETURN_NAMES = ("video_prompt", "voice_text", "image_prompt", "scene_number",
                    "total")
    DESCRIPTION = ("One scene of a shot plan, chosen by index, as plain values. Use it "
                   "inside a for-loop so each shot renders, saves and frees before the "
                   "next begins — which is what lets a folder of any size finish.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scenes_json": ("STRING", {
                    "multiline": True, "default": "[]", "forceInput": True,
                    "tooltip": "The JSON array from the story agent.",
                }),
                "index": ("INT", {
                    "default": 0, "min": 0, "max": 100000,
                    "tooltip": "0-based. Wire the loop's `index` here.",
                }),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")      # the index changes every iteration; never serve a cache

    def run(self, scenes_json, index):
        scenes = parse_scenes(scenes_json)
        total = len(scenes)
        if not scenes:
            print("[arkennemasis] scene at: the plan is empty or unparseable")
            return ("", "", "", 0, 0)

        i = max(0, min(int(index), total - 1))
        if i != int(index):
            print("[arkennemasis] scene at: asked for %d of %d — clamped to %d"
                  % (int(index), total, i))
        scene = scenes[i]

        video_prompt = _text(scene, "video_prompt", "videoPrompt", "camera", "prompt")
        voice = _text(scene, "voiceText", "voice_text", "narration", "voiceover")
        image_prompt = _text(scene, "image_prompt", "imagePrompt")
        number = scene.get("scene") or scene.get("scene_number") or (i + 1)
        try:
            number = int(number)
        except (TypeError, ValueError):
            number = i + 1

        print("[arkennemasis] scene %d/%d: %s" % (i + 1, total, (voice or "")[:60]))
        return (video_prompt, voice, image_prompt, number, total)


NODE_CLASS_MAPPINGS = {
    "ArkSceneAt": ArkSceneAt,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArkSceneAt": "arkennemasis Scene At (one scene of the plan, by index)",
}

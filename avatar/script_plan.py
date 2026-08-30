"""Take the script-writing model's answer apart into the things downstream needs.

One LLM call produces the spoken script, the social caption and a headline. Three
consumers need three different pieces of it — the voice needs the script and nothing else,
the subtitle burner needs it again in the shape a scene plan has, the save path needs a
filename — so the answer is parsed once, here, and handed out as separate sockets.

Doing it in one call rather than the original's two is not a shortcut. The caption has to
be about the same story the script tells, and a second independent call is a second chance
for the model to pick a different angle.

**Invalid JSON is an error, not a guess.** `ArkCodexLLM`'s `json_only` already parses and
re-raises inside the node, so anything arriving here malformed means something further
upstream changed. Guessing a script out of prose that was meant to be JSON produces a
video that sounds almost right, which is worse than a run that stops.

`scenes_json` is emitted in the shape `ArkVideoAssemble` reads — a one-scene plan carrying
`voiceText`. That node times each cue from the clip's REAL duration, so the plan only has
to supply the words; it does not need to know how long anything took.
"""

from __future__ import annotations

import json
import re

# The keys the model is asked for. `script` is the only one without which nothing can run.
REQUIRED = ("script",)
OPTIONAL = ("caption", "headline", "hashtags", "source_url")

_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def _slugify(text, limit=48):
    """A filename-safe stem.

    Apostrophes are DELETED rather than separated on, so "Apple's launch" gives
    `apples-launch` and not `apple-s-launch` — the same rule the hairstyle and thumbnail
    catalogs needed for their possessive rail groups.
    """
    text = (text or "").strip().lower()
    text = text.replace("'", "").replace("’", "")
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return (text[:limit].strip("-") or "story")


def parse(answer):
    """The model's answer -> a dict. Raises with a readable reason when it cannot."""
    raw = _FENCE.sub("", (answer or "").strip())
    if not raw:
        raise RuntimeError(
            "ArkAvatarScript got an empty answer. The script-writing model returned "
            "nothing — check its node for an error, and that `json_only` is on.")
    try:
        data = json.loads(raw)
    except Exception as exc:
        raise RuntimeError(
            "ArkAvatarScript could not read the model's answer as JSON (%s).\n"
            "It must be one object with a `script` key. First 300 characters:\n%s"
            % (exc, raw[:300]))
    if isinstance(data, list) and data:
        data = data[0]
    if not isinstance(data, dict):
        raise RuntimeError("ArkAvatarScript expected a JSON object, got %s."
                           % type(data).__name__)
    missing = [k for k in REQUIRED if not str(data.get(k) or "").strip()]
    if missing:
        raise RuntimeError(
            "ArkAvatarScript: the model's answer has no %s. It returned keys: %s"
            % (", ".join(missing), ", ".join(sorted(data)) or "(none)"))
    return data


def _caption(data):
    """Caption plus hashtags, however the model chose to hand them over."""
    caption = str(data.get("caption") or "").strip()
    tags = data.get("hashtags")
    if isinstance(tags, str):
        tags = tags.split()
    if isinstance(tags, (list, tuple)):
        marked = ["#" + str(t).lstrip("#").strip() for t in tags if str(t).strip()]
        # Only append tags the caption is not already carrying — models routinely put
        # them in both places, and a caption that repeats its own hashtags looks automated.
        fresh = [t for t in marked if t.lower() not in caption.lower()]
        if fresh:
            caption = (caption + "\n\n" + " ".join(fresh)).strip()
    return caption


def _caption_from(answer):
    """The caption, however its own call chose to answer.

    It is asked for PLAIN TEXT — a caption has one consumer and is the whole answer, so
    there is nothing for JSON to disambiguate. But a model told to return prose still
    sometimes returns `{"caption": ...}`, and a caption that arrives as a stringified JSON
    object is a visible defect in the post. So both are accepted, and neither is an error:
    JSON if it parses to an object, otherwise the text exactly as written.

    Deliberately NOT `parse()` — that one requires a `script` key, which a caption answer
    correctly does not have. Reusing it made every separate caption call fall back
    silently to the script call's caption.
    """
    raw = _FENCE.sub("", (answer or "").strip())
    if not raw:
        return ""
    try:
        data = json.loads(raw)
    except Exception:
        return raw
    if isinstance(data, list) and data:
        data = data[0]
    return _caption(data) if isinstance(data, dict) else raw


class ArkAvatarScript:
    CATEGORY = "arkennemasis/Avatar"
    FUNCTION = "run"
    # `slug` is the bare stem and `save_prefix` is the stem inside its folder. Both,
    # because they go to different kinds of consumer: a save node builds its own path and
    # wants the prefix, while a node that writes one named file wants a filename with no
    # separator in it — a slash there silently asks for a subdirectory.
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "INT",
                    "STRING")
    RETURN_NAMES = ("script", "caption", "headline", "slug", "save_prefix",
                    "scenes_json", "word_count", "report")
    DESCRIPTION = ("Split the script-writing model's JSON answer into the spoken script, "
                   "the social caption, a headline, a filename slug and a one-scene plan "
                   "for the subtitle burner.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "answer": ("STRING", {
                    "default": "", "multiline": True, "forceInput": True,
                    "tooltip": "The script-writing model's raw answer. Wire the LLM's "
                               "`text` output straight in.",
                }),
            },
            "optional": {
                "caption_answer": ("STRING", {
                    "default": "", "multiline": True, "forceInput": True,
                    "tooltip": "The CAPTION model's answer, when the caption is written "
                               "by its own call with its own instructions. Wired, it "
                               "wins; unwired, the caption is taken from the script "
                               "answer if it happens to carry one.",
                }),
                "prefix": ("STRING", {
                    "default": "avatar",
                    "tooltip": "Folder the outputs file under. The slug is appended, so "
                               "each story gets its own name inside it.",
                }),
                "max_words": ("INT", {
                    "default": 0, "min": 0, "max": 2000,
                    "tooltip": "0 = accept whatever was written. Above 0 the run STOPS "
                               "when the script is longer, because an over-long script "
                               "is only discovered as a video whose voice outruns its "
                               "picture — after the render has been paid for.",
                }),
            },
        }

    def run(self, answer, caption_answer="", prefix="avatar", max_words=0):
        data = parse(answer)
        script = " ".join(str(data["script"]).split())
        # The caption comes from its own call when there is one. Its instructions are
        # written for social copy and the script's are written for spoken words; asking
        # one prompt to do both is what the separate call exists to avoid.
        if (caption_answer or "").strip():
            caption = _caption_from(caption_answer)
        else:
            caption = _caption(data)
        headline = str(data.get("headline") or "").strip()
        source_url = str(data.get("source_url") or "").strip()
        words = len(script.split())

        if max_words and words > max_words:
            raise RuntimeError(
                "ArkAvatarScript: the script is %d words and the cap is %d. At about 2 "
                "words a second that is %.0f seconds of speech, and the clip is sized "
                "from it — so this would render long and cost accordingly. Lower "
                "`seconds` on the brief, or raise this cap deliberately."
                % (words, max_words, words / 2.0))

        slug = _slugify(headline or script)
        # The shape ArkVideoAssemble reads. `voiceText` is the spelling its `_voice_text`
        # looks for first; `scene` and `seconds` are carried for the same reason, so this
        # plan is interchangeable with one the story agent writes.
        scenes = json.dumps([{"scene": 1, "voiceText": script,
                              "seconds": round(words / 2.0, 2)}], ensure_ascii=False)

        report = ("%d words (~%.0fs) | slug %s%s"
                  % (words, words / 2.0, slug,
                     "" if not source_url else " | source %s" % source_url[:60]))
        print("[arkennemasis] avatar script: %s" % report)
        save_prefix = "%s/%s" % (prefix.strip("/ ") or "avatar", slug)
        return (script, caption, headline, slug, save_prefix, scenes, words, report)


NODE_CLASS_MAPPINGS = {"ArkAvatarScript": ArkAvatarScript}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ArkAvatarScript": "arkennemasis Avatar Script (split the model's answer)",
}

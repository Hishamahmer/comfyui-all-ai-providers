"""Write today's story into the history — but only once there is a video to show for it.

This exists because of a failure that looked like success. `ArkStoryPick` used to append
to the history the instant it chose a story, which is the earliest possible moment and the
wrong one. On 2026-08-22 a Variety article was picked, recorded as covered, and then the
article-text fetch timed out four minutes later. The run died with nothing rendered, and
the story was **permanently excluded** from every future run: consumed, with no video.

The rule the history is supposed to encode is "do not cover the same story twice". A run
that produced nothing did not cover anything. So the record belongs at the END of the
pipeline, gated on the finished file — which is exactly what `done` is for. ComfyUI orders
execution by data dependency, so taking the assembler's `final_path` as an input is what
makes this node run last, after the video is on disk.

`done` is checked, not merely depended on. A node that ran is not proof a file was
written, and the whole point here is to stop recording things that did not happen.
"""

from __future__ import annotations

import datetime
import os

from .story_history import append_history, load_history, normalise


class ArkStoryRecord:
    CATEGORY = "arkennemasis/Avatar"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    OUTPUT_NODE = True
    DESCRIPTION = ("Remember that this story has been covered, so tomorrow's run picks a "
                   "different one. Runs LAST, gated on the finished video — a run that "
                   "failed has covered nothing and must not consume a story.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {
                    "default": "", "forceInput": True,
                    "tooltip": "The story that was covered. Wire ArkStoryPick's `url`.",
                }),
                "done": ("STRING", {
                    "default": "", "forceInput": True,
                    "tooltip": "The finished video's path, from ArkVideoAssemble's "
                               "`final_path`. This is what makes this node run LAST, and "
                               "it is checked: no file, no record.",
                }),
                "history_file": ("STRING", {
                    "default": "",
                    "tooltip": "Where the covered stories are remembered. The SAME path "
                               "ArkStoryHistory reads. Blank records nothing, and the "
                               "run can then repeat itself tomorrow.",
                }),
            },
            "optional": {
                "headline": ("STRING", {
                    "default": "", "forceInput": True,
                    "tooltip": "Shown to the picking model next to the URL, so it can "
                               "avoid the same STORY from another publisher — which the "
                               "URL check cannot catch on its own.",
                }),
                "require_file": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Only record if the video actually exists on disk. Turn "
                               "this off only if you have moved the output elsewhere.",
                }),
            },
        }

    def run(self, url, done, history_file="", headline="", require_file=True):
        path = (history_file or "").strip()
        story = (url or "").strip()
        final = (done or "").strip()

        if not path:
            msg = ("ArkStoryRecord: no history_file, so nothing was remembered — this "
                   "story can be picked again tomorrow.")
            print("[arkennemasis] " + msg)
            return (msg,)
        if not story:
            msg = "ArkStoryRecord: no url, nothing to remember."
            print("[arkennemasis] " + msg)
            return (msg,)

        # The gate. A path that names no file means the render did not finish, and
        # recording here is what would burn the story.
        if require_file and not (final and os.path.isfile(final)):
            msg = ("ArkStoryRecord: no finished video at %r, so %s was NOT recorded — it "
                   "stays available for the next run." % (final, story))
            print("[arkennemasis] " + msg)
            return (msg,)

        already = any(normalise(e.get("url")) == normalise(story)
                      for e in load_history(path))
        try:
            total = append_history(path, story, (headline or "").strip(),
                                   datetime.date.today().isoformat())
        except Exception as exc:
            # A history that cannot be written is tomorrow's problem. Losing the video
            # that was just rendered over it would be the wrong trade.
            msg = ("ArkStoryRecord: could not write the history (%s) — this story may be "
                   "picked again tomorrow." % exc)
            print("[arkennemasis] " + msg)
            return (msg,)

        msg = ("recorded %s%s  |  %d covered  |  video %s"
               % (story, "" if not already else " (already there)", total,
                  os.path.basename(final) or "(unchecked)"))
        print("[arkennemasis] story record: " + msg)
        return (msg,)


NODE_CLASS_MAPPINGS = {"ArkStoryRecord": ArkStoryRecord}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ArkStoryRecord": "arkennemasis Story Record (remember it, once it rendered)"}

"""Remember which stories have already been covered, so the next one is genuinely new.

A daily workflow that picks from a front page will happily pick the same story two days
running, because a big story stays near the top for days and the model has no way to know
it already made that video. Nothing upstream can fix this: the candidate list is the page
as it is today, and today's page still leads with yesterday's news.

So the run keeps a file. This node reads it and hands the picker two things:

    covered   a readable block for the PROMPT — "you have already covered these"
    urls      the same URLs as plain lines, for `ArkStoryPick` to REJECT against

Both, deliberately. The prompt makes the model choose something else; the URL list makes
it *impossible* to choose the same thing anyway. Instructions are guidance and a check is
a guarantee, and the guarantee is what stops a duplicate video being rendered.

The file is written by `ArkStoryPick` after it accepts a story, so the pair works as a
loop across runs: read at the start, append at the end. Nothing else touches it, and a
missing or damaged file is treated as "nothing covered yet" rather than as an error —
losing the history should cost a repeated story, never a failed run.
"""

from __future__ import annotations

import json
import os

# What one story looks like on disk. `date` is when we covered it, not when it was
# published — publication dates are not available from a link, which is the whole reason
# recency is judged by page position elsewhere in this pipeline.
FIELDS = ("url", "headline", "date")


def load_history(path):
    """Every covered story, oldest first. Never raises — a lost history is not a failure."""
    if not path or not os.path.isfile(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception as exc:
        print("[arkennemasis] story history: %s is unreadable (%s) — treating it as "
              "empty. The worst case is a repeated story." % (path, exc))
        return []
    if not isinstance(data, list):
        return []
    return [e for e in data if isinstance(e, dict) and e.get("url")]


def normalise(url):
    """The comparison key. Two links to one article differ in trailing slash and #anchor."""
    return (url or "").split("#")[0].split("?")[0].rstrip("/").lower()


def append_history(path, url, headline="", date=""):
    """Add one story, atomically, without duplicating it. Returns the new length."""
    entries = load_history(path)
    key = normalise(url)
    if any(normalise(e.get("url")) == key for e in entries):
        return len(entries)                     # already there; nothing to do
    entries.append({"url": url, "headline": headline, "date": date})
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    tmp = path + ".part"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(entries, fh, indent=2, ensure_ascii=False)
    os.replace(tmp, path)                       # atomic, the house rule
    return len(entries)


class ArkStoryHistory:
    CATEGORY = "arkennemasis/Avatar"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING", "STRING", "INT", "STRING")
    RETURN_NAMES = ("covered", "urls", "count", "report")
    DESCRIPTION = ("Read the stories already covered, so today's pick is a new one. Wire "
                   "`covered` into the picker's prompt and `urls` into ArkStoryPick, "
                   "which appends to the same file after it accepts a story.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "history_file": ("STRING", {
                    "default": "", "multiline": False,
                    "tooltip": "Where the covered stories are recorded. The same path "
                               "goes on ArkStoryPick, which appends to it.",
                }),
                "remember": ("INT", {
                    "default": 40, "min": 0, "max": 1000,
                    "tooltip": "How many recent stories to show the picker. The file "
                               "keeps everything; this only limits how much reaches the "
                               "prompt, because an unbounded list would eventually be "
                               "most of what the model reads.",
                }),
            },
        }

    @classmethod
    def IS_CHANGED(cls, history_file="", **kwargs):
        # Stamp the file, so appending a story at the END of one run invalidates this node
        # at the START of the next. Without it the history is read once and then served
        # from cache forever, and the whole feature quietly stops working after run one.
        try:
            stat = os.stat(history_file)
            return "%d:%d" % (stat.st_mtime_ns, stat.st_size)
        except Exception:
            return "missing"

    def run(self, history_file="", remember=40):
        path = (history_file or "").strip()
        entries = load_history(path)
        recent = entries[-int(remember):] if remember else []

        lines = []
        for entry in recent:
            headline = (entry.get("headline") or "").strip()
            lines.append("%s\t%s" % (headline or "(no headline)", entry.get("url")))
        urls = "\n".join(e.get("url", "") for e in recent)

        if lines:
            covered = ("ALREADY COVERED — do NOT pick any of these again, and do not pick "
                       "a different article about the same development:\n" + "\n".join(lines))
        else:
            covered = ("ALREADY COVERED: nothing yet. This is the first run against this "
                       "history file.")

        report = "%d covered in total, %d shown to the picker | %s" % (
            len(entries), len(recent), path or "(no file set)")
        print("[arkennemasis] story history: %s" % report)
        return (covered, urls, len(entries), report)


NODE_CLASS_MAPPINGS = {"ArkStoryHistory": ArkStoryHistory}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ArkStoryHistory": "arkennemasis Story History (never cover the same thing twice)",
}

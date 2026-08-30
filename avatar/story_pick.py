"""Read the model's choice of story out of its answer, and check it is usable.

This is the local stand-in for the Perplexity tool in the workflow this was ported from.
That tool was given one job and one constraint — *"return accurate and timely citations …
Include only standalone article URLs — not homepages, section pages, or aggregators"* —
and it was domain-filtered to a single publisher. Both halves of that matter, and both are
enforced here rather than hoped for:

**A section page is not a story.** `https://site.com/music/` screenshots as a list of
headlines and reads as navigation, so the video ends up about nothing in particular. A URL
with no real path depth is rejected.

**A URL the model invented is worse than no URL**, because it fails four nodes later as a
screenshot of a 404. So the chosen URL must appear in the candidate list it was given.
That check is the whole reason the links are passed in as data instead of the model being
asked to remember what it saw.

The override exists because "make a video about *this* one" is a normal thing to want on
a day when the automatic pick is wrong, and it should not require rewiring the canvas.
"""

from __future__ import annotations

import json
import re

_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)

# A story lives at a path, not at a domain root or a one-segment section. `/artist-sues-
# label-over-royalties/` passes; `/music/` does not. Deliberately loose — publishers use
# dates, categories and slugs in every combination — it only has to reject the obvious.
def _looks_like_article(url):
    without_scheme = re.sub(r"^https?://", "", (url or "").strip()).rstrip("/")
    if "/" not in without_scheme:
        return False                              # bare domain
    path = without_scheme.split("/", 1)[1]
    if not path:
        return False
    last = path.rstrip("/").split("/")[-1]
    # A slug is wordy; a section is one short word.
    return len(last) >= 12 or "-" in last


def parse(answer):
    raw = _FENCE.sub("", (answer or "").strip())
    if not raw:
        raise RuntimeError(
            "ArkStoryPick got an empty answer. The picking model returned nothing — "
            "check that node for an error and that `json_only` is on.")
    try:
        data = json.loads(raw)
    except Exception as exc:
        raise RuntimeError(
            "ArkStoryPick could not read the answer as JSON (%s). It must be one object "
            "with a `url`. First 300 characters:\n%s" % (exc, raw[:300]))
    if isinstance(data, list) and data:
        data = data[0]
    if not isinstance(data, dict):
        raise RuntimeError("ArkStoryPick expected a JSON object, got %s."
                           % type(data).__name__)
    return data


class ArkStoryPick:
    CATEGORY = "arkennemasis/Avatar"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("url", "headline", "why", "report")
    DESCRIPTION = ("Take the chosen story out of the picking model's JSON answer, check "
                   "it is a real article from the candidate list, and hand its URL on to "
                   "the screenshot and the script.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "answer": ("STRING", {
                    "default": "", "multiline": True, "forceInput": True,
                    "tooltip": "The picking model's raw answer. Wire the LLM's `text` "
                               "output straight in.",
                }),
            },
            "optional": {
                "candidates": ("STRING", {
                    "default": "", "multiline": True, "forceInput": True,
                    "tooltip": "The `links` the model was shown. The chosen URL is "
                               "checked against these, so a hallucinated link fails HERE "
                               "instead of as a screenshot of a 404 three nodes later.",
                }),
                "override_url": ("STRING", {
                    "default": "",
                    "tooltip": "Make the video about THIS story instead. Set it and the "
                               "model's choice is ignored — for the day the automatic "
                               "pick is wrong.",
                }),
                "require_article": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Reject a bare domain or a section page. A section page "
                               "screenshots as a list of headlines and reads as "
                               "navigation, so the video ends up about nothing.",
                }),
                "covered_urls": ("STRING", {
                    "default": "", "multiline": True, "forceInput": True,
                    "tooltip": "URLs already covered, from ArkStoryHistory. A story in "
                               "this list is REJECTED — the prompt asks the model not to "
                               "repeat itself, and this makes sure it cannot.",
                }),
            },
        }

    def run(self, answer, candidates="", override_url="", require_article=True,
            covered_urls=""):
        override = (override_url or "").strip()
        if override:
            url = override
            headline, why = "", "operator override"
        else:
            data = parse(answer)
            url = str(data.get("url") or data.get("source_url") or "").strip()
            headline = str(data.get("headline") or data.get("title") or "").strip()
            why = str(data.get("why") or data.get("reason") or "").strip()
            if not url:
                raise RuntimeError(
                    "ArkStoryPick: the answer carries no `url`. It returned: %s"
                    % ", ".join(sorted(data)) or "(nothing)")

        if not re.match(r"^https?://", url):
            url = "https://" + url.lstrip("/")

        if require_article and not _looks_like_article(url):
            raise RuntimeError(
                "ArkStoryPick: %s is a section or a homepage, not a story. The "
                "instructions ask for a standalone article URL; re-run, or set "
                "`override_url`." % url)

        known = [l.split("\t")[-1].strip() for l in (candidates or "").splitlines()
                 if l.strip()]
        if known and not override:
            base = url.split("#")[0].rstrip("/")
            if not any(base == k.split("#")[0].rstrip("/") for k in known):
                raise RuntimeError(
                    "ArkStoryPick: %s was not in the %d links the model was shown, so it "
                    "was invented rather than chosen. Re-run; if it repeats, the source "
                    "page returned no usable links." % (url, len(known)))

        # Already covered? The prompt asked the model not to repeat itself; this is the
        # part that makes it impossible.
        #
        # An OVERRIDE is exempt, and that is a deliberate reversal. It used to be checked
        # too, on the reasoning that naming a URL by hand is exactly when someone forgets
        # they already covered it. In practice the opposite bites harder: wanting to redo
        # a story — because the captions were wrong, or the render failed — is a normal
        # thing to want, and being refused by your own history with no way through is
        # worse than a repeat you asked for. An explicit instruction should beat an
        # implicit rule. It still says so loudly.
        from .story_history import normalise

        key = normalise(url)
        covered = [normalise(u) for u in (covered_urls or "").splitlines() if u.strip()]
        if key in covered:
            if not override:
                raise RuntimeError(
                    "ArkStoryPick: %s has already been covered — it is in the history of "
                    "%d stories this run was shown. Re-run to pick a different one, or "
                    "set `override_url` to it if you deliberately want it again."
                    % (url, len(covered)))
            print("[arkennemasis] story pick: %s was already covered, but you named it "
                  "explicitly — covering it again." % url)

        # NOTHING is written to the history here, deliberately. Choosing a story is the
        # earliest moment in the run and recording it here means a failure downstream
        # consumes it for good: on 2026-08-22 a Variety article was picked, recorded, and
        # then the article fetch timed out — the story was excluded from every future run
        # with no video to show for it. `ArkStoryRecord` does the writing, at the far end,
        # gated on the finished file. A run that produced nothing covered nothing.
        report = "%s%s%s" % (url, "  |  " + headline if headline else "",
                             "  |  " + why if why else "")
        print("[arkennemasis] story pick: %s" % report)
        return (url, headline, why, report)


NODE_CLASS_MAPPINGS = {"ArkStoryPick": ArkStoryPick}
NODE_DISPLAY_NAME_MAPPINGS = {"ArkStoryPick": "arkennemasis Story Pick (choose today's story)"}

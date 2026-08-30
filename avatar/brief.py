"""The idea form — one labelled box per thing a person actually decides.

The alternative is a pre-formatted template in a text box that everyone edits by hand and
somebody eventually breaks by deleting a heading. A form cannot be broken that way: each
field is its own widget, and the assembly into a prompt happens here where it is testable.

Two rules carried over from `ArkStoryBrief`, both learned rather than chosen:

**An empty field is omitted, never sent blank.** A heading with nothing under it reads to
a language model as a constraint it has failed to satisfy, and it will invent something to
put there.

**`special_request` goes first**, because the model treats what it reads first as the
governing instruction, and that box is where the operator puts the thing they care about
most on this particular day.

**The word budget is computed, not typed.** Qwen3-TTS on this machine speaks at about
a rate measured on THIS voice, at the length it actually speaks — 2.65 words per second
across full scripts. A script written to someone else's word count runs long locally, the
voice outruns the clip, and the picture freezes to cover it; a script written to a rate
measured on twenty-word test lines runs short, and a thirty-second ask quietly returns a
twenty-two-second video. So the model is told a word count derived from the seconds asked
for and the rate this machine actually speaks at over a whole script. See
WORDS_PER_SECOND below for the measurements and why length changes the answer.
"""

from __future__ import annotations

# Measured on this install, 2026-08.
#
# SHORT lines: 22 words in 11.10 s, 21 in 8.62 s, 17 in 10.30 s — about 2.0 w/s.
# FULL scripts: 65 words in 23.58 s (2.76 w/s), 64 in 22.46 s (2.85), 158 in 54.54 s
#               (2.90), 160 in 56.06 s (2.85). Mean of the four: 2.84 w/s.
#
# Those are not contradictory measurements, they are a length effect. Every utterance
# carries a fixed lead-in and tail of near-silence, and on a 20-word clip that fixed cost
# is most of the difference; across a 60-word script it amortises away. Reading the rate
# off short lines and applying it to a long one is what made a 30-second request come
# back as a 22.5-second video — the budget was a quarter short before the model ever
# started writing.
#
# So the rate is taken from the FULL-SCRIPT measurements, which is the length this
# actually runs at. 2.65 was the first correction, made from the two shortest of those
# four, and it still came in low: two 60 s runs measured against it landed at 55.4 s and
# 57.0 s. 2.80 is just under the observed mean of 2.84, which puts a 60 s ask at 168 words
# and 58-61 s at the rates above, while staying inside ArkAvatarFrames' ceiling (20% above
# the target) even at the slowest rate seen. Still the conservative side, on purpose:
# short is a beat of silence, long has nowhere to go.
WORDS_PER_SECOND = 2.80

LANGUAGES = ["English", "Hindi", "Spanish", "French", "German", "Portuguese",
             "Italian", "Japanese", "Korean", "Chinese", "Arabic"]

# Where to go looking, by beat. These are SECTION pages: the model reads the headlines on
# them and picks one story.
#
# ONE PER LINE. Every one of these was tested by taking a real article from it and
# screenshotting that article through the hosted API — the only check that matters, since
# the shot is the video's background. Measured 2026-08-22:
#
#   musicbusinessworldwide  the best of them: big headline, date, byline, hero image, and
#                           its front page is genuinely chronological, so the top link is
#                           today's story. This is why it leads.
#   completemusicupdate     clean bold headline and standfirst
#   musicweek               clear headline, byline, hero image
#
# Deliberately NOT included, and why:
#   billboard.com/c/business  screenshots as "Disable Your Adblocker" — a wall, no headline
#   hypebot.com               a survey banner covers the top third of the article
#   digitalmusicnews.com      headline is small, under a logo and a subscribe bar
#
# A caveat worth knowing: only MBW was verified to list newest-first. Music Week and CMU
# returned 2024/2025 articles as their top link, so their contribution to "the last 24
# hours" is weaker — they add breadth, not recency.
MUSIC_SOURCES = (
    "https://www.musicbusinessworldwide.com/",
    "https://completemusicupdate.com/",
    "https://www.musicweek.com/",
)
DEFAULT_MUSIC_SOURCE = "\n".join(MUSIC_SOURCES)
# Entertainment is the SECOND beat and only used when music is thin, so one source is
# enough — and Variety is slow (58 s measured), so a second would cost a minute for
# candidates that usually lose to a music story anyway.
DEFAULT_ENTERTAINMENT_SOURCE = "https://variety.com/v/music/"


def _section(title, body):
    body = (body or "").strip()
    return "" if not body else "%s\n%s\n" % (title, body)


def compose(topic="", presenter="", seconds=30.0, call_to_action="",
            language="English", must_mention="", avoid="", special_request=""):
    """Assemble the operator's fields into one prompt. Pure, so it can be tested."""
    words = max(8, int(round(float(seconds) * WORDS_PER_SECOND)))
    lower, upper = max(6, words - 6), words + 6

    parts = []
    # First, on purpose — see the module docstring.
    parts.append(_section("TOP PRIORITY FOR THIS RUN:", special_request))
    parts.append(_section("WHAT THIS CHANNEL IS ABOUT:", topic))
    parts.append(_section("WHO IS SPEAKING:", presenter))
    parts.append(_section("MUST BE MENTIONED:", must_mention))
    parts.append(_section("MUST NOT APPEAR:", avoid))
    parts.append(_section("HOW IT SHOULD END:", call_to_action))
    parts.append(
        "LENGTH: the spoken script must be %d-%d words — about %.0f seconds at the "
        "%.1f words per second this voice actually speaks at. Going over does not make a "
        "longer video, it makes the voice outrun the picture.\n" % (lower, upper,
                                                                   float(seconds),
                                                                   WORDS_PER_SECOND))
    parts.append("LANGUAGE: write the script and the caption in %s.\n" % language)
    return "\n".join(p for p in parts if p).strip(), words


class ArkAvatarBrief:
    CATEGORY = "arkennemasis/Avatar"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "FLOAT", "INT")
    RETURN_NAMES = ("brief", "source_music", "source_entertainment", "override_url",
                    "seconds", "words")
    DESCRIPTION = ("The idea form for one presenter video. Fill in the boxes; this "
                   "assembles them into the prompt for the script-writing model and "
                   "hands the two source pages to the story hunt.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_music": ("STRING", {
                    "default": DEFAULT_MUSIC_SOURCE, "multiline": True,
                    "tooltip": "MUSIC sources, ONE PER LINE. The primary beat — a "
                               "strong music story always beats an entertainment "
                               "one. Prefix a line with # to switch a source off "
                               "without losing it. Every source here was verified by "
                               "screenshotting a real article from it.",
                }),
                "source_entertainment": ("STRING", {
                    "default": DEFAULT_ENTERTAINMENT_SOURCE, "multiline": True,
                    "tooltip": "The ENTERTAINMENT section page, read second and used "
                               "only when the music candidates are thin.",
                }),
                "topic": ("STRING", {
                    "default": "", "multiline": True,
                    "tooltip": "What this channel covers and who watches it. Steers "
                               "which part of the story is worth 30 seconds.",
                }),
                "presenter": ("STRING", {
                    "default": "", "multiline": True,
                    "tooltip": "Who is speaking and how — their register, pace and "
                               "attitude. This is the voice of the script, not the face; "
                               "the face comes from the reference photo.",
                }),
                "seconds": ("FLOAT", {
                    "default": 30.0, "min": 5.0, "max": 180.0, "step": 1.0,
                    "tooltip": "Target spoken length. The word budget is derived from "
                               "this, and so is the cost of the render — video time "
                               "scales with frames.",
                }),
                "language": (LANGUAGES, {
                    "tooltip": "The language of the script and the caption.",
                }),
            },
            "optional": {
                "call_to_action": ("STRING", {
                    "default": "", "multiline": True,
                    "tooltip": "How it should close. Left blank the model writes its "
                               "own, which is usually a generic 'follow for more'.",
                }),
                "must_mention": ("STRING", {
                    "default": "", "multiline": True,
                    "tooltip": "Anything that has to be in there — a name, a number, a "
                               "product.",
                }),
                "avoid": ("STRING", {
                    "default": "", "multiline": True,
                    "tooltip": "Words, claims or angles to keep out.",
                }),
                "special_request": ("STRING", {
                    "default": "", "multiline": True,
                    "tooltip": "The one thing that matters most today. Emitted FIRST, "
                               "because a model weights what it reads first.",
                }),
                "override_url": ("STRING", {
                    "default": "",
                    "tooltip": "Make today's video about THIS article instead, skipping "
                               "the automatic pick. For the day the model chooses badly, "
                               "or when you already know the story you want.",
                }),
            },
        }

    def run(self, source_music=DEFAULT_MUSIC_SOURCE,
            source_entertainment=DEFAULT_ENTERTAINMENT_SOURCE, topic="", presenter="",
            seconds=30.0, language="English", call_to_action="", must_mention="",
            avoid="", special_request="", override_url=""):
        brief, words = compose(topic=topic, presenter=presenter, seconds=seconds,
                               call_to_action=call_to_action, language=language,
                               must_mention=must_mention, avoid=avoid,
                               special_request=special_request)
        override = (override_url or "").strip()
        print("[arkennemasis] avatar brief: %.0fs -> %d words | music %s%s"
              % (seconds, words, (source_music or "").strip()[:50],
                 " | OVERRIDE " + override[:50] if override else ""))
        return (brief, (source_music or "").strip(),
                (source_entertainment or "").strip(), override, float(seconds), words)


NODE_CLASS_MAPPINGS = {"ArkAvatarBrief": ArkAvatarBrief}
NODE_DISPLAY_NAME_MAPPINGS = {"ArkAvatarBrief": "arkennemasis Avatar Brief (the idea form)"}

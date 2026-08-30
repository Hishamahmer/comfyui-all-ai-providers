"""ASS subtitle generation: font discovery, the five caption styles, and the burn args.

Shared by ``ArkCaptionStyle`` (which offers the choices) and ``ArkVideoAssemble`` (which
burns them). Kept apart from both so the style vocabulary has one home.

WHY ASS AND NOT SRT
    SRT carries text and a time range, nothing else — one look, one position, no way to
    mark a word. Everything past "white text at the bottom" needs ASS: per-word tags,
    the 3x3 position grid, an opaque box, outline and shadow control.

THE FIVE STYLES
    classic       one event per line, the whole line at once
    karaoke       one event, `\\k` durations, libass sweeps the fill left to right
    highlight     a base line on layer 0 + one recoloured overlay per word on layer 1
    underline     one full-line event per word, the active word wrapped in `\\u1..\\u0`
    word_by_word  one event per word, nothing else on screen

    Only `classic` runs on segment timings. The other four need to know when each WORD
    starts, which we do not get from a video model — see `estimate_words`.

FONTS
    libass matches on the FAMILY name recorded inside the file, which is not the
    filename: ARIBLK.TTF is "Arial Black", THEBOLDFONT.ttf is "The Bold Font". So the
    dropdown is built by reading each font's `name` table rather than by listing files.
"""

import os
import struct
import sys

# ---------------------------------------------------------------- font discovery

PACK_FONTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "fonts")


def _system_font_dirs():
    if sys.platform == "win32":
        dirs = [os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")]
        local = os.environ.get("LOCALAPPDATA")
        if local:
            dirs.append(os.path.join(local, "Microsoft", "Windows", "Fonts"))
        return dirs
    if sys.platform == "darwin":
        return ["/System/Library/Fonts", "/Library/Fonts",
                os.path.expanduser("~/Library/Fonts")]
    return ["/usr/share/fonts", "/usr/local/share/fonts",
            os.path.expanduser("~/.local/share/fonts")]


def _family_of(path):
    """Family name from a TrueType/OpenType `name` table, or None.

    Name ID 16 is the typographic family ("Roboto"); ID 1 is the legacy family, which
    for a weight-specific file says "Roboto Black" instead. Prefer 16 so all twelve
    Roboto files collapse to one dropdown entry, exactly as libass groups them.
    """
    try:
        with open(path, "rb") as fh:
            head = fh.read(12)
            if len(head) < 12 or head[:4] not in (b"\x00\x01\x00\x00", b"OTTO", b"true"):
                return None                     # .ttc and friends: skip, not worth it
            count = struct.unpack(">H", head[4:6])[0]
            directory = fh.read(16 * count)
            offset = length = None
            for i in range(count):
                entry = directory[16 * i:16 * i + 16]
                if entry[:4] == b"name":
                    offset, length = struct.unpack(">II", entry[8:16])
                    break
            if offset is None:
                return None
            fh.seek(offset)
            table = fh.read(length)
    except Exception:
        return None

    if len(table) < 6:
        return None
    _, records, strings = struct.unpack(">HHH", table[:6])
    legacy = None
    for i in range(records):
        rec = table[6 + 12 * i:18 + 12 * i]
        if len(rec) < 12:
            break
        platform, _, _, name_id, size, at = struct.unpack(">HHHHHH", rec)
        if name_id not in (1, 16):
            continue
        raw = table[strings + at:strings + at + size]
        try:
            text = raw.decode("utf-16-be" if platform in (0, 3) else "latin-1")
        except Exception:
            continue
        text = text.replace("\x00", "").strip()
        if not text:
            continue
        if name_id == 16:
            return text
        if legacy is None:
            legacy = text
    return legacy


def scan_fonts(extra_dirs=()):
    """{family: directory} for every font we can offer, pack fonts winning ties."""
    found = {}
    roots = list(_system_font_dirs()) + [d for d in extra_dirs if d] + [PACK_FONTS]
    for root in roots:                    # PACK_FONTS last so it overrides the system
        if not root or not os.path.isdir(root):
            continue
        try:
            names = os.listdir(root)
        except OSError:
            continue
        for name in names:
            if not name.lower().endswith((".ttf", ".otf")):
                continue
            family = _family_of(os.path.join(root, name))
            if family:
                found[family] = root
    return found


_CACHE = {}


def families(extra_dirs=()):
    key = tuple(extra_dirs)
    if key not in _CACHE:
        found = scan_fonts(extra_dirs)
        bundled = sorted(f for f, d in found.items() if d == PACK_FONTS)
        other = sorted(f for f, d in found.items() if d != PACK_FONTS)
        _CACHE[key] = (bundled + other, found)   # bundled first: the curated set
    return _CACHE[key]


# ---------------------------------------------------------------- style vocabulary

STYLES = ["classic", "karaoke", "highlight", "underline", "word_by_word"]
NEEDS_WORD_TIMING = {"karaoke", "highlight", "underline", "word_by_word"}
POSITIONS = ["bottom_center", "bottom_left", "bottom_right",
             "middle_center", "middle_left", "middle_right",
             "top_center", "top_left", "top_right"]
ALIGNMENTS = ["center", "left", "right"]
BORDERS = ["outline", "opaque box"]


def rgb_to_ass(value, default="#FFFFFF"):
    """#RRGGBB -> &H00BBGGRR. ASS stores colour byte-reversed with an alpha byte."""
    text = str(value or default).strip().lstrip("#")
    if len(text) == 3:
        text = "".join(c * 2 for c in text)
    if len(text) != 6:
        text = default.lstrip("#")
    try:
        r, g, b = (int(text[i:i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        r, g, b = 255, 255, 255
    return "&H00%02X%02X%02X" % (b, g, r)


def alignment_code(position, align, width, height):
    """(\\an code, x, y) from the 3x3 grid. Mirrors libass numbering: 1-3 bottom."""
    if "top" in position:
        base, y = 7, height / 6.0
    elif "middle" in position:
        base, y = 4, height / 2.0
    else:
        base, y = 1, 5 * height / 6.0
    if "left" in position:
        left, right, centre = 0.0, width / 3.0, width / 6.0
    elif "right" in position:
        left, right, centre = 2 * width / 3.0, float(width), 5 * width / 6.0
    else:
        left, right, centre = width / 3.0, 2 * width / 3.0, width / 2.0
    if align == "left":
        x, column = left, 1
    elif align == "right":
        x, column = right, 3
    else:
        x, column = centre, 2
    return base + column - 1, int(x), int(y)


def ass_time(seconds):
    seconds = max(0.0, float(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, rest = divmod(rest, 60)
    whole = int(rest)
    return "%d:%02d:%02d.%02d" % (hours, minutes, whole,
                                  min(99, int(round((rest - whole) * 100))))


# Trailing punctuation buys silence. Without this every word gets the same share and a
# line that ends on a full stop drifts a beat ahead of the speech by the next line.
_PAUSE = {",": 1.6, ";": 1.8, ":": 1.8, ".": 2.6, "!": 2.6, "?": 2.6, "-": 1.2}


def estimate_words(text, start, end):
    """Word timings spread across a known span, weighted by length and punctuation.

    A video model gives us no word timestamps, and the plan text is what it was ASKED to
    say rather than a transcript. So the moving styles run on an estimate: each word's
    share of the span is its character count plus a pause bonus for trailing punctuation.

    Good enough to track a line as it is spoken; NOT frame-accurate, and it drifts if
    the model pauses or ad-libs. Real per-word timing needs transcription of the
    finished audio.
    """
    words = str(text).split()
    if not words:
        return []
    span = max(0.05, float(end) - float(start))
    weights = []
    for word in words:
        weight = float(len(word)) + 1.0
        tail = word[-1:]
        weight += _PAUSE.get(tail, 0.0)
        weights.append(weight)
    total = sum(weights) or 1.0
    timed, clock = [], float(start)
    for word, weight in zip(words, weights):
        length = span * weight / total
        timed.append({"word": word, "start": clock, "end": clock + length})
        clock += length
    timed[-1]["end"] = float(end)          # absorb rounding into the last word
    return timed


def _normalise(word):
    """Compare words by their letters only.

    Whisper returns " Round" and "Hill," where the script has "Round" and "Hill" — case,
    leading space and trailing punctuation all differ without the word being different.
    """
    return "".join(ch for ch in str(word).lower() if ch.isalnum())


def cues_from_timings(timings, per_cue=4):
    """Build the cues FROM the measured words, so nothing has to be aligned.

    The first version of this tried to keep the script's wording and borrow the measured
    times, matching the two word by word. That is fragile for a reason no amount of fuzzy
    matching fixes: the script and the transcript are different TOKENISATIONS of the same
    speech. Measured on a real run — the script said "one billion dollars" (three words)
    and the transcript "$1 billion" (two), and every word after that sat one slot late
    until the end of the line. A one-slot shift is precisely the bug this exists to fix.

    So the measured words ARE the caption. Each carries its own start and end, there is
    nothing to match, and the mark cannot land on the wrong word by construction.

    It also produces better captions, which is a happy accident rather than the point:
    the script spells numbers out because a text-to-speech model reads "$1B" wrong, while
    a caption is read by eye and "$1 billion" is how anyone would write it. The script is
    written for the ear; the transcript is what was actually said, written for the page.

    Returns [(start, end, words, text)] per cue.
    """
    out = []
    step = max(1, int(per_cue))
    for index in range(0, len(timings), step):
        group = timings[index:index + step]
        words = [{"word": w["word"].strip(), "start": w["start"], "end": w["end"]}
                 for w in group if w["word"].strip()]
        if not words:
            continue
        out.append((words[0]["start"], words[-1]["end"], words,
                    " ".join(w["word"] for w in words)))
    return out


def _shape(text, style):
    if style.get("all_caps"):
        text = text.upper()
    return text.replace("{", "(").replace("}", ")")   # braces are ASS override tags


def _lines(items, per_line):
    if per_line and per_line > 0:
        return [items[i:i + per_line] for i in range(0, len(items), per_line)]
    return [items]


def header(style, resolution):
    width, height = resolution
    size = int(style.get("font_size") or 0) or max(12, int(height * 0.05))
    border = 3 if style.get("border_style") == "opaque box" else 1
    return (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        "PlayResX: %d\nPlayResY: %d\n"
        "ScaledBorderAndShadow: yes\n"
        "WrapStyle: 0\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, "
        "Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, "
        "MarginV, Encoding\n"
        "Style: Default,%s,%d,%s,%s,%s,%s,%d,%d,0,0,100,100,0,0,%d,%d,%d,5,%d,%d,%d,1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, "
        "Text\n" % (
            width, height,
            style.get("font", "Arial"), size,
            rgb_to_ass(style.get("line_color"), "#FFFFFF"),
            rgb_to_ass(style.get("word_color"), "#FFE600"),
            rgb_to_ass(style.get("outline_color"), "#000000"),
            rgb_to_ass(style.get("box_color"), "#000000"),
            -1 if style.get("bold") else 0,
            -1 if style.get("italic") else 0,
            border,
            int(style.get("outline_width", 3)),
            int(style.get("shadow_offset", 1)),
            int(style.get("margin_h", 40)), int(style.get("margin_h", 40)),
            int(style.get("margin_v", 40))))


def events(cues, style, resolution):
    """Dialogue lines for one of the five styles.

    `cues` is [{text, start, end, words:[{word,start,end}]}]; `words` is only consulted
    by the four moving styles.
    """
    kind = style.get("style", "classic")
    per_line = int(style.get("max_words_per_line", 0) or 0)
    code, x, y = alignment_code(style.get("position", "bottom_center"),
                                style.get("alignment", "center"), *resolution)
    at = "{\\an%d\\pos(%d,%d)}" % (code, x, y)
    line_colour = rgb_to_ass(style.get("line_color"), "#FFFFFF")
    word_colour = rgb_to_ass(style.get("word_color"), "#FFE600")
    out = []

    def emit(layer, start, end, body):
        out.append("Dialogue: %d,%s,%s,Default,,0,0,0,,%s%s"
                   % (layer, ass_time(start), ass_time(end), at, body))

    for cue in cues:
        text = _shape(cue.get("text", ""), style)
        if not text.strip():
            continue
        timed = cue.get("words") or []

        if kind == "classic" or not timed:
            chunks = _lines(text.split(), per_line or 0)
            emit(0, cue["start"], cue["end"], "\\N".join(" ".join(c) for c in chunks))
            continue

        timed = [dict(w, word=_shape(w["word"], style)) for w in timed]
        for group in _lines(timed, per_line):
            if not group:
                continue
            first, last = group[0]["start"], group[-1]["end"]
            words = [w["word"] for w in group]

            if kind == "karaoke":
                body = "".join("{\\k%d}%s " % (
                    max(1, int(round((w["end"] - w["start"]) * 100))), w["word"])
                    for w in group).strip()
                emit(0, first, last, "{\\c%s}%s" % (word_colour, body))

            elif kind == "highlight":
                emit(0, first, last, "{\\c%s}%s" % (line_colour, " ".join(words)))
                for i, w in enumerate(group):
                    marked = " ".join(
                        "{\\c%s}%s{\\c%s}" % (word_colour, v, line_colour) if j == i
                        else v for j, v in enumerate(words))
                    emit(1, w["start"], w["end"], "{\\c%s}%s" % (line_colour, marked))

            elif kind == "underline":
                for i, w in enumerate(group):
                    marked = " ".join("{\\u1}%s{\\u0}" % v if j == i else v
                                      for j, v in enumerate(words))
                    emit(0, w["start"], w["end"], "{\\c%s}%s" % (line_colour, marked))

            elif kind == "word_by_word":
                for w in group:
                    emit(0, w["start"], w["end"],
                         "{\\c%s}%s" % (word_colour, w["word"]))
    return "\n".join(out)


def build(cues, style, resolution):
    return header(style, resolution) + events(cues, style, resolution) + "\n"


def fontsdir_arg(style):
    """`fontsdir=...` for the subtitles filter, or "" when the family is a system font.

    Two Windows traps live here. The filter parses its own argument string, so the colon
    in `E:/…` reads as an option separator — and unlike the subtitle filename we cannot
    dodge it with a relative path, because the fonts may sit on another drive. The
    backslash escape alone is silently ignored in an option-value position; it only
    takes effect inside single quotes. Hence both.
    """
    directory = style.get("_font_dir") or ""
    if not directory or not os.path.isdir(directory):
        return ""
    return ":fontsdir='%s'" % directory.replace("\\", "/").replace(":", "\\:")

"""Caption Style — pick the subtitle look once, wire it into the assemble node.

Its own node rather than fifteen more widgets on Video Assemble: the look is a decision
you make once and then leave alone, while the assemble node's own settings change every
run. Keeping them apart also means the style can be previewed, swapped or wired into
more than one output without touching the rest of the graph.

The `font` dropdown is built by reading the family name out of every font file we can
find — the pack's own `fonts/` folder first (that curated set is what ships, so a
workflow saved here opens correctly on someone else's machine), then the system fonts.

Four of the five styles mark individual words, which needs per-word timings. A video
model does not give us any, so they are estimated from the planned line and the clip's
real duration — see `estimate_words` in `ass_captions`. `classic` needs no estimate.
"""

from . import ass_captions as ass

_FAMILIES, _DIRS = ass.families()
_DEFAULT_FONT = ("Oswald" if "Oswald" in _DIRS else
                 "Roboto" if "Roboto" in _DIRS else
                 (_FAMILIES[0] if _FAMILIES else "Arial"))


class ArkCaptionStyle:
    CATEGORY = "arkennemasis/Video"
    FUNCTION = "run"
    RETURN_TYPES = ("ARK_CAPTION_STYLE",)
    RETURN_NAMES = ("caption_style",)
    DESCRIPTION = ("Choose how burned-in subtitles look: one of five styles, any "
                   "installed font, colours, outline, box, size and position. Wire the "
                   "output into Video Assemble's caption_style input.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enabled": ("BOOLEAN", {
                    "default": True, "label_on": "captions ON",
                    "label_off": "captions OFF",
                    "tooltip": "Turn this off and the finished video is burned with no "
                               "subtitles at all. Everything below stays as you set it, "
                               "so switching back on needs one click.",
                }),
                "style": (ass.STYLES, {
                    "default": "highlight",
                    "tooltip": "classic = the whole line at once. karaoke = the colour "
                               "sweeps across the line. highlight = the spoken word "
                               "changes colour. underline = it gets underlined. "
                               "word_by_word = one word on screen at a time.\n\n"
                               "All but classic mark single words, and the word timings "
                               "are estimated from the script, so they track the speech "
                               "closely but are not frame-accurate.",
                }),
                "font": (_FAMILIES or ["Arial"], {
                    "default": _DEFAULT_FONT,
                    "tooltip": "Families bundled with this node pack are listed first; "
                               "everything after them is a font installed on this "
                               "machine and may be missing on someone else's.",
                }),
                "font_size": ("INT", {
                    "default": 0, "min": 0, "max": 400,
                    "tooltip": "0 = scale to the video: 5% of its height "
                               "(37 px at 736p).",
                }),
                "position": (ass.POSITIONS, {"default": "bottom_center"}),
                "alignment": (ass.ALIGNMENTS, {"default": "center"}),
            },
            "optional": {
                "line_color": ("STRING", {
                    "default": "#FFFFFF",
                    "tooltip": "Hex colour of the ordinary words.",
                }),
                "word_color": ("STRING", {
                    "default": "#FFE600",
                    "tooltip": "Hex colour of the word being spoken. Ignored by "
                               "classic, which never marks a single word.",
                }),
                "outline_color": ("STRING", {"default": "#000000"}),
                "outline_width": ("INT", {
                    "default": 3, "min": 0, "max": 20,
                    "tooltip": "The dark rim that keeps text readable over a bright "
                               "frame. 0 removes it — only safe with a shadow or a box.",
                }),
                "shadow_offset": ("INT", {"default": 1, "min": 0, "max": 20}),
                "border_style": (ass.BORDERS, {
                    "default": "outline",
                    "tooltip": "'opaque box' fills a solid card behind the text, using "
                               "outline_color as the fill and outline_width as padding.",
                }),
                "box_color": ("STRING", {"default": "#000000"}),
                "bold": ("BOOLEAN", {"default": True}),
                "italic": ("BOOLEAN", {"default": False}),
                "all_caps": ("BOOLEAN", {"default": False}),
                "max_words_per_line": ("INT", {
                    "default": 6, "min": 0, "max": 40,
                    "tooltip": "Break the line every N words. 0 puts the whole line on "
                               "one row, which overruns the frame on long sentences.",
                }),
                "margin_v": ("INT", {
                    "default": 40, "min": 0, "max": 400,
                    "tooltip": "Distance from the frame edge, in pixels.",
                }),
                "extra_fonts_dir": ("STRING", {
                    "default": "",
                    "tooltip": "Optional folder of extra .ttf/.otf files. Restart "
                               "ComfyUI after adding one for it to reach the dropdown; "
                               "it is picked up for rendering straight away.",
                }),
            },
        }

    def run(self, enabled, style, font, font_size, position, alignment,
            line_color="#FFFFFF", word_color="#FFE600", outline_color="#000000",
            outline_width=3, shadow_offset=1, border_style="outline",
            box_color="#000000", bold=True, italic=False, all_caps=False,
            max_words_per_line=6, margin_v=40, extra_fonts_dir=""):
        if not enabled:
            # Still return a dict rather than None: the assembler reads `enabled` and
            # skips the burn, which keeps every other setting intact for next time.
            print("[arkennemasis] captions OFF — the final video will carry none")
            return ({"enabled": False},)

        extra = (extra_fonts_dir or "").strip()
        directories = dict(_DIRS)
        if extra:
            # Rescan rather than trust the import-time snapshot: the folder may have
            # been created, or filled, after ComfyUI started.
            directories.update(ass.scan_fonts([extra]))

        chosen = directories.get(font)
        if chosen is None:
            print("[arkennemasis] caption font %r not found on this machine — libass "
                  "will fall back to its default. Installed families: %d"
                  % (font, len(directories)))

        settings = {
            "enabled": True,
            "style": style, "font": font, "font_size": int(font_size),
            "position": position, "alignment": alignment,
            "line_color": line_color, "word_color": word_color,
            "outline_color": outline_color, "outline_width": int(outline_width),
            "shadow_offset": int(shadow_offset), "border_style": border_style,
            "box_color": box_color, "bold": bool(bold), "italic": bool(italic),
            "all_caps": bool(all_caps),
            "max_words_per_line": int(max_words_per_line),
            "margin_v": int(margin_v), "margin_h": 40,
            "_font_dir": chosen or "",
        }
        print("[arkennemasis] caption style: %s / %s%s, %s"
              % (style, font, "" if chosen else " (MISSING)", position))
        return (settings,)


NODE_CLASS_MAPPINGS = {
    "ArkCaptionStyle": ArkCaptionStyle,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArkCaptionStyle": "arkennemasis Caption Style (font + subtitle style)",
}

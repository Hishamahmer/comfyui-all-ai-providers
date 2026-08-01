"""One place to say who the subject is, for a whole graph of prompts.

Character prompts should stay gender-neutral ("the person ...") so the same 24 shots work
for anybody. The gender then belongs in exactly one field, wired to every prompt, instead
of being hardcoded into each one - which is how a leftover "A woman" ends up on a set of
photos of a man.

Outputs a single line, e.g.::

    Subject: a man in his late twenties, athletic build.

Leave ``gender`` on "let the reference decide" when the reference photo is unambiguous and
you would rather not state it at all.
"""

INHERIT = "let the reference decide"

GENDERS = [
    INHERIT,
    "a man",
    "a woman",
    "a non-binary person",
    "a boy",
    "a girl",
]


class ArkSubjectLine:
    CATEGORY = "arkennemasis/Utility"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("subject",)
    DESCRIPTION = ("Build one 'Subject: ...' line from a gender choice plus free-text "
                   "notes, and wire it into every prompt. Keeps the prompts themselves "
                   "gender-neutral.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "gender": (GENDERS, {
                    "tooltip": "Stated once and applied to every prompt wired to this "
                               "node. 'let the reference decide' omits it entirely.",
                }),
                "notes": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "Optional, e.g. 'in his late twenties, athletic build'. "
                               "Describe age/build/ethnicity here, NOT hair or eye "
                               "colour - for a LoRA, captioned features stop being "
                               "learned.",
                }),
                "prefix": ("STRING", {
                    "default": "Subject:",
                    "tooltip": "Label in front of the line. Blank for none.",
                }),
            },
        }

    def run(self, gender, notes="", prefix="Subject:"):
        parts = []
        if gender != INHERIT:
            parts.append(gender.strip())
        note = (notes or "").strip().rstrip(".,")
        if note:
            parts.append(note)
        if not parts:
            return ("",)                     # nothing to say: add nothing to the prompt

        body = " ".join(parts)
        # "a man" + "in his 20s" reads as one phrase; a bare note stands on its own.
        line = body if body.endswith(".") else body + "."
        head = (prefix or "").strip()
        return (("%s %s" % (head, line)).strip() if head else line,)


NODE_CLASS_MAPPINGS = {
    "ArkSubjectLine": ArkSubjectLine,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArkSubjectLine": "arkennemasis Subject Line (gender + notes)",
}

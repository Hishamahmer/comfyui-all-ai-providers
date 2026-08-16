"""The generative variation flow: instructions -> prompts -> images.

    INPUTS            base product image + reference images + free-text notes
      |
    STEP 1  1A base system instructions (fixed template)
            1B an LLM turns them, plus the product, into CUSTOMISED system
               instructions for THIS product
      |
    STEP 2  2A an LLM uses those instructions to write one prompt per variation
            2B the individual prompts
      |
    STEP 3  for each prompt: base image + prompt + reference -> the image model
      |
    OUTPUTS one generated image per variation

Three nodes here, one per box the pipeline was missing:

    ArkInstructionBrief   1A — the fixed meta-prompt, product-neutral
    ArkPromptRequest      2A — asks for all N prompts in ONE call, as JSON
    ArkPromptAt           2B — pulls variation i's prompt out of that answer

**All N prompts come from a single LLM call, not N calls.** The set is what the model is
asked for, so it can see every variation at once and keep them parallel in structure —
and it costs one request instead of N. `ArkCodexLLM`'s `batch_total`/`batch_size` split
that into several short calls when N is large, which is about staying under the output
cap, not about asking N separate questions.

**The identity locks are still appended verbatim to every prompt.** The model writes the
part that must vary; the part that must not vary is a constant. That keeps what the
diagram asks for — genuinely authored per-variation prompts — without reintroducing the
drift that comes from re-authoring the fidelity clauses N times.
"""

from __future__ import annotations

import json

from .schema import ValidationError, canonical


def colour_phrase(hex_value):
    """`HEX #F4D94D / RGB 244, 217, 77` — a colour stated two ways.

    A hex code is an exact, unambiguous target, which is exactly why it belongs in the
    prompt TEXT rather than being handed over as a picture. Attaching a flat colour chip
    as a reference image tells an image-editing model "make this region look like this",
    and what it looks like is a flat fill — so a material that should keep its own
    structure comes back as paint.

    Both notations go in because they are read differently: the hex is the literal
    specification a client signed off, and the RGB triple is the form more image models
    have actually seen written next to colours in training text. Stating both costs a
    dozen characters and removes an entire class of misread.
    """
    from .colour import hex_to_rgb01
    try:
        rgb = hex_to_rgb01(hex_value)
    except ValueError:
        return str(hex_value)
    return "HEX %s / RGB %d, %d, %d" % (
        str(hex_value).upper(),
        round(rgb[0] * 255), round(rgb[1] * 255), round(rgb[2] * 255))

# ── 1A · the fixed base system instructions ─────────────────────────────────
# Product-neutral: it tells the model how to write instructions, never what the object
# is. Everything specific arrives as data.

BASE_INSTRUCTIONS = """\
You are writing the system instructions for an automated product-variation image
pipeline.

You will be given: one base photograph of a product, optionally some reference
images, optionally some notes from the operator, and the list of variations that
must be produced.

Your job is NOT to write the variation prompts. Your job is to write the system
instructions that a second model will follow when it writes those prompts.

Look at the photograph and write instructions that cover:

1. WHAT THE PRODUCT IS. Describe the object as it appears: its form, its parts, how
   the parts relate in size and position, and any distinctive detail. Describe only
   what you can see. Do not speculate about brand, price, quality or purpose.

2. WHICH PART EACH AXIS CHANGES. For every variation axis in the supplied list, say
   exactly which visible part of the product that axis alters, and — just as
   importantly — state that everything else stays untouched.

3. HOW THE CHANGED PART SHOULD BE DESCRIBED. If the variations differ in colour, say
   how colour should be specified. If they differ in material, pattern, texture or
   finish, say what has to be described for the result to read as that material:
   its structure, its scale, how light passes through or reflects off it. This is
   the section that decides whether the output looks like a real variant or a flat
   recolour, so be concrete.

4. WHAT MUST NEVER CHANGE. The camera, the framing, the crop, the background, the
   lighting, the shadow, and every part of the product that no axis targets.

5. HOUSE STYLE. The tone and structure each generated prompt should follow, so that
   all of them read as one set.

Write the instructions as clear prose addressed to the prompt-writing model. Do not
write any variation prompts yourself. Do not output JSON. Do not add commentary
before or after the instructions.
"""

# ── 2A · asking for the whole prompt set in one call ────────────────────────

PROMPT_REQUEST = """\
Write the image-generation prompts for the variations listed below.

Return ONLY a JSON array. Each element must be an object:

    {"n": <the variation number, an integer>,
     "key": "<the key exactly as given>",
     "prompt": "<the prompt for this variation>"}

Return exactly %(count)d objects, one for every variation listed, in the order given.

Rules for the prompts themselves:

* Each prompt describes ONE variation of the SAME product shown in the base image.
* Describe the target material or colour for the part that changes, in the terms the
  system instructions above call for. Where a variation supplies a reference image,
  write the prompt so the attached reference is what defines the appearance.
* Keep every prompt parallel in structure and length. They are a set: the only thing
  that should differ between two of them is the variation itself.
* Do not mention the camera, the background, the lighting or the framing. Fidelity
  clauses are appended automatically and must not be duplicated or paraphrased.
* Do not include the filename, the SKU, or any JSON other than the array itself.

THE VARIATIONS:

%(rows)s
"""


class ArkInstructionBrief:
    CATEGORY = "arkennemasis/Variation"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("system_instructions", "product_brief")
    DESCRIPTION = (
        "STEP 1A — the fixed base system instructions, plus a brief describing this "
        "product's axes. Feed both into an LLM together with the base photograph; what "
        "comes back is STEP 1B, the customised system instructions for this product."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "intake_json": ("STRING", {
                    "multiline": True, "default": "{}", "forceInput": True,
                }),
            },
            "optional": {
                "base_instructions": ("STRING", {
                    "multiline": True, "default": BASE_INSTRUCTIONS,
                    "tooltip": "1A. Product-neutral by design — it says how to write "
                               "instructions, never what the object is.",
                }),
                "operator_notes": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "Free-text direction for this product: 'the glass is "
                               "crushed, keep the crinkle structure', 'matte not "
                               "glossy'. Anything the photograph cannot say for itself.",
                }),
                "regions": ("STRING", {
                    "default": "", "forceInput": True,
                    "tooltip": "Optional. The plate's real region names, so the model "
                               "refers to parts by the names the rest of the pipeline "
                               "uses.",
                }),
            },
        }

    def run(self, intake_json, base_instructions=BASE_INSTRUCTIONS, operator_notes="",
            regions=""):
        intake = json.loads(intake_json or "{}")
        product = intake.get("product") or {}
        specs = intake.get("specs") or []

        by_axis = {}
        for entry in specs:
            by_axis.setdefault(canonical(entry.get("axis")), []).append(entry)

        lines = [
            "PRODUCT: %s" % (product.get("display_name") or product.get("product")),
            "TOTAL VARIATIONS: %s" % intake.get("cell_count"),
            "",
            "VARIATION AXES:",
        ]
        for axis, entries in sorted(by_axis.items()):
            kinds = sorted({("reference image" if e.get("refs") else
                             "hex colour" if e.get("hex") else "name only")
                            for e in entries})
            lines.append("")
            lines.append("  AXIS '%s' — %d values, given as %s"
                         % (axis, len(entries), " and ".join(kinds)))
            for entry in sorted(entries, key=lambda e: e.get("value", "")):
                bits = ["    - %s" % entry.get("value")]
                if entry.get("display") and entry["display"] != entry.get("value"):
                    bits.append("(%s)" % entry["display"])
                if entry.get("hex"):
                    bits.append("hex %s" % entry["hex"])
                if entry.get("refs"):
                    bits.append("+ reference image")
                if entry.get("description"):
                    bits.append("— %s" % entry["description"])
                lines.append(" ".join(bits))

        if str(regions or "").strip():
            lines += ["", "THE PRODUCT'S NAMED PARTS: %s" % regions.strip(),
                      "Refer to parts by these names."]
        if str(operator_notes or "").strip():
            lines += ["", "OPERATOR NOTES — these take priority over your own reading "
                          "of the photograph:", str(operator_notes).strip()]

        return (str(base_instructions), "\n".join(lines))


class ArkPromptRequest:
    CATEGORY = "arkennemasis/Variation"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("request", "count")
    DESCRIPTION = (
        "STEP 2A — build the single request that asks for every variation's prompt at "
        "once, as a JSON array. Wire it into an LLM with json_only ON, together with "
        "the customised system instructions from step 1B."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cells_json": ("STRING", {
                    "multiline": True, "default": "[]", "forceInput": True,
                }),
            },
            "optional": {
                "template": ("STRING", {
                    "multiline": True, "default": PROMPT_REQUEST,
                    "tooltip": "Uses %(count)d and %(rows)s.",
                }),
            },
        }

    def run(self, cells_json, template=PROMPT_REQUEST):
        cells = json.loads(cells_json or "[]")
        if not cells:
            raise ValidationError("ArkPromptRequest: the cell list is empty.")

        rows = []
        for index, cell in enumerate(cells, start=1):
            parts = []
            for slot in sorted(cell.get("slots") or [],
                               key=lambda s: s.get("order") or 99):
                bit = "%s = %s" % (slot.get("axis"), slot.get("display")
                                   or slot.get("value"))
                if slot.get("region"):
                    bit += " (changes the %s)" % slot["region"]
                if slot.get("hex"):
                    # Stated numerically and twice — this is the exact target the
                    # client specified, and it must appear in the prompt because no
                    # colour chip is attached for it.
                    bit += " — colour %s. Write this exact value into the prompt." \
                        % colour_phrase(slot["hex"])
                if slot.get("description"):
                    bit += " — %s" % slot["description"]
                references = slot.get("refs") or ([{"role": "material"}]
                                                  if slot.get("ref") else [])
                if references:
                    roles = ", ".join(r.get("role", "material") for r in references)
                    bit += (" — %d reference image(s) attached (%s). Write the prompt so "
                            "the ATTACHED REFERENCE defines this material's appearance, "
                            "and describe its structure, scale and finish rather than "
                            "only its colour." % (len(references), roles))
                parts.append(bit)
            rows.append("%d. key: %s\n   %s"
                        % (index, cell.get("key"), "\n   ".join(parts)))

        request = template % {"count": len(cells), "rows": "\n\n".join(rows)}
        print("[arkennemasis] prompt request: %d variations in one call" % len(cells))
        return (request, len(cells))


class ArkPromptAt:
    CATEGORY = "arkennemasis/Variation"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("prompt", "key", "report")
    DESCRIPTION = (
        "STEP 2B — one variation's prompt, out of the set the LLM returned. Appends the "
        "identity and scene locks verbatim, so the authored part varies and the "
        "fidelity part cannot."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompts_json": ("STRING", {
                    "multiline": True, "default": "[]", "forceInput": True,
                    "tooltip": "The LLM's answer from step 2A.",
                }),
                "cell_json": ("STRING", {
                    "multiline": True, "default": "{}", "forceInput": True,
                }),
                "index": ("INT", {"default": 0, "min": 0, "max": 100000}),
            },
            "optional": {
                "recipe_json": ("STRING", {
                    "multiline": True, "default": "", "forceInput": True,
                    "tooltip": "Supplies the two locks. Without it the authored prompt "
                               "is sent alone and identity is not anchored.",
                }),
                "plate_lock_json": ("STRING", {
                    "multiline": True, "default": "", "forceInput": True,
                    "tooltip": "Wire this. The hosted image backend IGNORES the tool's "
                               "size field, so the ONLY way to control the output shape "
                               "is to state it in the prompt — and the shape that is "
                               "wanted is always the locked plate's. Without it the "
                               "model picks its own canvas and reframes the product.",
                }),
                "append_locks": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Append the identity and scene locks verbatim. Off only "
                               "to see what the model wrote on its own.",
                }),
                "state_dimensions": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Append the plate's exact pixel size and aspect to every "
                               "prompt. Requires plate_lock_json.",
                }),
            },
        }

    @classmethod
    def IS_CHANGED(cls, prompts_json="", index=0, **kwargs):
        return "%s|%s" % (hash(str(prompts_json)), index)

    @staticmethod
    def _dimension_line(plate_lock_json):
        """State the plate's exact canvas in words, because `size` is ignored.

        Verified against this backend: asking the image tool for 1024x1536 and 1536x1024
        both came back 1254x1254, while stating the shape in the PROMPT returned both
        exactly. So the output shape is a sentence, not a parameter — and the shape that
        is always wanted is the locked plate's, since every check downstream compares
        the two frame to frame.
        """
        if not str(plate_lock_json or "").strip():
            return ""
        try:
            lock = json.loads(plate_lock_json)
        except ValueError:
            return ""
        width, height = int(lock.get("width") or 0), int(lock.get("height") or 0)
        if width < 2 or height < 2:
            return ""

        from fractions import Fraction
        ratio = Fraction(width, height).limit_denominator(32)
        shape = ("a SQUARE image" if width == height else
                 "a LANDSCAPE image, wider than it is tall" if width > height else
                 "a PORTRAIT image, taller than it is wide")
        line = ("Output format: %s of exactly %d x %d pixels, %d:%d aspect ratio — the "
                "identical canvas, framing and crop as the input image. Do not change "
                "the canvas shape, do not pad or letterbox it, and do not extend the "
                "scene beyond what the input image shows."
                % (shape, width, height, ratio.numerator, ratio.denominator))

        # Where the subject SITS, as measured numbers rather than an adjective.
        #
        # "Keep the subject's scale and position within the frame" is already in the
        # scene lock and the model still re-composed the shot: it shrank the product,
        # moved it down, and grew the ceiling. Prose describing a constraint is weak
        # against a generator; a figure it can check itself against is much stronger,
        # and these figures are already measured by ArkPlateLock — they were simply
        # never passed on.
        box = lock.get("subject_bbox")
        area = lock.get("subject_area_fraction")
        if box and len(box) == 4 and width and height:
            x, y, w, h = box
            left, right = 100.0 * x / width, 100.0 * (x + w) / width
            top, bottom = 100.0 * y / height, 100.0 * (y + h) / height
            line += (
                " The product itself must occupy the SAME footprint as in the input "
                "image: its bounding box spans %.0f%%-%.0f%% of the image width and "
                "%.0f%%-%.0f%% of the image height, measured from the top-left corner"
                % (left, right, top, bottom))
            if area:
                line += ", covering %.1f%% of the total image area" % (100.0 * area)
            line += (". Reproduce that placement exactly. Do not enlarge or shrink the "
                     "product, do not re-centre it, do not move it up or down, and do "
                     "not enlarge the ceiling, wall or floor area around it. This is an "
                     "edit of the supplied photograph, not a new photograph of the same "
                     "object.")
        return line

    def run(self, prompts_json, cell_json, index=0, recipe_json="",
            plate_lock_json="", append_locks=True, state_dimensions=True):
        cell = json.loads(cell_json or "{}")
        key = cell.get("key") or ""

        text = str(prompts_json or "").strip()
        if text.startswith("```"):
            body = text[3:]
            newline = body.find("\n")
            if newline != -1 and body[:newline].strip().isalpha():
                body = body[newline + 1:]
            text = body.rstrip()[:-3].strip() if body.rstrip().endswith("```") else body

        try:
            data = json.loads(text)
        except ValueError as exc:
            raise ValidationError(
                "The prompt-generating model did not return valid JSON (%s). Turn "
                "json_only ON for that LLM node. First 300 characters: %s"
                % (exc, text[:300]))
        if isinstance(data, dict):
            for candidate in ("prompts", "variations", "items", "output"):
                if isinstance(data.get(candidate), list):
                    data = data[candidate]
                    break
        if not isinstance(data, list) or not data:
            raise ValidationError("Step 2A did not return a JSON array of prompts.")

        # Match on the key first. The model is asked for the array in order, but a
        # single dropped or reordered element would otherwise silently pair every later
        # prompt with the wrong variation — and the images would all look plausible.
        chosen = None
        if key:
            for item in data:
                if isinstance(item, dict) and str(item.get("key") or "").strip() == key:
                    chosen = item
                    break
        matched_by = "key"
        if chosen is None:
            position = max(0, min(int(index), len(data) - 1))
            chosen = data[position]
            matched_by = "position %d (NO key match — check the model echoed the keys)" % position

        authored = ""
        if isinstance(chosen, dict):
            authored = str(chosen.get("prompt") or chosen.get("text") or "").strip()
        elif isinstance(chosen, str):
            authored = chosen.strip()
        if not authored:
            raise ValidationError("Variation '%s' has no prompt text in the answer." % key)

        final = authored
        locks = []
        if state_dimensions:
            line = self._dimension_line(plate_lock_json)
            if line:
                locks.append(line)
        if append_locks and str(recipe_json or "").strip():
            recipe = json.loads(recipe_json)
            invariants = recipe.get("invariants") or []
            if invariants:
                locks.append("The object shown has these fixed identifying features, "
                             "which must be true of the output: "
                             + "; ".join(str(i).strip().rstrip(".") for i in invariants)
                             + ".")
            for field in ("lock_product", "lock_scene"):
                value = str(recipe.get(field) or "").strip()
                if value:
                    locks.append(value)
        if locks:
            final = authored + " " + " ".join(locks)

        report = "\n".join([
            "PROMPT %d — %s" % (index + 1, key),
            "  matched by  : %s" % matched_by,
            "  authored    : %d chars" % len(authored),
            "  locks added : %d (%d chars)" % (len(locks), len(final) - len(authored)),
            "",
            "AUTHORED BY THE MODEL:",
            authored,
        ])
        print("[arkennemasis] prompt %d/%d for %s (%d chars authored)"
              % (index + 1, len(data), key, len(authored)))
        return (final, key, report)


NODE_CLASS_MAPPINGS = {
    "ArkInstructionBrief": ArkInstructionBrief,
    "ArkPromptRequest": ArkPromptRequest,
    "ArkPromptAt": ArkPromptAt,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArkInstructionBrief": "arkennemasis Instruction Brief (1A · base system prompt)",
    "ArkPromptRequest": "arkennemasis Prompt Request (2A · ask for all N prompts)",
    "ArkPromptAt": "arkennemasis Prompt At (2B · one variation's prompt)",
}

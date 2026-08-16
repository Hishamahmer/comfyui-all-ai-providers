"""Stage 5 — build one prompt by substitution. No model, no variance, no cost.

This is the one required change to the operator's original design, and it is the whole
reason the set stays consistent.

In the original, a language model authored each of the N prompts individually. That must
not be implemented. Independently authored prompts differ in incidental wording — one
says "softly illuminated", the next "warm glow", a third mentions the surface the object
rests on. Those incidental differences move the camera. Prompt variance is the mechanism
by which drift enters a set, and the drift already visible in the operator's existing
manual output is consistent with exactly this cause.

The required property is:

    same prompt except the slot  ->  same image except the slot

Free-form Tier 2 destroys it. Substitution preserves it, costs nothing, and adds no
latency.

Assembly order, per the spec, with only element [1] varying between cells:

    [1] the change instruction   <- template + slot   (VARIES)
    [2] the product invariants   <- recipe            (constant)
    [3] Lock A, product identity <- recipe            (constant)
    [4] Lock B, scene continuity <- recipe            (constant)

The node emits a `constant_hash` alongside the prompt: the hash of elements 2-4 alone.
Every cell in a run must produce the same value. Checking it costs nothing and catches
an entire class of silent drift before a single image is generated — which is why
`ArkPromptAudit` exists to collect them and fail loudly on a mismatch.
"""

from __future__ import annotations

import json
import os

from .schema import ValidationError, sha256_text

# A template may legitimately mention any of these. Anything else in braces is a typo,
# and a typo left in place ships literal `{regoin}` text to the generator on every cell.
KNOWN_SLOTS = ("region", "hex", "description", "display", "value", "axis",
               "filename_token", "product", "product_display")


def _format_template(template, fields):
    """Substitute known slots, and refuse unknown ones rather than crashing per-cell."""
    out = str(template or "")
    unknown = []
    import re

    # Validate every slot FIRST, across the whole template. Dropping sentences before
    # this would hide an unknown slot that happens to sit in a dropped sentence, and a
    # template error must surface on the first cell rather than on whichever cell later
    # happens to populate that field.
    for token in set(re.findall(r"\{([^}]+)\}", out)):
        if token not in fields:
            unknown.append(token)

    if not unknown:
        # A sentence whose slot is empty is dropped WHOLE. A description is optional —
        # a client who supplies a hex and no prose has given a complete specification —
        # but the stock hex template ends "The {region} is {description}." and an empty
        # value turned that into "The top-panel is ." and then, once punctuation was
        # tidied, the equally useless "The top-panel." Both are sent verbatim to the
        # image generator, where a fragment is not free: it is noise the model must
        # interpret, on every hex-only value.
        kept = []
        for sentence in re.split(r"(?<=[.!?])\s+", out):
            slots = re.findall(r"\{([^}]+)\}", sentence)
            if slots and any(not str(fields.get(s) or "").strip() for s in slots):
                continue
            kept.append(sentence)
        out = " ".join(kept)

    for token in set(re.findall(r"\{([^}]+)\}", out)):
        if token in fields:
            out = out.replace("{%s}" % token, str(fields[token] if fields[token] is not None else ""))
    if unknown:
        raise ValidationError(
            "Template contains unknown slot(s) %s. Known slots are: %s. Fix the "
            "template in the recipe rather than letting literal braces reach the "
            "generator." % (", ".join("{%s}" % u for u in sorted(unknown)),
                            ", ".join(KNOWN_SLOTS)))
    return _tidy(out)


def _tidy(text):
    """Remove the wreckage an empty substitution leaves behind.

    A description is optional — a client who supplies a hex and no prose is giving a
    complete specification — but the stock hex template ends "The {region} is
    {description}." and an empty value turns that into "The top-panel is .", which is
    sent verbatim to the image generator. It is not merely untidy: a malformed sentence
    in an instruction is noise the model has to interpret, and it appears on every
    hex-only value, which here is 4 of 11.

    Sentences that lost their only substantive content are dropped whole. Everything
    else is left exactly as the template author wrote it.
    """
    import re
    # "The top-panel is ." / "matching ,." — a clause ending in a dangling copula or
    # preposition immediately before its terminator.
    text = re.sub(r"\s*\b(?:is|are|of|to|with|in|as|like|matching)\s*([.,;:])",
                  r"\1", text)
    text = re.sub(r"\s+([.,;:])", r"\1", text)      # space before punctuation
    text = re.sub(r"([.,;:])\1+", r"\1", text)      # doubled terminators
    # A sentence with nothing left in it but punctuation.
    text = re.sub(r"(?<=[.!?])\s*[.,;:]+", "", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def build_prompt(recipe, cell):
    """`(prompt, constant_text, change_text)` for one cell."""
    slots = sorted(cell.get("slots") or [], key=lambda s: s.get("order") or 99)
    if not slots:
        raise ValidationError("Cell %s has no axis slots." % cell.get("key"))

    templates = recipe.get("templates") or {}
    changes = []
    for slot in slots:
        spec_type = slot.get("spec_type") or "reference_image"
        template = templates.get(spec_type)
        if not template:
            raise ValidationError(
                "No template for spec_type '%s' (axis '%s'). Add one to the recipe."
                % (spec_type, slot.get("axis")))
        region = slot.get("region")
        if not region:
            raise ValidationError(
                "Axis '%s' has no region to paint. The recipe gate should have caught "
                "this — an axis pointed at nothing produces an image changed at random."
                % slot.get("axis"))
        changes.append(_format_template(template, {
            "region": region,
            "hex": slot.get("hex") or "",
            "description": slot.get("description") or "",
            "display": slot.get("display") or slot.get("value") or "",
            "value": slot.get("value") or "",
            "axis": slot.get("axis") or "",
            "filename_token": slot.get("filename_token") or "",
            "product": recipe.get("product") or "",
            "product_display": recipe.get("product_display") or "",
        }).strip())

    change_text = " ".join(c for c in changes if c)

    invariants = recipe.get("invariants") or []
    invariant_text = ""
    if invariants:
        invariant_text = ("The object shown has these fixed identifying features, which "
                          "must be true of the output: " +
                          "; ".join(str(i).strip().rstrip(".") for i in invariants) + ".")

    # Regions that NO axis paints must come through untouched, and saying so by name is
    # far stronger than relying on "change only the X" to imply it. It matters most when
    # a product has one varying region and several fixed ones: the fixed ones are then
    # the majority of the object, and the correct amount of change in them is zero — not
    # "within the bleed tolerance", which is what a silence here would settle for.
    #
    # Derived rather than configured: any region the recipe lists that no axis claims is
    # locked by definition, so a new axis automatically stops locking its own region and
    # a removed one automatically starts.
    painted = {str(a.get("paints") or "").strip()
               for a in (recipe.get("axes") or [])}
    locked = [str(r).strip() for r in (recipe.get("regions") or [])
              if str(r).strip() and str(r).strip() not in painted]
    locked_text = ""
    if locked:
        locked_text = (
            "The %s must be reproduced EXACTLY as %s in the input image — identical "
            "colour, identical material, identical texture and identical brightness, "
            "pixel for pixel. %s not being changed on this image; %s part of the "
            "product's fixed identity, and any difference in %s is a defect."
            % (", ".join(locked[:-1]) + " and " + locked[-1] if len(locked) > 1
               else locked[0],
               "they are" if len(locked) > 1 else "it is",
               "They are" if len(locked) > 1 else "It is",
               "they are" if len(locked) > 1 else "it is",
               "them" if len(locked) > 1 else "it"))

    constant_parts = [invariant_text,
                      locked_text,
                      str(recipe.get("lock_product") or "").strip(),
                      str(recipe.get("lock_scene") or "").strip()]
    constant_text = " ".join(p for p in constant_parts if p)

    return (" ".join(p for p in (change_text, constant_text) if p),
            constant_text, change_text)


class ArkPromptBuild:
    CATEGORY = "arkennemasis/Variation"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("prompt", "constant_hash", "prompt_hash", "change_only")
    DESCRIPTION = (
        "Assemble one cell's prompt by pure string substitution: the change "
        "instruction, then the invariants, then both locks — the last three byte-"
        "identical across the whole run. Never invokes a language model."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "recipe_json": ("STRING", {
                    "multiline": True, "default": "{}", "forceInput": True,
                }),
                "cell_json": ("STRING", {
                    "multiline": True, "default": "{}", "forceInput": True,
                }),
            },
            "optional": {
                "extra_constant": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "Optional extra text appended to EVERY prompt, after the "
                               "locks — a backend-specific phrasing that helps this "
                               "particular model. It joins the constant block, so it "
                               "must not mention anything cell-specific.",
                }),
                "audit_dir": ("STRING", {
                    "default": "",
                    "tooltip": "Optional folder. Every built prompt is written there as "
                               "<key>.prompt.txt, and the constant hash is appended to "
                               "constants.log — which is what Prompt Audit reads.",
                }),
            },
        }

    def run(self, recipe_json, cell_json, extra_constant="", audit_dir=""):
        recipe = json.loads(recipe_json or "{}")
        cell = json.loads(cell_json or "{}")

        prompt, constant_text, change_text = build_prompt(recipe, cell)

        extra = str(extra_constant or "").strip()
        if extra:
            prompt = prompt + " " + extra
            constant_text = constant_text + " " + extra

        constant_hash = sha256_text(constant_text)
        prompt_hash = sha256_text(prompt)

        folder = str(audit_dir or "").strip()
        if folder:
            if not os.path.isabs(folder):
                try:
                    import folder_paths
                    folder = os.path.join(folder_paths.get_output_directory(), folder)
                except Exception:
                    folder = os.path.abspath(folder)
            os.makedirs(folder, exist_ok=True)
            key = str(cell.get("key") or "cell").replace("/", "_").replace("\\", "_")
            with open(os.path.join(folder, "%s.prompt.txt" % key), "w",
                      encoding="utf-8") as handle:
                handle.write(prompt + "\n")
            with open(os.path.join(folder, "constants.log"), "a",
                      encoding="utf-8") as handle:
                handle.write("%s\t%s\n" % (constant_hash, cell.get("key")))

        return (prompt, constant_hash, prompt_hash, change_text)


class ArkPromptAudit:
    CATEGORY = "arkennemasis/Variation"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING", "BOOLEAN", "INT")
    RETURN_NAMES = ("report", "all_identical", "distinct_constants")
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Confirm every prompt in the run shares one identical constant block. A single "
        "differing hash means one cell was built against a different lock or a "
        "different invariant list — the exact silent drift substitution exists to "
        "prevent. Costs nothing and catches a whole class of failure before delivery."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audit_dir": ("STRING", {
                    "default": "",
                    "tooltip": "The same folder Prompt Build wrote constants.log into.",
                }),
            },
            "optional": {
                "expected_hash": ("STRING", {
                    "default": "", "forceInput": True,
                    "tooltip": "Optional. A constant_hash from Prompt Build; every line "
                               "in the log must match it.",
                }),
                "strict": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "On: a mismatch raises. That is the right default — the "
                               "images are already paid for by the time you read a "
                               "warning.",
                }),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def run(self, audit_dir, expected_hash="", strict=True):
        folder = str(audit_dir or "").strip()
        if folder and not os.path.isabs(folder):
            try:
                import folder_paths
                folder = os.path.join(folder_paths.get_output_directory(), folder)
            except Exception:
                folder = os.path.abspath(folder)

        path = os.path.join(folder, "constants.log") if folder else ""
        if not path or not os.path.isfile(path):
            return ("No constants.log at %s — set audit_dir on Prompt Build to enable "
                    "the audit." % (path or "(unset)"), True, 0)

        groups = {}
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                digest, _, key = line.strip().partition("\t")
                groups.setdefault(digest, []).append(key)

        lines = ["PROMPT CONSTANT AUDIT",
                 "  log      : %s" % path,
                 "  prompts  : %d" % sum(len(v) for v in groups.values()),
                 "  distinct constant blocks: %d" % len(groups)]
        for digest, keys in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            lines.append("  %s  %4d prompt(s)" % (digest[:26], len(keys)))
            if len(groups) > 1:
                for key in keys[:5]:
                    lines.append("      %s" % key)
                if len(keys) > 5:
                    lines.append("      ... and %d more" % (len(keys) - 5))

        expected = str(expected_hash or "").strip()
        identical = len(groups) <= 1 and (not expected or expected in groups)

        if identical:
            lines += ["", "PASS — every prompt shares one identical constant block."]
        else:
            lines += ["", "FAIL — the constant block is not identical across the run.",
                      "Elements 2-4 (invariants, Lock A, Lock B) must be byte-identical "
                      "for every cell. A differing block means the set will drift."]
            if expected and expected not in groups:
                lines.append("Expected %s, which appears in none of the logged prompts."
                             % expected[:26])

        print("[arkennemasis] prompt audit: %d distinct constant block(s) — %s"
              % (len(groups), "PASS" if identical else "FAIL"))

        if not identical and strict:
            raise ValidationError("\n".join(lines))
        return ("\n".join(lines), identical, len(groups))


NODE_CLASS_MAPPINGS = {
    "ArkPromptBuild": ArkPromptBuild,
    "ArkPromptAudit": ArkPromptAudit,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArkPromptBuild": "arkennemasis Prompt Build (substitution, no LLM)",
    "ArkPromptAudit": "arkennemasis Prompt Audit (assert constants identical)",
}

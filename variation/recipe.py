"""Stages 4 and the human gate — compile one recipe per product, then freeze it.

The recipe is the whole configuration for a product: template, slot table, both locks,
the invariants, and the naming pattern. It is compiled once by a vision model that looks
at the locked plate, read once by a human, and then frozen — and every image in the run
inherits it.

Three nodes, in the order they run:

    ArkRecipeBrief    emits the fixed Tier 0 meta-prompt + the schema  -> feed to an LLM
    ArkRecipeCompile  validates the model's JSON, injects the constants, hashes it
    ArkRecipeGate     the single human checkpoint; blocks the run until approved

**Tier 0 is a compiler, not a pipeline stage.** It runs once per product, not once per
image. At runtime there are only two moving parts: this recipe, and mechanical
substitution per image.

The two locks and the tolerances are deliberately NOT written by the model. A model that
rewrites a lock will occasionally phrase it weakly, and that one cell is the one that
drifts — a failure that is effectively undiscoverable by reading prompts afterwards. The
locks are constants on `ArkRecipeCompile`, appended verbatim to every prompt, and the
compile step strips them out of the model's output if it tried anyway.
"""

from __future__ import annotations

import datetime
import json
import os

from .schema import (
    LLM_FORBIDDEN,
    SCHEMA_VERSION,
    ValidationError,
    canonical,
    normalise_hex,
    recipe_hash,
    spec_type_of,
    validate_recipe,
)

# ── Tier 0: fixed, hand-authored, and product-neutral ────────────────────────
# It discovers what the regions are by LOOKING at the photograph, never by being told
# what the object is. A meta-prompt carrying product vocabulary would quietly become a
# one-product pipeline wearing a configuration file.

try:
    from .prompts_local import META_PROMPT
except ImportError:  # public clone: supply your own wording on the canvas
    META_PROMPT = ""

# ── The two locks (constants, appended verbatim to every prompt) ─────────────

try:
    from .prompts_local import LOCK_PRODUCT
except ImportError:  # public clone: supply your own wording on the canvas
    LOCK_PRODUCT = ""

try:
    from .prompts_local import LOCK_SCENE
except ImportError:  # public clone: supply your own wording on the canvas
    LOCK_SCENE = ""

# The reference-image wording is deliberately blunt about what to TAKE and what to
# IGNORE. "Match the material shown in the attached reference image" reads, to an image
# model, as an invitation to borrow the reference as a whole — its lighting, its framing,
# sometimes its object — and a timber swatch photographed on a workbench then drags the
# workbench in with it. Naming the four properties to lift (colour, grain, texture,
# finish) and then naming what must not follow them is what keeps a material reference a
# material reference.
try:
    from .prompts_local import _TAKE_MATERIAL
except ImportError:  # public clone: supply your own wording on the canvas
    _TAKE_MATERIAL = ""

DEFAULT_TEMPLATES = {
    # The base photograph already shows this finish, so the instruction is to leave it
    # be. Stated positively and at length because "do not change X" alone reads, to an
    # image model in the middle of a list of changes, as a weak preference rather than
    # an exclusion.
    "unchanged":
        "Leave the {region} of the product EXACTLY as it is in the input image. Its "
        "colour, material, grain, texture, finish and every mark on it must come "
        "through completely untouched, pixel for pixel. This part is not being varied "
        "on this image — the photograph already shows the correct finish for it.",
    "reference_image": _TAKE_MATERIAL,
    "hex":
        "Change only the {region} of the product to the colour {hex}. The {region} is "
        "{description}.",
    "resolved_word": _TAKE_MATERIAL,
}


def _strip_fences(text):
    stripped = str(text or "").strip()
    if not stripped.startswith("```"):
        return stripped
    body = stripped[3:]
    newline = body.find("\n")
    if newline != -1 and body[:newline].strip().isalpha():
        body = body[newline + 1:]
    if body.rstrip().endswith("```"):
        body = body.rstrip()[:-3]
    return body.strip()


class ArkRecipeBrief:
    CATEGORY = "arkennemasis/Variation"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("system_instructions", "job_brief")
    DESCRIPTION = (
        "The fixed Tier 0 meta-prompt and the job brief for one product. Wire "
        "system_instructions and job_brief into a vision LLM together with the locked "
        "plate. Never changes between products — that is the point of it."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "intake_json": ("STRING", {
                    "multiline": True, "default": "{}", "forceInput": True,
                    "tooltip": "From Variation Intake.",
                }),
            },
            "optional": {
                "plate_lock_json": ("STRING", {
                    "multiline": True, "default": "", "forceInput": True,
                    "tooltip": "From Plate Lock. Wire this. It tells the model the "
                               "region names that MASKS ACTUALLY EXIST FOR, so it "
                               "assigns axes to those instead of inventing its own "
                               "wording — an invented name silently paints nothing.",
                }),
                "meta_prompt": ("STRING", {
                    "multiline": True, "default": META_PROMPT,
                    "tooltip": "The Tier 0 meta-prompt. Product-neutral by design: it "
                               "discovers regions by looking at the photograph rather "
                               "than being told what the object is. Edit with care.",
                }),
                "extra_notes": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "Optional product notes for the compiler — anything the "
                               "photograph cannot show. Appended to the job brief, "
                               "never to the meta-prompt.",
                }),
            },
        }

    def run(self, intake_json, plate_lock_json="", meta_prompt=META_PROMPT,
            extra_notes=""):
        try:
            intake = json.loads(intake_json or "{}")
        except ValueError as exc:
            raise ValidationError("intake_json is not valid JSON: %s" % exc)

        plate_lock = (json.loads(plate_lock_json)
                      if str(plate_lock_json or "").strip() else {})
        available = sorted((plate_lock.get("regions") or {}).keys())

        product = intake.get("product") or {}
        specs = intake.get("specs") or []

        by_axis = {}
        for entry in specs:
            by_axis.setdefault(canonical(entry.get("axis")), []).append(entry)

        lines = []
        if available:
            # The single most valuable thing this brief can carry. Left to itself the
            # model writes a reasonable-sounding region name of its own — 'upper-glass-
            # panel' where the mask on disk is 'top-panel' — and every lookup then misses,
            # so the axis paints zero pixels and the run completes looking successful.
            lines += [
                "REGION NAMES YOU MUST USE — these are the ONLY regions that have masks:",
                "    " + ", ".join(available),
                "",
                "Use these EXACT strings in \"regions\" and in each axis's \"paints\". "
                "Do not rephrase, pluralise, expand or prettify them. A name that is "
                "not in this list paints nothing at all.",
                "",
            ]
        lines += [
            "PRODUCT: %s" % (product.get("display_name") or product.get("product")),
            "PRODUCT ID: %s" % product.get("product"),
            "PLATES: %s" % ", ".join(str(p.get("id")) for p in (product.get("plates") or [])),
            "NAMING PATTERN (copy this verbatim into \"naming\"): %s"
            % (product.get("naming_pattern") or "(client filenames are used verbatim)"),
            "TOTAL CELLS: %s" % intake.get("cell_count"),
            "",
            "VARIATION AXES AND THEIR VALUES:",
        ]
        for axis, entries in sorted(by_axis.items()):
            kinds = sorted({spec_type_of(e) for e in entries})
            lines.append("")
            lines.append("  AXIS '%s' — %d values, specified by %s"
                         % (axis, len(entries), "/".join(kinds)))
            for entry in sorted(entries, key=lambda e: e.get("value", "")):
                bits = ["    - %s" % entry.get("value")]
                if entry.get("display") and entry["display"] != entry.get("value"):
                    bits.append("(%s)" % entry["display"])
                if entry.get("hex"):
                    bits.append("hex %s" % entry["hex"])
                if entry.get("refs"):
                    roles = ", ".join(r.get("role", "material") for r in entry["refs"])
                    bits.append("reference image(s): %s" % roles)
                if entry.get("description"):
                    bits.append("— %s" % entry["description"])
                lines.append(" ".join(bits))

        lines += ["", "REQUIRED OUTPUT SCHEMA (fill every key except the three named as "
                      "supplied separately):", json.dumps({
                          "schema_version": SCHEMA_VERSION,
                          "product": product.get("product"),
                          "product_display": product.get("display_name"),
                          "plates": product.get("plates"),
                          "regions": ["<discovered from the photograph>"],
                          "axes": [{
                              "name": "<axis name>",
                              "paints": "<one region name, or null>",
                              "spec_type": "hex | reference_image | word",
                              "order": 1,
                              "values": [{"id": "<value id>", "display": "<label>",
                                          "filename_token": "<token>",
                                          "description": "<3-10 words>"}],
                          }],
                          "templates": DEFAULT_TEMPLATES,
                          "invariants": ["<3-8 factual statements about this object>"],
                          "naming": product.get("naming_pattern") or "",
                          "needs_confirmation": [],
                      }, indent=2)]

        if str(extra_notes or "").strip():
            lines += ["", "OPERATOR NOTES:", str(extra_notes).strip()]

        return (str(meta_prompt), "\n".join(lines))


class ArkRecipeCompile:
    CATEGORY = "arkennemasis/Variation"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "BOOLEAN")
    RETURN_NAMES = ("recipe_json", "report", "needs_confirmation", "valid")
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Validate the Tier 0 model's JSON against the recipe schema, merge in the "
        "specifications resolved by the library, inject the two locks and the "
        "tolerances as constants, and hash the result. Validation failure is a hard "
        "error — an invalid recipe silently poisons every image in the run."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_json": ("STRING", {
                    "multiline": True, "default": "{}", "forceInput": True,
                    "tooltip": "The Tier 0 model's raw answer. Code fences are stripped.",
                }),
                "intake_json": ("STRING", {
                    "multiline": True, "default": "{}", "forceInput": True,
                }),
                "recipe_dir": ("STRING", {
                    "default": "variation/recipes",
                    "tooltip": "Where recipe.json is written. Relative paths resolve "
                               "inside ComfyUI's output directory.",
                }),
            },
            "optional": {
                "library_json": ("STRING", {
                    "multiline": True, "default": "", "forceInput": True,
                    "tooltip": "From Spec Library. Supplies each value's cached "
                               "reference path and hash, so the recipe points at frozen "
                               "local files rather than at URLs that can change.",
                }),
                "plate_lock_json": ("STRING", {
                    "multiline": True, "default": "", "forceInput": True,
                    "tooltip": "From Plate Lock. Supplies the frozen plate's hash and "
                               "measurements.",
                }),
                "lock_product": ("STRING", {
                    "multiline": True, "default": LOCK_PRODUCT,
                    "tooltip": "Lock A — product identity. A constant appended verbatim "
                               "to every prompt. The anti-improvement sentences are "
                               "load-bearing: image models silently 'help' by "
                               "straightening and idealising, and each such improvement "
                               "is a catalogue defect.",
                }),
                "lock_scene": ("STRING", {
                    "multiline": True, "default": LOCK_SCENE,
                    "tooltip": "Lock B — scene continuity. A constant appended verbatim "
                               "to every prompt.",
                }),
                "frame_tolerance": ("FLOAT", {
                    "default": 0.02, "min": 0.0, "max": 1.0, "step": 0.001,
                    "tooltip": "Allowed subject box drift as a fraction of frame. "
                               "CALIBRATE this against a labelled set with the "
                               "Calibrate node — a number chosen by feel either passes "
                               "drifted images or fails good ones.",
                }),
                "colour_tolerance_de": ("FLOAT", {
                    "default": 5.0, "min": 0.0, "max": 100.0, "step": 0.1,
                    "tooltip": "Allowed CIEDE2000 difference from the specified hex. "
                               "Commercially this is a contractual number as much as a "
                               "technical one — agree it with the client in advance.",
                }),
                "identity_tolerance": ("FLOAT", {
                    "default": 0.06, "min": 0.0, "max": 1.0, "step": 0.001,
                    "tooltip": "Allowed relative drift in the proportion ratios that "
                               "fingerprint this object. This is the check that catches "
                               "a redesigned product, which colour checking cannot.",
                }),
                "max_retries": ("INT", {
                    "default": 3, "min": 0, "max": 10,
                }),
            },
        }

    def run(self, llm_json, intake_json, recipe_dir="variation/recipes",
            library_json="", plate_lock_json="", lock_product=LOCK_PRODUCT,
            lock_scene=LOCK_SCENE, frame_tolerance=0.02, colour_tolerance_de=5.0,
            identity_tolerance=0.06, max_retries=3):
        try:
            recipe = json.loads(_strip_fences(llm_json))
        except ValueError as exc:
            raise ValidationError(
                "The Tier 0 model did not return valid JSON (%s). Turn json_only on for "
                "the LLM node. First 300 characters: %s"
                % (exc, str(llm_json)[:300]))
        if not isinstance(recipe, dict):
            raise ValidationError("The Tier 0 model returned JSON but not an object.")

        intake = json.loads(intake_json or "{}")
        library = json.loads(library_json) if str(library_json or "").strip() else {}
        plate_lock = json.loads(plate_lock_json) if str(plate_lock_json or "").strip() else {}

        product = intake.get("product") or {}
        stripped = [k for k in LLM_FORBIDDEN if k in recipe]
        for key in LLM_FORBIDDEN:
            recipe.pop(key, None)

        recipe["schema_version"] = SCHEMA_VERSION
        recipe.setdefault("product", product.get("product"))
        recipe.setdefault("product_display", product.get("display_name"))
        recipe.setdefault("templates", dict(DEFAULT_TEMPLATES))
        for key, value in DEFAULT_TEMPLATES.items():
            recipe["templates"].setdefault(key, value)

        # The naming pattern is never invented. The client already decided what each
        # file is called, and the intake's value wins over anything the model wrote.
        if product.get("naming_pattern"):
            recipe["naming"] = product["naming_pattern"]

        if plate_lock:
            recipe["plates"] = [{
                "id": plate_lock.get("plate_id"),
                "file": plate_lock.get("file"),
                "hash": plate_lock.get("hash"),
                "width": plate_lock.get("width"),
                "height": plate_lock.get("height"),
            }]
            recipe["plate_lock"] = {
                "ratios": plate_lock.get("ratios") or {},
                "subject_bbox": plate_lock.get("subject_bbox"),
                "regions": {k: v.get("bbox") for k, v in
                            (plate_lock.get("regions") or {}).items()},
                "colour_profile": plate_lock.get("colour_profile"),
                "lock_hash": plate_lock.get("lock_hash"),
            }
            recipe.setdefault("colour_profile", plate_lock.get("colour_profile"))
        elif not recipe.get("plates"):
            recipe["plates"] = product.get("plates") or [{"id": "front", "file": ""}]

        # Merge the resolved specifications into the axis values. The model wrote the
        # descriptions and the ordering; the library holds what the material actually
        # IS. Neither is authoritative about the other's half.
        values_index = (library.get("values") or {})
        intake_specs = {}
        for entry in intake.get("specs") or []:
            intake_specs[(canonical(entry.get("axis")), canonical(entry.get("value")))] = entry

        merged_refs = 0
        for axis in recipe.get("axes") or []:
            axis_name = canonical(axis.get("name"))
            axis["name"] = axis_name
            for value in axis.get("values") or []:
                value_id = canonical(value.get("id"))
                value["id"] = value_id
                key = "%s/%s" % (axis_name, value_id)
                resolved = values_index.get(key) or {}
                source = intake_specs.get((axis_name, value_id)) or {}

                if resolved.get("hex") or source.get("hex"):
                    value["hex"] = normalise_hex(resolved.get("hex") or source.get("hex"))
                token = (resolved.get("filename_token") or source.get("filename_token")
                         or value.get("filename_token") or value_id)
                value["filename_token"] = canonical(token) or value_id
                if not value.get("display"):
                    value["display"] = (resolved.get("display") or source.get("display")
                                        or value_id)

                references = resolved.get("refs") or []
                if references:
                    value["ref"] = references[0].get("path")
                    value["ref_hash"] = references[0].get("hash")
                    value["refs"] = references
                    merged_refs += len(references)
                if resolved.get("unchanged") or source.get("unchanged"):
                    value["unchanged"] = True
                if resolved.get("swatch"):
                    value["swatch"] = resolved["swatch"]
                if not value.get("description"):
                    value["description"] = (resolved.get("description")
                                            or source.get("description") or "")

            # spec_type is decided from what the library actually resolved, not from the
            # model's reading of the brief. The data on disk is the ground truth.
            #
            # It is stored PER VALUE, because a real client sheet mixes formats within a
            # single axis: a supplier sends a hex for one finish and a photograph of a
            # sample for the next. Holding one type for the whole axis forced a choice
            # between two wrong answers — call the axis `hex` and the referenced values
            # lose their picture, or call it `reference_image` and every hex-only value
            # fails validation for having no ref. The axis keeps a summary type for
            # display, but nothing downstream should decide anything from it.
            kinds = set()
            for value in axis.get("values") or []:
                if value.get("unchanged"):
                    value["spec_type"] = "unchanged"
                    kinds.add("unchanged")
                    continue
                if value.get("refs") or value.get("ref"):
                    value["spec_type"] = "reference_image"
                elif value.get("hex"):
                    value["spec_type"] = "hex"
                else:
                    value["spec_type"] = "word"
                kinds.add(value["spec_type"])
            if kinds:
                axis["spec_types"] = sorted(kinds)
                axis["spec_type"] = ("reference_image" if "reference_image" in kinds
                                     else "hex" if "hex" in kinds else "word")

        recipe["lock_product"] = str(lock_product or "").strip()
        recipe["lock_scene"] = str(lock_scene or "").strip()
        recipe["verification"] = {
            "frame_tolerance": float(frame_tolerance),
            "colour_tolerance_de": float(colour_tolerance_de),
            "identity_tolerance": float(identity_tolerance),
            "max_retries": int(max_retries),
            "calibrated": False,
        }
        recipe["cells"] = intake.get("cell_count")
        recipe["compiled_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        recipe.setdefault("approved_by", None)
        recipe.setdefault("needs_confirmation", [])

        problems = validate_recipe(recipe)

        # Cross-check every axis against the regions the PLATE actually has masks for.
        # `validate_recipe` only knows the recipe, so it can confirm an axis paints a
        # region the recipe lists — it cannot know that no mask of that name exists on
        # disk. That gap is how an axis ends up painting zero pixels while the run
        # reports success, which is the exact waste the human gate exists to prevent.
        available = sorted((plate_lock.get("regions") or {}).keys())
        if available:
            for axis in recipe.get("axes") or []:
                paints = canonical(axis.get("paints"))
                if not paints:
                    continue
                if paints not in available:
                    close = [r for r in available
                             if paints in r or r in paints
                             or paints.split("-")[0] == r.split("-")[0]]
                    problems.append(
                        "Axis '%s' paints region '%s', but the locked plate has no mask "
                        "of that name. It has: %s.%s Every cell on this axis would "
                        "change nothing at all."
                        % (axis.get("name"), paints, ", ".join(available),
                           (" Did you mean '%s'?" % close[0]) if close else ""))

        recipe["recipe_hash"] = recipe_hash(recipe)

        target = str(recipe_dir or "").strip() or "variation/recipes"
        if not os.path.isabs(target):
            try:
                import folder_paths
                target = os.path.join(folder_paths.get_output_directory(), target)
            except Exception:
                target = os.path.abspath(target)
        os.makedirs(target, exist_ok=True)
        path = os.path.join(target, "%s.recipe.json" % (recipe.get("product") or "product"))
        # Stamped so ArkRecipeGate writes the approval back to this exact file rather
        # than re-deriving a default path that may not be where this recipe landed.
        recipe["_path"] = path
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(recipe, handle, indent=2, ensure_ascii=False)

        unconfirmed = list(recipe.get("needs_confirmation") or [])
        for axis in recipe.get("axes") or []:
            if axis.get("paints") in (None, "", "null") and axis.get("name") not in unconfirmed:
                unconfirmed.append(axis.get("name"))

        lines = [
            "RECIPE COMPILED — %s" % (recipe.get("product_display") or recipe.get("product")),
            "  file      : %s" % path,
            "  hash      : %s" % recipe["recipe_hash"],
            "  regions   : %s" % ", ".join(recipe.get("regions") or []),
            "  cells     : %s" % recipe.get("cells"),
            "  naming    : %s" % recipe.get("naming"),
            "  references merged from the library: %d" % merged_refs,
        ]
        if stripped:
            lines.append("  NOTE: discarded model-written %s — those are constants."
                         % ", ".join(stripped))
        lines += ["", "AXES"]
        for axis in sorted(recipe.get("axes") or [], key=lambda a: a.get("order") or 99):
            lines.append("  %-18s paints %-14s %-16s %d values  (order %s)"
                         % (axis.get("name"), axis.get("paints") or "??",
                            axis.get("spec_type"), len(axis.get("values") or []),
                            axis.get("order")))
        lines += ["", "INVARIANTS — the per-product identity anchor"]
        for item in recipe.get("invariants") or []:
            lines.append("  - %s" % item)
        lines += ["", "VERIFICATION TOLERANCES (uncalibrated until the Calibrate node "
                      "has run against a labelled set)"]
        for key, value in sorted(recipe["verification"].items()):
            lines.append("  %-22s %s" % (key, value))

        if unconfirmed:
            lines += ["", "NEEDS CONFIRMATION (%d)" % len(unconfirmed)]
            lines += ["  - axis '%s' has no confident region assignment" % a
                      for a in unconfirmed]
        if problems:
            lines += ["", "VALIDATION PROBLEMS (%d)" % len(problems)]
            lines += ["  %d. %s" % (i, p) for i, p in enumerate(problems, start=1)]

        print("[arkennemasis] recipe: %s (%s), %d problems"
              % (recipe.get("product"), recipe["recipe_hash"][:19], len(problems)))

        if problems:
            raise ValidationError(
                "The compiled recipe is invalid (%d problem(s)). One incorrect field "
                "silently poisons every image in the run.\n\n%s"
                % (len(problems), "\n".join("  - " + p for p in problems)))

        return (json.dumps(recipe, ensure_ascii=False), "\n".join(lines),
                "\n".join(unconfirmed), True)


class ArkRecipeGate:
    CATEGORY = "arkennemasis/Variation"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("recipe_json", "summary")
    DESCRIPTION = (
        "The one human checkpoint before generation. Read the recipe once — about a "
        "minute — and type your name to approve it. Blocks everything downstream until "
        "you do. Reviewing 1 recipe replaces reviewing N prompts; reviewing nothing "
        "lets one wrong word poison the whole run."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "recipe_json": ("STRING", {
                    "multiline": True, "default": "{}", "forceInput": True,
                }),
                "approved_by": ("STRING", {
                    "default": "",
                    "tooltip": "Your name or initials. Non-empty = approved. This is the "
                               "gate: leave it blank and nothing downstream runs.",
                }),
            },
            "optional": {
                "confirm_paints": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "Resolve any axis the model was unsure about, one per "
                               "line as 'axis = region'. This is what the gate is really "
                               "protecting: an axis pointed at the wrong region applies "
                               "the wrong material to the wrong part of every image.",
                }),
                "write_back": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Write the approval into the recipe file on disk, so the "
                               "orchestrator can see the gate was passed.",
                }),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def run(self, recipe_json, approved_by="", confirm_paints="", write_back=True):
        try:
            from comfy_execution.graph_utils import ExecutionBlocker
        except ImportError:
            from execution import ExecutionBlocker

        recipe = json.loads(recipe_json or "{}")

        for line in str(confirm_paints or "").splitlines():
            if "=" not in line:
                continue
            axis_name, _, region = line.partition("=")
            axis_name, region = canonical(axis_name), canonical(region)
            for axis in recipe.get("axes") or []:
                if canonical(axis.get("name")) == axis_name:
                    axis["paints"] = region
                    print("[arkennemasis] gate: axis '%s' paints '%s' (operator)"
                          % (axis_name, region))

        unresolved = [a.get("name") for a in (recipe.get("axes") or [])
                      if a.get("paints") in (None, "", "null")]

        lines = [
            "RECIPE FOR APPROVAL — %s"
            % (recipe.get("product_display") or recipe.get("product")),
            "  hash    : %s" % recipe.get("recipe_hash"),
            "  cells   : %s" % recipe.get("cells"),
            "  naming  : %s" % recipe.get("naming"),
            "",
            "EACH AXIS PAINTS ONE REGION — this is the field to check",
        ]
        for axis in sorted(recipe.get("axes") or [], key=lambda a: a.get("order") or 99):
            lines.append("  %-18s ->  %-16s (%s, %d values)"
                         % (axis.get("name"), axis.get("paints") or "*** UNRESOLVED ***",
                            axis.get("spec_type"), len(axis.get("values") or [])))
        lines += ["", "INVARIANTS — a wrong one here is obvious on sight"]
        for item in recipe.get("invariants") or []:
            lines.append("  - %s" % item)

        if unresolved:
            lines += ["", "*** %d AXIS/AXES UNRESOLVED: %s"
                      % (len(unresolved), ", ".join(map(str, unresolved))),
                      "    Set them with confirm_paints, one per line: axis = region"]

        name = str(approved_by or "").strip()
        if not name:
            lines += ["", "NOT APPROVED — type your name in 'approved_by' to release "
                          "the run."]
            summary = "\n".join(lines)
            print("[arkennemasis] recipe gate: awaiting approval")
            return (ExecutionBlocker(
                "Recipe not approved. Read the summary on the Recipe Gate node and put "
                "your name in 'approved_by' to release the run."), summary)

        if unresolved:
            summary = "\n".join(lines)
            print("[arkennemasis] recipe gate: blocked, %d unresolved axis/axes"
                  % len(unresolved))
            return (ExecutionBlocker(
                "Recipe approved but %d axis/axes still have no region: %s. Resolve "
                "them in confirm_paints — an axis pointed at the wrong region wastes "
                "the entire run." % (len(unresolved), ", ".join(map(str, unresolved)))),
                summary)

        recipe["approved_by"] = name
        recipe["approved_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        recipe["recipe_hash"] = recipe_hash(recipe)

        if write_back:
            # `_path` is stamped by ArkRecipeCompile so the approval lands on the exact
            # file that was compiled, even when recipe_dir was customised.
            path = recipe.get("_path")
            if not path:
                target = "variation/recipes"
                try:
                    import folder_paths
                    target = os.path.join(folder_paths.get_output_directory(), target)
                except Exception:
                    target = os.path.abspath(target)
                path = os.path.join(target, "%s.recipe.json"
                                    % (recipe.get("product") or "product"))
            if os.path.isfile(path):
                with open(path, "w", encoding="utf-8") as handle:
                    json.dump(recipe, handle, indent=2, ensure_ascii=False)
                print("[arkennemasis] recipe approved by %s -> %s" % (name, path))

        lines += ["", "APPROVED by %s at %s" % (name, recipe["approved_at"]),
                  "Recipe is frozen. Every cell in this run inherits it."]
        return (json.dumps(recipe, ensure_ascii=False), "\n".join(lines))


NODE_CLASS_MAPPINGS = {
    "ArkRecipeBrief": ArkRecipeBrief,
    "ArkRecipeCompile": ArkRecipeCompile,
    "ArkRecipeGate": ArkRecipeGate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArkRecipeBrief": "arkennemasis Recipe Brief (Tier 0 meta-prompt)",
    "ArkRecipeCompile": "arkennemasis Recipe Compile (validate + freeze)",
    "ArkRecipeGate": "arkennemasis Recipe Gate (the one human checkpoint)",
}

"""Post-generation QC — a critic that LOOKS at the result, then fixes the prompt.

    generated image ─► QC / Critic Agent ─► PASS ─► save
                                          └► FAIL ─► diagnose ─► correct the prompt
                                                                  └► regenerate

`ArkVerifyCandidate` measures: bounding boxes, ΔE, structural similarity. Those catch
the failures that are arithmetic — it moved, it is the wrong colour, it bled. They
cannot catch the failures that are *judgement*: the material reads as plastic rather
than stone, the pattern is the right hue but the wrong structure, a fitting quietly
changed shape, the thing simply looks wrong.

That judgement is what the operator was doing by eye, and a vision model can do it —
provided it is shown BOTH images and told what was supposed to change. So the critic
receives the base plate and the candidate side by side, plus the one attribute that was
meant to differ, and answers in a fixed JSON shape.

The valuable half is what happens on a fail. A retry with the identical prompt asks the
same question again and tends to get the same answer. The critic therefore also returns
a REWRITTEN prompt addressing the specific defect it found, and that rewrite is what the
regeneration uses. The loop is bounded by the recipe's `max_retries`.

Two nodes, so the model call is explicit on the canvas like every other one in this pack:

    ArkQCRequest   builds the critic's question and hands over both images
    ArkQCVerdict   parses the answer into pass / fail + diagnosis + corrected prompt
"""

from __future__ import annotations

import json

import torch

from .schema import ValidationError

CRITIC_INSTRUCTIONS = """\
You are the quality-control critic for an automated product-variation image pipeline.

You will be shown TWO images, and sometimes a third:
  IMAGE 1 — the original base photograph of the product.
  IMAGE 2 — a generated variation of it.
  IMAGE 3 — OPTIONAL. The reference photograph of the material that IMAGE 2's changed
            part was supposed to be made of. When it is present it is the standard:
            judge the material against THIS PICTURE, not against what you imagine the
            material's name should look like. When it is absent, judge against the
            colour and description given below.

The change list below names every attribute that was supposed to change — it may be one
or several. Everything NOT on that list — the object itself, the camera, the framing,
the background, the lighting and the shadow — was supposed to stay identical.

Judge IMAGE 2. Be strict: this image is going onto a product page where a customer will
click between variants, so a difference a shopper would notice is a defect even if it is
subtle.

Check, in this order of importance:

1. IDENTITY. Is it the same physical object? Same proportions, same silhouette, same
   parts, same joins, same fittings, same cable or hardware. Has anything been
   redesigned, straightened, simplified, tidied or "improved"?
2. SCENE. Same camera angle, distance, crop and framing. Same background, same surface,
   same lighting direction and intensity, same shadow shape and position.
3. THE CHANGE ITSELF. Did the intended attribute actually change, on the correct part,
   and does it read convincingly as the material or colour it was meant to be? A flat
   tint where a textured material was asked for is a FAIL. A pattern at the wrong scale
   or with the wrong structure is a FAIL.
4. BLEED. Did the change spill onto any part it should not have touched?
5. ARTEFACTS. Warping, smearing, nonsense geometry, text, duplicated parts.

Answer with ONE JSON object and nothing else:

{
  "verdict": "pass" | "fail",
  "confidence": 0.0-1.0,
  "identity_ok": true | false,
  "scene_ok": true | false,
  "change_ok": true | false,
  "issues": ["one short factual sentence per defect, most serious first"],
  "diagnosis": "what went wrong and why, in one or two sentences; empty when it passes",
  "corrected_prompt": "the FULL rewritten prompt to regenerate with, addressing the
                       specific defects. Keep everything that was already correct.
                       Empty string when the verdict is pass."
}

Rules for `corrected_prompt`:
* Rewrite only what needs to change to fix the defects you listed. Do not restyle a
  prompt that was mostly working.
* Never weaken the fidelity language. If identity or scene drifted, make those
  requirements MORE explicit, not less.
* Never describe the camera, background or lighting as things to change — they are
  supposed to be untouched.
* Return it as one continuous prompt string, ready to send as-is.
"""


class ArkQCRequest:
    CATEGORY = "arkennemasis/Variation"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING", "STRING", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("system_instructions", "request", "plate", "candidate",
                    "reference")
    DESCRIPTION = (
        "Build the critic's question. Wire system_instructions and request into a vision "
        "LLM with json_only ON, and the two image outputs into its image_1 and image_2 — "
        "the critic must SEE both to judge them."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "plate": ("IMAGE", {"tooltip": "The locked base photograph — IMAGE 1."}),
                "candidate": ("IMAGE", {"tooltip": "The generated variation — IMAGE 2."}),
                "cell_json": ("STRING", {
                    "multiline": True, "default": "{}", "forceInput": True,
                }),
            },
            "optional": {
                "prompt_used": ("STRING", {
                    "multiline": True, "default": "", "forceInput": True,
                    "tooltip": "The prompt that produced the candidate. The critic needs "
                               "it to rewrite it — without this it can diagnose but not "
                               "correct.",
                }),
                "recipe_json": ("STRING", {
                    "multiline": True, "default": "", "forceInput": True,
                }),
                "measurements_json": ("STRING", {
                    "multiline": True, "default": "", "forceInput": True,
                    "tooltip": "Optional verdict from Verify Candidate. Giving the "
                               "critic the numbers alongside the pictures makes it a "
                               "second opinion rather than a duplicate of one.",
                }),
                "critic_instructions": ("STRING", {
                    "multiline": True, "default": CRITIC_INSTRUCTIONS,
                }),
                # An IMAGE socket, so it never appears in widgets_values and cannot
                # shift the positional values of any graph already saved against this
                # node — which is why it can be added safely at all.
                "reference": ("IMAGE", {
                    "tooltip": "The material reference photograph for the changed part, "
                               "when the value was specified by picture. Without it the "
                               "critic is asked whether a panel 'reads convincingly as' "
                               "a material it has never seen, and can only mark its own "
                               "guess — failing correct renders, which costs a paid "
                               "regeneration each time.",
                }),
            },
        }

    def run(self, plate, candidate, cell_json, prompt_used="", recipe_json="",
            reference=None,
            measurements_json="", critic_instructions=CRITIC_INSTRUCTIONS):
        cell = json.loads(cell_json or "{}")
        recipe = json.loads(recipe_json) if str(recipe_json or "").strip() else {}

        lines = ["THE VARIATION UNDER REVIEW: %s" % cell.get("key"), "",
                 "WHAT WAS SUPPOSED TO CHANGE:"]
        for slot in sorted(cell.get("slots") or [], key=lambda s: s.get("order") or 99):
            bit = "  - the %s becomes %s" % (slot.get("region") or "target part",
                                             slot.get("display") or slot.get("value"))
            if slot.get("hex"):
                bit += " (hex %s)" % slot["hex"]
            if slot.get("description"):
                bit += " — %s" % slot["description"]
            lines.append(bit)

        invariants = recipe.get("invariants") or []
        if invariants:
            lines += ["", "FEATURES THAT DEFINE THIS OBJECT AND MUST BE UNCHANGED:"]
            lines += ["  - %s" % str(i).strip().rstrip(".") for i in invariants]

        if str(measurements_json or "").strip():
            try:
                checks = (json.loads(measurements_json) or {}).get("checks") or {}
                interesting = {k: v for k, v in checks.items()
                               if k in ("colour_de", "ssim_untargeted",
                                        "silhouette_iou", "bbox_centre_shift",
                                        "bbox_scale_drift", "bleed_fraction",
                                        "sampled_hex")}
                if interesting:
                    lines += ["", "AUTOMATED MEASUREMENTS (a second opinion, not a "
                                  "verdict — trust your eyes over these):"]
                    lines += ["  %-22s %s" % (k, v)
                              for k, v in sorted(interesting.items())]
            except ValueError:
                pass

        if str(prompt_used or "").strip():
            lines += ["", "THE PROMPT THAT PRODUCED IMAGE 2 — rewrite this one if you "
                          "fail it:", str(prompt_used).strip()]
        else:
            lines += ["", "The prompt used was not supplied, so return an empty "
                          "corrected_prompt and describe the fix in the diagnosis."]

        # A reference-image value whose specification is ONLY a picture gives the critic
        # nothing to judge against but a name — it would be marking its own guess.
        # Passing the picture through costs no extra call: the LLM node already has four
        # image sockets and only two were in use.
        #
        # When there is no reference the PLATE is repeated rather than None returned:
        # RETURN_TYPES declares five outputs, and returning fewer makes ComfyUI's own
        # cache_update index past the end of the tuple (IndexError: list index out of
        # range) — a crash that surfaces after the whole run has been paid for.
        if reference is not None:
            lines += ["", "IMAGE 3 is the material reference for the changed part. "
                          "Judge the material against it."]
        return (str(critic_instructions), "\n".join(lines), plate, candidate,
                reference if reference is not None else plate)


class ArkQCVerdict:
    CATEGORY = "arkennemasis/Variation"
    FUNCTION = "run"
    RETURN_TYPES = ("BOOLEAN", "STRING", "STRING", "STRING", "FLOAT")
    RETURN_NAMES = ("passed", "corrected_prompt", "diagnosis", "report", "confidence")
    DESCRIPTION = (
        "Parse the critic's answer. On a fail it hands back a REWRITTEN prompt aimed at "
        "the specific defect — retrying with the identical prompt mostly reproduces the "
        "identical failure."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "critic_json": ("STRING", {
                    "multiline": True, "default": "{}", "forceInput": True,
                    "lazy": True,
                    "tooltip": "The critic's answer. LAZY: when `enabled` is off this "
                               "input is never requested, so the vision model upstream "
                               "of it is never called at all.",
                }),
                "original_prompt": ("STRING", {
                    "multiline": True, "default": "", "forceInput": True,
                    "tooltip": "Returned unchanged when the critic passes it, or when it "
                               "fails without offering a rewrite — so this output is "
                               "always safe to feed straight back to the generator.",
                }),
                "cell_json": ("STRING", {
                    "multiline": True, "default": "", "forceInput": True,
                }),
                "min_confidence": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Treat a fail below this confidence as a pass. 0 trusts "
                               "the critic completely. Raise it only after you have seen "
                               "it reject images you would have accepted.",
                }),
                "enabled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Off: skip the critic and pass. It is the most expensive "
                               "stage here - one vision call per cell, 80-110s each - so "
                               "this is the single biggest speed and cost saving "
                               "available.",
                }),
                "strict_json": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "On: unparseable output raises. Off: it is treated as a "
                               "pass with a warning, so one malformed answer cannot "
                               "block a long run.",
                }),
            },
        }

    def check_lazy_status(self, critic_json=None, original_prompt="", cell_json="",
                          min_confidence=0.0, enabled=True, strict_json=False):
        """Ask for the critic's answer ONLY when the critic is switched on.

        This is why the input is lazy rather than the node simply returning early.
        ComfyUI evaluates a node's inputs before it runs, so an `enabled` flag checked
        inside `run` still pays for everything upstream — the request node, and the
        vision call itself. Measured: switching the critic off left 11 model calls in a
        five-cell run where there should have been 2. Not naming the input here means
        the whole branch is never evaluated.
        """
        return ["critic_json"] if enabled else []

    def run(self, critic_json=None, original_prompt="", cell_json="",
            min_confidence=0.0, enabled=True, strict_json=False):
        cell = json.loads(cell_json) if str(cell_json or "").strip() else {}
        key = cell.get("key") or "(candidate)"

        if not enabled:
            print("[arkennemasis] QC %s: SKIPPED (disabled)" % key)
            return (True, str(original_prompt), "",
                    "QC CRITIC DISABLED - nothing looked at this image.", 0.0)

        text = str(critic_json or "").strip()
        if text.startswith("```"):
            body = text[3:]
            newline = body.find("\n")
            if newline != -1 and body[:newline].strip().isalpha():
                body = body[newline + 1:]
            text = body.rstrip()[:-3].strip() if body.rstrip().endswith("```") else body

        try:
            data = json.loads(text)
            if not isinstance(data, dict):
                raise ValueError("not an object")
        except ValueError as exc:
            message = ("The critic did not return valid JSON (%s). Turn json_only ON "
                       "for its LLM node. First 300 chars: %s" % (exc, text[:300]))
            if strict_json:
                raise ValidationError(message)
            print("[arkennemasis] QC %s: %s — treating as PASS" % (key, message))
            return (True, str(original_prompt), "", "QC UNPARSEABLE — passed by "
                    "default.\n" + message, 0.0)

        verdict = str(data.get("verdict") or "").strip().lower()
        confidence = float(data.get("confidence") or 0.0)
        issues = [str(i) for i in (data.get("issues") or [])]
        diagnosis = str(data.get("diagnosis") or "").strip()
        corrected = str(data.get("corrected_prompt") or "").strip()

        passed = verdict != "fail"
        overridden = False
        if not passed and confidence < float(min_confidence):
            passed, overridden = True, True

        # Always hand back something usable: a caller wiring this straight into the
        # generator must never receive an empty prompt.
        out_prompt = corrected if (not passed and corrected) else str(original_prompt)

        lines = ["QC CRITIC — %s: %s" % (key, "PASS" if passed else "FAIL"),
                 "  confidence : %.2f" % confidence,
                 "  identity   : %s" % data.get("identity_ok"),
                 "  scene      : %s" % data.get("scene_ok"),
                 "  change     : %s" % data.get("change_ok")]
        if overridden:
            lines.append("  NOTE: the critic failed it at %.2f confidence, below the "
                         "%.2f floor, so it was passed." % (confidence, min_confidence))
        if issues:
            lines += ["", "ISSUES"] + ["  - %s" % i for i in issues]
        if diagnosis:
            lines += ["", "DIAGNOSIS", "  " + diagnosis]
        if not passed:
            lines += ["", "CORRECTED PROMPT"
                      if corrected else "NO REWRITE OFFERED — regenerating with the "
                                        "original prompt is unlikely to help"]
            if corrected:
                lines.append("  " + corrected[:600])

        print("[arkennemasis] QC %s: %s (confidence %.2f)%s"
              % (key, "PASS" if passed else "FAIL", confidence,
                 " — rewrote the prompt" if (not passed and corrected) else ""))
        return (passed, out_prompt, diagnosis, "\n".join(lines), confidence)


NODE_CLASS_MAPPINGS = {
    "ArkQCRequest": ArkQCRequest,
    "ArkQCVerdict": ArkQCVerdict,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArkQCRequest": "arkennemasis QC Request (ask the critic to look)",
    "ArkQCVerdict": "arkennemasis QC Verdict (pass/fail + corrected prompt)",
}

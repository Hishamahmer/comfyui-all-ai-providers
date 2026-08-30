"""One switchboard for the whole canvas — what runs, and what is skipped.

The optional stages each grew their own `enabled` widget, which works and is horrible to
use: the toggles end up scattered across a canvas eight thousand pixels wide, so turning
the expensive stage off means hunting for it, and there is nowhere to look to answer
"what is this run actually going to do?".

So the switches live here, in one node, and are WIRED to the stages they control. That
keeps a single readable panel at the top-left while leaving each stage's own toggle
intact underneath — a node still works standalone if someone drops it into another
graph, it simply takes its answer from here when connected.

The widget LABELS are written for someone who has never read this code. ComfyUI shows an
input's name on the node face and hides its tooltip behind a hover, so a switch called
`qc_critic` tells a first-time operator nothing at the moment they have to decide. The
label carries the meaning; the short output name carries the wiring.

Deliberately not the `ArkGroupSwitch` approach used elsewhere in this pack. That one sets
node modes from a frontend extension, which is powerful but needs JavaScript kept in step
with the graph's group titles. Booleans on wires need no frontend and cannot silently
desynchronise from the canvas.
"""

from __future__ import annotations

#   output name        on-canvas label                                  default
SWITCHES = (
    ("verify",
     "1. Check every image (fast, free)",
     True,
     "Measures each result against the original: is it the right colour, did the "
     "product move, did the colour leak onto the wrong part. Takes milliseconds and "
     "costs nothing. Turn it off and nothing stands between a bad render and delivery."),

    # OFF by default, and it is the only switch that is. Every other stage here costs
    # milliseconds; this one costs about 90 seconds and a paid call PER IMAGE, so on a
    # 30-image run it is the difference between minutes and most of an hour. Defaulting
    # an expensive stage to on means the cost of forgetting to think about it is money,
    # which is the wrong way round — the cost of forgetting should be that a slow check
    # did not happen, and the operator can always turn it on deliberately for a final
    # pass. The free numeric check above stays on and still catches wrong colour, a
    # moved product and colour leaking into a locked region.
    ("qc_critic",
     "2. Let AI LOOK at every image (SLOW, costs money)",
     False,
     "An AI opens the original and the new image side by side and judges it like you "
     "would - does the material actually look like stone, is the pattern right. It also "
     "rewrites the prompt when it fails one, and it is shown the reference photograph "
     "when the finish was specified by picture.\n\nTHIS IS THE EXPENSIVE ONE: about 90 "
     "seconds and one AI call per image, so OFF by default. Turn it on for a final "
     "pass, once the prompts are settled."),

    ("excalidraw",
     "3. Make the Excalidraw board",
     True,
     "Writes matrix.excalidraw - all your images laid out in a labelled grid with the "
     "colour names and codes. The file you send a client."),

    ("html_sheet",
     "4. Make the review web page",
     True,
     "Writes review.html - every image on one page with Approve / Reject buttons. "
     "Clicking them saves your decision back into the run."),

    ("board_preview",
     "5. Turn the board into a picture",
     True,
     "Draws the Excalidraw grid as a normal image so you can see it inside ComfyUI, and "
     "saves matrix.png next to it."),

    ("store_export",
     "6. Make the shop upload file (CSV)",
     True,
     "Writes the spreadsheet the client imports into their shop, with each variation "
     "pointing at its image."),
)


class ArkRunSwitches:
    CATEGORY = "arkennemasis/Variation"
    FUNCTION = "run"
    RETURN_TYPES = tuple("BOOLEAN" for _ in SWITCHES) + ("STRING",)
    RETURN_NAMES = tuple(key for key, _l, _d, _t in SWITCHES) + ("report",)
    DESCRIPTION = (
        "Turn each optional part of this canvas on or off, in one place. Reading the "
        "images, making the boards and making the shop file are all optional; loading "
        "the sheet, generating the images and saving them are not, so they are not "
        "listed here."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                label: ("BOOLEAN", {
                    "default": default,
                    "label_on": "ON", "label_off": "off",
                    "tooltip": tip,
                })
                for _key, label, default, tip in SWITCHES
            },
        }

    def run(self, **kwargs):
        values = [bool(kwargs.get(label, default))
                  for _key, label, default, _tip in SWITCHES]

        lines = ["THIS RUN WILL:", ""]
        for (_key, label, _d, _t), value in zip(SWITCHES, values):
            lines.append("   %s  %s" % ("[ON ]" if value else "[off]", label))

        skipped = [label for (_k, label, _d, _t), value in zip(SWITCHES, values)
                   if not value]
        if skipped:
            lines += ["", "SKIPPED — %d of %d:" % (len(skipped), len(SWITCHES))]
            lines += ["   - %s" % label for label in skipped]
        else:
            lines += ["", "Everything is on."]

        if not values[0] and not values[1]:
            lines += ["", "!! WARNING !!",
                      "Both checks are off, so NOTHING is looking at these images",
                      "before they are delivered. Only do this while you are testing."]
        elif not values[1]:
            lines += ["", "Note: the AI reviewer is off, so runs are much faster and",
                      "cheaper. The fast numeric check is still catching wrong colours",
                      "and a moved product."]

        print("[arkennemasis] switches: %s"
              % ", ".join("%s=%s" % (key, "ON" if value else "off")
                          for (key, _l, _d, _t), value in zip(SWITCHES, values)))
        return tuple(values) + ("\n".join(lines),)


NODE_CLASS_MAPPINGS = {
    "ArkRunSwitches": ArkRunSwitches,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArkRunSwitches": "arkennemasis Run Switches (what runs, what is skipped)",
}

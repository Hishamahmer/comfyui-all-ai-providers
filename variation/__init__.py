"""Product-variation image pipeline — sheet in, verified image library out.

One product, N variation axes, a locked base photograph, and a guarantee: every
delivered image shows the SAME physical object, differing only in the specified
attribute. That guarantee — not image generation — is what this sub-package exists to
provide.

The ten stages, and the node that does each:

     1  INTAKE          ArkSheetProbe -> ArkVariationIntake
     2  RESOLVE         ArkSpecLibrary
     3  LOCK PLATE      ArkPlateLock
     4  COMPILE RECIPE  ArkRecipeBrief -> (a vision LLM) -> ArkRecipeCompile
        HUMAN GATE      ArkRecipeGate                        <- the one human checkpoint
     5  BUILD PROMPTS   ArkPromptBuild + ArkPromptAudit      <- substitution, no LLM
     6  GENERATE        ArkGenRoute -> ArkRegionRecolour | the image model
     7  VERIFY          ArkVerifyCandidate  (tolerances from ArkCalibrate)
     8  STATE           ArkJobSkip / ArkJobRecord / ArkRunReport
     9  DERIVE          ArkDeliver
    10  DELIVER         ArkReviewBoard + ArkStoreExport

    LOOP A  verify -> generate    bounded retry, inside the graph
    LOOP B  deliver -> the sheet  status and URLs written back

Every module here is product-neutral by construction. There are no region names, no axis
names, no counts and no filename patterns in this code — all of those arrive from the
recipe or the intake mapping. Searching this package for a product noun should find
nothing, and that is the test that it generalises rather than being one client's
pipeline wearing a configuration file.

Each module is imported independently so a missing optional dependency disables one node
rather than the whole sub-package.
"""

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

_MODULES = (
    "switches",
    "intake",
    "spec_library",
    "plate_lock",
    "recipe",
    "cells",
    "flow",
    "qc",
    "prompt_build",
    "job_store",
    "recolour",
    "verify",
    "deliver",
    "board_preview",
)


def _load():
    import importlib

    for name in _MODULES:
        try:
            module = importlib.import_module("." + name, __name__)
        except Exception as exc:                    # one bad module must not hide the rest
            print("[arkennemasis] variation.%s not loaded: %s" % (name, exc))
            continue
        NODE_CLASS_MAPPINGS.update(getattr(module, "NODE_CLASS_MAPPINGS", {}))
        NODE_DISPLAY_NAME_MAPPINGS.update(getattr(module, "NODE_DISPLAY_NAME_MAPPINGS", {}))


_load()

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

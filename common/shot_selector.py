"""Shot Selector — run only N of M expensive branches, chosen at random.

Drop one between each expensive generator (an API image node, a sampler, ...) and
whatever consumes it. Give every copy the same ``how_many``/``seed``/``total_shots``
and a distinct ``shot_index``. Each copy independently derives the SAME selected set
from ``(seed, how_many, total_shots)``, so no shared state or extra wiring is needed.

Unselected branches are never executed at all — ``image`` is a **lazy** input, so
``check_lazy_status`` simply does not request it and the whole upstream subtree is
skipped. That is the point: with a paid API node upstream, a skipped shot is a call
that is never billed.

Selected  -> passes the image straight through.
Unselected-> returns ``ExecutionBlocker(None)``, which silently skips everything
             downstream (saves, previews, collectors) without raising.

Note: because ComfyUI blocks a node when ANY of its inputs is blocked, a partial run
also skips any node that gathers ALL branches (e.g. a collect/unpack pair). That is
intended — a partial run is a test/top-up pass, and the full set is only assembled
when every branch runs.
"""

import random

try:                                            # ComfyUI >= the lazy/blocker rework
    from comfy_execution.graph_utils import ExecutionBlocker
except ImportError:                             # very old builds kept it in execution.py
    from execution import ExecutionBlocker


ORDERED = "first N in order"
RANDOM = "random from seed"
SELECTION_MODES = [ORDERED, RANDOM]


def selected_indices(how_many, total_shots, seed, selection=ORDERED):
    """The chosen slot numbers. Deterministic for a given set of inputs.

    ``first N in order``  -> slots 0..N-1. What you want when the first few slots are
                             the ones you are actively working on.
    ``random from seed``  -> an unbiased sample, for a representative spread across the
                             whole set without paying for all of it.
    """
    total = max(0, int(total_shots))
    n = max(0, min(int(how_many), total))
    if n >= total:
        return set(range(total))
    if selection == RANDOM:
        return set(random.Random(int(seed)).sample(range(total), n))
    return set(range(n))


class ArkShotSelector:
    CATEGORY = "arkennemasis/Utility"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    DESCRIPTION = ("Run only N of M expensive branches - either the first N in order or "
                   "a random sample from a seed. Put one after each generator, give them "
                   "all the same how_many/seed/selection and a unique shot_index. "
                   "Unselected branches never execute, so a paid API node upstream is "
                   "never called.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "lazy": True,
                    "tooltip": "Lazy: only pulled when this shot is selected, so the "
                               "upstream generator does not run otherwise.",
                }),
                "shot_index": ("INT", {
                    "default": 0, "min": 0, "max": 9999,
                    "tooltip": "This branch's slot number. Must be unique, 0-based.",
                }),
                "how_many": ("INT", {
                    "default": 24, "min": 0, "max": 9999,
                    "tooltip": "How many of the branches to actually run. "
                               "Wire one value into every copy.",
                }),
                "total_shots": ("INT", {
                    "default": 24, "min": 1, "max": 9999,
                    "tooltip": "How many branches exist in total.",
                }),
                # No control_after_generate: it would add a hidden extra widgets_values
                # slot, and this input is normally driven by a link anyway.
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF,
                    "tooltip": "Only used by 'random from seed'. Same seed = same set.",
                }),
                # MUST stay last: widgets_values is positional, so a new widget appended
                # anywhere else shifts every later value on every existing copy.
                "selection": (SELECTION_MODES, {
                    "default": ORDERED,
                    "tooltip": "'first N in order' runs slots 1..N - use it when the "
                               "first slots are the ones you are working on. 'random "
                               "from seed' takes an unbiased sample of the whole set.",
                }),
            },
        }

    def check_lazy_status(self, shot_index, how_many, total_shots, seed,
                          selection=ORDERED, image=None):
        # Returning [] leaves `image` unevaluated -> the upstream branch never runs.
        if int(shot_index) in selected_indices(how_many, total_shots, seed, selection):
            return ["image"]
        return []

    def run(self, shot_index, how_many, total_shots, seed, selection=ORDERED, image=None):
        if (int(shot_index) in selected_indices(how_many, total_shots, seed, selection)
                and image is not None):
            return (image,)
        return (ExecutionBlocker(None),)      # silent: skips everything downstream


NODE_CLASS_MAPPINGS = {
    "ArkShotSelector": ArkShotSelector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArkShotSelector": "arkennemasis Shot Selector (run N of M)",
}

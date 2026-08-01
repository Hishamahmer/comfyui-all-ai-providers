"""arkennemasis — ComfyUI nodes for many AI use cases, under one brand.

Menu layout:  arkennemasis/LLM  ·  arkennemasis/Image Gen  ·  arkennemasis/Utility

Currently bundled: Replicate (OpenAI GPT-5 LLM + gpt-image-2) and System Instructions.
More providers/use cases (Ollama, Fal, Excalidraw, ...) drop in as sibling packages.

Each module is loaded independently, so a missing optional dependency only disables
that one module instead of breaking the whole pack.
"""

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


def _load(desc, importer):
    try:
        cls_map, disp_map = importer()
        NODE_CLASS_MAPPINGS.update(cls_map)
        NODE_DISPLAY_NAME_MAPPINGS.update(disp_map)
    except Exception as e:  # never let one provider break the others
        print(f"[arkennemasis] '{desc}' not loaded: {e}")


def _common():
    from .common.system_instructions import (
        NODE_CLASS_MAPPINGS as c, NODE_DISPLAY_NAME_MAPPINGS as d,
    )
    return c, d


def _shot_selector():
    from .common.shot_selector import (
        NODE_CLASS_MAPPINGS as c, NODE_DISPLAY_NAME_MAPPINGS as d,
    )
    return c, d


def _text_file_save():
    from .common.text_file_save import (
        NODE_CLASS_MAPPINGS as c, NODE_DISPLAY_NAME_MAPPINGS as d,
    )
    return c, d


def _run_folder():
    from .common.run_folder import (
        NODE_CLASS_MAPPINGS as c, NODE_DISPLAY_NAME_MAPPINGS as d,
    )
    return c, d


def _codex():
    from .codex_provider.nodes import (
        NODE_CLASS_MAPPINGS as c, NODE_DISPLAY_NAME_MAPPINGS as d,
    )
    return c, d


def _replicate():
    from .replicate_provider.nodes import (
        NODE_CLASS_MAPPINGS as c, NODE_DISPLAY_NAME_MAPPINGS as d,
    )
    return c, d


_load("common nodes", _common)
_load("shot selector", _shot_selector)
_load("run folder", _run_folder)
_load("text file save", _text_file_save)
_load("replicate provider", _replicate)
_load("codex provider", _codex)
# Add future providers here, e.g.:
# _load("ollama provider", _ollama)
# _load("fal provider", _fal)

# Front-end assets (the running/"cooking" activity badge).
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

"""comfyui-all-ai-providers — ComfyUI nodes for multiple AI providers.

Currently bundled: Replicate (OpenAI GPT-5 LLM + gpt-image-2) and a provider-neutral
System Instructions node. More providers (Ollama, Fal, ...) drop in as sibling packages.

Each provider is loaded independently, so a missing optional dependency only disables
that one provider instead of breaking the whole pack.
"""

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


def _load(desc, importer):
    try:
        cls_map, disp_map = importer()
        NODE_CLASS_MAPPINGS.update(cls_map)
        NODE_DISPLAY_NAME_MAPPINGS.update(disp_map)
    except Exception as e:  # never let one provider break the others
        print(f"[comfyui-all-ai-providers] '{desc}' not loaded: {e}")


def _common():
    from .common.system_instructions import (
        NODE_CLASS_MAPPINGS as c, NODE_DISPLAY_NAME_MAPPINGS as d,
    )
    return c, d


def _replicate():
    from .replicate_provider.nodes import (
        NODE_CLASS_MAPPINGS as c, NODE_DISPLAY_NAME_MAPPINGS as d,
    )
    return c, d


_load("common nodes", _common)
_load("replicate provider", _replicate)
# Add future providers here, e.g.:
# _load("ollama provider", _ollama)
# _load("fal provider", _fal)

# Front-end assets (the running/"cooking" activity badge).
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

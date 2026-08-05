"""arkennemasis — ComfyUI nodes for many AI use cases, under one brand.

Menu layout:  arkennemasis/LLM  ·  /Image Gen  ·  /Utility  ·  /Video

Currently bundled:
  * Replicate  — OpenAI GPT-5 LLM (text + vision) and gpt-image-2, via an API key.
  * Codex      — the same image model through the ChatGPT/Codex CLI login, no API key.
                 Requires a paid ChatGPT plan and `codex login` already run in a terminal.
  * Utility    — System Instructions, Shot Selector, Run Folder, Text File Save,
                 Subject Line, Codex Login Status, and the shared Image Gen Settings.
  * Video      — Scene List (the loop), Hailuo Scene, Caption Style and Video Assemble:
                 a scene plan in, one narrated and subtitled video out.
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


def _subject_line():
    from .common.subject_line import (
        NODE_CLASS_MAPPINGS as c, NODE_DISPLAY_NAME_MAPPINGS as d,
    )
    return c, d


def _run_folder():
    from .common.run_folder import (
        NODE_CLASS_MAPPINGS as c, NODE_DISPLAY_NAME_MAPPINGS as d,
    )
    return c, d


def _run_log():
    from .common.run_log import (
        NODE_CLASS_MAPPINGS as c, NODE_DISPLAY_NAME_MAPPINGS as d,
    )
    return c, d


def _contact_sheet():
    from .common.contact_sheet import (
        NODE_CLASS_MAPPINGS as c, NODE_DISPLAY_NAME_MAPPINGS as d,
    )
    return c, d


def _story_brief():
    from .common.story_brief import (
        NODE_CLASS_MAPPINGS as c, NODE_DISPLAY_NAME_MAPPINGS as d,
    )
    return c, d


def _scene_split():
    from .common.scene_split import (
        NODE_CLASS_MAPPINGS as c, NODE_DISPLAY_NAME_MAPPINGS as d,
    )
    return c, d


def _hailuo_scene():
    from .common.hailuo_scene import (
        NODE_CLASS_MAPPINGS as c, NODE_DISPLAY_NAME_MAPPINGS as d,
    )
    return c, d


def _scene_list():
    from .common.scene_list import (
        NODE_CLASS_MAPPINGS as c, NODE_DISPLAY_NAME_MAPPINGS as d,
    )
    return c, d


def _caption_style():
    from .common.caption_style import (
        NODE_CLASS_MAPPINGS as c, NODE_DISPLAY_NAME_MAPPINGS as d,
    )
    return c, d


def _video_assemble():
    from .common.video_assemble import (
        NODE_CLASS_MAPPINGS as c, NODE_DISPLAY_NAME_MAPPINGS as d,
    )
    return c, d


def _codex():
    from .codex_provider.nodes import (
        NODE_CLASS_MAPPINGS as c, NODE_DISPLAY_NAME_MAPPINGS as d,
    )
    return c, d


def _codex_llm():
    from .codex_provider.llm import (
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
_load("run log", _run_log)
_load("story brief", _story_brief)
_load("contact sheet", _contact_sheet)
_load("scene split", _scene_split)
_load("scene list", _scene_list)
_load("hailuo scene", _hailuo_scene)
_load("caption style", _caption_style)
_load("video assemble", _video_assemble)
_load("subject line", _subject_line)
_load("text file save", _text_file_save)
_load("replicate provider", _replicate)
_load("codex provider", _codex)
_load("codex llm", _codex_llm)
# Add future providers here, e.g.:
# _load("ollama provider", _ollama)
# _load("fal provider", _fal)

# Front-end assets (the running/"cooking" activity badge).
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

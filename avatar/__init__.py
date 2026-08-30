"""Daily presenter video — a story in, a vertical captioned clip of a person telling it out.

The shape of the pipeline, and the node that does each part:

    1  THE STORY     ArkAvatarBrief -> (a text LLM) -> ArkAvatarScript
    2  THE PICTURE   ArkWebShot                       <- common/, generic
    3  THE VOICE     ArkQwenTTS                       <- common/, already existed
       THE CLOCK     ArkAvatarFrames                  <- how long the clip must be
    4  THE PRESENTER a talking-head video model, driven by that voice
    5  THE CUT-OUT   a segmenter -> ArkMaskRefine -> ArkOverlaySubject
    6  THE VIDEO     ArkCaptionStyle -> ArkVideoAssemble

Only the first and third rows are here. Everything else is either a node the pack already
had or a generic one that belongs in `common/`, and that split is deliberate: a node that
would be useful to somebody not making presenter videos does not belong in a package
named for presenter videos.

**No prompt text lives in this package.** The system instructions that tell a model how
to write a script and a caption are emitted onto the canvas by the workflow builder, into
`SystemInstructions` nodes, the same arrangement `nemasis-hyperconsistent` uses. They are
therefore visible and editable in the GUI, and a rebuild restores them — where a string
buried in a `.py` file is neither. These nodes only ever assemble what the operator typed.

Public. There is no authored catalog here to protect: what makes this workflow work is
the wiring, and the wiring is on the canvas.
"""

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

_MODULES = (
    "brief",
    "story_history",
    "story_pick",
    "story_record",
    "script_plan",
    "frames",
)


def _load():
    import importlib

    for name in _MODULES:
        try:
            module = importlib.import_module("." + name, __name__)
        except Exception as exc:                    # one bad module must not hide the rest
            print("[arkennemasis] avatar.%s not loaded: %s" % (name, exc))
            continue
        NODE_CLASS_MAPPINGS.update(getattr(module, "NODE_CLASS_MAPPINGS", {}))
        NODE_DISPLAY_NAME_MAPPINGS.update(getattr(module, "NODE_DISPLAY_NAME_MAPPINGS", {}))


_load()

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

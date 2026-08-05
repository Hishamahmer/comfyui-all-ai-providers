"""The idea form — one labelled box per field, assembled into a story-agent prompt.

This is the node equivalent of the spreadsheet row a storyboard pipeline reads:
one column per creative decision, filled in by a person, then handed to the LLM. Putting
them in ONE node keeps the canvas readable while still giving each field its own box, so
nobody has to edit a pre-formatted blob of text and risk breaking the template.

The fields: title, caption, story_idea, channel_description, character,
visual_style, colors, special_request, scenes. The three sheet columns that are NOT here
live where they actually belong — `image_reference` is the Load Image node,
`background_music` is the assembler's music_path, and `aspect_ratio` is the video
resolution.

Empty fields are omitted rather than sent as blanks: a heading with nothing under it
invites the model to invent a constraint that was never asked for.
"""


FIELDS = [
    ("title", "Video Title", "The title. The character's opening line conveys this in "
                             "her own words."),
    ("character", "Character", "Who the reference image is: name, hair, eyes, exact "
                               "clothing, and how they speak. Restated in every scene "
                               "to stop the character drifting."),
    ("story_idea", "Video Description", "What actually happens. The more specific the "
                                        "want and the obstacle, the less generic the "
                                        "result."),
    ("caption", "Video Caption", "One line, as it would appear under the post."),
    ("visual_style", "Visual style", "Held identical across every scene."),
    ("colors", "Colors", "Held identical across every scene."),
    ("channel_description", "Channel description", "Who this is for, and the register "
                                                   "to write in."),
    ("special_request", "Special request", "Anything that overrides the rest. The agent "
                                           "treats this as top priority."),
]


class ArkStoryBrief:
    CATEGORY = "arkennemasis/Utility"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("brief",)
    DESCRIPTION = ("The idea form: one box per creative field, assembled into the prompt "
                   "for a story agent. The node version of a storyboard spreadsheet row.")

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "scenes": ("INT", {
                "default": 6, "min": 1, "max": 200,
                "tooltip": "How many scenes to write. This must match the number of "
                           "scene branches the workflow was built with, or the extra "
                           "branches sit idle and the assembler cannot run.",
            }),
        }
        for key, label, tip in FIELDS:
            required[key] = ("STRING", {
                "multiline": True, "default": "",
                "tooltip": "%s — %s" % (label, tip),
            })
        return {"required": required}

    def run(self, scenes, **fields):
        lines = ["Make a video with %d scenes." % int(scenes)]

        special = (fields.get("special_request") or "").strip()
        if special:
            # First, because the agent is told to treat it as the top priority; burying
            # it under the boilerplate makes it read as an afterthought.
            lines += ["", "The user's special request (TOP PRIORITY):", special]

        character = (fields.get("character") or "").strip()
        if character:
            lines += ["", "***", "",
                      "Character (the person in the reference image — they appear in "
                      "EVERY scene and are the only one who speaks):", character]

        block = []
        for key, label, _ in FIELDS:
            if key in ("special_request", "character"):
                continue
            value = (fields.get(key) or "").strip()
            if value:
                block.append("%s: %s" % (label, value))
        if block:
            lines += ["", "***", ""] + block

        lines += ["", "***", "",
                  "Make sure there are exactly %d scenes in your output." % int(scenes)]
        return ("\n".join(lines),)


NODE_CLASS_MAPPINGS = {
    "ArkStoryBrief": ArkStoryBrief,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArkStoryBrief": "arkennemasis Story Brief (the idea form)",
}

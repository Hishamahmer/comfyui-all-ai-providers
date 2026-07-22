"""System Instructions — a provider-neutral node.

Holds reusable system instructions and outputs plain text (STRING), so it works with
ANY LLM node (Replicate, Ollama, Fal, ...) that accepts a system prompt / text input.
"""


class SystemInstructions:
    CATEGORY = "arkennemesis/Utility"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("system_instructions",)
    DESCRIPTION = ("Hold reusable system instructions and output them as text. Wire the "
                   "output into any LLM node's system prompt.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "system_instructions": ("STRING", {
                    "multiline": True,
                    "default": "You are a helpful assistant.",
                }),
            },
        }

    def run(self, system_instructions):
        return (system_instructions,)


NODE_CLASS_MAPPINGS = {
    "SystemInstructions": SystemInstructions,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SystemInstructions": "arkennemesis System Instructions",
}

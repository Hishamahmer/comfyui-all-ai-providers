# arkennemasis — ComfyUI Nodes

One pack, one menu (**arkennemasis**), many AI use cases: cloud LLMs, image generation,
and utility nodes. Providers and use cases keep growing — each module loads independently,
so nothing breaks anything else.

## Nodes

| Menu | Node | What it does | Out |
|---|---|---|---|
| arkennemasis/**LLM** | arkennemasis Replicate LLM (OpenAI GPT-5) | GPT-5 family (`gpt-5`, `-mini`, `-nano`, `-pro`, `-structured`, `5.1`, `5.2`, `5.4`, `5.6-luna/terra/sol`) — text + vision (4 image inputs) | `STRING` |
| arkennemasis/**Image Gen** | arkennemasis Replicate Image Gen (GPT-Image-2) | `openai/gpt-image-2` — text→image **and** image edit (4 image inputs) | `IMAGE` |
| arkennemasis/**Utility** | arkennemasis System Instructions | reusable system prompt for any LLM node | `STRING` |

## Install

```sh
cd ComfyUI/custom_nodes
git clone https://github.com/Hishamahmer/comfyui-arkennemasis
```

Install the dependency, then restart ComfyUI:

```sh
# portable build:
python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\comfyui-arkennemasis\requirements.txt
# normal install:
pip install replicate
```

Or in **ComfyUI-Manager** → *Install via Git URL* → paste the repo URL (deps auto-install).

## API keys — three ways (pasting is optional)

Keys are read in this order: **node field → OS env var → `.env` file**.

1. **Paste** into a node's `api_token` field, or
2. **Env var:** set `REPLICATE_API_TOKEN`, or
3. **`.env` file:** copy `.env.example` → `.env` (in this folder or your ComfyUI root) and add:
   ```
   REPLICATE_API_TOKEN=r8_your_token_here
   ```

Get a Replicate token at https://replicate.com/account/api-tokens. (`.env` is gitignored.)

## Usage

Typical LLM → image flow:

```
System Instructions ─► Replicate LLM (system_prompt)
Text  ───────────────► Replicate LLM (prompt)
Image(s) ────────────► Replicate LLM (image_1..4)   ← vision
                            │ text
                            ▼
                 Replicate Image Gen (prompt)  ◄─ image_1..4 (edit/reference)
                            │
                        Save Image
```

Optional params (`quality`, `aspect_ratio`, `reasoning_effort`, …) left on **`default`** are
not sent, so the model's own defaults apply. `timeout_seconds = 0` waits indefinitely.

## Notes

- Replicate calls are **paid** — each run bills your Replicate account.
- Long runs poll (no fixed timeout) and run off the UI thread, so ComfyUI stays responsive
  and **Cancel** works. A spinner + elapsed-time badge shows on the node while it runs.
- GPT-5 models output **text only**; image generation is done by the Image Gen node.

## Structure

```
comfyui-arkennemasis/
│
├── __init__.py              THE HUB — loads every module and merges their node maps.
│                            The only file you touch when adding something.
│
├── common/                  SHARED CODE — written once, reused by every module
│   ├── keys.py                 resolve_key(): node field → env var → .env
│   ├── image_utils.py          tensor ↔ data-URI ↔ bytes, text normalising
│   └── system_instructions.py  the System Instructions node
│
├── replicate_provider/      ONE PROVIDER = ONE FOLDER
│   └── nodes.py                Replicate LLM + Image Gen nodes
│
├── web/                     FRONT-END (auto-served via WEB_DIRECTORY)
│   └── activity.js             running / "cooking" badge on the node
│
├── requirements.txt         dependencies
├── pyproject.toml           ComfyUI-Manager metadata
├── .env.example             key template users copy to .env
├── README.md
└── LICENSE
```

Menu categories are set per node class, so **one provider folder can feed several
categories** (Replicate already serves both `LLM` and `Image Gen`):

```
arkennemasis/
├── LLM         ← replicate_provider · (ollama_provider · fal_provider · …)
├── Image Gen   ← replicate_provider · (fal_provider · …)
├── Video Gen   ← (new folder when needed)
├── Audio       ← (new folder when needed)
└── Utility     ← common/ modules (System Instructions, …)
```

## Adding a module — 3 steps

**1.** New folder with a `nodes.py` ending in the two standard dicts:

```python
NODE_CLASS_MAPPINGS = {"OllamaLLM": OllamaLLM}
NODE_DISPLAY_NAME_MAPPINGS = {"OllamaLLM": "arkennemasis Ollama LLM"}
```

**2.** Give each node class its menu placement:

```python
class OllamaLLM:
    CATEGORY = "arkennemasis/LLM"      # or /Image Gen, /Utility, /Video Gen, /Audio …
```

**3.** Register it in `__init__.py`:

```python
def _ollama():
    from .ollama_provider.nodes import (
        NODE_CLASS_MAPPINGS as c, NODE_DISPLAY_NAME_MAPPINGS as d,
    )
    return c, d

_load("ollama provider", _ollama)      # next to the existing _load calls
```

`_load()` isolates failures: if a module raises (missing dependency, upstream API change)
it logs `[arkennemasis] 'ollama provider' not loaded: …` and **every other module keeps
working**. Someone who only wants the Ollama nodes never needs a Replicate account.

### Reuse instead of rewriting

| Need | Use |
|---|---|
| API key from field / env / `.env` | `from ..common.keys import resolve_key` |
| ComfyUI image → send to an API | `collect_images_to_data_uris(img1, img2, …)` |
| API response → ComfyUI `IMAGE` | `bytes_list_to_image_tensor(output_to_bytes_list(out))` |
| stream / list / str → clean text | `output_to_text(out)` |
| long API call without freezing the UI | copy the `async run()` + `asyncio.to_thread(self._blocking, …)` pattern in `replicate_provider/nodes.py` |
| activity badge on a node | add the class key to `ANIMATED_NODES` in `web/activity.js` |

### Two rules

1. **Class keys are permanent.** `"OllamaLLM"` is the ID saved inside every workflow —
   renaming it breaks those workflows ("missing node" on load). Display names and
   categories are cosmetic and safe to change anytime.
2. **New dependencies go in `requirements.txt` and are imported *inside* the function**,
   not at module top level — so a user missing that package loses only that node.

## License

MIT

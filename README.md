# arkennemasis — ComfyUI Nodes

One pack, one menu (**arkennemasis**), many AI use cases: cloud LLMs, image generation,
and utility nodes. Providers and use cases keep growing — each module loads independently,
so nothing breaks anything else.

## Nodes

| Menu | Node | What it does | Out |
|---|---|---|---|
| arkennemasis/**LLM** | arkennemasis Replicate LLM (OpenAI GPT-5) | GPT-5 family (`gpt-5`, `-mini`, `-nano`, `-pro`, `-structured`, `5.1`, `5.2`, `5.4`, `5.6-luna/terra/sol`) — text + vision (4 image inputs) | `STRING` |
| arkennemasis/**Image Gen** | arkennemasis Replicate Image Gen (GPT-Image-2) | `openai/gpt-image-2` — text→image **and** image edit (4 image inputs) | `IMAGE` |
| arkennemasis/**Image Gen** | arkennemasis Image Gen Settings (shared) | one node driving `aspect_ratio` / `quality` / `run_mode` / `background` / `output_format` / `moderation` / `timeout_seconds` / `api_token` on **many** Image Gen nodes at once | `ARK_IMAGE_SETTINGS` |
| arkennemasis/**Utility** | arkennemasis System Instructions | reusable system prompt for any LLM node | `STRING` |
| arkennemasis/**Utility** | arkennemasis Shot Selector (run N of M) | run only N of M expensive branches, picked at random from a seed — unselected branches **never execute**, so a paid API node upstream is never called | `IMAGE` |
| arkennemasis/**Utility** | arkennemasis Run Folder (auto-numbered) | `<parent_dir>/<folder_name>_001`, `_002`, … — one fresh output folder per run | `STRING`, `INT` |

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
3. **`.env` file** containing:
   ```
   REPLICATE_API_TOKEN=r8_your_token_here
   ```

`.env` is looked for in **this folder** *and* in **ComfyUI's working directory** (its root).
**Prefer the ComfyUI root** — keeping the key outside the repo means re-cloning or updating
the pack never touches it, and no secret ever sits in a git working tree. `.env` is
gitignored either way; `.env.example` is the template.

Get a Replicate token at https://replicate.com/account/api-tokens.

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

### Driving many Image Gen nodes from one place

ComfyUI **rejects a `STRING` link into a `COMBO` widget**, so a shared `aspect_ratio` or
`quality` cannot be wired straight into the widgets. **Image Gen Settings** bundles them
into one typed value instead — wire its output into each node's optional `settings` socket:

```
Image Gen Settings ─┬─► Image Gen #1 (settings)
                    ├─► Image Gen #2 (settings)
                    └─► … 24 more
```

Any field left on **`use node's own`** (`-1` for `timeout_seconds`, blank for `api_token`)
falls through to that node's own widget, so you can share most settings and still override
one node locally.

`number_of_images` is deliberately **not** shared. Multiplying it across every wired node
is rarely what you want, and in a graph that names files deterministically the extra images
all land on the same filename and overwrite each other. Set it per node if you need it. `settings` is a *socket*, not
a widget, so adding it did not shift any existing `widgets_values` index.

### Rate limits and concurrency

ComfyUI runs `async def` nodes **concurrently**, so several API nodes in one graph create
their predictions in the same millisecond. Replicate drops to *"6 requests per minute with
a burst of 1"* while an account holds **under $5 credit**, so parallel calls reliably return
**429**.

Both API nodes therefore have a **`run_mode`** widget:

| `run_mode` | Behaviour |
|---|---|
| **`one at a time`** (default) | an asyncio lock serialises **every** arkennemasis API node in the graph |
| **`all at once`** | the original concurrent behaviour — faster when your rate limit allows it |

Independently of the mode, a 429 is retried with **exponential backoff**, honouring the
`Retry-After` header or the *"resets in ~Ns"* hint in the message. Non-rate-limit errors
still fail immediately.

### Running only some of many expensive branches

**Shot Selector** sits between each generator and its consumers. Give every copy the same
`how_many` / `seed` / `total_shots` and a unique `shot_index`; each copy derives the *same*
chosen set independently, so no extra wiring is needed.

`image` is a **lazy** input, so an unselected branch is never evaluated — the API node
upstream is never called, and never billed. Unselected returns `ExecutionBlocker(None)`,
which silently skips everything downstream.

> **Caveat:** ComfyUI blocks any node that has a blocked input, so a partial run also skips
> nodes that gather *every* branch (collect/unpack pairs). Give each branch its own save if
> you want partial runs to produce output.

### One output folder per run

**Run Folder** resolves `<parent_dir>/<folder_name>_<NNN>` for the current run, picking the
next free number (it scans existing siblings, so a manual `_009` yields `_010`). Wire
`folder_path` into every save node so a run's outputs land together and nothing is
overwritten.

It exists because save nodes that support time tokens evaluate them **at save time** — a
graph writing 24 images over an hour would scatter them across several `[time(%H-%M)]`
folders. `IS_CHANGED` returns NaN so the path is recomputed each queued run rather than
served from cache; only the saves re-execute, so re-running does **not** re-bill an
upstream API node.

## Notes

- Replicate calls are **paid** — each run bills your Replicate account.
- Long runs poll (no fixed timeout) and run off the UI thread, so ComfyUI stays responsive
  and **Cancel** works. A spinner + elapsed-time badge shows on the node while it runs.
- GPT-5 models output **text only**; image generation is done by the Image Gen node.

## Example workflows

Canvas-format workflow exports live in [`example workflows/`](example%20workflows/). See the
README there for conventions — most importantly **never commit an `api_token`**, since
workflow JSON stores widget values verbatim.

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
│   ├── throttle.py             serial_lock() + with_rate_limit_retry() (429 backoff)
│   ├── system_instructions.py  the System Instructions node
│   ├── shot_selector.py        the Shot Selector node (lazy input + ExecutionBlocker)
│   └── run_folder.py           the Run Folder node (auto-numbered per run)
│
├── replicate_provider/      ONE PROVIDER = ONE FOLDER
│   ├── nodes.py                Replicate LLM + Image Gen nodes
│   └── settings.py             the shared Image Gen Settings node
│
├── example workflows/       canvas .json exports demonstrating the nodes
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
└── Utility     ← common/ modules (System Instructions, Shot Selector, Run Folder, …)
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
| serialise calls across nodes | `async with serial_lock(): …` (`common/throttle.py`) |
| survive a provider's 429 | `with_rate_limit_retry(lambda: client…create(…))` |
| skip an expensive branch entirely | lazy input + `check_lazy_status` returning `[]`, then `ExecutionBlocker(None)` — see `common/shot_selector.py` |
| one output folder per run | `next_run_folder(parent, name)` (`common/run_folder.py`) |
| activity badge on a node | add the class key to `ANIMATED_NODES` in `web/activity.js` |

### Three rules

1. **Class keys are permanent.** `"OllamaLLM"` is the ID saved inside every workflow —
   renaming it breaks those workflows ("missing node" on load). Display names and
   categories are cosmetic and safe to change anytime.
2. **New dependencies go in `requirements.txt` and are imported *inside* the function**,
   not at module top level — so a user missing that package loses only that node.
3. **Append new widgets at the end** of the `optional` block. A workflow stores widget
   values as a positional list, so inserting one in the middle shifts every later value and
   silently corrupts saved workflows. Adding a *socket* (a non-widget type) is always safe.

## License

MIT

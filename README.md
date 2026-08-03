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
| arkennemasis/**Image Gen** | arkennemasis Codex Image Gen (ChatGPT login) | `gpt-image-2` through your **`codex login`** — no API key, billed to your ChatGPT plan | `IMAGE`, `STRING` |
| arkennemasis/**Utility** | arkennemasis Codex Login Status | which ChatGPT account this machine will use, and when its token expires | `STRING` |
| arkennemasis/**Utility** | arkennemasis System Instructions | reusable system prompt for any LLM node | `STRING` |
| arkennemasis/**Utility** | arkennemasis Shot Selector (run N of M) | run only N of M expensive branches — the **first N in order**, or a random sample from a seed. Unselected branches **never execute**, so a paid API node upstream is never called | `IMAGE` |
| arkennemasis/**Utility** | arkennemasis Text File Save (caption sidecar) | writes `<folder>/<filename>.txt` next to a saved image — the image/caption pairing training toolkits expect | `STRING` |
| arkennemasis/**Utility** | arkennemasis Run Folder (auto-numbered) | `<parent_dir>/<folder_name>_001`, `_002`, … — one fresh output folder per run | `STRING`, `INT` |

## Install

```sh
cd ComfyUI/custom_nodes
git clone https://github.com/Hishamahmer/comfyui-arkennemasis
```

Install the dependencies, then restart ComfyUI:

```sh
# portable build:
python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\comfyui-arkennemasis\requirements.txt
# normal install:
pip install replicate httpx
```

Or in **ComfyUI-Manager** → *Install via Git URL* → paste the repo URL (deps auto-install).

## What each path needs

The two image-generation paths reach the **same** `gpt-image-2` model. Pick whichever you
already pay for — you do not need both.

| Path | Nodes | What it requires |
|---|---|---|
| **Replicate** | Replicate LLM, Replicate Image Gen | A Replicate account and an API key. Pay-as-you-go per image. |
| **Codex / ChatGPT** | Codex Image Gen, Codex Login Status | A **paid ChatGPT subscription** and the **Codex CLI already logged in on this machine**. No API key. Images bill against your ChatGPT plan instead of per call. |

### Codex path — read this before you try it

1. **A paid ChatGPT plan is required.** The free tier cannot call the hosted image tool.
   **ChatGPT Plus at $20/month is the recommended plan** and is what this pack is
   developed against. Business/Pro plans work too.
2. **Install the Codex CLI and sign in from your own terminal**, on the same machine and
   the same user account that runs ComfyUI:
   ```sh
   codex login
   ```
   This writes `~/.codex/auth.json`. The nodes read that file directly — **there is no
   OAuth flow inside ComfyUI and nowhere to paste a password.** If you have not run
   `codex login`, the Codex nodes will tell you so and stop.
3. **Check it worked** by dropping in the **Codex Login Status** node — it reports the
   signed-in account, the plan and when the token expires.

> Availability is account-dependent: not every ChatGPT plan or region can call the hosted
> image tool. The node says so plainly rather than failing cryptically.

Running several ChatGPT accounts? Give each its own `CODEX_HOME` folder and set the node's
`codex_home` per node. The Codex Image Gen node's `account` output names the signed-in
email, so you can see which login produced an image.

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

`settings` is a *socket*, not a widget, so adding it did not shift any existing
`widgets_values` index.

`number_of_images` is deliberately **not** shared. Multiplying it across every wired node
is rarely what you want, and in a graph that names files deterministically the extra images
all land on the same filename and overwrite each other. Set it per node if you need it.

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

`max_concurrent` sets how many calls `all at once` may actually have in flight (default
**2**, `0` = uncapped).

### Retries — what is retried, and what is not

A node exception aborts the **whole ComfyUI prompt**, so on a 24-image batch one bad
minute of network throws away every shot still queued. `common/throttle.with_retry`
therefore retries anything that got **no complete answer**:

| Retried | Not retried |
|---|---|
| `429` — backs off exponentially, honouring `Retry-After` or the *"resets in ~Ns"* hint | a moderation refusal |
| `500 / 502 / 503 / 504` | a malformed request (`400`) |
| a dropped or truncated connection (`RemoteProtocolError`, read timeouts, *"incomplete chunked read"*) | an expired or wrong login (`401 / 403`) |
| an SSE stream that ends with no completion event | an account without the image tool |

Dropped connections retry after a couple of seconds — waiting a minute does not make a
socket healthier. Rate limits keep the long backoff. Six attempts, then the original
error is raised.

The Codex node also tells a **finished** image apart from a `partial_images` preview, so a
stream cut short mid-render can never be saved as if it were the final result.

### Running only some of many expensive branches

**Shot Selector** sits between each generator and its consumers. Give every copy the same
`how_many` / `seed` / `total_shots` / `selection` and a unique `shot_index`; each copy
derives the *same* chosen set independently, so no extra wiring is needed.

`selection` picks how the set is chosen:

| Mode | Behaviour | Use it when |
|---|---|---|
| `first N in order` *(default)* | runs slots `1..N` | the first branches are the ones you are iterating on — `how_many = 5` runs the first five, no seed needed |
| `random from seed` | an unbiased sample of the whole set | you want a representative spread across every branch without paying for all of them |

`seed` only matters in `random from seed`.

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

### Image generation with a ChatGPT login (no API key)

**Codex Image Gen** reuses the OAuth credentials the Codex CLI already wrote — the very
same login, not a copy:

```
codex login          # once, in a terminal; opens the browser
```

It reads `$CODEX_HOME/auth.json`, else `~/.codex/auth.json`. **No OAuth flow is
implemented in ComfyUI** — logging in stays the CLI's job. `codex logout` and the node
stops working; log in as someone else and the node follows.

**Multiple accounts:** give each its own folder and point `codex_home` at the one you
want. Different nodes can use different accounts in the same graph.

```powershell
$env:CODEX_HOME = "C:\CodexAccounts\work"
codex login
```

The node's `account` output and its console line both name the signed-in email, so you
can always see which account produced an image. **Codex Login Status** reports it without
generating anything.

Token handling: an expired access token is refreshed against
`https://auth.openai.com/oauth/token` and **saved back atomically**, preserving every
other key in the file. That write matters — the endpoint may return a *rotated* refresh
token, and dropping it would break the login on the next rotation. Set `allow_refresh`
off to fail instead of ever writing.

> Availability is account-dependent: not every ChatGPT plan can call the hosted image
> tool. If yours cannot, the node says so plainly instead of dumping an HTTP error.

### Caption sidecars

**Text File Save** writes `<folder_path>/<filename>.<extension>`. Give it the same folder
and filename stem the image save uses and you get the pairing kohya / ai-toolkit /
diffusers expect:

```
shot_001.png
shot_001.txt
```

Wire the generating image into `images`: it binds the caption to that branch, so a branch
skipped by a gate writes no orphan `.txt`.

**Caption rule for character LoRAs:** anything you caption is *excluded* from what the LoRA
learns. Caption the variable parts — pose, expression, wardrobe, background — and never the
face, hair colour or eye colour. This is why feeding it a VLM description of the finished
image is usually wrong: a VLM writes exactly the identity features you must not caption.

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
│   ├── throttle.py             serial_lock(), concurrency_gate(), with_retry()
│   ├── system_instructions.py  the System Instructions node
│   ├── shot_selector.py        the Shot Selector node (lazy input + ExecutionBlocker)
│   ├── text_file_save.py       the Text File Save node (caption sidecars)
│   └── run_folder.py           the Run Folder node (auto-numbered per run)
│
├── codex_provider/          ChatGPT/Codex OAuth — no API key
│   ├── auth.py                 reads `codex login` creds, refreshes + persists them
│   └── nodes.py                Codex Image Gen + Codex Login Status
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
| survive a 429, a 5xx or a dropped connection | `with_retry(lambda: client…create(…))` — retries only calls that got no complete answer; refusals and bad requests raise straight away |
| mark your own exception retryable | set `retryable = True` on it; `common/throttle.is_transient` honours it |
| skip an expensive branch entirely | lazy input + `check_lazy_status` returning `[]`, then `ExecutionBlocker(None)` — see `common/shot_selector.py` |
| one output folder per run | `next_run_folder(parent, name)` (`common/run_folder.py`) |
| reuse a `codex login` session | `codex_provider/auth.py` (`get_access_token`, `request_headers`) |
| write a sidecar file atomically | `common/text_file_save.py` (`.part` + `os.replace`) |
| activity badge on a node | add the class key to `ANIMATED_NODES` in `web/activity.js` |

### Four rules

1. **Class keys are permanent.** `"OllamaLLM"` is the ID saved inside every workflow —
   renaming it breaks those workflows ("missing node" on load). Display names and
   categories are cosmetic and safe to change anytime.
2. **New dependencies go in `requirements.txt` and are imported *inside* the function**,
   not at module top level — so a user missing that package loses only that node.
3. **Append new widgets at the end** of the `optional` block. A workflow stores widget
   values as a positional list, so inserting one in the middle shifts every later value and
   silently corrupts saved workflows. Adding a *socket* (a non-widget type) is always safe.
4. **Every node that makes the user wait gets the activity badge.** If it calls a network
   API, polls, or otherwise takes more than a moment, add its class key to
   `ANIMATED_NODES` in `web/activity.js` — otherwise the graph looks frozen and people
   re-queue it. Instant nodes stay out: a spinner that appears and vanishes in one frame
   is just flicker.

   ```js
   const ANIMATED_NODES = new Set([
     "ReplicateOpenAILLM",
     "ReplicateOpenAIGPTImage2",
     "ArkCodexImageGen",
     "YourNewLongRunningNode",   // <- add it here
   ]);
   ```

## License

MIT

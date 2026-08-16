# arkennemasis — ComfyUI Nodes

One pack, one menu (**arkennemasis**), many AI use cases: cloud LLMs, image generation,
and utility nodes. Providers and use cases keep growing — each module loads independently,
so nothing breaks anything else.

---

## ⭐ Generate images with your ChatGPT subscription — no API key

If you already pay for ChatGPT, **you can generate `gpt-image-2` images in ComfyUI without
buying any API credit.** Install the pack, run `codex login` once in a terminal, and the
**Codex Image Gen** node signs in with your existing ChatGPT account. Images bill against
your plan instead of per call.

**ChatGPT Plus at $20/month is enough** and is what this pack is developed against.

```sh
codex login      # once, in your own terminal — not inside ComfyUI
```

There is no API key field and no OAuth flow in ComfyUI — the node reads the Codex CLI's
own login. Drop in **Codex Login Status** to confirm the account and plan before you run
anything. Full details in [What each path needs](#what-each-path-needs).

## 🎬 Ready-made workflow: character LoRA training dataset

[`example workflows/`](example%20workflows/) ships a complete
**Character Dataset (GPT-Image-2) - Codex** workflow: drop in **one photo of a person** and
it produces a **25-image LoRA training dataset** — 24 generated shots plus the real
reference tile, each with a matching caption `.txt`, in an auto-numbered folder. Ready to
feed straight into LoRA training.

All 24 shots are distinct looks — street candid, studio campaign, black-and-white
editorial, festive ethnic wear, automotive, poolside, snow and more — with 24 different
outfits, so the model learns the *person* rather than the clothes or the room. No GPU
needed; every generation is an API call.

---

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
| arkennemasis/**Utility** | arkennemasis Subject Line (gender + notes) | one `Subject: …` line from a gender choice plus free-text notes, wired into every prompt — so the prompts themselves stay gender-neutral and the subject is stated once | `STRING` |
| arkennemasis/**Utility** | arkennemasis Text File Save (caption sidecar) | writes `<folder>/<filename>.txt` next to a saved image — the image/caption pairing training toolkits expect | `STRING` |
| arkennemasis/**Utility** | arkennemasis Run Folder (auto-numbered) | `<parent_dir>/<folder_name>_001`, `_002`, … — one fresh output folder per run | `STRING`, `INT` |
| arkennemasis/**Utility** | arkennemasis Story Brief / Run Log / Contact Sheet | the brief form, a JSON run log that needs no spreadsheet, and every still of a run on one sheet | `STRING`, `IMAGE` |
| arkennemasis/**Video** | arkennemasis Scene List (the loop) | fans a scene plan out so one chain runs once per scene — 5 or 50, same canvas | lists |
| arkennemasis/**Video** | arkennemasis Hailuo Scene | one scene start to finish: condition → sample → decode video **and** audio → mux → save → free | `VIDEO` |
| arkennemasis/**Audio** | arkennemasis Qwen3-TTS (voice clone) | local Qwen3-TTS. Text in, speech out; give it 5–30 s of someone speaking and it clones that voice. Runs in a subprocess — see below | `AUDIO`, `STRING` |
| arkennemasis/**Video** | arkennemasis Video Dub (narration over a clip) | swaps a clip's own soundtrack for a narration track, per clip. MiniMax H3 always generates audio and cannot be asked for silence, so the voice has to *replace* it | `VIDEO`, `STRING` |
| arkennemasis/**Video** | arkennemasis Narration Length (fit the shot to the voice) | measures a rendered narration and returns the shot length that covers it, snapped to H3's frame grid. Wire between the TTS node and the scene node and every shot outlasts its own voice-over | `INT`, `FLOAT`, `STRING` |
| arkennemasis/**Video** | arkennemasis Caption Style (font + subtitle style) | one of five subtitle styles, any installed font, colours, outline, box, size, 3×3 position — and an on/off switch | `ARK_CAPTION_STYLE` |
| arkennemasis/**Video** | arkennemasis Video Assemble (clips + music + subs) | joins every clip, levels each one's speech, ducks a music bed, burns the captions | `STRING`, `VIDEO` |
| arkennemasis/**Video** | arkennemasis Load Clips (finished clips from disk) | reads a run's finished clips back as a VIDEO list — join a film whose render was interrupted, without re-rendering | `VIDEO` list |

### arkennemasis/**Variation** — the product-variation pipeline

A client's variation spreadsheet plus one locked base photograph in; a verified,
consistently framed image library out. Any product, any number of variation axes. The
guarantee is not "good images" — it is that **every delivered image shows the same
physical object, differing only in the specified attribute**.

**A colour's specification format is a property of the VALUE, not the axis.** Four are
supported and a real client sheet mixes them inside a single axis:

| Format | `spec_type` | Reference sent? | Colour auto-checked? |
|---|---|---|---|
| hex only | `hex` | no — a flat swatch would ask for a flat fill | yes |
| hex + description | `hex` | no | yes |
| reference image | `reference_image` | yes | no — there is no target |
| reference image + hex | `reference_image` | yes | yes |

A bare word with neither is rejected: a word is a request for an opinion, and the model
gives a different opinion every time it is asked. `ref_url` accepts an `http(s)` URL, a
`file://` URL, or a bare local path.

Nothing in `variation/` knows what a product is. No region names, no axis names, no
counts, no filename patterns: all of those arrive from the recipe or the intake mapping,
and the cell generator is a cartesian product over however many axes the recipe declares.

| Node | What it does | Out |
|---|---|---|
| Sheet Probe | reads any client sheet (CSV/TSV/JSON) without interpreting it, and proposes a column mapping | `STRING` |
| Variation Intake | normalises whatever arrived into VARIANTS / SPECS / PRODUCT and runs every pre-generation validation. **Fails loudly rather than guessing** — a guessed material makes a plausible image that is wrong | `STRING`, `BOOLEAN`, `INT` |
| Spec Library | downloads, caches and content-hashes every reference; renders a swatch per hex. Per client, accumulates across products | `STRING`, `IMAGE` |
| Plate Lock | freezes one base photo and measures it: hash, dimensions, colour profile, per-region boxes, proportion ratios. After this it is never regenerated | `IMAGE`, `MASK`, `STRING` |
| Region Mask | pulls one named region's mask back out of the locked plate, resolving the cell's own target region by itself | `MASK`, `STRING`, `BOOLEAN` |
| Recipe Brief | the fixed, **product-neutral** Tier 0 meta-prompt — it discovers regions by looking at the photograph | `STRING` |
| Recipe Compile | validates the model's JSON, merges the library, and **injects both locks and the tolerances as constants** — anything the model wrote there is discarded | `STRING`, `BOOLEAN` |
| Recipe Gate | the one human checkpoint. Blocks everything downstream until a name is typed. Guards the `paints` field, which is where a wrong answer wastes a whole run | `STRING` |
| Cell Matrix / Cell At | the cartesian product of **N** axes × plates, then one cell by index. Two nested loops over exactly two axes is a defect, not a simplification | `STRING`, `IMAGE`, `INT` |
| Prompt Build | assembles one prompt by **pure substitution** — change instruction, then invariants, then both locks, the last three byte-identical across the run. Never calls a model | `STRING` |
| Prompt Audit | proves every prompt in the run shared one constant block, and fails if not. Prompt variance is *the* mechanism by which a set drifts | `BOOLEAN`, `INT` |
| Gen Route | picks the deterministic or the generative path per axis. Both inputs lazy, so the branch not taken is **never evaluated** — no API call at all | `IMAGE`, `STRING` |
| Region Recolour | retints a masked region to an exact hex in CIELAB while keeping the plate's own shading. Free, instant, and **cannot drift by construction** | `IMAGE`, `INT` |
| Verify Candidate | measures product identity, frame match, colour ΔE2000, bleed and hygiene, then returns pass / soft / hard. The step that used to live in the operator's head | `STRING`, `BOOLEAN`, `FLOAT` |
| Calibrate | derives tolerances from a real labelled set instead of guesswork, and reports the operator's true current pass rate. No generation spend | `STRING`, `FLOAT` |
| Job Skip / Job Record / Run Report | durable per-cell records on disk. Job Skip's generate branch is **lazy**, so a finished cell is never re-generated and never re-billed | `IMAGE`, `STRING`, `FLOAT` |
| Deliver | the format ladder (PNG / white-background JPG / WebP / thumb) filed one directory level per axis. Overwrites in place — no `_v2`, no timestamps | `STRING` |
| Review Board | an HTML contact sheet whose approve/reject buttons write **straight into the job records**, plus an `.excalidraw` matrix for the client | `STRING` |
| Store Export | the variation CSV with `meta:attribute_pa_{axis}` columns and each row's image — what turns a folder of images into "the variations are live" | `STRING`, `INT` |

Full instructions, the measured behaviour, and the five verification lessons that cost
real debugging:
`claude/workflow-runbooks/variation-pipeline/RUNBOOK.md` in the portable install.

### Qwen3-TTS — why it needs a one-off setup

**Qwen3-TTS runs in a subprocess, and that is deliberate.** It is written against
`transformers==4.57.3`; a normal ComfyUI install is on 5.x, and a dozen other node packs
depend on that. Every published wrapper for this model tells you to downgrade — **don't**.
The pin is also self-contradictory: `qwen-tts` 0.1.1 hardcodes 4.57.3 while its own
tokenizer imports `check_model_inputs`, which only exists in 5.x.

So the node keeps a private dependency tree in `vendor/tts_env` and puts it first on
`sys.path` in a **child process** — same interpreter, same torch, same CUDA, only the one
conflicting package differs, and only there. `vendor/` is **not** committed (114 MB, and
redistributing someone else's package is a deliberate decision, not a `git add .`), so a
fresh clone has to create it once. The command is in
[`common/qwen_tts_node.py`](common/qwen_tts_node.py)'s docstring.

Models go in `ComfyUI/models/qwen-tts/<model folder>`. A ***Base*** model is **clone-only**
— it has no preset voices, so `reference_audio` must be connected or the node stops
immediately and says so. For preset voices, use a *CustomVoice* model.

Generation is retried up to three times with fresh seeds: the model occasionally never
emits end-of-speech and generates until something stops it, which is a property of the
sampled path, so re-running the same seed reproduces it exactly. The node also rejects a
result far longer than the text can account for. It refuses to fall back to CPU — that
still produces correct audio, roughly 40× slower, which reads exactly like a hang.

### Captions

**Caption Style** feeds **Video Assemble**. Five styles:

| Style | On screen |
|---|---|
| `classic` | the whole line at once |
| `karaoke` | the fill sweeps across the line as it is spoken |
| `highlight` | the spoken word changes colour |
| `underline` | the spoken word is underlined |
| `word_by_word` | one word at a time, nothing else |

Everything but `classic` marks individual words, so it needs to know when each word is
spoken. A video model gives no word timestamps, so they are **estimated** from the script
and the clip's real duration, weighted by word length and by trailing punctuation. That
tracks speech closely; it is not frame-accurate, and it drifts if the model ad-libs.

Fonts come from [`fonts/`](fonts/) (bundled, listed first) and from the machine's own
installed fonts. See that folder's README for what ships and how to add more.

Toggle `enabled` off on the node for a video with no subtitles at all — every other
setting stays put.

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
│   ├── banner.py               the load-time ASCII banner (ARK_BANNER=0 silences it)
│   ├── system_instructions.py  the System Instructions node
│   ├── shot_selector.py        the Shot Selector node (lazy input + ExecutionBlocker)
│   ├── subject_line.py         the Subject Line node (gender stated once, not per prompt)
│   ├── text_file_save.py       the Text File Save node (caption sidecars)
│   ├── run_folder.py           the Run Folder node (auto-numbered per run)
│   ├── scene_list.py           the loop: OUTPUT_IS_LIST fans a scene plan out
│   ├── hailuo_scene.py         one scene start to finish, freed before the next
│   ├── ass_captions.py         font discovery + the five subtitle styles as ASS
│   ├── caption_style.py        the Caption Style node
│   └── video_assemble.py       the aggregate end: join, level, duck, burn
│
├── codex_provider/          ChatGPT/Codex OAuth — no API key
│   ├── auth.py                 reads `codex login` creds, refreshes + persists them
│   └── nodes.py                Codex Image Gen + Codex Login Status
│
├── replicate_provider/      ONE PROVIDER = ONE FOLDER
│   ├── nodes.py                Replicate LLM + Image Gen nodes
│   └── settings.py             the shared Image Gen Settings node
│
├── variation/               PRODUCT-VARIATION PIPELINE — a use case, not a provider
│   ├── schema.py               the canonical 3-table schema + every validator
│   ├── colour.py               sRGB↔Lab, ΔE2000, robust sampling, shading-preserving recolour
│   ├── intake.py               Sheet Probe + Variation Intake
│   ├── spec_library.py         download, cache and hash every reference
│   ├── plate_lock.py           freeze and measure the base plate; Region Mask
│   ├── recipe.py               Tier 0 brief, compile+validate, and the human gate
│   ├── cells.py                the N-axis cartesian product, and one cell by index
│   ├── prompt_build.py         substitution-only prompts + the constancy audit
│   ├── job_store.py            durable job records, lazy resume, the run report
│   ├── recolour.py             the non-generative path and the per-axis router
│   ├── verify.py               identity / frame / colour / bleed, and calibration
│   └── deliver.py              format ladder, review board, store import file
│
├── fonts/                   CAPTION FONTS — 13 OFL/Apache families, see fonts/README.md
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
├── LLM         ← replicate_provider · codex_provider
├── Image Gen   ← replicate_provider · codex_provider · (fal_provider · …)
├── Video       ← common/ modules (Scene List, Hailuo Scene, Video Assemble, …)
├── Audio       ← common/qwen_tts_node
├── Variation   ← variation/ (a USE CASE, not a provider — it calls the others)
└── Utility     ← common/ modules (System Instructions, Shot Selector, Run Folder, …)
```

`variation/` is the first sub-package organised around a **use case** rather than a
backend. It owns no model and no API client: it calls `codex_provider`'s LLM and image
nodes like any other consumer would. That is the shape to copy for the next pipeline —
providers stay thin and swappable, use cases compose them.

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

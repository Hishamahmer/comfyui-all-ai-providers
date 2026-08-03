# Example workflows

Drop workflow `.json` files here — ComfyUI **canvas** exports (Workflow → Export), not
API-format graphs, so they open by drag-and-drop onto the canvas.

## Conventions

- One `.json` per workflow, named after what it does.
- Add a row to the table below.
- **Never commit an API key.** Leave every `api_token` field blank — the nodes fall back
  to `REPLICATE_API_TOKEN` from the environment or a `.env` file. Workflow JSON stores
  widget values verbatim, so a pasted token *would* be committed.
- Absolute paths from your own machine leak your folder layout; prefer relative
  `filename_prefix` values where a node allows it.

## Workflows

| Workflow | Nodes it demonstrates | Notes |
|---|---|---|
| **Character Dataset (GPT-Image-2)** | Replicate Image Gen · Image Gen Settings · Shot Selector · Run Folder · Text File Save | Builds a 25-image character LoRA training dataset from a single photo: 24 generated shots + the real reference tile, each with a caption `.txt`. |
| **Character Dataset (GPT-Image-2) - Single Shot** | the same, with one API node | Prompt-testing rig: iterate a shot for one paid call instead of 24. |
| **Character Dataset (GPT-Image-2) - Codex** | Codex Image Gen · Image Gen Settings · Shot Selector · Run Folder · Text File Save | Identical to the 24-shot workflow but generated through your **ChatGPT/Codex login** — no API key, billed to your ChatGPT plan. |
| **Character Dataset (GPT-Image-2) - Single Shot - Codex** | the same, with one API node | One-call prompt testing on the ChatGPT login. |

### The Codex twins

Same graph, same prompts, same references, same saves and captions — only the generator
differs, so the two are directly comparable.

Run `codex login` once in a terminal; the node reads the CLI's own `~/.codex/auth.json`.
There is **no OAuth flow in ComfyUI** and no API key. For several ChatGPT accounts, give
each its own `CODEX_HOME` folder and set the node's `codex_home` per node — the `account`
output names the signed-in email so you can see which login made an image.

> Availability is account-dependent: not every ChatGPT plan can call the hosted image
> tool. The node says so plainly if yours cannot.

### Character Dataset (GPT-Image-2)

One photo in, a **LoRA-ready dataset folder** out.

**Needs:** a Replicate token (`REPLICATE_API_TOKEN` in your environment or a `.env` in the
ComfyUI root), plus KJNodes, WAS Node Suite, ComfyUI-AutoCropFaces and
comfyui-mickmumpitz-nodes.

**Set before running** — the shipped values are placeholders:

| Control | Ships as | Set to |
|---|---|---|
| `Load Image` (group 1) | `ccc_ref_placeholder.png` | your character photo (will show as missing until you do) |
| `NAME` (group 0) | `MyCharacter` | your character name — drives every filename, and doubles as the LoRA trigger word |
| `folder_name` (group 0) | `dataset` | your dataset name |
| shots to run (group 0) | `24` | **start with 2–3** — a full run is 24 paid API calls. The Shot Selector defaults to `first N in order`, so `3` runs the first three shots |

**Output:** `ComfyUI/output/CCC/<folder_name>_001/` with `<NAME>_<shot>_image_001.png`
… `_025.png`, **each with a matching `.txt` caption**. Each Run makes a new numbered folder.

**Captions** use `NAME` as the trigger word and describe only pose / expression / wardrobe /
background — never face, hair or eye colour, since anything captioned is *excluded* from
what the LoRA learns. The reference tile is captioned with the trigger word alone.

### The prompts

Every one of the 24 shots is a **JSON "system instruction"** rather than a paragraph of
prose. Each carries its own `framing`, `style_description` (aesthetic, fashion, lighting,
photographic treatment, colour palette), a `compositional_deconstruction` with bounding
boxes, and per-slot `constraints`.

They describe the **photograph, not the person** — face, hair, skin and build always come
from your reference image, and the garments drape to whatever build it shows. That is what
lets one fixed prompt set work for any subject. The prompts are gender-neutral throughout;
gender is stated once, in the **Subject** node, and flows to all 24.

**Shot list** is balanced for character-LoRA training:

- **Framing** — 5 tight (close-up / extreme close-up) · 5 waist-up · 6 three-quarter
  length · 8 full body
- **Angles** — 2 profiles, 1 back angle, 1 straight-down overhead, 2 low, 1 high
- **24 distinct wardrobes** and 24 distinct locations, so the model learns the person
  rather than the clothes or the room
- **24 distinct looks** — black-and-white editorial, pastel beauty, glasshouse, seaside
  backlit, street candid, studio campaign, phone selfie, quiet-luxury interior, festive
  ethnic wear, music room, automotive, 70s funk, creator wall, corporate lobby,
  laundromat, marble plaza, atelier, cobalt studio, overhead, direct flash, rain glass,
  running track, poolside, snow

Three clauses are shared by all 24 and are the place to edit global behaviour: **content
safety** (keeps every shot inside what an image model will generate), **rendering** (real
photographic texture, no glossy AI-render look) and **white balance** (neutral, never a
warm cast).

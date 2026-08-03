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
| **Character Dataset (GPT-Image-2) - Codex** | Codex Image Gen · Image Gen Settings · Shot Selector · Run Folder · Text File Save | Builds a 25-image character LoRA training dataset from a single photo: 24 generated shots + the real reference tile, each with a caption `.txt`. Generated through your **ChatGPT/Codex login** — no API key. |

### Requirements — this workflow will not run without these

1. **A paid ChatGPT subscription.** The free tier cannot call the hosted image tool.
   **ChatGPT Plus at $20/month is the recommended plan.**
2. **The Codex CLI signed in on the machine that runs ComfyUI**, under the same user
   account:
   ```sh
   codex login
   ```
   The node reads the CLI's own `~/.codex/auth.json`. There is **no OAuth flow in ComfyUI
   and no API key field** — if you have not logged in from a terminal first, every shot
   fails immediately. Verify with the **Codex Login Status** node, which reports the
   account, plan and token expiry.

> Availability is account-dependent: not every ChatGPT plan or region can call the hosted
> image tool. The node says so plainly if yours cannot.

Running several ChatGPT accounts? Give each its own `CODEX_HOME` folder and set the node's
`codex_home` per node — the `account` output names the signed-in email so you can see
which login made an image.

Prefer an API key? Swap every **Codex Image Gen** node for **Replicate Image Gen** — the
graph is otherwise identical, and the widgets map by name (`prompt`, `aspect_ratio`,
`quality`, `background`, `timeout_seconds`, `run_mode`, `max_concurrent`).

### Character Dataset (GPT-Image-2) - Codex

One photo in, a **LoRA-ready dataset folder** out.

**Needs:** a ChatGPT login via `codex login`, plus KJNodes, WAS Node Suite,
ComfyUI-AutoCropFaces and comfyui-mickmumpitz-nodes.

**Set before running** — the shipped values are placeholders:

| Control | Ships as | Set to |
|---|---|---|
| `Load Image` (group 1) | `ccc_ref_placeholder.png` | your character photo (will show as missing until you do) |
| `NAME` (group 0) | `MyCharacter` | your character name — drives every filename, and doubles as the LoRA trigger word |
| `folder_name` (group 0) | `dataset` | your dataset name |
| shots to run (group 0) | `24` | **start with 2–3** — a full run is 24 image generations against your ChatGPT plan. The Shot Selector defaults to `first N in order`, so `3` runs the first three shots |

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

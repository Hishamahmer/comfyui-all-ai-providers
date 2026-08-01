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

### Character Dataset (GPT-Image-2)

One photo in, a **LoRA-ready dataset folder** out.

**Needs:** a Replicate token (`REPLICATE_API_TOKEN` in your environment or a `.env` in the
ComfyUI root), plus KJNodes, WAS Node Suite, ComfyUI-AutoCropFaces and
comfyui-mickmumpitz-nodes.

**Set before running** — the shipped values are placeholders:

| Control | Ships as | Set to |
|---|---|---|
| `Load Image` (group 1) | `ccc_ref_placeholder.png` | your character photo (will show as missing until you do) |
| `NAME` (group 0) | `Al1n4_02` | your character name — drives every filename |
| `folder_name` (group 0) | `dataset` | your dataset name |
| shots to run (group 0) | `24` | **start with 2–3** — a full run is 24 paid API calls |

**Output:** `ComfyUI/output/CCC/<folder_name>_001/` with `<NAME>_<shot>_image_001.png`
… `_025.png`, **each with a matching `.txt` caption**. Each Run makes a new numbered folder.

**Captions** use `NAME` as the trigger word and describe only pose / expression / wardrobe /
background — never face, hair or eye colour, since anything captioned is *excluded* from
what the LoRA learns. The reference tile is captioned with the trigger word alone.

**Shot list** is balanced for character-LoRA training — roughly front 46% /
three-quarter 29% / profile 8% / back 8% / high-low 8%, a mix of close-up, waist-up and
full-body framing, and **a different outfit in every shot** so the model learns the person
rather than the clothes.

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
| arkennemasis/**Utility** | arkennemasis Color Picker | pick a color: drag a pin on the image, screen eyedropper, or manual hex | `STRING` ×3 + `IMAGE` |
| arkennemasis/**Utility** | arkennemasis Palette Analyzer | top-N dominant colors of an image | `STRING` ×2 + `IMAGE` |

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

### Utility: color tools

- **Color Picker** — three ways to pick: **drag the pin** on the image shown on the node
  (live #hex/RGB readout; exact full-res value sampled at run time), **🎯 pick from screen**
  (Chromium eyedropper — works over anything on your screen), or type a **manual hex**.
  Outputs `hex` / `rgb` / `prompt_text` strings for LLM prompts and a solid `swatch` IMAGE
  for image nodes. Need several colors? Duplicate the node — one pin each.
  The image preview appears instantly when fed by Load Image; for generated images it
  appears after the first run.
- **Palette Analyzer** — top-N dominant colors → numbered list with hex/RGB/share (for
  LLM prompts), a compact hex list, and a palette-strip IMAGE.

## Notes

- Replicate calls are **paid** — each run bills your Replicate account.
- Long runs poll (no fixed timeout) and run off the UI thread, so ComfyUI stays responsive
  and **Cancel** works. A spinner + elapsed-time badge shows on the node while it runs.
- GPT-5 models output **text only**; image generation is done by the Image Gen node.

## Repo layout (for contributors)

```
common/             shared + utility nodes: System Instructions, color tools, key/.env resolution
replicate_provider/ Replicate (OpenAI) nodes
web/                front-end: activity badge, color-picker pin UI
```
Add a module = new package exposing `NODE_CLASS_MAPPINGS` / `NODE_DISPLAY_NAME_MAPPINGS`,
then register it in `__init__.py`. Put new nodes under an `arkennemasis/<category>` menu.

## License

MIT

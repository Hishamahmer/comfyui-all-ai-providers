# ComfyUI — All AI Providers

ComfyUI nodes for calling multiple AI providers. **Currently bundled: Replicate** (OpenAI
GPT-5 LLM + `gpt-image-2`) plus a provider-neutral **System Instructions** node. Built to
grow — Ollama, Fal, etc. drop in as new providers.

| Node | Menu | Model(s) | Out |
|---|---|---|---|
| **OpenAI LLM (Replicate)** | Replicate/OpenAI | GPT-5 family: `gpt-5`, `-mini`, `-nano`, `-pro`, `-structured`, `5.1`, `5.2`, `5.4`, `5.6-luna/terra/sol` | `STRING` |
| **OpenAI GPT-Image-2 (Replicate)** | Replicate/OpenAI | `openai/gpt-image-2` (text→image **and** edit) | `IMAGE` |
| **System Instructions** | AI/Prompt | — (holds reusable text; works with any LLM) | `STRING` |

## Install

```sh
cd ComfyUI/custom_nodes
git clone https://github.com/Hishamahmer/comfyui-all-ai-providers
```

Install the dependency, then restart ComfyUI:

```sh
# portable build:
python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\comfyui-all-ai-providers\requirements.txt
# normal install:
pip install replicate
```

Or in **ComfyUI-Manager** → *Install via Git URL* → paste the repo URL (deps auto-install).

## API keys — three ways (pasting is optional)

Keys are read in this order: **node field → OS env var → `.env` file**. Use whichever you like.

1. **Paste** into a node's `api_token` field, or
2. **Env var:** set `REPLICATE_API_TOKEN`, or
3. **`.env` file:** copy `.env.example` → `.env` (in this folder or your ComfyUI root) and add:
   ```
   REPLICATE_API_TOKEN=r8_your_token_here
   ```

Get a Replicate token at https://replicate.com/account/api-tokens. (`.env` is gitignored.)

## Usage

Search "Replicate" or "System Instructions" in the node menu. Typical flow:

```
System Instructions ─► OpenAI LLM (system_prompt)
Text  ───────────────► OpenAI LLM (prompt)
Image(s) ────────────► OpenAI LLM (image_1..4)   ← vision
                            │ text
                            ▼
                      GPT-Image-2 (prompt)  ◄─ image_1..4 (edit/reference)
                            │
                        Save Image
```

- **LLM** takes a text prompt, optional system prompt, and up to 4 image inputs; returns text.
- **GPT-Image-2** takes a prompt + optional images to edit; returns an image. Wire the LLM's
  text into its `prompt`, or type one directly.
- **System Instructions** outputs reusable text → wire into any LLM's system prompt.

Optional params (`quality`, `aspect_ratio`, `reasoning_effort`, …) left on **`default`** are
not sent, so the model's own defaults apply. `timeout_seconds = 0` waits indefinitely.

## Notes

- Replicate calls are **paid** — each run bills your Replicate account.
- Long runs poll (no fixed timeout) and run off the UI thread, so ComfyUI stays responsive
  and **Cancel** works. A spinner + elapsed-time badge shows on the node while it runs.
- GPT-5 models output **text only**; image generation is done by the GPT-Image-2 node.

## Repo layout (for contributors)

```
common/             shared, provider-neutral: System Instructions, image utils, key/.env resolution
replicate_provider/ Replicate (OpenAI) nodes
web/                activity badge (front-end)
```
Add a provider = new `*_provider/` package exposing `NODE_CLASS_MAPPINGS` /
`NODE_DISPLAY_NAME_MAPPINGS`, then register it in `__init__.py`.

## License

MIT

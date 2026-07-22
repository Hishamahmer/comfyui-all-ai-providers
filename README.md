# Replicate OpenAI — ComfyUI nodes

Call OpenAI models hosted on [Replicate](https://replicate.com) from ComfyUI: GPT-5 family
(text) and `gpt-image-2` (image generate/edit). Three nodes, under the **Replicate/OpenAI**
menu.

| Node | Model(s) | Out |
|---|---|---|
| **OpenAI LLM (Replicate)** | GPT-5 family: `gpt-5`, `-mini`, `-nano`, `-pro`, `-structured`, `5.1`, `5.2`, `5.4`, `5.6-luna/terra/sol` | `STRING` |
| **OpenAI GPT-Image-2 (Replicate)** | `openai/gpt-image-2` (text→image **and** image edit) | `IMAGE` |
| **System Instructions (Replicate/OpenAI)** | — (holds reusable text) | `STRING` |

## Install

```sh
cd ComfyUI/custom_nodes
git clone https://github.com/Hishamahmer/replicate-comfyui
```

Install the one dependency, then restart ComfyUI:

```sh
# portable build:
python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\replicate-comfyui\requirements.txt
# normal install:
pip install replicate
```

Or in **ComfyUI-Manager** → *Install via Git URL* → paste the repo URL (deps auto-install).

## API token

Get one at https://replicate.com/account/api-tokens, then either:
- paste it into a node's **`api_token`** field, or
- set the `REPLICATE_API_TOKEN` environment variable.

## Usage

Drop the nodes in (search "Replicate"). Typical flow:

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
  text into its `prompt` to auto-generate the image prompt, or type one directly.
- **System Instructions** just outputs reusable text → wire into the LLM's `system_prompt`.

Optional params (`quality`, `aspect_ratio`, `reasoning_effort`, …) left on **`default`** are
not sent, so the model's own defaults apply. `timeout_seconds = 0` waits indefinitely.

## Notes

- These are **paid** Replicate API calls — each run bills your Replicate account.
- Long runs are handled by polling (no fixed timeout) and run off the UI thread, so ComfyUI
  stays responsive and **Cancel** works. A spinner + elapsed-time badge shows on the node
  while it runs.
- GPT-5 models output **text only**; image generation is done by the GPT-Image-2 node.

## License

MIT

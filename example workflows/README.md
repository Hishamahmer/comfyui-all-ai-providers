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
| _(none yet)_ | | |

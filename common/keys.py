"""Shared API-key resolution.

Every provider resolves its key the same way, in priority order:
  1. the node's own field (paste)             -> optional
  2. an OS environment variable               -> optional
  3. a `.env` file (repo root or ComfyUI root) -> optional

So pasting a key is never required — set an env var or a .env file instead.
No external dependency (python-dotenv not needed); the .env parser is stdlib.
"""

import os


def _dotenv_paths():
    here = os.path.dirname(os.path.abspath(__file__))       # .../common
    repo_root = os.path.dirname(here)                        # the custom-node folder
    seen, out = set(), []
    for p in (os.path.join(repo_root, ".env"), os.path.join(os.getcwd(), ".env")):
        ap = os.path.abspath(p)
        if ap not in seen:
            seen.add(ap)
            if os.path.isfile(ap):
                out.append(ap)
    return out


def _parse_dotenv(path):
    data = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                if line.lower().startswith("export "):
                    line = line[7:].strip()
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key:
                    data[key] = val
    except Exception:
        pass
    return data


def dotenv_values():
    """Merged key/values from .env files (repo root, then ComfyUI cwd)."""
    merged = {}
    for p in _dotenv_paths():
        merged.update(_parse_dotenv(p))
    return merged


def resolve_key(field_value, *env_vars):
    """First non-empty of: node field, then env_vars (OS env, then .env). '' if none."""
    v = (field_value or "").strip()
    if v:
        return v
    for name in env_vars:
        v = os.environ.get(name, "").strip()
        if v:
            return v
    dv = dotenv_values()
    for name in env_vars:
        v = (dv.get(name, "") or "").strip()
        if v:
            return v
    return ""

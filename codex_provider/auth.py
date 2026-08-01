"""Reuse the Codex CLI's ChatGPT login instead of an API key.

`codex login` writes an OAuth credential file. This reads it, keeps the access token
fresh, and builds the headers the ChatGPT Codex backend expects. **No OAuth flow is
implemented here** — logging in stays the CLI's job, which is the point: one login,
reused.

Credential file (first hit wins):
  1. ``$CODEX_HOME/auth.json``
  2. ``~/.codex/auth.json``

Shape written by the CLI::

    {"auth_mode": "...", "OPENAI_API_KEY": null,
     "tokens": {"id_token": "...", "access_token": "...",
                "refresh_token": "...", "account_id": "..."},
     "last_refresh": "..."}

**Why refreshes are persisted:** the token endpoint may return a *rotated*
``refresh_token``. Refreshing without saving it would strand the old one and break the
login the next time it rotates. Writes are atomic (``.part`` + ``os.replace``) and
preserve every key we did not touch, since the CLI owns this file too.

Nothing here logs or returns a token value.
"""

import base64
import json
import os
import threading
import time

# The Codex CLI's public OAuth client id — an identifier, not a secret.
OAUTH_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
OAUTH_TOKEN_URL = "https://auth.openai.com/oauth/token"

# Cloudflare in front of chatgpt.com/backend-api/codex only admits first-party
# originators; a generic client gets a 403 challenge no matter how good the token is.
ORIGINATOR = "codex_cli_rs"
USER_AGENT = "codex_cli_rs/0.0.0 (comfyui-arkennemasis)"

# Refresh this long before the JWT actually expires, so a long render never dies mid-call.
EXPIRY_MARGIN_SECONDS = 300

_lock = threading.Lock()


class CodexAuthError(RuntimeError):
    """No usable Codex login. The message says how to fix it."""


def auth_path(codex_home=""):
    """Path to the Codex credential file."""
    home = (codex_home or "").strip() or os.environ.get("CODEX_HOME", "").strip()
    base = home or os.path.join(os.path.expanduser("~"), ".codex")
    return os.path.join(base, "auth.json")


def _read(path):
    if not os.path.isfile(path):
        raise CodexAuthError(
            "No Codex login found at %s. Install the Codex CLI and run `codex login`, "
            "or set CODEX_HOME / the node's codex_home to the folder holding auth.json."
            % path
        )
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        raise CodexAuthError("Could not read %s: %s" % (path, exc))
    if not isinstance(data, dict) or not isinstance(data.get("tokens"), dict):
        raise CodexAuthError(
            "%s has no `tokens` object. Run `codex login` to re-authenticate." % path)
    return data


def _write_atomic(path, data):
    tmp = path + ".part"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def jwt_claims(token):
    """Claims from a JWT access token, or {} if it isn't one."""
    try:
        parts = token.split(".")
        if len(parts) < 2:
            return {}
        payload = parts[1] + "=" * (-len(parts[1]) % 4)
        return json.loads(base64.urlsafe_b64decode(payload))
    except Exception:
        return {}


def _expires_at(token):
    exp = jwt_claims(token).get("exp")
    return float(exp) if isinstance(exp, (int, float)) and exp else 0.0


def _refresh(refresh_token, timeout=30.0):
    """Exchange a refresh token. Returns {'access_token', 'refresh_token'}."""
    import httpx

    if not (refresh_token or "").strip():
        raise CodexAuthError(
            "Codex login has no refresh_token. Run `codex login` to re-authenticate.")
    try:
        resp = httpx.post(
            OAUTH_TOKEN_URL,
            headers={"Content-Type": "application/x-www-form-urlencoded",
                     "Accept": "application/json", "User-Agent": USER_AGENT},
            data={"grant_type": "refresh_token",
                  "refresh_token": refresh_token.strip(),
                  "client_id": OAUTH_CLIENT_ID},
            timeout=httpx.Timeout(max(5.0, timeout)),
        )
    except Exception as exc:
        raise CodexAuthError("Codex token refresh could not reach %s: %s"
                             % (OAUTH_TOKEN_URL, exc))

    if resp.status_code == 429:
        raise CodexAuthError(
            "Codex token endpoint is rate limited (429). Your login is still valid — "
            "retry once the limit resets.")
    if resp.status_code != 200:
        raise CodexAuthError(
            "Codex token refresh failed (HTTP %s). Run `codex login` to re-authenticate."
            % resp.status_code)
    try:
        payload = resp.json()
    except Exception:
        raise CodexAuthError("Codex token refresh returned invalid JSON.")

    access = (payload.get("access_token") or "").strip()
    if not access:
        raise CodexAuthError("Codex token refresh returned no access_token.")
    # The endpoint MAY rotate the refresh token; keep the new one when it does.
    rotated = (payload.get("refresh_token") or "").strip()
    return {"access_token": access, "refresh_token": rotated or refresh_token.strip()}


def get_access_token(codex_home="", allow_refresh=True, timeout=30.0):
    """A currently-valid Codex access token.

    Refreshes and persists when the stored one is expired or nearly so. Serialised with
    a lock so two nodes running at once cannot both refresh and race the file.
    """
    path = auth_path(codex_home)
    with _lock:
        data = _read(path)
        tokens = data["tokens"]
        access = (tokens.get("access_token") or "").strip()
        exp = _expires_at(access)

        fresh = access and (not exp or time.time() < exp - EXPIRY_MARGIN_SECONDS)
        if fresh:
            return access

        if not allow_refresh:
            raise CodexAuthError(
                "Codex access token is expired and allow_refresh is off. "
                "Run `codex login`, or turn allow_refresh on.")

        new = _refresh(tokens.get("refresh_token") or "", timeout=timeout)
        rotated = new["refresh_token"] != (tokens.get("refresh_token") or "").strip()

        # Re-read before writing: the CLI may have refreshed while we were on the wire.
        # Preserve every other key — this file is not ours alone.
        try:
            latest = _read(path)
        except CodexAuthError:
            latest = data
        latest.setdefault("tokens", {}).update(new)
        latest["last_refresh"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        try:
            _write_atomic(path, latest)
            print("[arkennemasis] refreshed Codex token%s"
                  % (" (refresh token rotated)" if rotated else ""))
        except Exception as exc:
            # Losing a ROTATED refresh token breaks the next refresh, so say so loudly.
            print("[arkennemasis] WARNING: refreshed the Codex token but could not save "
                  "it to %s (%s).%s" % (path, exc,
                  " The refresh token rotated, so run `codex login` if auth starts failing."
                  if rotated else ""))
        return new["access_token"]


def account_email(token):
    """Signed-in ChatGPT email from the token claims, or '' if the claim is absent."""
    claims = jwt_claims(token)
    profile = claims.get("https://api.openai.com/profile")
    if isinstance(profile, dict):
        value = profile.get("email")
        if isinstance(value, str) and value.strip():
            return value.strip()
    value = claims.get("email")
    return value.strip() if isinstance(value, str) else ""


def account_id(access_token):
    """ChatGPT account id from the token's claims, or '' if absent."""
    claims = jwt_claims(access_token)
    auth = claims.get("https://api.openai.com/auth")
    if isinstance(auth, dict):
        for key in ("chatgpt_account_id", "account_id"):
            value = auth.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    value = claims.get("chatgpt_account_id")
    return value.strip() if isinstance(value, str) else ""


def request_headers(access_token):
    """Headers the ChatGPT Codex backend requires (see ORIGINATOR above)."""
    headers = {
        "Authorization": "Bearer %s" % access_token,
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "User-Agent": USER_AGENT,
        "originator": ORIGINATOR,
    }
    acct = account_id(access_token)
    if acct:
        headers["ChatGPT-Account-ID"] = acct       # casing matches codex-rs auth.rs
    return headers


def describe(codex_home=""):
    """Non-secret summary for the status node: never returns a token."""
    path = auth_path(codex_home)
    try:
        data = _read(path)
    except CodexAuthError as exc:
        return {"ok": False, "path": path, "detail": str(exc)}
    tokens = data.get("tokens") or {}
    access = (tokens.get("access_token") or "").strip()
    exp = _expires_at(access)
    claims = jwt_claims(access)
    auth = claims.get("https://api.openai.com/auth") or {}
    left = (exp - time.time()) / 3600.0 if exp else None
    id_token = (tokens.get("id_token") or "").strip()
    return {
        "ok": bool(access),
        "path": path,
        "email": account_email(id_token) or account_email(access) or "unknown",
        "has_refresh_token": bool((tokens.get("refresh_token") or "").strip()),
        "account_id_present": bool(account_id(access)),
        "plan": auth.get("chatgpt_plan_type") or "unknown",
        "expires_in_hours": round(left, 1) if left is not None else None,
        "expired": bool(exp and time.time() > exp),
    }

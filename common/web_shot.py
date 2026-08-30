"""Photograph a web page, read its text, and list its links — using the installed browser.

An automated news pipeline needs a picture of the story it is talking about. The hosted
answer is a screenshot API, which means an account, a key, a per-shot charge and a third
party that has to be up, all to render a page this machine can already render.

Chrome and Edge ship a headless mode and a debugging protocol. So this drives one of them
over **CDP**, using the `aiohttp` websocket client ComfyUI already depends on. No pip
install, no key.

CDP rather than the simpler `--screenshot` flag, for one reason that matters:

**A screenshot at scroll-zero is a navigation bar, not a headline.** The hosted service
this replaces knew that — its call carried
`scroll_into_view=body > div.wp-site-blocks > header > div > div` and
`scroll_into_view_adjust_top=500`, which scrolls past a WordPress site header and lands on
the article title. Hard-wiring one publisher's selector is why "only some sources
screenshot correctly". Here `scroll_to` takes a **list of selectors tried in order**, so a
site-specific one can lead and a near-universal `h1` can catch everything else.

Three outputs, because a discovery step and a writing step want different things:

    image       what the page looks like, framed on the headline
    page_text   `document.body.innerText` — what it says, already rendered, so no tag
                stripping and no JavaScript left in the middle of the article
    links       `title <TAB> url` per line — what a model needs to pick one story
                out of a section page and hand its URL to the next stage

Nothing here is specific to news: a URL in, a picture, the words, and the links out.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import re
import socket
import subprocess
import tempfile
import threading

import numpy as np
import torch
from PIL import Image

# Where Chrome and Edge actually install on Windows. Chrome first because its headless is
# the one Google maintains; Edge is the fallback present on every Windows 11 machine.
BROWSERS = (
    ("chrome", r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    ("chrome", r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
    ("edge", r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    ("edge", r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
)

AUTO = "auto"
CHOICES = [AUTO, "chrome", "edge"]

WHEN_CHANGED = "when the inputs change"
ONCE_A_DAY = "once a day"
EVERY_RUN = "every run"
REFRESH = [WHEN_CHANGED, ONCE_A_DAY, EVERY_RUN]

PHONE_UA = ("Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) "
            "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 "
            "Safari/604.1")

# Tried in order, first hit wins. The WordPress block-theme wrapper leads because it is
# what the hosted call targeted and it frames a TechCrunch-style article exactly; `h1` is
# the catch-all, because an article headline is an h1 on essentially every publisher.
DEFAULT_SCROLL_TO = "body > div.wp-site-blocks > header > div > div, article h1, main h1, h1"

# Consent walls, by platform. Removing them is not cosmetic: a consent overlay is usually
# 90% of the picture AND it sets `overflow: hidden` on the body, which silently defeats
# the scroll — that is why several publishers came back framed at y=0 with a wall of
# cookie text at the front of `innerText`. The hosted service this replaces had
# `block_cookie_banners=true` for exactly this reason.
CONSENT_SELECTORS = ",".join([
    "#CybotCookiebotDialog", "#CybotCookiebotDialogBodyUnderlay",   # Cookiebot
    "#onetrust-consent-sdk", "#onetrust-banner-sdk",                # OneTrust
    ".qc-cmp2-container", "#qc-cmp2-container",                     # Quantcast
    "#truste-consent-track",                                        # TrustArc
    ".osano-cm-window",                                             # Osano
    "#didomi-host",                                                 # Didomi
    "[id^='sp_message_container']", ".sp_message_container",        # Sourcepoint
    "#cmplz-cookiebanner-container",                                # Complianz
    "#usercentrics-root", "#cookie-law-info-bar",
    "[class*='cookie-consent']", "[class*='cookie-banner']",
    "[id*='cookie-banner']", "[class*='gdpr']",
])

# Injected once, after load. Returns everything the node needs in a single round trip so
# the page is measured in the state it is photographed in.
PAGE_SCRIPT = r"""
(function (selectors, adjustTop, linkLimit, consent) {
  var killed = 0;
  function drop(el) {
    if (el && el.parentNode) { el.parentNode.removeChild(el); killed++; }
  }
  // 1 - known consent platforms, by selector.
  try {
    var walls = document.querySelectorAll(consent);
    for (var w = 0; w < walls.length; w++) drop(walls[w]);
  } catch (e) {}
  // 2 - anything PINNED and LARGE whose own text is about consent. Catches the platforms
  //     not on the list without touching ordinary fixed furniture like a nav bar, which
  //     is neither large nor talking about cookies.
  try {
    var all = document.querySelectorAll("div,section,aside");
    var vw = window.innerWidth, vh = window.innerHeight;
    for (var k = 0; k < all.length; k++) {
      var el = all[k], cs = window.getComputedStyle(el);
      if (cs.position !== "fixed" && cs.position !== "sticky") continue;
      var r = el.getBoundingClientRect();
      if (r.width * r.height < vw * vh * 0.25) continue;
      var t = (el.innerText || "").slice(0, 400).toLowerCase();
      if (t.indexOf("cookie") >= 0 || t.indexOf("consent") >= 0 ||
          t.indexOf("privacy") >= 0 || t.indexOf("subscribe") >= 0) drop(el);
    }
  } catch (e) {}
  // 3 - give the scroll back. A wall that is gone can still have left the body locked.
  try {
    document.documentElement.style.overflow = "auto";
    document.body.style.overflow = "auto";
    document.body.style.position = "static";
  } catch (e) {}

  var scrolled = null;
  var list = selectors.split(",").map(function (s) { return s.trim(); })
                      .filter(function (s) { return s.length; });
  for (var i = 0; i < list.length; i++) {
    var el = null;
    try { el = document.querySelector(list[i]); } catch (e) { el = null; }
    if (el) {
      var top = el.getBoundingClientRect().top + window.scrollY;
      window.scrollTo(0, Math.max(0, top + adjustTop));
      scrolled = list[i];
      break;
    }
  }
  var seen = {}, links = [];
  var anchors = document.querySelectorAll("a[href]");
  // Section machinery, not stories. These sit at the TOP of most publisher pages, so
  // leaving them in pushes the actual headlines down — and the top of the list is the
  // only recency signal a reader has, because a link carries no publication date.
  var SECTION = /\/(tag|tags|category|categories|topic|topics|author|authors|section|page|feed|subscribe|newsletter|jobs|events|about|contact|privacy|terms)\//i;
  for (var j = 0; j < anchors.length && links.length < linkLimit; j++) {
    var a = anchors[j];
    var href = a.href;
    if (!href || href.indexOf("http") !== 0) continue;
    if (href.indexOf("#") === 0) continue;
    if (SECTION.test(href)) continue;
    var text = (a.innerText || a.textContent || "").replace(/\s+/g, " ").trim();
    // A headline is a sentence; navigation is a word. Anything under 25 characters is
    // menu furniture, and keeping it buries the real stories in the list a model reads.
    if (text.length < 25) continue;
    var key = href.split("#")[0];
    if (seen[key]) continue;
    seen[key] = 1;
    links.push(text + "\t" + key);
  }
  return JSON.stringify({
    title: document.title || "",
    text: (document.body ? document.body.innerText : "") || "",
    links: links,
    scrolled: scrolled,
    scrollY: window.scrollY,
    height: document.body ? document.body.scrollHeight : 0
  });
})(%s, %s, %s, %s)
"""


def find_browser(preference=AUTO):
    """(kind, exe) for the first installed browser, honouring an explicit choice."""
    for kind, path in BROWSERS:
        if preference not in (AUTO, kind):
            continue
        if os.path.isfile(path):
            return kind, path
    raise RuntimeError(
        "ArkWebShot found no browser to drive. Looked for:\n  "
        + "\n  ".join(p for _, p in BROWSERS)
        + "\nInstall Chrome or Edge, or set `browser_path` to the executable.")


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _CDP:
    """The smallest CDP client that does this job. One page, one navigation, one shot."""

    def __init__(self, ws):
        self.ws = ws
        self.n = 0

    async def call(self, method, params=None, timeout=60):
        self.n += 1
        wanted = self.n
        await self.ws.send_json({"id": wanted, "method": method, "params": params or {}})
        while True:
            raw = await asyncio.wait_for(self.ws.receive(), timeout=timeout)
            if raw.data is None:
                raise RuntimeError("ArkWebShot: the browser closed the connection.")
            msg = json.loads(raw.data)
            if msg.get("id") == wanted:
                if "error" in msg:
                    raise RuntimeError("ArkWebShot: %s failed — %s"
                                       % (method, msg["error"].get("message")))
                return msg.get("result", {})

    async def until(self, event, timeout=60):
        """Wait for one event, ignoring command replies that arrive meanwhile."""
        try:
            while True:
                raw = await asyncio.wait_for(self.ws.receive(), timeout=timeout)
                if raw.data is None:
                    return
                if json.loads(raw.data).get("method") == event:
                    return
        except asyncio.TimeoutError:
            # A page that never fires `load` is usually still perfectly photographable —
            # a stalled tracker or a video keeps the event pending forever. Carry on.
            return


async def _session(port, url, width, height, scale, mobile, agent, wait_seconds,
                   scroll_to, adjust_top, link_limit, timeout):
    import aiohttp

    async with aiohttp.ClientSession() as http:
        # The browser needs a moment to open its debugging port.
        target = None
        # Capped independently of `timeout`. A browser that has not opened its debugging
        # port within half a minute is not going to, and spending the caller's whole
        # timeout discovering that turns a clear error into a long silence.
        for _ in range(int(min(timeout, 30) * 4)):
            try:
                async with http.get("http://127.0.0.1:%d/json" % port,
                                    timeout=aiohttp.ClientTimeout(total=2)) as r:
                    for t in await r.json():
                        if t.get("type") == "page" and t.get("webSocketDebuggerUrl"):
                            target = t
                            break
                if target:
                    break
            except Exception:
                pass
            await asyncio.sleep(0.25)
        if not target:
            raise RuntimeError(
                "ArkWebShot: the browser never opened its debugging port. Something is "
                "blocking loopback connections, or the executable is not a Chromium.")

        async with http.ws_connect(target["webSocketDebuggerUrl"],
                                   max_msg_size=0) as ws:
            cdp = _CDP(ws)
            await cdp.call("Page.enable")
            # Emulation, not just --window-size: `mobile` is what makes a site serve its
            # phone layout when it branches on the media query rather than the UA.
            await cdp.call("Emulation.setDeviceMetricsOverride", {
                "width": int(width), "height": int(height),
                "deviceScaleFactor": float(scale), "mobile": bool(mobile),
            })
            if agent:
                await cdp.call("Emulation.setUserAgentOverride", {"userAgent": agent})
            await cdp.call("Page.navigate", {"url": url})
            await cdp.until("Page.loadEventFired", timeout=timeout)
            if wait_seconds > 0:
                await asyncio.sleep(float(wait_seconds))

            # FOUR arguments, matching the four the JS declares. It took three for a
            # while, which left `consent` undefined inside the page: every
            # `querySelectorAll(undefined)` threw straight into the try/catch, so the
            # whole targeted consent-wall pass silently did nothing and only the generic
            # heuristic below it was ever running. A JS arity mismatch cannot be caught
            # by anything on the Python side, which is why it survived.
            script = PAGE_SCRIPT % (json.dumps(scroll_to), json.dumps(int(adjust_top)),
                                    json.dumps(int(link_limit)),
                                    json.dumps(CONSENT_SELECTORS))
            got = await cdp.call("Runtime.evaluate",
                                 {"expression": script, "returnByValue": True},
                                 timeout=timeout)
            payload = json.loads((got.get("result") or {}).get("value") or "{}")
            # Let the scroll settle before the shutter — a smooth-scrolling site is still
            # moving when the command returns.
            await asyncio.sleep(0.4)
            shot = await cdp.call("Page.captureScreenshot",
                                  {"format": "png", "captureBeyondViewport": False},
                                  timeout=timeout)
            payload["png"] = shot.get("data") or ""
            return payload


def _tail(path, limit=600):
    """The last of whatever the browser wrote, for an error message.

    Written to a file rather than a pipe precisely so reading it here is safe: there is
    no buffer to fill and nothing to deadlock against.
    """
    try:
        with open(path, "rb") as fh:
            data = fh.read()
    except Exception:
        return "(no browser log)"
    text = data.decode("utf-8", "replace").strip()
    return text[-limit:] if text else "(the browser said nothing)"


def _drive(argv, port, kwargs, timeout, log_path):
    """Run the browser and the CDP session, on a private event loop in its own thread.

    ComfyUI already owns an event loop, and a node's `run` is synchronous. Rather than
    guess how those interact, this gets a loop of its own.

    **Both child streams go to a FILE or DEVNULL, never a pipe**, and that is the whole
    difference between this working and hanging. `stderr=subprocess.PIPE` with nothing
    draining it gives Chrome a 64 KB buffer and then blocks it forever on the next write.
    Standalone that never showed, because Chrome had little to say. Inside ComfyUI it hung
    every time: CUDA is already initialised in the process, Chrome's GPU probing produces
    far more output, the buffer fills, and the page never finishes loading — so the node
    sat until its own timeout with no error to report.

    It is the same failure `video_assemble` hit with ffmpeg and `stdin`, and the same one
    ComfyUI itself hits when its console is contended: a child process blocked on a pipe
    nobody is reading. A file cannot fill.
    """
    errlog = open(log_path, "wb")
    try:
        proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL,
                                stdout=subprocess.DEVNULL, stderr=errlog)
    finally:
        errlog.close()          # the child holds its own handle
    box = {}

    def worker():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            box["value"] = loop.run_until_complete(_session(port=port, timeout=timeout,
                                                            **kwargs))
        except BaseException as exc:            # reported on the calling thread
            box["error"] = exc
        finally:
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            finally:
                loop.close()

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join(timeout + 30)
    try:
        proc.terminate()
        proc.wait(timeout=10)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    if thread.is_alive():
        raise RuntimeError(
            "ArkWebShot: the browser did not finish within %ds.\n%s"
            % (timeout, _tail(log_path)))
    if "error" in box:
        raise box["error"]
    if "value" not in box:
        raise RuntimeError("ArkWebShot: the browser produced nothing.\n%s"
                           % _tail(log_path))
    return box["value"]


def clean_text(text, limit=12000):
    """Tidy `innerText` into something worth putting in a prompt.

    Far less work than it used to be: `innerText` is already the rendered text, so there
    is no markup to strip and no risk of a wall of minified JavaScript in the middle of
    the article. All that is left is collapsing blank runs and dropping the repeated
    furniture a site shows in both its header and its footer.
    """
    lines, seen = [], set()
    for raw in (text or "").splitlines():
        line = " ".join(raw.split())
        if len(line) < 3 or line in seen:
            continue
        seen.add(line)
        lines.append(line)
    return "\n".join(lines)[:limit]


class ArkWebShot:
    CATEGORY = "arkennemasis/Utility"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("image", "page_text", "links", "title", "report")
    DESCRIPTION = ("Screenshot a web page with the Chrome or Edge already installed, and "
                   "return its text and its links. Scrolls to a headline first, so the "
                   "picture is the story rather than a navigation bar. No API key.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {
                    "default": "", "multiline": True,
                    "tooltip": "The page to read. ONE PER LINE for several — their text "
                               "and links are concatenated, and the image comes from the "
                               "first. Blank lines and lines starting with # are ignored, "
                               "so a source can be commented out rather than deleted. "
                               "`https://` is added when the scheme is missing.",
                }),
                "width": ("INT", {
                    "default": 430, "min": 64, "max": 4096,
                    "tooltip": "Viewport width in CSS pixels. 430 is an iPhone 15 Pro "
                               "Max, which is what makes a site serve its phone layout.",
                }),
                "height": ("INT", {
                    "default": 932, "min": 64, "max": 8192,
                    "tooltip": "Viewport height. The capture is the viewport, so this is "
                               "the output height.",
                }),
                "scale": ("FLOAT", {
                    "default": 2.0, "min": 0.5, "max": 4.0, "step": 0.5,
                    "tooltip": "Device pixel ratio. 2.0 renders at retina density, so "
                               "the headline stays legible behind a video.",
                }),
                "wait_seconds": ("FLOAT", {
                    "default": 3.0, "min": 0.0, "max": 60.0, "step": 0.5,
                    "tooltip": "Settling time after load, before scrolling and the "
                               "shutter. Too low and you photograph a spinner.",
                }),
                "scroll_to": ("STRING", {
                    "default": DEFAULT_SCROLL_TO, "multiline": False,
                    "tooltip": "CSS selectors tried IN ORDER; the first that exists is "
                               "scrolled to. This is what puts the headline in frame "
                               "instead of the nav bar. Leave blank to shoot the top.",
                }),
                "scroll_adjust_top": ("INT", {
                    "default": 500, "min": -4000, "max": 8000,
                    "tooltip": "Pixels to scroll PAST the matched element. 500 with a "
                               "site-header selector lands on the article title.",
                }),
            },
            "optional": {
                "mobile": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Emulate a phone. Sites that branch on the user-agent or "
                               "on touch support need this for their mobile layout.",
                }),
                "browser": (CHOICES, {
                    "tooltip": "Which browser to drive. `auto` takes Chrome if it is "
                               "installed and Edge otherwise.",
                }),
                "browser_path": ("STRING", {
                    "default": "",
                    "tooltip": "Full path to a browser executable, when it is somewhere "
                               "this node does not look.",
                }),
                "user_agent": ("STRING", {
                    "default": "",
                    "tooltip": "Override the user-agent. Blank uses a phone UA when "
                               "`mobile` is on, and the browser's own otherwise.",
                }),
                "timeout_seconds": ("INT", {
                    "default": 90, "min": 10, "max": 600,
                    "tooltip": "Hard limit. A hung browser fails here instead of "
                               "stalling the queue silently.",
                }),
                "text_limit": ("INT", {
                    "default": 12000, "min": 500, "max": 200000,
                    "tooltip": "Characters of page text to keep. A whole site is mostly "
                               "navigation; the article is near the front.",
                }),
                "link_limit": ("INT", {
                    "default": 60, "min": 0, "max": 500,
                    "tooltip": "How many links to return. Links under 25 characters of "
                               "anchor text are dropped as navigation, so what is left "
                               "is mostly headlines.",
                }),
                "refresh": (REFRESH, {
                    "tooltip": "How often to actually go and look.\n"
                               "'when the inputs change' caches forever — right when the "
                               "page is a fixed article.\n"
                               "'once a day' refetches on the first run of each new date "
                               "— right for a front page you are mining for today's "
                               "story.\n"
                               "'every run' refetches always, and re-runs EVERYTHING "
                               "downstream of it, including any render.",
                }),
            },
        }

    @classmethod
    def IS_CHANGED(cls, refresh=WHEN_CHANGED, **kwargs):
        """Decide whether ComfyUI may serve this from cache.

        The three settings exist because two obvious answers are both wrong here.

        Cache forever and a "daily" workflow shows yesterday's news — the same URL is a
        different page tomorrow. Never cache and every press of Run re-fetches, which
        invalidates every node downstream: the story pick is re-billed, the script is
        re-billed, and the video re-renders. That makes nudging a composite setting cost a
        full render, which is how a canvas becomes unusable.

        Returning the DATE is the setting that matches what this is for. It is stable
        within a day, so tuning is free; it changes at midnight, so the first run each
        morning genuinely goes and looks.
        """
        if refresh == EVERY_RUN:
            return float("nan")           # never equal to itself, so never cached
        if refresh == ONCE_A_DAY:
            import datetime
            return datetime.date.today().isoformat()
        return False

    def _many(self, urls, width, height, scale, wait_seconds, scroll_to,
              scroll_adjust_top, mobile, browser, browser_path, user_agent,
              timeout_seconds, text_limit, link_limit):
        """Read several pages and hand back one combined candidate list.

        Per source rather than pooled: the link budget is divided so one prolific site
        cannot crowd the others out of the list, and each block is labelled so the picker
        can see which publication a headline came from.

        **A source that fails does not fail the hunt.** Publishers go down, put up a wall,
        or simply time out, and losing today's video because the third of four sources was
        slow would be the wrong trade. The failure is printed and the run continues on
        whatever answered — unless nothing did.
        """
        per_source = max(5, int(link_limit) // len(urls)) if link_limit else 0
        first_image, texts, links, titles, failed = None, [], [], [], []
        for one_url in urls:
            try:
                image, text, link_block, title, _ = self.run(
                    url=one_url, width=width, height=height, scale=scale,
                    wait_seconds=wait_seconds, scroll_to=scroll_to,
                    scroll_adjust_top=scroll_adjust_top, mobile=mobile, browser=browser,
                    browser_path=browser_path, user_agent=user_agent,
                    timeout_seconds=timeout_seconds,
                    text_limit=max(500, int(text_limit) // len(urls)),
                    link_limit=per_source, refresh=WHEN_CHANGED)
            except Exception as exc:
                failed.append("%s (%s)" % (one_url, str(exc)[:80]))
                print("[arkennemasis] web shot: SKIPPING %s — %s" % (one_url, exc))
                continue
            if first_image is None:
                first_image = image
            host = re.sub(r"^https?://(www\.)?", "", one_url).split("/")[0]
            titles.append(title or host)
            if text:
                texts.append("----- %s -----\n%s" % (host, text))
            if link_block:
                links.append(link_block)
        if first_image is None:
            raise RuntimeError(
                "ArkWebShot: every source failed.\n  " + "\n  ".join(failed))
        report = ("%d of %d sources read | %d links%s"
                  % (len(urls) - len(failed), len(urls),
                     sum(len(b.splitlines()) for b in links),
                     "" if not failed else " | FAILED: " + "; ".join(failed)))
        print("[arkennemasis] web shot: %s" % report)
        return (first_image, "\n\n".join(texts), "\n".join(links),
                " | ".join(titles), report)

    def run(self, url, width=430, height=932, scale=2.0, wait_seconds=3.0,
            scroll_to=DEFAULT_SCROLL_TO, scroll_adjust_top=500, mobile=True,
            browser=AUTO, browser_path="", user_agent="", timeout_seconds=90,
            text_limit=12000, link_limit=60, refresh=WHEN_CHANGED):
        # One URL or many. A hunt across three publications is three page loads and one
        # candidate list, which is what lets the picker weigh them against each other —
        # three separate nodes would give it three lists and no way to compare.
        urls = []
        for line in (url or "").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue                       # commented out, not deleted
            if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", line):
                line = "https://" + line
            if line not in urls:
                urls.append(line)
        if not urls:
            raise RuntimeError("ArkWebShot needs at least one `url`.")
        if len(urls) > 1:
            return self._many(urls, width, height, scale, wait_seconds, scroll_to,
                              scroll_adjust_top, mobile, browser, browser_path,
                              user_agent, timeout_seconds, text_limit, link_limit)
        url = urls[0]

        if browser_path.strip():
            exe, kind = browser_path.strip(), "custom"
            if not os.path.isfile(exe):
                raise RuntimeError("ArkWebShot: browser_path does not exist: %s" % exe)
        else:
            kind, exe = find_browser(browser)

        agent = user_agent.strip() or (PHONE_UA if mobile else "")
        port = _free_port()

        # A throwaway profile per call. Two browsers sharing one profile directory is how
        # a headless run exits instantly having written nothing.
        #
        # `ignore_cleanup_errors` is REQUIRED on Windows, not tidiness. Terminating the
        # browser does not take its children with it, and one of them holds the
        # optimization-guide model files open for a moment longer. Without this the
        # directory cannot be removed and `TemporaryDirectory.__exit__` raises
        # PermissionError — AFTER the screenshot has been taken successfully, so a
        # perfectly good capture is thrown away by its own cleanup. It is intermittent,
        # because those files are only present on the runs where Chrome fetched them.
        with tempfile.TemporaryDirectory(prefix="arkwebshot_",
                                         ignore_cleanup_errors=True) as work:
            argv = [
                exe,
                "--headless=new",
                "--disable-gpu",
                "--hide-scrollbars",
                "--no-first-run",
                "--no-default-browser-check",
                "--disable-extensions",
                "--disable-background-networking",
                "--disable-features=Translate,BackForwardCache",
                "--disable-software-rasterizer",
                "--log-level=3",
                "--remote-debugging-port=%d" % port,
                "--user-data-dir=" + os.path.join(work, "profile"),
                "--window-size=%d,%d" % (int(width), int(height)),
                "about:blank",
            ]
            payload = _drive(argv, port, {
                "url": url, "width": width, "height": height, "scale": scale,
                "mobile": mobile, "agent": agent, "wait_seconds": wait_seconds,
                "scroll_to": scroll_to or "", "adjust_top": scroll_adjust_top,
                "link_limit": link_limit,
            }, int(timeout_seconds), os.path.join(work, "browser.log"))

        data = payload.get("png") or ""
        if not data:
            raise RuntimeError("ArkWebShot: %s returned no screenshot for %s."
                               % (kind, url))
        with Image.open(io.BytesIO(base64.b64decode(data))) as opened:
            pil = opened.convert("RGB")
            pixels = np.asarray(pil, dtype=np.float32) / 255.0
        image = torch.from_numpy(pixels)[None, ...]

        text = clean_text(payload.get("text"), int(text_limit))
        title = " ".join((payload.get("title") or "").split())
        links = "\n".join(payload.get("links") or [])

        matched = payload.get("scrolled")
        report = ("%s | %s | %dx%d @%sx | scrolled to %s (y=%s) | %d chars | %d links"
                  % (url, kind, image.shape[2], image.shape[1], scale,
                     matched or "nothing matched", payload.get("scrollY"),
                     len(text), len(payload.get("links") or [])))
        if scroll_to and not matched:
            # Worth saying out loud: it means the picture is whatever is at the top of
            # the page, which on a news site is the navigation bar.
            print("[arkennemasis] web shot: no selector matched on %s — the shot is the "
                  "TOP of the page, not the headline. Add a selector for this site."
                  % url)
        print("[arkennemasis] web shot: %s" % report)
        return (image, text, links, title, report)


NODE_CLASS_MAPPINGS = {"ArkWebShot": ArkWebShot}
NODE_DISPLAY_NAME_MAPPINGS = {"ArkWebShot": "arkennemasis Web Shot (page -> picture, text, links)"}

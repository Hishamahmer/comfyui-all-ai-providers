"""Photograph a web page through ScreenshotOne — the hosted path, ad- and consent-free.

`ArkWebShot` drives the local Chrome and is free, and for reading a page it is the better
tool: it returns the text and the links, which an API that hands back a JPEG cannot. What
it is not good at is the *picture*, because a real publisher page is a consent wall, a
notification prompt and three ad slots stacked over the article. Fighting that with
injected JavaScript is an arms race against every CMP vendor in the world.

ScreenshotOne already won that fight. `block_ads`, `block_cookie_banners` and
`block_trackers` are server-side, and `scroll_into_view` + `scroll_into_view_adjust_top`
put the article headline in frame instead of the navigation bar.

So the two nodes divide the work rather than competing:

    ArkWebShot        the words and the links      free, local, every run
    ArkScreenshotOne  the picture                  billed per shot, only the hero image

The defaults here are the ones from the workflow this was ported from, kept verbatim
because they were arrived at against real pages: an iPhone 15 Pro Max viewport, jpg at
quality 80, ads and consent banners blocked, `scroll_into_view` on the WordPress
block-theme header with a 500px adjustment.

**The key is never typed into the graph.** `resolve_key` reads the node field first, then
the OS environment, then the `.env` at the portable root — so a saved workflow carries no
secret and can be shared as-is.
"""

from __future__ import annotations

import hashlib
import hmac
import io
import urllib.parse
import urllib.request

import numpy as np
import torch
from PIL import Image

from .keys import resolve_key

ENDPOINT = "https://api.screenshotone.com/take"

# The n8n call, parameter for parameter. `scroll_into_view` is the one that decides
# whether the shot is the story or the site's navigation; see ArkWebShot for why a single
# hard-wired selector is also why only some publishers screenshot correctly.
DEFAULT_SCROLL_INTO_VIEW = "body > div.wp-site-blocks > header > div > div"

FORMATS = ["jpg", "png", "webp"]
DEVICES = [
    "iphone_15_pro_max", "iphone_15_pro", "iphone_15", "iphone_14_pro_max",
    "pixel_7", "galaxy_s22_ultra", "ipad_pro_12", "none (use width/height)",
]
NO_DEVICE = "none (use width/height)"


def build_url(url, access_key, *, device=DEVICES[0], width=430, height=932,
              image_format="jpg", image_quality=80, block_ads=True,
              block_cookie_banners=True, block_trackers=True,
              block_banners_by_heuristics=False, delay=0, timeout=60,
              scroll_into_view=DEFAULT_SCROLL_INTO_VIEW, scroll_adjust_top=500,
              full_page=False, secret_key=""):
    """The signed (or unsigned) request URL. Pure, so it can be tested without spending."""
    params = [
        ("url", url),
        ("format", image_format),
        ("image_quality", str(int(image_quality))),
        ("block_ads", "true" if block_ads else "false"),
        ("block_cookie_banners", "true" if block_cookie_banners else "false"),
        ("block_banners_by_heuristics",
         "true" if block_banners_by_heuristics else "false"),
        ("block_trackers", "true" if block_trackers else "false"),
        ("delay", str(int(delay))),
        ("timeout", str(int(timeout))),
        ("response_type", "by_format"),
    ]
    if device and device != NO_DEVICE:
        params += [("viewport_mobile", "true"), ("viewport_device", device)]
    else:
        params += [("viewport_width", str(int(width))),
                   ("viewport_height", str(int(height)))]
    if full_page:
        params.append(("full_page", "true"))
    if scroll_into_view.strip():
        params.append(("scroll_into_view", scroll_into_view.strip()))
        params.append(("scroll_into_view_adjust_top", str(int(scroll_adjust_top))))

    query = urllib.parse.urlencode(params + [("access_key", access_key)])
    if secret_key:
        # Only some accounts enforce signing. When a secret is present it costs nothing
        # to sign, and an account that later turns enforcement on keeps working.
        signature = hmac.new(secret_key.encode(), query.encode(),
                             hashlib.sha256).hexdigest()
        query += "&signature=" + signature
    return ENDPOINT + "?" + query


class ArkScreenshotOne:
    CATEGORY = "arkennemasis/Utility"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "report")
    DESCRIPTION = ("Screenshot a web page through ScreenshotOne, with ads, cookie "
                   "banners and trackers blocked server-side and the article headline "
                   "scrolled into frame. Billed per shot; the key comes from .env.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {
                    "default": "", "multiline": False,
                    "tooltip": "The page to photograph.",
                }),
                "device": (DEVICES, {
                    "tooltip": "Viewport to emulate. The phone devices are what give a "
                               "tall portrait shot that sits behind a vertical video.",
                }),
                "image_format": (FORMATS, {
                    "tooltip": "jpg is what the original used; png if you need a clean "
                               "edge for compositing.",
                }),
                "image_quality": ("INT", {
                    "default": 80, "min": 20, "max": 100,
                    "tooltip": "JPEG quality. Ignored for png.",
                }),
            },
            "optional": {
                "scroll_into_view": ("STRING", {
                    "default": DEFAULT_SCROLL_INTO_VIEW, "multiline": False,
                    "tooltip": "CSS selector to scroll to before the shutter. This is "
                               "what frames the headline instead of the nav bar — and "
                               "the reason a selector that fits one publisher does not "
                               "fit another. Blank shoots the top of the page.",
                }),
                "scroll_adjust_top": ("INT", {
                    "default": 500, "min": -4000, "max": 8000,
                    "tooltip": "Pixels past the matched element. 500 with a site-header "
                               "selector lands on the article title.",
                }),
                "block_ads": ("BOOLEAN", {"default": True}),
                "block_cookie_banners": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "The single most valuable setting here. Without it most "
                               "publisher pages photograph as a consent dialog.",
                }),
                "block_trackers": ("BOOLEAN", {"default": True}),
                "block_banners_by_heuristics": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Aggressive extra pass. Off by default because it also "
                               "removes legitimate page furniture.",
                }),
                "full_page": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Capture the whole scroll height instead of one viewport. "
                               "Produces a very tall image; `scroll_into_view` then has "
                               "nothing to do.",
                }),
                "delay": ("INT", {
                    "default": 0, "min": 0, "max": 30,
                    "tooltip": "Extra settle time server-side, in seconds.",
                }),
                "timeout": ("INT", {"default": 60, "min": 10, "max": 120}),
                "width": ("INT", {
                    "default": 430, "min": 64, "max": 4096,
                    "tooltip": "Only used when `device` is the none option.",
                }),
                "height": ("INT", {"default": 932, "min": 64, "max": 8192}),
                "access_key": ("STRING", {
                    "default": "",
                    "tooltip": "Leave BLANK. The key is read from SCREENSHOTONE_ACCESS_KEY "
                               "in the environment or the .env at the portable root, so "
                               "no secret is saved into the workflow file.",
                }),
                "force_refetch": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Shoot again even though nothing changed. Every fetch is "
                               "billed, so this is opt-in — the cached image is what you "
                               "want while tuning anything downstream.",
                }),
            },
        }

    @classmethod
    def IS_CHANGED(cls, force_refetch=False, **kwargs):
        # Default FALSE, unlike most fetchers: this one costs money per call, so the
        # cache is the safe default and re-shooting is the deliberate act.
        return float("nan") if force_refetch else False

    def run(self, url, device=DEVICES[0], image_format="jpg", image_quality=80,
            scroll_into_view=DEFAULT_SCROLL_INTO_VIEW, scroll_adjust_top=500,
            block_ads=True, block_cookie_banners=True, block_trackers=True,
            block_banners_by_heuristics=False, full_page=False, delay=0, timeout=60,
            width=430, height=932, access_key="", force_refetch=False):
        url = (url or "").strip()
        if not url:
            raise RuntimeError("ArkScreenshotOne needs a `url`.")
        if "://" not in url:
            url = "https://" + url

        key = resolve_key(access_key, "SCREENSHOTONE_ACCESS_KEY")
        if not key:
            raise RuntimeError(
                "ArkScreenshotOne has no access key. Put SCREENSHOTONE_ACCESS_KEY in the "
                ".env at the portable root, or in the environment. Do not type it into "
                "the node — it would be saved inside the workflow file.")
        secret = resolve_key("", "SCREENSHOTONE_SECRET_KEY")

        request_url = build_url(
            url, key, device=device, width=width, height=height,
            image_format=image_format, image_quality=image_quality,
            block_ads=block_ads, block_cookie_banners=block_cookie_banners,
            block_trackers=block_trackers,
            block_banners_by_heuristics=block_banners_by_heuristics,
            delay=delay, timeout=timeout, scroll_into_view=scroll_into_view,
            scroll_adjust_top=scroll_adjust_top, full_page=full_page,
            secret_key=secret)

        try:
            with urllib.request.urlopen(request_url, timeout=timeout + 30) as response:
                blob = response.read()
        except urllib.error.HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode("utf-8", "replace")[:400]
            except Exception:
                pass
            # The URL carries the key, so it must never reach a log or an exception.
            raise RuntimeError("ArkScreenshotOne: HTTP %s from the API. %s"
                               % (exc.code, body))
        except Exception as exc:
            raise RuntimeError("ArkScreenshotOne: the request failed (%s)." % exc)

        if not blob:
            raise RuntimeError("ArkScreenshotOne returned an empty response.")
        with Image.open(io.BytesIO(blob)) as opened:
            pil = opened.convert("RGB")
            pixels = np.asarray(pil, dtype=np.float32) / 255.0
        image = torch.from_numpy(pixels)[None, ...]

        report = ("%s | %dx%d %s | ads:%s cookies:%s | %.0f KB"
                  % (url, image.shape[2], image.shape[1], image_format,
                     block_ads, block_cookie_banners, len(blob) / 1024.0))
        print("[arkennemasis] screenshotone: %s" % report)
        return (image, report)


NODE_CLASS_MAPPINGS = {"ArkScreenshotOne": ArkScreenshotOne}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ArkScreenshotOne": "arkennemasis ScreenshotOne (hosted, ad-free page shot)",
}

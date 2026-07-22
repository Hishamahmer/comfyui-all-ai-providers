// Color Picker front-end: shows the image on the node, lets you drag a pin over it
// with a live #hex / RGB readout, plus a "pick from screen" eyedropper button
// (Chromium EyeDropper API). The authoritative color is sampled by Python at run time
// from the full-resolution tensor; this preview is for interaction/feedback.

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE = "AIColorPicker";
const MAX_H = 240;

function w(node, name) {
  return node.widgets?.find((x) => x.name === name);
}

// ---- preview image loading ---------------------------------------------------

function setPreview(node, url) {
  try {
    const pv = node._aapPv;
    if (!pv || pv._url === url) return;
    const img = new Image();
    img.onload = () => {
      pv._img = img;
      pv._url = url;
      pv._cv = null; // rebuild sampler lazily
      const sz = node.computeSize();
      if (node.size[1] < sz[1]) node.setSize([node.size[0], sz[1]]);
      node.setDirtyCanvas(true, true);
    };
    img.src = url;
  } catch (e) {}
}

function parseLoadImageValue(v) {
  // "img.png", "sub/img.png", possibly with " [input]" / " [output]" / " [temp]"
  let type = "input";
  let name = String(v);
  const m = name.match(/^(.*) \[(input|output|temp)\]$/);
  if (m) {
    name = m[1];
    type = m[2];
  }
  let subfolder = "";
  const i = name.lastIndexOf("/");
  if (i >= 0) {
    subfolder = name.substring(0, i);
    name = name.substring(i + 1);
  }
  return { name, subfolder, type };
}

function loadFromUpstream(node) {
  try {
    const inp = node.inputs?.find((i) => i.name === "image");
    if (!inp?.link) return;
    const link = app.graph.links[inp.link];
    if (!link) return;
    const src = app.graph.getNodeById(link.origin_id);
    if (!src) return;
    const t = src.comfyClass || src.type;
    if (t === "LoadImage") {
      const iw = src.widgets?.find((x) => x.name === "image");
      if (iw?.value) {
        const p = parseLoadImageValue(iw.value);
        setPreview(node, api.apiURL(
          `/view?filename=${encodeURIComponent(p.name)}&type=${p.type}` +
          `&subfolder=${encodeURIComponent(p.subfolder)}`));
      }
    } else if (src.imgs?.[0]?.src) {
      // any upstream node that already shows a preview image
      setPreview(node, src.imgs[0].src);
    }
  } catch (e) {}
}

// after a run, Python sends the received image back (custom ui key -> no double preview)
api.addEventListener("executed", (e) => {
  try {
    const d = e.detail;
    const info = d?.output?.aap_preview?.[0];
    if (!info) return;
    const node = app.graph.getNodeById(d.display_node ?? d.node) ||
                 app.graph.getNodeById(Number(d.display_node ?? d.node));
    if (!node || (node.comfyClass || node.type) !== NODE) return;
    setPreview(node, api.apiURL(
      `/view?filename=${encodeURIComponent(info.filename)}&type=${info.type}` +
      `&subfolder=${encodeURIComponent(info.subfolder || "")}`));
  } catch (err) {}
});

// ---- pixel sampling from the preview ----------------------------------------

function sampleAt(pv, u, v) {
  try {
    if (!pv._img) return null;
    if (!pv._cv) {
      const cv = document.createElement("canvas");
      cv.width = pv._img.naturalWidth;
      cv.height = pv._img.naturalHeight;
      cv.getContext("2d").drawImage(pv._img, 0, 0);
      pv._cv = cv;
      pv._ctx2 = cv.getContext("2d", { willReadFrequently: true });
    }
    const x = Math.max(0, Math.min(pv._cv.width - 1, Math.round(u * (pv._cv.width - 1))));
    const y = Math.max(0, Math.min(pv._cv.height - 1, Math.round(v * (pv._cv.height - 1))));
    const d = pv._ctx2.getImageData(x, y, 1, 1).data;
    return [d[0], d[1], d[2]];
  } catch (e) {
    return null;
  }
}

const hex2 = (n) => n.toString(16).toUpperCase().padStart(2, "0");

// ---- the widget --------------------------------------------------------------

function makePreviewWidget(node) {
  const pv = {
    type: "AAP_PICK_PREVIEW",
    name: "pick_preview",
    options: { serialize: false },
    _img: null,
    _url: null,
    _cv: null,
    _ctx2: null,
    _rect: null,
    _drag: false,

    computeSize(width) {
      if (!this._img) return [width, 46];
      const availW = Math.max(50, (width || node.size[0]) - 20);
      const h = Math.min(MAX_H, Math.round(availW * this._img.naturalHeight / this._img.naturalWidth));
      return [width, h + 12];
    },

    draw(ctx, n, widgetWidth, y) {
      try {
        const margin = 10;
        const availW = widgetWidth - margin * 2;

        if (!this._img) {
          ctx.save();
          ctx.fillStyle = "rgba(255,255,255,0.06)";
          ctx.fillRect(margin, y + 4, availW, 36);
          ctx.fillStyle = "#888";
          ctx.font = "11px sans-serif";
          ctx.textAlign = "center";
          ctx.fillText("connect an image (preview shows after 1st run)", widgetWidth / 2, y + 26);
          ctx.restore();
          this._rect = null;
          return;
        }

        const iw = this._img.naturalWidth, ih = this._img.naturalHeight;
        const scale = Math.min(availW / iw, MAX_H / ih);
        const dw = iw * scale, dh = ih * scale;
        const dx = margin + (availW - dw) / 2, dy = y + 6;
        this._rect = [dx, dy, dw, dh];

        ctx.save();
        ctx.drawImage(this._img, dx, dy, dw, dh);
        ctx.strokeStyle = "rgba(255,255,255,0.25)";
        ctx.lineWidth = 1;
        ctx.strokeRect(dx, dy, dw, dh);

        // pin
        const u = Number(w(n, "pick_x")?.value ?? 0.5);
        const v = Number(w(n, "pick_y")?.value ?? 0.5);
        const px = dx + u * dw, py = dy + v * dh;
        ctx.lineWidth = 2;
        ctx.strokeStyle = "#fff";
        ctx.beginPath(); ctx.arc(px, py, 8, 0, Math.PI * 2); ctx.stroke();
        ctx.strokeStyle = "#000";
        ctx.beginPath(); ctx.arc(px, py, 6, 0, Math.PI * 2); ctx.stroke();

        // live readout chip
        const rgb = sampleAt(this, u, v);
        const label = rgb
          ? `#${hex2(rgb[0])}${hex2(rgb[1])}${hex2(rgb[2])}  RGB ${rgb[0]},${rgb[1]},${rgb[2]}`
          : `x ${u.toFixed(3)}  y ${v.toFixed(3)}`;
        ctx.font = "11px monospace";
        const tw = ctx.measureText(label).width;
        const chipW = tw + (rgb ? 26 : 12), chipH = 18;
        let cx = Math.max(dx, Math.min(px - chipW / 2, dx + dw - chipW));
        let cy = py - 30 < dy ? py + 14 : py - 30;
        ctx.fillStyle = "rgba(10,10,12,0.85)";
        ctx.beginPath();
        ctx.roundRect ? ctx.roundRect(cx, cy, chipW, chipH, 4) : ctx.rect(cx, cy, chipW, chipH);
        ctx.fill();
        if (rgb) {
          ctx.fillStyle = `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;
          ctx.fillRect(cx + 5, cy + 4, 10, 10);
        }
        ctx.fillStyle = "#eee";
        ctx.textAlign = "left";
        ctx.fillText(label, cx + (rgb ? 20 : 6), cy + 13);
        ctx.restore();
      } catch (e) {}
    },

    mouse(event, pos, n) {
      try {
        if (!this._rect) return false;
        const [rx, ry, rw, rh] = this._rect;
        const inside = pos[0] >= rx && pos[0] <= rx + rw && pos[1] >= ry && pos[1] <= ry + rh;
        const t = event.type;
        const isDown = t === "pointerdown" || t === "mousedown";
        const isMove = t === "pointermove" || t === "mousemove";
        const isUp = t === "pointerup" || t === "mouseup";

        if (isDown && inside) this._drag = true;
        if (isUp) this._drag = false;
        if ((isDown && inside) || (isMove && this._drag)) {
          const u = Math.max(0, Math.min(1, (pos[0] - rx) / rw));
          const v = Math.max(0, Math.min(1, (pos[1] - ry) / rh));
          const wx = w(n, "pick_x"), wy = w(n, "pick_y"), ws = w(n, "source");
          if (wx) wx.value = Math.round(u * 1000) / 1000;
          if (wy) wy.value = Math.round(v * 1000) / 1000;
          if (ws) ws.value = "pin_on_image";
          n.setDirtyCanvas(true, true);
          return true;
        }
        return inside;
      } catch (e) {
        return false;
      }
    },
  };
  return pv;
}

// ---- extension ---------------------------------------------------------------

app.registerExtension({
  name: "aap.color_picker",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      onNodeCreated?.apply(this, arguments);
      const node = this;

      // screen eyedropper (Chromium EyeDropper API; ComfyUI desktop = Chromium)
      const btn = node.addWidget("button", "🎯 pick from screen", null, async () => {
        if (!window.EyeDropper) {
          alert("Screen eyedropper needs a Chromium browser (Chrome/Edge/ComfyUI desktop).");
          return;
        }
        try {
          const r = await new window.EyeDropper().open();
          const whex = w(node, "manual_hex"), ws = w(node, "source");
          if (whex) whex.value = r.sRGBHex.toUpperCase();
          if (ws) ws.value = "manual_hex";
          node.setDirtyCanvas(true, true);
        } catch (e) { /* user cancelled */ }
      });
      if (btn) btn.options = { ...(btn.options || {}), serialize: false };

      const pv = makePreviewWidget(node);
      node.addCustomWidget(pv);
      node._aapPv = pv;

      setTimeout(() => loadFromUpstream(node), 100);
    };

    const onConnectionsChange = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onConnectionsChange = function () {
      onConnectionsChange?.apply(this, arguments);
      loadFromUpstream(this);
    };
  },
  async setup() {
    console.log("[arkennemasis] color picker UI loaded");
  },
});

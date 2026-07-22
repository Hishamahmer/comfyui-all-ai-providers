// Minimal "cooking" activity indicator for the Replicate/OpenAI nodes.
// While a node calls Replicate it shows:
//   - a title spinner + elapsed seconds  (works in ANY renderer, incl. Vue nodes)
//   - a pulsing border + pill             (classic canvas renderer)
// Detection uses the "executing" WebSocket event (fires reliably) with app.runningNodeId
// as a fallback. One light 150ms timer; it only repaints while a node is running.

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const ANIMATED_NODES = new Set([
  "ReplicateOpenAILLM",
  "ReplicateOpenAIGPTImage2",
]);

const SPINNER = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
const COOK = ["🍳", "🔥", "♨️", "🍲", "✨"];

let eventNodeId = null; // id from the "executing" event (null when idle)
let frame = 0;
let startTime = 0;
let current = null; // the node instance currently "running"
let loggedDraw = false;

function isOurs(node) {
  return node && ANIMATED_NODES.has(node.comfyClass || node.type);
}

function resolveRunningNode() {
  let id = eventNodeId;
  if (id === null || id === undefined) {
    try { id = app.runningNodeId; } catch (e) { id = null; }
  }
  if (id === null || id === undefined) return null;
  const g = app.graph;
  if (!g) return null;
  const n = g.getNodeById(id) || g.getNodeById(Number(id));
  return isOurs(n) ? n : null;
}

function restore(node) {
  if (node && node.__repOrigTitle !== undefined) {
    node.title = node.__repOrigTitle;
    node.__repOrigTitle = undefined;
    node.__repRunning = false;
  }
}

// ---- detection via WebSocket events -----------------------------------------
api.addEventListener("executing", (e) => {
  const d = e && e.detail;
  const id = d && typeof d === "object" ? (d.node ?? d.display_node ?? null) : (d ?? null);
  eventNodeId = id === undefined ? null : id;
});
const clearExec = () => { eventNodeId = null; };
api.addEventListener("execution_success", clearExec);
api.addEventListener("execution_error", clearExec);
api.addEventListener("execution_interrupted", clearExec);
api.addEventListener("execution_start", () => { eventNodeId = null; });

// ---- one light heartbeat -----------------------------------------------------
setInterval(() => {
  try {
    const node = resolveRunningNode();

    if (node !== current) {
      if (current) restore(current);
      current = node;
      if (current) {
        if (current.__repOrigTitle === undefined) current.__repOrigTitle = current.title;
        current.__repRunning = true;
        startTime = performance.now();
      }
    }
    if (!current) return; // idle: no repaint

    frame++;
    const secs = ((performance.now() - startTime) / 1000).toFixed(0);
    const spin = SPINNER[frame % SPINNER.length];
    const cook = COOK[Math.floor(frame / 6) % COOK.length];

    // guaranteed-visible: reflect status in the node title
    current.title = `${cook} ${current.__repOrigTitle} · ${spin} ${secs}s`;

    if (app.graph) app.graph.setDirtyCanvas(true, true);
  } catch (e) {
    /* never let the indicator break the UI */
  }
}, 150);

// ---- pretty canvas overlay (classic renderer) -------------------------------
function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

function drawCooking(node, ctx) {
  if (!node.__repRunning || node.flags?.collapsed) return;
  if (!loggedDraw) {
    loggedDraw = true;
    console.log("[arkennemasis] cooking badge drawing on canvas");
  }

  const now = performance.now();
  const W = node.size[0];
  const H = node.size[1];

  ctx.save();
  // pulsing border
  const pulse = 0.5 + 0.5 * Math.sin(now / 300);
  ctx.strokeStyle = `rgba(90,170,255,${0.3 + 0.55 * pulse})`;
  ctx.lineWidth = 2.5;
  roundRect(ctx, 1, 1, W - 2, H - 2, 8);
  ctx.stroke();

  // small sliding shimmer bar under the title
  const barW = W * 0.28;
  const t = (now / 900) % 1;
  const bx = (W - barW) * (0.5 - 0.5 * Math.cos(t * Math.PI * 2));
  ctx.fillStyle = "rgba(90,170,255,0.9)";
  roundRect(ctx, bx, 2, barW, 3, 1.5);
  ctx.fill();
  ctx.restore();
}

app.registerExtension({
  name: "arkennemasis.activity",
  async setup() {
    console.log("[arkennemasis] cooking indicator loaded");
  },
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!ANIMATED_NODES.has(nodeData.name)) return;
    const orig = nodeType.prototype.onDrawForeground;
    nodeType.prototype.onDrawForeground = function (ctx) {
      orig?.apply(this, arguments);
      try {
        drawCooking(this, ctx);
      } catch (e) {
        /* cosmetic only */
      }
    };
  },
});

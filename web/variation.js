// Grey out the stages that the Run Switches node has turned off.
//
// The switches already WORK without this — each one is wired to its stage, and the
// expensive branch is skipped through a lazy input so the model is never called. But a
// canvas where nothing changes appearance gives the operator no way to see what this
// run is going to do, and "did that toggle actually take?" is exactly the question a
// switchboard exists to answer.
//
// This is DELIBERATELY cosmetic. The obvious alternative — setting the controlled nodes
// to Never/Bypass the way ArkGroupSwitch does — would break the graph here: Verify
// Candidate feeds a REQUIRED verdict_json into Job Record, and a muted node supplies
// nothing, so switching verification off would fail the run rather than skip a stage.
// The wires decide behaviour; this only makes the decision visible.

import { app } from "../../scripts/app.js";

const SWITCH_NODE = "ArkRunSwitches";

// Switch widget name -> which nodes it governs. Widget names are the on-canvas labels,
// so they are matched by their leading number: the wording may be reworded for clarity
// without silently detaching the highlighting from the switch.
const CONTROLS = [
  { prefix: "1.", types: ["ArkVerifyCandidate"] },
  { prefix: "2.", types: ["ArkQCRequest", "ArkQCVerdict"], titleMatch: /qc|critic|looks at/i },
  { prefix: "3.", types: [], sharesNode: "ArkReviewBoard" },
  { prefix: "4.", types: [], sharesNode: "ArkReviewBoard" },
  { prefix: "5.", types: ["ArkBoardPreview"] },
  { prefix: "6.", types: ["ArkStoreExport"] },
];

const OFF_COLOR = "#2a2a2e";
const OFF_BGCOLOR = "#1c1c20";
const SUFFIX = "   ⏸ OFF";

function switchValues(node) {
  const out = {};
  for (const widget of node.widgets || []) {
    const label = String(widget.name || "");
    const match = label.match(/^(\d+)\./);
    if (match) out[match[1] + "."] = widget.value !== false;
  }
  return out;
}

function remember(node) {
  if (node.__arkOrig === undefined) {
    node.__arkOrig = {
      color: node.color,
      bgcolor: node.bgcolor,
      title: node.title,
    };
  }
  return node.__arkOrig;
}

function setDimmed(node, dimmed) {
  const original = remember(node);
  if (dimmed) {
    node.color = OFF_COLOR;
    node.bgcolor = OFF_BGCOLOR;
    if (!String(node.title || "").endsWith(SUFFIX)) {
      node.title = (original.title || node.type) + SUFFIX;
    }
  } else {
    node.color = original.color;
    node.bgcolor = original.bgcolor;
    node.title = original.title;
  }
}

function apply(graph) {
  if (!graph || !graph._nodes) return;
  const board = graph._nodes.filter((n) => n.type === "ArkReviewBoard");

  for (const control of graph._nodes.filter((n) => n.type === SWITCH_NODE)) {
    const values = switchValues(control);

    for (const rule of CONTROLS) {
      const on = values[rule.prefix];
      if (on === undefined) continue;

      for (const node of graph._nodes) {
        if (!rule.types.includes(node.type)) continue;
        // The critic's LLM is an ArkCodexLLM like the other two, so it is identified by
        // its title rather than its type — dimming all three would be wrong.
        if (rule.titleMatch && node.type === "ArkCodexLLM" &&
            !rule.titleMatch.test(String(node.title || ""))) continue;
        setDimmed(node, !on);
      }
    }

    // The board node writes two different artefacts, so it only counts as off when
    // BOTH of its switches are.
    const boardOff = values["3."] === false && values["4."] === false;
    for (const node of board) setDimmed(node, boardOff);
  }

  // The critic's own LLM node, matched by title.
  for (const control of graph._nodes.filter((n) => n.type === SWITCH_NODE)) {
    const values = switchValues(control);
    if (values["2."] === undefined) continue;
    for (const node of graph._nodes) {
      if (node.type === "ArkCodexLLM" && /qc|critic|looks at/i.test(String(node.title || ""))) {
        setDimmed(node, !values["2."]);
      }
    }
  }
  app.graph.setDirtyCanvas(true, false);
}

app.registerExtension({
  name: "arkennemasis.variation.switches",

  nodeCreated(node) {
    if (node.comfyClass !== SWITCH_NODE && node.type !== SWITCH_NODE) return;
    for (const widget of node.widgets || []) {
      const previous = widget.callback;
      widget.callback = function (...args) {
        const result = previous ? previous.apply(this, args) : undefined;
        // Defer: the widget's own value is assigned after the callback returns.
        setTimeout(() => apply(app.graph), 0);
        return result;
      };
    }
    setTimeout(() => apply(app.graph), 0);
  },

  // A freshly loaded graph must show the state it was saved in, not a default.
  afterConfigureGraph() {
    setTimeout(() => apply(app.graph), 50);
  },
});

(function () {
  "use strict";

  var api = typeof window !== "undefined" ? window.comfyAPI : null;
  var appApi = api && api.app;
  var app = appApi && (appApi.app || appApi.default);
  if (!app || typeof app.registerExtension !== "function") return;

  var NODE_CLASS = "DepthPro2FocalLength";
  var WIDGET_NAME = "$$depthpro2-focal-readout";
  var EMPTY_TEXT =
    "Connect IMAGE and queue the prompt to see focal pixels, 35mm-equivalent and FOV.";

  function getWidget(node, create) {
    var widget = (node.widgets || []).find(function (item) {
      return item.name === WIDGET_NAME;
    });
    if (widget || !create || typeof node.addDOMWidget !== "function") return widget;

    var text = EMPTY_TEXT;
    var element = document.createElement("div");
    element.className = "depthpro2-focal-readout";
    element.textContent = text;
    Object.assign(element.style, {
      boxSizing: "border-box",
      width: "100%",
      height: "100%",
      minHeight: "64px",
      maxHeight: "180px",
      overflow: "auto",
      padding: "8px 10px",
      border: "1px solid var(--border-color, #555)",
      borderRadius: "6px",
      background: "var(--comfy-input-bg, #222)",
      color: "var(--fg-color, #eee)",
      font: "11px/1.45 ui-monospace, SFMono-Regular, Consolas, monospace",
      whiteSpace: "pre-wrap",
      overflowWrap: "anywhere",
      userSelect: "text"
    });

    widget = node.addDOMWidget(WIDGET_NAME, "depthpro2-focal-readout", element, {
      getMinHeight: function () { return 64; },
      getMaxHeight: function () { return 180; },
      getValue: function () { return text; },
      setValue: function (value) {
        text = String(value);
        element.textContent = text;
      },
      serialize: false
    });
    widget.serialize = false;
    return widget;
  }

  function updateWidget(node, value) {
    var widget = getWidget(node, true);
    if (!widget) return;
    if (widget.options && typeof widget.options.setValue === "function") {
      widget.options.setValue(value);
    }
    widget.value = value;
    if (widget.element) widget.element.textContent = value;
    node.setDirtyCanvas(true, true);
  }

  function findNode(graph, executionId) {
    if (!graph || typeof graph.getNodeById !== "function") return null;
    var node = graph.getNodeById(Number(executionId));
    if (node) return node;
    for (var end = executionId.indexOf(":"); end !== -1; end = executionId.indexOf(":", end + 1)) {
      var host = graph.getNodeById(Number(executionId.slice(0, end)));
      if (host && host.subgraph) {
        var nested = findNode(host.subgraph, executionId.slice(end + 1));
        if (nested) return nested;
      }
    }
    return null;
  }

  app.registerExtension({
    name: "DepthPro2.FocalLengthReadout",
    nodeCreated: function (node) {
      if (!node || node.comfyClass !== NODE_CLASS) return;
      getWidget(node, true);
      var onExecuted = node.onExecuted;
      node.onExecuted = function (output) {
        if (onExecuted) onExecuted.call(this, output);
        var text = output && output.text && output.text[0];
        if (typeof text === "string") updateWidget(this, text);
      };
    },
    onNodeOutputsUpdated: function (nodeOutputs) {
      if (!nodeOutputs) return;
      var graph = app.rootGraph || app.graph;
      Object.keys(nodeOutputs).forEach(function (id) {
        var node = findNode(graph, id);
        if (!node || node.comfyClass !== NODE_CLASS) return;
        var output = nodeOutputs[id];
        var text = output && output.text && output.text[0];
        if (typeof text === "string") updateWidget(node, text);
      });
    }
  });
})();

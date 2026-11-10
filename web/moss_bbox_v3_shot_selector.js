import { app } from "../../scripts/app.js";

const NODE_TYPE = "YebanMossBBoxShotControlV1";
const SHOTS = [
    ["reference", "沿用原图", "原图大小"],
    ["close", "近景", "约 90% 画高"],
    ["environmental", "中景", "约 58% 画高"],
    ["wide", "远景", "约 34% 画高"],
];
const POSITIONS = [
    ["upper_left", "左上"], ["upper_center", "上中"], ["upper_right", "右上"],
    ["middle_left", "左中"], ["center", "居中"], ["middle_right", "右中"],
    ["lower_left", "左下"], ["lower_center", "下中"], ["lower_right", "右下"],
];

function mount(node) {
    if (node?.comfyClass !== NODE_TYPE || node.__yebanMossShotSelector) return;
    const shotWidget = node.widgets?.find((widget) => widget.name === "shot_size");
    const positionWidget = node.widgets?.find((widget) => widget.name === "composition_position");
    if (!shotWidget || !positionWidget) return;

    const root = document.createElement("div");
    root.className = "yeban-moss-shot-selector";
    root.setAttribute("role", "group");
    root.setAttribute("aria-label", "景别与构图位置选择");
    const shotPanel = document.createElement("section");
    shotPanel.className = "yeban-moss-shot-selector__panel";
    const shotHeading = document.createElement("div");
    shotHeading.className = "yeban-moss-shot-selector__heading";
    shotHeading.textContent = "景别";
    const shotGrid = document.createElement("div");
    shotGrid.className = "yeban-moss-shot-selector__shots";
    shotPanel.append(shotHeading, shotGrid);
    const positionPanel = document.createElement("section");
    positionPanel.className = "yeban-moss-shot-selector__panel";
    const positionHeading = document.createElement("div");
    positionHeading.className = "yeban-moss-shot-selector__heading";
    positionHeading.textContent = "构图位置";
    const positionGrid = document.createElement("div");
    positionGrid.className = "yeban-moss-shot-selector__positions";
    positionPanel.append(positionHeading, positionGrid);
    root.append(shotPanel, positionPanel);

    const shotButtons = new Map();
    for (const [value, title, detail] of SHOTS) {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "yeban-moss-shot-selector__option";
        button.dataset.value = value;
        button.innerHTML = `<span class="yeban-moss-shot-selector__title"></span><span class="yeban-moss-shot-selector__detail"></span>`;
        button.querySelector(".yeban-moss-shot-selector__title").textContent = title;
        button.querySelector(".yeban-moss-shot-selector__detail").textContent = detail;
        button.addEventListener("click", (event) => {
            event.preventDefault();
            event.stopPropagation();
            setWidgetValue(shotWidget, value);
            render();
        });
        shotButtons.set(value, button);
        shotGrid.append(button);
    }
    const positionButtons = new Map();
    for (const [value, title] of POSITIONS) {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "yeban-moss-shot-selector__position";
        button.textContent = title;
        button.dataset.value = value;
        button.addEventListener("click", (event) => {
            event.preventDefault();
            event.stopPropagation();
            if (shotWidget.value === "reference") return;
            setWidgetValue(positionWidget, value);
            render();
        });
        positionButtons.set(value, button);
        positionGrid.append(button);
    }

    function setWidgetValue(widget, value) {
        if (widget.value === value) return;
        widget.value = value;
        widget.callback?.call(widget, value, app.canvas, node, node.widgets.indexOf(widget));
        if (node.graph) node.graph._version++;
        node.setDirtyCanvas?.(true, true);
        app.graph?.setDirtyCanvas?.(true, true);
    }

    function render() {
        for (const [value, button] of shotButtons) {
            const selected = shotWidget.value === value;
            button.dataset.selected = String(selected);
            button.setAttribute("aria-pressed", String(selected));
        }
        const positionDisabled = shotWidget.value === "reference";
        positionPanel.dataset.disabled = String(positionDisabled);
        positionHeading.title = positionDisabled
            ? "沿用原图时保留参考图中的人物位置"
            : "选择人物组在画面中的九宫格位置";
        for (const [value, button] of positionButtons) {
            const selected = positionWidget.value === value;
            button.dataset.selected = String(selected);
            button.disabled = positionDisabled;
            button.setAttribute("aria-pressed", String(selected));
        }
    }

    // Keep the original widget as the workflow's serialized value carrier.
    // The compact visual selector changes only its value, so saved workflows
    // keep the same widget order and existing shot selections remain intact.
    for (const widget of [shotWidget, positionWidget]) {
        widget.type = "hidden";
        widget.computeSize = () => [0, -4];
    }
    node.addDOMWidget("景别与构图位置", "yeban-moss-shot-selector", root, {
        serialize: false,
        hideOnZoom: false,
        getValue: () => undefined,
        setValue: () => {},
        getMinHeight: () => 116,
        getMaxHeight: () => 116,
    });
    render();
    node.__yebanMossShotSelector = { root, render };
    requestAnimationFrame(() => node.setDirtyCanvas?.(true, true));
}

function loadStyle() {
    if (document.querySelector('link[data-yeban-moss-shot-selector="true"]')) return;
    const link = document.createElement("link");
    link.rel = "stylesheet";
    link.href = new URL("./moss_bbox_v3_shot_selector.css?v=20261109-2", import.meta.url).href;
    link.dataset.yebanMossShotSelector = "true";
    document.head.append(link);
}

loadStyle();
app.registerExtension({
    name: "Yeban.MossBBoxV3ShotSelector",
    nodeCreated(node) { mount(node); },
    loadedGraphNode(node) { queueMicrotask(() => mount(node)); },
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== NODE_TYPE) return;
        const originalCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = originalCreated?.apply(this, arguments);
            queueMicrotask(() => mount(this));
            return result;
        };
        const originalConfigured = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = originalConfigured?.apply(this, arguments);
            queueMicrotask(() => mount(this));
            return result;
        };
    },
});


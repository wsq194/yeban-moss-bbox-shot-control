import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_TYPE = "YebanMossBBoxStandaloneV3";
const SUPPORTED = new Set(["local_grok", "dsh_official"]);

function mount(node) {
    if (node?.comfyClass !== NODE_TYPE || node.__yebanMossSessionPanel) return;
    const backendWidget = node.widgets?.find((widget) => widget.name === "backend");
    const customModelWidget = node.widgets?.find((widget) => widget.name === "custom_model");
    const modelWidget = node.widgets?.find((widget) => widget.name === "backend_model");
    if (!backendWidget || !modelWidget) return;

    const root = document.createElement("div");
    root.className = "yeban-moss-session";
    root.innerHTML = `<div class="yeban-moss-session__row"><span>后端会话</span><b>尚未运行</b><button type="button">清除会话</button></div><small>本机 Grok CLI、官方 DSH 会按节点、模型分别续接；其他后端不支持持续会话。</small>`;
    const status = root.querySelector("b");
    const button = root.querySelector("button");

    function modelName() {
        if (String(backendWidget.value || "") === "dsh_official") {
            const slotIndex = node.inputs?.findIndex((slot) => slot.name === "dsh_config") ?? -1;
            const configNode = slotIndex >= 0 ? node.getInputNode?.(slotIndex) : null;
            const configured = configNode?.widgets?.find((widget) => widget.name === "model")?.value;
            if (configured) return String(configured).trim();
        }
        const selected = String(customModelWidget?.value || modelWidget?.value || "").trim();
        return selected || (String(backendWidget.value || "") === "local_grok" ? "grok-4.7" : "");
    }

    async function refresh() {
        const backend = String(backendWidget.value || "");
        root.hidden = !SUPPORTED.has(backend);
        if (!SUPPORTED.has(backend)) return;
        try {
            const query = new URLSearchParams({ node: String(node.id), backend, model: modelName() });
            const response = await api.fetchApi(`/yeban/moss-bbox-standalone/session/status?${query}`);
            const payload = await response.json();
            status.textContent = payload?.status || "尚未运行";
            button.disabled = !payload?.has_session;
        } catch {
            status.textContent = "状态暂不可用";
            button.disabled = true;
        }
        node.setDirtyCanvas?.(true, true);
    }

    button.addEventListener("click", async (event) => {
        event.preventDefault();
        event.stopPropagation();
        button.disabled = true;
        status.textContent = "正在清除会话";
        try {
            const response = await api.fetchApi("/yeban/moss-bbox-standalone/session/reset", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ node: String(node.id), backend: String(backendWidget.value || ""), model: modelName() }),
            });
            const payload = await response.json();
            status.textContent = payload?.status || payload?.error || "清除失败";
            button.disabled = true;
        } catch {
            status.textContent = "清除失败";
        }
        node.setDirtyCanvas?.(true, true);
    });

    for (const widget of [backendWidget, modelWidget, customModelWidget].filter(Boolean)) {
        const original = widget.callback;
        widget.callback = function (...args) {
            const result = original?.apply(this, args);
            setTimeout(refresh, 400);
            return result;
        };
    }
    node.addDOMWidget("后端会话", "yeban-moss-session", root, {
        serialize: false,
        hideOnZoom: false,
        getValue: () => undefined,
        setValue: () => {},
        getMinHeight: () => 44,
        getMaxHeight: () => 44,
    });
    node.__yebanMossSessionPanel = { root, refresh };
    void refresh();
}

function loadStyle() {
    if (document.querySelector('link[data-yeban-moss-session="true"]')) return;
    const link = document.createElement("link");
    link.rel = "stylesheet";
    link.href = new URL("./moss_bbox_v3_session.css?v=20261112-1", import.meta.url).href;
    link.dataset.yebanMossSession = "true";
    document.head.append(link);
}

loadStyle();
api.addEventListener("yeban_prompt_writer_progress", ({ detail }) => {
    if (detail?.stage !== "backend_status") return;
    const node = app.graph?.getNodeById?.(Number(detail?.node)) ?? app.graph?.getNodeById?.(detail?.node);
    const panel = node?.__yebanMossSessionPanel;
    if (panel && typeof detail?.backend_status === "string") {
        const label = panel.root.querySelector("b");
        if (label) label.textContent = detail.backend_status;
        node.setDirtyCanvas?.(true, true);
        void panel.refresh();
    }
});

app.registerExtension({
    name: "Yeban.MossBBoxV3Session",
    nodeCreated(node) { queueMicrotask(() => mount(node)); },
    loadedGraphNode(node) { queueMicrotask(() => mount(node)); },
});

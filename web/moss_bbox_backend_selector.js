import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_ID = "YebanMossBBoxStandaloneV3";
const ROUTE = "/yeban/moss-bbox-standalone";
const BOUND = Symbol("yebanMossBBoxBackendBound");
const EFFORTS = new Set(["none", "minimal", "low", "medium", "high", "xhigh", "max"]);

function bind(node) {
    if (node[BOUND]) return;
    const get = (name) => node.widgets?.find((widget) => widget.name === name);
    const backend = get("backend"), model = get("backend_model"), effort = get("reasoning");
    if (!backend || !model || !effort) return;

    let revision = 0;
    let queued = false;
    const manual = get("custom_model");
    const controls = [model, effort, manual].filter(Boolean).map((widget) => ({
        widget, type: widget.type, computeSize: widget.computeSize,
    }));
    const refreshButton = node.addWidget("button", "刷新模型选项", null, () => schedule(), { serialize: false });
    const hideModelControls = (hidden) => {
        for (const { widget, type, computeSize } of controls) {
            widget.hidden = hidden;
            widget.type = hidden ? "converted-widget" : type;
            widget.computeSize = hidden ? () => [0, -4] : computeSize;
            if (widget.element) widget.element.style.display = hidden ? "none" : "";
        }
    };
    const status = (text) => {
        refreshButton.label = refreshButton.name = text;
        node.setDirtyCanvas?.(true, true);
    };

    async function refresh() {
        const requestId = ++revision;
        const selectedBackend = backend.value;
        const isDsh = selectedBackend === "dsh_official";
        hideModelControls(isDsh);
        status(isDsh ? "请在 DSH 配置节点选择模型" : "正在读取模型选项…");
        let data = {};
        let hint = "";

        try {
            if (isDsh) {
                hint = node.inputs?.some((slot) => slot.name === "dsh_config" && slot.link != null)
                    ? "模型和对话分支由已连接的 DSH 配置节点提供"
                    : "请连接 MOSS BBox 官方 DSH 配置节点";
            } else if (selectedBackend === "openai_compatible") {
                const index = node.inputs?.findIndex((slot) => slot.name === "provider_config") ?? -1;
                const provider = index >= 0 ? node.getInputNode?.(index) : null;
                const profile = provider?.widgets?.find((widget) => widget.name === "供应商档案")?.value;
                if (profile) {
                    const response = await api.fetchApi(`${ROUTE}/providers/models`, {
                        method: "POST", headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ id: profile }),
                    });
                    const result = await response.json();
                    if (!response.ok || !result.ok) throw new Error("provider model catalog unavailable");
                    data.models = Object.fromEntries(result.models.map((id) => [id, []]));
                } else {
                    hint = provider ? "此配置没有模型列表，可手填模型" : "请连接 MOSS BBox 供应商管理器";
                }
            } else {
                const query = new URLSearchParams({ backend: selectedBackend });
                const response = await api.fetchApi(`${ROUTE}/backend-options?${query}`);
                data = await response.json();
                if (!response.ok || data.available === false) throw new Error("backend model catalog unavailable");
            }
        } catch {
            hint = "暂时无法读取模型列表，可使用后端默认值或手填模型";
        }

        if (requestId !== revision) return;
        const modelIds = Object.keys(data.models ?? {});
        const previousModel = model.value || "";
        model.options.values = ["", ...modelIds, ...(!modelIds.length && previousModel && !isDsh ? [previousModel] : [])];
        if (!model.options.values.includes(model.value)) model.value = model.options.values[0];
        if (!model.value && modelIds.includes(data.default_model)) model.value = data.default_model;
        const effectiveModel = manual?.value || model.value;
        const advertised = data.models?.[effectiveModel] ?? (modelIds.length === 1 ? data.models[modelIds[0]] : undefined);
        const efforts = advertised?.length ? advertised : data.efforts ?? [];
        effort.options.values = ["default", ...new Set(efforts.filter((value) => EFFORTS.has(value)))];
        if (!effort.options.values.includes(effort.value)) effort.value = "default";
        model.label = model.value ? "模型" : "模型（后端默认）";
        effort.label = effort.value === "default" ? "推理强度（后端默认）" : "推理强度";
        model.options.tooltip = hint || "留空使用后端默认模型；手填模型优先";
        effort.options.tooltip = data.effort_scope === "cli"
            ? "本机 CLI 提供的推理档位，具体支持情况取决于模型"
            : efforts.length ? "该模型提供的推理档位" : "后端未提供可选档位，使用默认值";
        status(hint || (modelIds.length ? `刷新模型选项（${modelIds.length} 个模型）` : "后端未返回模型列表 · 点击重试"));
    }

    function schedule() {
        if (queued) return;
        queued = true;
        queueMicrotask(() => { queued = false; void refresh(); });
    }

    for (const widget of [backend, model, manual].filter(Boolean)) {
        const original = widget.callback;
        widget.callback = function (...args) {
            const result = original?.apply(this, args);
            if (widget === backend) {
                model.value = "";
                effort.value = "default";
                if (manual) manual.value = "";
            }
            schedule();
            return result;
        };
    }
    const originalConnectionsChange = node.onConnectionsChange;
    node.onConnectionsChange = function (...args) {
        const result = originalConnectionsChange?.apply(this, args);
        const slot = this.inputs?.[args[1]];
        if (args[0] === 1 && ["provider_config", "dsh_config"].includes(slot?.name)) schedule();
        return result;
    };
    node[BOUND] = schedule;
    queueMicrotask(schedule);
}

app.registerExtension({
    name: "Yeban.MossBBoxStandalone.BackendSelector",
    nodeCreated(node) { if (node?.comfyClass === NODE_ID) bind(node); },
    loadedGraphNode(node) { if (node?.comfyClass === NODE_ID) queueMicrotask(() => bind(node)); },
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== NODE_ID) return;
        for (const method of ["onNodeCreated", "onConfigure"]) {
            const original = nodeType.prototype[method];
            nodeType.prototype[method] = function (...args) {
                const result = original?.apply(this, args);
                queueMicrotask(() => bind(this));
                return result;
            };
        }
    },
});

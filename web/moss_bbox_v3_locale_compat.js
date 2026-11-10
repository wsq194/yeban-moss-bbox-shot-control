import { app } from "../../scripts/app.js";

const NODE_TYPE = "YebanMossBBoxStandaloneV3";
const LABELS = {
    image: "参考图",
    provider_config: "供应商配置（API 后端）",
    dsh_config: "DSH 配置（DSH 后端）",
    llama_model: "本地 Llama 模型",
    llama_parameters: "本地 Llama 参数",
    layout_aspect: "采样画幅比例",
    layout_long_edge: "采样长边（像素）",
    shot_size: "景别",
    composition_position: "构图位置",
    backend: "后端",
    backend_model: "模型",
    custom_model: "手填模型（可选）",
    reasoning: "推理强度",
    subject_setting: "人物基础设定（可选）",
    photography_style: "摄影风格（可选）",
    additional_requirements: "附加提示词（可选，可留空）",
    timeout: "超时（秒）",
    use_cache: "使用缓存",
    cache_seed: "缓存种子",
    target_word_count: "智能反推目标英文词数",
};

function isChineseLocale() {
    const locales = [
        app.ui?.settings?.getSettingValue?.("Comfy.Locale"),
        app.ui?.settings?.getSettingValue?.("AGL.Locale"),
        localStorage.getItem("Comfy.Settings.Comfy.Locale"),
        localStorage.getItem("Comfy.Settings.AGL.Locale"),
        localStorage.getItem("AGL.Locale"),
    ];
    return locales.some((value) => /^zh(?:[-_]|$)/i.test(String(value ?? "").replace(/^"|"$/g, "")));
}

function translate(node) {
    if (!isChineseLocale() || (node?.comfyClass ?? node?.type) !== NODE_TYPE) return;

    for (const slot of [...(node.inputs ?? []), ...(node.outputs ?? [])]) {
        const label = LABELS[slot.name];
        if (label) slot.label = label;
    }
    for (const widget of node.widgets ?? []) {
        const label = LABELS[widget.name];
        if (label) widget.label = label;
    }
    node.setDirtyCanvas?.(true, true);
}

function translateAfterOtherExtensions(node) {
    requestAnimationFrame(() => translate(node));
}

app.registerExtension({
    name: "Yeban.MOSSBBoxV3LocaleCompat",
    nodeCreated: translateAfterOtherExtensions,
    loadedGraphNode: translateAfterOtherExtensions,
});

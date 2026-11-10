import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_ID = "YebanMossBBoxProviderManager";
const BASE = "/yeban/moss-bbox-standalone/providers";

function widget(node, name) {
    return node.widgets?.find((item) => item.name === name);
}

function notify(message) {
    if (app.ui?.dialog?.show) app.ui.dialog.show(message);
    else window.alert(message);
}

async function requestJson(path, options = {}) {
    const response = await api.fetchApi(path, options);
    const data = await response.json().catch(() => ({}));
    if (!response.ok || data.error) throw new Error(data.error || `HTTP ${response.status}`);
    return data;
}

function post(path, payload) {
    return requestJson(path, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
    });
}

function field(label, value = "", type = "text") {
    const wrap = document.createElement("label");
    wrap.style.cssText = "display:grid;grid-template-columns:112px 1fr;gap:8px;align-items:center;margin:4px 0;color:var(--fg-color);font-size:12px";
    const title = document.createElement("span");
    title.textContent = label;
    const input = document.createElement("input");
    input.type = type;
    input.value = value;
    input.style.cssText = "min-width:0;background:var(--comfy-input-bg,#222);color:var(--input-text,var(--fg-color));border:1px solid var(--border-color,#666);border-radius:5px;padding:5px 7px";
    wrap.append(title, input);
    return { wrap, input };
}

function button(text, handler, danger = false) {
    const value = document.createElement("button");
    value.textContent = text;
    value.type = "button";
    value.style.cssText = `padding:6px 7px;border-radius:5px;border:1px solid ${danger ? "#b54" : "var(--border-color,#666)"};background:var(--comfy-input-bg,#222);color:var(--fg-color);cursor:pointer;font-size:12px`;
    value.onclick = (event) => {
        event.preventDefault();
        event.stopPropagation();
        void handler(value);
    };
    return value;
}

function passwordDialog() {
    return new Promise((resolve) => {
        const overlay = document.createElement("div");
        overlay.style.cssText = "position:fixed;inset:0;z-index:100000;background:#0009;display:flex;align-items:center;justify-content:center";
        const box = document.createElement("div");
        box.style.cssText = "width:min(460px,90vw);background:var(--comfy-menu-bg,#292929);color:var(--fg-color);border:1px solid var(--border-color,#666);border-radius:10px;padding:18px;box-shadow:0 16px 48px #000";
        const title = document.createElement("h3");
        title.textContent = "配置 API Key";
        title.style.margin = "0 0 8px";
        const help = document.createElement("p");
        help.textContent = "密钥会立即发送到本机 ComfyUI，并使用当前 Windows 用户的 DPAPI 加密保存；不会写入工作流。";
        help.style.cssText = "font-size:13px;line-height:1.5;opacity:.85";
        const input = document.createElement("input");
        input.type = "password";
        input.autocomplete = "new-password";
        input.placeholder = "粘贴 API Key";
        input.style.cssText = "width:100%;box-sizing:border-box;padding:9px;background:var(--comfy-input-bg,#222);color:var(--fg-color);border:1px solid var(--border-color,#666);border-radius:6px";
        const actions = document.createElement("div");
        actions.style.cssText = "display:flex;justify-content:flex-end;gap:8px;margin-top:14px";
        const finish = (result) => {
            input.value = "";
            overlay.remove();
            resolve(result);
        };
        actions.append(button("取消", () => finish("")), button("加密保存", () => finish(input.value.trim())));
        box.append(title, help, input, actions);
        overlay.append(box);
        document.body.append(overlay);
        overlay.addEventListener("mousedown", (event) => { if (event.target === overlay) finish(""); });
        input.addEventListener("keydown", (event) => {
            if (event.key === "Enter") finish(input.value.trim());
            if (event.key === "Escape") finish("");
        });
        setTimeout(() => input.focus(), 0);
    });
}

function modelDialog(models, apply) {
    const overlay = document.createElement("div");
    overlay.style.cssText = "position:fixed;inset:0;z-index:100000;background:#0009;display:flex;align-items:center;justify-content:center";
    const box = document.createElement("div");
    box.style.cssText = "width:min(620px,92vw);background:var(--comfy-menu-bg,#292929);color:var(--fg-color);border:1px solid var(--border-color,#666);border-radius:10px;padding:18px";
    const title = document.createElement("h3");
    title.textContent = `可用模型（${models.length}）`;
    title.style.margin = "0 0 10px";
    const select = document.createElement("select");
    select.style.cssText = "width:100%;padding:8px;background:var(--comfy-input-bg,#222);color:var(--fg-color);border-radius:6px";
    for (const id of models) {
        const option = document.createElement("option");
        option.value = id;
        option.textContent = id;
        select.append(option);
    }
    const actions = document.createElement("div");
    actions.style.cssText = "display:grid;grid-template-columns:repeat(3,1fr);gap:7px;margin-top:12px";
    const close = () => overlay.remove();
    for (const [label, target] of [["设为 Luna", "luna_model"], ["设为 Sol", "sol_model"], ["设为 Terra", "terra_model"], ["设为默认模型", "custom_model"], ["设为 Fallback", "fallback_model"]]) {
        actions.append(button(label, () => { apply(target, select.value); close(); }));
    }
    actions.append(button("关闭", close));
    box.append(title, select, actions);
    overlay.append(box);
    document.body.append(overlay);
}

function createPanel(node) {
    const root = document.createElement("div");
    root.style.cssText = "box-sizing:border-box;width:100%;padding:7px 5px 5px;font-family:system-ui;user-select:none";
    const heading = document.createElement("div");
    heading.textContent = "供应商档案管理";
    heading.style.cssText = "font-weight:700;margin:0 0 6px;color:var(--fg-color)";
    const status = document.createElement("div");
    status.textContent = "正在读取本机档案…";
    status.style.cssText = "padding:7px;border-radius:5px;background:#0003;color:#9dc;margin-bottom:7px;font-size:12px;white-space:normal";
    const savedHeading = document.createElement("div");
    savedHeading.textContent = "已保存供应商";
    savedHeading.style.cssText = "font-weight:700;margin:8px 0 5px;color:var(--fg-color);font-size:13px";
    const savedList = document.createElement("div");
    savedList.style.cssText = "display:flex;flex-direction:column;gap:5px;margin-bottom:10px;max-height:190px;overflow:auto";
    const fields = {
        name: field("供应商名称", "自定义供应商"),
        base_url: field("兼容 API 地址", "https://api.openai.com/v1"),
        luna_model: field("Luna 模型 ID", "gpt-5.6-luna"),
        sol_model: field("Sol 模型 ID", "gpt-5.6-sol"),
        terra_model: field("Terra 模型 ID", "gpt-5.6-terra"),
        custom_model: field("默认自定义模型", ""),
        fallback_model: field("Fallback 模型", "gpt-5.6-sol"),
    };
    const form = document.createElement("div");
    for (const item of Object.values(fields)) form.append(item.wrap);
    form.style.display = "none";
    const actions = document.createElement("div");
    actions.style.cssText = "display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin-top:8px";
    root.append(heading, status, savedHeading, savedList, form, actions);

    const profileWidget = widget(node, "供应商档案");
    const enabledWidget = widget(node, "启用供应商");
    let profiles = [];
    let current = null;
    let editing = false;
    let saveButton = null;
    let cancelButton = null;

    function setStatus(text, error = false) {
        status.textContent = text;
        status.style.color = error ? "#ff9b91" : "#9dc";
        node.setDirtyCanvas?.(true, true);
    }

    function setEditing(active) {
        editing = Boolean(active);
        form.style.display = editing ? "block" : "none";
        if (saveButton) saveButton.style.display = editing ? "block" : "none";
        if (cancelButton) cancelButton.style.display = editing ? "block" : "none";
        node.setSize?.([Math.max(node.size?.[0] || 0, 460), editing ? 680 : 500]);
        node.setDirtyCanvas?.(true, true);
    }

    function renderProfileList() {
        savedList.replaceChildren();
        if (!profiles.length) {
            const empty = document.createElement("div");
            empty.textContent = "暂无已保存供应商，请点击“新建档案”。";
            empty.style.cssText = "padding:9px;border:1px dashed var(--border-color,#666);border-radius:6px;opacity:.7;font-size:12px";
            savedList.append(empty);
            return;
        }
        for (const profile of profiles) {
            const row = document.createElement("button");
            row.type = "button";
            const selected = current?.id === profile.id;
            row.style.cssText = `display:grid;grid-template-columns:18px 1fr;gap:5px;width:100%;padding:8px;text-align:left;border-radius:6px;cursor:pointer;color:var(--fg-color);background:${selected ? "#477a6655" : "var(--comfy-input-bg,#222)"};border:1px solid ${selected ? "#71c59c" : "var(--border-color,#666)"}`;
            const marker = document.createElement("span");
            marker.textContent = selected ? "●" : "○";
            marker.style.color = selected ? "#71c59c" : "#888";
            const content = document.createElement("span");
            const name = document.createElement("strong");
            name.textContent = profile.name || "未命名供应商";
            name.style.display = "block";
            const detail = document.createElement("small");
            const model = profile.custom_model || profile.luna_model || profile.sol_model || "未映射模型";
            let endpoint = profile.base_url || "未设置 API 地址";
            try { endpoint = new URL(endpoint).host || endpoint; } catch {}
            detail.textContent = `${endpoint} · ${model} · ${profile.credential_configured ? "Key 已配置" : "Key 未配置"}`;
            detail.style.cssText = "display:block;margin-top:2px;opacity:.72;overflow:hidden;text-overflow:ellipsis;white-space:nowrap";
            content.append(name, detail);
            row.append(marker, content);
            row.onclick = (event) => {
                event.preventDefault();
                event.stopPropagation();
                load(profile);
            };
            savedList.append(row);
        }
    }

    function collect() {
        const value = { id: current?.id || "", enabled: enabledWidget?.value !== false };
        for (const [key, item] of Object.entries(fields)) value[key] = item.input.value.trim();
        return value;
    }

    function load(profile) {
        current = profile || null;
        const source = profile || { name: "自定义供应商", base_url: "https://api.openai.com/v1", luna_model: "gpt-5.6-luna", sol_model: "gpt-5.6-sol", terra_model: "gpt-5.6-terra", custom_model: "", fallback_model: "gpt-5.6-sol" };
        for (const [key, item] of Object.entries(fields)) item.input.value = source[key] || "";
        if (enabledWidget) enabledWidget.value = source.enabled !== false;
        // This hidden legacy widget is the actual workflow/backend value.
        // Never put a display label here: legacy LiteGraph serializes .value
        // directly and does not honor serializeValue().
        if (profileWidget) profileWidget.value = source.id || "";
        setStatus(profile ? `${profile.credential_configured ? "🔐 API Key 已加密保存" : "⚠ 尚未配置 API Key"}｜${profile.name}` : "新档案：填写后点击“保存/更新”");
        setEditing(false);
        renderProfileList();
    }

    function rebuildProfileCombo() {
        if (!profileWidget) return;
        // Selection is presented by the DOM list above. Keep this legacy
        // widget only as the stable workflow/backend id carrier.
        profileWidget.type = "hidden";
        profileWidget.computeSize = () => [0, -4];
    }

    async function refresh(selectId = current?.id || profileWidget?.value) {
        try {
            const data = await requestJson(`${BASE}?t=${Date.now()}`, { cache: "no-store" });
            profiles = data.profiles || [];
            rebuildProfileCombo();
            const selected = profiles.find((item) => item.id === selectId)
                // Compatibility migration for workflows saved by the previous
                // UI revision, which accidentally serialized the display name.
                || profiles.find((item) => item.name === selectId)
                || profiles[0];
            if (selected) load(selected);
            else load(null);
            renderProfileList();
        } catch (error) {
            setStatus(`读取失败：${error.message}`, true);
        }
    }

    async function save(apiKey = "") {
        try {
            setStatus("正在加密并保存…");
            const payload = collect();
            if (apiKey) payload.api_key = apiKey;
            const data = await post(`${BASE}/save`, payload);
            await refresh(data.profile.id);
            setStatus(`${data.profile.credential_configured ? "🔐 API Key 已加密保存" : "⚠ 档案已保存，尚未配置 API Key"}｜${data.profile.name}`);
            setEditing(false);
        } catch (error) {
            setStatus(`保存失败：${error.message}`, true);
        }
    }

    const newButton = button("新建档案", () => { load(null); setEditing(true); });
    const editButton = button("编辑档案", () => {
        if (!current?.id) return setStatus("请先选择档案", true);
        setEditing(true);
        fields.name.input.focus();
    });
    const copyButton = button("复制档案", () => {
        if (!current?.id) return setStatus("请先选择档案", true);
        current = null;
        fields.name.input.value = `${fields.name.input.value} 副本`;
        if (profileWidget) profileWidget.value = "";
        setStatus("副本尚未保存");
        setEditing(true);
    });
    saveButton = button("保存/更新", () => save());
    cancelButton = button("取消编辑", () => {
        const selected = profiles.find((item) => item.id === current?.id) || profiles[0] || null;
        load(selected);
    });
    saveButton.style.display = "none";
    cancelButton.style.display = "none";

    actions.append(
        newButton,
        editButton,
        copyButton,
        button("配置/替换 API Key", async () => { const key = await passwordDialog(); if (key) await save(key); }),
        button("删除 API Key", async () => {
            if (!current?.id) return setStatus("请先选择档案", true);
            if (!window.confirm(`确认删除“${current.name}”的本机加密 API Key？`)) return;
            try { await post(`${BASE}/credential/delete`, { id: current.id }); await refresh(current.id); } catch (error) { setStatus(`删除失败：${error.message}`, true); }
        }, true),
        button("删除档案", async () => {
            if (!current?.id) return setStatus("请先选择档案", true);
            if (!window.confirm(`确认删除供应商档案“${current.name}”及其加密 API Key？`)) return;
            try { await post(`${BASE}/delete`, { id: current.id }); await refresh(""); } catch (error) { setStatus(`删除失败：${error.message}`, true); }
        }, true),
        button("测试连接 / 获取模型", async () => {
            if (!current?.id) return setStatus("请先保存档案", true);
            try {
                setStatus("正在请求供应商 /models（不会生成内容）…");
                const data = await post(`${BASE}/models`, { id: current.id, timeout: 15 });
                setStatus(`连接成功｜${data.elapsed_ms} ms｜发现 ${data.count} 个模型`);
                modelDialog(data.models, (target, value) => { fields[target].input.value = value; setEditing(true); setStatus(`已将 ${value} 填入对应映射，请点击“保存/更新”`); });
            } catch (error) { setStatus(error.message, true); }
        }),
        button("刷新档案", () => refresh()),
        saveButton,
        cancelButton,
    );
    return { root, refresh };
}

app.registerExtension({
    name: "Yeban.MossBBox.ProviderManager",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== NODE_ID) return;
        const original = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = original?.apply(this, arguments);
            const panel = createPanel(this);
            this.addDOMWidget("供应商管理面板", "div", panel.root, { serialize: false, hideOnZoom: false });
            this.setSize?.([Math.max(this.size?.[0] || 0, 460), 500]);
            const originalConfigure = this.onConfigure;
            this.onConfigure = function () {
                const configureResult = originalConfigure?.apply(this, arguments);
                // ComfyUI restores widgets_values during configure. Defer the
                // profile lookup until that saved id is available.
                setTimeout(() => {
                    const savedId = widget(this, "供应商档案")?.value || "";
                    void panel.refresh(savedId);
                }, 0);
                return configureResult;
            };
            // New nodes do not receive a configure callback. This deferred
            // fallback therefore selects the first profile only when no saved
            // workflow id exists.
            setTimeout(() => {
                const savedId = widget(this, "供应商档案")?.value || "";
                void panel.refresh(savedId);
            }, 0);
            return result;
        };
    },
});



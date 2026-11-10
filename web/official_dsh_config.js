import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
    name: "Yeban.MossBBox.OfficialDshConfig",
    async nodeCreated(node) {
        if (node.comfyClass !== "YebanMossBBoxDshConfig") return;
        const widget = name => node.widgets.find(item => item.name === name);
        const provider = widget("dsh_provider"), model = widget("model"), branch = widget("dsh_branch_session");
        if (!provider || !model || !branch) return;
        let groups = [], branches = [], sourceId = branch.value === "new_session" ? "" : branch.value;
        provider.type = "converted-widget";
        provider.hidden = true;
        provider.computeSize = () => [0, -4];
        model.label = "模型";
        const providerLabel = item => {
            const name = item.name || item.id;
            return groups.filter(other => (other.name || other.id) === name).length > 1 ? `${name} (${item.id})` : name;
        };
        const providerChoice = node.addWidget("combo", "供应商", provider.value, label => {
            const selected = groups.find(item => providerLabel(item) === label);
            if (!selected) return;
            provider.value = selected.id;
            provider.callback?.(selected.id);
        }, { values: [provider.value], serialize: false });
        providerChoice.serialize = false;
        const updateProviderChoice = () => {
            const selected = groups.find(item => item.id === provider.value);
            providerChoice.value = selected ? providerLabel(selected) : provider.value;
            providerChoice.options.values = [...new Set([providerChoice.value, ...groups.map(providerLabel)])];
        };
        let action = widget("dsh_action");
        if (!action) action = node.addWidget("combo", "dsh_action", "continue", () => {}, { values: ["continue", "fork", "blank"] });
        action.type = "converted-widget"; action.hidden = true; action.computeSize = () => [0, -4];
        const modes = { "接续此对话": "continue", "从此对话新建分支": "fork", "新建空白对话": "blank" };
        const info = node.addWidget("text", "分支标题与时间", "", () => {}, { serialize: false });
        info.options.read_only = true; info.serialize = false;
        const baseTitle = item => item.title + (item.created_at ? " · " + new Date(item.created_at).toLocaleString("zh-CN", { timeZone: "Asia/Shanghai", hour12: false }) : "");
        const titleFor = item => baseTitle(item) + (branches.filter(other => baseTitle(other) === baseTitle(item)).length > 1 ? ` (${item.id})` : "");
        const pick = node.addWidget("combo", "选择对话", "请选择对话", label => {
            if (busy) return;
            const selected = branches.find(item => titleFor(item) === label);
            sourceId = selected?.id || ""; requestId = null;
            branch.value = action.value === "blank" ? "new_session" : sourceId || "new_session";
            updateInfo();
        }, { values: ["请选择对话"], serialize: false });
        pick.serialize = false;
        branch.type = "converted-widget"; branch.hidden = true; branch.computeSize = () => [0, -4];
        const updateInfo = () => {
            const selected = branches.find(item => item.id === sourceId);
            const time = selected?.created_at ? new Date(selected.created_at).toLocaleString("zh-CN", { timeZone: "Asia/Shanghai", hour12: false }) : "";
            info.value = selected ? `${selected.title} · ${time}` : branch.value === "new_session" ? "每次新建空白对话" : "已保存分支；刷新以读取标题";
            pick.options.values = ["请选择对话", ...branches.map(titleFor)];
            pick.value = selected ? titleFor(selected) : sourceId ? "已保存对话（目录暂不可用）" : "请选择对话";
            mode.value = Object.keys(modes).find(label => modes[label] === action.value);
            for (const control of [title, create]) {
                control.hidden = action.value !== "fork";
                control.type = control.hidden ? "converted-widget" : control === title ? "text" : "button";
                control.computeSize = control.hidden ? () => [0, -4] : undefined;
            }
            info.value = action.value === "blank" ? "每次任务新建空白对话" : info.value;
            if (node.size && node.computeSize && node.setSize) node.setSize([node.size[0], node.computeSize()[1]]);
            node.setDirtyCanvas(true, true);
        };
        const mode = node.addWidget("combo", "使用方式", "接续此对话", label => {
            if (busy) return;
            action.value = modes[label];
            branch.value = action.value === "blank" ? "new_session" : sourceId || "new_session";
            updateInfo();
        }, { values: Object.keys(modes), serialize: false });
        mode.serialize = false;
        const title = node.addWidget("text", "新分支标题", "", () => {}, { serialize: false });
        title.serialize = false; title.callback = () => { requestId = null; };
        let busy = false, requestId = null;
        const create = node.addWidget("button", "创建并使用新分支", null, async () => {
            if (busy) return;
            if (!sourceId) { info.value = "请先选择来源对话"; return; }
            const parent = branches.find(item => item.id === sourceId);
            const name = title.value.trim() || (parent?.title || "对话").slice(0, 100) + " · 新分支";
            requestId ||= crypto.randomUUID(); busy = true; info.value = "正在创建分支…";
            try {
                const response = await api.fetchApi("/yeban/moss-bbox-standalone/official-dsh/branches", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ source_session: sourceId, title: name, request_id: requestId }) });
                const data = await response.json();
                if (!response.ok || !data.ok) throw new Error(data.error || "分支创建失败");
                branches.push(data.branch); sourceId = branch.value = data.branch.id; action.value = "continue";
                branch.options.values = [...new Set([...branch.options.values, sourceId])];
                title.value = ""; requestId = null; updateInfo();
            } catch (error) { info.value = error.message; }
            finally { busy = false; }
        }, { serialize: false });
        create.serialize = false;
        const oldConfigure = node.onConfigure;
        node.onConfigure = function(...args) { const result = oldConfigure?.apply(this, args); sourceId = branch.value === "new_session" ? "" : branch.value; if (!sourceId) action.value = "blank"; updateProviderChoice(); updateModels(); updateInfo(); return result; };
        const updateModels = () => {
            const options = groups.find(item => item.id === provider.value)?.models.map(item => item.id) || [];
            if (options.length) {
                model.options.values = options;
                if (!options.includes(model.value)) model.value = options[0];
            }
        };
        const oldProvider = provider.callback, oldBranch = branch.callback;
        provider.callback = (...args) => { oldProvider?.(...args); updateProviderChoice(); updateModels(); node.setDirtyCanvas(true, true); };
        branch.callback = (...args) => { oldBranch?.(...args); sourceId = branch.value === "new_session" ? "" : branch.value; if (!sourceId) action.value = "blank"; updateInfo(); };
        const refresh = async () => {
            try {
                const response = await api.fetchApi("/yeban/moss-bbox-standalone/official-dsh/catalog");
                const data = await response.json();
                if (!response.ok || !data.ok) throw new Error(data.error);
                groups = data.groups || []; branches = data.branches || [];
                provider.options.values = [...new Set([provider.value, ...groups.map(item => item.id)])];
                branch.options.values = [...new Set(["new_session", branch.value, ...branches.map(item => item.id)])];
                updateProviderChoice(); updateModels(); updateInfo();
            } catch { info.value = "读取失败，请检查官方 DSH 与故事板服务"; }
        };
        const manage = node.addWidget("button", "管理 DSH 供应商", null, async () => {
            if (busy) return;
            busy = true;
            try {
                const response = await api.fetchApi("/yeban/moss-bbox-standalone/official-dsh/open-desktop", { method: "POST" });
                const data = await response.json();
                if (!response.ok || !data.ok) throw new Error(data.error || "无法打开 DSH 桌面版");
                info.value = "在 DSH 中保存供应商后，回这里点刷新";
            } catch (error) {
                info.value = error.message;
            } finally {
                busy = false;
                node.setDirtyCanvas(true, true);
            }
        }, { serialize: false });
        manage.serialize = false;
        node.addWidget("button", "刷新官方 DSH 配置", null, refresh, { serialize: false });
        if (!sourceId) action.value = "blank";
        updateInfo();
        // Delay initial refresh until saved workflow values have been restored.
        setTimeout(refresh, 100);
    },
});



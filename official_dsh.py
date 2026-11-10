"""Official DSH transport bridge; credentials stay in the official profile."""
import contextvars
import functools
import importlib.util
import ipaddress
import json
import os
import re
import tempfile
import time
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

CURRENT_CONFIG = contextvars.ContextVar("yeban_official_dsh_config", default=None)

@functools.lru_cache(maxsize=1)
def transport():
    source = Path(__file__).resolve().parent / "official_dsh_adapter.py"
    if not source.is_file():
        raise RuntimeError("官方 DSH SDK 适配器不可用，请安装故事板集成")
    spec = importlib.util.spec_from_file_location("yeban_official_dsh_transport", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def catalog():
    return transport().model_catalog()

def _home():
    return transport().official_home()

def session_file(session_id):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", str(session_id)):
        return None
    return next((_home() / "sessions").glob("*/" + session_id + "/session.v4.jsonl.zstd"), None)

def branches():
    cache = _home() / "cache" / "yeban_branch_catalog.json"
    try:
        with urlopen("http://127.0.0.1:8348/api/director/dsh-sessions", timeout=10) as response:
            data = json.load(response)
            rows = data.get("sessions", data.get("settings", {}).get("dsh_branches", []))
        rows = [{key: row.get(key, "") for key in ("id", "title", "created_at", "parent_session")} for row in rows if re.fullmatch(r"[A-Za-z0-9_-]{1,100}", str(row.get("id", "")))]
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=cache.parent, delete=False) as handle:
                json.dump(rows, handle, ensure_ascii=False)
                temporary = Path(handle.name)
            try: os.replace(temporary, cache)
            finally: temporary.unlink(missing_ok=True)
        except OSError:
            pass
        return rows
    except Exception:
        try:
            rows = json.loads(cache.read_text(encoding="utf-8"))
            return [row for row in rows if isinstance(row, dict) and re.fullmatch(r"[A-Za-z0-9_-]{1,100}", str(row.get("id", "")))] if isinstance(rows, list) else []
        except (OSError, ValueError): return []

EXECUTION = contextvars.ContextVar("yeban_prompt_execution", default=None)

def remaining_timeout(requested, record_call=True):
    state = EXECUTION.get()
    if state is None:
        return requested
    remaining = state["deadline"] - time.monotonic()
    if remaining < 1:
        raise TimeoutError("节点总时限已到，已停止后续模型调用")
    if record_call: state["calls"] += 1
    return min(int(requested), int(remaining))

def error_message(error):
    messages = {
        "DSH_OFFICIAL_NOT_INSTALLED": "未找到官方 DSH 桌面版安装，请检查安装目录",
        "DSH_OFFICIAL_PROFILE_INVALID": "官方 DSH 配置格式无效，请检查供应商设置",
        "DSH_OFFICIAL_CATALOG_EMPTY": "官方 DSH 尚未配置可用模型",
        "DSH_OFFICIAL_INSTALL_UNAVAILABLE": "未找到官方 DSH 桌面版安装，请检查安装目录",
        "DSH_OFFICIAL_ENTRY_UNAVAILABLE": "官方 DSH SDK 文件缺失，请修复桌面版安装",
        "DSH_OFFICIAL_TIMEOUT": "官方 DSH 调用超时，请检查网络或增加总时限",
        "DSH_OFFICIAL_SIGN_IN_REQUIRED": "请先在官方 DSH 桌面版登录",
        "DSH_OFFICIAL_AUTH_REQUIRED": "官方 DSH 供应商认证不可用，请检查凭据",
        "DSH_OFFICIAL_MODEL_UNAVAILABLE": "官方 DSH 模型不可用，请检查供应商和模型配置",
        "DSH_OFFICIAL_PROFILE_UNAVAILABLE": "未找到官方 DSH 配置，请先配置官方桌面版",
        "DSH_OFFICIAL_BRANCH_INVALID": "DSH 分支标识无效，请重新选择分支",
        "DSH_OFFICIAL_PROMPT_REJECTED": "官方 DSH 未接受请求，请检查分支是否存在或正在执行",
        "DSH_OFFICIAL_EXITED": "官方 DSH SDK 进程意外退出，请检查桌面版安装",
        "DSH_OFFICIAL_EMPTY_RESPONSE": "官方 DSH 没有返回有效文本",
        "DSH_OFFICIAL_INVALID_RPC_FRAME": "官方 DSH 返回了无法解析的 SDK 消息",
        "DSH_OFFICIAL_MODEL_FAILED": "官方 DSH 模型执行失败，请检查供应商服务",
    }
    if isinstance(error, TimeoutError):
        return messages["DSH_OFFICIAL_TIMEOUT"]
    return messages.get(str(error), "官方 DSH 调用失败，请检查供应商、模型、分支和登录状态")


def defaults():
    try: data = catalog()
    except Exception as error: raise RuntimeError(error_message(error)) from None
    groups = data.get("groups", [])
    default = data.get("default", {})
    provider = default.get("provider") or (groups[0]["id"] if groups else "deepseek-account")
    return {"dsh_provider": provider, "model": default.get("model") or "deepseek-v4.1-flash", "dsh_branch_session": ""}

def scoped(function):
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        token = CURRENT_CONFIG.set(kwargs.get("dsh_config"))
        try:
            return function(*args, **kwargs)
        finally:
            CURRENT_CONFIG.reset(token)
    return wrapped

class YebanOfficialDshConfig:
    @classmethod
    def INPUT_TYPES(cls):
        try:
            data = catalog()
        except Exception:
            data = {"groups": [], "default": {}}
        groups = data.get("groups", [])
        providers = [group["id"] for group in groups] or ["deepseek-account"]
        models = list(dict.fromkeys(model["id"] for group in groups for model in group.get("models", []))) or ["deepseek-v4.1-flash"]
        source = branches()
        ids = ["new_session"] + [branch["id"] for branch in source]
        hint = "new_session：每次新建对话；已有分支继续上下文。\n" + "\n".join(f"{branch['id']}: {branch.get('title', '')} · {branch.get('created_at', '')}" for branch in source)
        default = data.get("default", {})
        return {"required": {
            "dsh_provider": (providers, {"default": default.get("provider") if default.get("provider") in providers else providers[0], "display_name": "官方 DSH 供应商"}),
            "model": (models, {"default": default.get("model") if default.get("model") in models else models[0], "display_name": "官方 DSH 模型"}),
            "dsh_branch_session": (ids, {"default": "new_session", "display_name": "接续对话分支", "tooltip": hint}),
        }, "optional": {"dsh_action": (["continue", "fork", "blank"], {"default": "continue", "display_name": "使用方式"})}}
    @classmethod
    def VALIDATE_INPUTS(cls, dsh_provider, model, dsh_branch_session="new_session", dsh_action="continue"):
        if dsh_action not in ("continue", "fork", "blank"):
            return "DSH 使用方式无效"
        if not re.fullmatch(r"[A-Za-z0-9_.:-]{1,100}", str(dsh_provider)):
            return "DSH 供应商标识无效"
        if not model or len(str(model)) > 200 or re.search(r"[\r\n\x00]", str(model)):
            return "DSH 模型标识无效"
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", str(dsh_branch_session)):
            return "DSH 分支标识无效"
        return True

    RETURN_TYPES = ("YEBAN_OFFICIAL_DSH_CONFIG",)
    RETURN_NAMES = ("官方 DSH 配置",)
    FUNCTION = "configure"
    CATEGORY = "夜半/MOSS BBox"
    def configure(self, dsh_provider, model, dsh_branch_session="new_session", dsh_action="continue"):
        validation = self.VALIDATE_INPUTS(dsh_provider, model, dsh_branch_session, dsh_action)
        if validation is not True:
            raise ValueError(validation)
        if dsh_action == "fork":
            raise ValueError("请先点击创建并使用新分支，再运行工作流")
        if dsh_action == "blank":
            dsh_branch_session = "new_session"
        if dsh_branch_session != "new_session" and session_file(dsh_branch_session) is None:
            raise ValueError("所选 DSH 分支已不存在，请重新选择；不会自动新建空白对话")
        try: data = catalog()
        except Exception as error: raise ValueError(error_message(error)) from None
        group = next((row for row in data.get("groups", []) if row["id"] == dsh_provider), None)
        if not group or model not in [row["id"] for row in group.get("models", [])]:
            raise ValueError("所选官方 DSH 供应商没有配置此模型")
        return ({"dsh_provider": dsh_provider, "model": model, "dsh_branch_session": "" if dsh_branch_session == "new_session" else dsh_branch_session},)


def open_desktop():
    if os.name != "nt":
        raise RuntimeError("当前系统不支持从此节点打开 DSH 桌面版")
    installed = Path(os.environ.get("DSH_OFFICIAL_INSTALL") or Path.home() / "AppData/Local/Programs/DeepSeek Harness")
    executable = installed / "DeepSeek Harness.exe"
    if not executable.is_file():
        raise RuntimeError("未找到官方 DSH 桌面版")
    os.startfile(str(executable))


def register_routes():
    try:
        import asyncio
        from aiohttp import web
        from server import PromptServer
        routes = PromptServer.instance.routes
    except (ImportError, AttributeError):
        return
    @routes.post("/yeban/moss-bbox-standalone/official-dsh/open-desktop")
    async def launch_desktop(request):
        try:
            address = ipaddress.ip_address(request.remote or "")
            local = address.is_loopback or bool(getattr(address, "ipv4_mapped", None) and address.ipv4_mapped.is_loopback)
            origin = request.headers.get("Origin")
            if not local or (origin and urlsplit(origin).netloc.casefold() != request.host.casefold()):
                return web.json_response({"ok": False, "error": "请从本机 ComfyUI 页面打开 DSH 桌面版"}, status=403)
        except ValueError:
            return web.json_response({"ok": False, "error": "无法确认本机访问"}, status=403)
        try:
            await asyncio.to_thread(open_desktop)
            return web.json_response({"ok": True})
        except (OSError, RuntimeError) as exc:
            message = str(exc) if isinstance(exc, RuntimeError) else "打开 DSH 桌面版失败，请从开始菜单启动"
            return web.json_response({"ok": False, "error": message}, status=503)

    @routes.get("/yeban/moss-bbox-standalone/official-dsh/catalog")
    async def read_catalog(request):
        try:
            data = await asyncio.to_thread(catalog)
            source = await asyncio.to_thread(branches)
            return web.json_response({"ok": True, **data, "branches": source})
        except Exception:
            return web.json_response({"ok": False, "error": "官方 DSH 配置读取失败"}, status=503)
    @routes.post("/yeban/moss-bbox-standalone/official-dsh/branches")
    async def create_branch(request):
        def create(payload):
            body = json.dumps(payload).encode("utf-8")
            with urlopen(Request("http://127.0.0.1:8348/api/director/dsh-branches", data=body, headers={"Content-Type": "application/json"}), timeout=50) as response:
                return json.load(response)
        try:
            payload = await request.json()
            result = await asyncio.to_thread(create, payload)
            return web.json_response(result)
        except Exception:
            return web.json_response({"ok": False, "error": "新建分支失败，请检查来源对话与故事板服务"}, status=400)



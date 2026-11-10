"""Local encrypted provider profiles for Yeban Codex nodes."""

from __future__ import annotations

import asyncio
import base64
import ctypes
from ctypes import wintypes
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import uuid


DEFAULT_BASE_URL = "https://api.openai.com/v1"
DATA_SCHEMA_VERSION = 1
PROFILE_FILENAME = "providers.json"
CREDENTIAL_FILENAME = "credentials.json"
LOG_PREFIX = "[Yeban Codex Provider]"
PROVIDER_TRANSPORT_OPENAI = "openai_compatible"
PROVIDER_TRANSPORT_CODEX_CLI = "codex_cli"


class ProviderManagerError(RuntimeError):
    """A safe, user-facing provider-manager error."""


def _log(message: str) -> None:
    print(f"{LOG_PREFIX} {message}", flush=True)


def _data_directory() -> Path:
    try:
        import folder_paths  # type: ignore

        root = Path(folder_paths.get_user_directory())
    except (ImportError, AttributeError, TypeError):
        root = Path(__file__).resolve().parent / ".user_data"
    path = root / "Yeban_MOSS_BBox"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _profile_path() -> Path:
    return _data_directory() / PROFILE_FILENAME


def _credential_path() -> Path:
    return _data_directory() / CREDENTIAL_FILENAME


def _empty_profiles() -> dict[str, Any]:
    return {"schema_version": DATA_SCHEMA_VERSION, "profiles": {}}


def _empty_credentials() -> dict[str, Any]:
    return {"schema_version": DATA_SCHEMA_VERSION, "credentials": {}}


def _read_json(path: Path, fallback: dict[str, Any]) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else fallback
    except (OSError, UnicodeError, ValueError):
        return fallback


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass


class _DataBlob(ctypes.Structure):
    _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_byte))]


def _blob(data: bytes) -> tuple[_DataBlob, Any]:
    buffer = ctypes.create_string_buffer(data)
    return _DataBlob(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_byte))), buffer


def _dpapi_encrypt(secret: str) -> str:
    raw = secret.encode("utf-8")
    if not raw:
        raise ProviderManagerError("API Key 不能为空")
    if os.name != "nt":
        raise ProviderManagerError("加密保存 API Key 目前仅支持 Windows")
    input_blob, input_buffer = _blob(raw)
    output_blob = _DataBlob()
    crypt32 = ctypes.windll.crypt32
    kernel32 = ctypes.windll.kernel32
    success = crypt32.CryptProtectData(
        ctypes.byref(input_blob),
        "Yeban Codex Provider",
        None,
        None,
        None,
        0x01,
        ctypes.byref(output_blob),
    )
    del input_buffer
    if not success:
        raise ProviderManagerError(f"Windows DPAPI 加密失败：{ctypes.get_last_error()}")
    try:
        encrypted = ctypes.string_at(output_blob.pbData, output_blob.cbData)
        return base64.b64encode(encrypted).decode("ascii")
    finally:
        kernel32.LocalFree(output_blob.pbData)


def _dpapi_decrypt(ciphertext: str) -> str:
    if os.name != "nt":
        raise ProviderManagerError("读取加密 API Key 目前仅支持 Windows")
    try:
        encrypted = base64.b64decode(ciphertext.encode("ascii"), validate=True)
    except (ValueError, UnicodeError) as exc:
        raise ProviderManagerError("本机 API Key 密文已损坏") from exc
    input_blob, input_buffer = _blob(encrypted)
    output_blob = _DataBlob()
    crypt32 = ctypes.windll.crypt32
    kernel32 = ctypes.windll.kernel32
    success = crypt32.CryptUnprotectData(
        ctypes.byref(input_blob), None, None, None, None, 0x01, ctypes.byref(output_blob)
    )
    del input_buffer
    if not success:
        raise ProviderManagerError("无法解密 API Key；它可能由其他 Windows 用户保存")
    try:
        return ctypes.string_at(output_blob.pbData, output_blob.cbData).decode("utf-8")
    except UnicodeError as exc:
        raise ProviderManagerError("解密后的 API Key 无效") from exc
    finally:
        kernel32.LocalFree(output_blob.pbData)


def _normalise_profile_id(value: Any) -> str:
    profile_id = str(value or "").strip()
    if profile_id and re.fullmatch(r"[A-Za-z0-9_-]{1,80}", profile_id):
        return profile_id
    return uuid.uuid4().hex


def _normalise_base_url(value: Any) -> str:
    base_url = str(value or DEFAULT_BASE_URL).strip().rstrip("/")
    if not base_url.startswith(("https://", "http://")):
        raise ProviderManagerError("兼容 API 地址必须以 http:// 或 https:// 开头")
    return base_url


def _normalise_transport(payload: Mapping[str, Any]) -> str:
    """Resolve the hidden transport for a provider profile.

    Existing Packy profiles are detected by name or endpoint so upgrading the
    node does not require the user to edit and resave the profile.
    """
    raw = str(payload.get("transport") or payload.get("execution_mode") or "").strip().casefold()
    if raw in {"codex_cli", "codex-cli", "packy", "packycode", "packy_code"}:
        return PROVIDER_TRANSPORT_CODEX_CLI
    name = str(payload.get("name") or "").strip().casefold()
    base_url = str(payload.get("base_url") or "").strip().casefold()
    if "packy" in name or "cf.api.fan" in base_url or "packyapi" in base_url:
        return PROVIDER_TRANSPORT_CODEX_CLI
    return PROVIDER_TRANSPORT_OPENAI


def _clean_profile(payload: Mapping[str, Any], *, profile_id: str | None = None) -> dict[str, Any]:
    now = int(time.time())
    return {
        "id": _normalise_profile_id(profile_id or payload.get("id")),
        "name": str(payload.get("name") or "自定义供应商").strip()[:100],
        "enabled": bool(payload.get("enabled", True)),
        "base_url": _normalise_base_url(payload.get("base_url")),
        "luna_model": str(payload.get("luna_model") or "gpt-5.6-luna").strip()[:200],
        "sol_model": str(payload.get("sol_model") or "gpt-5.6-sol").strip()[:200],
        "terra_model": str(payload.get("terra_model") or "gpt-5.6-terra").strip()[:200],
        "custom_model": str(payload.get("custom_model") or "").strip()[:200],
        "fallback_model": str(payload.get("fallback_model") or "gpt-5.6-sol").strip()[:200],
        "transport": _normalise_transport(payload),
        "updated_at": now,
    }


def _load_profiles() -> dict[str, Any]:
    data = _read_json(_profile_path(), _empty_profiles())
    profiles = data.get("profiles")
    data["profiles"] = profiles if isinstance(profiles, dict) else {}
    return data


def _load_credentials() -> dict[str, Any]:
    data = _read_json(_credential_path(), _empty_credentials())
    credentials = data.get("credentials")
    data["credentials"] = credentials if isinstance(credentials, dict) else {}
    return data


def _credential_exists(profile_id: str) -> bool:
    item = _load_credentials()["credentials"].get(profile_id)
    return isinstance(item, Mapping) and bool(item.get("ciphertext"))


def _public_profile(profile: Mapping[str, Any]) -> dict[str, Any]:
    public = {key: value for key, value in profile.items() if key != "ciphertext"}
    public["credential_configured"] = _credential_exists(str(profile.get("id") or ""))
    return public


def list_profiles() -> list[dict[str, Any]]:
    profiles = [_public_profile(item) for item in _load_profiles()["profiles"].values() if isinstance(item, Mapping)]
    return sorted(profiles, key=lambda item: (str(item.get("name", "")).casefold(), str(item.get("id", ""))))


def save_profile(payload: Mapping[str, Any], api_key: str = "") -> dict[str, Any]:
    profile_id = _normalise_profile_id(payload.get("id"))
    profiles_data = _load_profiles()
    existing = profiles_data["profiles"].get(profile_id)
    profile = _clean_profile(payload, profile_id=profile_id)
    profile["created_at"] = int(existing.get("created_at", time.time())) if isinstance(existing, Mapping) else int(time.time())
    profiles_data["profiles"][profile_id] = profile
    _atomic_write_json(_profile_path(), profiles_data)
    if str(api_key or "").strip():
        credentials_data = _load_credentials()
        credentials_data["credentials"][profile_id] = {
            "ciphertext": _dpapi_encrypt(str(api_key).strip()),
            "updated_at": int(time.time()),
        }
        _atomic_write_json(_credential_path(), credentials_data)
        _log(f"credential_saved profile_id={profile_id}")
    _log(f"profile_saved profile_id={profile_id} name={profile['name']}")
    return _public_profile(profile)


def delete_profile(profile_id: str) -> bool:
    profile_id = _normalise_profile_id(profile_id)
    profiles_data = _load_profiles()
    existed = profiles_data["profiles"].pop(profile_id, None) is not None
    _atomic_write_json(_profile_path(), profiles_data)
    credentials_data = _load_credentials()
    credentials_data["credentials"].pop(profile_id, None)
    _atomic_write_json(_credential_path(), credentials_data)
    _log(f"profile_deleted profile_id={profile_id} existed={str(existed).lower()}")
    return existed


def delete_credential(profile_id: str) -> bool:
    profile_id = _normalise_profile_id(profile_id)
    credentials_data = _load_credentials()
    existed = credentials_data["credentials"].pop(profile_id, None) is not None
    _atomic_write_json(_credential_path(), credentials_data)
    _log(f"credential_deleted profile_id={profile_id} existed={str(existed).lower()}")
    return existed


def _get_saved_profile(profile_id: str) -> dict[str, Any]:
    item = _load_profiles()["profiles"].get(str(profile_id or "").strip())
    if not isinstance(item, Mapping):
        raise ProviderManagerError("请选择并保存一个供应商档案")
    return dict(item)


def _get_api_key(profile_id: str) -> str:
    item = _load_credentials()["credentials"].get(profile_id)
    if not isinstance(item, Mapping) or not item.get("ciphertext"):
        raise ProviderManagerError("该供应商尚未配置 API Key，请点击“配置/替换 API Key”")
    return _dpapi_decrypt(str(item["ciphertext"]))


def provider_runtime(profile_id: str, enabled: bool = True) -> dict[str, Any]:
    profile = _get_saved_profile(profile_id)
    effective_enabled = bool(enabled) and bool(profile.get("enabled", True))
    if not effective_enabled:
        return {"enabled": False, "name": str(profile.get("name") or "自定义供应商")}
    api_key = _get_api_key(str(profile["id"]))
    return {
        "enabled": True,
        "name": str(profile.get("name") or "自定义供应商"),
        "base_url": _normalise_base_url(profile.get("base_url")),
        "key_mode": "节点内直接填写（明文）",
        "api_key": api_key,
        "luna_model": str(profile.get("luna_model") or ""),
        "sol_model": str(profile.get("sol_model") or ""),
        "terra_model": str(profile.get("terra_model") or ""),
        "custom_model": str(profile.get("custom_model") or ""),
        "fallback_model": str(profile.get("fallback_model") or ""),
        "transport": _normalise_transport(profile),
        "profile_id": str(profile["id"]),
        "managed": True,
    }


def fetch_models(profile_id: str, timeout: int = 15) -> dict[str, Any]:
    profile = _get_saved_profile(profile_id)
    api_key = _get_api_key(profile_id)
    request = Request(
        _normalise_base_url(profile.get("base_url")) + "/models",
        headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json", "User-Agent": "Yeban-Codex-Provider-Manager/1.0"},
        method="GET",
    )
    started = time.monotonic()
    try:
        with urlopen(request, timeout=max(1, min(int(timeout), 60))) as response:  # nosec B310
            raw = response.read()
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500].replace(api_key, "[REDACTED]")
        raise ProviderManagerError(f"连接失败：HTTP {exc.code}" + (f"，{detail}" if detail else "")) from exc
    except (URLError, TimeoutError, OSError) as exc:
        message = str(exc).replace(api_key, "[REDACTED]")
        raise ProviderManagerError(f"连接失败：{message}") from exc
    try:
        body = json.loads(raw.decode("utf-8"))
    except (UnicodeError, ValueError) as exc:
        raise ProviderManagerError("供应商 /models 返回了无效 JSON") from exc
    candidates = body.get("data", []) if isinstance(body, Mapping) else []
    models = sorted({str(item.get("id")) for item in candidates if isinstance(item, Mapping) and item.get("id")})
    elapsed_ms = round((time.monotonic() - started) * 1000)
    _log(f"models_ok profile_id={profile_id} count={len(models)} elapsed_ms={elapsed_ms}")
    return {"ok": True, "models": models, "count": len(models), "elapsed_ms": elapsed_ms}


def _revision(profile_id: str) -> str:
    digest = hashlib.sha256(str(profile_id or "").encode("utf-8"))
    for path in (_profile_path(), _credential_path()):
        try:
            stat = path.stat()
            digest.update(f"{stat.st_mtime_ns}:{stat.st_size}".encode("ascii"))
        except OSError:
            digest.update(b"missing")
    return digest.hexdigest()


class YebanCodexProviderManager:
    """Load an encrypted local provider profile into a provider pipe."""

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {"required": {
            "供应商档案": ("STRING", {"default": "", "display_name": "供应商档案（选择）"}),
            "启用供应商": ("BOOLEAN", {"default": True, "label_on": "启用", "label_off": "停用"}),
        }}

    RETURN_TYPES = ("YEBAN_CODEX_PROVIDER", "STRING", "STRING")
    RETURN_NAMES = ("供应商配置", "连接状态", "供应商信息")
    FUNCTION = "load_provider"
    CATEGORY = "夜半/MOSS BBox"

    @classmethod
    def IS_CHANGED(cls, **kwargs: Any) -> str:
        return _revision(str(kwargs.get("供应商档案") or ""))

    def load_provider(self, **kwargs: Any) -> tuple[dict[str, Any], str, str]:
        profile_id = str(kwargs.get("供应商档案") or "").strip()
        enabled = bool(kwargs.get("启用供应商", True))
        try:
            runtime = provider_runtime(profile_id, enabled)
            name = str(runtime.get("name") or "自定义供应商")
            if not runtime.get("enabled"):
                return runtime, "已停用", f"{name}｜不会覆盖下游节点的后端设置"
            profile = _get_saved_profile(profile_id)
            info = f"{name}｜{profile.get('base_url')}｜Luna: {profile.get('luna_model') or '未映射'}"
            return runtime, "已加载加密凭据", info
        except ProviderManagerError as exc:
            return {"enabled": True, "name": "供应商未就绪", "error": str(exc)}, f"配置错误：{exc}", "请在节点管理面板中完成并保存配置"


NODE_CLASS_MAPPINGS = {"YebanCodexProviderManager": YebanCodexProviderManager}
NODE_DISPLAY_NAME_MAPPINGS = {"YebanCodexProviderManager": "夜半 Codex 供应商管理器"}


async def _json_payload(request: Any) -> Mapping[str, Any]:
    value = await request.json()
    if not isinstance(value, Mapping):
        raise ProviderManagerError("请求内容必须是 JSON 对象")
    return value


def _register_routes() -> None:
    try:
        from aiohttp import web
        from server import PromptServer
    except (ImportError, AttributeError):
        return
    try:
        routes = PromptServer.instance.routes
    except AttributeError:
        return

    @routes.get("/yeban/moss-bbox-standalone/providers")
    async def providers_list(_request: Any) -> Any:
        return web.json_response({"profiles": await asyncio.to_thread(list_profiles)})

    @routes.post("/yeban/moss-bbox-standalone/providers/save")
    async def providers_save(request: Any) -> Any:
        try:
            payload = await _json_payload(request)
            profile = await asyncio.to_thread(save_profile, payload, str(payload.get("api_key") or ""))
            return web.json_response({"ok": True, "profile": profile})
        except ProviderManagerError as exc:
            return web.json_response({"error": str(exc)}, status=400)
        except Exception as exc:
            _log(f"save_failed error={type(exc).__name__}")
            return web.json_response({"error": "保存供应商失败，请查看后台日志"}, status=500)

    @routes.post("/yeban/moss-bbox-standalone/providers/delete")
    async def providers_delete(request: Any) -> Any:
        try:
            payload = await _json_payload(request)
            existed = await asyncio.to_thread(delete_profile, str(payload.get("id") or ""))
            return web.json_response({"ok": True, "deleted": existed})
        except ProviderManagerError as exc:
            return web.json_response({"error": str(exc)}, status=400)

    @routes.post("/yeban/moss-bbox-standalone/providers/credential/delete")
    async def credential_delete(request: Any) -> Any:
        try:
            payload = await _json_payload(request)
            existed = await asyncio.to_thread(delete_credential, str(payload.get("id") or ""))
            return web.json_response({"ok": True, "deleted": existed})
        except ProviderManagerError as exc:
            return web.json_response({"error": str(exc)}, status=400)

    @routes.post("/yeban/moss-bbox-standalone/providers/models")
    async def provider_models(request: Any) -> Any:
        try:
            payload = await _json_payload(request)
            result = await asyncio.to_thread(fetch_models, str(payload.get("id") or ""), int(payload.get("timeout", 15)))
            return web.json_response(result)
        except ProviderManagerError as exc:
            return web.json_response({"error": str(exc)}, status=400)
        except Exception as exc:
            _log(f"models_failed error={type(exc).__name__}")
            return web.json_response({"error": "连接测试失败，请查看后台日志"}, status=500)


_register_routes()


__all__ = [
    "YebanCodexProviderManager",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "ProviderManagerError",
    "list_profiles",
    "save_profile",
    "delete_profile",
    "delete_credential",
    "provider_runtime",
    "fetch_models",
]


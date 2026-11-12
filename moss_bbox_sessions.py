"""Bounded persistent backend conversations for the standalone MOSS BBox node."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Mapping

_LOCK = threading.RLock()
_SESSIONS: dict[str, dict[str, Any]] = {}
_STATUS: dict[str, str] = {}
_ACTIVE: set[str] = set()
_LOADED = False
_MAX_TURNS = 8
_MAX_TOKENS = 60000
_MAX_AGE = 30 * 86400


def _cache_path() -> Path | None:
    try:
        import folder_paths

        root = folder_paths.get_user_directory()
        return Path(root) / "Yeban_MOSS_BBox" / "conversations.json" if root else None
    except (ImportError, AttributeError, OSError, TypeError):
        return None


def _load() -> None:
    global _LOADED
    if _LOADED:
        return
    _LOADED = True
    path = _cache_path()
    if path is None or not path.is_file():
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("conversations") if isinstance(payload, Mapping) else None
        if not isinstance(rows, Mapping):
            return
        now = time.time()
        for key, value in rows.items():
            if not isinstance(key, str) or not isinstance(value, Mapping):
                continue
            session_id = str(value.get("session_id") or "")
            signature = str(value.get("signature") or "")
            updated = float(value.get("updated_at") or 0)
            if not session_id or len(session_id) > 2048 or not re.fullmatch(r"[0-9a-f]{64}", signature):
                continue
            if now - updated > _MAX_AGE:
                continue
            _SESSIONS[key] = {
                "session_id": session_id,
                "signature": signature,
                "turns": max(0, int(value.get("turns") or 0)),
                "estimated_tokens": max(0, int(value.get("estimated_tokens") or 0)),
                "generation": max(1, int(value.get("generation") or 1)),
                "updated_at": updated,
                "restored": True,
            }
    except (OSError, UnicodeError, ValueError, TypeError):
        return


def _save() -> None:
    path = _cache_path()
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        now = time.time()
        rows = {
            key: {
                "session_id": str(value.get("session_id") or ""),
                "signature": str(value.get("signature") or ""),
                "turns": int(value.get("turns") or 0),
                "estimated_tokens": int(value.get("estimated_tokens") or 0),
                "generation": int(value.get("generation") or 1),
                "updated_at": now,
            }
            for key, value in list(_SESSIONS.items())[-128:]
            if value.get("session_id") and re.fullmatch(r"[0-9a-f]{64}", str(value.get("signature") or ""))
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps({"version": 1, "conversations": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    except (OSError, TypeError, ValueError):
        return


def _key(backend: str, node_id: Any, model: str) -> str:
    return f"{backend}:{str(node_id or 'unknown')}:{model}:moss_bbox_layout"


def _stable_cwd(key: str) -> str:
    try:
        import folder_paths

        user_dir = Path(folder_paths.get_user_directory())
    except (ImportError, AttributeError, OSError, TypeError):
        return ""
    path = user_dir / "Yeban_MOSS_BBox" / "grok_sessions" / hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
    try:
        path.mkdir(parents=True, exist_ok=True)
        return str(path)
    except OSError:
        return ""


def prepare(
    *, backend: str, node_id: Any, model: str, signature: str,
    prompt: str, image_png: bytes, target_word_count: int, forced_session_id: str = "",
) -> dict[str, Any]:
    """Return the persistent session context; layout-only values are excluded by caller."""
    key = _key(backend, node_id, model)
    image_tokens = 256
    try:
        from PIL import Image
        import io

        with Image.open(io.BytesIO(image_png)) as image:
            image_tokens = max(256, math.ceil(image.width * image.height / 576))
    except Exception:
        image_tokens = max(256, len(image_png) // 256)
    next_tokens = math.ceil(len(prompt) / 3.5) + image_tokens + max(512, round(max(200, target_word_count) * 1.45))
    status = ""
    with _LOCK:
        _load()
        state = _SESSIONS.get(key)
        reset_reason = "new" if not state else ""
        if state and state.get("signature") != signature:
            reset_reason = "settings_changed"
        elif state and int(state.get("turns", 0)) >= _MAX_TURNS:
            reset_reason = "turn_limit"
        elif state and int(state.get("estimated_tokens", 0)) + next_tokens > _MAX_TOKENS:
            reset_reason = "context_limit"
        elif state and not state.get("session_id"):
            reset_reason = "missing_session"
        if reset_reason:
            state = {
                "session_id": (forced_session_id or ("mossbbox-" + uuid.uuid4().hex if backend == "dsh_official" else "")), "signature": signature, "turns": 0,
                "estimated_tokens": 0, "generation": int((state or {}).get("generation", 0)) + 1,
                "updated_at": time.time(),
            }
            _SESSIONS[key] = state
            status = "新建会话" if reset_reason in {"new", "missing_session"} else (
                "设定已变化，新建会话" if reset_reason == "settings_changed" else "上下文达到上限，新建会话"
            )
        else:
            status = "已恢复上次会话" if state.pop("restored", False) else "继续使用会话"
            if forced_session_id:
                state["session_id"] = forced_session_id
        context = {
            "key": key,
            "state": state,
            "session_id": str(state.get("session_id") or ""),
            "cwd": _stable_cwd(key) if backend == "local_grok" else "",
            "status": status,
            "reset": bool(reset_reason),
            "continuation": not bool(reset_reason),
            "estimated_tokens": next_tokens,
            "full_prompt": prompt,
        }
        _STATUS[key] = status
    return context


def record(context: Mapping[str, Any] | None, result: Any) -> None:
    if not isinstance(context, Mapping):
        return
    state = context.get("state")
    if not isinstance(state, dict):
        return
    with _LOCK:
        if context.get("session_id"):
            state["session_id"] = str(context["session_id"])
        state["turns"] = int(state.get("turns", 0)) + 1
        state["estimated_tokens"] = int(state.get("estimated_tokens", 0)) + int(context.get("estimated_tokens", 0)) + math.ceil(len(str(result or "")) / 3.5)
        state["updated_at"] = time.time()
        state.pop("restored", None)
        _STATUS[str(context.get("key") or "")] = str(context.get("status") or "会话已使用")
        _save()


def status(backend: str, node_id: Any, model: str) -> dict[str, Any]:
    key = _key(backend, node_id, model)
    with _LOCK:
        _load()
        value = _SESSIONS.get(key)
        label = _STATUS.get(key) or ("可继续上次会话" if value and value.get("session_id") else "尚未运行")
        return {"status": label, "has_session": bool(value and value.get("session_id")), "turns": int((value or {}).get("turns", 0))}


def reset(backend: str, node_id: Any, model: str) -> dict[str, Any]:
    key = _key(backend, node_id, model)
    with _LOCK:
        if key in _ACTIVE:
            raise RuntimeError("节点正在运行，请完成后再清除会话")
        _load()
        existed = _SESSIONS.pop(key, None) is not None
        _STATUS[key] = "已清除会话，下次运行将新建"
        _save()
    return {"status": _STATUS[key], "had_session": existed}


def mark_active(backend: str, node_id: Any, model: str, active: bool) -> None:
    key = _key(backend, node_id, model)
    with _LOCK:
        if active:
            _ACTIVE.add(key)
        else:
            _ACTIVE.discard(key)


def register_routes() -> None:
    try:
        from aiohttp import web
        from server import PromptServer

        routes = PromptServer.instance.routes
    except (ImportError, AttributeError):
        return

    @routes.get("/yeban/moss-bbox-standalone/session/status")
    async def get_session_status(request: Any) -> Any:
        node_id = str(request.query.get("node") or "")
        backend = str(request.query.get("backend") or "")
        model = str(request.query.get("model") or "")
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", node_id) or backend not in {"local_grok", "dsh_official"}:
            return web.json_response({"error": "invalid session key"}, status=400)
        return web.json_response(status(backend, node_id, model))

    @routes.post("/yeban/moss-bbox-standalone/session/reset")
    async def reset_session(request: Any) -> Any:
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"error": "invalid request"}, status=400)
        if not isinstance(payload, Mapping):
            return web.json_response({"error": "invalid request"}, status=400)
        node_id = str(payload.get("node") or "")
        backend = str(payload.get("backend") or "")
        model = str(payload.get("model") or "")
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", node_id) or backend not in {"local_grok", "dsh_official"}:
            return web.json_response({"error": "invalid session key"}, status=400)
        try:
            return web.json_response(reset(backend, node_id, model))
        except RuntimeError as error:
            return web.json_response({"error": str(error)}, status=409)

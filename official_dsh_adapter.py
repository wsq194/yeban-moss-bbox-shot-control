"""Run a director turn through the official DeepSeek Harness SDK profile."""

import base64
import json
import os
from pathlib import Path
import queue
import re
import subprocess
import tempfile
import threading
import time
import uuid


class OfficialDshError(RuntimeError):
    pass


def official_home():
    return Path(os.environ.get("DSH_OFFICIAL_HOME") or Path.home() / ".dsh")


def _desktop_profile_rows():
    """Read only model-provider rows from the official Desktop profile patch."""
    patch = official_home() / "profiles" / "desktop" / "cordis.patch.yml"
    if not patch.is_file():
        raise OfficialDshError("DSH_OFFICIAL_PROFILE_UNAVAILABLE")
    try:
        import yaml
        lines = patch.read_text(encoding="utf-8-sig").splitlines(keepends=True)
        starts = [(index, line.split(":", 1)[1].strip()) for index, line in enumerate(lines) if line.startswith("- id:")]
        wanted = {"agent-default-model", "llm-deepseek", "llm-pi-ai"}
        rows = {}
        for position, (start, row_id) in enumerate(starts):
            if row_id not in wanted:
                continue
            end = starts[position + 1][0] if position + 1 < len(starts) else len(lines)
            parsed = yaml.safe_load("".join(lines[start:end]))
            if isinstance(parsed, list) and len(parsed) == 1 and isinstance(parsed[0], dict):
                rows[row_id] = parsed[0]
        return rows
    except OfficialDshError:
        raise
    except Exception as error:
        raise OfficialDshError("DSH_OFFICIAL_PROFILE_INVALID") from error


def _public_model(model):
    if isinstance(model, str):
        model_id = model.strip()
        name = model_id
    elif isinstance(model, dict):
        model_id = str(model.get("id") or "").strip()
        name = str(model.get("name") or model_id).strip()
    else:
        return None
    if not model_id or len(model_id) > 200 or re.search(r"[\r\n\x00]", model_id):
        return None
    return {"id": model_id, "name": name[:200] or model_id}


def model_catalog():
    """Return public provider/model IDs configured in the official Desktop profile."""
    rows = _desktop_profile_rows()
    default = rows.get("agent-default-model", {}).get("config", {})
    account_model = _public_model(default.get("model")) if default.get("provider") == "deepseek-account" else None
    groups = [{
        "id": "deepseek-account",
        "name": "DeepSeek 账号",
        "models": [account_model or {"id": "deepseek-flash", "name": "deepseek-flash"}],
    }]
    deepseek = rows.get("llm-deepseek", {}).get("config", {})
    deepseek_models = [item for item in (_public_model(model) for model in deepseek.get("models", [])) if item]
    if deepseek_models:
        groups.append({"id": "deepseek-official", "name": "DeepSeek Official", "models": deepseek_models})

    pi_ai = rows.get("llm-pi-ai", {}).get("config", {})
    for provider_id, provider in (pi_ai.get("providers") or {}).items():
        if provider_id in {"deepseek-account", "deepseek-official"} or not isinstance(provider, dict) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,100}", str(provider_id)):
            continue
        models = [item for item in (_public_model(model) for model in provider.get("models", [])) if item]
        if models:
            groups.append({"id": str(provider_id), "name": str(provider.get("displayName") or provider_id)[:200], "models": models})

    return {
        "groups": groups,
        "default": {"provider": default.get("provider"), "model": default.get("model")},
        "failed_providers": [],
    }


def _without_secrets(value):
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            normalized = re.sub(r"[^a-z]", "", str(key).lower())
            if normalized in {"apikey", "token", "secret", "password", "credential", "credentials"}:
                continue
            result[key] = _without_secrets(item)
        return result
    if isinstance(value, list):
        return [_without_secrets(item) for item in value]
    return value


def _provider_sdk_patch(provider_id):
    """Overlay only the selected public LLM provider config onto the SDK profile."""
    rows = _desktop_profile_rows()
    if provider_id == "deepseek-official":
        row = rows.get("llm-deepseek")
    else:
        pi_ai = rows.get("llm-pi-ai", {}).get("config", {})
        row = rows.get("llm-pi-ai") if provider_id in (pi_ai.get("providers") or {}) else None
    if not row:
        return None
    try:
        import yaml
        patch = yaml.safe_dump([_without_secrets(row)], allow_unicode=True, sort_keys=False)
        handle = tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".yml", prefix="storyboard-dsh-sdk-", delete=False)
        try:
            handle.write(patch)
            handle.close()
            return Path(handle.name)
        except Exception:
            handle.close()
            Path(handle.name).unlink(missing_ok=True)
            raise
    except OfficialDshError:
        raise
    except Exception as error:
        raise OfficialDshError("DSH_OFFICIAL_PROFILE_INVALID") from error


def runtime_paths():
    installed = Path(os.environ.get("DSH_OFFICIAL_INSTALL", Path.home() / "AppData/Local/Programs/DeepSeek Harness"))
    executable = installed / "DeepSeek Harness.exe"
    entry = installed / "resources/app.asar/dsh/node_modules/@deepseek-ai/dsh/lib/bin.js"
    if not executable.is_file() or not (installed / "resources/app.asar").is_file():
        raise OfficialDshError("DSH_OFFICIAL_NOT_INSTALLED")
    return executable, entry


def run(prompt, images, config, schema, cwd, timeout=240):
    executable, entry = runtime_paths()
    env = os.environ.copy()
    env["ELECTRON_RUN_AS_NODE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["DSH_HOME"] = str(official_home())
    branch_id = str(config.get("dsh_branch_session") or "").strip()
    if branch_id and not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", branch_id):
        raise OfficialDshError("DSH_OFFICIAL_BRANCH_INVALID")
    patch_path = _provider_sdk_patch(config.get("dsh_provider") or "deepseek-account")
    env.pop("STORYBOARD_DSH_BRANCH_SESSION", None)
    command = [str(executable)]
    if branch_id:
        env["STORYBOARD_DSH_BRANCH_SESSION"] = branch_id
        command.extend(["--import", Path(__file__).with_name("director_dsh_branch.mjs").resolve().as_uri()])
    command.extend([str(entry), "--profile", "sdk"])
    if patch_path:
        command.extend(["--patch", str(patch_path)])
    try:
        child = subprocess.Popen(
            command,
            cwd=str(cwd), env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, encoding="utf-8", bufsize=1,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except Exception:
        if patch_path:
            patch_path.unlink(missing_ok=True)
        raise
    frames = queue.Queue()
    threading.Thread(target=lambda: [frames.put(line) for line in child.stdout], daemon=True).start()
    deadline = time.monotonic() + timeout
    session_id = branch_id or "storyboard-" + uuid.uuid4().hex

    def send(number, method, params):
        child.stdin.write(json.dumps({"jsonrpc": "2.0", "id": number, "method": method, "params": params}, ensure_ascii=False) + "\n")
        child.stdin.flush()

    def next_frame():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("DSH_OFFICIAL_TIMEOUT")
        try:
            line = frames.get(timeout=min(remaining, 1)).strip()
        except queue.Empty:
            if child.poll() is not None:
                raise OfficialDshError("DSH_OFFICIAL_EXITED")
            return None
        if not line or re.match(r"^\[[A-Za-z0-9_.:-]+\](?:\s|$)", line):
            return None
        try:
            frame = json.loads(line)
        except ValueError as exc:
            raise OfficialDshError("DSH_OFFICIAL_INVALID_RPC_FRAME") from exc
        return frame if isinstance(frame, dict) else None

    try:
        send(1, "initialize", {"cwd": str(cwd), "provider": config.get("dsh_provider") or "deepseek-account", "model": config.get("model") or "deepseek-flash"})
        while True:
            frame = next_frame()
            if frame and frame.get("id") == 1:
                if "error" in frame:
                    raise OfficialDshError("DSH_OFFICIAL_MODEL_UNAVAILABLE")
                break
        blocks = [{"type": "text", "text": prompt + "\nReturn ONLY JSON matching this schema:\n" + json.dumps(schema, ensure_ascii=False)}]
        blocks.extend({"type": "image", "mimeType": "image/png", "data": base64.b64encode(image).decode("ascii")} for image in images)
        send(2, "session/prompt", {"sessionId": session_id, "contentBlocks": blocks})
        accepted = False
        last_text = ""
        turn_error = False
        while True:
            frame = next_frame()
            if not frame:
                continue
            if frame.get("id") == 2:
                if "error" in frame:
                    raise OfficialDshError("DSH_OFFICIAL_PROMPT_REJECTED")
                accepted = True
            elif frame.get("method") == "session.event" and frame.get("params", {}).get("sessionId") == session_id:
                event = frame["params"].get("event") or {}
                if event.get("type") == "assistant/message":
                    blocks = ((event.get("data") or {}).get("message") or {}).get("content") or []
                    candidate = "".join(str(block.get("text") or "") for block in blocks if isinstance(block, dict) and block.get("type") == "text").strip()
                    if candidate:
                        last_text = candidate
                elif event.get("type") == "turn/end":
                    reason = (event.get("data") or {}).get("reason") or {}
                    if isinstance(reason, dict) and reason.get("kind") == "error":
                        code = ((reason.get("error") or {}).get("code") or "").upper()
                        turn_error = "DSH_OFFICIAL_SIGN_IN_REQUIRED" if code == "ACCOUNT_SIGN_IN_REQUIRED" else "DSH_OFFICIAL_AUTH_REQUIRED" if code in ("AUTH", "MISSING_CREDENTIAL") else "DSH_OFFICIAL_MODEL_FAILED"
            elif frame.get("method") == "session.status" and frame.get("params", {}).get("sessionId") == session_id and frame["params"].get("status") == "idle" and accepted:
                if turn_error:
                    raise OfficialDshError(turn_error)
                if not last_text:
                    raise OfficialDshError("DSH_OFFICIAL_EMPTY_RESPONSE")
                return last_text
    finally:
        if child.poll() is None:
            try:
                send(3, "shutdown", {})
                child.wait(timeout=4)
            except (OSError, ValueError, subprocess.TimeoutExpired):
                if child.poll() is None:
                    child.terminate()
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=5)
        if patch_path:
            patch_path.unlink(missing_ok=True)

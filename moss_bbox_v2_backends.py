"""Local SAM detection and backend routing for MOSS BBox V2."""

import hashlib
import json
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path

from . import nodes

DETECTOR = "sam3.1_multiplex_fp16.safetensors"
DETECTION_SCHEMA = "moss-sam31-v2.2"


def _find_detector_checkpoint(folder_paths):
    """Find the required SAM checkpoint in ComfyUI's registered checkpoint roots."""
    checkpoint = folder_paths.get_full_path("checkpoints", DETECTOR)
    if checkpoint:
        return checkpoint

    # get_filename_list recursively searches every configured checkpoint root and
    # returns paths relative to those roots. This also supports nested folders and
    # extra_model_paths.yaml entries without scanning unrelated disks/directories.
    for filename in folder_paths.get_filename_list("checkpoints"):
        if Path(filename).name.casefold() != DETECTOR.casefold():
            continue
        checkpoint = folder_paths.get_full_path("checkpoints", filename)
        if checkpoint:
            return checkpoint
    return None


@lru_cache(maxsize=1)
def _load_detector(path, modified, size):
    import comfy.sd
    import folder_paths

    return comfy.sd.load_checkpoint_guess_config(
        path, output_vae=False, output_clip=True,
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
    )[:2]


def _overlap(a, b):
    w = max(0, min(a["x"] + a["width"], b["x"] + b["width"]) - max(a["x"], b["x"]))
    h = max(0, min(a["y"] + a["height"], b["y"] + b["height"]) - max(a["y"], b["y"]))
    area = w * h
    union = a["width"] * a["height"] + b["width"] * b["height"] - area
    return area / union if union > 0 else 0


def detect_boxes(frame, use_cache):
    import folder_paths
    import torch
    from comfy_extras.nodes_sam3 import SAM3_Detect
    from nodes import CLIPTextEncode

    checkpoint = _find_detector_checkpoint(folder_paths)
    if not checkpoint:
        raise nodes.PromptWriterError(
            f"自动检测没有在 ComfyUI 已注册的 checkpoints 模型目录（含子文件夹）中找到 SAM 3.1 权重：{DETECTOR}。"
            "请将该文件放入任一 checkpoints 搜索目录，或把自定义模型目录注册为 checkpoints 路径。"
        )
    stat = Path(checkpoint).stat()
    identity = (checkpoint, stat.st_mtime_ns, stat.st_size)
    digest = hashlib.sha256(frame.tobytes())
    digest.update(repr((frame.shape, identity, DETECTION_SCHEMA)).encode())
    key = digest.hexdigest()
    cached = nodes._read_cache(key, schema_version=DETECTION_SCHEMA) if use_cache else None
    if cached:
        try:
            people, props = json.loads(cached["prompt_text"])
            return people, props
        except (TypeError, ValueError):
            pass

    model, clip = _load_detector(*identity)
    image = torch.from_numpy(frame.copy()).unsqueeze(0).float() / 255

    def detect(query, threshold):
        conditioning, = CLIPTextEncode().encode(clip, query)
        return SAM3_Detect.execute(model, image, conditioning=conditioning,
                                   threshold=threshold, refine_iterations=0)[1][0]

    candidates = detect("person:8", 0.35)
    best_score = max((box["score"] for box in candidates), default=0)
    frame_area = frame.shape[0] * frame.shape[1]
    people = [box for box in candidates
              if box["score"] >= best_score * 0.72
              and box["width"] * box["height"] >= frame_area * 0.008]
    if not people and candidates:
        people = [max(candidates, key=lambda box: box["score"])]
    props = detect("chair:2, sofa:2, bench:2, ottoman:2, table:2, handbag:2, railing, fence", 0.55)
    selected = []
    for prop in sorted(props, key=lambda box: box.get("score", 0), reverse=True):
        if prop["width"] * prop["height"] < frame_area * 0.012 or any(
            _overlap(prop, other) >= 0.60 for other in [*people, *selected]
        ):
            continue
        selected.append(prop)
        if len(selected) == 5:
            break
    if use_cache and people:
        nodes._write_cache(key, {"prompt_text": json.dumps([people, selected])},
                           schema_version=DETECTION_SCHEMA)
    return people, selected


def resolve_backend(backend, model, provider_config, dsh_config):
    base_url, api_key, identity = "", None, None
    if backend == "openai_compatible":
        profile = nodes._provider_runtime(provider_config)
        if not profile or profile["transport"] != "openai_compatible":
            raise nodes.PromptWriterError("请接入启用的兼容 API 供应商配置。")
        model = model or next((profile["models"][key] for key in ("custom", "luna", "sol", "terra")
                               if key in profile["models"]), "")
        if not model:
            raise nodes.PromptWriterError("请选择支持图片输入的 API 模型。")
        base_url, api_key = profile["base_url"], profile["api_key"]
        identity = (base_url, profile["profile_id"])
    elif backend == "dsh_official":
        if not isinstance(dsh_config, Mapping):
            raise nodes.PromptWriterError("请接入「夜半 官方 DSH 配置」。")
        model = dsh_config.get("model") or "deepseek-v4.1-flash"
        identity = (dsh_config.get("dsh_provider"), dsh_config.get("dsh_branch_session"))
    elif backend == "local_grok":
        model = model or "grok-4.7"
        if not model.casefold().startswith("grok"):
            raise nodes.PromptWriterError("请输入有效的 Grok 模型名称。")
    elif backend == "local_codex":
        model = model or nodes.DEFAULT_MODEL
    else:
        raise nodes.PromptWriterError("不支持的 MOSS V2 后端。")
    return model, base_url, api_key, identity

# Yeban MOSS BBox 景别控制 V1

Windows-only ComfyUI custom node package for MOSS BBox scene-scale and composition control.

> **仅面向 Windows 用户。** API 密钥由 Windows DPAPI 加密保存；本机 Grok/Codex CLI 和官方 DSH 桌面后端也按 Windows 环境设计。本插件不会下载或捆绑第三方模型、CLI 或密钥。

## Install

Install through ComfyUI Manager/Registry after publication, or clone this repository into `ComfyUI/custom_nodes/Yeban_MOSS_BBox` and restart ComfyUI. The primary node appears as **Yeban MOSS BBox 景别控制 V1** under **Yeban/MOSS BBox**. Its internal node type remains `YebanMossBBoxStandaloneV3` to keep existing workflows compatible. It can coexist with `Yeban_Codex_Prompt_Writer`; the standalone node IDs, API routes, and saved provider profiles are namespaced separately.

Chinese node labels are included in `locales/zh/nodeDefs.json` for ComfyUI's native Chinese locale. A small frontend compatibility layer also reapplies these labels when the AIGODLIKE translation extension resets untranslated node labels to their internal IDs.

## What it does

The node detects people and selected foreground props locally with ComfyUI's SAM 3.1 detector, then calls one selected vision backend to describe the reference image. It compiles the returned visual facts into a MOSS BBox layout prompt and also returns a standalone smart reverse prompt. Switching shot size and composition position reuses cached image analysis when caching is enabled; it does not require another vision call.

Shot sizes are **沿用原图**, **近景** (about 90% of frame height), **中景** (about 58%), and **远景** (about 34%). The nine-position selector controls the person group placement for non-reference shot sizes. Values are target layout guidance; the image model can still vary from the requested framing.

## 效果示意

以下是 Yeban MOSS BBox 景别控制 V1 的真实采样对照，使用同一参考图、固定 MOSS 模型和 seed 切换景别或构图位置。它们展示提示词引导的实际变化，不代表人物姿势、身份或服装能被像素级锁定；不同模型、seed 和工作流会有差异。

### 景别

![便利店场景：沿用原图、近景、中景、远景对比](examples/shot-sizes-store.jpg)

![雨夜街景：沿用原图、近景、中景、远景对比](examples/shot-sizes-rainy-street.jpg)

![夜景屋顶：近景、中景、远景跨图复验](examples/shot-sizes-rooftop.jpg)

### 九宫格构图位置

![左中、居中、右中位置对比](examples/composition-horizontal.jpg)

![上中、居中、下中位置对比](examples/composition-vertical.jpg)

## 示例工作流

发布包附带完整的 **Yeban MOSS BBox 景别控制 V1 示例**：
[`examples/Yeban-MOSS-BBox-Shot-Control-V1.json`](examples/Yeban-MOSS-BBox-Shot-Control-V1.json)。
它演示如何把布局提示词接入 MOSS Image 正向提示词，并把节点输出的宽、高接入空 Latent。
打开后请在“加载图像”节点上传自己的参考图。示例不包含个人参考图、SAM 权重、MOSS 模型或 API 凭据。

## Requirements

- Windows and a ComfyUI build that provides `comfy_api.latest` and `comfy_extras.nodes_sam3`.
- The SAM checkpoint `sam3.1_multiplex_fp16.safetensors`. The node searches ComfyUI's registered `checkpoints` model paths recursively, including nested folders and paths added through `extra_model_paths.yaml`; it does not scan unrelated drives. The recommended location is `ComfyUI/models/checkpoints/`. The official Comfy-Org Hugging Face file is about 1.75 GB: [download the checkpoint](https://huggingface.co/Comfy-Org/sam3.1/resolve/main/checkpoints/sam3.1_multiplex_fp16.safetensors) ([model page and license](https://huggingface.co/Comfy-Org/sam3.1)). The model is not bundled with this plugin. The published file's SHA-256 is `9ba99c92703c2e8b4f47de2d34a539bb8e18923049e238b780d70dbe6368eb03`; review and follow the SAM License linked on the model page.
- One vision backend configured for use: local Grok CLI (default), local Codex CLI, official DSH desktop configuration, an OpenAI-compatible provider profile, or a connected local image-capable Llama model.
- For the local Llama option only, the existing `ComfyUI-SkillReversePrompt` integration must be installed. This integration is optional for all other backends.
- Official DSH requires the official DeepSeek Harness desktop application. Provider API credentials and provider profiles are stored by this package separately from the legacy prompt-writer plugin.

OpenAI-compatible API keys are encrypted with Windows DPAPI and are local to the Windows user account that saved them; they are not portable to another computer or Windows account.

No model weights, API credentials, or third-party CLI executables are bundled.

## Included helper nodes

- **夜半 MOSS BBox 供应商管理器** stores OpenAI-compatible provider profiles and encrypted API credentials.
- **夜半 MOSS BBox 官方 DSH 配置** reads model and conversation-branch options from the official DSH desktop application.

Connect the relevant helper output to the matching optional input on the main node when using those backends.

## Package notes

The standalone bundle includes a namespaced compatibility core (`nodes.py`) from the existing node package so it does not import or require `Yeban_Codex_Prompt_Writer`. Only the three MOSS BBox helper/main nodes are registered from this package. Shared ComfyUI runtime components remain supplied by ComfyUI itself.

The package is published under the MIT License. Third-party models, SAM weights, provider services, and CLI applications keep their own licenses and terms.

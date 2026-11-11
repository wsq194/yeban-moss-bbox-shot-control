# Yeban MOSS BBox 景别控制 V1

通过参考图生成 MOSS BBox 布局提示词，并用“景别”和“构图位置”快速调整人物在画面中的大小与位置。另有一条独立的智能反推提示词，可供普通正向提示词流程使用。

**仅支持 Windows。** 节点本身不包含 SAM、MOSS 模型或视觉模型，也不会替你安装或登录后端客户端。

## 快速开始

### 1. 安装节点

在 ComfyUI Manager 搜索 **Yeban MOSS BBox 景别控制 V1** 并安装；如果 Registry 中暂时还搜不到，可从本仓库下载 ZIP，解压到 ComfyUI 的 `custom_nodes` 文件夹。安装后重启 ComfyUI。

### 2. 准备 SAM 检测模型

下载文件 **`sam3.1_multiplex_fp16.safetensors`**，放入 `ComfyUI/models/checkpoints/`，或放入 ComfyUI 已登记的其他 `checkpoints` 搜索目录。节点会自动查找这个文件名，包括已登记目录的子文件夹。

- [下载 SAM 3.1 权重（约 1.75 GB）](https://huggingface.co/Comfy-Org/sam3.1/resolve/main/checkpoints/sam3.1_multiplex_fp16.safetensors)
- [查看模型说明与许可](https://huggingface.co/Comfy-Org/sam3.1)
- SHA-256：`9ba99c92703c2e8b4f47de2d34a539bb8e18923049e238b780d70dbe6368eb03`
- 百度网盘备用下载：https://pan.baidu.com/s/1EMgLhpDgDjHHvhitZGHGjA?pwd=rdtd

权重不随插件分发。下载和使用前请阅读模型页面的 SAM License。

### 3. 打开示例工作流并上传参考图

打开 [示例工作流](examples/Yeban-MOSS-BBox-Shot-Control-V1.json)，在“加载图像”节点选择自己的图片。示例不包含参考图片、SAM 权重、MOSS 模型或后端凭据。

### 4. 配置后端并运行

在主节点选择已经安装并登录的视觉后端，确认模型选项可用，然后点击运行。第一次分析会调用视觉后端；启用缓存后，只改景别或构图位置会复用图像分析结果。

## 节点怎么接

示例工作流已经接好主要输出。手动搭建时按下面连接：

| 节点输出 | 连接到 | 用途 |
| --- | --- | --- |
| **MOSS 布局提示词** | MOSS Image 的布局提示词输入 | 按参考图生成布局与场景描述 |
| **采样宽度** | 空 Latent 的宽度 | 让空 Latent 使用节点计算出的宽度 |
| **采样高度** | 空 Latent 的高度 | 让空 Latent 使用节点计算出的高度 |
| **智能反推提示词** | 普通正向提示词流程（可选） | 单独描述原图事实，供其他提示词或修复流程使用 |

把参考图接入节点左侧的**参考图**输入。使用 DSH 或兼容 API 时，还要连接对应配置节点的输出。

## 景别和构图位置

景别按钮控制人物在目标画面中的大致高度；百分比是构图目标，模型生成时可能有偏差。

| 选项 | 人物目标高度 |
| --- | ---: |
| 沿用原图 | 尽量保留参考图中的人物大小和位置 |
| 近景 | 约为画面高度的 90% |
| 中景 | 约为画面高度的 58% |
| 远景 | 约为画面高度的 34% |

九宫格用于调整人物组在画面中的位置。选择“沿用原图”时位置控制不生效；切换到近景、中景或远景后即可使用。主体安全边距和人物完整度会限制移动范围。

“采样画幅比例”和“采样长边”决定节点输出的采样宽高，可直接连接到空 Latent。把长边设为 `0` 时沿用原图尺寸。

## 效果示意

下面是不同景别和构图位置的实际生成对照。它们展示提示词带来的构图趋势，不是像素级锁定；人物姿势、外观和最终占比仍会受模型影响。

### 景别变化

![便利店场景：沿用原图、近景、中景、远景](examples/shot-sizes-store.jpg)

![雨夜街景：沿用原图、近景、中景、远景](examples/shot-sizes-rainy-street.jpg)

![夜景屋顶：近景、中景、远景跨图示意](examples/shot-sizes-rooftop.jpg)

### 构图位置变化

![左中、居中、右中位置对比](examples/composition-horizontal.jpg)

![上中、居中、下中位置对比](examples/composition-vertical.jpg)

## 后端设置

每位用户需要在自己的 Windows 账户下安装并登录所选服务。节点不会替你创建账号、转移订阅或安装第三方 CLI。

### 本机 Grok CLI

1. 按 [Grok Build 官方 Windows 说明](https://docs.x.ai/build/overview)安装并登录。
2. 在 PowerShell 确认可用：

   ```powershell
   grok --version
   grok models
   ```

3. 节点“后端”选择**本机 Grok CLI**，点击**刷新模型选项**读取模型。

### 本机 Codex CLI

1. 按 [OpenAI Codex CLI 官方说明](https://developers.openai.com/codex/cli)安装并登录。
2. 在 PowerShell 确认可用：

   ```powershell
   codex --version
   ```

3. 节点“后端”选择**本机 Codex CLI**，点击**刷新模型选项**读取模型和推理强度。

本机 CLI 指由节点在本地启动客户端；图像仍会由客户端连接的服务处理，并按对应账户的服务规则计量，不代表离线推理。

### 官方 DSH 桌面版

1. 安装并登录**官方 DeepSeek Harness 桌面版**，在其中配置可用的供应商和模型。
2. 在工作流中添加**夜半 MOSS BBox 官方 DSH 配置**节点，刷新并选择供应商、模型，以及可选的对话分支。
3. 将该节点的输出接到主节点的**DSH 配置**输入；主节点后端选择**官方 DSH**。

这一路径读取官方桌面版的账号与配置，不需要在工作流中重复填写 API 密钥。

### OpenAI 兼容 API

1. 添加**夜半 MOSS BBox 供应商管理器**节点，填写兼容接口地址、模型和自己的 API 密钥。
2. 将供应商管理器输出连接到主节点的**供应商配置**输入。
3. 主节点后端选择**供应商 API**，刷新模型选项。

API 密钥使用 Windows DPAPI 加密，绑定当前 Windows 用户账户；更换电脑或 Windows 账户后需要重新配置。

**当前 V1 不提供本地 Llama 推理。**

## 常见问题

**模型下拉列表为空**

确认所选 CLI 或官方 DSH 已安装并登录，再点**刷新模型选项**。本机 CLI 不在 PATH 时，按后端对应方式设置可执行文件路径后重启 ComfyUI。

**节点提示找不到 SAM**

确认文件名完整为 `sam3.1_multiplex_fp16.safetensors`，并放在 ComfyUI 已登记的 `checkpoints` 搜索目录中。节点不会扫描未登记的任意磁盘路径。

**旧工作流节点仍显示已移除的输入口**

删除旧节点，从 **Yeban/MOSS BBox** 分类重新添加 **Yeban MOSS BBox 景别控制 V1**。开发期间节点类型已更新，旧实例不会自动迁移。

**换景别或位置后仍然很慢**

确认节点已启用缓存且缓存种子未改变。缓存会复用图像事实分析，但首次反推仍需调用选定的视觉后端。

## 兼容性与隐私

- 需要 Windows，以及提供 `comfy_api.latest` 和 `comfy_extras.nodes_sam3` 的 ComfyUI 版本。
- SAM 人物与道具检测在本机运行。参考图会发送到所选视觉后端分析；请依据对应服务的隐私政策选择后端。
- 插件不捆绑模型权重、API 密钥、CLI 或桌面客户端。
- 插件代码采用 MIT License。SAM 权重、MOSS 模型、供应商服务及客户端分别遵循其自身许可和服务条款。

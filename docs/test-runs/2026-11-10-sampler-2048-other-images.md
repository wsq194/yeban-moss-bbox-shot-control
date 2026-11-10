# MOSS BBox sampler test at 2048 long edge

Date: 2026-11-10 (Asia/Shanghai)
Purpose: Compare the packaged MOSS BBox Shot Control V1 workflow on three other references after reducing only the sampling long edge to 2048. The user requested the size reduction after the 3072 runs were unusually slow.

## Controlled settings

- Workflow graph: `Yeban-MOSS-BBox-Shot-Control-V1.json` (executed as an API copy; the packaged workflow file was not edited)
- UNET: `Moss/moss_image_sft_bf16.safetensors`
- CLIP: `moss/qwen3.5_9b_w4a8_gptq.safetensors`
- VAE: `flux2-vae.safetensors`
- MOSS BBox V3: aspect `3:2 <-> 2:3`, long edge `2048`, shot size `environmental`, backend `local_grok`, model `grok-4.5`, reasoning `default`, center composition, cache enabled, target English words `220`
- KSampler: fixed seed `438829726369282`, 25 steps, CFG 2.5, `euler`, `sgm_uniform`, denoise 1.0
- Derived output dimensions: 1360 × 2048 (portrait 2:3 ratio rounded to a supported multiple of 16)
- Before each run, called ComfyUI `/free` with `unload_models=true` and `free_memory=true`; checked the queue was clear before starting.

## Runs

| Ref | Source (SHA-256) | Prompt ID | Result | End-to-end |
|---|---|---|---|---|
| Exterior walking portrait | `小梦大半__爱在每一个角落_02.jpeg` — `1267a417184a2d2b3f2072dc3870f7df91266a8c37c71479a534ccc75b7fbaaa` | `a3b68a77-9406-414c-a76c-a01525a6e206` | Success, 1360×2048; scene fills edges with no white border. Architecture and full-body walking setup retained; clothes/face drifted. | 201 s (3:21) |
| Lakeside seated portrait | `小梦大半__花好还是我窈窕_05.jpeg` — `ac215b6986917a543d2c941fac41322d56d474f4aa4bb7d45edc40167eddcc81` | `a3cfaa20-d1f8-44ec-b02f-b83e3ef40bb8` | Success, 1360×2048; lake, pavilion/covered walkway and full-edge scene retained; seated pose broadly retained, clothing/identity details drifted. | 228 s (3:48) |
| Supermarket portrait | `小梦大半__升职啦！可以邀请前辈喝一杯吗_04.jpeg` — `e9ff729f25457d6f16374820adef9e60a5f3656cf91ed53bbe52e0605d7b6c95` | `2d4912e5-91c2-4781-b18a-ebdc6a19d15a` | Success, 1360×2048; supermarket shelves fill both sides and full-body framing is preserved; clothing/pose details drifted. | 648 s (10:48) |

Outputs:

- `apps/comfyui/ComfyUI/output/Yeban_MOSS_BBox_Test_20261110_2048/01_exterior_00001_.png`
- `apps/comfyui/ComfyUI/output/Yeban_MOSS_BBox_Test_20261110_2048/02_lakeside_00001_.png`
- `apps/comfyui/ComfyUI/output/Yeban_MOSS_BBox_Test_20261110_2048/03_supermarket_00001_.png`

## Timing evidence and interpretation

ComfyUI history `execution_start` and `execution_success` timestamps give the elapsed times above. The first two 2048 runs completed in roughly 3–4 minutes, while the third took 10:48. Earlier 3072 tests on the first two references took about 16 and 18 minutes end-to-end; the third 3072 supermarket attempt failed in `YebanMossBBoxStandaloneV3` with `Local Grok ACP session/prompt failed (-32603): Internal error` before sampling.

The current instrumentation does not separately time Grok reverse analysis, SAM work, KSampler, and VAE decode for these 2048 runs. Therefore, do not attribute all elapsed time to sampling or claim that reducing resolution directly speeds reverse analysis. The 3× timing spread at identical 2048 settings, along with the earlier ACP error, shows substantial local Grok/backend variability. Reducing from 3072 to 2048 reduces pixel area and can shorten KSampler/decode work, but does not remove the reverse-call latency. Visual acceptance was manual; this was not an automated similarity evaluation.

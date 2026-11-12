# Release validation record

## 0.1.1 candidate

- **Purpose:** check the updated standalone node package, local Llama backend interface, session-aware Grok/DSH path, English/Chinese documentation, and packaged example workflow before publication.
- **Source:** release repository files synchronized from the active Yeban MOSS BBox package. The example workflow contains no embedded image bytes or credentials; its Load Image node points to a local image filename and should be changed by the user.
- **Unit-test image:** synthetic 48×32 RGB array generated in memory.
- **Model/backend:** unit tests mock SAM detection, vision backends, cache, and layout compiler. No network model request or image sampling was part of this validation run.
- **Sampling settings:** not applicable to unit tests. The example workflow's KSampler is fixed at 8 steps, CFG 1, Euler, `sgm_uniform`, denoise 1; its width and height are driven by the MOSS node outputs.
- **Command:** `python_embeded\python.exe -m unittest discover -s tests -v` from the package repository.
- **Result:** 9 tests passed, including shot-size scale/position transforms, cache reuse and invalidation, BBox output, and presence of both local Llama model inputs.
- **Frontend:** `node --check` passed for all shipped JavaScript files.
- **Data files:** Chinese locale JSON and example workflow JSON parsed successfully. Workflow node type is `YebanMossBBoxStandaloneV3`; image bytes and credential-like fields were not embedded.
- **Package metadata:** version `0.1.1`; publisher `wsq194`; Windows-only classifier; no runtime pip installation or bundled model weights.
- **Unverified in this run:** a clean-profile ComfyUI install and fresh end-to-end image-generation run from the packaged workflow. Backend calls in unit tests are mocked.

## Notes for users

- Local Llama inference uses a connected `LLAMACPPMODEL` and the existing `ComfyUI-SkillReversePrompt` local vision bridge. The backend requires a vision-capable model/projector configuration.
- SAM weights, MOSS models, backend accounts, CLI applications, and API credentials are external prerequisites and are not included in the release.

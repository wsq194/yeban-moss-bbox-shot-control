# Standalone package validation

Last updated: 2026-11-10

## Package and code checks

- **Package source:** standalone release copy in this repository. The Python, frontend, locale, and node entrypoint files were SHA-256 compared against the active installed `Yeban_MOSS_BBox` source; all compared files matched.
- **Test input:** the unit tests create a synthetic 48×32 RGB image in memory; no user image or secret is included.
- **Model/workflow:** unit tests mock the vision backend, SAM detector, cache, and MOSS layout compiler. No inference model was invoked by these tests.
- **Sampling parameters:** not applicable to the unit tests.
- **Unit test command:** `python_embeded\python.exe -m unittest discover -s tests -v` using the user's ComfyUI embedded Python runtime.
- **Unit test result:** 8 tests passed. Coverage includes cached analysis across shot-size changes, cache invalidation, output behavior, approved shot targets, full-canvas bounds, reference framing, and character-first layout ordering.
- **Syntax:** embedded Python `compileall` passed; `node --check` passed for every shipped frontend `.js` file.
- **Metadata/assets:** `pyproject.toml`, Chinese locale JSON, and the example workflow JSON parsed successfully. Publisher ID and package ID match the intended `wsq194` release.
- **Workflow note:** repaired the example's corrupted Chinese MarkdownNote, re-parsed the JSON, and verified the private Baidu SAM link remains present.
- **Package import:** loading this standalone folder under an isolated package name registered exactly the three expected nodes. A first SAM probe used an unqualified submodule import and failed due to the isolated package alias; the corrected qualified import passed and resolved `sam3.1_multiplex_fp16.safetensors` from the registered ComfyUI checkpoint roots.
- **Registry source scan:** no `eval()`/`exec()` use, runtime pip installation, or bundled credentials found. `subprocess` is used only to invoke a user-selected local CLI / official DSH desktop bridge; README documents the local and API backend choices.

## User-visible evidence and remaining acceptance

- The gallery images in `examples/` are real V3 feature-validation outputs created before standalone extraction. They demonstrate shot-size and nine-position composition changes, but are not a fresh image-generation acceptance run from a clean install of this repository.
- The standalone node is already installed in the user's active ComfyUI, and the user confirmed that Chinese UI labels display correctly. The release copy itself has not yet been installed into a separate clean ComfyUI profile.
- A clean-profile install, ComfyUI restart, live `/object_info` check, SAM auto-discovery from the current runtime, and one packaged-workflow image run remain the final runtime acceptance items before Registry publication.
- No Registry release has been submitted. The initial GitHub publish workflow is manual-only (`workflow_dispatch`) and requires the repository secret `REGISTRY_ACCESS_TOKEN`.
- SAM checkpoint, vision backends, API credentials, MOSS model files, and CLI applications are external prerequisites and are not bundled. The checkpoint remains governed by its upstream SAM License.

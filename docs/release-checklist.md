# Release checklist

## Package

- [x] Separate Windows-only repository under the `wsq194` GitHub publisher.
- [x] Registry metadata with package ID, display name, semantic version, repository URL, Publisher ID, MIT license, and Windows classifier.
- [x] `.comfyignore` excludes test and development docs while keeping the example workflow and effect sheets.
- [x] Registry publish action is manual-only; pushing the initial package does not publish a Registry version.
- [x] Private Baidu SAM download link retained in the workflow note as requested.

## Before publishing a Registry version

- [ ] Install this repository into a clean Windows ComfyUI profile and restart that profile.
- [x] Import the standalone release package with the ComfyUI embedded Python runtime and confirm all three expected node mappings.
- [x] Confirm SAM is discovered from the registered ComfyUI checkpoint search paths (`models/checkpoints` and configured roots).
- [x] Run package unit tests, Python compile check, frontend syntax checks, and parse the sample workflow/locale/Registry metadata.
- [x] Repair the sample workflow's Chinese instructions and confirm the user-provided Baidu download link remains intact.
- [ ] Run the packaged example workflow from a clean install and save its output/evidence.
- [ ] Add the Comfy Registry API key as the GitHub repository secret `REGISTRY_ACCESS_TOKEN`.
- [ ] Review the final Registry archive and publish version `0.1.0` only after explicit release approval.

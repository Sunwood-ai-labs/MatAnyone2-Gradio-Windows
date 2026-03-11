# Architecture

![MatAnyone runtime architecture](/matanyone-architecture.svg)

Diagram source files:

- `media/matanyone-architecture.drawio`
- `media/matanyone-architecture.svg`

## Repository layout

- `hugging_face/app.py`: Gradio entrypoint and UI wiring
- `hugging_face/matanyone2_wrapper.py`: matting loop wrapper around the inference core
- `hugging_face/tools/`: click prompting, mask painting, download helpers, UI support code
- `matanyone2/demo_core.py`: shared runtime for Gradio and CLI, including ffmpeg setup, media loading, SAM integration, matting orchestration, output writing, and debug artifact export
- `matanyone2/cli.py`: direct CLI entrypoint for validation and reproducible runs
- `matanyone2/gradio.py`: package-friendly Gradio launcher used by the `matanyone-gradio` console script
- `matanyone2/`: upstream model and inference implementation
- `pretrained_models/`: downloaded checkpoints
- `results/`: generated output artifacts

## Runtime flow

1. `matanyone-gradio`, `matanyone-cli`, `matanyone2-runtime`, `python -m matanyone2 <cli|webui>`, or the legacy wrapper script parse runtime flags and forward execution into `matanyone2/demo_core.py`.
2. ffmpeg and required checkpoints are resolved and downloaded into `pretrained_models/` if missing.
3. Media is loaded into a shared session state with source size, working size, frame count, FPS, and optional audio metadata.
4. SAM prompt points are converted into a template mask on the selected frame.
5. The matting loop runs through the shared inference path and produces foreground/alpha outputs.
6. Final outputs and debug artifacts are written into a dedicated run folder under `results/<input-name>_<timestamp>/`.

## Diagram reading notes

- Both the Gradio app and the CLI feed into the same orchestration layer in `matanyone2/demo_core.py`.
- SAM prompt processing and MatAnyone inference are separated so we can optimize them independently.
- `pretrained_models/` is a shared runtime dependency for both SAM and MatAnyone checkpoints.
- Public docs and README previews use tracked assets from `media/` instead of ignored files from `results/`.
- Debugging is easier now because every run writes its own input snapshots, SAM previews, masks, output previews, and `metadata.json` into the run folder.

## Why docs live separately

This repository is expected to grow beyond a single README. The `docs/` site gives us a stable place for:

- onboarding pages
- architecture notes
- future model management guides
- release notes or migration docs
- contributor-facing explanations

## Deployment

The repository now has a simple split between validation, documentation publishing, and release packaging:

- `Repo Checks` validates docs, maintained Python files, and package entrypoints
- `Docs Pages` publishes the VitePress site
- `Release Package` builds wheel and sdist artifacts for tagged releases

That gives us one workflow for day-to-day safety and separate workflows for published outputs.

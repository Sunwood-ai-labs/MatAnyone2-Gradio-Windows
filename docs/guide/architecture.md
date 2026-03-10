# Architecture

## Repository layout

- `hugging_face/app.py`: Gradio entrypoint, checkpoint loading, ffmpeg detection, UI wiring
- `hugging_face/matanyone2_wrapper.py`: matting loop wrapper around the inference core
- `hugging_face/tools/`: click prompting, mask painting, download helpers, UI support code
- `matanyone2/`: upstream model and inference implementation
- `pretrained_models/`: downloaded checkpoints
- `results/`: generated output artifacts

## Runtime flow

1. `hugging_face/app.py` parses CLI arguments and discovers ffmpeg.
2. SAM checkpoints and MatAnyone checkpoints are downloaded into `pretrained_models/` if missing.
3. The Gradio UI collects point prompts and builds masks from the selected frame.
4. `hugging_face/matanyone2_wrapper.py` runs the actual matting loop on frames.
5. Outputs are rendered back to the UI and written into `results/` for video jobs.

## Why docs live separately

This repository is expected to grow beyond a single README. The `docs/` site gives us a stable place for:

- onboarding pages
- architecture notes
- future model management guides
- release notes or migration docs
- contributor-facing explanations

## Deployment

The docs are built with VitePress and deployed by GitHub Actions through the `Docs Pages` workflow.

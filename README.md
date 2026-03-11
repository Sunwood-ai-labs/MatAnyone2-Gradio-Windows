<p align="center">
  <img src="https://sunwood-ai-labs.github.io/MatAnyone2-Gradio-Windows/matanyone-hero.svg" alt="MatAnyone hero banner" width="100%" />
</p>

# MatAnyone

Windows-friendly local runtime for `MatAnyone` and `MatAnyone 2`, with a shared core that powers both the Gradio WebUI and a reproducible CLI.

<p align="center">
  <a href="./README.ja.md">日本語</a> |
  <a href="https://sunwood-ai-labs.github.io/MatAnyone2-Gradio-Windows/">Docs</a> |
  <a href="https://pypi.org/project/matanyone2-runtime/">PyPI</a> |
  <a href="https://github.com/pq-yang/MatAnyone2">MatAnyone 2</a> |
  <a href="https://github.com/pq-yang/MatAnyone">MatAnyone</a> |
  <a href="https://huggingface.co/spaces/PeiqingYang/MatAnyone">Original Space</a>
</p>

<p align="center">
  <a href="https://github.com/Sunwood-ai-labs/MatAnyone2-Gradio-Windows/actions/workflows/repo-checks.yml"><img src="https://github.com/Sunwood-ai-labs/MatAnyone2-Gradio-Windows/actions/workflows/repo-checks.yml/badge.svg" alt="Repo Checks" /></a>
  <a href="https://github.com/Sunwood-ai-labs/MatAnyone2-Gradio-Windows/actions/workflows/docs-pages.yml"><img src="https://github.com/Sunwood-ai-labs/MatAnyone2-Gradio-Windows/actions/workflows/docs-pages.yml/badge.svg" alt="Docs Pages" /></a>
  <a href="https://github.com/Sunwood-ai-labs/MatAnyone2-Gradio-Windows/actions/workflows/release-package.yml"><img src="https://github.com/Sunwood-ai-labs/MatAnyone2-Gradio-Windows/actions/workflows/release-package.yml/badge.svg" alt="Release Package" /></a>
</p>

## What This Repo Provides

- A shared runtime core in `matanyone2/demo_core.py` used by both the CLI and the Gradio app
- A package entrypoint, `matanyone2-runtime`, with `cli` and `webui` subcommands
- Timestamped run folders under `results/<input-name>_<timestamp>/`
- Debug artifacts for every run, including SAM previews, masks, frame snapshots, and `metadata.json`
- Windows-friendly setup with `uv`, automatic checkpoint downloads, and ffmpeg discovery

## Quick Start

Requirements:

- Windows 10 or Windows 11
- Python 3.10
- `uv`
- `git`
- `ffmpeg`
- NVIDIA GPU recommended for practical speed

Install the base tools with `winget`:

```powershell
winget install astral-sh.uv
winget install Git.Git
winget install Gyan.FFmpeg
```

Install from PyPI if you just want to run the packaged runtime:

```powershell
uv venv --python 3.10
uv pip install --python .\.venv\Scripts\python.exe --upgrade pip setuptools wheel
uv pip install --python .\.venv\Scripts\python.exe torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
uv pip install --python .\.venv\Scripts\python.exe matanyone2-runtime
```

PyPI package:

- [`matanyone2-runtime` on PyPI](https://pypi.org/project/matanyone2-runtime/)

Install from source instead if you want to modify this repository locally:

```powershell
uv venv --python 3.10
uv pip install --python .\.venv\Scripts\python.exe --upgrade pip setuptools wheel
uv pip install --python .\.venv\Scripts\python.exe torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
uv pip install --python .\.venv\Scripts\python.exe -e .
```

Installed entrypoints:

- `matanyone2-runtime`
- `matanyone-cli`
- `matanyone-gradio`
- `python -m matanyone2`

Recommended unified entrypoint:

- `matanyone2-runtime webui ...`
- `matanyone2-runtime cli ...`

Launch the WebUI:

```powershell
uv run --python .\.venv\Scripts\python.exe matanyone2-runtime webui --device cuda --port 7860 --server_name 127.0.0.1
```

Run the shared CLI:

```powershell
uv run --python .\.venv\Scripts\python.exe matanyone2-runtime cli --input .\media\bookcat.mp4 --device cpu --performance_profile fast --cpu_threads 8 --positive_point 280,180 --output_dir .\results
```

Then open `http://127.0.0.1:7860` for the WebUI, or inspect the generated run folder for CLI outputs.

## Run Output Layout

Every run creates a dedicated directory such as `results/bookcat_1773163828_6577592/`.

Typical contents:

- final outputs such as `*_foreground.mp4`, `*_alpha.mp4`, `*_mask.png`, and `*_sam_preview.png`
- debug artifacts such as `input_first_frame.png`, `input_selected_frame.png`, `sam_selected_preview.png`, and `sam_selected_mask.png`
- first and last matting snapshots such as `matting_output_first_*` and `matting_output_last_*`
- `metadata.json` with the exact runtime configuration

This makes it easier to compare runs, reproduce experiments, and inspect intermediate state without launching the UI.

## CLI and WebUI Share The Same Core

The Gradio app and the CLI both run through `matanyone2/demo_core.py`. Shared responsibilities include:

- runtime configuration and ffmpeg setup
- image and video loading
- SAM prompt handling
- matting orchestration
- final output writing
- debug artifact export

Thin entrypoints:

- `hugging_face/app.py` for the WebUI
- `matanyone2/cli.py` for direct CLI execution
- `matanyone2/runtime.py` for the package-friendly `matanyone2-runtime` launcher
- `scripts/run_pipeline_check.py` as a compatibility wrapper around the CLI

## Python API

The PyPI package now exposes an import-friendly API for host applications that want to call MatAnyone directly instead of shelling out to the CLI.

```python
from matanyone2 import run_pipeline

result = run_pipeline(
    input_path="bookcat.mp4",
    device="cpu",
    model="MatAnyone 2",
    positive_points=["280,180"],
)

print(result["foreground_path"])
print(result["alpha_path"])
```

The returned dictionary includes:

- `run_output_dir`
- `debug_dir`
- `foreground_path`
- `alpha_path`
- `mask_path`
- `sam_preview_path`
- `is_video`
- `fps`

## Reproducible Validation

To reproduce the historical `bookcat-profile-exp` style run more closely, pin the same prompt points and FPS settings:

```powershell
uv run --python .\.venv\Scripts\python.exe matanyone2-runtime cli --input .\media\bookcat.mp4 --device cpu --performance_profile fast --cpu_threads 8 --sam_model_type vit_h --frame_limit 241 --video_target_fps 0 --output_fps 12 --positive_point 280,180 --negative_point 30,30 --negative_point 530,30 --model "MatAnyone 2" --output_dir .\results
```

This workflow is also useful for CI smoke checks, scripted regression checks, and local benchmarking.

By default, the CLI now writes video outputs at the loaded media's processing FPS so playback duration stays aligned with the processed frames. Use `--output_fps` only when you intentionally want to override that for benchmarking or comparison runs.

## CI/CD

GitHub Actions now covers the main repository lifecycle:

- `Repo Checks`: builds docs, compiles maintained Python sources, lints the maintained runtime files, builds the package, and verifies the packaged entrypoints
- `Docs Pages`: builds the VitePress site and deploys it to GitHub Pages on pushes to `main`
- `Release Package`: builds wheel and sdist artifacts on `v*` tags, attaches them to a GitHub Release with SHA256 checksums, and publishes them to PyPI

If you want to cut a release:

```powershell
git tag v0.2.0
git push origin v0.2.0
```

That tag will trigger the package release workflow automatically.

To make PyPI publishing work, configure a trusted publisher for this repository on PyPI and bind it to the `pypi` GitHub Actions environment.

Published package:

- [`matanyone2-runtime` on PyPI](https://pypi.org/project/matanyone2-runtime/)

## Documentation

Full docs live at [sunwood-ai-labs.github.io/MatAnyone2-Gradio-Windows](https://sunwood-ai-labs.github.io/MatAnyone2-Gradio-Windows/) and in [`docs/`](./docs/index.md).

- Getting started: [`docs/guide/getting-started.md`](./docs/guide/getting-started.md)
- Usage guide: [`docs/guide/usage.md`](./docs/guide/usage.md)
- Performance notes: [`docs/guide/performance.md`](./docs/guide/performance.md)
- Architecture notes: [`docs/guide/architecture.md`](./docs/guide/architecture.md)
- CI/CD guide: [`docs/guide/ci-cd.md`](./docs/guide/ci-cd.md)
- Troubleshooting: [`docs/guide/troubleshooting.md`](./docs/guide/troubleshooting.md)

To preview the docs site locally:

```powershell
cd docs
npm install
npm run docs:dev
```

## Repository Layout

| Path | Purpose |
| --- | --- |
| `hugging_face/app.py` | Gradio app entrypoint and UI wiring |
| `hugging_face/tools/` | UI helper utilities used by the demo |
| `matanyone2/demo_core.py` | Shared runtime for WebUI and CLI |
| `matanyone2/cli.py` | Direct CLI entrypoint |
| `matanyone2/runtime.py` | Unified package entrypoint with `cli` and `webui` subcommands |
| `matanyone2/` | Upstream model and inference implementation |
| `pretrained_models/` | Auto-downloaded checkpoints, ignored by git |
| `results/` | Generated outputs and debug artifacts, ignored by git |
| `media/` | Repository branding and documentation assets |

## Troubleshooting

- If ffmpeg is not detected, confirm `ffmpeg.exe` is on `PATH` or installed via `winget`
- If CUDA startup fails, install a matching PyTorch build for your driver or switch to `--device cpu`
- If the first launch takes a while, checkpoints and sample media are being downloaded on demand
- If you want different host or port settings, use `--server_name` and `--port`

## Attribution And License

This repository is a derivative adaptation of:

- [MatAnyone2](https://github.com/pq-yang/MatAnyone2)
- [MatAnyone](https://github.com/pq-yang/MatAnyone)
- [Original Hugging Face demo](https://huggingface.co/spaces/PeiqingYang/MatAnyone)

The included [`LICENSE`](./LICENSE) is the upstream `S-Lab License 1.0`. Commercial use still requires permission from the original authors listed in that license file.

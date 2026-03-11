# Getting Started

## What you need

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

## Install dependencies

From the repository root:

```powershell
uv venv --python 3.10
uv pip install --python .\.venv\Scripts\python.exe --upgrade pip setuptools wheel
uv pip install --python .\.venv\Scripts\python.exe torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
uv pip install --python .\.venv\Scripts\python.exe -e .
```

Installed entrypoints:

- `matanyone2-runtime`
- `matanyone-gradio`
- `matanyone-cli`
- `python -m matanyone2`

Recommended entrypoint:

- `matanyone2-runtime webui ...` for the Gradio app
- `matanyone2-runtime cli ...` for direct pipeline execution
- `python -m matanyone2 webui ...` and `python -m matanyone2 cli ...` follow the same unified runtime

## Start the app

GPU mode:

```powershell
uv run --python .\.venv\Scripts\python.exe matanyone2-runtime webui --device cuda --port 7860 --server_name 127.0.0.1
```

CPU mode:

```powershell
uv run --python .\.venv\Scripts\python.exe matanyone2-runtime webui --device cpu --port 7860 --server_name 127.0.0.1
```

Then open `http://127.0.0.1:7860`.

## Run the shared CLI

The Gradio app and the validation path now share the same runtime core in `matanyone2/demo_core.py`. For quick local checks, you can run the same pipeline without opening the web UI:

```powershell
uv run --python .\.venv\Scripts\python.exe matanyone2-runtime cli --input .\media\bookcat.mp4 --device cpu --performance_profile fast --cpu_threads 8 --positive_point 280,180 --output_dir .\results
```

This creates a per-run folder such as `results/bookcat_1773163828_6577592/`.

## First-run downloads

The app downloads these files automatically when needed:

- `sam_vit_h_4b8939.pth`
- `matanyone.pth`
- `matanyone2.pth`
- example videos and images under `hugging_face/test_sample/`

Runtime checkpoints are stored in `pretrained_models/`.

## Output layout

Each run folder contains both final outputs and debug artifacts:

- `*_foreground.mp4` / `*_alpha.mp4` or image outputs
- `*_mask.png` and `*_sam_preview.png`
- `input_first_frame.png`, `input_selected_frame.png`
- `sam_selected_preview.png`, `sam_selected_mask.png`
- `matting_output_first_*`, `matting_output_last_*`
- `metadata.json`

## Local docs preview

If you want to preview this documentation site locally:

```powershell
cd docs
npm install
npm run docs:dev
```

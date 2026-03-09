---
title: MatAnyone
emoji: 🤡
colorFrom: red
colorTo: green
sdk: gradio
sdk_version: 4.31.0
python_version: 3.10.13
app_file: hugging_face/app.py
pinned: false
license: other
short_description: Gradio demo for MatAnyone 1 & 2
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

## Attribution

This repository is based on the original projects by Peiqing Yang and collaborators:

- MatAnyone2: https://github.com/pq-yang/MatAnyone2
- MatAnyone: https://github.com/pq-yang/MatAnyone
- Hugging Face demo source: https://huggingface.co/spaces/PeiqingYang/MatAnyone

This repo is a derivative/fork-style adaptation for local Windows + `uv` usage.

## Local Windows run with uv

Requirements:

- NVIDIA GPU with recent CUDA driver
- `uv`
- `git`
- `ffmpeg`

Install the required tools with `winget`:

```powershell
winget install astral-sh.uv
winget install Git.Git
winget install Gyan.FFmpeg
```

Create the virtual environment and install dependencies from the repository root:

```powershell
uv venv --python 3.10
uv pip install --python .\.venv\Scripts\python.exe --upgrade pip setuptools wheel
uv pip install --python .\.venv\Scripts\python.exe torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
uv pip install --python .\.venv\Scripts\python.exe -r requirements.txt
```

Run the Gradio app:

GPU mode:

```powershell
uv run python hugging_face/app.py --device cuda --port 7860 --server_name 127.0.0.1
```

CPU mode:

```powershell
uv run python hugging_face/app.py --device cpu --port 7860 --server_name 127.0.0.1
```

The app will start on [http://127.0.0.1:7860](http://127.0.0.1:7860).

Useful options:

```powershell
uv run python hugging_face/app.py --device cuda --port 7861 --server_name 127.0.0.1
```

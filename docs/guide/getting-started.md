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
uv pip install --python .\.venv\Scripts\python.exe -r requirements.txt
```

## Start the app

GPU mode:

```powershell
uv run --python .\.venv\Scripts\python.exe python hugging_face\app.py --device cuda --port 7860 --server_name 127.0.0.1
```

CPU mode:

```powershell
uv run --python .\.venv\Scripts\python.exe python hugging_face\app.py --device cpu --port 7860 --server_name 127.0.0.1
```

Then open `http://127.0.0.1:7860`.

## First-run downloads

The app downloads these files automatically when needed:

- `sam_vit_h_4b8939.pth`
- `matanyone.pth`
- `matanyone2.pth`
- example videos and images under `hugging_face/test_sample/`

Runtime checkpoints are stored in `pretrained_models/`.

## Local docs preview

If you want to preview this documentation site locally:

```powershell
cd docs
npm install
npm run docs:dev
```

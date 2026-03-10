# Usage

## Video workflow

1. Open the `Video` tab.
2. Load an input video.
3. Choose positive and negative points on the start frame.
4. Press `Add Mask` after the target region looks correct.
5. Run `Video Matting`.
6. Collect the generated foreground and alpha videos from `results/`.

## Image workflow

1. Open the `Image` tab.
2. Load an input image.
3. Use point prompts to isolate the subject.
4. Press `Add Mask`.
5. Run `Image Matting`.
6. Save the generated foreground and alpha outputs from the UI.

## Model selection

The UI exposes both checkpoints when they are available:

- `MatAnyone`
- `MatAnyone 2`

The app prefers `MatAnyone 2` as the default option when both model files exist.

## Useful runtime flags

```powershell
uv run --python .\.venv\Scripts\python.exe python hugging_face\app.py --help
```

Common flags:

- `--device cuda` or `--device cpu`
- `--port 7860`
- `--server_name 127.0.0.1`
- `--sam_model_type vit_h`

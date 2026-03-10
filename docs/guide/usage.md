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
- `--performance_profile auto|quality|balanced|fast`
- `--cpu_threads 8`

## CLI validation script

You can validate the full `SAM -> MatAnyone -> save outputs` path without launching Gradio:

```powershell
.\.venv\Scripts\python.exe scripts\run_pipeline_check.py --input media\bookcat.mp4 --device cpu --performance_profile balanced --cpu_threads 8 --sam_model_type vit_h --frame_limit 241 --positive_point 280,180 --negative_point 30,30 --negative_point 530,30 --output_dir results\bookcat-profile-exp\balanced
```

This script is what we used for the profile comparison documented in [`performance.md`](./performance.md). It writes mask previews, foreground outputs, alpha outputs, and timing-friendly artifacts under the chosen `--output_dir`.

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
uv run --python .\.venv\Scripts\python.exe matanyone2-runtime webui --help
```

Common flags:

- `--device cuda` or `--device cpu`
- `--port 7860`
- `--server_name 127.0.0.1`
- `--sam_model_type vit_h`
- `--performance_profile auto|quality|balanced|fast`
- `--cpu_threads 8`

If you installed from PyPI instead of an editable checkout, the same commands still apply. On Windows the generated launcher is usually available as either:

- `matanyone2-runtime ...`
- `.\.venv\Scripts\matanyone2-runtime.exe ...`

## CLI validation script

You can validate the full `SAM -> MatAnyone -> save outputs` path without launching Gradio:

```powershell
uv run --python .\.venv\Scripts\python.exe matanyone2-runtime cli --input media\bookcat.mp4 --device cpu --performance_profile balanced --cpu_threads 8 --sam_model_type vit_h --frame_limit 241 --video_target_fps 0 --output_fps 12 --positive_point 280,180 --negative_point 30,30 --negative_point 530,30 --output_dir results
```

You can also use `matanyone-cli ...`, `python -m matanyone2 cli ...`, or the legacy `scripts/run_pipeline_check.py`. The unified `matanyone2-runtime` entrypoint is the recommended one.

This workflow is what we used for the profile comparison documented in [`performance.md`](./performance.md). It writes mask previews, foreground outputs, alpha outputs, and timing-friendly artifacts under the chosen `--output_dir`.

If you do not pass `--output_fps`, the CLI preserves the loaded processing FPS so the output video duration matches the processed frame sequence. Set `--output_fps` only when you explicitly want a different playback rate.

## Shared runtime

The Gradio app and the CLI both run through the same shared runtime in `matanyone2/demo_core.py`. That means:

- CLI runs are a valid way to debug the same core behavior used by the web UI
- checkpoint loading, video/image preprocessing, SAM prompting, matting, and artifact writing stay aligned across both entrypoints
- the compatibility wrapper `scripts/run_pipeline_check.py` and `python -m matanyone2.cli` should produce equivalent outputs when given the same flags

## Run output structure

Current runs create a dedicated folder like `results/bookcat_1773163828_6577592/` instead of writing files directly into the top-level `results/` directory.

Inside that folder you will typically find:

- final outputs such as `bookcat_foreground.mp4` and `bookcat_alpha.mp4`
- the selected SAM preview and mask
- input frame snapshots used for debugging
- first and last matting output snapshots
- `metadata.json` with the run configuration

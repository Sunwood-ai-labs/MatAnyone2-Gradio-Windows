# Performance Notes

## What we measured

We benchmarked the end-to-end `SAM -> MatAnyone -> output video` pipeline with the local validation script:

```powershell
.\.venv\Scripts\python.exe scripts\run_pipeline_check.py --input media\bookcat.mp4 --device cpu --performance_profile fast --cpu_threads 8 --sam_model_type vit_h --frame_limit 241 --positive_point 280,180 --negative_point 30,30 --negative_point 530,30 --output_dir results\bookcat-profile-exp\fast
```

Test asset:

- Input: `media/bookcat.mp4`
- Resolution: `560x560`
- Frames: `241`
- Prompt points: `280,180` (positive), `30,30` and `530,30` (negative)

Generated comparison artifacts live under:

- `results/bookcat-profile-exp/`
- `results/bookcat-profile-exp-gpu/`
- `results/bookcat-profile-exp-compare.json`
- `media/bookcat-quality-preview.webp`
- `media/bookcat-balanced-preview.webp`
- `media/bookcat-fast-preview.webp`

Representative visual comparison:

- `results/bookcat-profile-exp/comparison_frame_120.png`

Compact animated previews:

| `quality` | `balanced` | `fast` |
| --- | --- | --- |
| <img src="/bookcat-quality-preview.webp" alt="Quality profile preview" width="180"> | <img src="/bookcat-balanced-preview.webp" alt="Balanced profile preview" width="180"> | <img src="/bookcat-fast-preview.webp" alt="Fast profile preview" width="180"> |

## Speed results

The timings below include mask generation, matting, and output writing.

| Profile | CPU time | GPU time | GPU speedup vs CPU |
| --- | ---: | ---: | ---: |
| `quality` | `227.76s` | `72.84s` | `3.13x` |
| `balanced` | `225.62s` | `78.72s` | `2.87x` |
| `fast` | `211.33s` | `71.43s` | `2.96x` |

## Quality deltas

`quality` was used as the reference output.

| Profile | Foreground MAE | Alpha MAE | Foreground PSNR | Alpha PSNR |
| --- | ---: | ---: | ---: | ---: |
| `balanced` | `0.87` | `0.63` | `33.82 dB` | `30.31 dB` |
| `fast` | `1.48` | `1.92` | `29.04 dB` | `24.14 dB` |

## Practical takeaways

- On this `560x560` sample, `balanced` stayed visually close to `quality`.
- `fast` was the quickest profile, but the gain was modest because the source video was already small.
- GPU stayed roughly `3x` faster than CPU for the full application path on this sample.
- The most visible quality tradeoff shows up in the alpha matte before it shows up in the composited foreground.

## Re-running the experiment

CPU:

```powershell
.\.venv\Scripts\python.exe scripts\run_pipeline_check.py --input media\bookcat.mp4 --device cpu --performance_profile balanced --cpu_threads 8 --sam_model_type vit_h --frame_limit 241 --positive_point 280,180 --negative_point 30,30 --negative_point 530,30 --output_dir results\bookcat-profile-exp\balanced
```

GPU:

```powershell
.\.venv\Scripts\python.exe scripts\run_pipeline_check.py --input media\bookcat.mp4 --device cuda --performance_profile balanced --sam_model_type vit_h --frame_limit 241 --positive_point 280,180 --negative_point 30,30 --negative_point 530,30 --output_dir results\bookcat-profile-exp-gpu\balanced
```

Animated WebP preview generation:

```powershell
uv run --project D:\Prj\video-background-remover video-background-remover D:\Prj\MatAnyone\results\bookcat-profile-exp\balanced\bookcat_foreground.mp4 D:\Prj\MatAnyone\media\bookcat-balanced-preview.webp --matanyone --alpha-video D:\Prj\MatAnyone\results\bookcat-profile-exp\balanced\bookcat_alpha.mp4 --animated webp --webp-fps 6 --max-frames 96 --size 280x280
```

The `--size 280x280` setting is the half-resolution post-process for the original `560x560` sample.

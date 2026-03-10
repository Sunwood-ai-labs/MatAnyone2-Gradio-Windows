# Troubleshooting

## ffmpeg is not found

- Confirm `ffmpeg.exe` is on `PATH`
- Or install it with `winget install Gyan.FFmpeg`
- The app also checks common Winget installation paths on Windows

## CUDA launch fails

- Confirm you installed a PyTorch build that matches your NVIDIA driver and CUDA setup
- Try `--device cpu` to verify the rest of the app is working

## First launch is slow

This is normal if the app is downloading checkpoints or sample media for the first time.

## Docs build fails locally

- Run `npm install` inside `docs/`
- Confirm your Node version is modern enough for VitePress
- Re-run `npm run docs:build`

## CI passes locally but fails on GitHub

Check the latest workflow logs:

```powershell
gh run list --repo Sunwood-ai-labs/MatAnyone2-Gradio-Windows --limit 5
gh run view RUN_ID --repo Sunwood-ai-labs/MatAnyone2-Gradio-Windows --log-failed
```

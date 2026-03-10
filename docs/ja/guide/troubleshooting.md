# トラブルシュート

## ffmpeg が見つからない

- `ffmpeg.exe` が `PATH` にあるか確認してください
- または `winget install Gyan.FFmpeg` で導入してください
- アプリは Windows 上の代表的な Winget 配置先も確認します

## CUDA 実行に失敗する

- NVIDIA ドライバと CUDA 環境に合った PyTorch を入れているか確認してください
- まず `--device cpu` でその他の経路が正常か切り分けるのも有効です

## 初回起動が遅い

初回は重みやサンプルメディアのダウンロードが走るため、ある程度正常です。

## docs の build がローカルで失敗する

- `docs/` 配下で `npm install` を実行してください
- VitePress が動く Node バージョンか確認してください
- `npm run docs:build` を再実行してください

## ローカルでは通るのに GitHub Actions で失敗する

最新の workflow ログを確認してください。

```powershell
gh run list --repo Sunwood-ai-labs/MatAnyone2-Gradio-Windows --limit 5
gh run view RUN_ID --repo Sunwood-ai-labs/MatAnyone2-Gradio-Windows --log-failed
```

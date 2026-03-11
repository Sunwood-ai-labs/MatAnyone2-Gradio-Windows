# MatAnyone

<p align="center">
  <img src="media/matanyone-hero.svg" alt="MatAnyone hero banner" width="100%" />
</p>

`MatAnyone` / `MatAnyone 2` をローカルで扱いやすくした Windows 向けランタイムです。Gradio WebUI と CLI が `matanyone2/demo_core.py` の共通コアを使うので、UI 検証と CLI 検証の結果を揃えやすくしています。

<p align="center">
  <a href="./README.md">English</a> |
  <a href="https://sunwood-ai-labs.github.io/MatAnyone2-Gradio-Windows/">Docs</a> |
  <a href="https://github.com/pq-yang/MatAnyone2">MatAnyone 2</a> |
  <a href="https://github.com/pq-yang/MatAnyone">MatAnyone</a> |
  <a href="https://huggingface.co/spaces/PeiqingYang/MatAnyone">Original Space</a>
</p>

## できること

- `matanyone2-runtime webui` で Gradio UI を起動
- `matanyone2-runtime cli` で同じ処理系を CLI から実行
- `results/<input-name>_<timestamp>/` に最終成果物と中間ファイルをまとめて保存
- `metadata.json` と SAM / matting の途中成果物でデバッグしやすい実行ログを残す

## セットアップ

必要なもの:

- Windows 10 / 11
- Python 3.10
- `uv`
- `git`
- `ffmpeg`

基本ツールのインストール:

```powershell
winget install astral-sh.uv
winget install Git.Git
winget install Gyan.FFmpeg
```

仮想環境とパッケージのインストール:

```powershell
uv venv --python 3.10
uv pip install --python .\.venv\Scripts\python.exe --upgrade pip setuptools wheel
uv pip install --python .\.venv\Scripts\python.exe torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
uv pip install --python .\.venv\Scripts\python.exe -e .
```

利用できるエントリポイント:

- `matanyone2-runtime`
- `matanyone-cli`
- `matanyone-gradio`
- `python -m matanyone2`

おすすめは unified runtime です。

```powershell
uv run --python .\.venv\Scripts\python.exe matanyone2-runtime webui --device cuda --port 7860 --server_name 127.0.0.1
uv run --python .\.venv\Scripts\python.exe matanyone2-runtime cli --input .\media\bookcat.mp4 --device cpu --performance_profile fast --cpu_threads 8 --positive_point 280,180 --output_dir .\results
```

## 出力構成

各 run は `results/bookcat_1773163828_6577592/` のような専用フォルダを作ります。

中には次のようなファイルが入ります。

- `*_foreground.mp4`, `*_alpha.mp4`, `*_mask.png`
- `input_first_frame.png`, `input_selected_frame.png`
- `sam_selected_preview.png`, `sam_selected_mask.png`
- `matting_output_first_*`, `matting_output_last_*`
- `metadata.json`

## CI/CD

GitHub Actions では次を整備しています。

- `Repo Checks`: docs build、Python compile、lint、package build、entrypoint 確認
- `Docs Pages`: `main` への push で VitePress を GitHub Pages にデプロイ
- `Release Package`: `v*` タグで wheel / sdist / SHA256SUMS を作成して GitHub Release に添付

リリース例:

```powershell
git tag v0.1.0
git push origin v0.1.0
```

## ドキュメント

- Getting Started: [`docs/guide/getting-started.md`](./docs/guide/getting-started.md)
- Usage: [`docs/guide/usage.md`](./docs/guide/usage.md)
- Performance: [`docs/guide/performance.md`](./docs/guide/performance.md)
- Architecture: [`docs/guide/architecture.md`](./docs/guide/architecture.md)
- CI/CD: [`docs/guide/ci-cd.md`](./docs/guide/ci-cd.md)
- Troubleshooting: [`docs/guide/troubleshooting.md`](./docs/guide/troubleshooting.md)

公開サイト:

- [sunwood-ai-labs.github.io/MatAnyone2-Gradio-Windows](https://sunwood-ai-labs.github.io/MatAnyone2-Gradio-Windows/)

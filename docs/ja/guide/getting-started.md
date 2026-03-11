# Getting Started

## 必要なもの

- Windows 10 または Windows 11
- Python 3.10
- `uv`
- `git`
- `ffmpeg`
- 実用速度を求めるなら NVIDIA GPU 推奨

基本ツールは `winget` で導入できます。

```powershell
winget install astral-sh.uv
winget install Git.Git
winget install Gyan.FFmpeg
```

## PyPI からインストール

実行だけしたい場合は、PyTorch を入れたあとに公開済みの PyPI パッケージを入れるのが最短です。

```powershell
uv venv --python 3.10
uv pip install --python .\.venv\Scripts\python.exe --upgrade pip setuptools wheel
uv pip install --python .\.venv\Scripts\python.exe torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
uv pip install --python .\.venv\Scripts\python.exe matanyone2-runtime
```

パッケージページ:

- [`matanyone2-runtime` on PyPI](https://pypi.org/project/matanyone2-runtime/)

## ソースからインストール

このリポジトリを編集しながら使う場合は、リポジトリルートで editable install を使います。

```powershell
uv venv --python 3.10
uv pip install --python .\.venv\Scripts\python.exe --upgrade pip setuptools wheel
uv pip install --python .\.venv\Scripts\python.exe torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
uv pip install --python .\.venv\Scripts\python.exe -e .
```

使えるエントリポイント:

- `matanyone2-runtime`
- `matanyone-gradio`
- `matanyone-cli`
- `python -m matanyone2`

推奨エントリポイント:

- `matanyone2-runtime webui ...` で Gradio アプリを起動
- `matanyone2-runtime cli ...` で同じパイプラインを CLI 実行
- `python -m matanyone2 webui ...` と `python -m matanyone2 cli ...` も同じ統一ランタイムを使います

## アプリの起動

GPU モード:

```powershell
uv run --python .\.venv\Scripts\python.exe matanyone2-runtime webui --device cuda --port 7860 --server_name 127.0.0.1
```

CPU モード:

```powershell
uv run --python .\.venv\Scripts\python.exe matanyone2-runtime webui --device cpu --port 7860 --server_name 127.0.0.1
```

起動後は `http://127.0.0.1:7860` を開きます。

## 共通 CLI の実行

Gradio アプリと検証用 CLI はどちらも `matanyone2/demo_core.py` の共通コアを使っています。Web UI を開かずに同じ処理を試すなら、次のように実行できます。

```powershell
uv run --python .\.venv\Scripts\python.exe matanyone2-runtime cli --input .\media\bookcat.mp4 --device cpu --performance_profile fast --cpu_threads 8 --positive_point 280,180 --output_dir .\results
```

この実行で `results/bookcat_1773163828_6577592/` のような run 専用フォルダが作られます。

## 初回ダウンロード

必要になったタイミングで次のファイルを自動ダウンロードします。

- `sam_vit_h_4b8939.pth`
- `matanyone.pth`
- `matanyone2.pth`
- `hugging_face/test_sample/` 配下のサンプル動画と画像

ランタイム用チェックポイントは `pretrained_models/` に保存されます。

## 出力構成

各 run フォルダには最終成果物とデバッグ成果物の両方が入ります。

- `*_foreground.mp4` / `*_alpha.mp4` または画像出力
- `*_mask.png` と `*_sam_preview.png`
- `input_first_frame.png`, `input_selected_frame.png`
- `sam_selected_preview.png`, `sam_selected_mask.png`
- `matting_output_first_*`, `matting_output_last_*`
- `metadata.json`

## docs サイトのローカル確認

ドキュメントサイトをローカル表示する場合:

```powershell
cd docs
npm install
npm run docs:dev
```

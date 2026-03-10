# セットアップ

## 必要なもの

- Windows 10 または Windows 11
- Python 3.10
- `uv`
- `git`
- `ffmpeg`
- 実用速度を考えると NVIDIA GPU 推奨

基本ツールは `winget` で入れられます。

```powershell
winget install astral-sh.uv
winget install Git.Git
winget install Gyan.FFmpeg
```

## 依存関係の導入

リポジトリのルートで実行します。

```powershell
uv venv --python 3.10
uv pip install --python .\.venv\Scripts\python.exe --upgrade pip setuptools wheel
uv pip install --python .\.venv\Scripts\python.exe torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
uv pip install --python .\.venv\Scripts\python.exe -r requirements.txt
```

## アプリの起動

GPU 実行:

```powershell
uv run --python .\.venv\Scripts\python.exe python hugging_face\app.py --device cuda --port 7860 --server_name 127.0.0.1
```

CPU 実行:

```powershell
uv run --python .\.venv\Scripts\python.exe python hugging_face\app.py --device cpu --port 7860 --server_name 127.0.0.1
```

起動後は `http://127.0.0.1:7860` を開きます。

## 初回ダウンロード

必要に応じて次のファイルが自動取得されます。

- `sam_vit_h_4b8939.pth`
- `matanyone.pth`
- `matanyone2.pth`
- `hugging_face/test_sample/` 配下のサンプル画像と動画

重みは `pretrained_models/` に保存されます。

## docs サイトのローカル確認

```powershell
cd docs
npm install
npm run docs:dev
```

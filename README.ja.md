# MatAnyone

<p align="center">
  <img src="media/matanyone-hero.svg" alt="MatAnyone hero banner" width="100%" />
</p>

`MatAnyone` / `MatAnyone 2` の Gradio デモを、Windows でローカル実行しやすいように調整したリポジトリです。元の研究実装と Hugging Face Space をベースにしつつ、`uv` ベースのセットアップ、ffmpeg 検出、ランタイム生成物の整理を加えています。

<p align="center">
  <a href="./README.md">English</a> |
  <a href="https://sunwood-ai-labs.github.io/MatAnyone2-Gradio-Windows/">Docs</a> |
  <a href="https://huggingface.co/spaces/PeiqingYang/MatAnyone">元の Space</a> |
  <a href="https://github.com/pq-yang/MatAnyone2">MatAnyone 2</a> |
  <a href="https://github.com/pq-yang/MatAnyone">MatAnyone</a>
</p>

## ✨ このリポジトリでできること

- [`hugging_face/app.py`](./hugging_face/app.py) を起点に、Windows 上でローカル実行しやすい形でデモを動かせます。
- `MatAnyone` と `MatAnyone 2` の両方のチェックポイントを同じ UI から切り替えられます。
- `uv` を前提にした導入手順、ffmpeg の自動検出、生成物の `.gitignore` 整備が入っています。

## 🎯 主なワークフロー

| ワークフロー | 入力 | 出力 |
| --- | --- | --- |
| 画像マッティング | 画像 1 枚と positive / negative クリック | 前景プレビューと alpha プレビュー |
| 動画マッティング | 動画 1 本と開始フレーム上のマスク指定 | `results/` に前景 MP4 と alpha MP4 |
| モデル比較 | `MatAnyone` / `MatAnyone 2` の切り替え | 同じ UI で挙動を比較 |

初回起動時には、必要なランタイム資産が自動ダウンロードされます。

- `sam_vit_h_4b8939.pth`
- `matanyone.pth`
- `matanyone2.pth`
- `hugging_face/test_sample/` 配下のサンプルメディア

チェックポイントは `pretrained_models/` に保存され、git 管理からは除外されています。

## 🧰 前提条件

- Windows 10 または Windows 11
- Python 3.10
- `uv`
- `git`
- `ffmpeg`
- 実用速度を考えると NVIDIA GPU 推奨。`--device cpu` でも動きますがかなり遅くなります

基本ツールは `winget` で導入できます。

```powershell
winget install astral-sh.uv
winget install Git.Git
winget install Gyan.FFmpeg
```

## 🚀 セットアップ

リポジトリのルートで仮想環境を作り、依存関係を入れます。

```powershell
uv venv --python 3.10
uv pip install --python .\.venv\Scripts\python.exe --upgrade pip setuptools wheel
uv pip install --python .\.venv\Scripts\python.exe torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
uv pip install --python .\.venv\Scripts\python.exe -r requirements.txt
```

GPU 実行:

```powershell
uv run --python .\.venv\Scripts\python.exe python hugging_face\app.py --device cuda --port 7860 --server_name 127.0.0.1
```

CPU 実行:

```powershell
uv run --python .\.venv\Scripts\python.exe python hugging_face\app.py --device cpu --port 7860 --server_name 127.0.0.1
```

CPU 環境でより軽快に動かしたい場合は、`fast` プロファイルとスレッド数指定から試すのがおすすめです。

```powershell
uv run --python .\.venv\Scripts\python.exe python hugging_face\app.py --device cpu --performance_profile fast --cpu_threads 8 --sam_model_type vit_b --port 7860 --server_name 127.0.0.1
```

起動後は `http://127.0.0.1:7860` を開いてください。

補助コマンド:

```powershell
uv run --python .\.venv\Scripts\python.exe python hugging_face\app.py --device cuda --port 7861 --server_name 127.0.0.1
uv run --python .\.venv\Scripts\python.exe python hugging_face\app.py --device cpu --performance_profile fast --port 7860 --server_name 127.0.0.1
uv run --python .\.venv\Scripts\python.exe python hugging_face\app.py --help
```

`--performance_profile auto` は CPU では `fast`、GPU では `quality` を自動選択します。`--sam_model_type auto` も CPU では `vit_b`、GPU では `vit_h` を選びます。

## 📚 ドキュメント

将来の拡張に備えて、構造化されたドキュメントを公開サイト [sunwood-ai-labs.github.io/MatAnyone2-Gradio-Windows](https://sunwood-ai-labs.github.io/MatAnyone2-Gradio-Windows/) とソースツリーの [`docs/`](./docs/index.md) に用意しています。

- セットアップ: [`docs/ja/guide/getting-started.md`](./docs/ja/guide/getting-started.md)
- 使い方: [`docs/ja/guide/usage.md`](./docs/ja/guide/usage.md)
- パフォーマンス検証: [`docs/ja/guide/performance.md`](./docs/ja/guide/performance.md)
- アーキテクチャ: [`docs/ja/guide/architecture.md`](./docs/ja/guide/architecture.md)
- トラブルシュート: [`docs/ja/guide/troubleshooting.md`](./docs/ja/guide/troubleshooting.md)

ローカルで docs サイトを確認する場合:

```powershell
cd docs
npm install
npm run docs:dev
```

## Performance Snapshot

`media/bookcat.mp4` を使って、ローカル検証スクリプト [`scripts/run_pipeline_check.py`](./scripts/run_pipeline_check.py) から `SAM -> MatAnyone -> 動画出力` の全体パイプラインを計測しました。

| Profile | CPU time | GPU time | GPU speedup vs CPU |
| --- | ---: | ---: | ---: |
| `quality` | `227.76s` | `72.84s` | `3.13x` |
| `balanced` | `225.62s` | `78.72s` | `2.87x` |
| `fast` | `211.33s` | `71.43s` | `2.96x` |

比較結果の JSON は `results/bookcat-profile-exp-compare.json`、代表フレーム比較は `results/bookcat-profile-exp/comparison_frame_120.png` にあります。実験条件と `quality` 基準の差分は [`docs/ja/guide/performance.md`](./docs/ja/guide/performance.md) にまとめています。

## 🖱️ 使い方

1. `Video` か `Image` タブを開きます。
2. 入力ファイルを読み込みます。
3. positive / negative クリックで対象を指定します。
4. 対象が取れたら `Add Mask` を押します。
5. `Video Matting` または `Image Matting` を実行します。
6. 生成結果は `results/` から取り出せます。

## 🗂️ ディレクトリ構成

| パス | 役割 |
| --- | --- |
| `hugging_face/app.py` | Gradio アプリ本体とローカル実行用の補助処理 |
| `hugging_face/tools/` | デモ UI が使う補助ユーティリティ |
| `matanyone2/` | upstream 由来のモデル・推論コード |
| `pretrained_models/` | 自動ダウンロードされる重み。git では無視 |
| `results/` | 生成動画の出力先。git では無視 |
| `media/` | README で使うブランド用 SVG 資産 |

## 🛠️ トラブルシュート

- ffmpeg が見つからない場合は、`ffmpeg.exe` が `PATH` にあるか、`winget` で入っているかを確認してください。アプリは Winget の典型的な配置先も自動探索します。
- CUDA 実行に失敗する場合は、ドライバに合う PyTorch を入れるか、`--device cpu` に切り替えてください。
- 初回起動が遅いのは、チェックポイントやサンプルを都度ダウンロードしているためで、ある程度正常です。
- ポートやバインドアドレスを変えたい場合は `--port` と `--server_name` を使ってください。

## 🙏 クレジットとライセンス

このリポジトリは以下の upstream をベースにした派生版です。

- [MatAnyone2](https://github.com/pq-yang/MatAnyone2)
- [MatAnyone](https://github.com/pq-yang/MatAnyone)
- [元の Hugging Face デモ](https://huggingface.co/spaces/PeiqingYang/MatAnyone)

同梱されている [`LICENSE`](./LICENSE) は upstream の `S-Lab License 1.0` です。商用利用には、ライセンスに記載された元著者への確認が必要です。

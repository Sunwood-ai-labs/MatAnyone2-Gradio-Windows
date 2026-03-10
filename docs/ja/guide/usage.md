# 使い方

## 動画ワークフロー

1. `Video` タブを開きます。
2. 入力動画を読み込みます。
3. 開始フレーム上で positive / negative クリックを使って対象を指定します。
4. 対象領域が取れたら `Add Mask` を押します。
5. `Video Matting` を実行します。
6. 出力された foreground と alpha 動画を `results/` から回収します。

## 画像ワークフロー

1. `Image` タブを開きます。
2. 入力画像を読み込みます。
3. point prompt で対象を切り出します。
4. `Add Mask` を押します。
5. `Image Matting` を実行します。
6. UI から foreground と alpha を保存します。

## モデル選択

利用可能な重みがあれば、UI 上で次を切り替えられます。

- `MatAnyone`
- `MatAnyone 2`

両方ある場合は `MatAnyone 2` が既定値になります。

## よく使う起動オプション

```powershell
uv run --python .\.venv\Scripts\python.exe python hugging_face\app.py --help
```

主なフラグ:

- `--device cuda` または `--device cpu`
- `--port 7860`
- `--server_name 127.0.0.1`
- `--sam_model_type vit_h`
- `--performance_profile auto|quality|balanced|fast`
- `--cpu_threads 8`

## CLI 検証スクリプト

Gradio を立ち上げずに `SAM -> MatAnyone -> 出力保存` の全体経路を確認したい場合は、`scripts/run_pipeline_check.py` を使えます。

```powershell
.\.venv\Scripts\python.exe scripts\run_pipeline_check.py --input media\bookcat.mp4 --device cpu --performance_profile balanced --cpu_threads 8 --sam_model_type vit_h --frame_limit 241 --video_target_fps 0 --output_fps 12 --positive_point 280,180 --negative_point 30,30 --negative_point 530,30 --output_dir results\bookcat-profile-exp\balanced
```

このスクリプトは [`performance.md`](./performance.md) のプロファイル比較でも使っています。指定した `--output_dir` 配下に、マスクのプレビュー、foreground、alpha、比較用の生成物を保存します。

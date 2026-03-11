# Usage

## 動画ワークフロー

1. `Video` タブを開きます。
2. 入力動画を読み込みます。
3. 開始フレーム上で positive / negative の点を置きます。
4. 対象領域が狙えていることを確認して `Add Mask` を押します。
5. `Video Matting` を実行します。
6. 生成された foreground / alpha 動画を `results/` から回収します。

## 画像ワークフロー

1. `Image` タブを開きます。
2. 入力画像を読み込みます。
3. point prompt で対象を切り出します。
4. `Add Mask` を押します。
5. `Image Matting` を実行します。
6. UI から foreground / alpha 出力を保存します。

## モデル選択

利用可能なチェックポイントがそろっている場合、UI では次の 2 つを選べます。

- `MatAnyone`
- `MatAnyone 2`

両方ある場合の既定値は `MatAnyone 2` です。

## よく使うランタイムフラグ

```powershell
uv run --python .\.venv\Scripts\python.exe matanyone2-runtime webui --help
```

主なフラグ:

- `--device cuda` または `--device cpu`
- `--port 7860`
- `--server_name 127.0.0.1`
- `--sam_model_type vit_h`
- `--performance_profile auto|quality|balanced|fast`
- `--cpu_threads 8`

PyPI から入れた環境でも、同じ `matanyone2-runtime` コマンドで起動できます。Windows では次のどちらでも実行しやすいです。

- `matanyone2-runtime ...`
- `.\.venv\Scripts\matanyone2-runtime.exe ...`

## CLI での検証実行

Gradio を開かずに `SAM -> MatAnyone -> 保存` までの経路を検証したい場合は、共通ランタイムを CLI から実行できます。

```powershell
uv run --python .\.venv\Scripts\python.exe matanyone2-runtime cli --input media\bookcat.mp4 --device cpu --performance_profile balanced --cpu_threads 8 --sam_model_type vit_h --frame_limit 241 --video_target_fps 0 --output_fps 12 --positive_point 280,180 --negative_point 30,30 --negative_point 530,30 --output_dir results
```

`matanyone-cli ...`、`python -m matanyone2 cli ...`、従来の `scripts/run_pipeline_check.py` でも近い実行はできますが、推奨は統一エントリポイントの `matanyone2-runtime` です。

このワークフローは [`performance.md`](./performance.md) の比較でも使っています。指定した `--output_dir` の下に mask preview、foreground、alpha、各種デバッグ成果物がまとまって出力されます。

`--output_fps` を省略した場合、CLI は読み込んだ処理 FPS を保ったまま動画を書き出すため、出力動画の再生時間が処理フレーム列と一致しやすくなります。意図的に再生速度を変えたいときだけ `--output_fps` を指定してください。

## 共通ランタイム

Gradio アプリと CLI はどちらも `matanyone2/demo_core.py` を通ります。つまり:

- CLI 実行は Web UI と同じコア挙動のデバッグ手段になります
- checkpoint 読み込み、前処理、SAM prompting、matting、成果物保存の流れが両 entrypoint でそろいます
- `scripts/run_pipeline_check.py` や `python -m matanyone2.cli` も、同じフラグならほぼ同等の結果を期待できます

## run 出力構造

現在の run は `results/bookcat_1773163828_6577592/` のような専用フォルダを作り、トップレベルの `results/` に直接ばらまかない構成です。

その中には通常次が入ります。

- `bookcat_foreground.mp4` や `bookcat_alpha.mp4` などの最終成果物
- 選択した SAM preview と mask
- デバッグ用の入力フレームスナップショット
- matting の先頭 / 末尾スナップショット
- run 設定を記録した `metadata.json`

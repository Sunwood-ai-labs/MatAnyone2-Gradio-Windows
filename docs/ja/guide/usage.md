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

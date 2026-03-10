# アーキテクチャ

## リポジトリ構成

- `hugging_face/app.py`: Gradio の入口、重み読み込み、ffmpeg 検出、UI 配線
- `hugging_face/matanyone2_wrapper.py`: 推論コアを呼ぶマッティング用ラッパー
- `hugging_face/tools/`: クリック入力、マスク描画、ダウンロード補助、UI サポート
- `matanyone2/`: upstream 由来のモデル・推論実装
- `pretrained_models/`: ダウンロード済みの重み
- `results/`: 生成結果

## 実行の流れ

1. `hugging_face/app.py` が CLI 引数を読み、ffmpeg を検出します。
2. 必要であれば SAM と MatAnyone の重みを `pretrained_models/` にダウンロードします。
3. Gradio UI で選択したフレームからポイント入力を受け、マスクを組み立てます。
4. `hugging_face/matanyone2_wrapper.py` がフレーム列に対してマッティングを実行します。
5. UI と `results/` に出力を書き戻します。

## docs を分ける理由

このリポジトリは README 1 枚では足りなくなる前提で育てるため、`docs/` を次の情報の置き場にしています。

- 導入ガイド
- 構成メモ
- 将来のモデル管理手順
- リリースノートや移行ガイド
- 開発者向け説明

## デプロイ

docs は VitePress で build され、GitHub Actions の `Docs Pages` workflow から GitHub Pages に配信されます。

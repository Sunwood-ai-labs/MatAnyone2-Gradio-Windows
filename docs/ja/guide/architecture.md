# アーキテクチャ

![MatAnyone runtime architecture](/matanyone-architecture-ja.svg)

図のソース:

- `media/matanyone-architecture-ja.drawio`
- `media/matanyone-architecture-ja.svg`

## リポジトリ構成

- `hugging_face/app.py`: Gradio の起点、CLI 引数処理、ffmpeg 検出、モデル選択
- `hugging_face/matanyone2_wrapper.py`: image / video の matting ループをまとめるラッパー
- `hugging_face/tools/`: クリック入力、マスク生成、ダウンロード補助などの UI 周辺コード
- `matanyone2/`: upstream 由来のモデル本体と推論実装
- `pretrained_models/`: 実行時に取得されるチェックポイント
- `results/`: 生成された出力や検証結果

## 実行フロー

1. `hugging_face/app.py` が CLI 引数を解釈し、ffmpeg を見つけます。
2. 必要な SAM と MatAnyone のチェックポイントがなければ `pretrained_models/` に取得します。
3. Gradio UI か `scripts/run_pipeline_check.py` が入力フレームと point prompt を渡します。
4. `hugging_face/tools/interact_tools.py` が SAM ベースのマスクを作り、template mask を用意します。
5. `hugging_face/matanyone2_wrapper.py` がフレーム列を処理し、`matanyone2/` の推論コアを呼び出します。
6. 出力は UI プレビュー、動画ファイル、実験用の生成物として返されます。

## 図の見どころ

- Gradio と CLI 検証スクリプトは、途中から同じ実行パスを通ります。
- SAM のマスク生成段と MatAnyone の推論段を分けてあるので、別々に最適化しやすい構造です。
- `pretrained_models/` は SAM と MatAnyone の両方から参照される共有のランタイム依存です。
- README と docs に載せる画像は、無視される `results/` ではなく追跡される `media/` に置いています。

## docs を分けている理由

このリポジトリは README 1 枚では収まらない説明が増えていく前提なので、`docs/` を次の情報の受け皿として使っています。

- セットアップ手順
- アーキテクチャ解説
- モデル運用メモ
- リリースノートや移行ガイド
- コントリビュータ向けの補足説明

## デプロイ

docs は VitePress で build され、GitHub Actions の `Docs Pages` workflow から GitHub Pages へ公開されます。

# CI/CD

## ワークフロー

このリポジトリでは GitHub Actions を 3 つ使っています。

- `Repo Checks`: Pull Request と `main` / `codex/**` への push で実行
- `Docs Pages`: `main` への push で VitePress を GitHub Pages に配信
- `Release Package`: `v*` タグの push で package artifact を作成して GitHub Release に添付

## Repo Checks

日常開発の主な安全網です。現在は次を確認します。

- docs の必須アセットが揃っていること
- docs サイトが build できること
- 管理対象の Python ファイルが compile できること
- 管理対象の runtime / package 周辺が `ruff` を通ること
- wheel と sdist が build できること
- editable install 後に期待する entrypoint が存在すること
- `matanyone2-runtime --help` と `python -m matanyone2 --help` が動くこと

## Docs Pages

`docs/` からサイトを build し、`docs/.vitepress/dist` を GitHub Pages にデプロイします。

つまり docs は 2 段階で守られます。

- `Repo Checks` で Pull Request 時点の build を確認
- `Docs Pages` で `main` 更新後に同じ成果物を公開

## Release Package

`v0.1.0` のようなタグを push すると、次の流れで配布物を作ります。

1. sdist と wheel を build
2. `SHA256SUMS.txt` を生成
3. workflow artifact として保存
4. GitHub Release を作成または更新
5. 生成したファイルを Release に添付
6. 同じ配布物を PyPI に公開

例:

```powershell
git tag v0.1.0
git push origin v0.1.0
```

## PyPI の事前設定

PyPI 公開には、リポジトリ外で一度だけ設定が必要です。

1. PyPI 上で `matanyone2-runtime` を作成するか、使いたい配布名を確保する
2. PyPI 側でこの GitHub リポジトリと workflow を trusted publisher として登録する
3. `.github/workflows/release-package.yml` で使っている `pypi` environment と対応付ける
4. tag release は `main` 系の公開フローに揃える

現在の workflow は公式の `pypa/gh-action-pypi-publish` と OIDC を使うので、trusted publishing を設定すれば長期トークンを GitHub secrets に置かずに運用できます。

## push 前のローカル確認

```powershell
.\.venv\Scripts\python.exe -m compileall hugging_face matanyone2 scripts
.\.venv\Scripts\python.exe -m ruff check pyproject.toml hugging_face\app.py hugging_face\__init__.py scripts\run_pipeline_check.py
.\.venv\Scripts\python.exe -m ruff check matanyone2\__init__.py matanyone2\__main__.py matanyone2\cli.py matanyone2\demo_core.py matanyone2\gradio.py matanyone2\runtime.py
.\.venv\Scripts\python.exe -m build
```

editable install 済みなら、入口の確認もできます。

```powershell
.\.venv\Scripts\matanyone2-runtime.exe --help
.\.venv\Scripts\python.exe -m matanyone2 --help
```

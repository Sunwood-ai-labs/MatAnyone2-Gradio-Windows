---
layout: home

hero:
  name: MatAnyone
  text: Windows 向けローカルランタイム
  tagline: CLI と Gradio WebUI の共通ドキュメント。再現しやすい実行方法、パッケージ利用、リリース運用をまとめています。
  image:
    src: /matanyone-hero.svg
    alt: MatAnyone hero
  actions:
    - theme: brand
      text: セットアップ
      link: /ja/guide/getting-started
    - theme: alt
      text: PyPI パッケージ
      link: https://pypi.org/project/matanyone2-runtime/
    - theme: alt
      text: アーキテクチャ
      link: /ja/guide/architecture
    - theme: alt
      text: CI/CD
      link: /ja/guide/ci-cd
    - theme: alt
      text: English
      link: /

features:
  - title: 共通ランタイム
    details: CLI と WebUI はどちらも matanyone2/demo_core.py を通るので、検証結果と UI 動作をそろえやすい構成です。
  - title: デバッグしやすい出力
    details: 各 run はタイムスタンプ付きフォルダにまとまり、成果物、SAM preview、mask、snapshot、metadata.json を残します。
  - title: リリース運用込み
    details: GitHub Actions が docs build、package build、Pages 配信、version tag での公開までカバーします。
  - title: PyPI 公開済み
    details: matanyone2-runtime は PyPI から直接インストールでき、リポジトリを clone しなくても使い始められます。
---

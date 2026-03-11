---
layout: home

hero:
  name: MatAnyone
  text: Windows 向けローカルランタイム
  tagline: CLI と Gradio WebUI を同じコアで動かし、再現しやすい検証と配布を支えるドキュメントです。
  image:
    src: /matanyone-hero.svg
    alt: MatAnyone hero
  actions:
    - theme: brand
      text: セットアップ
      link: /ja/guide/getting-started
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
    details: CLI と WebUI はどちらも matanyone2/demo_core.py を通るので、検証経路を揃えやすくなっています。
  - title: デバッグしやすい出力
    details: 各 run はタイムスタンプ付きフォルダへ保存され、中間ファイルと metadata.json も一緒に残ります。
  - title: 配布しやすい運用
    details: GitHub Actions で docs build、package build、Pages 配信、release artifact 作成まで自動化しています。
---

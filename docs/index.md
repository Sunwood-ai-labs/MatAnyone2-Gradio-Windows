---
layout: home

hero:
  name: MatAnyone
  text: Windows-Friendly Local Runtime
  tagline: Shared CLI and Gradio docs for reproducible matting runs, package usage, and release automation.
  image:
    src: /matanyone-hero.svg
    alt: MatAnyone hero
  actions:
    - theme: brand
      text: Get Started
      link: /guide/getting-started
    - theme: alt
      text: Read the Architecture
      link: /guide/architecture
    - theme: alt
      text: CI/CD
      link: /guide/ci-cd
    - theme: alt
      text: 日本語
      link: /ja/

features:
  - title: Shared runtime core
    details: The CLI and WebUI both execute through the same orchestration path in matanyone2/demo_core.py.
  - title: Debuggable outputs
    details: Every run creates a timestamped folder with final outputs, SAM previews, masks, snapshots, and metadata.json.
  - title: Release-ready repository
    details: GitHub Actions now verifies the package, builds docs, deploys Pages, and publishes release artifacts on version tags.
---

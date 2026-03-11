# CI/CD

## Workflows

This repository ships with three GitHub Actions workflows:

- `Repo Checks`: runs on pull requests and pushes to `main` or `codex/**`
- `Docs Pages`: deploys the VitePress site to GitHub Pages on pushes to `main`
- `Release Package`: builds and publishes package artifacts when a `v*` tag is pushed

## Repo Checks

`Repo Checks` is the main safety net for everyday development. It currently verifies:

- required documentation assets exist
- the docs site builds successfully
- maintained Python surfaces compile
- maintained runtime and package files pass `ruff`
- the Python package builds as wheel and sdist
- the editable package exposes the expected entrypoints
- `matanyone2-runtime --help` and `python -m matanyone2 --help` work from the packaged install

This job intentionally focuses on maintained runtime surfaces instead of the entire upstream model tree, so we can keep CI signal tight while still inheriting upstream code.

## Docs Pages

`Docs Pages` builds the site from `docs/` and deploys `docs/.vitepress/dist` to GitHub Pages.

This means documentation quality is checked in two stages:

- `Repo Checks` confirms the docs build is healthy on every pull request
- `Docs Pages` publishes the same output after `main` is updated

## Release Package

When you push a tag like `v0.1.0`, the release workflow:

1. builds the source distribution and wheel
2. generates `SHA256SUMS.txt`
3. uploads the artifacts to the workflow run
4. creates or updates the GitHub Release for that tag
5. attaches the generated files to the release

Example:

```powershell
git tag v0.1.0
git push origin v0.1.0
```

## Local verification before pushing

These are the same checks we use locally before relying on CI:

```powershell
.\.venv\Scripts\python.exe -m compileall hugging_face matanyone2 scripts
.\.venv\Scripts\python.exe -m ruff check pyproject.toml hugging_face\app.py hugging_face\__init__.py scripts\run_pipeline_check.py
.\.venv\Scripts\python.exe -m ruff check matanyone2\__init__.py matanyone2\__main__.py matanyone2\cli.py matanyone2\demo_core.py matanyone2\gradio.py matanyone2\runtime.py
.\.venv\Scripts\python.exe -m build
```

If you already installed the project in editable mode, you can also smoke test the packaged entrypoints:

```powershell
.\.venv\Scripts\matanyone2-runtime.exe --help
.\.venv\Scripts\python.exe -m matanyone2 --help
```

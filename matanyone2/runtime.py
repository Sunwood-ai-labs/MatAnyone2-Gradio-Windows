from __future__ import annotations

from pathlib import Path
import runpy
import sys


def _render_help(program_name: str):
    return f"""usage: {program_name} [-h] {{cli,webui}} ...

Launch the shared MatAnyone runtime as either the CLI pipeline or the Gradio WebUI.

positional arguments:
  {{cli,webui}}
    cli        Run the shared CLI pipeline.
    webui      Launch the Gradio WebUI.

options:
  -h, --help   show this help message and exit
"""


def _run_module(module_name: str, argv: list[str]):
    original_argv = sys.argv[:]
    try:
        sys.argv = [module_name, *argv]
        runpy.run_module(module_name, run_name="__main__")
    finally:
        sys.argv = original_argv


def _resolve_program_name():
    if not sys.argv:
        return "matanyone2-runtime"
    entry_name = Path(sys.argv[0]).name
    if entry_name == "__main__.py":
        return "python -m matanyone2"
    if entry_name.lower().endswith(".exe"):
        return Path(entry_name).stem
    return entry_name


def main(argv: list[str] | None = None):
    original_argv = sys.argv[1:] if argv is None else argv
    args = list(original_argv)
    program_name = _resolve_program_name()
    runtime_help = _render_help(program_name)
    if not args or args[0] in {"-h", "--help"}:
        print(runtime_help)
        return

    command, *rest = args
    if command == "cli":
        _run_module("matanyone2.cli", rest)
        return

    if command == "webui":
        _run_module("hugging_face.app", rest)
        return

    print(f"Unknown command: {command}\n")
    print(runtime_help)
    raise SystemExit(2)


if __name__ == "__main__":
    main()

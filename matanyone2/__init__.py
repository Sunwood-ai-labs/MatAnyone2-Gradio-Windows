from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = ["InferenceCore", "MatAnyone2", "MatAnyoneRunResult", "run_pipeline"]

if TYPE_CHECKING:
    from matanyone2.inference.inference_core import InferenceCore as InferenceCore
    from matanyone2.model.matanyone2 import MatAnyone2 as MatAnyone2


def __getattr__(name: str):
    if name == "InferenceCore":
        from matanyone2.inference.inference_core import InferenceCore

        return InferenceCore
    if name == "MatAnyone2":
        from matanyone2.model.matanyone2 import MatAnyone2

        return MatAnyone2
    if name == "MatAnyoneRunResult":
        from matanyone2.api import MatAnyoneRunResult

        return MatAnyoneRunResult
    if name == "run_pipeline":
        from matanyone2.api import run_pipeline

        return run_pipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)

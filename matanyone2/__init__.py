from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = ["InferenceCore", "MatAnyone2"]

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
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)

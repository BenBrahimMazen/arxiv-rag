"""Evaluation package (lazy exports)."""
from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

__all__ = ["RAGASEvaluator", "TEST_SET"]

_EXPORTS = {
    "RAGASEvaluator": "src.evaluation.ragas_eval",
    "TEST_SET": "src.evaluation.test_set",
}

if TYPE_CHECKING:
    from src.evaluation.ragas_eval import RAGASEvaluator
    from src.evaluation.test_set import TEST_SET


def __getattr__(name: str):
    if name in _EXPORTS:
        return getattr(importlib.import_module(_EXPORTS[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

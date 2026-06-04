"""Retrieval package (lazy exports)."""
from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

__all__ = ["HybridSearcher", "Reranker"]

_EXPORTS = {
    "HybridSearcher": "src.retrieval.hybrid_search",
    "Reranker": "src.retrieval.reranker",
}

if TYPE_CHECKING:
    from src.retrieval.hybrid_search import HybridSearcher
    from src.retrieval.reranker import Reranker


def __getattr__(name: str):
    if name in _EXPORTS:
        return getattr(importlib.import_module(_EXPORTS[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

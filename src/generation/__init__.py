"""Generation package (lazy exports)."""
from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

__all__ = ["RAGChain"]

_EXPORTS = {"RAGChain": "src.generation.chain"}

if TYPE_CHECKING:
    from src.generation.chain import RAGChain


def __getattr__(name: str):
    if name in _EXPORTS:
        return getattr(importlib.import_module(_EXPORTS[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

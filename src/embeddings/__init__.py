"""Embeddings package (lazy exports)."""
from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

__all__ = [
    "Embedder",
    "VectorStore",
    "ChromaStore",
    "PineconeStore",
    "get_vector_store",
]

_EXPORTS = {
    "Embedder": "src.embeddings.embedder",
    "VectorStore": "src.embeddings.vector_store",
    "ChromaStore": "src.embeddings.vector_store",
    "PineconeStore": "src.embeddings.vector_store",
    "get_vector_store": "src.embeddings.vector_store",
}

if TYPE_CHECKING:
    from src.embeddings.embedder import Embedder
    from src.embeddings.vector_store import (
        ChromaStore,
        PineconeStore,
        VectorStore,
        get_vector_store,
    )


def __getattr__(name: str):
    if name in _EXPORTS:
        return getattr(importlib.import_module(_EXPORTS[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

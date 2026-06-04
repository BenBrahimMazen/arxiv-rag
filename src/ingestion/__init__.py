"""Ingestion package (lazy exports to avoid importing heavy deps eagerly)."""
from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

__all__ = ["ArxivClient", "PDFParser", "SemanticChunker"]

_EXPORTS = {
    "ArxivClient": "src.ingestion.arxiv_client",
    "PDFParser": "src.ingestion.pdf_parser",
    "SemanticChunker": "src.ingestion.chunker",
}

if TYPE_CHECKING:
    from src.ingestion.arxiv_client import ArxivClient
    from src.ingestion.chunker import SemanticChunker
    from src.ingestion.pdf_parser import PDFParser


def __getattr__(name: str):
    if name in _EXPORTS:
        return getattr(importlib.import_module(_EXPORTS[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

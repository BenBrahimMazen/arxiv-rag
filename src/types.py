"""Shared domain dataclasses used across ingestion, retrieval and generation.

Kept in one module to avoid circular imports between the pipeline stages.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


@dataclass
class ArxivPaper:
    """A paper's metadata as returned by the ArXiv API."""

    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    published_date: date
    categories: list[str]
    pdf_url: str
    full_text: str = ""


@dataclass
class ParsedPaper:
    """Result of extracting text from a downloaded PDF."""

    arxiv_id: str
    full_text: str
    sections: dict[str, str] = field(default_factory=dict)
    page_count: int = 0
    extraction_success: bool = True


@dataclass
class Chunk:
    """A retrievable unit of text from a paper."""

    chunk_id: str
    arxiv_id: str
    section: str
    text: str
    token_count: int
    chunk_index: int


@dataclass
class SearchResult:
    """A single retrieved chunk with a relevance score."""

    chunk_id: str
    arxiv_id: str
    section: str
    text: str
    score: float


@dataclass
class SourcePaper:
    """A cited paper surfaced to the user."""

    arxiv_id: str
    title: str
    url: str


@dataclass
class RAGResponse:
    """The full result of a RAG query."""

    answer: str
    sources: list[SourcePaper]
    retrieved_chunks: list[SearchResult]
    latency_ms: int

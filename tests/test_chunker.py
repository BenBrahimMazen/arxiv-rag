"""Tests for section-aware semantic chunking."""
from __future__ import annotations

import pytest

from src.ingestion.chunker import SemanticChunker
from src.types import ParsedPaper


@pytest.fixture(scope="module")
def chunker() -> SemanticChunker:
    return SemanticChunker(chunk_size=50, overlap=10)


def test_chunks_respect_section_boundaries(chunker: SemanticChunker) -> None:
    paper = ParsedPaper(
        arxiv_id="2301.00001",
        full_text="",
        sections={
            "Introduction": "This is the introduction. " * 20,
            "Results": "These are the results. " * 20,
        },
    )
    chunks = chunker.chunk_paper(paper)
    assert chunks, "expected at least one chunk"
    # No chunk should mix two sections.
    assert {c.section for c in chunks} <= {"Introduction", "Results"}
    for chunk in chunks:
        assert chunk.arxiv_id == "2301.00001"
        assert chunk.token_count > 0


def test_chunk_indices_are_sequential(chunker: SemanticChunker) -> None:
    paper = ParsedPaper(
        arxiv_id="2301.00002",
        full_text="",
        sections={"Body": "Sentence number one. Sentence number two. " * 30},
    )
    chunks = chunker.chunk_paper(paper)
    indices = [c.chunk_index for c in chunks]
    assert indices == list(range(len(chunks)))


def test_token_count_under_budget(chunker: SemanticChunker) -> None:
    paper = ParsedPaper(
        arxiv_id="2301.00003",
        full_text="",
        sections={"Body": "Short sentence here. " * 40},
    )
    chunks = chunker.chunk_paper(paper)
    # Allow a small slack because overlap re-adds sentences.
    assert all(c.token_count <= 50 + 15 for c in chunks)


def test_unique_chunk_ids(chunker: SemanticChunker) -> None:
    paper = ParsedPaper(
        arxiv_id="2301.00004",
        full_text="",
        sections={"Body": "Alpha beta gamma delta. " * 25},
    )
    chunks = chunker.chunk_paper(paper)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_empty_paper_returns_no_chunks(chunker: SemanticChunker) -> None:
    paper = ParsedPaper(arxiv_id="2301.00005", full_text="", sections={})
    assert chunker.chunk_paper(paper) == []

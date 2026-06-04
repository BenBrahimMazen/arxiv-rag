"""Tests for hybrid dense + BM25 retrieval and RRF fusion."""
from __future__ import annotations

import numpy as np

from src.embeddings.vector_store import VectorStore
from src.retrieval.hybrid_search import ChunkRecord, HybridSearcher
from src.types import SearchResult


class FakeVectorStore(VectorStore):
    """Returns a fixed dense ranking, ignoring the query embedding."""

    def __init__(self, results: list[SearchResult]) -> None:
        self._results = results

    def upsert(self, chunks, embeddings) -> None:  # noqa: D102
        pass

    def query(self, query_embedding, top_k=20):  # noqa: D102
        return self._results[:top_k]

    def delete_collection(self) -> None:  # noqa: D102
        pass

    def count(self) -> int:  # noqa: D102
        return len(self._results)


def _record(i: int, text: str) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=f"c{i}", arxiv_id=f"2301.0000{i}", section="Body", text=text
    )


def _result(i: int, text: str, score: float) -> SearchResult:
    return SearchResult(
        chunk_id=f"c{i}", arxiv_id=f"2301.0000{i}", section="Body", text=text,
        score=score,
    )


def _make_searcher() -> HybridSearcher:
    corpus = [
        _record(1, "transformers use self attention for sequence modeling"),
        _record(2, "convolutional networks process images with filters"),
        _record(3, "reinforcement learning optimizes a reward signal"),
        _record(4, "attention is all you need for translation tasks"),
    ]
    dense = [
        _result(2, corpus[1].text, 0.9),
        _result(1, corpus[0].text, 0.8),
        _result(3, corpus[2].text, 0.5),
    ]
    return HybridSearcher(FakeVectorStore(dense), corpus)


def test_search_returns_results() -> None:
    searcher = _make_searcher()
    out = searcher.search("attention", np.zeros(384), top_k=3)
    assert out
    assert len(out) <= 3


def test_bm25_surfaces_keyword_match() -> None:
    searcher = _make_searcher()
    out = searcher.search("attention translation", np.zeros(384), top_k=4)
    ids = [r.chunk_id for r in out]
    # Chunk 4 mentions attention+translation but is absent from dense list;
    # BM25 should surface it through fusion.
    assert "c4" in ids


def test_rrf_scores_are_descending() -> None:
    searcher = _make_searcher()
    out = searcher.search("attention", np.zeros(384), top_k=4)
    scores = [r.score for r in out]
    assert scores == sorted(scores, reverse=True)


def test_rrf_constant_formula() -> None:
    # Two lists agreeing on the top item should rank it first.
    a = [_result(1, "x", 1.0), _result(2, "y", 0.5)]
    b = [_result(1, "x", 1.0), _result(3, "z", 0.5)]
    fused = HybridSearcher._reciprocal_rank_fusion([a, b])
    assert fused[0].chunk_id == "c1"


def test_empty_corpus_uses_dense_only() -> None:
    dense = [_result(1, "only dense", 0.9)]
    searcher = HybridSearcher(FakeVectorStore(dense), [])
    out = searcher.search("anything", np.zeros(384), top_k=5)
    assert [r.chunk_id for r in out] == ["c1"]

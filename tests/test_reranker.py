"""Tests for the reranker re-scoring logic (model stubbed for speed)."""
from __future__ import annotations

from src.retrieval.reranker import Reranker
from src.types import SearchResult


class StubCrossEncoder:
    """Scores a passage by how many query words it contains."""

    def predict(self, pairs):
        scores = []
        for query, passage in pairs:
            q_words = set(query.lower().split())
            overlap = sum(1 for w in passage.lower().split() if w in q_words)
            scores.append(float(overlap))
        return scores


def _make_local_reranker() -> Reranker:
    # Bypass __init__ to avoid downloading the cross-encoder model.
    reranker = object.__new__(Reranker)
    reranker.backend = "local"
    reranker.model_name = "stub"
    reranker._model = StubCrossEncoder()
    reranker._cohere = None
    return reranker


def _result(i: int, text: str) -> SearchResult:
    return SearchResult(
        chunk_id=f"c{i}", arxiv_id=f"id{i}", section="Body", text=text, score=0.0
    )


def test_rerank_orders_by_relevance() -> None:
    reranker = _make_local_reranker()
    results = [
        _result(1, "the sky is blue and clear"),
        _result(2, "neural attention mechanisms in transformers"),
        _result(3, "attention transformers attention models"),
    ]
    out = reranker.rerank("attention transformers", results, top_n=2)
    assert len(out) == 2
    assert out[0].chunk_id == "c3"  # most query-word overlap


def test_rerank_respects_top_n() -> None:
    reranker = _make_local_reranker()
    results = [_result(i, f"doc {i} attention") for i in range(5)]
    out = reranker.rerank("attention", results, top_n=3)
    assert len(out) == 3


def test_rerank_empty_input() -> None:
    reranker = _make_local_reranker()
    assert reranker.rerank("anything", [], top_n=5) == []


def test_rerank_updates_scores() -> None:
    reranker = _make_local_reranker()
    results = [_result(1, "attention attention attention")]
    out = reranker.rerank("attention", results, top_n=1)
    assert out[0].score == 3.0

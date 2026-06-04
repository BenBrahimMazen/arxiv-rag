"""Hybrid retrieval: dense (vector) + sparse (BM25) fused with RRF."""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from src.embeddings.vector_store import VectorStore
from src.logging_conf import get_logger
from src.types import SearchResult

logger = get_logger(__name__)

_RRF_K = 60  # standard Reciprocal Rank Fusion constant
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class ChunkRecord:
    """A chunk as loaded from PostgreSQL for the BM25 corpus."""

    chunk_id: str
    arxiv_id: str
    section: str
    text: str


class HybridSearcher:
    """Combine dense vector search and BM25 keyword search via RRF."""

    def __init__(
        self, vector_store: VectorStore, bm25_corpus: list[ChunkRecord]
    ) -> None:
        """``bm25_corpus`` is every chunk (with metadata) loaded from Postgres."""
        self.vector_store = vector_store
        self._records = bm25_corpus
        self._tokenized = [_tokenize(r.text) for r in bm25_corpus]
        # rank_bm25 cannot index an empty corpus.
        self._bm25 = BM25Okapi(self._tokenized) if self._tokenized else None
        logger.info("HybridSearcher initialized over %d chunks", len(bm25_corpus))

    def search(
        self,
        query: str,
        query_embedding: np.ndarray,
        top_k: int = 20,
        alpha: float = 0.7,  # reserved for future weighted fusion
    ) -> list[SearchResult]:
        """Return ``top_k`` results fused from dense + BM25 rankings via RRF."""
        fetch = top_k * 2
        dense = self.vector_store.query(query_embedding, top_k=fetch)
        sparse = self._bm25_search(query, top_k=fetch)

        fused = self._reciprocal_rank_fusion([dense, sparse])
        return fused[:top_k]

    # ── BM25 ──────────────────────────────────────────────────────────────
    def _bm25_search(self, query: str, top_k: int) -> list[SearchResult]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        if not len(scores):
            return []
        top_idx = np.argsort(scores)[::-1][:top_k]
        results: list[SearchResult] = []
        for idx in top_idx:
            rec = self._records[int(idx)]
            results.append(
                SearchResult(
                    chunk_id=rec.chunk_id,
                    arxiv_id=rec.arxiv_id,
                    section=rec.section,
                    text=rec.text,
                    score=float(scores[idx]),
                )
            )
        return results

    # ── fusion ────────────────────────────────────────────────────────────
    @staticmethod
    def _reciprocal_rank_fusion(
        ranked_lists: list[list[SearchResult]],
    ) -> list[SearchResult]:
        """Fuse multiple ranked lists: score(d) = Σ 1/(k + rank_i(d))."""
        fused_scores: dict[str, float] = {}
        by_id: dict[str, SearchResult] = {}
        for ranked in ranked_lists:
            for rank, result in enumerate(ranked):
                fused_scores[result.chunk_id] = fused_scores.get(
                    result.chunk_id, 0.0
                ) + 1.0 / (_RRF_K + rank + 1)
                by_id.setdefault(result.chunk_id, result)

        ordered = sorted(fused_scores.items(), key=lambda kv: kv[1], reverse=True)
        out: list[SearchResult] = []
        for chunk_id, score in ordered:
            result = by_id[chunk_id]
            out.append(
                SearchResult(
                    chunk_id=result.chunk_id,
                    arxiv_id=result.arxiv_id,
                    section=result.section,
                    text=result.text,
                    score=score,
                )
            )
        return out

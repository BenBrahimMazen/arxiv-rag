"""Cross-encoder / Cohere reranking of retrieved passages."""
from __future__ import annotations

import time
from dataclasses import replace

from src.config import Settings, get_settings
from src.logging_conf import get_logger
from src.types import SearchResult

logger = get_logger(__name__)


class Reranker:
    """Re-score (query, passage) pairs with a more precise model.

    Backend chosen by ``RERANKER_BACKEND`` (``local`` or ``cohere``).
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.backend = self.settings.reranker_backend
        self._model = None
        self._cohere = None
        if self.backend == "cohere":
            self._init_cohere()
        else:
            self._init_local()

    def _init_cohere(self) -> None:
        import cohere

        if not self.settings.cohere_api_key:
            raise ValueError("COHERE_API_KEY required for cohere reranker backend")
        self._cohere = cohere.Client(api_key=self.settings.cohere_api_key)
        self.model_name = "rerank-english-v3.0"

    def _init_local(self) -> None:
        from sentence_transformers import CrossEncoder

        self.model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        self._model = CrossEncoder(self.model_name)

    def rerank(
        self, query: str, results: list[SearchResult], top_n: int = 5
    ) -> list[SearchResult]:
        """Return the ``top_n`` most relevant results, re-scored."""
        if not results:
            return []
        start = time.perf_counter()
        if self.backend == "cohere":
            reranked = self._rerank_cohere(query, results, top_n)
        else:
            reranked = self._rerank_local(query, results, top_n)
        latency_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "Rerank (%s): %d -> %d in %.0f ms",
            self.backend,
            len(results),
            len(reranked),
            latency_ms,
        )
        return reranked

    def _rerank_local(
        self, query: str, results: list[SearchResult], top_n: int
    ) -> list[SearchResult]:
        pairs = [(query, r.text) for r in results]
        scores = self._model.predict(pairs)
        scored = [
            replace(r, score=float(s))
            for r, s in zip(results, scores, strict=False)
        ]
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_n]

    def _rerank_cohere(
        self, query: str, results: list[SearchResult], top_n: int
    ) -> list[SearchResult]:
        resp = self._cohere.rerank(
            model=self.model_name,
            query=query,
            documents=[r.text for r in results],
            top_n=min(top_n, len(results)),
        )
        out: list[SearchResult] = []
        for item in resp.results:
            out.append(replace(results[item.index], score=float(item.relevance_score)))
        return out

"""End-to-end RAG chain: embed -> hybrid search -> rerank -> generate."""
from __future__ import annotations

import json
import re
import time
from collections.abc import AsyncGenerator, Callable

from src.embeddings.embedder import Embedder
from src.generation.llm import LLM
from src.generation.prompts import SYSTEM_PROMPT, build_user_prompt
from src.logging_conf import get_logger
from src.retrieval.hybrid_search import HybridSearcher
from src.retrieval.reranker import Reranker
from src.types import RAGResponse, SearchResult, SourcePaper

logger = get_logger(__name__)

_CITATION_RE = re.compile(
    r"arxiv:([0-9]+\.[0-9]+(?:v\d+)?|[a-z\-]+/\d+)", re.IGNORECASE
)

# Resolves an arxiv_id to (title, url); injected by the API from the DB.
PaperLookup = Callable[[str], tuple[str, str]]


def _default_lookup(arxiv_id: str) -> tuple[str, str]:
    return ("", f"https://arxiv.org/abs/{arxiv_id}")


class RAGChain:
    """Orchestrates retrieval and generation for a user question."""

    def __init__(
        self,
        searcher: HybridSearcher,
        reranker: Reranker,
        embedder: Embedder,
        llm: LLM,
        paper_lookup: PaperLookup | None = None,
        top_k: int = 20,
        top_n: int = 5,
    ) -> None:
        self.searcher = searcher
        self.reranker = reranker
        self.embedder = embedder
        self.llm = llm
        self.paper_lookup = paper_lookup or _default_lookup
        self.top_k = top_k
        self.top_n = top_n

    # ── retrieval shared by query() and stream_query() ────────────────────
    def _retrieve(self, question: str) -> list[SearchResult]:
        query_embedding = self.embedder.embed_query(question)
        candidates = self.searcher.search(
            question, query_embedding, top_k=self.top_k
        )
        return self.reranker.rerank(question, candidates, top_n=self.top_n)

    def _sources_from(
        self, answer: str, chunks: list[SearchResult]
    ) -> list[SourcePaper]:
        """Build the cited-paper list, preferring ids actually cited in answer."""
        cited = {m.lower() for m in _CITATION_RE.findall(answer)}
        retrieved_ids = list(dict.fromkeys(c.arxiv_id for c in chunks))
        # Prefer papers cited in the answer; fall back to all retrieved papers.
        chosen = [i for i in retrieved_ids if i.lower() in cited] or retrieved_ids
        sources: list[SourcePaper] = []
        for arxiv_id in chosen:
            title, url = self.paper_lookup(arxiv_id)
            sources.append(SourcePaper(arxiv_id=arxiv_id, title=title, url=url))
        return sources

    # ── public API ────────────────────────────────────────────────────────
    async def query(self, question: str) -> RAGResponse:
        """Run the full pipeline and return a structured response."""
        start = time.perf_counter()
        chunks = self._retrieve(question)
        prompt = build_user_prompt(question, chunks)
        answer = await self.llm.complete(SYSTEM_PROMPT, prompt)
        latency_ms = int((time.perf_counter() - start) * 1000)
        return RAGResponse(
            answer=answer,
            sources=self._sources_from(answer, chunks),
            retrieved_chunks=chunks,
            latency_ms=latency_ms,
        )

    async def stream_query(self, question: str) -> AsyncGenerator[str, None]:
        """Stream the answer as SSE, then emit a final ``sources`` event."""
        chunks = self._retrieve(question)
        prompt = build_user_prompt(question, chunks)

        answer_parts: list[str] = []
        async for token in self.llm.stream(SYSTEM_PROMPT, prompt):
            answer_parts.append(token)
            yield f"data: {json.dumps({'type': 'token', 'data': token})}\n\n"

        sources = self._sources_from("".join(answer_parts), chunks)
        payload = {
            "type": "sources",
            "data": [
                {"arxiv_id": s.arxiv_id, "title": s.title, "url": s.url}
                for s in sources
            ],
        }
        yield f"data: {json.dumps(payload)}\n\n"
        yield "data: [DONE]\n\n"

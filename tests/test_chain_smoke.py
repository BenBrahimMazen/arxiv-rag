"""Offline end-to-end smoke test of the RAG chain (no models, no network)."""
from __future__ import annotations

import numpy as np
import pytest

from src.embeddings.vector_store import VectorStore
from src.generation.chain import RAGChain
from src.generation.llm import EchoLLM
from src.retrieval.hybrid_search import ChunkRecord, HybridSearcher
from src.types import SearchResult


class FakeVectorStore(VectorStore):
    def __init__(self, results):
        self._results = results

    def upsert(self, chunks, embeddings):
        pass

    def query(self, query_embedding, top_k=20):
        return self._results[:top_k]

    def delete_collection(self):
        pass

    def count(self):
        return len(self._results)


class FakeEmbedder:
    def embed_query(self, text):
        return np.zeros(384, dtype=np.float32)


class IdentityReranker:
    def rerank(self, query, results, top_n=5):
        return results[:top_n]


@pytest.mark.asyncio
async def test_chain_end_to_end_offline():
    corpus = [
        ChunkRecord("c1", "2301.00001", "Results",
                    "LoRA freezes weights and adds low-rank matrices."),
        ChunkRecord("c2", "2301.00002", "Methodology",
                    "FlashAttention is IO-aware exact attention."),
    ]
    dense = [
        SearchResult("c1", "2301.00001", "Results", corpus[0].text, 0.9),
        SearchResult("c2", "2301.00002", "Methodology", corpus[1].text, 0.7),
    ]
    searcher = HybridSearcher(FakeVectorStore(dense), corpus)
    chain = RAGChain(
        searcher=searcher,
        reranker=IdentityReranker(),
        embedder=FakeEmbedder(),
        llm=EchoLLM(),
        top_k=5,
        top_n=5,
    )

    response = await chain.query("How does LoRA work?")
    assert response.answer
    assert response.retrieved_chunks
    assert response.latency_ms >= 0
    # Echo LLM cites retrieved papers, so sources should be populated.
    assert {s.arxiv_id for s in response.sources} <= {"2301.00001", "2301.00002"}


@pytest.mark.asyncio
async def test_chain_streaming_offline():
    corpus = [ChunkRecord("c1", "2301.00001", "Body", "evidence text here")]
    dense = [SearchResult("c1", "2301.00001", "Body", corpus[0].text, 0.9)]
    chain = RAGChain(
        searcher=HybridSearcher(FakeVectorStore(dense), corpus),
        reranker=IdentityReranker(),
        embedder=FakeEmbedder(),
        llm=EchoLLM(),
    )
    events = [chunk async for chunk in chain.stream_query("question?")]
    joined = "".join(events)
    assert "data:" in joined
    assert '"type": "sources"' in joined
    assert "[DONE]" in joined

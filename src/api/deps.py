"""Shared application state and component initialization for the API."""
from __future__ import annotations

from dataclasses import dataclass

from src.config import get_settings
from src.db.repository import load_bm25_corpus, paper_title_url
from src.db.session import init_db, session_scope
from src.embeddings.embedder import Embedder
from src.embeddings.vector_store import get_vector_store
from src.generation.chain import RAGChain
from src.generation.llm import get_llm
from src.logging_conf import get_logger
from src.retrieval.hybrid_search import HybridSearcher
from src.retrieval.reranker import Reranker

logger = get_logger(__name__)


@dataclass
class AppState:
    """Holds warm, reusable components for request handling."""

    chain: RAGChain
    embedder: Embedder
    searcher: HybridSearcher


_state: AppState | None = None


def _paper_lookup(arxiv_id: str) -> tuple[str, str]:
    with session_scope() as session:
        return paper_title_url(session, arxiv_id)


def build_state() -> AppState:
    """Initialize DB, vector store, BM25 index, embedder, reranker and chain."""
    get_settings()
    init_db()

    embedder = Embedder()
    vector_store = get_vector_store()
    with session_scope() as session:
        corpus = load_bm25_corpus(session)
    searcher = HybridSearcher(vector_store, corpus)
    reranker = Reranker()
    llm = get_llm()

    chain = RAGChain(
        searcher=searcher,
        reranker=reranker,
        embedder=embedder,
        llm=llm,
        paper_lookup=_paper_lookup,
    )
    logger.info("AppState built (corpus=%d chunks)", len(corpus))
    return AppState(chain=chain, embedder=embedder, searcher=searcher)


def set_state(state: AppState) -> None:
    global _state
    _state = state


def get_state() -> AppState:
    if _state is None:
        raise RuntimeError("AppState not initialized")
    return _state

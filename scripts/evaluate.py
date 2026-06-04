"""Run RAGAS evaluation over the 20-question test set.

Run:  python -m scripts.evaluate
Exits non-zero if faithfulness drops below the threshold (for CI gating).
"""
from __future__ import annotations

import argparse
import asyncio
import sys

from src.db.repository import load_bm25_corpus, paper_title_url
from src.db.session import init_db, session_scope
from src.embeddings.embedder import Embedder
from src.embeddings.vector_store import get_vector_store
from src.evaluation.ragas_eval import RAGASEvaluator
from src.generation.chain import RAGChain
from src.generation.llm import get_llm
from src.logging_conf import get_logger
from src.retrieval.hybrid_search import HybridSearcher
from src.retrieval.reranker import Reranker

logger = get_logger(__name__)


def _build_chain() -> RAGChain:
    init_db()
    embedder = Embedder()
    vector_store = get_vector_store()
    with session_scope() as session:
        corpus = load_bm25_corpus(session)
    searcher = HybridSearcher(vector_store, corpus)

    def lookup(arxiv_id: str) -> tuple[str, str]:
        with session_scope() as session:
            return paper_title_url(session, arxiv_id)

    return RAGChain(
        searcher=searcher,
        reranker=Reranker(),
        embedder=embedder,
        llm=get_llm(),
        paper_lookup=lookup,
    )


async def _run(threshold: float) -> int:
    chain = _build_chain()
    report = await RAGASEvaluator().run_evaluation(chain)
    logger.info("Aggregate scores: %s", report.aggregate)
    print(report.to_markdown())

    faithfulness = report.aggregate.get("faithfulness")
    if faithfulness is not None and faithfulness == faithfulness:  # not NaN
        if faithfulness < threshold:
            logger.error(
                "Faithfulness %.3f below threshold %.2f", faithfulness, threshold
            )
            return 1
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation")
    parser.add_argument("--threshold", type=float, default=0.7)
    args = parser.parse_args()
    sys.exit(asyncio.run(_run(args.threshold)))


if __name__ == "__main__":
    main()

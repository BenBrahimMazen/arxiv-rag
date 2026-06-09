"""RAGAS-based evaluation of the RAG chain over the hand-crafted test set."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime

from src.config import get_settings
from src.evaluation.test_set import TEST_SET
from src.generation.chain import RAGChain
from src.logging_conf import get_logger

logger = get_logger(__name__)

_METRICS = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]
_RESULTS_DIR = "evaluation_results"


class _LocalEmbeddings:
    """LangChain-compatible embeddings backed by our local sentence-transformer.

    Lets RAGAS compute embedding-based metrics for free (no OpenAI), reusing the
    same MiniLM model the retriever uses.
    """

    def __init__(self) -> None:
        from src.embeddings.embedder import Embedder

        self._embedder = Embedder()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._embedder.embed_batch(texts)]

    def embed_query(self, text: str) -> list[float]:
        return self._embedder.embed_query(text).tolist()


def _judge_components():
    """Return ``(llm, embeddings)`` for RAGAS, or ``(None, None)`` for OpenAI default.

    For the free ``groq`` backend this returns a Groq chat model as the judge and
    local MiniLM embeddings, so the whole evaluation runs without paid APIs.
    """
    settings = get_settings()
    if settings.llm_backend != "groq":
        return None, None  # let RAGAS use its OpenAI default (paid)

    from langchain_openai import ChatOpenAI

    judge = ChatOpenAI(
        model=settings.groq_model,
        temperature=0,
        api_key=settings.groq_api_key,
        base_url=settings.groq_base_url,
    )
    embeddings = _LocalEmbeddings()

    # Wrap with RAGAS adapters when available (newer RAGAS versions require it).
    try:
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper

        return LangchainLLMWrapper(judge), LangchainEmbeddingsWrapper(embeddings)
    except Exception:  # noqa: BLE001 — older RAGAS accepts raw langchain objects
        return judge, embeddings


@dataclass
class EvalReport:
    """Aggregate + per-question RAGAS scores."""

    aggregate: dict[str, float]
    per_question: list[dict] = field(default_factory=list)
    timestamp: str = ""

    def to_markdown(self) -> str:
        rows = "\n".join(
            f"| {m.replace('_', ' ').title()} | "
            f"{self.aggregate.get(m, float('nan')):.3f} |"
            for m in _METRICS
        )
        return f"| Metric | Score |\n| --- | --- |\n{rows}\n"


class RAGASEvaluator:
    """Run the chain over the test set and compute RAGAS metrics."""

    async def _collect(self, chain: RAGChain) -> list[dict]:
        samples: list[dict] = []
        for item in TEST_SET:
            response = await chain.query(item["question"])
            samples.append(
                {
                    "question": item["question"],
                    "answer": response.answer,
                    "contexts": [c.text for c in response.retrieved_chunks],
                    "ground_truth": item["ground_truth"],
                }
            )
            logger.info("Evaluated: %s", item["question"][:60])
        return samples

    async def run_evaluation(self, chain: RAGChain) -> EvalReport:
        """Collect predictions, score with RAGAS, persist JSON + markdown."""
        samples = await self._collect(chain)
        aggregate, per_question = self._score(samples)

        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        report = EvalReport(
            aggregate=aggregate, per_question=per_question, timestamp=timestamp
        )
        self._persist(report)
        return report

    def _score(self, samples: list[dict]) -> tuple[dict[str, float], list[dict]]:
        """Compute RAGAS metrics; degrade gracefully if RAGAS is unavailable."""
        try:
            from datasets import Dataset
            from ragas import evaluate
            from ragas.metrics import (
                answer_relevancy,
                context_precision,
                context_recall,
                faithfulness,
            )

            dataset = Dataset.from_list(samples)
            judge_llm, judge_embeddings = _judge_components()
            result = evaluate(
                dataset,
                metrics=[
                    faithfulness,
                    answer_relevancy,
                    context_recall,
                    context_precision,
                ],
                # When set (e.g. Groq + local embeddings) RAGAS uses these instead
                # of its OpenAI default, keeping evaluation completely free.
                llm=judge_llm,
                embeddings=judge_embeddings,
            )
            df = result.to_pandas()
            aggregate = {m: float(df[m].mean()) for m in _METRICS if m in df}
            per_question = df.to_dict(orient="records")
            return aggregate, per_question
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "RAGAS scoring unavailable (%s); returning empty scores", exc
            )
            return {m: float("nan") for m in _METRICS}, samples

    def _persist(self, report: EvalReport) -> None:
        os.makedirs(_RESULTS_DIR, exist_ok=True)
        json_path = os.path.join(_RESULTS_DIR, f"{report.timestamp}.json")
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "timestamp": report.timestamp,
                    "aggregate": report.aggregate,
                    "per_question": report.per_question,
                },
                fh,
                indent=2,
                default=str,
            )
        md_path = os.path.join(_RESULTS_DIR, f"{report.timestamp}.md")
        with open(md_path, "w", encoding="utf-8") as fh:
            fh.write(report.to_markdown())
        logger.info("Saved evaluation report to %s", json_path)

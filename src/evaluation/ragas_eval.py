"""RAGAS-style evaluation of the RAG chain over the hand-crafted test set.

Metrics follow the RAGAS definitions (faithfulness, answer relevancy, context
recall, context precision) but are computed by a self-contained LLM-as-judge
([judge.LLMJudge]) using the configured Groq model and local MiniLM embeddings,
because the RAGAS library is not compatible with the installed LangChain 1.x.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime

from src.evaluation.judge import LLMJudge
from src.evaluation.test_set import load_test_set
from src.generation.chain import RAGChain
from src.logging_conf import get_logger

logger = get_logger(__name__)

_METRICS = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]
_RESULTS_DIR = "evaluation_results"


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
        for item in load_test_set():
            response = await self._query_with_retry(chain, item["question"])
            if response is None:
                logger.warning("Skipping (gen failed): %s", item["question"][:45])
                continue
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

    @staticmethod
    async def _query_with_retry(chain: RAGChain, question: str, retries: int = 4):
        """Run chain.query with backoff so transient 429s don't abort the run."""
        import asyncio

        delay = 5.0
        for attempt in range(retries):
            try:
                return await chain.query(question)
            except Exception as exc:  # noqa: BLE001 — rate limits etc.
                if attempt == retries - 1:
                    logger.warning("Generation failed permanently: %s", exc)
                    return None
                await asyncio.sleep(delay)
                delay *= 2
        return None

    async def run_evaluation(self, chain: RAGChain) -> EvalReport:
        """Collect predictions, score with the LLM judge, persist JSON + markdown."""
        samples = await self._collect(chain)
        aggregate, per_question = await self._score(chain, samples)

        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        report = EvalReport(
            aggregate=aggregate, per_question=per_question, timestamp=timestamp
        )
        self._persist(report)
        return report

    async def _score(
        self, chain: RAGChain, samples: list[dict]
    ) -> tuple[dict[str, float], list[dict]]:
        """Score every sample with the LLM judge and aggregate the means."""
        judge = LLMJudge(self._judge_llm(chain), chain.embedder)
        per_question: list[dict] = []
        for i, sample in enumerate(samples, start=1):
            scored = await judge.score_sample(sample)
            per_question.append(scored)
            logger.info(
                "Judged %d/%d: %s", i, len(samples), sample["question"][:50]
            )

        aggregate: dict[str, float] = {}
        for metric in _METRICS:
            values = [row[metric] for row in per_question if metric in row]
            aggregate[metric] = float(sum(values) / len(values)) if values else 0.0
        return aggregate, per_question

    @staticmethod
    def _judge_llm(chain: RAGChain):
        """Use a lighter, higher-rate-limit Groq model for judging when on Groq.

        The judge only does classification/counting, so the 8B instant model is
        ample and keeps us comfortably within free-tier limits; otherwise reuse
        the chain's own LLM.
        """
        from src.config import get_settings
        from src.generation.llm import OpenAIChatLLM

        settings = get_settings()
        if settings.llm_backend != "groq":
            return chain.llm
        judge_settings = settings.model_copy(
            update={"groq_model": "llama-3.1-8b-instant"}
        )
        return OpenAIChatLLM(judge_settings)

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

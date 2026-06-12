"""Self-contained LLM-as-judge evaluator implementing the RAGAS metric set.

The RAGAS library is incompatible with LangChain 1.x (it hard-imports removed
VertexAI modules), so this module computes the same four metric definitions
directly with our configured LLM (Groq) as the judge and the local MiniLM model
for the embedding-based relevancy metric. This keeps evaluation completely free
and dependency-light while following the RAGAS definitions:

- faithfulness:      fraction of answer claims supported by the retrieved context
- answer_relevancy:  similarity between the question and questions the answer implies
- context_recall:    fraction of ground-truth claims covered by the retrieved context
- context_precision: fraction of retrieved context chunks that are relevant
"""
from __future__ import annotations

import asyncio
import json
import re

import numpy as np

from src.embeddings.embedder import Embedder
from src.generation.llm import LLM
from src.logging_conf import get_logger

logger = get_logger(__name__)

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_json(text: str) -> dict:
    """Extract the first JSON object from an LLM response, tolerating fences."""
    match = _JSON_RE.search(text)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def _ratio(supported: float, total: float) -> float:
    return 1.0 if total <= 0 else max(0.0, min(1.0, supported / total))


class LLMJudge:
    """Scores RAG samples with an LLM judge and local embeddings."""

    def __init__(self, llm: LLM, embedder: Embedder, pause_s: float = 0.4) -> None:
        self.llm = llm
        self.embedder = embedder
        self.pause_s = pause_s  # light pacing for free-tier rate limits

    async def _ask(self, system: str, user: str, retries: int = 4) -> str:
        """Call the judge LLM with simple backoff on transient errors."""
        delay = 2.0
        for attempt in range(retries):
            try:
                return await self.llm.complete(system, user)
            except Exception as exc:  # noqa: BLE001 — rate limits etc.
                if attempt == retries - 1:
                    logger.warning("Judge call failed permanently: %s", exc)
                    return "{}"
                await asyncio.sleep(delay)
                delay *= 2
        return "{}"

    # ── individual metrics ────────────────────────────────────────────────
    async def faithfulness(self, answer: str, contexts: list[str]) -> float:
        context = "\n\n".join(contexts)[:3500]
        system = (
            "You are a strict evaluator. Count factual claims in an answer and how "
            "many are directly supported by the context. Reply ONLY with JSON: "
            '{"total": <int>, "supported": <int>}.'
        )
        user = f"CONTEXT:\n{context}\n\nANSWER:\n{answer}"
        data = _parse_json(await self._ask(system, user))
        return _ratio(data.get("supported", 0), data.get("total", 0))

    async def answer_relevancy(self, question: str, answer: str) -> float:
        if "could not find sufficient information" in answer.lower():
            return 0.0
        system = (
            "Generate exactly 3 distinct questions that the given answer directly "
            'answers. Reply ONLY with JSON: {"questions": ["...", "...", "..."]}.'
        )
        data = _parse_json(await self._ask(system, f"ANSWER:\n{answer}"))
        generated = [q for q in data.get("questions", []) if isinstance(q, str)]
        if not generated:
            return 0.0
        q_vec = self.embedder.embed_query(question)
        gen_vecs = self.embedder.embed_batch(generated)
        sims = gen_vecs @ q_vec  # vectors are unit-normalized -> cosine
        return float(max(0.0, min(1.0, float(np.mean(sims)))))

    async def context_recall(self, ground_truth: str, contexts: list[str]) -> float:
        context = "\n\n".join(contexts)[:3500]
        system = (
            "Break the reference answer into factual claims and count how many are "
            "supported by the context. Reply ONLY with JSON: "
            '{"total": <int>, "supported": <int>}.'
        )
        user = f"CONTEXT:\n{context}\n\nREFERENCE ANSWER:\n{ground_truth}"
        data = _parse_json(await self._ask(system, user))
        return _ratio(data.get("supported", 0), data.get("total", 0))

    async def context_precision(
        self, question: str, ground_truth: str, contexts: list[str]
    ) -> float:
        if not contexts:
            return 0.0
        listing = "\n\n".join(
            f"[{i + 1}] {c[:900]}" for i, c in enumerate(contexts)
        )
        system = (
            "For EACH numbered context chunk, decide if it is relevant for answering "
            "the question (given the reference answer). Reply ONLY with JSON: "
            '{"relevant": [<0 or 1 for each chunk, in order>]}.'
        )
        user = f"QUESTION: {question}\nREFERENCE: {ground_truth}\n\nCHUNKS:\n{listing}"
        data = _parse_json(await self._ask(system, user))
        raw = data.get("relevant", [])
        labels = [1 if x in (1, "1", True) else 0 for x in raw][: len(contexts)]
        labels += [0] * (len(contexts) - len(labels))  # pad if judge under-returns

        # Average precision@k: reward relevant chunks ranked earlier.
        hits = 0
        precisions: list[float] = []
        for i, rel in enumerate(labels, start=1):
            if rel:
                hits += 1
                precisions.append(hits / i)
        return float(np.mean(precisions)) if precisions else 0.0

    # ── orchestration ─────────────────────────────────────────────────────
    async def score_sample(self, sample: dict) -> dict:
        question = sample["question"]
        answer = sample["answer"]
        contexts = sample["contexts"]
        ground_truth = sample["ground_truth"]

        faith = await self.faithfulness(answer, contexts)
        await asyncio.sleep(self.pause_s)
        relev = await self.answer_relevancy(question, answer)
        await asyncio.sleep(self.pause_s)
        recall = await self.context_recall(ground_truth, contexts)
        await asyncio.sleep(self.pause_s)
        precision = await self.context_precision(question, ground_truth, contexts)
        await asyncio.sleep(self.pause_s)

        return {
            "question": question,
            "faithfulness": faith,
            "answer_relevancy": relev,
            "context_recall": recall,
            "context_precision": precision,
        }

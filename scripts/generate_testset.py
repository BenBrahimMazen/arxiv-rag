"""Generate a corpus-grounded evaluation set from the indexed papers.

For each indexed paper, an LLM writes one specific question the paper answers
plus a concise ground-truth answer drawn from the paper's own text. This is the
correct RAGAS methodology: the test questions come from the corpus under test.

Run:  python -m scripts.generate_testset --n 20
Writes: src/evaluation/generated_test_set.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re

from sqlalchemy import select

from src.db.models import Chunk, Paper
from src.db.session import init_db, session_scope
from src.generation.llm import get_llm
from src.logging_conf import get_logger

logger = get_logger(__name__)

_OUT_PATH = os.path.join("src", "evaluation", "generated_test_set.json")
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

_SYSTEM = (
    "You write evaluation data for a retrieval-augmented QA system over research "
    "papers. Given a paper's text, produce ONE specific, self-contained question "
    "that the paper answers (do not mention 'the paper' or 'this study'), and a "
    "concise ground-truth answer (1-3 sentences) grounded strictly in the text. "
    'Reply ONLY with JSON: {"question": "...", "ground_truth": "..."}.'
)


def _parse(text: str) -> dict | None:
    m = _JSON_RE.search(text)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
        if d.get("question") and d.get("ground_truth"):
            return {"question": d["question"], "ground_truth": d["ground_truth"]}
    except json.JSONDecodeError:
        return None
    return None


def _paper_sources(n: int) -> list[tuple[str, str]]:
    """Return up to ``n`` (arxiv_id, source_text) for papers that have chunks."""
    out: list[tuple[str, str]] = []
    with session_scope() as session:
        papers = session.scalars(
            select(Paper).where(Paper.chunk_count > 0).limit(n)
        ).all()
        for p in papers:
            chunks = session.scalars(
                select(Chunk.text)
                .where(Chunk.arxiv_id == p.arxiv_id)
                .order_by(Chunk.chunk_index)
                .limit(4)
            ).all()
            source = (p.abstract or "") + "\n\n" + "\n\n".join(chunks)
            out.append((p.arxiv_id, source[:6000]))
    return out


async def _generate(n: int) -> list[dict]:
    init_db()
    llm = get_llm()
    sources = _paper_sources(n)
    logger.info("Generating Q&A from %d papers", len(sources))
    items: list[dict] = []
    for arxiv_id, source in sources:
        try:
            raw = await llm.complete(_SYSTEM, f"PAPER TEXT:\n{source}")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Generation failed for %s: %s", arxiv_id, exc)
            continue
        qa = _parse(raw)
        if qa:
            items.append(qa)
            logger.info("[%d] %s -> %s", len(items), arxiv_id, qa["question"][:60])
        await asyncio.sleep(0.6)  # pace for free-tier limits
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a corpus-grounded test set")
    parser.add_argument("--n", type=int, default=20)
    args = parser.parse_args()

    items = asyncio.run(_generate(args.n))
    os.makedirs(os.path.dirname(_OUT_PATH), exist_ok=True)
    with open(_OUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(items, fh, indent=2, ensure_ascii=False)
    logger.info("Wrote %d Q&A pairs to %s", len(items), _OUT_PATH)


if __name__ == "__main__":
    main()

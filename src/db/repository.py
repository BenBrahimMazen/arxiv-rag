"""Reusable read/write helpers over the ORM models."""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.db.models import Chunk, Paper, QueryLog
from src.retrieval.hybrid_search import ChunkRecord
from src.types import ArxivPaper


def upsert_paper(
    session: Session, paper: ArxivPaper, pdf_path: str | None = None
) -> None:
    """Insert or update a paper's metadata (idempotent for resumable runs)."""
    existing = session.get(Paper, paper.arxiv_id)
    if existing is None:
        session.add(
            Paper(
                arxiv_id=paper.arxiv_id,
                title=paper.title,
                authors=paper.authors,
                abstract=paper.abstract,
                published_date=paper.published_date,
                categories=paper.categories,
                pdf_url=paper.pdf_url,
                pdf_path=pdf_path,
            )
        )
    elif pdf_path:
        existing.pdf_path = pdf_path


def save_chunks(
    session: Session, chunks: list[Chunk], embedding_model: str
) -> None:
    """Persist chunk rows and update the parent paper's chunk count/model."""
    if not chunks:
        return
    arxiv_id = chunks[0].arxiv_id
    session.add_all(
        Chunk(
            chunk_id=c.chunk_id,
            arxiv_id=c.arxiv_id,
            section=c.section,
            text=c.text,
            token_count=c.token_count,
            chunk_index=c.chunk_index,
        )
        for c in chunks
    )
    paper = session.get(Paper, arxiv_id)
    if paper is not None:
        paper.chunk_count = len(chunks)
        paper.embedding_model = embedding_model


def load_bm25_corpus(session: Session) -> list[ChunkRecord]:
    """Load every chunk as a BM25 corpus record."""
    rows = session.execute(
        select(Chunk.chunk_id, Chunk.arxiv_id, Chunk.section, Chunk.text)
    ).all()
    return [
        ChunkRecord(chunk_id=r[0], arxiv_id=r[1], section=r[2], text=r[3])
        for r in rows
    ]


def paper_title_url(session: Session, arxiv_id: str) -> tuple[str, str]:
    """Return ``(title, abs_url)`` for a paper id, with a safe fallback."""
    paper = session.get(Paper, arxiv_id)
    url = f"https://arxiv.org/abs/{arxiv_id}"
    if paper is None:
        return ("", url)
    return (paper.title, url)


def log_query(
    session: Session,
    question: str,
    answer: str,
    sources: list[str],
    latency_ms: int,
) -> None:
    session.add(
        QueryLog(
            question=question,
            answer=answer,
            sources=sources,
            latency_ms=latency_ms,
        )
    )


def counts(session: Session) -> tuple[int, int]:
    """Return ``(paper_count, chunk_count)``."""
    from sqlalchemy import func

    papers = session.scalar(select(func.count()).select_from(Paper)) or 0
    chunks = session.scalar(select(func.count()).select_from(Chunk)) or 0
    return int(papers), int(chunks)

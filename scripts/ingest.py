"""End-to-end ingestion: ArXiv fetch -> PDF parse -> chunk -> embed -> store.

Run:  python -m scripts.ingest --max-papers 50

Metadata is written to Postgres before PDFs are downloaded so interrupted runs
can be resumed (already-ingested papers/chunks are skipped).
"""
from __future__ import annotations

import argparse

from src.config import get_settings
from src.db.models import Chunk as ChunkRow
from src.db.models import Paper
from src.db.repository import save_chunks, upsert_paper
from src.db.session import init_db, session_scope
from src.embeddings.embedder import Embedder
from src.embeddings.vector_store import get_vector_store
from src.ingestion.arxiv_client import ArxivClient
from src.ingestion.chunker import SemanticChunker
from src.ingestion.pdf_parser import PDFParser
from src.logging_conf import get_logger

logger = get_logger(__name__)
RAW_DIR = "data/raw"


def _already_chunked(session, arxiv_id: str) -> bool:
    paper = session.get(Paper, arxiv_id)
    return paper is not None and paper.chunk_count > 0


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Run the ArXiv ingestion pipeline")
    parser.add_argument("--max-papers", type=int, default=settings.max_papers)
    parser.add_argument("--category", default=settings.arxiv_category)
    parser.add_argument("--start-date", default=settings.arxiv_start_date)
    parser.add_argument("--end-date", default=settings.arxiv_end_date)
    parser.add_argument("--query", default="")
    args = parser.parse_args()

    init_db()
    client = ArxivClient()
    pdf_parser = PDFParser()
    chunker = SemanticChunker(
        chunk_size=settings.chunk_size, overlap=settings.chunk_overlap
    )
    embedder = Embedder()
    vector_store = get_vector_store()

    logger.info("Fetching up to %d papers from %s", args.max_papers, args.category)
    papers = client.search(
        query=args.query,
        category=args.category,
        max_results=args.max_papers,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    # Persist all metadata first (resumability).
    with session_scope() as session:
        for paper in papers:
            upsert_paper(session, paper)
    logger.info("Stored metadata for %d papers", len(papers))

    processed = 0
    for paper in papers:
        with session_scope() as session:
            if _already_chunked(session, paper.arxiv_id):
                logger.info("Skipping already-chunked %s", paper.arxiv_id)
                continue
        try:
            pdf_path = client.download_pdf(paper, RAW_DIR)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Download failed for %s: %s", paper.arxiv_id, exc)
            continue

        parsed = pdf_parser.extract_text(pdf_path)
        parsed.arxiv_id = paper.arxiv_id  # ensure canonical id
        if not parsed.extraction_success:
            logger.warning("Extraction failed for %s; skipping", paper.arxiv_id)
            continue

        chunks = chunker.chunk_paper(parsed)
        if not chunks:
            logger.warning("No chunks produced for %s", paper.arxiv_id)
            continue

        embeddings = embedder.embed_batch([c.text for c in chunks])
        vector_store.upsert(chunks, embeddings)

        with session_scope() as session:
            upsert_paper(session, paper, pdf_path=pdf_path)
            # Avoid duplicate chunk rows if partially ingested before.
            session.query(ChunkRow).filter(
                ChunkRow.arxiv_id == paper.arxiv_id
            ).delete()
            save_chunks(session, chunks, embedding_model=embedder.model_name)

        processed += 1
        logger.info(
            "[%d/%d] %s -> %d chunks", processed, len(papers),
            paper.arxiv_id, len(chunks)
        )

    logger.info(
        "Ingestion complete: %d papers chunked, %d vectors in store",
        processed,
        vector_store.count(),
    )


if __name__ == "__main__":
    main()

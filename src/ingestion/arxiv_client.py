"""Fetch paper metadata and PDFs from the ArXiv API."""
from __future__ import annotations

import os
from datetime import date, datetime

import arxiv
import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.logging_conf import get_logger
from src.types import ArxivPaper

logger = get_logger(__name__)


def _to_date(value: datetime | date) -> date:
    return value.date() if isinstance(value, datetime) else value


def _yyyymmdd(iso_date: str) -> str:
    """Convert ``YYYY-MM-DD`` to the ArXiv ``YYYYMMDD`` query format."""
    return iso_date.replace("-", "")


class ArxivClient:
    """Thin wrapper over the ``arxiv`` library with retries and date filtering."""

    def __init__(self, page_size: int = 100, delay_seconds: float = 3.0) -> None:
        # ArXiv asks for >=3s between requests; the client enforces this.
        self._client = arxiv.Client(
            page_size=page_size, delay_seconds=delay_seconds, num_retries=3
        )

    def search(
        self,
        query: str = "",
        category: str = "cs.LG",
        max_results: int = 500,
        start_date: str = "2023-01-01",
        end_date: str = "2024-12-31",
    ) -> list[ArxivPaper]:
        """Search ArXiv, filtering by category and submission-date window.

        ``query`` is an optional free-text term ANDed with the category filter.
        """
        date_clause = (
            f"submittedDate:[{_yyyymmdd(start_date)} TO {_yyyymmdd(end_date)}]"
        )
        terms = [f"cat:{category}", date_clause]
        if query:
            terms.append(f"all:{query}")
        search_query = " AND ".join(terms)
        logger.info("ArXiv search: %s (max_results=%d)", search_query, max_results)

        search = arxiv.Search(
            query=search_query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
        )

        papers: list[ArxivPaper] = []
        for result in self._client.results(search):
            papers.append(
                ArxivPaper(
                    arxiv_id=result.get_short_id(),
                    title=result.title.strip(),
                    authors=[a.name for a in result.authors],
                    abstract=result.summary.strip(),
                    published_date=_to_date(result.published),
                    categories=list(result.categories),
                    pdf_url=result.pdf_url,
                )
            )
        logger.info("Fetched %d papers from ArXiv", len(papers))
        return papers

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        reraise=True,
    )
    def download_pdf(self, paper: ArxivPaper, save_dir: str) -> str:
        """Download ``paper``'s PDF into ``save_dir`` and return the local path.

        Retries with exponential backoff (max 3 attempts) on failure.
        """
        os.makedirs(save_dir, exist_ok=True)
        filename = f"{paper.arxiv_id.replace('/', '_')}.pdf"
        target = os.path.join(save_dir, filename)
        if os.path.exists(target) and os.path.getsize(target) > 0:
            logger.debug("PDF already present, skipping: %s", target)
            return target

        # Download the PDF directly over HTTP. This avoids depending on the
        # arxiv library's downloader, whose API differs across major versions.
        headers = {"User-Agent": "arxiv-rag/1.0 (academic research assistant)"}
        with requests.get(
            paper.pdf_url, headers=headers, stream=True, timeout=60
        ) as resp:
            resp.raise_for_status()
            with open(target, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=8192):
                    fh.write(chunk)
        logger.info("Downloaded %s", target)
        return target

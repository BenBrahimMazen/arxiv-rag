"""Extract text and section structure from ArXiv PDFs using pdfplumber."""
from __future__ import annotations

import os
import re

import pdfplumber

from src.logging_conf import get_logger
from src.types import ParsedPaper

logger = get_logger(__name__)

# Canonical section names mapped to the heading patterns that introduce them.
# Matches headings like "1 Introduction", "2. Related Work", "ABSTRACT".
_SECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("Abstract", re.compile(r"^\s*(?:\d+\.?\s+)?abstract\b", re.IGNORECASE)),
    ("Introduction", re.compile(r"^\s*(?:\d+\.?\s+)?introduction\b", re.IGNORECASE)),
    (
        "Related Work",
        re.compile(r"^\s*(?:\d+\.?\s+)?related\s+work\b", re.IGNORECASE),
    ),
    (
        "Methodology",
        re.compile(
            r"^\s*(?:\d+\.?\s+)?(?:method(?:ology|s)?|approach|model)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "Experiments",
        re.compile(
            r"^\s*(?:\d+\.?\s+)?(?:experiment(?:s|al setup)?|setup)\b", re.IGNORECASE
        ),
    ),
    ("Results", re.compile(r"^\s*(?:\d+\.?\s+)?results?\b", re.IGNORECASE)),
    (
        "Conclusion",
        re.compile(
            r"^\s*(?:\d+\.?\s+)?(?:conclusion(?:s)?|discussion)\b", re.IGNORECASE
        ),
    ),
    ("References", re.compile(r"^\s*(?:\d+\.?\s+)?references?\b", re.IGNORECASE)),
]


class PDFParser:
    """Parse a PDF into full text plus a best-effort section map."""

    def extract_text(self, pdf_path: str) -> ParsedPaper:
        """Extract text page by page; never raises on a bad PDF.

        On any failure (scanned/encrypted/corrupt PDF) returns a ParsedPaper
        with ``extraction_success=False`` and logs a warning.
        """
        arxiv_id = os.path.splitext(os.path.basename(pdf_path))[0].replace("_", "/")
        try:
            pages: list[str] = []
            with pdfplumber.open(pdf_path) as pdf:
                page_count = len(pdf.pages)
                for page in pdf.pages:
                    pages.append(page.extract_text() or "")
        except Exception as exc:  # noqa: BLE001 — must degrade gracefully
            logger.warning("Failed to parse %s: %s", pdf_path, exc)
            return ParsedPaper(
                arxiv_id=arxiv_id,
                full_text="",
                sections={},
                page_count=0,
                extraction_success=False,
            )

        full_text = "\n".join(pages).strip()
        if not full_text:
            logger.warning("No extractable text (likely scanned): %s", pdf_path)
            return ParsedPaper(
                arxiv_id=arxiv_id,
                full_text="",
                sections={},
                page_count=page_count,
                extraction_success=False,
            )

        sections = self._split_sections(full_text)
        return ParsedPaper(
            arxiv_id=arxiv_id,
            full_text=full_text,
            sections=sections,
            page_count=page_count,
            extraction_success=True,
        )

    def _split_sections(self, full_text: str) -> dict[str, str]:
        """Group lines into canonical sections by detecting heading lines."""
        sections: dict[str, list[str]] = {}
        current = "Body"  # text before the first recognised heading
        for line in full_text.splitlines():
            matched = self._match_heading(line)
            if matched is not None:
                current = matched
                sections.setdefault(current, [])
                continue
            sections.setdefault(current, []).append(line)
        return {
            name: "\n".join(lines).strip()
            for name, lines in sections.items()
            if "".join(lines).strip()
        }

    @staticmethod
    def _match_heading(line: str) -> str | None:
        stripped = line.strip()
        # Headings are short lines; skip long prose to avoid false positives.
        if not stripped or len(stripped) > 60:
            return None
        for name, pattern in _SECTION_PATTERNS:
            if pattern.match(stripped):
                return name
        return None

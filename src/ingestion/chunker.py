"""Section-aware, token-budgeted semantic chunking."""
from __future__ import annotations

import uuid

import tiktoken

from src.logging_conf import get_logger
from src.types import Chunk, ParsedPaper

logger = get_logger(__name__)


def _build_sentencizer():
    """Return a lightweight spaCy sentence splitter.

    Uses a blank English pipeline with the rule-based ``sentencizer`` so no
    model download is required (keeps the project free and CI-friendly).
    """
    import spacy

    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    # Academic pages can be long; raise the cap to avoid spaCy length errors.
    nlp.max_length = 2_000_000
    return nlp


class SemanticChunker:
    """Split papers into overlapping, sentence-aligned, section-bounded chunks."""

    def __init__(self, chunk_size: int = 512, overlap: int = 50) -> None:
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap
        self._encoding = tiktoken.get_encoding("cl100k_base")
        self._nlp = _build_sentencizer()

    def _count_tokens(self, text: str) -> int:
        return len(self._encoding.encode(text))

    def _sentences(self, text: str) -> list[str]:
        doc = self._nlp(text)
        return [s.text.strip() for s in doc.sents if s.text.strip()]

    def chunk_paper(self, paper: ParsedPaper) -> list[Chunk]:
        """Chunk a parsed paper, never splitting across detected sections."""
        sections = paper.sections or {"Body": paper.full_text}
        chunks: list[Chunk] = []
        index = 0
        for section_name, section_text in sections.items():
            if not section_text.strip():
                continue
            for text in self._chunk_section(section_text):
                chunks.append(
                    Chunk(
                        chunk_id=str(uuid.uuid4()),
                        arxiv_id=paper.arxiv_id,
                        section=section_name,
                        text=text,
                        token_count=self._count_tokens(text),
                        chunk_index=index,
                    )
                )
                index += 1
        logger.debug("Chunked %s into %d chunks", paper.arxiv_id, len(chunks))
        return chunks

    def _chunk_section(self, section_text: str) -> list[str]:
        """Greedily pack sentences into ~chunk_size token windows with overlap."""
        sentences = self._sentences(section_text)
        if not sentences:
            return []

        chunks: list[str] = []
        current: list[str] = []
        current_tokens = 0

        for sentence in sentences:
            sent_tokens = self._count_tokens(sentence)

            # A single oversized sentence becomes its own chunk.
            if sent_tokens >= self.chunk_size:
                if current:
                    chunks.append(" ".join(current))
                    current, current_tokens = [], 0
                chunks.append(sentence)
                continue

            if current_tokens + sent_tokens > self.chunk_size and current:
                chunks.append(" ".join(current))
                current, current_tokens = self._carry_overlap(current)

            current.append(sentence)
            current_tokens += sent_tokens

        if current:
            chunks.append(" ".join(current))
        return chunks

    def _carry_overlap(self, sentences: list[str]) -> tuple[list[str], int]:
        """Return the trailing sentences (within ``overlap`` tokens) to repeat."""
        carried: list[str] = []
        tokens = 0
        for sentence in reversed(sentences):
            sent_tokens = self._count_tokens(sentence)
            if tokens + sent_tokens > self.overlap:
                break
            carried.insert(0, sentence)
            tokens += sent_tokens
        return carried, tokens

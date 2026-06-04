"""System and user prompt templates for the RAG chain."""
from __future__ import annotations

from src.types import SearchResult

SYSTEM_PROMPT = """\
You are a research assistant specializing in machine learning papers from ArXiv.
Answer questions using ONLY the provided context passages.
Every factual claim must be followed by a citation in the format [arxiv:XXXX.XXXXX].
If the context does not contain enough information to answer the question,
say "I could not find sufficient information in the retrieved papers."
Do not hallucinate paper titles, authors, or results not present in the context.
Structure your answers clearly: start with a direct answer, then elaborate with evidence.\
"""


def build_user_prompt(query: str, context_chunks: list[SearchResult]) -> str:
    """Format retrieved chunks into a numbered, cited context block + question."""
    blocks: list[str] = []
    for i, chunk in enumerate(context_chunks, start=1):
        blocks.append(
            f"[{i}] arxiv:{chunk.arxiv_id} (section: {chunk.section})\n{chunk.text}"
        )
    context = "\n\n".join(blocks) if blocks else "(no context retrieved)"
    return f"{context}\n\nQuestion: {query}"

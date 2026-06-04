"""Pydantic request/response models for the API."""
from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=10)


class SourceModel(BaseModel):
    arxiv_id: str
    title: str
    url: str


class ChunkModel(BaseModel):
    chunk_id: str
    arxiv_id: str
    section: str
    text: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceModel]
    latency_ms: int
    retrieved_chunks: list[ChunkModel] = []


class PaperModel(BaseModel):
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    published_date: date | None = None
    categories: list[str]
    pdf_url: str
    chunk_count: int = 0


class PaperListResponse(BaseModel):
    total: int
    limit: int
    offset: int
    papers: list[PaperModel]


class HealthResponse(BaseModel):
    status: str
    papers_indexed: int
    chunks_indexed: int

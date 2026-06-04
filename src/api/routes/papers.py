"""Paper listing and detail endpoints."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import func, select

from src.api.schemas import PaperListResponse, PaperModel
from src.db.models import Paper
from src.db.session import session_scope

router = APIRouter(tags=["papers"])


def _to_model(paper: Paper) -> PaperModel:
    return PaperModel(
        arxiv_id=paper.arxiv_id,
        title=paper.title,
        authors=paper.authors or [],
        abstract=paper.abstract,
        published_date=paper.published_date,
        categories=paper.categories or [],
        pdf_url=paper.pdf_url,
        chunk_count=paper.chunk_count,
    )


@router.get("/papers", response_model=PaperListResponse)
async def list_papers(
    query: str | None = Query(default=None, description="Substring match on title"),
    category: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> PaperListResponse:
    """Return a paginated, filterable list of indexed papers."""
    stmt = select(Paper)
    if query:
        stmt = stmt.where(Paper.title.ilike(f"%{query}%"))
    if category:
        stmt = stmt.where(Paper.categories.contains([category]))
    if start_date:
        stmt = stmt.where(Paper.published_date >= start_date)
    if end_date:
        stmt = stmt.where(Paper.published_date <= end_date)

    with session_scope() as session:
        total = session.scalar(
            select(func.count()).select_from(stmt.subquery())
        ) or 0
        rows = session.scalars(
            stmt.order_by(Paper.published_date.desc()).limit(limit).offset(offset)
        ).all()
        papers = [_to_model(p) for p in rows]

    return PaperListResponse(
        total=int(total), limit=limit, offset=offset, papers=papers
    )


@router.get("/papers/{arxiv_id:path}", response_model=PaperModel)
async def get_paper(arxiv_id: str) -> PaperModel:
    """Return a single paper's metadata plus chunk count."""
    with session_scope() as session:
        paper = session.get(Paper, arxiv_id)
        if paper is None:
            raise HTTPException(status_code=404, detail="Paper not found")
        return _to_model(paper)

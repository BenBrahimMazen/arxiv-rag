"""Query endpoints: synchronous JSON and SSE streaming."""
from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from src.api.deps import get_state
from src.api.schemas import ChunkModel, QueryRequest, QueryResponse, SourceModel
from src.db.repository import log_query
from src.db.session import session_scope

router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """Answer a question with cited sources."""
    state = get_state()
    state.chain.top_n = request.top_k
    response = await state.chain.query(request.question)

    with session_scope() as session:
        log_query(
            session,
            question=request.question,
            answer=response.answer,
            sources=[s.arxiv_id for s in response.sources],
            latency_ms=response.latency_ms,
        )

    return QueryResponse(
        answer=response.answer,
        sources=[SourceModel(**s.__dict__) for s in response.sources],
        latency_ms=response.latency_ms,
        retrieved_chunks=[
            ChunkModel(
                chunk_id=c.chunk_id,
                arxiv_id=c.arxiv_id,
                section=c.section,
                text=c.text,
                score=c.score,
            )
            for c in response.retrieved_chunks
        ],
    )


@router.get("/query/stream")
async def query_stream(question: str) -> StreamingResponse:
    """Stream the answer token-by-token as Server-Sent Events."""
    state = get_state()
    generator = state.chain.stream_query(question)
    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

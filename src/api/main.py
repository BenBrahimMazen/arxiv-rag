"""FastAPI application entrypoint."""
from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.deps import build_state, get_state, set_state
from src.api.routes import papers, query
from src.api.schemas import HealthResponse
from src.config import get_settings
from src.db.repository import counts
from src.db.session import session_scope
from src.logging_conf import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up components on startup."""
    logger.info("Starting API: initializing components...")
    set_state(build_state())
    logger.info("API ready")
    yield


app = FastAPI(
    title="ArXiv Research Assistant",
    description="RAG-powered Q&A over ArXiv ML papers",
    version="1.0.0",
    lifespan=lifespan,
)

settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = (time.perf_counter() - start) * 1000
    logger.info("%s %s -> %d (%.0f ms)", request.method, request.url.path,
                response.status_code, elapsed)
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s: %s", request.url.path, exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    # Ensure state is initialized (raises clearly if not).
    get_state()
    with session_scope() as session:
        paper_count, chunk_count = counts(session)
    return HealthResponse(
        status="ok", papers_indexed=paper_count, chunks_indexed=chunk_count
    )


app.include_router(query.router)
app.include_router(papers.router)

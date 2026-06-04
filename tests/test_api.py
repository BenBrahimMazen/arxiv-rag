"""API tests with the chain and DB stubbed (no models, no Postgres)."""
from __future__ import annotations

from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient

from src.types import RAGResponse, SearchResult, SourcePaper


class FakeChain:
    top_n = 5

    async def query(self, question: str) -> RAGResponse:
        return RAGResponse(
            answer=f"Answer to: {question} [arxiv:2301.00001]",
            sources=[
                SourcePaper(
                    arxiv_id="2301.00001",
                    title="A Great Paper",
                    url="https://arxiv.org/abs/2301.00001",
                )
            ],
            retrieved_chunks=[
                SearchResult(
                    chunk_id="c1", arxiv_id="2301.00001", section="Body",
                    text="evidence", score=0.9,
                )
            ],
            latency_ms=12,
        )

    async def stream_query(self, question: str):
        for token in ["Hello ", "world "]:
            yield f'data: {{"type": "token", "data": "{token}"}}\n\n'
        yield 'data: {"type": "sources", "data": []}\n\n'
        yield "data: [DONE]\n\n"


@contextmanager
def _fake_session_scope():
    yield None


@pytest.fixture()
def client(monkeypatch):
    from src.api import deps, main
    from src.api.routes import query as query_route

    fake_state = deps.AppState(chain=FakeChain(), embedder=None, searcher=None)
    monkeypatch.setattr(main, "build_state", lambda: fake_state)
    monkeypatch.setattr(main, "session_scope", _fake_session_scope)
    monkeypatch.setattr(main, "counts", lambda session: (2, 3))
    monkeypatch.setattr(query_route, "session_scope", _fake_session_scope)
    monkeypatch.setattr(query_route, "log_query", lambda *a, **k: None)

    with TestClient(main.app) as test_client:
        yield test_client


def test_health(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["papers_indexed"] == 2
    assert body["chunks_indexed"] == 3


def test_query(client: TestClient) -> None:
    resp = client.post("/query", json={"question": "What is attention?", "top_k": 3})
    assert resp.status_code == 200
    body = resp.json()
    assert "attention" in body["answer"]
    assert body["sources"][0]["arxiv_id"] == "2301.00001"
    assert body["latency_ms"] == 12


def test_query_validation(client: TestClient) -> None:
    resp = client.post("/query", json={"question": "", "top_k": 3})
    assert resp.status_code == 422


def test_query_stream(client: TestClient) -> None:
    resp = client.get("/query/stream", params={"question": "hi"})
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
    assert "Hello" in resp.text
    assert "[DONE]" in resp.text

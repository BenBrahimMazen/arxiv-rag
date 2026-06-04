"""Test configuration: force free/local/offline backends before imports."""
from __future__ import annotations

import os

# Default every backend to something that needs no API key or network.
os.environ.setdefault("EMBEDDING_BACKEND", "local")
os.environ.setdefault("RERANKER_BACKEND", "local")
os.environ.setdefault("LLM_BACKEND", "echo")
os.environ.setdefault("VECTOR_BACKEND", "chroma")
os.environ.setdefault("POSTGRES_URL", "sqlite+pysqlite:///:memory:")

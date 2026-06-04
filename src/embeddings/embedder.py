"""Embedding backends: OpenAI (paid) or sentence-transformers (free local)."""
from __future__ import annotations

import numpy as np

from src.config import Settings, get_settings
from src.logging_conf import get_logger

logger = get_logger(__name__)

# text-embedding-3-small price (USD per 1M tokens), for cost logging only.
_OPENAI_PRICE_PER_1M = 0.02


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Normalize each row to unit length (safe against zero vectors)."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


class Embedder:
    """Create unit-normalized embeddings from either backend.

    Backend is chosen by ``EMBEDDING_BACKEND`` (``local`` or ``openai``).
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.backend = self.settings.embedding_backend
        self._model = None
        self._openai = None
        self._tiktoken = None
        if self.backend == "openai":
            self._init_openai()
        else:
            self._init_local()

    # ── backend init ──────────────────────────────────────────────────────
    def _init_openai(self) -> None:
        from openai import OpenAI

        if not self.settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY required for openai embedding backend")
        self._openai = OpenAI(api_key=self.settings.openai_api_key)
        import tiktoken

        self._tiktoken = tiktoken.get_encoding("cl100k_base")
        self.model_name = "text-embedding-3-small"
        self.dim = self.settings.openai_embedding_dim

    def _init_local(self) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self._model = SentenceTransformer(self.model_name)
        self.dim = self.settings.local_embedding_dim

    # ── public API ────────────────────────────────────────────────────────
    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Return a ``(len(texts), dim)`` float32 array of unit vectors."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        if self.backend == "openai":
            vectors = self._embed_openai(texts)
        else:
            vectors = self._embed_local(texts)
        return _l2_normalize(vectors.astype(np.float32))

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query and return a 1-D ``(dim,)`` vector."""
        return self.embed_batch([text])[0]

    # ── implementations ───────────────────────────────────────────────────
    def _embed_openai(self, texts: list[str], batch_size: int = 100) -> np.ndarray:
        out: list[list[float]] = []
        total_tokens = 0
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            total_tokens += sum(len(self._tiktoken.encode(t)) for t in batch)
            resp = self._openai.embeddings.create(
                model=self.model_name, input=batch
            )
            out.extend(item.embedding for item in resp.data)
        cost = total_tokens / 1_000_000 * _OPENAI_PRICE_PER_1M
        logger.info(
            "OpenAI embeddings: %d texts, ~%d tokens, est. $%.5f",
            len(texts),
            total_tokens,
            cost,
        )
        return np.array(out, dtype=np.float32)

    def _embed_local(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            self._model.encode(
                texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True
            ),
            dtype=np.float32,
        )

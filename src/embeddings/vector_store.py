"""Vector store abstraction with Chroma (local) and Pinecone (prod) backends."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from src.config import Settings, get_settings
from src.logging_conf import get_logger
from src.types import Chunk, SearchResult

logger = get_logger(__name__)

_COLLECTION = "arxiv_rag"


class VectorStore(ABC):
    """Common interface for vector stores."""

    @abstractmethod
    def upsert(self, chunks: list[Chunk], embeddings: np.ndarray) -> None: ...

    @abstractmethod
    def query(
        self, query_embedding: np.ndarray, top_k: int = 20
    ) -> list[SearchResult]: ...

    @abstractmethod
    def delete_collection(self) -> None: ...

    @abstractmethod
    def count(self) -> int: ...

    @staticmethod
    def _metadata(chunk: Chunk) -> dict:
        return {
            "arxiv_id": chunk.arxiv_id,
            "section": chunk.section,
            "chunk_index": chunk.chunk_index,
            "text": chunk.text,
        }


class ChromaStore(VectorStore):
    """Local, persistent Chroma collection."""

    def __init__(self, settings: Settings | None = None) -> None:
        import chromadb

        self.settings = settings or get_settings()
        self._client = chromadb.PersistentClient(path=self.settings.chroma_path)
        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION, metadata={"hnsw:space": "cosine"}
        )

    def upsert(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        if not chunks:
            return
        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=[e.tolist() for e in embeddings],
            documents=[c.text for c in chunks],
            metadatas=[self._metadata(c) for c in chunks],
        )
        logger.info("Chroma upsert: %d chunks", len(chunks))

    def query(
        self, query_embedding: np.ndarray, top_k: int = 20
    ) -> list[SearchResult]:
        res = self._collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            include=["metadatas", "documents", "distances"],
        )
        ids = res["ids"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]
        results: list[SearchResult] = []
        for chunk_id, meta, dist in zip(ids, metas, dists, strict=False):
            results.append(
                SearchResult(
                    chunk_id=chunk_id,
                    arxiv_id=meta["arxiv_id"],
                    section=meta["section"],
                    text=meta["text"],
                    score=1.0 - float(dist),  # cosine distance -> similarity
                )
            )
        return results

    def delete_collection(self) -> None:
        self._client.delete_collection(_COLLECTION)
        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION, metadata={"hnsw:space": "cosine"}
        )

    def count(self) -> int:
        return self._collection.count()


class PineconeStore(VectorStore):
    """Production Pinecone index, single index + namespace."""

    def __init__(self, settings: Settings | None = None) -> None:
        from pinecone import Pinecone, ServerlessSpec

        self.settings = settings or get_settings()
        if not self.settings.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY required for pinecone backend")
        self._pc = Pinecone(api_key=self.settings.pinecone_api_key)
        self._index_name = self.settings.pinecone_index
        self._namespace = self.settings.pinecone_namespace

        existing = {idx["name"] for idx in self._pc.list_indexes()}
        if self._index_name not in existing:
            self._pc.create_index(
                name=self._index_name,
                dimension=self.settings.embedding_dim,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=self.settings.pinecone_cloud,
                    region=self.settings.pinecone_region,
                ),
            )
        self._index = self._pc.Index(self._index_name)

    def upsert(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        if not chunks:
            return
        vectors = [
            {
                "id": c.chunk_id,
                "values": e.tolist(),
                "metadata": self._metadata(c),
            }
            for c, e in zip(chunks, embeddings, strict=False)
        ]
        for start in range(0, len(vectors), 100):
            self._index.upsert(
                vectors=vectors[start : start + 100], namespace=self._namespace
            )
        logger.info("Pinecone upsert: %d chunks", len(chunks))

    def query(
        self, query_embedding: np.ndarray, top_k: int = 20
    ) -> list[SearchResult]:
        res = self._index.query(
            vector=query_embedding.tolist(),
            top_k=top_k,
            namespace=self._namespace,
            include_metadata=True,
        )
        results: list[SearchResult] = []
        for match in res["matches"]:
            meta = match["metadata"]
            results.append(
                SearchResult(
                    chunk_id=match["id"],
                    arxiv_id=meta["arxiv_id"],
                    section=meta["section"],
                    text=meta["text"],
                    score=float(match["score"]),
                )
            )
        return results

    def delete_collection(self) -> None:
        self._index.delete(delete_all=True, namespace=self._namespace)

    def count(self) -> int:
        stats = self._index.describe_index_stats()
        ns = stats.get("namespaces", {}).get(self._namespace, {})
        return int(ns.get("vector_count", 0))


def get_vector_store(settings: Settings | None = None) -> VectorStore:
    """Factory returning the configured vector store implementation."""
    settings = settings or get_settings()
    if settings.vector_backend == "pinecone":
        return PineconeStore(settings)
    return ChromaStore(settings)

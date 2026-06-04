"""Central, typed configuration loaded from environment / .env.

Everything that varies between local dev, CI and production lives here so the
rest of the codebase can stay free of ``os.getenv`` calls.
"""
from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings, sourced from environment variables / ``.env``."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Database
    postgres_url: str = Field(
        default="postgresql+psycopg2://arxiv:arxiv@localhost:5432/arxiv_rag"
    )

    # OpenAI
    openai_api_key: str | None = Field(default=None)

    # Vector store
    vector_backend: str = Field(default="chroma")  # chroma | pinecone
    chroma_path: str = Field(default="./chroma_db")
    pinecone_api_key: str | None = Field(default=None)
    pinecone_index: str = Field(default="arxiv-rag")
    pinecone_namespace: str = Field(default="cs-lg")
    pinecone_cloud: str = Field(default="aws")
    pinecone_region: str = Field(default="us-east-1")

    # Embeddings
    embedding_backend: str = Field(default="local")  # local | openai

    # Reranker
    reranker_backend: str = Field(default="local")  # local | cohere
    cohere_api_key: str | None = Field(default=None)

    # Generation
    llm_backend: str = Field(default="openai")  # openai | echo
    llm_model: str = Field(default="gpt-4o-mini")

    # Ingestion
    max_papers: int = Field(default=500)
    arxiv_category: str = Field(default="cs.LG")
    arxiv_start_date: str = Field(default="2023-01-01")
    arxiv_end_date: str = Field(default="2024-12-31")
    chunk_size: int = Field(default=512)
    chunk_overlap: int = Field(default=50)

    # Misc
    log_level: str = Field(default="INFO")
    api_url: str = Field(default="http://localhost:8000")

    @property
    def openai_embedding_dim(self) -> int:
        return 1536

    @property
    def local_embedding_dim(self) -> int:
        return 384

    @property
    def embedding_dim(self) -> int:
        return (
            self.openai_embedding_dim
            if self.embedding_backend == "openai"
            else self.local_embedding_dim
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton ``Settings`` instance."""
    return Settings()

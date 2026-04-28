"""
config.py — Centralized configuration using Pydantic BaseSettings.
All settings loaded from environment variables or .env file.
"""
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # App
    app_name: str = "LegalMind RAG"
    app_version: str = "1.0.0"
    environment: Literal["development", "staging", "production"] = "development"
    debug: bool = False
    log_level: str = "INFO"

    # LLM Provider — selects which backend the app calls
    llm_provider: Literal["openai", "gemini"] = "openai"

    # OpenAI (used when llm_provider == "openai")
    openai_api_key: str = Field(default="", description="OpenAI API key")
    embedding_model: str = "text-embedding-ada-002"
    embedding_dimensions: int = 1536
    llm_model: str = "gpt-4o"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 2048

    # Gemini (used when llm_provider == "gemini") — accessed through the
    # Google OpenAI-compatible endpoint so we can reuse the OpenAI SDK
    gemini_api_key: str = Field(default="", description="Gemini API key")
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    gemini_model: str = "gemini-2.0-flash-lite"
    gemini_vision_model: str = "gemini-2.0-flash-lite"
    gemini_embedding_model: str = "gemini-embedding-001"
    gemini_embedding_dimensions: int = 3072

    @property
    def active_llm_model(self) -> str:
        return self.gemini_model if self.llm_provider == "gemini" else self.llm_model

    @property
    def active_embedding_model(self) -> str:
        return (
            self.gemini_embedding_model
            if self.llm_provider == "gemini"
            else self.embedding_model
        )

    @property
    def active_embedding_dimensions(self) -> int:
        return (
            self.gemini_embedding_dimensions
            if self.llm_provider == "gemini"
            else self.embedding_dimensions
        )

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 64
    min_chunk_length: int = 50  # discard tiny chunks

    # Retrieval
    top_k: int = 5
    similarity_threshold: float = 0.70  # min cosine similarity to include
    reranker_enabled: bool = False

    # Vector Store
    vector_store_backend: Literal["chroma", "pinecone"] = "chroma"
    chroma_path: Path = Path("./data/vectorstore")
    chroma_collection_name: str = "legalMind_docs"
    pinecone_api_key: str | None = None
    pinecone_index_name: str = "legalMind"
    pinecone_environment: str = "us-east-1-aws"

    # PostgreSQL
    postgres_url: str = "postgresql+asyncpg://user:pass@localhost:5432/legalMind"
    postgres_pool_size: int = 10
    postgres_max_overflow: int = 20

    # Redis Cache
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = 3600  # 1 hour
    cache_enabled: bool = True

    # Storage
    raw_docs_path: Path = Path("./data/raw")
    processed_docs_path: Path = Path("./data/processed")
    s3_bucket: str | None = None
    s3_prefix: str = "legalMind/documents/"

    # Security
    secret_key: str = Field(
        default="change-me-in-production-change-me-in-production",
        min_length=32,
    )
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60 * 24  # 24 hours
    allowed_origins: list[str] = ["http://localhost:3000"]

    # Rate Limiting
    rate_limit_requests: int = 100
    rate_limit_window_seconds: int = 60

    # Observability
    prometheus_enabled: bool = True
    otel_endpoint: str | None = None

    @field_validator("chroma_path", "raw_docs_path", "processed_docs_path", mode="before")
    @classmethod
    def ensure_path(cls, v: str | Path) -> Path:
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        return path

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }


@lru_cache
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()

"""
config.py — Central settings for the Advanced RAG pipeline.

Loaded once via lru_cache; values come from env or .env. Exposes provider-aware
accessors (`active_llm_model`, `active_embedding_model`) so the rest of the code
doesn't branch on provider.
"""
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # App
    app_name: str = "SupportMind RAG"
    app_version: str = "1.0.0"
    environment: Literal["development", "staging", "production"] = "development"
    debug: bool = False
    log_level: str = "INFO"

    # LLM provider
    llm_provider: Literal["openai", "gemini"] = "gemini"

    # OpenAI
    openai_api_key: str = Field(default="", description="OpenAI API key")
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 1024

    # Gemini (via Google's OpenAI-compat endpoint)
    gemini_api_key: str = Field(default="", description="Gemini API key")
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    gemini_model: str = "gemini-2.5-flash-lite"
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
    chunk_size: int = 480
    chunk_overlap: int = 64
    min_chunk_length: int = 50
    semantic_chunking: bool = True

    # Hybrid retrieval — BM25 + dense, fused via Reciprocal Rank Fusion
    dense_top_k: int = 20
    sparse_top_k: int = 20
    hybrid_top_k: int = 20
    final_top_k: int = 5
    rrf_k: int = 60  # RRF dampening constant; 60 is the value from the original paper
    similarity_threshold: float = 0.0  # 0 disables threshold; let reranker decide

    # Re-ranking
    reranker_enabled: bool = True
    reranker_backend: Literal["cross_encoder", "llm", "off"] = "llm"
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_k: int = 20  # how many candidates to send to the reranker

    # Query rewriting
    query_rewrite_enabled: bool = True
    query_rewrite_variants: int = 3  # how many alternate phrasings to generate
    hyde_enabled: bool = True  # generate a hypothetical answer for embedding

    # Vector store
    vector_store_backend: Literal["chroma"] = "chroma"
    chroma_path: Path = Path("./data/vectorstore")
    chroma_collection_name: str = "support_kb"

    # BM25 keyword index
    bm25_index_path: Path = Path("./data/bm25/index.pkl")

    # Storage
    raw_docs_path: Path = Path("./data/raw")
    processed_docs_path: Path = Path("./data/processed")

    # Security
    secret_key: str = Field(
        default="local-dev-only-change-me-local-dev-only-change-me",
        min_length=32,
    )
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60 * 24
    allowed_origins: list[str] = ["http://localhost:3000"]

    # Observability
    prometheus_enabled: bool = True

    @field_validator(
        "chroma_path", "raw_docs_path", "processed_docs_path", mode="before"
    )
    @classmethod
    def ensure_dir(cls, v: str | Path) -> Path:
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @field_validator("bm25_index_path", mode="before")
    @classmethod
    def ensure_parent_dir(cls, v: str | Path) -> Path:
        path = Path(v)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }


@lru_cache
def get_settings() -> Settings:
    return Settings()

"""
schemas.py — Pydantic models for API request/response validation.
"""
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator


# Query

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000, description="User question")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve")
    filters: dict[str, Any] | None = Field(default=None, description="Metadata filters")
    stream: bool = Field(default=False, description="Stream tokens via SSE")

    @field_validator("question")
    @classmethod
    def question_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Question cannot be blank")
        return v.strip()


class SourceCitation(BaseModel):
    source_index: int
    filename: str
    chunk_id: str
    similarity_score: float
    metadata: dict[str, Any] = {}


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceCitation]
    question: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    retrieval_count: int
    cached: bool = False
    latency_ms: float
    correlation_id: str


# Ingestion

class IngestRequest(BaseModel):
    source_path: str = Field(..., description="File path or URL to ingest")
    recursive: bool = Field(default=True, description="Recurse into directories")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Extra metadata to attach")


class IngestResponse(BaseModel):
    message: str
    documents_processed: int
    chunks_created: int
    vectors_upserted: int
    duration_seconds: float
    errors: list[str] = []


# Health

class HealthStatus(BaseModel):
    status: str                   # "healthy" | "degraded" | "unhealthy"
    version: str
    environment: str
    vector_store: bool
    cache: bool
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    vector_store_size: int = 0


# Document

class DocumentListItem(BaseModel):
    chunk_id: str
    source: str
    chunk_index: int
    char_count: int
    metadata: dict[str, Any] = {}


class DocumentListResponse(BaseModel):
    items: list[DocumentListItem]
    total: int
    page: int
    page_size: int


# Error

class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
    correlation_id: str | None = None

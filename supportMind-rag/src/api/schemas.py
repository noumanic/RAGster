"""
schemas.py — Pydantic request/response models for the API.

Response models mirror the per-stage shape of `AnswerEnvelope` so the frontend
can show what happened at each pipeline stage.
"""
from typing import Any

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    path: str = Field(..., description="File or directory to ingest")
    recursive: bool = Field(True, description="Recurse into subdirectories")


class IngestResponse(BaseModel):
    documents: int
    chunks: int
    elapsed_s: float
    sources: list[str]


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)


class SourceItem(BaseModel):
    index: int
    chunk_id: str
    source: str
    section_path: str
    rerank_score: float
    fused_score: float
    dense_score: float | None
    sparse_score: float | None
    preview: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    retrieval: dict[str, Any]
    generation: dict[str, Any]
    total_elapsed_s: float


class HealthResponse(BaseModel):
    status: str
    version: str
    vector_store_size: int
    bm25_index_size: int


class StatsResponse(BaseModel):
    app_name: str
    version: str
    llm_provider: str
    llm_model: str
    embedding_model: str
    reranker_backend: str
    vector_store_size: int
    bm25_index_size: int
    final_top_k: int
    rrf_k: int
    query_rewrite_enabled: bool
    hyde_enabled: bool
    semantic_chunking: bool

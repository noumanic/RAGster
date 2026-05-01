"""
routes.py — API endpoints.

  POST /ingest    walk a path, chunk, embed, write to dense + BM25 indexes
  POST /query     run rewrite → hybrid → rerank → generate, return cited answer
  GET  /health    liveness probe
  GET  /stats     introspection: which models, which backends, current sizes
"""
from fastapi import APIRouter, HTTPException

from src.api.schemas import (
    HealthResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    StatsResponse,
)
from src.generation.generator import Generator
from src.ingestion.pipeline import IngestionPipeline
from src.retrieval.bm25_index import get_bm25_index
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.vector_store import get_vector_store
from src.utils.config import get_settings
from src.utils.logging import get_logger

log = get_logger(__name__)
settings = get_settings()
router = APIRouter()

# Singletons — built lazily; the first request pays for any cold start.
_retriever: HybridRetriever | None = None
_generator: Generator | None = None
_pipeline: IngestionPipeline | None = None


def _get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever


def _get_generator() -> Generator:
    global _generator
    if _generator is None:
        _generator = Generator(retriever=_get_retriever())
    return _generator


def _get_pipeline() -> IngestionPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = IngestionPipeline(retriever=_get_retriever())
    return _pipeline


@router.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest) -> IngestResponse:
    try:
        report = await _get_pipeline().ingest_path(req.path, recursive=req.recursive)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return IngestResponse(**report)


@router.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest) -> QueryResponse:
    envelope = await _get_generator().answer(req.query)
    return QueryResponse(
        answer=envelope.answer,
        sources=envelope.sources,
        retrieval=envelope.retrieval,
        generation=envelope.generation,
        total_elapsed_s=envelope.total_elapsed_s,
    )


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    vs = get_vector_store()
    return HealthResponse(
        status="ok",
        version=settings.app_version,
        vector_store_size=await vs.count(),
        bm25_index_size=get_bm25_index().count(),
    )


@router.get("/stats", response_model=StatsResponse)
async def stats() -> StatsResponse:
    vs = get_vector_store()
    return StatsResponse(
        app_name=settings.app_name,
        version=settings.app_version,
        llm_provider=settings.llm_provider,
        llm_model=settings.active_llm_model,
        embedding_model=settings.active_embedding_model,
        reranker_backend=settings.reranker_backend
        if settings.reranker_enabled
        else "off",
        vector_store_size=await vs.count(),
        bm25_index_size=get_bm25_index().count(),
        final_top_k=settings.final_top_k,
        rrf_k=settings.rrf_k,
        query_rewrite_enabled=settings.query_rewrite_enabled,
        hyde_enabled=settings.hyde_enabled,
        semantic_chunking=settings.semantic_chunking,
    )

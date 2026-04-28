"""
routes.py — All FastAPI route handlers.
Endpoints:
  POST /query          — Ask a question (buffered or streaming)
  POST /ingest         — Ingest a document or directory
  GET  /health         — Health check
  GET  /documents      — List indexed documents
  DELETE /documents/{id} — Remove a document chunk
"""
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from src.api.schemas import (
    DocumentListResponse,
    ErrorResponse,
    HealthStatus,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    SourceCitation,
)
from src.generation.citation import CitationExtractor
from src.generation.llm_client import LLMClient
from src.generation.prompt_builder import PromptBuilder
from src.ingestion.chunker import RecursiveTextChunker
from src.ingestion.cleaner import TextCleaner
from src.ingestion.loader import DocumentLoader
from src.retrieval.reranker import get_reranker
from src.retrieval.retriever import Retriever
from src.utils.cache import get_cache
from src.utils.config import get_settings
from src.utils.logging import get_logger, get_correlation_id
from src.utils import metrics

router = APIRouter()
log = get_logger(__name__)
settings = get_settings()


# Dependency singletons

def get_retriever() -> Retriever:
    return Retriever()


def get_llm() -> LLMClient:
    return LLMClient()


def get_prompt_builder() -> PromptBuilder:
    return PromptBuilder(firm_name=settings.app_name)


def get_citation_extractor() -> CitationExtractor:
    return CitationExtractor()


# Routes

@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Ask a question over your documents",
    responses={400: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
async def query(
    request: QueryRequest,
    retriever: Retriever = Depends(get_retriever),
    llm: LLMClient = Depends(get_llm),
    prompt_builder: PromptBuilder = Depends(get_prompt_builder),
    citation_extractor: CitationExtractor = Depends(get_citation_extractor),
) -> QueryResponse:
    """Main RAG query endpoint.

    Pipeline:
    1. Check Redis cache
    2. Embed question → retrieve top-k chunks
    3. Optional: rerank chunks
    4. Build prompt with context
    5. Generate answer via LLM
    6. Extract citations
    7. Cache response
    """
    start = time.perf_counter()
    cid = get_correlation_id()
    cache_client = await get_cache()

    # 1. Cache check
    cached = await cache_client.get(request.question, request.filters)
    if cached:
        metrics.cache_hits_total.labels(cache_type="query").inc()
        cached["cached"] = True
        cached["correlation_id"] = cid
        return QueryResponse(**cached)
    metrics.cache_misses_total.labels(cache_type="query").inc()

    # 2. Retrieve
    chunks = await retriever.retrieve(
        query=request.question,
        top_k=request.top_k,
        filters=request.filters,
    )

    if not chunks:
        log.warning("No chunks retrieved for query", question=request.question[:80])

    # 3. Optional rerank
    reranker = get_reranker()
    if reranker and chunks:
        chunks = reranker.rerank(request.question, chunks)

    # 4. Build prompt
    prompt = prompt_builder.build(request.question, chunks)

    # 5. Generate
    result = await llm.generate(prompt)

    # 6. Extract citations
    cited = citation_extractor.extract(result.answer, prompt.context_chunks)

    # 7. Build response
    sources = [
        SourceCitation(
            source_index=c.index,
            filename=_basename(c.source),
            chunk_id=c.chunk_id,
            similarity_score=round(c.similarity_score, 4),
            metadata=c.metadata,
        )
        for c in cited.citations
    ]

    elapsed_ms = (time.perf_counter() - start) * 1000

    response_data = {
        "answer": cited.answer,
        "sources": [s.model_dump() for s in sources],
        "question": request.question,
        "model": result.model,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "total_tokens": result.total_tokens,
        "retrieval_count": len(chunks),
        "cached": False,
        "latency_ms": round(elapsed_ms, 1),
        "correlation_id": cid,
    }

    # 8. Cache it
    await cache_client.set(request.question, response_data, request.filters)

    log.info(
        "Query handled",
        latency_ms=round(elapsed_ms),
        chunks=len(chunks),
        tokens=result.total_tokens,
    )
    metrics.api_request_duration_seconds.labels(endpoint="/query").observe(elapsed_ms / 1000)

    return QueryResponse(**response_data)


@router.post(
    "/query/stream",
    summary="Stream a RAG answer via Server-Sent Events",
)
async def query_stream(
    request: QueryRequest,
    retriever: Retriever = Depends(get_retriever),
    llm: LLMClient = Depends(get_llm),
    prompt_builder: PromptBuilder = Depends(get_prompt_builder),
) -> StreamingResponse:
    """Streaming SSE endpoint — yields answer tokens as they arrive."""
    chunks = await retriever.retrieve(request.question, request.top_k, request.filters)
    prompt = prompt_builder.build(request.question, chunks)

    async def event_generator():
        async for token in llm.generate_stream(prompt):
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Ingest documents into the vector store",
)
async def ingest(
    request: IngestRequest,
    retriever: Retriever = Depends(get_retriever),
) -> IngestResponse:
    """Ingest a file, directory, or URL into the RAG index."""
    start = time.perf_counter()
    loader = DocumentLoader()
    cleaner = TextCleaner()
    chunker = RecursiveTextChunker()

    docs_processed = 0
    chunks_all = []
    errors = []

    source = request.source_path

    try:
        if source.startswith("http://") or source.startswith("https://"):
            raw_doc = await loader.load_url(source)
            clean_doc = cleaner.clean(raw_doc)
            chunks_all.extend(chunker.chunk(clean_doc))
            docs_processed = 1
        else:
            path = Path(source)
            if not path.exists():
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Path not found: {source}",
                )
            async for raw_doc in loader.load_path(path, recursive=request.recursive):
                try:
                    clean_doc = cleaner.clean(raw_doc)
                    chunks = chunker.chunk(clean_doc)
                    chunks_all.extend(chunks)
                    docs_processed += 1
                except Exception as e:
                    errors.append(f"{raw_doc.source}: {str(e)}")

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )

    # Embed and upsert
    vectors_upserted = await retriever.ingest_chunks(chunks_all)

    elapsed = time.perf_counter() - start
    metrics.docs_ingested_total.labels(doc_type="mixed", status="success").inc(docs_processed)

    log.info(
        "Ingestion complete",
        docs=docs_processed,
        chunks=len(chunks_all),
        vectors=vectors_upserted,
        elapsed=round(elapsed, 2),
    )

    return IngestResponse(
        message=f"Successfully ingested {docs_processed} document(s)",
        documents_processed=docs_processed,
        chunks_created=len(chunks_all),
        vectors_upserted=vectors_upserted,
        duration_seconds=round(elapsed, 2),
        errors=errors,
    )


@router.get(
    "/health",
    response_model=HealthStatus,
    summary="System health check",
)
async def health() -> HealthStatus:
    """Check health of all system components."""
    from src.retrieval.vector_store import get_vector_store

    vs = get_vector_store()
    cache_client = await get_cache()

    vs_healthy = await vs.health_check()
    cache_healthy = await cache_client.health_check()
    vs_size = await vs.count() if vs_healthy else 0

    overall = "healthy" if (vs_healthy and cache_healthy) else "degraded"
    if not vs_healthy:
        overall = "unhealthy"

    return HealthStatus(
        status=overall,
        version=settings.app_version,
        environment=settings.environment,
        vector_store=vs_healthy,
        cache=cache_healthy,
        vector_store_size=vs_size,
    )


@router.get(
    "/stats",
    summary="Vector store statistics",
)
async def stats() -> dict:
    from src.retrieval.vector_store import get_vector_store
    vs = get_vector_store()
    count = await vs.count()
    return {
        "vector_store_backend": settings.vector_store_backend,
        "total_vectors": count,
        "llm_provider": settings.llm_provider,
        "embedding_model": settings.active_embedding_model,
        "llm_model": settings.active_llm_model,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "top_k": settings.top_k,
        "reranker_enabled": settings.reranker_enabled,
    }


# Helpers

def _basename(path: str) -> str:
    import os
    return os.path.basename(path) if path else "unknown"

"""
main.py — FastAPI entrypoint.

Same conventions as legalMind-rag: CORS, request-logging middleware that
attaches a correlation ID, Prometheus mount, static frontend at /app, root
redirect, global 500 handler.
"""
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import make_asgi_app

from src.api.routes import router
from src.utils import metrics
from src.utils.config import get_settings
from src.utils.logging import get_logger, set_correlation_id, setup_logging

settings = get_settings()
setup_logging(settings.log_level, settings.environment)
log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info(
        f"Starting {settings.app_name} v{settings.app_version}",
        environment=settings.environment,
    )

    # Pre-warm the vector store + BM25 index so the first /query is honest
    from src.retrieval.bm25_index import get_bm25_index
    from src.retrieval.vector_store import get_vector_store

    vs = get_vector_store()
    bm25 = get_bm25_index()
    try:
        n_dense = await vs.count()
        n_sparse = bm25.count()
        metrics.vector_store_size.set(n_dense)
        metrics.bm25_index_size.set(n_sparse)
        log.info(
            "Indexes ready",
            dense_vectors=n_dense,
            bm25_chunks=n_sparse,
        )
    except Exception as exc:
        log.warning(f"Index warm-up failed: {exc}")

    yield
    log.info(f"Shutting down {settings.app_name}")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Advanced RAG service for customer-support Q&A. "
        "Pipeline: query rewrite (multi-query + HyDE) → hybrid retrieval (BM25 + dense, "
        "fused via RRF) → cross-encoder / LLM-judge reranking → cited answer."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    cid = request.headers.get("X-Correlation-ID", str(uuid.uuid4())[:8])
    set_correlation_id(cid)

    start = time.perf_counter()
    log.info(
        "Request started",
        method=request.method,
        path=request.url.path,
    )

    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000

    metrics.api_requests_total.labels(
        method=request.method,
        endpoint=request.url.path,
        status_code=response.status_code,
    ).inc()
    log.info(
        "Request completed",
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        latency_ms=round(elapsed_ms),
    )
    response.headers["X-Correlation-ID"] = cid
    response.headers["X-Response-Time-Ms"] = str(round(elapsed_ms))
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    log.exception("Unhandled exception", path=request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "detail": str(exc) if settings.debug else "An unexpected error occurred",
            "correlation_id": request.headers.get("X-Correlation-ID"),
        },
    )


app.include_router(router, prefix="/api/v1", tags=["RAG"])

if settings.prometheus_enabled:
    app.mount("/metrics", make_asgi_app())

# Static frontend at /app
_web_dir = Path(__file__).resolve().parents[2] / "web"
if _web_dir.is_dir():
    app.mount("/app", StaticFiles(directory=_web_dir, html=True), name="web")


@app.get("/", include_in_schema=False)
async def root():
    if _web_dir.is_dir():
        return RedirectResponse(url="/app/")
    return {
        "app": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/api/v1/health",
    }

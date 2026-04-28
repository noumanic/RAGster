"""
main.py — FastAPI application entry point.

Includes:
- Lifespan management (startup/shutdown hooks)
- CORS middleware
- Request logging middleware
- Prometheus metrics endpoint
- Global exception handler
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
from src.utils.cache import get_cache
from src.utils.config import get_settings
from src.utils.logging import setup_logging, get_logger, set_correlation_id
from src.utils import metrics

settings = get_settings()
setup_logging(settings.log_level, settings.environment)
log = get_logger(__name__)


# Lifespan (startup/shutdown)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown hooks."""
    log.info(
        f"Starting {settings.app_name} v{settings.app_version}",
        environment=settings.environment,
    )
    # Pre-warm connections — Redis is optional, the app degrades gracefully
    cache = None
    try:
        cache = await get_cache()
    except Exception as exc:
        log.warning(f"Redis unavailable, running without cache: {exc}")

    from src.retrieval.vector_store import get_vector_store
    vs = get_vector_store()
    try:
        count = await vs.count()
        metrics.vector_store_size.set(count)
        log.info(f"Vector store ready", backend=settings.vector_store_backend, vectors=count)
    except Exception as exc:
        log.warning(f"Vector store count failed: {exc}")

    yield

    # Shutdown
    if cache is not None:
        await cache.disconnect()
    log.info(f"Shutting down {settings.app_name}")


# Application

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Production Naive RAG API for legal document Q&A. "
        "Ingests PDFs, DOCX, and text files; answers questions with cited sources."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Middleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    """Log every request with correlation ID, method, path, and latency."""
    cid = request.headers.get("X-Correlation-ID", str(uuid.uuid4())[:8])
    set_correlation_id(cid)

    start = time.perf_counter()
    log.info(
        "Request started",
        method=request.method,
        path=request.url.path,
        client=request.client.host if request.client else "unknown",
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


# Global exception handler

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


# Routes

app.include_router(router, prefix="/api/v1", tags=["RAG"])

# Prometheus metrics endpoint
if settings.prometheus_enabled:
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)

# Static frontend at /app (single-file SPA in web/)
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

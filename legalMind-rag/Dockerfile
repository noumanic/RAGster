# Build stage
FROM python:3.11-slim AS builder

WORKDIR /build
COPY pyproject.toml .
RUN pip install --upgrade pip hatch && \
    pip install --no-cache-dir -e "." 2>/dev/null || \
    pip install fastapi uvicorn openai chromadb pdfplumber python-docx \
        aiohttp aiofiles redis pydantic pydantic-settings loguru prometheus-client \
        python-multipart httpx

# Runtime stage
FROM python:3.11-slim AS runtime

# Security: non-root user
RUN groupadd --gid 1001 appgroup && \
    useradd --uid 1001 --gid appgroup --no-create-home appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source
COPY src/ ./src/
COPY config/ ./config/
COPY scripts/ ./scripts/

# Create data directories
RUN mkdir -p /app/data/raw /app/data/processed /app/data/vectorstore && \
    chown -R appuser:appgroup /app/data

USER appuser

EXPOSE 8000

ENV PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')"

CMD ["uvicorn", "src.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--log-level", "info"]

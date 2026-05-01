"""
embeddings.py — Async embedding service (provider-aware: OpenAI or Gemini).

Calls Google's OpenAI-compat endpoint when `llm_provider=gemini`, so the same
AsyncOpenAI client is reused. Embeds queries and chunks; the chunk path
batches up to 100 inputs per call.
"""
import time

from openai import AsyncOpenAI

from src.ingestion.chunker import Chunk
from src.utils import metrics
from src.utils.config import get_settings
from src.utils.logging import get_logger

log = get_logger(__name__)
settings = get_settings()


class EmbeddingResult:
    def __init__(self, chunk: Chunk, embedding: list[float]) -> None:
        self.chunk = chunk
        self.embedding = embedding


class EmbeddingService:
    """Wraps the embedding API."""

    def __init__(self) -> None:
        if settings.llm_provider == "gemini":
            self._client = AsyncOpenAI(
                api_key=settings.gemini_api_key,
                base_url=settings.gemini_base_url,
                timeout=60.0,
                max_retries=3,
            )
        else:
            self._client = AsyncOpenAI(
                api_key=settings.openai_api_key,
                timeout=60.0,
                max_retries=3,
            )

    async def embed_query(self, text: str) -> list[float]:
        text = text.strip().replace("\n", " ")
        start = time.perf_counter()
        response = await self._client.embeddings.create(
            input=text,
            model=settings.active_embedding_model,
        )
        metrics.embedding_latency_seconds.observe(time.perf_counter() - start)
        return response.data[0].embedding

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Batch-embed an arbitrary list of strings (used by HyDE)."""
        if not texts:
            return []
        cleaned = [t.strip().replace("\n", " ") for t in texts]
        start = time.perf_counter()
        response = await self._client.embeddings.create(
            input=cleaned,
            model=settings.active_embedding_model,
        )
        metrics.embedding_latency_seconds.observe(time.perf_counter() - start)
        return [d.embedding for d in response.data]

    async def embed_chunks(self, chunks: list[Chunk]) -> list[EmbeddingResult]:
        if not chunks:
            return []

        results: list[EmbeddingResult] = []
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            texts = [c.text.strip().replace("\n", " ") for c in batch]
            start = time.perf_counter()
            response = await self._client.embeddings.create(
                input=texts,
                model=settings.active_embedding_model,
            )
            metrics.embedding_latency_seconds.observe(time.perf_counter() - start)
            for chunk, emb in zip(batch, response.data):
                results.append(EmbeddingResult(chunk=chunk, embedding=emb.embedding))
        return results

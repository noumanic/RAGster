"""
retriever.py — The core Naive RAG retriever.

Responsibilities:
1. Embed the user query via OpenAI embeddings
2. Run top-k cosine similarity search in the vector store
3. Filter by similarity threshold
4. Return ranked SearchResults with metadata
"""
import time
from openai import AsyncOpenAI

from src.retrieval.vector_store import BaseVectorStore, SearchResult, get_vector_store, EmbeddingResult
from src.ingestion.chunker import Chunk
from src.utils.config import get_settings
from src.utils.logging import get_logger
from src.utils import metrics

log = get_logger(__name__)
settings = get_settings()


class EmbeddingService:
    """Wraps OpenAI embeddings API.

    Handles:
    - Single string embedding (queries)
    - Batch embedding (document chunks)
    - Retry on transient errors
    - Metrics instrumentation
    """

    def __init__(self) -> None:
        if settings.llm_provider == "gemini":
            self._client = AsyncOpenAI(
                api_key=settings.gemini_api_key,
                base_url=settings.gemini_base_url,
            )
        else:
            self._client = AsyncOpenAI(api_key=settings.openai_api_key)

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        text = text.strip().replace("\n", " ")
        start = time.perf_counter()
        try:
            response = await self._client.embeddings.create(
                input=text,
                model=settings.active_embedding_model,
            )
            vector = response.data[0].embedding
            elapsed = time.perf_counter() - start
            metrics.embedding_latency_seconds.observe(elapsed)
            log.debug("Query embedded", model=settings.active_embedding_model, elapsed=elapsed)
            return vector
        except Exception as exc:
            log.error("Embedding failed", error=str(exc))
            raise

    async def embed_chunks(self, chunks: list[Chunk]) -> list[EmbeddingResult]:
        """Embed a list of chunks in batches of 100 (OpenAI limit)."""
        if not chunks:
            return []

        results = []
        batch_size = 100
        total_batches = (len(chunks) + batch_size - 1) // batch_size

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            texts = [c.text.strip().replace("\n", " ") for c in batch]

            start = time.perf_counter()
            response = await self._client.embeddings.create(
                input=texts,
                model=settings.active_embedding_model,
            )
            elapsed = time.perf_counter() - start
            metrics.embedding_latency_seconds.observe(elapsed)

            batch_num = i // batch_size + 1
            log.debug(
                f"Embedded batch {batch_num}/{total_batches}",
                batch_size=len(batch),
                elapsed=elapsed,
            )

            for chunk, emb_data in zip(batch, response.data):
                results.append(EmbeddingResult(chunk=chunk, embedding=emb_data.embedding))

        metrics.chunks_created_total.inc(len(results))
        return results


class Retriever:
    """Naive RAG Retriever — embed query, search vector store, return chunks.

    This is the 'R' in RAG. It does NOT call the LLM.
    """

    def __init__(
        self,
        vector_store: BaseVectorStore | None = None,
        embedding_service: EmbeddingService | None = None,
    ) -> None:
        self._vs = vector_store or get_vector_store()
        self._embedder = embedding_service or EmbeddingService()

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        """Main retrieval method.

        Args:
            query: Natural language question from the user.
            top_k: Number of chunks to retrieve (default from settings).
            filters: Optional metadata filters (e.g. {"doc_type": "contract"}).

        Returns:
            List of SearchResult sorted by similarity score descending.
        """
        if not query.strip():
            raise ValueError("Query cannot be empty")

        top_k = top_k or settings.top_k
        start = time.perf_counter()

        # 1. Embed the query
        query_vector = await self._embedder.embed_query(query)

        # 2. Similarity search
        results = await self._vs.search(
            query_vector=query_vector,
            top_k=top_k,
            filters=filters,
        )

        elapsed = time.perf_counter() - start
        metrics.retrieval_latency_seconds.observe(elapsed)
        metrics.retrieval_results_count.observe(len(results))

        if results:
            avg_score = sum(r.score for r in results) / len(results)
            metrics.avg_similarity_score.observe(avg_score)

        log.info(
            "Retrieval complete",
            query_preview=query[:60],
            results=len(results),
            top_score=results[0].score if results else None,
            elapsed=round(elapsed, 3),
        )

        # Sort by score descending (usually already sorted, but ensure it)
        return sorted(results, key=lambda r: r.score, reverse=True)

    async def ingest_chunks(self, chunks: list[Chunk]) -> int:
        """Embed and upsert chunks into the vector store.

        Returns number of chunks ingested.
        """
        if not chunks:
            return 0
        embedding_results = await self._embedder.embed_chunks(chunks)
        n = await self._vs.upsert(embedding_results)
        log.info(f"Ingested {n} chunks into vector store")
        return n

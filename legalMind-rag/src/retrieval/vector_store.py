"""
vector_store.py — Abstract vector store interface with ChromaDB and Pinecone backends.

The VectorStore abstraction allows swapping backends without changing retrieval logic.
Both backends expose the same async interface.
"""
import asyncio
import time
from abc import ABC, abstractmethod
from typing import Protocol

from src.ingestion.chunker import Chunk
from src.utils.config import get_settings
from src.utils.logging import get_logger
from src.utils import metrics

log = get_logger(__name__)
settings = get_settings()


class EmbeddingResult:
    """Container for a chunk + its embedding vector."""
    def __init__(self, chunk: Chunk, embedding: list[float]) -> None:
        self.chunk = chunk
        self.embedding = embedding


class SearchResult:
    """A single retrieval result."""
    def __init__(
        self,
        chunk_id: str,
        text: str,
        source: str,
        score: float,
        metadata: dict,
    ) -> None:
        self.chunk_id = chunk_id
        self.text = text
        self.source = source
        self.score = score
        self.metadata = metadata

    def __repr__(self) -> str:
        return f"SearchResult(score={self.score:.3f}, source={self.source!r})"


class BaseVectorStore(ABC):
    """Abstract base for all vector store backends."""

    @abstractmethod
    async def upsert(self, results: list[EmbeddingResult]) -> int:
        """Upsert embeddings. Returns number of vectors upserted."""

    @abstractmethod
    async def search(
        self,
        query_vector: list[float],
        top_k: int,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        """Return top-k most similar chunks."""

    @abstractmethod
    async def delete(self, chunk_ids: list[str]) -> int:
        """Delete chunks by ID. Returns count deleted."""

    @abstractmethod
    async def count(self) -> int:
        """Return total number of vectors stored."""

    @abstractmethod
    async def health_check(self) -> bool:
        """Return True if the store is reachable."""


# ChromaDB Backend

class ChromaVectorStore(BaseVectorStore):
    """Persistent ChromaDB vector store (default for local/dev)."""

    def __init__(self) -> None:
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        self._client = chromadb.PersistentClient(
            path=str(settings.chroma_path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        log.info(
            "ChromaDB initialized",
            path=str(settings.chroma_path),
            collection=settings.chroma_collection_name,
        )

    async def upsert(self, results: list[EmbeddingResult]) -> int:
        if not results:
            return 0
        loop = asyncio.get_event_loop()

        ids = [r.chunk.chunk_id for r in results]
        embeddings = [r.embedding for r in results]
        documents = [r.chunk.text for r in results]
        metadatas = [
            {
                **r.chunk.metadata,
                "source": r.chunk.source,
                "chunk_index": r.chunk.chunk_index,
            }
            for r in results
        ]

        def _upsert():
            self._collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )

        await loop.run_in_executor(None, _upsert)
        count = await self.count()
        metrics.vector_store_size.set(count)
        log.debug(f"Upserted {len(results)} vectors to ChromaDB")
        return len(results)

    async def search(
        self,
        query_vector: list[float],
        top_k: int,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        loop = asyncio.get_event_loop()
        where = filters if filters else None

        def _query():
            return self._collection.query(
                query_embeddings=[query_vector],
                n_results=min(top_k, max(await_count, 1)),
                where=where,
                include=["documents", "metadatas", "distances"],
            )

        # ChromaDB distances are already cosine distances [0,2]; convert to similarity
        await_count = await self.count()
        if await_count == 0:
            return []

        result = await loop.run_in_executor(None, _query)

        search_results = []
        for i, (doc, meta, dist) in enumerate(
            zip(
                result["documents"][0],
                result["metadatas"][0],
                result["distances"][0],
            )
        ):
            # Convert cosine distance to cosine similarity: sim = 1 - dist/2
            similarity = 1.0 - (dist / 2.0)
            if similarity < settings.similarity_threshold:
                continue
            search_results.append(
                SearchResult(
                    chunk_id=result["ids"][0][i],
                    text=doc,
                    source=meta.get("source", "unknown"),
                    score=similarity,
                    metadata=meta,
                )
            )

        return search_results

    async def delete(self, chunk_ids: list[str]) -> int:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: self._collection.delete(ids=chunk_ids))
        return len(chunk_ids)

    async def count(self) -> int:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._collection.count)

    async def health_check(self) -> bool:
        try:
            await self.count()
            return True
        except Exception:
            return False


# Pinecone Backend

class PineconeVectorStore(BaseVectorStore):
    """Pinecone serverless vector store (production cloud)."""

    def __init__(self) -> None:
        from pinecone import Pinecone  # type: ignore

        pc = Pinecone(api_key=settings.pinecone_api_key)
        self._index = pc.Index(settings.pinecone_index_name)
        log.info("Pinecone initialized", index=settings.pinecone_index_name)

    async def upsert(self, results: list[EmbeddingResult]) -> int:
        if not results:
            return 0
        loop = asyncio.get_event_loop()
        vectors = [
            {
                "id": r.chunk.chunk_id,
                "values": r.embedding,
                "metadata": {
                    **r.chunk.metadata,
                    "text": r.chunk.text[:1000],  # Pinecone metadata limit
                    "source": r.chunk.source,
                },
            }
            for r in results
        ]
        # Pinecone recommends batches of 100
        batch_size = 100
        total = 0
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i : i + batch_size]
            await loop.run_in_executor(None, lambda b=batch: self._index.upsert(vectors=b))
            total += len(batch)
        return total

    async def search(
        self,
        query_vector: list[float],
        top_k: int,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        loop = asyncio.get_event_loop()
        kwargs = {"vector": query_vector, "top_k": top_k, "include_metadata": True}
        if filters:
            kwargs["filter"] = filters

        response = await loop.run_in_executor(None, lambda: self._index.query(**kwargs))

        return [
            SearchResult(
                chunk_id=match["id"],
                text=match["metadata"].get("text", ""),
                source=match["metadata"].get("source", "unknown"),
                score=match["score"],
                metadata=match["metadata"],
            )
            for match in response["matches"]
            if match["score"] >= settings.similarity_threshold
        ]

    async def delete(self, chunk_ids: list[str]) -> int:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: self._index.delete(ids=chunk_ids))
        return len(chunk_ids)

    async def count(self) -> int:
        loop = asyncio.get_event_loop()
        stats = await loop.run_in_executor(None, self._index.describe_index_stats)
        return stats.get("total_vector_count", 0)

    async def health_check(self) -> bool:
        try:
            await self.count()
            return True
        except Exception:
            return False


# Factory

_store: BaseVectorStore | None = None


def get_vector_store() -> BaseVectorStore:
    """Return the configured vector store singleton."""
    global _store
    if _store is None:
        backend = settings.vector_store_backend
        if backend == "chroma":
            _store = ChromaVectorStore()
        elif backend == "pinecone":
            _store = PineconeVectorStore()
        else:
            raise ValueError(f"Unknown vector store backend: {backend}")
    return _store

"""
vector_store.py — ChromaDB-backed dense vector store.

Pared-down version of the legalMind store: no Pinecone branch, no similarity
threshold (the reranker handles relevance now), but otherwise identical
async wrapper over chromadb's sync client.
"""
import asyncio
from abc import ABC, abstractmethod

from src.retrieval.embeddings import EmbeddingResult
from src.utils import metrics
from src.utils.config import get_settings
from src.utils.logging import get_logger

log = get_logger(__name__)
settings = get_settings()


class SearchResult:
    """Single dense-search hit."""

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
    @abstractmethod
    async def upsert(self, results: list[EmbeddingResult]) -> int: ...

    @abstractmethod
    async def search(
        self, query_vector: list[float], top_k: int, filters: dict | None = None
    ) -> list[SearchResult]: ...

    @abstractmethod
    async def delete(self, chunk_ids: list[str]) -> int: ...

    @abstractmethod
    async def count(self) -> int: ...


class ChromaVectorStore(BaseVectorStore):
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
            "ChromaDB ready",
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

        await loop.run_in_executor(
            None,
            lambda: self._collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            ),
        )
        metrics.vector_store_size.set(await self.count())
        return len(results)

    async def search(
        self, query_vector: list[float], top_k: int, filters: dict | None = None
    ) -> list[SearchResult]:
        total = await self.count()
        if total == 0:
            return []

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self._collection.query(
                query_embeddings=[query_vector],
                n_results=min(top_k, total),
                where=filters or None,
                include=["documents", "metadatas", "distances"],
            ),
        )

        out: list[SearchResult] = []
        for i, (doc, meta, dist) in enumerate(
            zip(result["documents"][0], result["metadatas"][0], result["distances"][0])
        ):
            similarity = 1.0 - (dist / 2.0)  # cosine distance → similarity
            out.append(
                SearchResult(
                    chunk_id=result["ids"][0][i],
                    text=doc,
                    source=meta.get("source", "unknown"),
                    score=similarity,
                    metadata=meta,
                )
            )
        return out

    async def delete(self, chunk_ids: list[str]) -> int:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, lambda: self._collection.delete(ids=chunk_ids)
        )
        return len(chunk_ids)

    async def count(self) -> int:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._collection.count)


_store: BaseVectorStore | None = None


def get_vector_store() -> BaseVectorStore:
    global _store
    if _store is None:
        _store = ChromaVectorStore()
    return _store

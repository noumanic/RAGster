"""
test_retrieval.py — Unit tests for the retrieval layer.

Covers:
- SearchResult / EmbeddingResult containers
- EmbeddingService (OpenAI client mocked)
- Retriever orchestration (vector store + embedder mocked)
"""
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.ingestion.chunker import Chunk
from src.retrieval.retriever import EmbeddingService, Retriever
from src.retrieval.vector_store import EmbeddingResult, SearchResult


# Container shape

class TestSearchResult:
    def test_repr_includes_score_and_source(self):
        r = SearchResult(
            chunk_id="c1",
            text="hello",
            source="contract.pdf",
            score=0.876,
            metadata={"page": 1},
        )
        rep = repr(r)
        assert "0.876" in rep
        assert "contract.pdf" in rep

    def test_metadata_preserved(self):
        r = SearchResult(
            chunk_id="c1", text="t", source="s", score=0.5, metadata={"k": "v"}
        )
        assert r.metadata == {"k": "v"}


# EmbeddingService

class TestEmbeddingService:
    @pytest.mark.asyncio
    async def test_embed_query_returns_vector(self):
        svc = EmbeddingService()
        fake_resp = MagicMock()
        fake_resp.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
        svc._client = MagicMock()
        svc._client.embeddings = MagicMock()
        svc._client.embeddings.create = AsyncMock(return_value=fake_resp)

        vec = await svc.embed_query("what is X?")

        assert vec == [0.1, 0.2, 0.3]
        svc._client.embeddings.create.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_embed_query_strips_newlines(self):
        svc = EmbeddingService()
        fake_resp = MagicMock()
        fake_resp.data = [MagicMock(embedding=[0.0])]
        svc._client = MagicMock()
        svc._client.embeddings = MagicMock()
        svc._client.embeddings.create = AsyncMock(return_value=fake_resp)

        await svc.embed_query("  line1\nline2  ")

        kwargs = svc._client.embeddings.create.await_args.kwargs
        assert "\n" not in kwargs["input"]

    @pytest.mark.asyncio
    async def test_embed_chunks_empty_list_short_circuits(self):
        svc = EmbeddingService()
        svc._client = MagicMock()
        svc._client.embeddings = MagicMock()
        svc._client.embeddings.create = AsyncMock()

        results = await svc.embed_chunks([])

        assert results == []
        svc._client.embeddings.create.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_embed_chunks_returns_one_result_per_chunk(self):
        svc = EmbeddingService()
        chunks = [
            Chunk(chunk_id=f"c{i}", text=f"text {i}", source="src", chunk_index=i, total_chunks=3)
            for i in range(3)
        ]
        fake_resp = MagicMock()
        fake_resp.data = [MagicMock(embedding=[float(i)]) for i in range(3)]
        svc._client = MagicMock()
        svc._client.embeddings = MagicMock()
        svc._client.embeddings.create = AsyncMock(return_value=fake_resp)

        results = await svc.embed_chunks(chunks)

        assert len(results) == 3
        assert all(isinstance(r, EmbeddingResult) for r in results)
        assert [r.embedding[0] for r in results] == [0.0, 1.0, 2.0]


# Retriever

def _make_retriever_with_mocks(search_results: list[SearchResult]) -> Retriever:
    vs = MagicMock()
    vs.search = AsyncMock(return_value=search_results)
    vs.upsert = AsyncMock(return_value=len(search_results))
    embedder = MagicMock()
    embedder.embed_query = AsyncMock(return_value=[0.1] * 4)
    embedder.embed_chunks = AsyncMock(return_value=[])
    return Retriever(vector_store=vs, embedding_service=embedder)


class TestRetriever:
    @pytest.mark.asyncio
    async def test_empty_query_raises(self):
        r = _make_retriever_with_mocks([])
        with pytest.raises(ValueError, match="empty"):
            await r.retrieve("   ")

    @pytest.mark.asyncio
    async def test_results_sorted_by_score_desc(self):
        results = [
            SearchResult("c1", "lo", "s", 0.4, {}),
            SearchResult("c2", "hi", "s", 0.9, {}),
            SearchResult("c3", "mid", "s", 0.6, {}),
        ]
        r = _make_retriever_with_mocks(results)

        out = await r.retrieve("anything")

        assert [s.score for s in out] == [0.9, 0.6, 0.4]

    @pytest.mark.asyncio
    async def test_passes_top_k_and_filters_to_vector_store(self):
        r = _make_retriever_with_mocks([])

        await r.retrieve("q", top_k=7, filters={"doc_type": "contract"})

        call = r._vs.search.await_args
        assert call.kwargs["top_k"] == 7
        assert call.kwargs["filters"] == {"doc_type": "contract"}

    @pytest.mark.asyncio
    async def test_ingest_chunks_empty_returns_zero(self):
        r = _make_retriever_with_mocks([])
        n = await r.ingest_chunks([])
        assert n == 0
        r._embedder.embed_chunks.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_ingest_chunks_calls_upsert(self):
        results = [SearchResult("c1", "t", "s", 1.0, {})]
        r = _make_retriever_with_mocks(results)
        chunk = Chunk(chunk_id="c1", text="t", source="s", chunk_index=0, total_chunks=1)
        r._embedder.embed_chunks = AsyncMock(
            return_value=[EmbeddingResult(chunk=chunk, embedding=[0.0])]
        )

        n = await r.ingest_chunks([chunk])

        assert n == 1
        r._vs.upsert.assert_awaited_once()

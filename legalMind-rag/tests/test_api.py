"""
test_api.py — Integration tests for the FastAPI endpoints.
Uses pytest-asyncio + httpx AsyncClient. Mocks OpenAI and vector store.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from httpx import AsyncClient, ASGITransport

from src.api.main import app
from src.retrieval.vector_store import SearchResult
from src.generation.llm_client import GenerationResult


MOCK_CHUNKS = [
    SearchResult(
        chunk_id="chunk-001",
        text="The termination clause requires 30 days written notice.",
        source="smith_contract.pdf",
        score=0.92,
        metadata={"page_number": 5, "doc_type": "contract"},
    ),
    SearchResult(
        chunk_id="chunk-002",
        text="Penalties for early termination shall be $10,000 per occurrence.",
        source="smith_contract.pdf",
        score=0.85,
        metadata={"page_number": 6, "doc_type": "contract"},
    ),
]

MOCK_GENERATION = GenerationResult(
    answer=(
        "The termination clause requires 30 days written notice [Source 1]. "
        "Early termination penalties are $10,000 per occurrence [Source 2]."
    ),
    prompt_tokens=450,
    completion_tokens=80,
    model="gpt-4o",
    finish_reason="stop",
)


@pytest.fixture
def mock_retriever():
    with patch("src.api.routes.Retriever") as MockRetriever:
        instance = MockRetriever.return_value
        instance.retrieve = AsyncMock(return_value=MOCK_CHUNKS)
        instance.ingest_chunks = AsyncMock(return_value=10)
        yield instance


@pytest.fixture
def mock_llm():
    with patch("src.api.routes.LLMClient") as MockLLM:
        instance = MockLLM.return_value
        instance.generate = AsyncMock(return_value=MOCK_GENERATION)
        yield instance


@pytest.fixture
def mock_cache():
    with patch("src.api.routes.get_cache") as mock:
        cache = AsyncMock()
        cache.get = AsyncMock(return_value=None)
        cache.set = AsyncMock()
        mock.return_value = cache
        yield cache


@pytest.mark.asyncio
class TestQueryEndpoint:
    async def test_query_success(self, mock_retriever, mock_llm, mock_cache):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v1/query",
                json={"question": "What are the termination clauses in the Smith contract?"},
            )
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert data["retrieval_count"] == 2
        assert data["cached"] is False

    async def test_query_empty_question_fails(self, mock_cache):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post("/api/v1/query", json={"question": "  "})
        assert response.status_code == 422

    async def test_query_too_long_fails(self, mock_cache):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v1/query", json={"question": "x" * 3000}
            )
        assert response.status_code == 422

    async def test_query_cache_hit(self, mock_retriever, mock_llm, mock_cache):
        cached_response = {
            "answer": "Cached answer",
            "sources": [],
            "question": "test",
            "model": "gpt-4o",
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "retrieval_count": 0,
            "cached": False,
            "latency_ms": 50.0,
            "correlation_id": "abc123",
        }
        mock_cache.get.return_value = cached_response

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v1/query", json={"question": "What is in the contract?"}
            )
        assert response.status_code == 200
        data = response.json()
        assert data["cached"] is True
        # LLM should NOT have been called
        mock_llm.generate.assert_not_called()


@pytest.mark.asyncio
class TestHealthEndpoint:
    async def test_health_check(self):
        with (
            patch("src.api.routes.get_vector_store") as mock_vs,
            patch("src.api.routes.get_cache") as mock_cache,
        ):
            vs = AsyncMock()
            vs.health_check = AsyncMock(return_value=True)
            vs.count = AsyncMock(return_value=1500)
            mock_vs.return_value = vs

            cache = AsyncMock()
            cache.health_check = AsyncMock(return_value=True)
            mock_cache.return_value = cache

            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.get("/api/v1/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["vector_store"] is True
        assert data["vector_store_size"] == 1500

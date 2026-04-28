"""
test_generation.py — Unit tests for the generation layer.

Covers:
- PromptBuilder (system/user assembly, budget trimming, source numbering)
- CitationExtractor (parse [Source N], coverage, format references)
- LLMClient (OpenAI client mocked — happy path + streaming shape)
"""
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.generation.citation import CitationExtractor
from src.generation.llm_client import GenerationResult, LLMClient
from src.generation.prompt_builder import BuiltPrompt, PromptBuilder
from src.retrieval.vector_store import SearchResult


def _sr(idx: int, text: str = "chunk text", score: float = 0.9) -> SearchResult:
    return SearchResult(
        chunk_id=f"c{idx}",
        text=text,
        source=f"doc{idx}.pdf",
        score=score,
        metadata={"page_number": idx},
    )


# PromptBuilder

class TestPromptBuilder:
    def test_build_returns_built_prompt(self):
        pb = PromptBuilder(firm_name="Acme LLP")
        result = pb.build("What is the penalty?", [_sr(1), _sr(2)])

        assert isinstance(result, BuiltPrompt)
        assert "Acme LLP" in result.system_prompt
        assert "What is the penalty?" in result.user_message
        assert len(result.context_chunks) == 2
        assert result.estimated_tokens > 0

    def test_numbers_sources_starting_at_one(self):
        pb = PromptBuilder()
        chunks = [_sr(1, text="A"), _sr(2, text="B"), _sr(3, text="C")]

        out = pb.build("q", chunks)

        assert "[Source 1]" in out.user_message
        assert "[Source 2]" in out.user_message
        assert "[Source 3]" in out.user_message

    def test_zero_chunks_still_builds_prompt(self):
        pb = PromptBuilder()
        out = pb.build("q", [])
        assert "No relevant documents found." in out.user_message

    def test_budget_trims_chunks_when_overflowing(self):
        # max_context_tokens=50 → ~200 chars total budget; each chunk ~250+ chars
        pb = PromptBuilder(max_context_tokens=50)
        big = "x" * 1000
        chunks = [_sr(1, text=big), _sr(2, text=big), _sr(3, text=big)]

        out = pb.build("q", chunks)

        assert len(out.context_chunks) < len(chunks)


# CitationExtractor

class TestCitationExtractor:
    def test_extracts_source_references(self):
        ext = CitationExtractor()
        chunks = [_sr(1), _sr(2)]
        answer = "The penalty is 10% [Source 1]. Termination is 30 days [Source 2]."

        cited = ext.extract(answer, chunks)

        assert len(cited.citations) == 2
        assert {c.index for c in cited.citations} == {1, 2}

    def test_uncited_answer_strips_markers(self):
        ext = CitationExtractor()
        cited = ext.extract("Yes [Source 1].", [_sr(1)])
        assert "[Source 1]" not in cited.uncited_answer
        assert cited.uncited_answer.startswith("Yes")

    def test_invalid_source_index_skipped(self):
        ext = CitationExtractor()
        # Answer cites Source 5 but only 2 chunks were provided
        cited = ext.extract("Claim [Source 5].", [_sr(1), _sr(2)])
        assert cited.citations == []

    def test_duplicate_citations_deduped(self):
        ext = CitationExtractor()
        cited = ext.extract(
            "Fact [Source 1]. Same fact again [Source 1].",
            [_sr(1)],
        )
        assert len(cited.citations) == 1

    def test_empty_answer_returns_empty_citations(self):
        cited = CitationExtractor().extract("", [_sr(1)])
        assert cited.citations == []
        assert cited.coverage_ratio == 0.0

    def test_coverage_ratio_in_unit_range(self):
        ext = CitationExtractor()
        cited = ext.extract(
            "Cited sentence [Source 1]. Uncited sentence here.",
            [_sr(1)],
        )
        assert 0.0 < cited.coverage_ratio <= 1.0

    def test_format_references_includes_filename(self):
        ext = CitationExtractor()
        cited = ext.extract("A [Source 1].", [_sr(1)])
        refs = ext.format_references(cited)
        assert "doc1.pdf" in refs
        assert "References" in refs

    def test_format_references_empty_when_no_citations(self):
        ext = CitationExtractor()
        cited = ext.extract("No sources here.", [_sr(1)])
        assert ext.format_references(cited) == ""


# LLMClient

def _built_prompt() -> BuiltPrompt:
    return BuiltPrompt(
        system_prompt="sys",
        user_message="usr",
        context_chunks=[_sr(1)],
        estimated_tokens=10,
    )


class TestLLMClient:
    @pytest.mark.asyncio
    async def test_generate_returns_generation_result(self):
        client = LLMClient()
        fake_resp = MagicMock()
        fake_resp.choices = [MagicMock(message=MagicMock(content="An answer."))]
        fake_resp.usage = MagicMock(
            prompt_tokens=10, completion_tokens=5, total_tokens=15
        )
        fake_resp.model = "gpt-4o"
        client._client = MagicMock()
        client._client.chat = MagicMock()
        client._client.chat.completions = MagicMock()
        client._client.chat.completions.create = AsyncMock(return_value=fake_resp)

        result = await client.generate(_built_prompt())

        assert isinstance(result, GenerationResult)
        assert "An answer." in result.answer
        assert result.prompt_tokens == 10
        assert result.completion_tokens == 5
        assert result.total_tokens == 15
        client._client.chat.completions.create.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_generate_passes_system_and_user_messages(self):
        client = LLMClient()
        fake_resp = MagicMock()
        fake_resp.choices = [MagicMock(message=MagicMock(content="ok"))]
        fake_resp.usage = MagicMock(prompt_tokens=1, completion_tokens=1, total_tokens=2)
        fake_resp.model = "gpt-4o"
        client._client = MagicMock()
        client._client.chat = MagicMock()
        client._client.chat.completions = MagicMock()
        client._client.chat.completions.create = AsyncMock(return_value=fake_resp)

        await client.generate(_built_prompt())

        kwargs = client._client.chat.completions.create.await_args.kwargs
        roles = [m["role"] for m in kwargs["messages"]]
        assert "system" in roles
        assert "user" in roles

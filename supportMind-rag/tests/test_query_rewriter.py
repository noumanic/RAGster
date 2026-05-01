"""Tests for the QueryRewriter — focuses on parsing logic; LLM calls are mocked."""
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.retrieval.query_rewriter import QueryRewriter


def test_parse_variants_handles_plain_json_array():
    out = QueryRewriter._parse_variants(
        '["why is my device not charging?", "charger problem"]', expected=3
    )
    assert out == ["why is my device not charging?", "charger problem"]


def test_parse_variants_strips_markdown_fence():
    fenced = '```json\n["a", "b", "c"]\n```'
    out = QueryRewriter._parse_variants(fenced, expected=3)
    assert out == ["a", "b", "c"]


def test_parse_variants_falls_back_to_line_split():
    text = "1. how do I charge my device\n2. charger error\n- charging fault"
    out = QueryRewriter._parse_variants(text, expected=3)
    assert out == [
        "how do I charge my device",
        "charger error",
        "charging fault",
    ]


def test_parse_variants_caps_to_expected():
    out = QueryRewriter._parse_variants('["a","b","c","d","e"]', expected=2)
    assert out == ["a", "b"]


@pytest.mark.asyncio
async def test_rewrite_runs_multi_query_and_hyde_in_parallel(monkeypatch):
    rw = QueryRewriter.__new__(QueryRewriter)
    # stub out the two private async methods
    rw._multi_query = AsyncMock(return_value=["alt 1", "alt 2"])
    rw._hyde = AsyncMock(return_value="A plausible KB answer.")

    # also bypass the real __init__'s OpenAI client
    rw._client = MagicMock()

    from src.utils import config

    monkeypatch.setattr(config.get_settings(), "query_rewrite_enabled", True, raising=False)
    monkeypatch.setattr(config.get_settings(), "hyde_enabled", True, raising=False)

    out = await rw.rewrite("how do I charge?")
    assert out.original == "how do I charge?"
    assert out.variants == ["alt 1", "alt 2"]
    assert out.hyde_passage == "A plausible KB answer."
    assert out.elapsed_s >= 0

"""
query_rewriter.py — Two query-rewriting techniques the Advanced RAG pipeline runs
in parallel before any retrieval happens.

1. Multi-query expansion. We ask the LLM for N alternate phrasings of the same
   user question. Each variant becomes its own dense+sparse query; results are
   union'd before fusion. This raises recall when the user's wording diverges
   from the docs (conversational input vs. terse documentation).

2. HyDE (Hypothetical Document Embeddings). We ask the LLM for a *plausible*
   answer to the question, then embed THAT answer instead of the question. The
   embedding lands in answer-space, much closer to the relevant chunks than a
   question-space embedding would. Especially useful for "how do I X" → docs
   that say "to X, do Y".

Both rewrites are best-effort: if the LLM call fails or rewriting is disabled,
the pipeline degrades gracefully to the raw query.
"""
import asyncio
import json
import time
from dataclasses import dataclass

from openai import AsyncOpenAI

from src.utils import metrics
from src.utils.config import get_settings
from src.utils.logging import get_logger

log = get_logger(__name__)
settings = get_settings()


@dataclass
class RewrittenQuery:
    """Output of the rewriter — every retrieval signal we generated."""

    original: str
    variants: list[str]            # alternate phrasings (multi-query)
    hyde_passage: str | None       # hypothetical answer (for embedding)
    elapsed_s: float


_MULTI_QUERY_SYSTEM = """You expand customer-support questions into alternate \
phrasings to improve search recall. Generate concise rewrites that preserve the \
user's intent but vary vocabulary, specificity, and phrasing. Include at least \
one rewrite that uses likely product/technical terminology. Respond with a JSON \
array of strings only — no commentary, no markdown."""

_HYDE_SYSTEM = """You write short, plausible passages that look like they came \
from a product knowledge base. Given a user question, write a 2–4 sentence \
answer in declarative documentation style — as if you were quoting the help \
center. Do not preface with "Sure" or "Here is". Output the passage only."""


class QueryRewriter:
    def __init__(self) -> None:
        if settings.llm_provider == "gemini":
            self._client = AsyncOpenAI(
                api_key=settings.gemini_api_key,
                base_url=settings.gemini_base_url,
                timeout=30.0,
                max_retries=2,
            )
        else:
            self._client = AsyncOpenAI(
                api_key=settings.openai_api_key,
                timeout=30.0,
                max_retries=2,
            )

    async def rewrite(self, query: str) -> RewrittenQuery:
        start = time.perf_counter()

        if not settings.query_rewrite_enabled:
            return RewrittenQuery(
                original=query, variants=[], hyde_passage=None, elapsed_s=0.0
            )

        # Run multi-query and HyDE in parallel — neither depends on the other.
        tasks: list = [self._multi_query(query)]
        if settings.hyde_enabled:
            tasks.append(self._hyde(query))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        variants = results[0] if not isinstance(results[0], Exception) else []
        if isinstance(results[0], Exception):
            log.warning(f"multi-query rewrite failed: {results[0]}")

        hyde = None
        if settings.hyde_enabled:
            hyde_res = results[1]
            if isinstance(hyde_res, Exception):
                log.warning(f"HyDE failed: {hyde_res}")
            else:
                hyde = hyde_res

        elapsed = time.perf_counter() - start
        metrics.query_rewrite_latency_seconds.observe(elapsed)
        log.info(
            "Query rewrite complete",
            variants=len(variants),
            has_hyde=hyde is not None,
            elapsed=round(elapsed, 2),
        )

        return RewrittenQuery(
            original=query,
            variants=variants,
            hyde_passage=hyde,
            elapsed_s=elapsed,
        )

    async def _multi_query(self, query: str) -> list[str]:
        n = settings.query_rewrite_variants
        if n <= 0:
            return []

        user = (
            f'Original question: "{query}"\n\n'
            f"Produce exactly {n} alternate phrasings as a JSON array of strings."
        )
        response = await self._client.chat.completions.create(
            model=settings.active_llm_model,
            messages=[
                {"role": "system", "content": _MULTI_QUERY_SYSTEM},
                {"role": "user", "content": user},
            ],
            temperature=0.3,
            max_tokens=300,
        )
        content = (response.choices[0].message.content or "").strip()
        return self._parse_variants(content, expected=n)

    async def _hyde(self, query: str) -> str | None:
        response = await self._client.chat.completions.create(
            model=settings.active_llm_model,
            messages=[
                {"role": "system", "content": _HYDE_SYSTEM},
                {"role": "user", "content": query},
            ],
            temperature=0.2,
            max_tokens=180,
        )
        passage = (response.choices[0].message.content or "").strip()
        return passage or None

    @staticmethod
    def _parse_variants(content: str, expected: int) -> list[str]:
        """Tolerant JSON parser — LLMs sometimes wrap arrays in ```json fences."""
        cleaned = content
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()
            # drop trailing fence remnants
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3].strip()

        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, list):
                out = [str(x).strip() for x in parsed if str(x).strip()]
                return out[:expected]
        except json.JSONDecodeError:
            pass

        # Fallback: split lines, strip numbering / bullets / quotes
        out: list[str] = []
        for line in cleaned.splitlines():
            line = line.strip().lstrip("-*0123456789. ").strip().strip('"').strip("'")
            if line:
                out.append(line)
        return out[:expected]

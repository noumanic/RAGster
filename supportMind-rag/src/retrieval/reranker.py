"""
reranker.py — Two reranking backends, selected by RERANKER_BACKEND.

Why rerank at all? Recall and precision are different problems. Hybrid retrieval
goes wide (top-20 candidates) for recall. The reranker is a tighter, more
expensive scorer that picks the top-5 to actually feed the LLM. Cross-encoders
read (query, candidate) jointly — they catch relevance signals a bi-encoder
embedding can't.

Backends:
  - cross_encoder : sentence-transformers cross-encoder (default in prod).
                    Heavy dep (torch). Lazy-imported so cold-start doesn't crash
                    if it's not installed.
  - llm           : LLM-as-judge fallback. No torch dep. We send the query plus
                    a numbered list of candidates and ask Gemini to rank by
                    relevance. Slower but zero local compute and works
                    everywhere.
  - off           : pass-through, for ablation.

Both backends produce the same RerankedHit type so the pipeline doesn't care.
"""
import time

from openai import AsyncOpenAI

from src.retrieval.types import FusedHit, RerankedHit
from src.utils import metrics
from src.utils.config import get_settings
from src.utils.logging import get_logger

log = get_logger(__name__)
settings = get_settings()


class CrossEncoderReranker:
    """sentence-transformers cross-encoder (lazy-loaded so the import only fires
    when this backend is actually used)."""

    def __init__(self) -> None:
        self._model = None  # lazy

    def _ensure_loaded(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            log.info(f"Loading cross-encoder: {settings.cross_encoder_model}")
            self._model = CrossEncoder(settings.cross_encoder_model)
        return self._model

    async def rerank(
        self, query: str, candidates: list[FusedHit], top_k: int
    ) -> list[RerankedHit]:
        if not candidates:
            return []
        model = self._ensure_loaded()
        pairs = [(query, c.text) for c in candidates]
        scores = model.predict(pairs).tolist()
        scored = sorted(
            zip(candidates, scores), key=lambda t: t[1], reverse=True
        )[:top_k]
        return [
            RerankedHit(
                chunk_id=c.chunk_id,
                text=c.text,
                source=c.source,
                metadata=c.metadata,
                rerank_score=float(s),
                fused_score=c.fused_score,
                dense_score=c.dense_score,
                sparse_score=c.sparse_score,
            )
            for c, s in scored
        ]


_LLM_RERANK_SYSTEM = """You are a relevance grader for a customer-support \
search engine. You will see a user question and a numbered list of candidate \
passages from a knowledge base. Score each candidate from 0.0 to 1.0 by how \
directly it answers the question. Be strict — passages that mention the topic \
but don't actually answer the question score under 0.5. Respond with a JSON \
array where each element is {"id": <int>, "score": <float>}, ordered best to \
worst. Output JSON only — no commentary."""


class LLMReranker:
    """LLM-as-judge reranker — no extra deps, runs against the configured LLM."""

    def __init__(self) -> None:
        if settings.llm_provider == "gemini":
            self._client = AsyncOpenAI(
                api_key=settings.gemini_api_key,
                base_url=settings.gemini_base_url,
                timeout=45.0,
                max_retries=2,
            )
        else:
            self._client = AsyncOpenAI(
                api_key=settings.openai_api_key,
                timeout=45.0,
                max_retries=2,
            )

    async def rerank(
        self, query: str, candidates: list[FusedHit], top_k: int
    ) -> list[RerankedHit]:
        if not candidates:
            return []

        # Truncate each candidate so the prompt stays bounded.
        listing = "\n\n".join(
            f"[{i}] {c.text[:600]}" for i, c in enumerate(candidates)
        )
        user = (
            f'User question: "{query}"\n\n'
            f"Candidates:\n{listing}\n\n"
            f"Score each candidate."
        )

        try:
            response = await self._client.chat.completions.create(
                model=settings.active_llm_model,
                messages=[
                    {"role": "system", "content": _LLM_RERANK_SYSTEM},
                    {"role": "user", "content": user},
                ],
                temperature=0.0,
                max_tokens=600,
            )
            scores = self._parse_scores(
                response.choices[0].message.content or "", n=len(candidates)
            )
        except Exception as exc:
            log.warning(f"LLM rerank failed, falling back to fused order: {exc}")
            scores = [c.fused_score for c in candidates]

        scored = sorted(
            zip(candidates, scores), key=lambda t: t[1], reverse=True
        )[:top_k]
        return [
            RerankedHit(
                chunk_id=c.chunk_id,
                text=c.text,
                source=c.source,
                metadata=c.metadata,
                rerank_score=float(s),
                fused_score=c.fused_score,
                dense_score=c.dense_score,
                sparse_score=c.sparse_score,
            )
            for c, s in scored
        ]

    @staticmethod
    def _parse_scores(content: str, n: int) -> list[float]:
        import json

        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3].strip()

        try:
            arr = json.loads(cleaned)
        except json.JSONDecodeError:
            return [0.0] * n

        out = [0.0] * n
        if isinstance(arr, list):
            for entry in arr:
                if isinstance(entry, dict):
                    idx = entry.get("id")
                    score = entry.get("score")
                    if isinstance(idx, int) and 0 <= idx < n and isinstance(
                        score, (int, float)
                    ):
                        out[idx] = float(score)
        return out


class PassthroughReranker:
    """No-op reranker — keeps the fused order. Useful for A/B comparisons."""

    async def rerank(
        self, query: str, candidates: list[FusedHit], top_k: int
    ) -> list[RerankedHit]:
        return [
            RerankedHit(
                chunk_id=c.chunk_id,
                text=c.text,
                source=c.source,
                metadata=c.metadata,
                rerank_score=c.fused_score,
                fused_score=c.fused_score,
                dense_score=c.dense_score,
                sparse_score=c.sparse_score,
            )
            for c in candidates[:top_k]
        ]


# Factory + timing wrapper

class Reranker:
    """User-facing reranker — wraps the chosen backend with metrics + timing."""

    def __init__(self) -> None:
        backend = settings.reranker_backend
        if not settings.reranker_enabled or backend == "off":
            self._impl = PassthroughReranker()
            self._name = "passthrough"
        elif backend == "cross_encoder":
            try:
                self._impl = CrossEncoderReranker()
                self._name = "cross_encoder"
            except Exception as exc:
                log.warning(
                    f"cross-encoder unavailable ({exc}); falling back to LLM reranker"
                )
                self._impl = LLMReranker()
                self._name = "llm"
        elif backend == "llm":
            self._impl = LLMReranker()
            self._name = "llm"
        else:
            raise ValueError(f"Unknown reranker backend: {backend}")
        log.info(f"Reranker backend: {self._name}")

    @property
    def name(self) -> str:
        return self._name

    async def rerank(
        self, query: str, candidates: list[FusedHit], top_k: int
    ) -> list[RerankedHit]:
        start = time.perf_counter()
        out = await self._impl.rerank(query, candidates, top_k)
        metrics.rerank_latency_seconds.observe(time.perf_counter() - start)
        return out

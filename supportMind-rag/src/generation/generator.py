"""
generator.py — End-to-end generation orchestrator.

Takes a query, runs the HybridRetriever (rewrite → hybrid → rerank), builds a
prompt, calls the LLM, and returns one fully-traced AnswerEnvelope. The
envelope carries the per-stage timings so the API can show the user exactly
what happened.
"""
import time
from dataclasses import asdict, dataclass

from src.generation.llm_client import GenerationResult, LLMClient
from src.generation.prompts import build_prompt
from src.retrieval.hybrid_retriever import HybridRetriever, RetrievalResult


@dataclass
class AnswerEnvelope:
    answer: str
    sources: list[dict]
    retrieval: dict           # mirror of RetrievalResult, JSON-friendly
    generation: dict
    total_elapsed_s: float


class Generator:
    def __init__(
        self,
        retriever: HybridRetriever | None = None,
        llm: LLMClient | None = None,
    ) -> None:
        self._retriever = retriever or HybridRetriever()
        self._llm = llm or LLMClient()

    @property
    def retriever(self) -> HybridRetriever:
        return self._retriever

    async def answer(self, query: str) -> AnswerEnvelope:
        start = time.perf_counter()

        retrieval = await self._retriever.retrieve(query)
        prompt = build_prompt(query, retrieval.reranked)
        gen: GenerationResult = await self._llm.generate(prompt)

        elapsed = time.perf_counter() - start

        return AnswerEnvelope(
            answer=gen.answer,
            sources=prompt.sources,
            retrieval=_serialize_retrieval(retrieval, self._retriever.reranker_name),
            generation={
                "model": gen.model,
                "prompt_tokens": gen.prompt_tokens,
                "completion_tokens": gen.completion_tokens,
                "total_tokens": gen.total_tokens,
                "finish_reason": gen.finish_reason,
            },
            total_elapsed_s=round(elapsed, 3),
        )


def _serialize_retrieval(r: RetrievalResult, reranker_name: str) -> dict:
    return {
        "rewriter": {
            "original": r.rewritten.original,
            "variants": r.rewritten.variants,
            "hyde_passage": r.rewritten.hyde_passage,
            "elapsed_s": round(r.rewritten.elapsed_s, 3),
        },
        "dense_candidates": r.dense_candidates,
        "sparse_candidates": r.sparse_candidates,
        "fused_top": [
            {
                "chunk_id": h.chunk_id,
                "source": h.source,
                "fused_score": round(h.fused_score, 4),
                "dense_score": round(h.dense_score, 4) if h.dense_score is not None else None,
                "sparse_score": round(h.sparse_score, 4) if h.sparse_score is not None else None,
                "contributing_queries": h.contributing_queries,
                "section_path": h.metadata.get("section_path", ""),
            }
            for h in r.fused[:10]
        ],
        "reranked_top": [
            {
                "chunk_id": h.chunk_id,
                "source": h.source,
                "rerank_score": round(h.rerank_score, 4),
                "fused_score": round(h.fused_score, 4),
                "section_path": h.metadata.get("section_path", ""),
            }
            for h in r.reranked
        ],
        "reranker": reranker_name,
        "elapsed_s": round(r.elapsed_s, 3),
    }

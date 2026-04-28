"""
reranker.py — Optional cross-encoder reranker (improves Naive RAG precision).

In Naive RAG the retriever uses bi-encoder embeddings (fast but approximate).
The reranker uses a cross-encoder to re-score the top-k results with the full
(query, chunk) pair — much more accurate but slower.

Set RERANKER_ENABLED=true in your .env to activate.
Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (free, ~25ms per pair on CPU)
"""
from __future__ import annotations

from src.retrieval.vector_store import SearchResult
from src.utils.config import get_settings
from src.utils.logging import get_logger

log = get_logger(__name__)
settings = get_settings()


class CrossEncoderReranker:
    """Reranks retrieved chunks using a cross-encoder model.

    Only activated when settings.reranker_enabled is True.
    Falls back to original order if model unavailable.
    """

    MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self) -> None:
        self._model = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
            self._model = CrossEncoder(self.MODEL_NAME)
            log.info("Cross-encoder reranker loaded", model=self.MODEL_NAME)
        except ImportError:
            log.warning(
                "sentence-transformers not installed; reranker disabled. "
                "pip install sentence-transformers"
            )

    def rerank(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        """Re-score results with the cross-encoder and return re-sorted list."""
        if not self._model or not results:
            return results

        pairs = [(query, r.text) for r in results]
        scores = self._model.predict(pairs).tolist()

        for result, score in zip(results, scores):
            result.metadata["rerank_score"] = score
            result.metadata["bi_encoder_score"] = result.score
            result.score = float(score)

        reranked = sorted(results, key=lambda r: r.score, reverse=True)
        log.debug(
            "Reranked results",
            original_top=results[0].source if results else None,
            reranked_top=reranked[0].source if reranked else None,
        )
        return reranked


_reranker: CrossEncoderReranker | None = None


def get_reranker() -> CrossEncoderReranker | None:
    if not settings.reranker_enabled:
        return None
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoderReranker()
    return _reranker

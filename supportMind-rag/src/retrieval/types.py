"""
types.py — Shared dataclasses used across the retrieval pipeline.

Lives in its own module to break the cycle between hybrid_retriever (produces
FusedHit, calls Reranker) and reranker (consumes FusedHit, produces RerankedHit).
"""
from dataclasses import dataclass


@dataclass
class FusedHit:
    """A candidate that survived RRF fusion. Carries provenance so the API can
    surface whether each hit came from BM25, dense, or both."""

    chunk_id: str
    text: str
    source: str
    metadata: dict
    fused_score: float
    dense_score: float | None
    sparse_score: float | None
    contributing_queries: list[str]


@dataclass
class RerankedHit:
    """Final candidate after the reranker — ordered by `rerank_score`."""

    chunk_id: str
    text: str
    source: str
    metadata: dict
    rerank_score: float
    fused_score: float
    dense_score: float | None
    sparse_score: float | None

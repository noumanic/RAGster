"""
hybrid_retriever.py — The Advanced RAG retriever.

Pipeline (per user query):
    1. QueryRewriter         original + N variants + HyDE passage
    2. parallel:
         - dense search       embed each query string, search Chroma
         - sparse search      run BM25 over each query string
    3. RRF fusion             merge ranked lists by reciprocal rank
    4. Reranker               cross-encoder or LLM-as-judge picks final top-K

Reciprocal Rank Fusion is unweighted by design: it doesn't need score
calibration between BM25 and cosine similarity (which live on different
scales), it just looks at *rank position*. Formula per doc:
    RRF_score(d) = Σ_q  1 / (k + rank_q(d))
where k=60 is the constant from Cormack et al. (2009) and the sum is over
every query (original + variants + HyDE).
"""
import asyncio
import time
from dataclasses import dataclass

from src.ingestion.chunker import Chunk
from src.retrieval.bm25_index import BM25Hit, BM25Index, get_bm25_index
from src.retrieval.embeddings import EmbeddingService
from src.retrieval.query_rewriter import QueryRewriter, RewrittenQuery
from src.retrieval.reranker import Reranker
from src.retrieval.types import FusedHit, RerankedHit
from src.retrieval.vector_store import (
    BaseVectorStore,
    SearchResult,
    get_vector_store,
)
from src.utils import metrics
from src.utils.config import get_settings
from src.utils.logging import get_logger

log = get_logger(__name__)
settings = get_settings()


@dataclass
class RetrievalResult:
    """Full retrieval trace — answers feed off these but the API also returns
    them so the frontend can show *what happened* per stage."""

    rewritten: RewrittenQuery
    dense_candidates: int
    sparse_candidates: int
    fused: list[FusedHit]
    reranked: list[RerankedHit]
    elapsed_s: float


class HybridRetriever:
    def __init__(
        self,
        vector_store: BaseVectorStore | None = None,
        bm25_index: BM25Index | None = None,
        embedder: EmbeddingService | None = None,
        rewriter: QueryRewriter | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self._vs = vector_store or get_vector_store()
        self._bm25 = bm25_index or get_bm25_index()
        self._embedder = embedder or EmbeddingService()
        self._rewriter = rewriter or QueryRewriter()
        self._reranker = reranker or Reranker()

    @property
    def reranker_name(self) -> str:
        return self._reranker.name

    async def ingest_chunks(self, chunks: list[Chunk]) -> int:
        """Ingest into BOTH the dense store and the BM25 index in lock-step."""
        if not chunks:
            return 0

        embedded = await self._embedder.embed_chunks(chunks)
        # dense and sparse can be written in parallel; sparse is sync but cheap
        n_dense = await self._vs.upsert(embedded)
        self._bm25.add(chunks)
        log.info(
            "Indexed chunks",
            dense=n_dense,
            sparse=self._bm25.count(),
        )
        return n_dense

    async def retrieve(self, query: str) -> RetrievalResult:
        if not query.strip():
            raise ValueError("Query cannot be empty")
        total_start = time.perf_counter()

        # 1. Query rewriting (multi-query + HyDE)
        rewritten = await self._rewriter.rewrite(query)

        # 2. Build the set of search strings
        dense_queries = [rewritten.original, *rewritten.variants]
        if rewritten.hyde_passage:
            dense_queries.append(rewritten.hyde_passage)
        sparse_queries = [rewritten.original, *rewritten.variants]
        # HyDE is excluded from BM25: a hypothetical answer doesn't help keyword
        # matching, it just dilutes rare-term signal.

        # 3. Parallel dense + sparse retrieval, each over every query
        dense_lists, sparse_lists = await asyncio.gather(
            self._dense_multi(dense_queries),
            self._sparse_multi(sparse_queries),
        )

        # 4. RRF fusion
        fuse_start = time.perf_counter()
        fused = self._rrf_fuse(dense_lists, sparse_lists, dense_queries, sparse_queries)
        fused = fused[: settings.hybrid_top_k]
        metrics.fusion_latency_seconds.observe(time.perf_counter() - fuse_start)

        # 5. Rerank
        reranked = await self._reranker.rerank(
            rewritten.original, fused, settings.final_top_k
        )

        elapsed = time.perf_counter() - total_start
        metrics.retrieval_total_latency_seconds.observe(elapsed)

        log.info(
            "Hybrid retrieval complete",
            queries=len(dense_queries),
            dense_total=sum(len(x) for x in dense_lists),
            sparse_total=sum(len(x) for x in sparse_lists),
            fused=len(fused),
            reranked=len(reranked),
            elapsed=round(elapsed, 2),
        )

        return RetrievalResult(
            rewritten=rewritten,
            dense_candidates=sum(len(x) for x in dense_lists),
            sparse_candidates=sum(len(x) for x in sparse_lists),
            fused=fused,
            reranked=reranked,
            elapsed_s=elapsed,
        )

    async def _dense_multi(self, queries: list[str]) -> list[list[SearchResult]]:
        """Embed each query, run vector search, return one ranked list per query."""
        if not queries:
            return []

        embed_start = time.perf_counter()
        vectors = await self._embedder.embed_texts(queries)
        # query_rewrite_latency already covers the rewrite call; this is the
        # embedding portion of the dense pipeline
        log.debug(
            "Embedded query variants",
            count=len(queries),
            elapsed=round(time.perf_counter() - embed_start, 2),
        )

        search_start = time.perf_counter()
        results = await asyncio.gather(
            *(self._vs.search(v, settings.dense_top_k) for v in vectors)
        )
        metrics.dense_search_latency_seconds.observe(time.perf_counter() - search_start)
        return list(results)

    async def _sparse_multi(self, queries: list[str]) -> list[list[BM25Hit]]:
        """Run BM25 over each query (synchronous lib, but cheap)."""
        # rank_bm25 is sync; offload to executor to keep the event loop clean
        loop = asyncio.get_event_loop()
        return await asyncio.gather(
            *(
                loop.run_in_executor(
                    None, self._bm25.search, q, settings.sparse_top_k
                )
                for q in queries
            )
        )

    def _rrf_fuse(
        self,
        dense_lists: list[list[SearchResult]],
        sparse_lists: list[list[BM25Hit]],
        dense_queries: list[str],
        sparse_queries: list[str],
    ) -> list[FusedHit]:
        """Reciprocal Rank Fusion across every (query, hit) pair from both sides."""
        k = settings.rrf_k
        scores: dict[str, float] = {}
        meta: dict[str, FusedHit] = {}

        def _stamp(hit_id: str, hit_score_field: str, value: float):
            existing = meta.get(hit_id)
            if existing is None:
                return
            current = getattr(existing, hit_score_field)
            if current is None or value > current:
                setattr(existing, hit_score_field, value)

        for q_idx, hits in enumerate(dense_lists):
            qstr = dense_queries[q_idx]
            for rank, h in enumerate(hits):
                contrib = 1.0 / (k + rank + 1)
                scores[h.chunk_id] = scores.get(h.chunk_id, 0.0) + contrib
                if h.chunk_id not in meta:
                    meta[h.chunk_id] = FusedHit(
                        chunk_id=h.chunk_id,
                        text=h.text,
                        source=h.source,
                        metadata=h.metadata,
                        fused_score=0.0,
                        dense_score=h.score,
                        sparse_score=None,
                        contributing_queries=[qstr],
                    )
                else:
                    if qstr not in meta[h.chunk_id].contributing_queries:
                        meta[h.chunk_id].contributing_queries.append(qstr)
                    _stamp(h.chunk_id, "dense_score", h.score)

        for q_idx, hits in enumerate(sparse_lists):
            qstr = sparse_queries[q_idx]
            for rank, h in enumerate(hits):
                contrib = 1.0 / (k + rank + 1)
                scores[h.chunk_id] = scores.get(h.chunk_id, 0.0) + contrib
                if h.chunk_id not in meta:
                    meta[h.chunk_id] = FusedHit(
                        chunk_id=h.chunk_id,
                        text=h.text,
                        source=h.source,
                        metadata=h.metadata,
                        fused_score=0.0,
                        dense_score=None,
                        sparse_score=h.score,
                        contributing_queries=[qstr],
                    )
                else:
                    if qstr not in meta[h.chunk_id].contributing_queries:
                        meta[h.chunk_id].contributing_queries.append(qstr)
                    _stamp(h.chunk_id, "sparse_score", h.score)

        for chunk_id, hit in meta.items():
            hit.fused_score = scores[chunk_id]

        return sorted(meta.values(), key=lambda h: h.fused_score, reverse=True)

"""Test the RRF fusion math in HybridRetriever in isolation, without hitting the
real embedding/LLM/Chroma stack."""
from src.retrieval.bm25_index import BM25Hit
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.vector_store import SearchResult


class _FakeRetriever(HybridRetriever):
    def __init__(self):
        # bypass __init__ — we only care about the pure _rrf_fuse method
        pass


def test_rrf_fuse_combines_dense_and_sparse_evidence():
    r = _FakeRetriever()

    # Same chunk surfaced in BOTH lists should outscore singletons.
    dense = [
        [
            SearchResult("c1", "doc 1", "a.md", 0.91, {}),
            SearchResult("c2", "doc 2", "b.md", 0.85, {}),
            SearchResult("c3", "doc 3", "c.md", 0.80, {}),
        ]
    ]
    sparse = [
        [
            BM25Hit("c2", "doc 2", "b.md", 12.4, {}),
            BM25Hit("c4", "doc 4", "d.md", 8.2, {}),
        ]
    ]

    fused = r._rrf_fuse(
        dense_lists=dense,
        sparse_lists=sparse,
        dense_queries=["q1"],
        sparse_queries=["q1"],
    )
    ids = [h.chunk_id for h in fused]
    # c2 appears in both — must rank first
    assert ids[0] == "c2"
    # c2 has both dense and sparse scores stamped
    c2 = next(h for h in fused if h.chunk_id == "c2")
    assert c2.dense_score is not None
    assert c2.sparse_score is not None
    # singletons have only one side populated
    c1 = next(h for h in fused if h.chunk_id == "c1")
    assert c1.dense_score is not None
    assert c1.sparse_score is None
    c4 = next(h for h in fused if h.chunk_id == "c4")
    assert c4.sparse_score is not None
    assert c4.dense_score is None


def test_rrf_fuse_aggregates_across_query_variants():
    r = _FakeRetriever()
    # Same chunk appears in two different query variants — sum of contributions
    dense = [
        [SearchResult("c1", "x", "a.md", 0.9, {})],
        [SearchResult("c1", "x", "a.md", 0.9, {})],
    ]
    sparse = [[], []]
    fused = r._rrf_fuse(dense, sparse, ["q1", "q2"], ["q1", "q2"])
    assert fused[0].chunk_id == "c1"
    # Two queries surfacing it at rank 1 → 1/(60+1) + 1/(60+1)
    assert fused[0].fused_score > 1 / 61
    assert "q1" in fused[0].contributing_queries
    assert "q2" in fused[0].contributing_queries

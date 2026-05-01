"""
metrics.py — Prometheus metric definitions for the Advanced RAG pipeline.

Each retrieval stage gets its own histogram so a Grafana dashboard can show
where latency lives: rewrite → dense → sparse → fuse → rerank → generate.
"""
from prometheus_client import Counter, Gauge, Histogram

# Ingestion
documents_ingested_total = Counter(
    "supportmind_documents_ingested_total",
    "Number of documents ingested",
)
chunks_created_total = Counter(
    "supportmind_chunks_created_total",
    "Number of chunks produced by the chunker",
)

# Embeddings
embedding_latency_seconds = Histogram(
    "supportmind_embedding_latency_seconds",
    "Latency of an embedding API call",
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

# Retrieval — per-stage histograms
query_rewrite_latency_seconds = Histogram(
    "supportmind_query_rewrite_latency_seconds",
    "Latency of the query-rewriter (multi-query + HyDE)",
)
dense_search_latency_seconds = Histogram(
    "supportmind_dense_search_latency_seconds",
    "Latency of the dense vector search",
)
sparse_search_latency_seconds = Histogram(
    "supportmind_sparse_search_latency_seconds",
    "Latency of the BM25 keyword search",
)
fusion_latency_seconds = Histogram(
    "supportmind_fusion_latency_seconds",
    "Latency of the RRF fusion step",
)
rerank_latency_seconds = Histogram(
    "supportmind_rerank_latency_seconds",
    "Latency of the cross-encoder / LLM reranker",
)
retrieval_total_latency_seconds = Histogram(
    "supportmind_retrieval_total_latency_seconds",
    "End-to-end retrieval latency",
)

# Generation
generation_latency_seconds = Histogram(
    "supportmind_generation_latency_seconds",
    "Latency of the LLM generation call",
)
llm_tokens_used = Counter(
    "supportmind_llm_tokens_used_total",
    "LLM tokens consumed",
    ["token_type"],
)
llm_errors_total = Counter(
    "supportmind_llm_errors_total",
    "LLM errors by type",
    ["error_type"],
)

# API
api_requests_total = Counter(
    "supportmind_api_requests_total",
    "API request count",
    ["method", "endpoint", "status_code"],
)
active_queries = Gauge(
    "supportmind_active_queries",
    "Queries currently being processed",
)

# Stores
vector_store_size = Gauge(
    "supportmind_vector_store_size",
    "Vectors held by the dense store",
)
bm25_index_size = Gauge(
    "supportmind_bm25_index_size",
    "Documents held by the BM25 index",
)

"""
metrics.py — Prometheus metrics definitions for the RAG pipeline.
"""
from prometheus_client import Counter, Gauge, Histogram, Summary

# Ingestion
docs_ingested_total = Counter(
    "rag_docs_ingested_total",
    "Total documents ingested into the vector store",
    ["doc_type", "status"],
)

chunks_created_total = Counter(
    "rag_chunks_created_total",
    "Total text chunks created during ingestion",
)

embedding_latency_seconds = Histogram(
    "rag_embedding_latency_seconds",
    "Time to embed a batch of chunks",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
)

# Retrieval
retrieval_latency_seconds = Histogram(
    "rag_retrieval_latency_seconds",
    "Time to retrieve top-k chunks from vector store",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0],
)

retrieval_results_count = Histogram(
    "rag_retrieval_results_count",
    "Number of chunks returned per query",
    buckets=[0, 1, 2, 3, 4, 5, 8, 10],
)

avg_similarity_score = Summary(
    "rag_avg_similarity_score",
    "Average cosine similarity score of retrieved chunks",
)

# Generation
generation_latency_seconds = Histogram(
    "rag_generation_latency_seconds",
    "LLM generation latency (time-to-first-token excluded)",
    buckets=[0.5, 1.0, 2.0, 4.0, 8.0, 15.0, 30.0],
)

llm_tokens_used = Counter(
    "rag_llm_tokens_total",
    "Total LLM tokens consumed",
    ["token_type"],  # prompt | completion
)

llm_errors_total = Counter(
    "rag_llm_errors_total",
    "LLM API errors",
    ["error_type"],
)

# API
api_requests_total = Counter(
    "rag_api_requests_total",
    "Total API requests",
    ["method", "endpoint", "status_code"],
)

api_request_duration_seconds = Histogram(
    "rag_api_request_duration_seconds",
    "API request duration",
    ["endpoint"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

cache_hits_total = Counter(
    "rag_cache_hits_total",
    "Total cache hits",
    ["cache_type"],  # query
)

cache_misses_total = Counter(
    "rag_cache_misses_total",
    "Total cache misses",
    ["cache_type"],
)

# System
vector_store_size = Gauge(
    "rag_vector_store_size",
    "Current number of embeddings in the vector store",
)

active_queries = Gauge(
    "rag_active_queries",
    "Number of queries currently being processed",
)

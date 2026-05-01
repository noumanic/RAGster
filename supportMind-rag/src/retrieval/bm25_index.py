"""
bm25_index.py — Persisted BM25 keyword index over the same chunks the dense
store holds.

Why hybrid search needs BM25:
- Customer-support queries often mention exact product names, error codes,
  version numbers, or CLI flags. Dense embeddings can paraphrase those away
  ("E_47 → general failure"). BM25 ranks by exact token overlap and keeps
  rare/specific terms decisive.
- BM25 + dense fused via RRF is the production default at retrieval-system
  scale (Elastic, Pinecone, Vespa).

Implementation:
- `rank_bm25.BM25Okapi` over a tokenized corpus.
- Tokenizer is intentionally simple: lowercase, alphanumeric + `_/.-` (so
  "E_47", "v2.3", "wi-fi" survive intact), drops a small English stoplist.
- The whole index — corpus, tokens, chunk metadata — is pickled to one file
  so cold start is `pickle.load`, not re-tokenize-the-world.
"""
import pickle
import re
import time
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from src.ingestion.chunker import Chunk
from src.utils import metrics
from src.utils.config import get_settings
from src.utils.logging import get_logger

log = get_logger(__name__)
settings = get_settings()

_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "of", "to", "in", "on", "at",
    "for", "with", "by", "from", "as", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "this", "that", "these",
    "those", "it", "its", "you", "your", "we", "our", "they", "their", "i",
    "me", "my", "what", "which", "who", "whom", "how", "when", "where", "why",
    "can", "could", "should", "would", "may", "might", "will", "shall",
}

_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_./-]*")


def tokenize(text: str) -> list[str]:
    """Lowercase + alphanumeric tokens, preserving common technical chars (._/-).

    Keeps strings like `E_47`, `v2.3`, `usb-c`, `wi-fi`, `ssh-keygen` intact —
    these are the high-signal tokens BM25 leans on.
    """
    return [
        tok.lower()
        for tok in _TOKEN_RE.findall(text)
        if tok.lower() not in _STOPWORDS and len(tok) > 1
    ]


@dataclass
class BM25Hit:
    chunk_id: str
    text: str
    source: str
    score: float
    metadata: dict


class BM25Index:
    """Persisted BM25 keyword index.

    Operations:
      - add(chunks)          incremental — also reloads from disk first
      - search(query, k)     returns top-k BM25 hits
      - save() / load()      pickle round-trip
      - count()              number of indexed chunks
    """

    def __init__(self) -> None:
        self._tokenized: list[list[str]] = []
        self._chunk_meta: list[dict] = []  # parallel to _tokenized
        self._bm25: BM25Okapi | None = None
        self.load()

    def add(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0

        # Allow re-indexing — replace chunks with the same id
        seen_ids = {c.chunk_id for c in chunks}
        if seen_ids:
            kept_idx = [
                i for i, m in enumerate(self._chunk_meta) if m["chunk_id"] not in seen_ids
            ]
            self._tokenized = [self._tokenized[i] for i in kept_idx]
            self._chunk_meta = [self._chunk_meta[i] for i in kept_idx]

        for c in chunks:
            self._tokenized.append(tokenize(c.text))
            self._chunk_meta.append(
                {
                    "chunk_id": c.chunk_id,
                    "text": c.text,
                    "source": c.source,
                    "metadata": {
                        **c.metadata,
                        "source": c.source,
                        "chunk_index": c.chunk_index,
                    },
                }
            )

        self._rebuild()
        self.save()
        metrics.bm25_index_size.set(len(self._tokenized))
        log.info(f"BM25 index: +{len(chunks)} chunks, total {len(self._tokenized)}")
        return len(chunks)

    def search(self, query: str, top_k: int) -> list[BM25Hit]:
        if self._bm25 is None or not self._tokenized:
            return []

        start = time.perf_counter()
        tokens = tokenize(query)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)
        # argsort top_k
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        ranked = [i for i in ranked if scores[i] > 0][:top_k]

        metrics.sparse_search_latency_seconds.observe(time.perf_counter() - start)

        hits = []
        for i in ranked:
            m = self._chunk_meta[i]
            hits.append(
                BM25Hit(
                    chunk_id=m["chunk_id"],
                    text=m["text"],
                    source=m["source"],
                    score=float(scores[i]),
                    metadata=m["metadata"],
                )
            )
        return hits

    def count(self) -> int:
        return len(self._tokenized)

    def save(self) -> None:
        path = settings.bm25_index_path
        path.parent.mkdir(parents=True, exist_ok=True)
        # Stale .gitkeep style: never pickle bm25 itself — re-fit on load is fine.
        with path.open("wb") as fh:
            pickle.dump(
                {"tokenized": self._tokenized, "chunk_meta": self._chunk_meta},
                fh,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    def load(self) -> None:
        path = settings.bm25_index_path
        if not path.exists():
            return
        try:
            with path.open("rb") as fh:
                data = pickle.load(fh)
            self._tokenized = data.get("tokenized", [])
            self._chunk_meta = data.get("chunk_meta", [])
            self._rebuild()
            metrics.bm25_index_size.set(len(self._tokenized))
            log.info(f"BM25 index loaded — {len(self._tokenized)} chunks")
        except Exception as exc:
            log.warning(f"BM25 load failed, starting empty: {exc}")
            self._tokenized = []
            self._chunk_meta = []

    def _rebuild(self) -> None:
        if self._tokenized:
            self._bm25 = BM25Okapi(self._tokenized)
        else:
            self._bm25 = None


_index: BM25Index | None = None


def get_bm25_index() -> BM25Index:
    global _index
    if _index is None:
        _index = BM25Index()
    return _index

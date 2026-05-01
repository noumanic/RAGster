"""
pipeline.py — Orchestrates load → clean → chunk → index.

Indexes go to BOTH the dense vector store AND the BM25 sparse index. Hybrid
retrieval needs both populated in lock-step or recall on one side will lag.
"""
import time

from src.ingestion.chunker import Chunk, get_chunker
from src.ingestion.cleaner import normalize
from src.ingestion.loaders import RawDocument, load_path
from src.utils import metrics
from src.utils.logging import get_logger

log = get_logger(__name__)


class IngestionPipeline:
    def __init__(self, retriever) -> None:
        # `retriever` is a HybridRetriever — exposes ingest_chunks()
        self._retriever = retriever
        self._chunker = get_chunker()

    async def ingest_path(self, path, recursive: bool = True) -> dict:
        """Ingest one file or every supported file under `path`.

        Returns a small report — useful for the API response.
        """
        from pathlib import Path

        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"{path} not found")

        start = time.perf_counter()
        docs: list[RawDocument] = load_path(p) if recursive or p.is_file() else []
        if not recursive and p.is_dir():
            for child in p.iterdir():
                if child.is_file():
                    docs.extend(load_path(child))

        if not docs:
            log.warning(f"No supported documents found at {path}")
            return {"documents": 0, "chunks": 0, "elapsed_s": 0.0}

        for d in docs:
            d.text = normalize(d.text)
        metrics.documents_ingested_total.inc(len(docs))

        all_chunks: list[Chunk] = []
        for d in docs:
            all_chunks.extend(self._chunker.chunk(d))
        metrics.chunks_created_total.inc(len(all_chunks))

        n = await self._retriever.ingest_chunks(all_chunks)
        elapsed = time.perf_counter() - start

        log.info(
            "Ingestion complete",
            documents=len(docs),
            chunks=n,
            elapsed=round(elapsed, 2),
        )
        return {
            "documents": len(docs),
            "chunks": n,
            "elapsed_s": round(elapsed, 2),
            "sources": [d.source for d in docs],
        }

#!/usr/bin/env python
"""
ingest_documents.py — CLI script to bulk-ingest documents into the RAG index.

Usage:
    python scripts/ingest_documents.py --source ./data/raw
    python scripts/ingest_documents.py --source ./data/raw --recursive
    python scripts/ingest_documents.py --source https://example.com/doc.pdf
"""
import argparse
import asyncio
import sys
import time
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.chunker import RecursiveTextChunker
from src.ingestion.cleaner import TextCleaner
from src.ingestion.loader import DocumentLoader
from src.retrieval.retriever import Retriever
from src.utils.config import get_settings
from src.utils.logging import setup_logging, get_logger

settings = get_settings()
setup_logging(settings.log_level, settings.environment)
log = get_logger("ingest_cli")


async def run_ingestion(source: str, recursive: bool, dry_run: bool) -> None:
    loader = DocumentLoader()
    cleaner = TextCleaner()
    chunker = RecursiveTextChunker()
    retriever = Retriever()

    start = time.perf_counter()
    total_docs = 0
    total_chunks = 0
    errors = []
    all_chunks = []

    print(f"\n{'=' * 60}")
    print(f"  LegalMind RAG — Document Ingestion")
    print(f"  Source: {source}")
    print(f"  Recursive: {recursive}")
    print(f"  Dry run: {dry_run}")
    print(f"{'=' * 60}\n")

    if source.startswith("http"):
        raw = await loader.load_url(source)
        clean = cleaner.clean(raw)
        chunks = chunker.chunk(clean)
        all_chunks.extend(chunks)
        total_docs = 1
        total_chunks = len(chunks)
        print(f"  ✓ URL loaded: {len(chunks)} chunks")
    else:
        async for raw_doc in loader.load_path(Path(source), recursive=recursive):
            try:
                clean_doc = cleaner.clean(raw_doc)
                chunks = chunker.chunk(clean_doc)
                all_chunks.extend(chunks)
                total_docs += 1
                total_chunks += len(chunks)
                print(f"  ✓ {Path(raw_doc.source).name}: {len(chunks)} chunks")
            except Exception as e:
                errors.append(f"  ✗ {raw_doc.source}: {e}")
                print(f"  ✗ {raw_doc.source}: {e}")

    if errors:
        print(f"\n  Errors ({len(errors)}):")
        for err in errors:
            print(err)

    print(f"\n  Summary:")
    print(f"    Documents: {total_docs}")
    print(f"    Chunks:    {total_chunks}")
    print(f"    Chunk size: {settings.chunk_size} chars")

    if not dry_run and all_chunks:
        print(f"\n  Embedding and upserting {total_chunks} chunks...")
        n = await retriever.ingest_chunks(all_chunks)
        elapsed = time.perf_counter() - start
        print(f"  ✓ Upserted {n} vectors in {elapsed:.1f}s")
    elif dry_run:
        print("\n  Dry run — skipping upsert")

    print(f"\n{'=' * 60}\n")


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into LegalMind RAG")
    parser.add_argument("--source", required=True, help="File path, directory, or URL")
    parser.add_argument("--recursive", action="store_true", default=True, help="Recurse directories")
    parser.add_argument("--dry-run", action="store_true", help="Parse but don't upsert")
    args = parser.parse_args()

    asyncio.run(run_ingestion(args.source, args.recursive, args.dry_run))


if __name__ == "__main__":
    main()

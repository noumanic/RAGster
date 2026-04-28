"""
chunker.py — Recursive character text splitter with metadata preservation.
Implements the standard Naive RAG chunking strategy:
  - Recursive split on [paragraph, newline, sentence, word, char]
  - Configurable chunk_size and chunk_overlap (in characters)
  - Each chunk tagged with source metadata + chunk index
"""
import re
import uuid
from dataclasses import dataclass, field

from src.ingestion.loader import RawDocument
from src.utils.config import get_settings
from src.utils.logging import get_logger

log = get_logger(__name__)
settings = get_settings()


@dataclass
class Chunk:
    """A single text chunk ready for embedding."""

    chunk_id: str
    text: str
    source: str
    chunk_index: int
    total_chunks: int
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.chunk_id:
            self.chunk_id = str(uuid.uuid4())


class RecursiveTextChunker:
    """Splits text recursively on a hierarchy of separators.

    Splitting hierarchy (tried in order):
      1. Double newline (paragraph break)
      2. Single newline
      3. Sentence end (. ! ?)
      4. Comma / semicolon
      5. Space (word boundary)
      6. Single character (last resort)

    This ensures chunks are semantically meaningful: we prefer to split
    at paragraph boundaries before sentence boundaries.
    """

    SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        min_chunk_length: int | None = None,
    ) -> None:
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.min_chunk_length = min_chunk_length or settings.min_chunk_length

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be < chunk_size ({self.chunk_size})"
            )

    def chunk(self, doc: RawDocument) -> list[Chunk]:
        """Split a document into overlapping chunks."""
        if not doc.text.strip():
            log.warning("Empty document, skipping", source=doc.source)
            return []

        raw_chunks = self._split(doc.text, self.SEPARATORS)
        # Merge small chunks up to chunk_size, then apply overlap
        merged = self._merge_splits(raw_chunks)

        chunks = []
        for i, text in enumerate(merged):
            if len(text.strip()) < self.min_chunk_length:
                continue
            chunks.append(
                Chunk(
                    chunk_id=str(uuid.uuid4()),
                    text=text.strip(),
                    source=doc.source,
                    chunk_index=i,
                    total_chunks=len(merged),
                    metadata={
                        **doc.metadata,
                        "chunk_index": i,
                        "chunk_char_count": len(text.strip()),
                    },
                )
            )

        log.debug(
            "Chunked document",
            source=doc.source,
            total_chunks=len(chunks),
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        return chunks

    def chunk_batch(self, docs: list[RawDocument]) -> list[Chunk]:
        """Chunk a list of documents."""
        all_chunks = []
        for doc in docs:
            all_chunks.extend(self.chunk(doc))
        return all_chunks

    # Private helpers

    def _split(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split text using the first applicable separator."""
        if not separators:
            return [text]

        separator = separators[0]
        remaining_separators = separators[1:]

        if separator == "":
            # Last resort: character-level split
            return list(text)

        splits = text.split(separator)

        result = []
        for split in splits:
            if not split:
                continue
            if len(split) <= self.chunk_size:
                result.append(split)
            else:
                # Recursively split larger pieces
                result.extend(self._split(split, remaining_separators))

        return result

    def _merge_splits(self, splits: list[str]) -> list[str]:
        """Merge small splits into chunks of ~chunk_size with overlap."""
        chunks: list[str] = []
        current_chunks: list[str] = []
        current_length = 0

        for split in splits:
            split_len = len(split)

            if current_length + split_len > self.chunk_size and current_chunks:
                # Flush current buffer as a chunk
                chunk_text = " ".join(current_chunks)
                chunks.append(chunk_text)

                # Keep overlap: pop from front until we're within overlap budget
                while current_chunks and current_length > self.chunk_overlap:
                    removed = current_chunks.pop(0)
                    current_length -= len(removed) + 1

            current_chunks.append(split)
            current_length += split_len + 1  # +1 for the space joiner

        # Don't forget the last batch
        if current_chunks:
            chunks.append(" ".join(current_chunks))

        return chunks

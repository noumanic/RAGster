"""
chunker.py — Two chunking strategies for the Advanced RAG pipeline.

1. RecursiveTextChunker — character-budgeted recursive splitter.
   The baseline strategy. Splits on paragraph → sentence → word → char.

2. SemanticChunker — structure-aware splitter.
   Uses markdown headings as primary boundaries, then sentence packing within
   each section. Each chunk gets `section_path` metadata (e.g. "Setup > Wi-Fi
   troubleshooting") which the BM25 index uses as a strong keyword signal.

The two strategies share the same output type so the rest of the pipeline
doesn't care which one ran.
"""
import re
import uuid
from dataclasses import dataclass, field

from src.ingestion.loaders import RawDocument
from src.utils.config import get_settings
from src.utils.logging import get_logger

log = get_logger(__name__)
settings = get_settings()


@dataclass
class Chunk:
    """A single text chunk ready for embedding + BM25 indexing."""

    chunk_id: str
    text: str
    source: str
    chunk_index: int
    total_chunks: int
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.chunk_id:
            self.chunk_id = str(uuid.uuid4())


# Recursive baseline

class RecursiveTextChunker:
    """Recursive character-budgeted splitter (paragraph → sentence → word → char)."""

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
            raise ValueError("chunk_overlap must be < chunk_size")

    def chunk(self, doc: RawDocument) -> list[Chunk]:
        if not doc.text.strip():
            return []
        raw_splits = self._split(doc.text, self.SEPARATORS)
        merged = self._merge(raw_splits)
        return [
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
                    "chunker": "recursive",
                },
            )
            for i, text in enumerate(merged)
            if len(text.strip()) >= self.min_chunk_length
        ]

    def _split(self, text: str, seps: list[str]) -> list[str]:
        if not seps:
            return [text]
        sep, rest = seps[0], seps[1:]
        if sep == "":
            return list(text)
        out: list[str] = []
        for piece in text.split(sep):
            if not piece:
                continue
            if len(piece) <= self.chunk_size:
                out.append(piece)
            else:
                out.extend(self._split(piece, rest))
        return out

    def _merge(self, splits: list[str]) -> list[str]:
        chunks: list[str] = []
        buf: list[str] = []
        cur = 0
        for s in splits:
            slen = len(s)
            if cur + slen > self.chunk_size and buf:
                chunks.append(" ".join(buf))
                while buf and cur > self.chunk_overlap:
                    cur -= len(buf.pop(0)) + 1
            buf.append(s)
            cur += slen + 1
        if buf:
            chunks.append(" ".join(buf))
        return chunks


# Semantic — heading-aware

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")


class SemanticChunker:
    """Heading-aware chunker for KB articles.

    Algorithm:
      1. Split the doc into sections using markdown headings (#, ##, ...).
         If no headings are present, fall back to a single section.
      2. Build a `section_path` (breadcrumb) for each section using the heading
         hierarchy — e.g. "Setup > Wi-Fi > Reset router".
      3. Within each section, pack sentences greedily up to chunk_size with
         chunk_overlap. Code blocks (```...```) are kept intact — they never
         cross a chunk boundary.

    The chunk text is prefixed with the section_path so dense embeddings get
    extra context, and BM25 ranking gets more keyword signal.
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        min_chunk_length: int | None = None,
    ) -> None:
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.min_chunk_length = min_chunk_length or settings.min_chunk_length

    def chunk(self, doc: RawDocument) -> list[Chunk]:
        if not doc.text.strip():
            return []

        sections = self._split_sections(doc.text)
        chunks: list[Chunk] = []
        idx = 0
        provisional: list[tuple[str, str]] = []  # (section_path, text)

        for section_path, body in sections:
            for piece in self._pack_section(body):
                provisional.append((section_path, piece))

        for section_path, text in provisional:
            text = text.strip()
            if len(text) < self.min_chunk_length:
                continue
            prefixed = f"[{section_path}]\n{text}" if section_path else text
            chunks.append(
                Chunk(
                    chunk_id=str(uuid.uuid4()),
                    text=prefixed,
                    source=doc.source,
                    chunk_index=idx,
                    total_chunks=0,  # filled in below
                    metadata={
                        **doc.metadata,
                        "chunk_index": idx,
                        "section_path": section_path,
                        "chunk_char_count": len(prefixed),
                        "chunker": "semantic",
                    },
                )
            )
            idx += 1

        for c in chunks:
            c.total_chunks = len(chunks)
        return chunks

    def _split_sections(self, text: str) -> list[tuple[str, str]]:
        """Return [(section_path, body), ...] using markdown headings."""
        matches = list(_HEADING_RE.finditer(text))
        if not matches:
            return [("", text)]

        sections: list[tuple[str, str]] = []
        stack: list[str] = []  # heading at each level (1-indexed)

        # leading body before the first heading
        if matches[0].start() > 0:
            head = text[: matches[0].start()].strip()
            if head:
                sections.append(("", head))

        for i, m in enumerate(matches):
            level = len(m.group(1))
            title = m.group(2).strip()
            stack = stack[: level - 1] + [title]
            body_start = m.end()
            body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[body_start:body_end].strip()
            if not body:
                continue
            sections.append((" > ".join(stack), body))
        return sections

    def _pack_section(self, body: str) -> list[str]:
        """Pack sentences (and intact code blocks) into ~chunk_size pieces."""
        units = self._units(body)
        chunks: list[str] = []
        buf: list[str] = []
        cur = 0
        for u in units:
            ulen = len(u)
            if cur + ulen > self.chunk_size and buf:
                chunks.append(" ".join(buf).strip())
                # carry an overlap tail forward
                tail: list[str] = []
                tail_len = 0
                for piece in reversed(buf):
                    if tail_len + len(piece) > self.chunk_overlap:
                        break
                    tail.insert(0, piece)
                    tail_len += len(piece) + 1
                buf = tail
                cur = tail_len
            buf.append(u)
            cur += ulen + 1
        if buf:
            chunks.append(" ".join(buf).strip())
        return chunks

    def _units(self, body: str) -> list[str]:
        """Break a section body into atomic units: sentences + intact code blocks."""
        units: list[str] = []
        i = 0
        while i < len(body):
            if body.startswith("```", i):
                end = body.find("```", i + 3)
                end = end + 3 if end != -1 else len(body)
                units.append(body[i:end].strip())
                i = end
            else:
                # find next code block (or end)
                next_code = body.find("```", i)
                segment_end = next_code if next_code != -1 else len(body)
                segment = body[i:segment_end]
                for sent in _SENTENCE_RE.split(segment):
                    sent = sent.strip()
                    if sent:
                        units.append(sent)
                i = segment_end
        return units


# Factory

def get_chunker():
    """Return SemanticChunker if enabled in settings, otherwise the recursive baseline."""
    if settings.semantic_chunking:
        log.debug("Using SemanticChunker")
        return SemanticChunker()
    log.debug("Using RecursiveTextChunker")
    return RecursiveTextChunker()

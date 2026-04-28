"""
test_ingestion.py — Unit tests for the document ingestion pipeline.
"""
import asyncio
import tempfile
from pathlib import Path

import pytest

from src.ingestion.chunker import RecursiveTextChunker
from src.ingestion.cleaner import TextCleaner, CleanerConfig
from src.ingestion.loader import DocumentLoader, RawDocument


# Loader tests

class TestDocumentLoader:
    def setup_method(self):
        self.loader = DocumentLoader()

    @pytest.mark.asyncio
    async def test_load_text_file(self, tmp_path: Path):
        p = tmp_path / "test.txt"
        p.write_text("Hello, this is a test document with some content.\n\nSecond paragraph.")
        doc = await self.loader.load_file(p)
        assert doc.text == "Hello, this is a test document with some content.\n\nSecond paragraph."
        assert doc.metadata["source_type"] == "txt"
        assert doc.metadata["filename"] == "test.txt"

    @pytest.mark.asyncio
    async def test_load_markdown_file(self, tmp_path: Path):
        p = tmp_path / "test.md"
        p.write_text("# Heading\n\nParagraph content here.")
        doc = await self.loader.load_file(p)
        assert "Heading" in doc.text
        assert doc.metadata["source_type"] == "md"

    @pytest.mark.asyncio
    async def test_load_unsupported_raises(self, tmp_path: Path):
        p = tmp_path / "test.xyz"
        p.write_text("data")
        with pytest.raises(ValueError, match="Unsupported"):
            await self.loader.load_file(p)

    @pytest.mark.asyncio
    async def test_load_directory(self, tmp_path: Path):
        (tmp_path / "a.txt").write_text("Document A content here.")
        (tmp_path / "b.txt").write_text("Document B content here.")
        (tmp_path / "skip.xyz").write_text("Not loaded.")
        docs = [doc async for doc in self.loader.load_path(tmp_path)]
        assert len(docs) == 2
        sources = {Path(d.source).name for d in docs}
        assert sources == {"a.txt", "b.txt"}


# Cleaner tests

class TestTextCleaner:
    def setup_method(self):
        self.cleaner = TextCleaner()

    def _make_doc(self, text: str) -> RawDocument:
        return RawDocument(text=text, source="test.txt")

    def test_removes_boilerplate(self):
        doc = self._make_doc("Real content here.\nPage 5 of 20\nMore real content.")
        cleaned = self.cleaner.clean(doc)
        assert "Page 5 of 20" not in cleaned.text
        assert "Real content here" in cleaned.text

    def test_normalizes_unicode(self):
        doc = self._make_doc("Smart \u2018quotes\u2019 and \u2013 dashes.")
        cleaned = self.cleaner.clean(doc)
        assert "\u2018" not in cleaned.text
        assert "'" in cleaned.text

    def test_collapses_whitespace(self):
        doc = self._make_doc("Line one.\n\n\n\n\nLine two.")
        cleaned = self.cleaner.clean(doc)
        assert "\n\n\n" not in cleaned.text

    def test_strips_page_markers(self):
        doc = self._make_doc("[Page 1]\nSome legal text.\n[Page 2]\nMore text.")
        cleaned = self.cleaner.clean(doc)
        assert "[Page" not in cleaned.text

    def test_filters_short_lines(self):
        config = CleanerConfig(min_line_length=10)
        cleaner = TextCleaner(config)
        doc = self._make_doc("ok\nThis is a real sentence with enough content.\nhi")
        cleaned = cleaner.clean(doc)
        assert "ok\n" not in cleaned.text or "ok" not in cleaned.text.split("\n")


# Chunker tests

class TestRecursiveTextChunker:
    def setup_method(self):
        self.chunker = RecursiveTextChunker(chunk_size=200, chunk_overlap=20, min_chunk_length=10)

    def _make_doc(self, text: str) -> RawDocument:
        return RawDocument(text=text, source="test.txt")

    def test_basic_chunking(self):
        text = ("This is a sentence. " * 30).strip()
        doc = self._make_doc(text)
        chunks = self.chunker.chunk(doc)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.text) <= self.chunker.chunk_size + 50  # small tolerance

    def test_metadata_propagated(self):
        doc = RawDocument(text="A " * 100, source="contract.pdf", metadata={"doc_type": "contract"})
        chunks = self.chunker.chunk(doc)
        for c in chunks:
            assert c.metadata.get("doc_type") == "contract"
            assert c.source == "contract.pdf"

    def test_empty_doc_returns_empty(self):
        doc = self._make_doc("")
        chunks = self.chunker.chunk(doc)
        assert chunks == []

    def test_chunk_ids_unique(self):
        doc = self._make_doc("Word " * 500)
        chunks = self.chunker.chunk(doc)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_invalid_overlap_raises(self):
        with pytest.raises(ValueError, match="chunk_overlap"):
            RecursiveTextChunker(chunk_size=100, chunk_overlap=100)

    def test_chunk_index_assigned(self):
        doc = self._make_doc("Sentence. " * 100)
        chunks = self.chunker.chunk(doc)
        for i, c in enumerate(chunks):
            assert c.chunk_index == i

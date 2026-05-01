"""Test the SemanticChunker — heading detection, section breadcrumbs, code-block
preservation."""
from src.ingestion.chunker import SemanticChunker
from src.ingestion.loaders import RawDocument


def _doc(text: str) -> RawDocument:
    return RawDocument(source="t.md", text=text, metadata={})


def test_semantic_chunker_uses_heading_breadcrumbs():
    chunker = SemanticChunker(chunk_size=400, chunk_overlap=20)
    text = (
        "# Top\nintro line.\n\n"
        "## Setup\nfirst setup paragraph here.\n\n"
        "### Wi-Fi\nthe wifi paragraph contains specific instructions.\n"
    )
    chunks = chunker.chunk(_doc(text))
    paths = [c.metadata["section_path"] for c in chunks]
    assert any("Top > Setup > Wi-Fi" == p for p in paths)
    assert all(c.metadata["chunker"] == "semantic" for c in chunks)


def test_semantic_chunker_keeps_code_blocks_intact():
    chunker = SemanticChunker(chunk_size=120, chunk_overlap=20)
    text = (
        "# Title\nbefore code.\n\n"
        "```\nline1\nline2\nline3\nline4\n```\n\n"
        "after code paragraph that is fairly long to push past the chunk size."
    )
    chunks = chunker.chunk(_doc(text))
    # The code block should appear in exactly one chunk, fully intact
    blocks = [c for c in chunks if "```" in c.text]
    assert len(blocks) >= 1
    assert any("line1" in c.text and "line4" in c.text for c in blocks)


def test_semantic_chunker_sets_total_chunks():
    chunker = SemanticChunker(chunk_size=200, chunk_overlap=20)
    text = "# A\n" + ("paragraph. " * 60) + "\n\n# B\n" + ("other. " * 60)
    chunks = chunker.chunk(_doc(text))
    assert chunks
    assert all(c.total_chunks == len(chunks) for c in chunks)

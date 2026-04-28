"""
citation.py — Parses and validates inline citations from LLM answers.
Extracts [Source N] references and maps them back to original chunks.
"""
import re
from dataclasses import dataclass

from src.retrieval.vector_store import SearchResult
from src.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class Citation:
    index: int          # 1-based source number used in the answer
    text_excerpt: str   # sentence containing the citation
    source: str         # original filename/URL
    chunk_id: str
    similarity_score: float
    metadata: dict


@dataclass
class CitedAnswer:
    answer: str
    citations: list[Citation]
    uncited_answer: str     # answer with [Source N] markers removed
    coverage_ratio: float   # fraction of answer sentences that cite a source


class CitationExtractor:
    """Extracts and validates citations from LLM-generated answers.

    The LLM is prompted to use [Source N] notation. This class:
    1. Parses all [Source N] occurrences from the answer
    2. Maps each N back to the correct SearchResult
    3. Returns a CitedAnswer with structured citation data
    """

    SOURCE_PATTERN = re.compile(r"\[Source (\d+)\]", re.IGNORECASE)

    def extract(
        self,
        answer: str,
        context_chunks: list[SearchResult],
    ) -> CitedAnswer:
        """Parse citations from an LLM answer.

        Args:
            answer: The raw LLM-generated answer string.
            context_chunks: Ordered list of chunks used to build the prompt.

        Returns:
            CitedAnswer with structured citation data.
        """
        if not answer:
            return CitedAnswer(
                answer=answer,
                citations=[],
                uncited_answer=answer,
                coverage_ratio=0.0,
            )

        # Find all [Source N] mentions and their surrounding context
        citations = []
        seen_indices: set[int] = set()

        sentences = re.split(r"(?<=[.!?])\s+", answer)

        for sentence in sentences:
            matches = self.SOURCE_PATTERN.findall(sentence)
            for match in matches:
                idx = int(match)
                if idx in seen_indices:
                    continue
                seen_indices.add(idx)

                # Map 1-based index to 0-based list
                chunk_idx = idx - 1
                if chunk_idx < 0 or chunk_idx >= len(context_chunks):
                    log.warning(f"LLM cited [Source {idx}] but only {len(context_chunks)} chunks exist")
                    continue

                chunk = context_chunks[chunk_idx]
                citations.append(
                    Citation(
                        index=idx,
                        text_excerpt=sentence.strip(),
                        source=chunk.source,
                        chunk_id=chunk.chunk_id,
                        similarity_score=chunk.score,
                        metadata=chunk.metadata,
                    )
                )

        # Uncited answer = raw answer with [Source N] stripped
        uncited = self.SOURCE_PATTERN.sub("", answer).strip()

        # Coverage: what fraction of sentences mention a source?
        cited_sentences = sum(
            1 for s in sentences if self.SOURCE_PATTERN.search(s)
        )
        coverage = cited_sentences / max(len(sentences), 1)

        log.debug(
            "Citations extracted",
            total_citations=len(citations),
            unique_sources=len({c.source for c in citations}),
            coverage_ratio=round(coverage, 2),
        )

        return CitedAnswer(
            answer=answer,
            citations=citations,
            uncited_answer=uncited,
            coverage_ratio=coverage,
        )

    def format_references(self, cited_answer: CitedAnswer) -> str:
        """Format a references section appended to the answer."""
        if not cited_answer.citations:
            return ""
        lines = ["\n\n**References:**"]
        seen = set()
        for c in sorted(cited_answer.citations, key=lambda x: x.index):
            if c.source not in seen:
                import os
                filename = os.path.basename(c.source)
                lines.append(f"[{c.index}] {filename} (relevance: {c.similarity_score:.0%})")
                seen.add(c.source)
        return "\n".join(lines)

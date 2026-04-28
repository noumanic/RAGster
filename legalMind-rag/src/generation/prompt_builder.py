"""
prompt_builder.py — Builds structured prompts for RAG generation.

The prompt follows the standard Naive RAG template:
  1. System prompt: persona + instructions
  2. Context: numbered source chunks with metadata
  3. User question
  4. Answer instruction: cite sources, say "I don't know" if not in context
"""
from dataclasses import dataclass

from src.retrieval.vector_store import SearchResult
from src.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class BuiltPrompt:
    system_prompt: str
    user_message: str
    context_chunks: list[SearchResult]
    estimated_tokens: int


SYSTEM_PROMPT_TEMPLATE = """You are LegalMind, an expert AI assistant for {firm_name}.
Your role is to answer legal questions based strictly on the provided document excerpts.

RULES:
1. Answer ONLY from the provided context. Do NOT use outside knowledge.
2. Always cite your sources using [Source N] notation where N is the source number.
3. If the answer cannot be found in the provided context, say:
   "I cannot find this information in the available documents."
4. If context is partially relevant, use what is relevant and note gaps.
5. Be precise and concise. Avoid hedging language.
6. For contract clauses or legal terms, quote the exact text from the source.

RESPONSE FORMAT:
- Lead with the direct answer
- Follow with supporting evidence from sources
- End with citations in the format: [Source 1: filename, page N]
"""

USER_MESSAGE_TEMPLATE = """CONTEXT DOCUMENTS:
{context_block}

---

QUESTION: {question}

Please answer based on the context above. Cite all sources used."""


class PromptBuilder:
    """Constructs prompts for the RAG generation step.

    Handles:
    - Context window budget management
    - Source numbering and citation formatting
    - Firm-specific system prompt customization
    """

    # Rough tokens-per-character estimate for budget calculation
    CHARS_PER_TOKEN = 4

    def __init__(
        self,
        firm_name: str = "your law firm",
        max_context_tokens: int = 3000,
    ) -> None:
        self.firm_name = firm_name
        self.max_context_tokens = max_context_tokens

    def build(
        self,
        question: str,
        retrieved_chunks: list[SearchResult],
    ) -> BuiltPrompt:
        """Build a complete prompt from question + retrieved chunks.

        Chunks are truncated to fit within max_context_tokens.
        """
        if not retrieved_chunks:
            log.warning("Building prompt with zero retrieved chunks", question=question[:60])

        # Trim chunks to context budget
        trimmed_chunks = self._fit_to_budget(retrieved_chunks)

        context_block = self._format_context(trimmed_chunks)
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(firm_name=self.firm_name)
        user_message = USER_MESSAGE_TEMPLATE.format(
            context_block=context_block,
            question=question,
        )

        total_chars = len(system_prompt) + len(user_message)
        estimated_tokens = total_chars // self.CHARS_PER_TOKEN

        log.debug(
            "Prompt built",
            chunks_used=len(trimmed_chunks),
            estimated_tokens=estimated_tokens,
        )

        return BuiltPrompt(
            system_prompt=system_prompt,
            user_message=user_message,
            context_chunks=trimmed_chunks,
            estimated_tokens=estimated_tokens,
        )

    # Private helpers

    def _fit_to_budget(self, chunks: list[SearchResult]) -> list[SearchResult]:
        """Select as many chunks as fit within the token budget."""
        budget_chars = self.max_context_tokens * self.CHARS_PER_TOKEN
        used = 0
        selected = []
        for chunk in chunks:
            chunk_chars = len(chunk.text) + 200  # overhead for formatting
            if used + chunk_chars > budget_chars:
                break
            selected.append(chunk)
            used += chunk_chars

        if len(selected) < len(chunks):
            log.debug(
                f"Context truncated: {len(selected)}/{len(chunks)} chunks fit in budget"
            )
        return selected

    @staticmethod
    def _format_context(chunks: list[SearchResult]) -> str:
        """Format chunks as a numbered, cited context block."""
        if not chunks:
            return "No relevant documents found."

        parts = []
        for i, chunk in enumerate(chunks, start=1):
            source_label = _make_source_label(chunk, i)
            score_pct = f"{chunk.score * 100:.0f}%"
            parts.append(
                f"[Source {i}] {source_label} (relevance: {score_pct})\n"
                f"{chunk.text}"
            )
        return "\n\n---\n\n".join(parts)


def _make_source_label(chunk: SearchResult, index: int) -> str:
    """Build a human-readable source label from chunk metadata."""
    import os
    filename = os.path.basename(chunk.source) if chunk.source else f"Document {index}"
    page = chunk.metadata.get("page_number") or chunk.metadata.get("chunk_index")
    if page is not None:
        return f"{filename} (section {page})"
    return filename

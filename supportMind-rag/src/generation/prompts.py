"""
prompts.py — Prompt templates for the support-Q&A generator.

The system prompt is strict about citations: every factual claim must reference
[Source N], and the model must say "I don't know" if the context doesn't cover
the question. This is the right behavior for a customer-support Q&A assistant
where hallucinated workarounds cause real downstream tickets.
"""
from dataclasses import dataclass

from src.retrieval.types import RerankedHit
from src.utils.config import get_settings

settings = get_settings()

SYSTEM_PROMPT = """You are SupportMind, a careful customer-support assistant. \
You answer questions using ONLY the knowledge-base passages provided. Follow \
these rules without exception:

1. Cite sources inline. After every factual claim, add a citation like [Source 1] \
or [Source 2, 3]. The numbers refer to the numbered passages in the user message.

2. If the passages do not contain enough information to answer the question, \
say exactly: "I don't have enough information in the knowledge base to answer \
that." Do NOT speculate, do NOT invent steps, and do NOT use outside knowledge.

3. Prefer concrete, actionable steps. Quote exact error codes, command names, \
version numbers, and UI labels from the passages verbatim — do not paraphrase \
them away.

4. If multiple passages disagree, say so and cite each.

5. Keep the answer tight: 2–6 sentences for simple questions, a numbered list \
for procedures."""


@dataclass
class BuiltPrompt:
    system_prompt: str
    user_message: str
    sources: list[dict]  # parallel to [Source N] indices, for the API response


def build_prompt(query: str, hits: list[RerankedHit]) -> BuiltPrompt:
    """Construct the user message with numbered passages + the question."""
    if not hits:
        passages_block = "(no passages retrieved)"
    else:
        parts = []
        for i, h in enumerate(hits, start=1):
            section = h.metadata.get("section_path") or h.metadata.get("title") or ""
            header = f"Source {i} — {h.source}"
            if section:
                header += f" :: {section}"
            parts.append(f"{header}\n{h.text}")
        passages_block = "\n\n---\n\n".join(parts)

    user_message = (
        f"Knowledge-base passages:\n\n{passages_block}\n\n"
        f"---\n\n"
        f"User question: {query}\n\n"
        f"Answer using only the passages above. Cite each claim with [Source N]."
    )

    sources = [
        {
            "index": i,
            "chunk_id": h.chunk_id,
            "source": h.source,
            "section_path": h.metadata.get("section_path", ""),
            "rerank_score": h.rerank_score,
            "fused_score": h.fused_score,
            "dense_score": h.dense_score,
            "sparse_score": h.sparse_score,
            "preview": h.text[:240],
        }
        for i, h in enumerate(hits, start=1)
    ]

    return BuiltPrompt(
        system_prompt=SYSTEM_PROMPT,
        user_message=user_message,
        sources=sources,
    )

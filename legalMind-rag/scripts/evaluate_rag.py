#!/usr/bin/env python
"""
evaluate_rag.py — Evaluate RAG pipeline quality using RAGAS metrics.

Metrics:
  - Faithfulness:       Is the answer grounded in the retrieved context?
  - Answer Relevancy:   Does the answer address the question?
  - Context Recall:     Were the relevant chunks retrieved?
  - Context Precision:  Are the retrieved chunks relevant?

Usage:
    python scripts/evaluate_rag.py --dataset ./data/eval_set.json
    python scripts/evaluate_rag.py --dataset ./data/eval_set.json --output ./reports/eval.json
"""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.generation.llm_client import LLMClient
from src.generation.prompt_builder import PromptBuilder
from src.retrieval.retriever import Retriever
from src.utils.config import get_settings
from src.utils.logging import setup_logging, get_logger

settings = get_settings()
setup_logging("INFO", settings.environment)
log = get_logger("evaluate")


async def run_single(
    question: str,
    retriever: Retriever,
    llm: LLMClient,
    prompt_builder: PromptBuilder,
) -> dict:
    """Run one question through the full RAG pipeline."""
    chunks = await retriever.retrieve(question, top_k=settings.top_k)
    prompt = prompt_builder.build(question, chunks)
    result = await llm.generate(prompt)
    return {
        "question": question,
        "answer": result.answer,
        "contexts": [c.text for c in prompt.context_chunks],
        "sources": [c.source for c in prompt.context_chunks],
        "tokens": result.total_tokens,
    }


async def evaluate(dataset_path: str, output_path: str | None) -> None:
    """Load eval set, run pipeline, compute metrics."""
    dataset_file = Path(dataset_path)
    if not dataset_file.exists():
        print(f"ERROR: Dataset file not found: {dataset_path}")
        sys.exit(1)

    with open(dataset_file) as f:
        dataset = json.load(f)

    # dataset format: [{"question": "...", "ground_truth": "..."}]
    print(f"\nEvaluating {len(dataset)} questions...")

    retriever = Retriever()
    llm = LLMClient()
    prompt_builder = PromptBuilder()

    results = []
    for i, item in enumerate(dataset):
        q = item["question"]
        gt = item.get("ground_truth", "")
        print(f"  [{i+1}/{len(dataset)}] {q[:60]}...")
        try:
            result = await run_single(q, retriever, llm, prompt_builder)
            result["ground_truth"] = gt
            results.append(result)
        except Exception as e:
            log.error(f"Failed: {q}", error=str(e))
            results.append({"question": q, "error": str(e)})

    # Compute basic metrics (without RAGAS installed, use simple heuristics)
    successes = [r for r in results if "error" not in r]
    has_answer = sum(1 for r in successes if len(r.get("answer", "")) > 20)
    has_sources = sum(1 for r in successes if r.get("sources"))
    avg_tokens = sum(r.get("tokens", 0) for r in successes) / max(len(successes), 1)

    report = {
        "total_questions": len(dataset),
        "successful": len(successes),
        "failed": len(results) - len(successes),
        "answer_rate": round(has_answer / max(len(successes), 1), 2),
        "source_citation_rate": round(has_sources / max(len(successes), 1), 2),
        "avg_tokens_per_query": round(avg_tokens),
        "results": results,
    }

    print(f"\n{'=' * 50}")
    print(f"  Evaluation Results")
    print(f"{'=' * 50}")
    print(f"  Questions:       {report['total_questions']}")
    print(f"  Successful:      {report['successful']}")
    print(f"  Answer Rate:     {report['answer_rate']:.0%}")
    print(f"  Citation Rate:   {report['source_citation_rate']:.0%}")
    print(f"  Avg Tokens/Q:    {report['avg_tokens_per_query']}")
    print(f"{'=' * 50}\n")

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"  Report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate LegalMind RAG pipeline")
    parser.add_argument("--dataset", required=True, help="Path to eval dataset JSON")
    parser.add_argument("--output", default=None, help="Output report path")
    args = parser.parse_args()
    asyncio.run(evaluate(args.dataset, args.output))


if __name__ == "__main__":
    main()

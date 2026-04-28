"""
llm_client.py — Async OpenAI LLM client with streaming, retry, and token tracking.
"""
import time
from typing import AsyncIterator

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from src.generation.prompt_builder import BuiltPrompt
from src.utils.config import get_settings
from src.utils.logging import get_logger
from src.utils import metrics

log = get_logger(__name__)
settings = get_settings()


class GenerationResult:
    """Output from the LLM generation step."""

    def __init__(
        self,
        answer: str,
        prompt_tokens: int,
        completion_tokens: int,
        model: str,
        finish_reason: str,
    ) -> None:
        self.answer = answer
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = prompt_tokens + completion_tokens
        self.model = model
        self.finish_reason = finish_reason


class LLMClient:
    """Async OpenAI client wrapper for RAG generation.

    Supports:
    - Standard (buffered) generation
    - Streaming generation (yields tokens as they arrive)
    - Token usage tracking
    - Prometheus metrics
    """

    def __init__(self) -> None:
        if settings.llm_provider == "gemini":
            self._client = AsyncOpenAI(
                api_key=settings.gemini_api_key,
                base_url=settings.gemini_base_url,
                timeout=60.0,
                max_retries=3,
            )
        else:
            self._client = AsyncOpenAI(
                api_key=settings.openai_api_key,
                timeout=60.0,
                max_retries=3,  # OpenAI SDK handles retries with exponential backoff
            )

    async def generate(self, prompt: BuiltPrompt) -> GenerationResult:
        """Generate a non-streaming answer from the built prompt."""
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": prompt.system_prompt},
            {"role": "user", "content": prompt.user_message},
        ]

        start = time.perf_counter()
        metrics.active_queries.inc()
        try:
            response = await self._client.chat.completions.create(
                model=settings.active_llm_model,
                messages=messages,
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens,
                stream=False,
            )
        except Exception as exc:
            error_type = type(exc).__name__
            metrics.llm_errors_total.labels(error_type=error_type).inc()
            log.error("LLM generation failed", error=str(exc))
            raise
        finally:
            metrics.active_queries.dec()

        elapsed = time.perf_counter() - start
        usage = response.usage

        metrics.generation_latency_seconds.observe(elapsed)
        metrics.llm_tokens_used.labels(token_type="prompt").inc(usage.prompt_tokens)
        metrics.llm_tokens_used.labels(token_type="completion").inc(usage.completion_tokens)

        answer = response.choices[0].message.content or ""
        finish = response.choices[0].finish_reason

        log.info(
            "Generation complete",
            model=settings.active_llm_model,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            elapsed=round(elapsed, 2),
            finish_reason=finish,
        )

        return GenerationResult(
            answer=answer,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            model=settings.active_llm_model,
            finish_reason=finish,
        )

    async def generate_stream(
        self, prompt: BuiltPrompt
    ) -> AsyncIterator[str]:
        """Stream tokens as they arrive from the LLM.

        Yields:
            str: Each token/delta as it streams in.
        """
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": prompt.system_prompt},
            {"role": "user", "content": prompt.user_message},
        ]

        metrics.active_queries.inc()
        try:
            stream = await self._client.chat.completions.create(
                model=settings.active_llm_model,
                messages=messages,
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens,
                stream=True,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield delta.content
        finally:
            metrics.active_queries.dec()

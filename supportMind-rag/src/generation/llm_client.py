"""
llm_client.py — Async chat-completion client with token accounting.

Same pattern as the rest of the project: single AsyncOpenAI instance, configured
to point at Gemini's OpenAI-compat endpoint when LLM_PROVIDER=gemini.
"""
import time

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from src.generation.prompts import BuiltPrompt
from src.utils import metrics
from src.utils.config import get_settings
from src.utils.logging import get_logger

log = get_logger(__name__)
settings = get_settings()


class GenerationResult:
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
                max_retries=3,
            )

    async def generate(self, prompt: BuiltPrompt) -> GenerationResult:
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
            )
        except Exception as exc:
            metrics.llm_errors_total.labels(error_type=type(exc).__name__).inc()
            log.error(f"LLM generation failed: {exc}")
            raise
        finally:
            metrics.active_queries.dec()

        elapsed = time.perf_counter() - start
        usage = response.usage
        metrics.generation_latency_seconds.observe(elapsed)
        metrics.llm_tokens_used.labels(token_type="prompt").inc(usage.prompt_tokens)
        metrics.llm_tokens_used.labels(token_type="completion").inc(
            usage.completion_tokens
        )

        return GenerationResult(
            answer=response.choices[0].message.content or "",
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            model=settings.active_llm_model,
            finish_reason=response.choices[0].finish_reason,
        )

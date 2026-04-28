"""
cache.py — Redis cache wrapper with async support.
Provides hash-based query caching for RAG responses.
"""
import hashlib
import json
from typing import Any

import redis.asyncio as aioredis

from src.utils.config import get_settings
from src.utils.logging import get_logger

log = get_logger(__name__)
settings = get_settings()


class CacheClient:
    """Async Redis cache client with JSON serialization."""

    def __init__(self) -> None:
        self._client: aioredis.Redis | None = None

    @property
    def connected(self) -> bool:
        return self._client is not None

    @property
    def redis(self) -> aioredis.Redis:
        if self._client is None:
            raise RuntimeError("CacheClient not connected — call connect() first")
        return self._client

    async def connect(self) -> None:
        self._client = aioredis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
            socket_connect_timeout=3,
            socket_timeout=3,
        )
        await self._client.ping()
        log.info("Redis cache connected", url=settings.redis_url)

    async def disconnect(self) -> None:
        if self._client:
            await self._client.aclose()
            log.info("Redis cache disconnected")

    @staticmethod
    def _make_key(query: str, filters: dict | None = None) -> str:
        """Deterministic cache key from query + filters."""
        payload = json.dumps({"q": query, "f": filters or {}}, sort_keys=True)
        digest = hashlib.sha256(payload.encode()).hexdigest()[:16]
        return f"rag:response:{digest}"

    async def get(self, query: str, filters: dict | None = None) -> dict | None:
        if not settings.cache_enabled or self._client is None:
            return None
        key = self._make_key(query, filters)
        try:
            raw = await self._client.get(key)
            if raw:
                log.debug("Cache hit", key=key)
                return json.loads(raw)
            log.debug("Cache miss", key=key)
            return None
        except Exception as exc:
            log.warning("Cache GET failed", error=str(exc))
            return None

    async def set(
        self,
        query: str,
        response: dict[str, Any],
        filters: dict | None = None,
        ttl: int | None = None,
    ) -> None:
        if not settings.cache_enabled or self._client is None:
            return
        key = self._make_key(query, filters)
        ttl = ttl or settings.cache_ttl_seconds
        try:
            await self._client.setex(key, ttl, json.dumps(response, default=str))
            log.debug("Cache SET", key=key, ttl=ttl)
        except Exception as exc:
            log.warning("Cache SET failed", error=str(exc))

    async def invalidate_prefix(self, prefix: str) -> int:
        """Invalidate all keys matching a prefix pattern."""
        if not self._client:
            return 0
        keys = await self._client.keys(f"{prefix}*")
        if keys:
            return await self._client.delete(*keys)
        return 0

    async def health_check(self) -> bool:
        try:
            return await self._client.ping() if self._client else False
        except Exception:
            return False


# Module-level singleton
_cache: CacheClient | None = None


async def get_cache() -> CacheClient:
    global _cache
    if _cache is None:
        _cache = CacheClient()
        await _cache.connect()
    return _cache

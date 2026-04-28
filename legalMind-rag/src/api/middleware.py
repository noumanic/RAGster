"""
middleware.py — Reusable Starlette middleware for the FastAPI app.

- JWTAuthMiddleware: validates `Authorization: Bearer <token>` and attaches
  the decoded claims to `request.state.user`. Public paths bypass the check.
- RateLimitMiddleware: fixed-window counter in Redis, keyed by client IP +
  route. Returns 429 once the limit for the window is exceeded.

Both classes read their config from `src.utils.config.Settings`. Wire them
into `app` via `app.add_middleware(...)` in `src.api.main`.
"""
from __future__ import annotations

import time
from typing import Iterable

import jwt
from fastapi import status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from src.utils.cache import get_cache
from src.utils.config import get_settings
from src.utils.logging import get_logger

log = get_logger(__name__)


# JWT Auth

DEFAULT_PUBLIC_PATHS: tuple[str, ...] = (
    "/",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/metrics",
    "/api/v1/health",
)


class JWTAuthMiddleware(BaseHTTPMiddleware):
    """Validate a Bearer JWT and stash decoded claims on `request.state.user`.

    Paths in `public_paths` are not authenticated. Requests with a missing or
    invalid token to a protected path get a 401.
    """

    def __init__(
        self,
        app,
        public_paths: Iterable[str] = DEFAULT_PUBLIC_PATHS,
    ) -> None:
        super().__init__(app)
        self.public_paths = tuple(public_paths)
        self._settings = get_settings()

    async def dispatch(self, request: Request, call_next) -> Response:
        if self._is_public(request.url.path):
            return await call_next(request)

        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            return _json_error(
                status.HTTP_401_UNAUTHORIZED,
                "Missing or malformed Authorization header",
            )

        token = auth.removeprefix("Bearer ").strip()
        try:
            claims = jwt.decode(
                token,
                self._settings.secret_key,
                algorithms=[self._settings.jwt_algorithm],
            )
        except jwt.ExpiredSignatureError:
            return _json_error(status.HTTP_401_UNAUTHORIZED, "Token expired")
        except jwt.InvalidTokenError as e:
            log.warning("JWT validation failed", reason=str(e))
            return _json_error(status.HTTP_401_UNAUTHORIZED, "Invalid token")

        request.state.user = claims
        return await call_next(request)

    def _is_public(self, path: str) -> bool:
        return any(path == p or path.startswith(p + "/") for p in self.public_paths)


# Rate Limiting


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Fixed-window rate limiter backed by Redis.

    Key: `ratelimit:{client_ip}:{path}:{window_start}`. Counter is INCR'd and
    given a TTL equal to the window. Once the limit is exceeded, return 429
    with `Retry-After` set to the remaining seconds in the window.
    """

    def __init__(
        self,
        app,
        requests: int | None = None,
        window_seconds: int | None = None,
    ) -> None:
        super().__init__(app)
        s = get_settings()
        self.limit = requests or s.rate_limit_requests
        self.window = window_seconds or s.rate_limit_window_seconds

    async def dispatch(self, request: Request, call_next) -> Response:
        cache = await get_cache()
        if not cache.connected:
            return await call_next(request)

        now = int(time.time())
        window_start = now - (now % self.window)
        client_ip = request.client.host if request.client else "anon"
        key = f"ratelimit:{client_ip}:{request.url.path}:{window_start}"

        try:
            count = await cache.redis.incr(key)
            if count == 1:
                await cache.redis.expire(key, self.window)
        except Exception as e:
            log.warning("Rate limit Redis error — allowing request", error=str(e))
            return await call_next(request)

        if count > self.limit:
            retry_after = self.window - (now - window_start)
            return _json_error(
                status.HTTP_429_TOO_MANY_REQUESTS,
                f"Rate limit exceeded: {self.limit} requests per {self.window}s",
                headers={"Retry-After": str(retry_after)},
            )

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self.limit)
        response.headers["X-RateLimit-Remaining"] = str(max(0, self.limit - count))
        return response


# helpers


def _json_error(
    status_code: int,
    detail: str,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": detail},
        headers=headers,
    )

"""
logging.py — Structured JSON logging via Loguru.
Provides correlation ID injection and request-scoped log context.
"""
import json
import sys
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any

from loguru import logger

# Per-request correlation ID stored in a ContextVar
_correlation_id: ContextVar[str] = ContextVar("correlation_id", default="")


def get_correlation_id() -> str:
    return _correlation_id.get() or str(uuid.uuid4())


def set_correlation_id(cid: str) -> None:
    _correlation_id.set(cid)


def _json_serializer(record: dict[str, Any]) -> str:
    """Format each log record as a single-line JSON string."""
    log_entry = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "level": record["level"].name,
        "message": record["message"],
        "module": record["module"],
        "function": record["function"],
        "line": record["line"],
        "correlation_id": get_correlation_id(),
    }
    # Merge any extra kwargs passed to logger.bind(...)
    log_entry.update(record.get("extra", {}))

    # Include exception info if present
    if record["exception"]:
        exc = record["exception"]
        log_entry["exception"] = {
            "type": exc.type.__name__ if exc.type else None,
            "value": str(exc.value) if exc.value else None,
        }
    return json.dumps(log_entry, default=str)


def setup_logging(log_level: str = "INFO", environment: str = "development") -> None:
    """Configure loguru for the application.

    In development: colorized human-readable format.
    In staging/production: structured JSON to stdout.
    """
    logger.remove()  # Remove default handler

    if environment == "development":
        fmt = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<yellow>cid={extra[cid]}</yellow> — "
            "<level>{message}</level>"
        )
        logger.configure(extra={"cid": ""})
        logger.add(
            sys.stdout,
            format=fmt,
            level=log_level,
            colorize=True,
            backtrace=True,
            diagnose=True,
        )
    else:
        logger.add(
            sys.stdout,
            format="{message}",
            level=log_level,
            serialize=False,
            colorize=False,
            backtrace=False,
            diagnose=False,
            filter=lambda r: True,
            sink=lambda msg: print(_json_serializer(msg.record)),
        )

    logger.info(
        f"Logging initialized",
        environment=environment,
        level=log_level,
    )


def get_logger(name: str):
    """Return a logger bound to the given module name."""
    return logger.bind(module=name)

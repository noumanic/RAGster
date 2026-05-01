"""
logging.py — Loguru-based structured logger with correlation IDs.

`set_correlation_id` stores an ID per task/request so log lines can be grouped.
`get_logger(__name__)` returns a logger that automatically attaches the current ID.
"""
import contextvars
import sys
from loguru import logger

_correlation_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "correlation_id", default="-"
)


def set_correlation_id(cid: str) -> None:
    _correlation_id.set(cid)


def get_correlation_id() -> str:
    return _correlation_id.get()


def setup_logging(level: str = "INFO", environment: str = "development") -> None:
    logger.remove()
    if environment == "development":
        fmt = (
            "<green>{time:HH:mm:ss}</green> "
            "<level>{level:<7}</level> "
            "<cyan>[{extra[cid]}]</cyan> "
            "<dim>{name}</dim> | {message}"
        )
        logger.add(
            sys.stdout,
            level=level,
            format=fmt,
            colorize=True,
            backtrace=False,
            diagnose=False,
        )
    else:
        logger.add(sys.stdout, level=level, serialize=True)

    logger.configure(extra={"cid": "-"})


def get_logger(name: str):
    """Return a bound logger that always emits the current correlation ID."""

    class _Bound:
        def _bind(self):
            return logger.bind(name=name, cid=get_correlation_id())

        def info(self, msg, **kw):
            self._bind().info(msg, **kw)

        def debug(self, msg, **kw):
            self._bind().debug(msg, **kw)

        def warning(self, msg, **kw):
            self._bind().warning(msg, **kw)

        def error(self, msg, **kw):
            self._bind().error(msg, **kw)

        def exception(self, msg, **kw):
            self._bind().exception(msg, **kw)

    return _Bound()

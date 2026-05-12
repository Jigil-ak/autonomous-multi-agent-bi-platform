"""
Retry Handler with Exponential Backoff
========================================
Provides both synchronous and asynchronous retry wrappers for resilient
API calls, especially important for Gemini Pro rate-limit protection.

Features:
- Configurable retry delays (default: [2, 4, 8] seconds)
- Exception logging on each retry attempt
- Final exception propagation or graceful None return
- Async variant for FastAPI / async workflows
"""

import asyncio
import time
import functools
from typing import Any, Callable, TypeVar

from config import RETRY_DELAYS, MAX_RETRIES
from utils.logger import get_logger

logger = get_logger("retry_handler")

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Synchronous Retry
# ---------------------------------------------------------------------------
def retry_with_backoff(
    func: Callable[..., T] | None = None,
    *,
    max_retries: int = MAX_RETRIES,
    delays: list[int] | None = None,
    raise_on_failure: bool = False,
    fallback_value: Any = None,
) -> Any:
    """
    Decorator / wrapper that retries a function with exponential backoff.

    Can be used as a decorator::

        @retry_with_backoff
        def call_api():
            ...

    Or with arguments::

        @retry_with_backoff(max_retries=5, raise_on_failure=True)
        def call_api():
            ...

    Parameters
    ----------
    func : callable, optional
        The function to wrap (auto-supplied when used as bare decorator).
    max_retries : int
        Maximum number of retry attempts.
    delays : list[int]
        Sleep durations between retries (cycles if retries > len(delays)).
    raise_on_failure : bool
        If True, re-raise the last exception after exhausting retries.
        If False, return ``fallback_value``.
    fallback_value : Any
        Value to return when all retries are exhausted and raise_on_failure
        is False.
    """
    if delays is None:
        delays = RETRY_DELAYS

    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> T | Any:
            last_exception: Exception | None = None

            for attempt in range(1, max_retries + 1):
                try:
                    return fn(*args, **kwargs)
                except Exception as exc:
                    last_exception = exc
                    delay = delays[(attempt - 1) % len(delays)]
                    logger.warning(
                        f"Attempt {attempt}/{max_retries} for "
                        f"{fn.__name__} failed: {exc}. "
                        f"Retrying in {delay}s..."
                    )
                    time.sleep(delay)

            # All retries exhausted
            logger.error(
                f"All {max_retries} retries exhausted for {fn.__name__}. "
                f"Last error: {last_exception}"
            )
            if raise_on_failure:
                raise last_exception  # type: ignore[misc]
            return fallback_value

        return wrapper  # type: ignore[return-value]

    # Support both @retry_with_backoff and @retry_with_backoff(...)
    if func is not None:
        return decorator(func)
    return decorator


# ---------------------------------------------------------------------------
# Asynchronous Retry
# ---------------------------------------------------------------------------
def async_retry_with_backoff(
    func: Callable[..., Any] | None = None,
    *,
    max_retries: int = MAX_RETRIES,
    delays: list[int] | None = None,
    raise_on_failure: bool = False,
    fallback_value: Any = None,
) -> Any:
    """
    Async version of :func:`retry_with_backoff`.

    Usage::

        @async_retry_with_backoff(max_retries=3)
        async def call_api():
            ...
    """
    if delays is None:
        delays = RETRY_DELAYS

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception: Exception | None = None

            for attempt in range(1, max_retries + 1):
                try:
                    return await fn(*args, **kwargs)
                except Exception as exc:
                    last_exception = exc
                    delay = delays[(attempt - 1) % len(delays)]
                    logger.warning(
                        f"[async] Attempt {attempt}/{max_retries} for "
                        f"{fn.__name__} failed: {exc}. "
                        f"Retrying in {delay}s..."
                    )
                    await asyncio.sleep(delay)

            logger.error(
                f"[async] All {max_retries} retries exhausted for "
                f"{fn.__name__}. Last error: {last_exception}"
            )
            if raise_on_failure:
                raise last_exception  # type: ignore[misc]
            return fallback_value

        return wrapper

    if func is not None:
        return decorator(func)
    return decorator


# ---------------------------------------------------------------------------
# One-shot retry helper (non-decorator)
# ---------------------------------------------------------------------------
def call_with_retry(
    fn: Callable[..., T],
    *args: Any,
    max_retries: int = MAX_RETRIES,
    delays: list[int] | None = None,
    fallback_value: Any = None,
    **kwargs: Any,
) -> T | Any:
    """
    Call ``fn(*args, **kwargs)`` with retry logic.

    Useful when you don't want to decorate a function but still want
    retry behavior on a specific call.
    """
    if delays is None:
        delays = RETRY_DELAYS

    last_exception: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            last_exception = exc
            delay = delays[(attempt - 1) % len(delays)]
            logger.warning(
                f"call_with_retry: attempt {attempt}/{max_retries} for "
                f"{fn.__name__} failed: {exc}. Retrying in {delay}s..."
            )
            time.sleep(delay)

    logger.error(
        f"call_with_retry: all retries exhausted for {fn.__name__}. "
        f"Last error: {last_exception}"
    )
    return fallback_value

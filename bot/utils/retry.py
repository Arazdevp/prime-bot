"""Async retry utility with exponential backoff and jitter."""

import asyncio
import logging
import random
from typing import Any, Callable, Coroutine, TypeVar
from bot.services.llm_client import LLMError, LLMTruncationError

logger = logging.getLogger(__name__)

T = TypeVar("T")


async def async_retry(
    fn: Callable[[], Coroutine[Any, Any, T]],
    retries: int = 3,
    base_delay: float = 2.0,
    rate_limit_base_delay: float = 20.0,
    on_rate_limit: Callable[[float], None] | None = None,
) -> T:
    """Execute async function with exponential backoff for retryable errors.

    Retries only on retryable LLMErrors (e.g. 408, 429, 5xx, network timeouts).
    Uses rate-limit-aware backoff (honoring Retry-After or spanning >60s) on HTTP 429
    so per-minute RPM/TPM quotas can reset before retrying.
    Fails fast immediately on non-retryable errors (e.g. 400, 401, 403, safety blocks)
    and LLMTruncationError (which is handled with a single retry by the summarizer).
    """
    attempt = 0
    while True:
        try:
            return await fn()
        except LLMError as exc:
            if isinstance(exc, LLMTruncationError) or not exc.is_retryable or attempt >= retries:
                if exc.status_code == 429 and on_rate_limit is not None:
                    on_rate_limit(exc.retry_after if exc.retry_after else 30.0)
                if not isinstance(exc, LLMTruncationError):
                    logger.error(
                        "LLM call failed permanently (attempt %s/%s, status %s): %s",
                        attempt + 1,
                        retries + 1,
                        exc.status_code,
                        exc.message,
                    )
                raise

            attempt += 1
            if exc.status_code == 429:
                if exc.retry_after is not None and exc.retry_after > 0:
                    delay = min(120.0, max(1.0, float(exc.retry_after))) + random.uniform(0.5, 1.5)
                else:
                    # Rate-limit backoff: 20s, 35s, 50s... (spans >60s TPM/RPM reset window)
                    delay = min(90.0, rate_limit_base_delay + 15.0 * (attempt - 1)) + random.uniform(0.5, 2.0)
                if on_rate_limit is not None:
                    on_rate_limit(delay)
            else:
                # Exponential backoff: 2s, 4s, 8s... + uniform jitter [0.1, 0.5]
                delay = (base_delay * (2 ** (attempt - 1))) + random.uniform(0.1, 0.5)
            logger.warning(
                "Retryable LLM error on attempt %s/%s (status %s): %s. Retrying in %.2fs...",
                attempt,
                retries,
                exc.status_code,
                exc.message,
                delay,
            )
            await asyncio.sleep(delay)
        except Exception as exc:
            # If unexpected generic exception, retry up to limit
            if attempt >= retries:
                logger.error(
                    "Non-LLM exception failed permanently after %s retries: %s",
                    retries,
                    exc,
                )
                raise
            attempt += 1
            delay = (base_delay * (2 ** (attempt - 1))) + random.uniform(0.1, 0.5)
            logger.warning(
                "Unexpected exception on attempt %s/%s: %s. Retrying in %.2fs...",
                attempt,
                retries,
                exc,
                delay,
            )
            await asyncio.sleep(delay)

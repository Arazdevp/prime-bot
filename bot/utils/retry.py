"""Async retry utility with exponential backoff and jitter."""

import asyncio
import logging
import random
from typing import Any, Callable, Coroutine, TypeVar
from bot.services.llm_client import LLMError

logger = logging.getLogger(__name__)

T = TypeVar("T")


async def async_retry(
    fn: Callable[[], Coroutine[Any, Any, T]],
    retries: int = 3,
    base_delay: float = 2.0,
) -> T:
    """Execute async function with exponential backoff for retryable errors.

    Retries only on retryable LLMErrors (e.g. 429, 5xx, network timeouts).
    Fails fast immediately on non-retryable errors (e.g. 400, 401, 403, safety blocks).
    """
    attempt = 0
    while True:
        try:
            return await fn()
        except LLMError as exc:
            if not exc.is_retryable or attempt >= retries:
                logger.error(
                    "LLM call failed permanently (attempt %s/%s, status %s): %s",
                    attempt + 1,
                    retries + 1,
                    exc.status_code,
                    exc.message,
                )
                raise

            attempt += 1
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

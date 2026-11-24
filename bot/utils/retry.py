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
    throttle_retries: int | None = None,
    on_rate_limit: Callable[[float], None] | None = None,
    on_in_flight_budget: Callable[[float], None] | None = None,
) -> T:
    """Execute async function with exponential backoff for retryable errors.

    Retries only on retryable LLMErrors (e.g. 408, 429, transient OpenRouter 402
    in-flight budget exhaustion, 5xx, network timeouts).
    Uses throttle-aware backoff (honoring Retry-After up to 300s or escalating across
    the >60s RPM/TPM and credit settlement window) on 429 and transient 402 errors.
    Fails fast immediately on non-retryable errors (e.g. 400, 401, 403, permanent 402
    credit exhaustion, safety blocks) and LLMTruncationError (handled by the summarizer).
    """
    effective_throttle_retries = retries if throttle_retries is None else throttle_retries
    attempt = 0
    while True:
        try:
            return await fn()
        except LLMError as exc:
            is_in_flight = bool(getattr(exc, "is_in_flight_budget", False))
            is_throttle = bool(getattr(exc, "is_throttle", False) or exc.status_code == 429 or is_in_flight)
            max_attempts = effective_throttle_retries if is_throttle else retries

            if isinstance(exc, LLMTruncationError) or not exc.is_retryable or attempt >= max_attempts:
                if is_throttle:
                    cooldown = (
                        float(exc.retry_after)
                        if exc.retry_after is not None and exc.retry_after > 0
                        else (45.0 if is_in_flight else 30.0)
                    )
                    if on_rate_limit is not None:
                        on_rate_limit(cooldown)
                    if is_in_flight and on_in_flight_budget is not None:
                        on_in_flight_budget(cooldown)
                if not isinstance(exc, LLMTruncationError):
                    logger.error(
                        "LLM call failed permanently (attempt %s/%s, status %s): %s",
                        attempt + 1,
                        max_attempts + 1,
                        exc.status_code,
                        exc.message,
                    )
                raise

            attempt += 1
            if is_throttle:
                if exc.retry_after is not None and exc.retry_after > 0:
                    delay = min(300.0, max(1.0, float(exc.retry_after))) + random.uniform(0.5, 1.5)
                else:
                    # Throttle/settlement backoff: 20s, 35s, 50s, 65s, 80s... (spans >60s reset/settle window)
                    delay = min(120.0, rate_limit_base_delay + 15.0 * (attempt - 1)) + random.uniform(0.5, 2.0)
                if on_rate_limit is not None:
                    on_rate_limit(delay)
                if is_in_flight and on_in_flight_budget is not None:
                    on_in_flight_budget(delay)
            else:
                # Exponential backoff: 2s, 4s, 8s... + uniform jitter [0.1, 0.5]
                delay = (base_delay * (2 ** (attempt - 1))) + random.uniform(0.1, 0.5)
            logger.warning(
                "Retryable LLM error on attempt %s/%s (status %s): %s. Retrying in %.2fs...",
                attempt,
                max_attempts,
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

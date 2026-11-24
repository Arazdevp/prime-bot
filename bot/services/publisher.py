"""Safe Telegram message publisher handling splitting, rate limits, and HTML parsing fallbacks."""

import asyncio
import html
import logging
import re
from typing import Any
from aiogram import Bot
from aiogram.exceptions import (
    TelegramBadRequest,
    TelegramNetworkError,
    TelegramRetryAfter,
    TelegramServerError,
)
from bot.config import get_settings
from bot.utils.general_topic import to_send_thread_id
from bot.utils.text_split import TELEGRAM_MESSAGE_MAX_CHARS, smart_split

logger = logging.getLogger(__name__)


def strip_html_tags(text: str) -> str:
    """Strip all HTML tags and unescape entities for fallback plain-text sending."""
    clean = re.sub(r"<[^>]+>", "", text)
    return html.unescape(clean)


async def send_summary_parts(
    bot: Bot,
    chat_id: int,
    thread_id: int | None,
    parts: list[str],
    start_index: int = 0,
    on_part_sent: Any | None = None,
) -> tuple[bool, int]:
    """Send pre-split summary parts starting from start_index.

    Returns (all_delivered: bool, sent_count: int).
    Invokes on_part_sent(sent_count) after each successfully delivered part.
    """
    if not parts:
        return False, 0

    if start_index >= len(parts):
        return True, len(parts)

    settings = get_settings()
    current_thread_id: int | None = to_send_thread_id(thread_id)
    sent_count = start_index

    for i in range(start_index, len(parts)):
        part = parts[i]
        delivered = False
        use_html = True
        retry_count = 0
        max_retries = 5

        while not delivered and retry_count < max_retries:
            send_text = part if use_html else strip_html_tags(part)
            send_parse_mode = "HTML" if use_html else None
            try:
                await bot.send_message(
                    chat_id=chat_id,
                    message_thread_id=current_thread_id,
                    text=send_text,
                    parse_mode=send_parse_mode,
                    disable_web_page_preview=True,
                )
                delivered = True

            except TelegramRetryAfter as exc:
                retry_count += 1
                wait_time = exc.retry_after + 1
                logger.warning(
                    "Telegram flood control hit. Sleeping %s seconds before retry %s/%s...",
                    wait_time,
                    retry_count,
                    max_retries,
                )
                await asyncio.sleep(wait_time)

            except (TelegramNetworkError, TelegramServerError, asyncio.TimeoutError) as exc:
                retry_count += 1
                if retry_count >= max_retries:
                    logger.error(
                        "Transient Telegram network/server error persisted for chunk %s/%s: %s",
                        i + 1,
                        len(parts),
                        exc,
                    )
                    return False, sent_count
                backoff = min(16.0, 2.0 * (2 ** (retry_count - 1)))
                logger.warning(
                    "Transient Telegram error on chunk %s/%s (%s). Retrying %s/%s in %.1fs...",
                    i + 1,
                    len(parts),
                    exc,
                    retry_count,
                    max_retries,
                    backoff,
                )
                await asyncio.sleep(backoff)

            except TelegramBadRequest as exc:
                error_msg = str(exc).lower()

                # Handle entity parsing errors -> retry cleanly as plain text within the same loop
                if use_html and (
                    "can't parse entities" in error_msg
                    or "unsupported start tag" in error_msg
                    or "parsing" in error_msg
                ):
                    logger.warning(
                        "HTML parse error from Telegram (%s). Falling back to plain text for chunk %s/%s.",
                        exc,
                        i + 1,
                        len(parts),
                    )
                    use_html = False
                    continue

                # Handle deleted or non-existent thread -> fall back to General topic
                elif "message thread not found" in error_msg or "thread_not_found" in error_msg:
                    if current_thread_id is not None:
                        logger.warning(
                            "Topic thread %s not found in chat %s. Falling back to General topic.",
                            current_thread_id,
                            chat_id,
                        )
                        current_thread_id = None
                        retry_count += 1
                        continue
                    else:
                        logger.error("Failed sending to chat %s even in General: %s", chat_id, exc)
                        return False, sent_count
                else:
                    logger.error("TelegramBadRequest while sending summary: %s", exc)
                    return False, sent_count

            except Exception as exc:
                logger.error("Unexpected error delivering summary chunk %s/%s: %s", i + 1, len(parts), exc)
                return False, sent_count

        if not delivered:
            logger.error("Failed to deliver chunk %s/%s after %s retries.", i + 1, len(parts), max_retries)
            return False, sent_count

        sent_count = i + 1
        if on_part_sent:
            try:
                res = on_part_sent(sent_count)
                if asyncio.iscoroutine(res):
                    await res
            except Exception as exc:
                logger.warning("Error in on_part_sent callback: %s", exc)

        # Sleep configured delay between parts to avoid flood limits
        if i < len(parts) - 1:
            await asyncio.sleep(settings.SEND_DELAY_SECONDS)

    return True, sent_count


async def send_summary(
    bot: Bot,
    chat_id: int,
    thread_id: int | None,
    text: str,
    destination_override: tuple[int, int | None] | None = None,
    dry_run: bool = False,
) -> bool:
    """Send summary text safely to Telegram chat/topic or an explicit destination override.

    - Splits into chunks <= 4096 rendered Telegram characters while preserving HTML balance.
    - Sends each part with HTML parse mode.
    - If Telegram returns parse entity error, automatically resends as plain text.
    - Handles flood limits (TelegramRetryAfter) with sleep.
    - If target thread does not exist, falls back to General topic.
    - Applies delay between parts to avoid Telegram flood limits.
    - Returns True only if all parts were successfully delivered.
    """
    _ = dry_run  # Accepted for caller symmetry; send_summary never writes to the database
    parts = smart_split(text, limit=TELEGRAM_MESSAGE_MAX_CHARS, count_rendered=True)
    if not parts:
        return False

    target_chat_id, target_thread_id = (
        destination_override if destination_override is not None else (chat_id, thread_id)
    )

    delivered, _ = await send_summary_parts(
        bot=bot,
        chat_id=target_chat_id,
        thread_id=target_thread_id,
        parts=parts,
        start_index=0,
    )
    return delivered

"""Safe Telegram message publisher handling splitting, rate limits, and HTML parsing fallbacks."""

import asyncio
import html
import logging
import re
from aiogram import Bot
from aiogram.exceptions import TelegramBadRequest, TelegramRetryAfter
from bot.config import get_settings
from bot.utils.general_topic import to_send_thread_id
from bot.utils.text_split import smart_split

logger = logging.getLogger(__name__)


def strip_html_tags(text: str) -> str:
    """Strip all HTML tags and unescape entities for fallback plain-text sending."""
    clean = re.sub(r"<[^>]+>", "", text)
    return html.unescape(clean)


async def send_summary(
    bot: Bot,
    chat_id: int,
    thread_id: int,
    text: str,
) -> bool:
    """Send summary text safely to Telegram supergroup topic.

    - Splits into chunks <= 4000 chars while preserving HTML balance.
    - Sends each part with HTML parse mode.
    - If Telegram returns parse entity error, automatically resends as plain text.
    - Handles flood limits (TelegramRetryAfter) with sleep.
    - If target thread does not exist, falls back to General topic.
    - Applies delay between parts to avoid Telegram flood limits.
    - Returns True only if all parts were successfully delivered.
    """
    settings = get_settings()
    parts = smart_split(text, limit=4000)
    if not parts:
        return False

    current_thread_id: int | None = to_send_thread_id(thread_id)

    for i, part in enumerate(parts):
        delivered = False
        retry_count = 0
        max_retries = 5

        while not delivered and retry_count < max_retries:
            try:
                await bot.send_message(
                    chat_id=chat_id,
                    message_thread_id=current_thread_id,
                    text=part,
                    parse_mode="HTML",
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

            except TelegramBadRequest as exc:
                error_msg = str(exc).lower()

                # Handle entity parsing errors -> resend as plain text
                if "can't parse entities" in error_msg or "unsupported start tag" in error_msg or "parsing" in error_msg:
                    logger.warning(
                        "HTML parse error from Telegram (%s). Falling back to plain text for chunk %s/%s.",
                        exc,
                        i + 1,
                        len(parts),
                    )
                    plain_text = strip_html_tags(part)
                    try:
                        await bot.send_message(
                            chat_id=chat_id,
                            message_thread_id=current_thread_id,
                            text=plain_text,
                            parse_mode=None,
                            disable_web_page_preview=True,
                        )
                        delivered = True
                    except Exception as fallback_exc:
                        logger.error("Plain text fallback failed: %s", fallback_exc)
                        return False

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
                        return False
                else:
                    logger.error("TelegramBadRequest while sending summary: %s", exc)
                    return False

            except Exception as exc:
                logger.error("Unexpected error delivering summary chunk %s/%s: %s", i + 1, len(parts), exc)
                return False

        if not delivered:
            logger.error("Failed to deliver chunk %s/%s after %s retries.", i + 1, len(parts), max_retries)
            return False

        # Sleep configured delay between parts to avoid flood limits
        if i < len(parts) - 1:
            await asyncio.sleep(settings.SEND_DELAY_SECONDS)

    return True

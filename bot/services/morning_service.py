"""Daily Hills/Prime morning greeting service: scheduling, catch-up, and admin test sending."""

import asyncio
from datetime import datetime
import logging
from zoneinfo import ZoneInfo
from aiogram import Bot
from aiogram.exceptions import (
    TelegramBadRequest,
    TelegramForbiddenError,
    TelegramNetworkError,
    TelegramRetryAfter,
    TelegramServerError,
)
from bot.config import get_settings
from bot.database import repository
from bot.texts.morning_texts import (
    MORNING_GREETINGS,
    compute_hills_day,
    format_morning_greeting,
)
from bot.utils.general_topic import to_send_thread_id
from bot.utils.timer_parser import to_persian_digits

logger = logging.getLogger(__name__)


def _resolve_local_now(now_tehran: datetime | None, tz_name: str) -> datetime:
    """Return timezone-aware datetime in tz_name."""
    tz = ZoneInfo(tz_name)
    if now_tehran is None:
        return datetime.now(tz)
    if now_tehran.tzinfo is None:
        return now_tehran.replace(tzinfo=tz)
    return now_tehran.astimezone(tz)


async def _send_greeting_message(
    bot: Bot,
    chat_id: int,
    thread_id: int,
    text: str,
) -> bool:
    """Send HTML morning greeting into target forum topic with flood/network retries and General fallback."""
    target_thread = to_send_thread_id(thread_id)
    max_attempts = 4
    attempt = 0

    while attempt < max_attempts:
        try:
            await bot.send_message(
                chat_id=chat_id,
                message_thread_id=target_thread,
                text=text,
                parse_mode="HTML",
                disable_web_page_preview=True,
            )
            return True
        except TelegramRetryAfter as exc:
            attempt += 1
            if attempt >= max_attempts:
                logger.error("Morning greeting flood limit persisted for chat %s: %s", chat_id, exc)
                return False
            wait_s = min(60.0, float(exc.retry_after) + 1.0)
            logger.warning("Morning greeting hit TelegramRetryAfter for chat %s; sleeping %.1fs...", chat_id, wait_s)
            await asyncio.sleep(wait_s)
        except (TelegramNetworkError, TelegramServerError, asyncio.TimeoutError) as exc:
            attempt += 1
            if attempt >= max_attempts:
                logger.error("Transient network/server error sending morning greeting to chat %s: %s", chat_id, exc)
                return False
            backoff = min(16.0, 2.0 * (2 ** (attempt - 1)))
            logger.warning(
                "Transient Telegram error sending morning greeting to chat %s (%s). Retrying in %.1fs...",
                chat_id,
                exc,
                backoff,
            )
            await asyncio.sleep(backoff)
        except TelegramBadRequest as exc:
            err_msg = str(exc).lower()
            if ("message thread not found" in err_msg or "thread_not_found" in err_msg) and target_thread is not None:
                logger.warning(
                    "Morning greeting target topic %s not found in chat %s. Falling back to General topic.",
                    target_thread,
                    chat_id,
                )
                target_thread = None
                attempt += 1
                continue
            logger.error("TelegramBadRequest sending morning greeting to chat %s: %s", chat_id, exc)
            return False
        except TelegramForbiddenError as exc:
            logger.warning("Bot lacks permission to send morning greeting to chat %s: %s", chat_id, exc)
            return False
        except Exception as exc:
            logger.error("Unexpected error sending morning greeting to chat %s: %s", chat_id, exc, exc_info=True)
            return False

    return False


async def send_morning_greeting_to_group(
    bot: Bot,
    chat_id: int,
    *,
    is_test: bool = False,
    now_tehran: datetime | None = None,
) -> tuple[bool, str]:
    """Send a unique daily Hills morning greeting to a group's configured topic.

    When is_test=True, sends the exact live greeting format into the configured topic without
    updating last_sent_day (so the scheduled 05:01 automatic send is unaffected).
    """
    settings = get_settings()
    cfg = await repository.get_morning_config(chat_id)
    if cfg is None or cfg.get("thread_id") is None:
        return False, "هنوز تاپیک مقصد برای پیام صبح‌بخیر انتخاب نشده است."

    if not is_test and int(cfg.get("enabled", 0)) == 0:
        return False, "سیستم صبح‌بخیر روزانه برای این گروه غیرفعال است."

    thread_id = int(cfg["thread_id"])
    local_now = _resolve_local_now(now_tehran, settings.TIMEZONE)
    today_day_str = local_now.strftime("%Y-%m-%d")
    day_number = compute_hills_day(
        now_tehran=local_now,
        start_date_iso=settings.HILLS_START_DATE,
        tz_name=settings.TIMEZONE,
    )

    draw = await repository.claim_morning_greeting_draw(
        chat_id,
        today_day_str=today_day_str,
        is_test=is_test,
    )
    if draw is None:
        return False, "پیام صبح‌بخیر امروز قبلاً به صورت خودکار ارسال شده است."

    chosen_idx, prev_last_day, _remaining = draw
    text = format_morning_greeting(
        chosen_idx,
        day_number=day_number,
        now_tehran=local_now,
    )

    delivered = await _send_greeting_message(bot, chat_id, thread_id, text)
    if not delivered:
        if not is_test:
            await repository.rollback_morning_last_sent_day(chat_id, prev_last_day)
        return False, "خطا در ارسال پیام صبح‌بخیر به تلگرام (دسترسی ربات یا وجود تاپیک را بررسی کنید)."

    topic_name = await repository.get_topic_name(chat_id, thread_id)
    if not topic_name:
        topic_name = "عمومی" if thread_id == 1 else f"تاپیک {thread_id}"

    total_pool = len(MORNING_GREETINGS)
    logger.info(
        "Sent %s morning greeting to chat %s topic '%s' (%s): Hills day %s (variant %s/%s).",
        "TEST" if is_test else "scheduled",
        chat_id,
        topic_name,
        thread_id,
        day_number,
        chosen_idx + 1,
        total_pool,
    )
    return (
        True,
        f"پیام صبح‌بخیر روز {to_persian_digits(day_number)} هیلز در تاپیک «{topic_name}» ارسال شد! "
        f"(متن {to_persian_digits(chosen_idx + 1)} از {to_persian_digits(total_pool)})",
    )


async def morning_greeting_job(
    bot: Bot,
    *,
    now_tehran: datetime | None = None,
) -> dict[str, int]:
    """Scheduled 05:01 daily job dispatching morning greetings to all enabled groups."""
    settings = get_settings()
    groups = await repository.list_enabled_morning_groups()
    if settings.ALLOWED_CHAT_IDS:
        allowed_set = set(settings.ALLOWED_CHAT_IDS)
        groups = [g for g in groups if int(g["chat_id"]) in allowed_set]

    report = {"sent": 0, "skipped": 0, "failed": 0}
    for idx, g in enumerate(groups):
        chat_id = int(g["chat_id"])
        ok, details = await send_morning_greeting_to_group(
            bot,
            chat_id,
            is_test=False,
            now_tehran=now_tehran,
        )
        if ok:
            report["sent"] += 1
        elif "قبلاً" in details:
            report["skipped"] += 1
        else:
            report["failed"] += 1

        if idx < len(groups) - 1:
            await asyncio.sleep(settings.SEND_DELAY_SECONDS)

    if groups:
        logger.info(
            "Daily morning greeting job finished: %s sent, %s skipped, %s failed.",
            report["sent"],
            report["skipped"],
            report["failed"],
        )
    return report


async def run_morning_catchup_if_needed(
    bot: Bot,
    *,
    now_tehran: datetime | None = None,
) -> int:
    """On startup, send missed today's morning greeting ONLY if still within the morning window."""
    settings = get_settings()
    local_now = _resolve_local_now(now_tehran, settings.TIMEZONE)

    scheduled_today = local_now.replace(
        hour=settings.MORNING_GREETING_HOUR,
        minute=settings.MORNING_GREETING_MINUTE,
        second=0,
        microsecond=0,
    )
    catchup_cutoff = local_now.replace(
        hour=settings.MORNING_CATCHUP_END_HOUR,
        minute=0,
        second=0,
        microsecond=0,
    )

    # Never send a 'Good Morning' greeting before 05:01 or after late morning (default 11:00)
    if local_now < scheduled_today or local_now >= catchup_cutoff:
        return 0

    today_day_str = local_now.strftime("%Y-%m-%d")
    groups = await repository.list_enabled_morning_groups()
    if settings.ALLOWED_CHAT_IDS:
        allowed_set = set(settings.ALLOWED_CHAT_IDS)
        groups = [g for g in groups if int(g["chat_id"]) in allowed_set]

    pending = [g for g in groups if g.get("last_sent_day") != today_day_str]
    if not pending:
        return 0

    logger.info(
        "Morning greeting catch-up triggered within morning window (%s) for %s group(s).",
        local_now.strftime("%H:%M"),
        len(pending),
    )
    sent_count = 0
    for idx, g in enumerate(pending):
        ok, _ = await send_morning_greeting_to_group(
            bot,
            int(g["chat_id"]),
            is_test=False,
            now_tehran=local_now,
        )
        if ok:
            sent_count += 1
        if idx < len(pending) - 1:
            await asyncio.sleep(settings.SEND_DELAY_SECONDS)

    return sent_count

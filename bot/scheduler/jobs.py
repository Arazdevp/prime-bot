"""Scheduler jobs for nightly summarization, database maintenance, and cleanup."""

import asyncio
from datetime import datetime, timezone
import logging
from pathlib import Path
from zoneinfo import ZoneInfo
from aiogram import Bot
from aiogram.exceptions import TelegramBadRequest, TelegramForbiddenError
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from bot.config import get_settings
from bot.database import repository
from bot.database.db import get_db, get_write_lock
from bot.services.llm_client import BaseLLMClient
from bot.services.publisher import send_summary
from bot.services.summarizer import summarize_topic_messages
from bot.utils.html_utils import escape_html

logger = logging.getLogger(__name__)

# Concurrency lock ensuring nightly and manual runs do not execute concurrently
_summary_execution_lock = asyncio.Lock()


def get_summary_lock() -> asyncio.Lock:
    """Return shared summarization lock."""
    return _summary_execution_lock


async def notify_admins_failure(
    bot: Bot,
    chat_title: str,
    topic_name: str,
    error_type: str,
    error_message: str,
) -> None:
    """Send Persian failure alert to configured bot admins via private message."""
    settings = get_settings()
    if not settings.ADMIN_IDS:
        return

    text = (
        "⚠️ <b>گزارش خطای خلاصه‌سازی تاپیک:</b>\n\n"
        f"🏢 <b>گروه:</b> {escape_html(chat_title)}\n"
        f"📌 <b>تاپیک:</b> {escape_html(topic_name)}\n"
        f"❌ <b>نوع خطا:</b> {escape_html(error_type)}\n"
        f"📝 <b>پیام خطا:</b> <code>{escape_html(error_message[:400])}</code>"
    )

    for admin_id in settings.ADMIN_IDS:
        try:
            await bot.send_message(
                chat_id=admin_id,
                text=text,
                parse_mode="HTML",
                disable_web_page_preview=True,
            )
        except (TelegramForbiddenError, TelegramBadRequest):
            # Admin hasn't started the bot or blocked it
            continue
        except Exception as exc:
            logger.warning("Failed sending failure alert to admin %s: %s", admin_id, exc)


async def notify_admins_nightly_completion(
    bot: Bot,
    total_groups: int,
    total_success: int,
    total_skipped: int,
    total_failed: int,
) -> None:
    """Send Persian nightly summary completion report to admins."""
    settings = get_settings()
    if not settings.ADMIN_IDS:
        return

    text = (
        "🌙 <b>گزارش اجرای خودکار شبانه خلاصه‌ساز:</b>\n\n"
        f"🏢 گروه‌های بررسی‌شده: <b>{total_groups}</b>\n"
        f"✅ تاپیک‌های با موفقیت تلخیص‌شده: <b>{total_success}</b>\n"
        f"⏭ تاپیک‌های کمتر از حد نصاب (ردشده): <b>{total_skipped}</b>\n"
        f"❌ تاپیک‌های با خطا: <b>{total_failed}</b>"
    )

    for admin_id in settings.ADMIN_IDS:
        try:
            await bot.send_message(
                chat_id=admin_id,
                text=text,
                parse_mode="HTML",
                disable_web_page_preview=True,
            )
        except Exception:
            continue


async def run_summary_for_topic(
    bot: Bot,
    chat_id: int,
    thread_id: int,
    mark_summarized: bool = True,
    is_nightly: bool = True,
    llm_client: BaseLLMClient | None = None,
) -> tuple[bool, str]:
    """Execute end-to-end summarization pipeline for a single topic.

    Returns (success_bool, details_str).
    """
    settings = get_settings()
    group_cfg = await repository.ensure_group_settings(chat_id, settings.MIN_MESSAGES_PER_TOPIC)

    if group_cfg.get("enabled", 1) == 0:
        return False, "سیستم خلاصه‌سازی این گروه غیرفعال است."

    if not await repository.is_topic_monitored(chat_id, thread_id):
        return False, "این تاپیک در تنظیمات فعلی گروه تحت نظارت نیست."

    topic_name = await repository.get_topic_name(chat_id, thread_id)
    if not topic_name:
        topic_name = "عمومی" if thread_id == 1 else f"تاپیک {thread_id}"

    messages = await repository.fetch_unsummarized_messages(chat_id, thread_id)
    min_msgs = group_cfg.get("min_messages", settings.MIN_MESSAGES_PER_TOPIC)

    # Threshold check
    if len(messages) < min_msgs:
        logger.info(
            "Topic '%s' in chat %s has %s messages (< min %s).",
            topic_name,
            chat_id,
            len(messages),
            min_msgs,
        )
        if is_nightly and mark_summarized and messages:
            msg_ids = [m["id"] for m in messages]
            deleted_cnt = await repository.delete_messages_by_ids(msg_ids)
            logger.info("Deleted %s below-threshold messages for topic '%s' to free storage.", deleted_cnt, topic_name)
        return False, f"تعداد پیام‌ها ({len(messages)}) کمتر از حداقل حد نصاب ({min_msgs}) بود."

    # Generate summary with AI
    try:
        summary_text = await summarize_topic_messages(
            topic_name=topic_name,
            messages=messages,
            chat_id=chat_id,
            min_messages=min_msgs,
            is_nightly=is_nightly,
            llm_client=llm_client,
        )
    except Exception as exc:
        logger.error("Summarization AI failed for topic '%s': %s", topic_name, exc, exc_info=True)
        try:
            chat = await bot.get_chat(chat_id)
            chat_title = chat.title or str(chat_id)
        except Exception:
            chat_title = str(chat_id)

        await notify_admins_failure(
            bot=bot,
            chat_title=chat_title,
            topic_name=topic_name,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
        return False, f"خطای هوش مصنوعی در تولید خلاصه: {exc}"

    if not summary_text:
        return False, "خلاصه معتبری توسط مدل تولید نشد."

    # Destination handling
    dest_mode = group_cfg.get("destination_mode", "per_topic")
    dest_tid = group_cfg.get("destination_thread_id")

    if dest_mode == "aggregate" and dest_tid is not None:
        target_thread = dest_tid
        content_to_send = f"📂 <b>تاپیک مبدا: {escape_html(topic_name)}</b>\n\n\n{summary_text}"
    else:
        target_thread = thread_id
        content_to_send = summary_text

    # Publish to Telegram
    try:
        delivered = await send_summary(
            bot=bot,
            chat_id=chat_id,
            thread_id=target_thread,
            text=content_to_send,
        )
    except Exception as exc:
        logger.error("Publishing summary failed: %s", exc, exc_info=True)
        delivered = False

    if delivered:
        if mark_summarized:
            msg_ids = [m["id"] for m in messages]
            deleted_cnt = await repository.delete_messages_by_ids(msg_ids)
            logger.info("Deleted %s summarized messages in topic '%s' to free database space.", deleted_cnt, topic_name)
        return True, "خلاصه با موفقیت ارسال شد."
    else:
        logger.error("Failed delivering summary for topic '%s' in chat %s.", topic_name, chat_id)
        try:
            chat = await bot.get_chat(chat_id)
            chat_title = chat.title or str(chat_id)
        except Exception:
            chat_title = str(chat_id)

        await notify_admins_failure(
            bot=bot,
            chat_title=chat_title,
            topic_name=topic_name,
            error_type="PublishDeliveryError",
            error_message="عدم موفقیت در ارسال بخش‌های پیام به تلگرام.",
        )
        return False, "خطا در ارسال پیام به تلگرام."


async def run_summary_for_group(
    bot: Bot,
    chat_id: int,
    mark_summarized: bool = True,
    is_nightly: bool = False,
    llm_client: BaseLLMClient | None = None,
) -> dict[str, int]:
    """Run summarization sequentially for all monitored topics in a group."""
    settings = get_settings()
    distinct_threads = await repository.list_distinct_unsummarized_topics(chat_id)

    report = {"success": 0, "skipped": 0, "failed": 0}

    for tid in distinct_threads:
        if not await repository.is_topic_monitored(chat_id, tid):
            continue

        try:
            success, details = await run_summary_for_topic(
                bot=bot,
                chat_id=chat_id,
                thread_id=tid,
                mark_summarized=mark_summarized,
                is_nightly=is_nightly,
                llm_client=llm_client,
            )
            if success:
                report["success"] += 1
            else:
                if "کمتر از حداقل حد نصاب" in details:
                    report["skipped"] += 1
                else:
                    report["failed"] += 1
        except Exception as exc:
            logger.error("Unexpected error in group topic summary: %s", exc)
            report["failed"] += 1

        await asyncio.sleep(settings.SEND_DELAY_SECONDS)

    return report


async def nightly_summary_job(bot: Bot, llm_client: BaseLLMClient | None = None) -> None:
    """Nightly cron job executing at 00:01 Asia/Tehran."""
    logger.info("Starting nightly summarization job...")
    lock = get_summary_lock()

    if lock.locked():
        logger.warning("Another summarization job is already active. Skipping duplicate run.")
        return

    async with lock:
        groups = await repository.list_enabled_groups()
        total_groups = len(groups)
        total_success = 0
        total_skipped = 0
        total_failed = 0

        for g in groups:
            chat_id = g["chat_id"]
            logger.info("Processing nightly summary for group %s...", chat_id)
            group_report = await run_summary_for_group(
                bot=bot,
                chat_id=chat_id,
                mark_summarized=True,
                is_nightly=True,
                llm_client=llm_client,
            )
            total_success += group_report["success"]
            total_skipped += group_report["skipped"]
            total_failed += group_report["failed"]

        logger.info(
            "Nightly summary completed: %s groups, %s success, %s skipped, %s failed.",
            total_groups,
            total_success,
            total_skipped,
            total_failed,
        )
        await notify_admins_nightly_completion(
            bot=bot,
            total_groups=total_groups,
            total_success=total_success,
            total_skipped=total_skipped,
            total_failed=total_failed,
        )


async def cleanup_job() -> None:
    """Daily database maintenance and retention cleanup job at 03:30 Asia/Tehran."""
    logger.info("Starting scheduled cleanup job...")
    settings = get_settings()

    # 1. Purge messages older than retention days
    deleted_count = await repository.delete_old_messages(settings.RETENTION_DAYS)
    logger.info("Deleted %s messages older than %s days.", deleted_count, settings.RETENTION_DAYS)

    # 2. SQLite PRAGMA optimize
    conn = get_db()
    lock = get_write_lock()
    async with lock:
        await conn.execute("PRAGMA optimize;")
        await conn.commit()

    # 3. Weekly VACUUM on Saturdays if database exceeds 50 MB
    tehran_now = datetime.now(ZoneInfo(settings.TIMEZONE))
    # Python weekday(): Monday is 0, Saturday is 5
    if tehran_now.weekday() == 5:
        db_path = Path(settings.DB_PATH)
        if db_path.exists():
            size_mb = db_path.stat().st_size / (1024 * 1024)
            if size_mb > 50.0:
                logger.info("Running weekly VACUUM on DB (current size: %.2f MB)...", size_mb)
                async with lock:
                    await conn.execute("VACUUM;")
                    await conn.commit()


def build_scheduler(bot: Bot) -> AsyncIOScheduler:
    """Build and configure the APScheduler AsyncIOScheduler instance."""
    settings = get_settings()
    tz = ZoneInfo(settings.TIMEZONE)

    scheduler = AsyncIOScheduler(timezone=tz)

    # 1. Nightly summary job at 00:01
    scheduler.add_job(
        nightly_summary_job,
        trigger=CronTrigger(
            hour=settings.SUMMARY_TIME_HOUR,
            minute=settings.SUMMARY_TIME_MINUTE,
            timezone=tz,
        ),
        args=[bot],
        id="nightly_summary",
        name="Nightly Topic Summarization",
        misfire_grace_time=3600,
        coalesce=True,
        max_instances=1,
    )

    # 2. Daily cleanup job at 03:30
    scheduler.add_job(
        cleanup_job,
        trigger=CronTrigger(
            hour=3,
            minute=30,
            timezone=tz,
        ),
        id="daily_cleanup",
        name="Daily Database Retention Cleanup",
        misfire_grace_time=3600,
        coalesce=True,
        max_instances=1,
    )

    return scheduler

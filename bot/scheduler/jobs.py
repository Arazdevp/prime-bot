"""Scheduler jobs for nightly summarization, database maintenance, and cleanup."""

import asyncio
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Awaitable, Callable
from zoneinfo import ZoneInfo
from aiogram import Bot
from aiogram.exceptions import TelegramBadRequest, TelegramForbiddenError
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from bot.config import get_settings
from bot.database import repository
from bot.database.db import get_db, get_write_lock
from bot.services.llm_client import BaseLLMClient
from bot.services.publisher import send_summary, send_summary_parts
from bot.services.summarizer import summarize_topic_messages
from bot.utils.dates import compute_nightly_cutoff
from bot.utils.html_utils import escape_html
from bot.utils.text_split import smart_split

logger = logging.getLogger(__name__)

# Concurrency lock ensuring nightly and manual runs do not execute concurrently
_summary_execution_lock = asyncio.Lock()
_scheduler_ref: AsyncIOScheduler | None = None


def get_summary_lock() -> asyncio.Lock:
    """Return shared summarization lock."""
    return _summary_execution_lock


def set_scheduler(scheduler: AsyncIOScheduler | None) -> None:
    """Store active scheduler instance for runtime inspection (e.g. Global Status)."""
    global _scheduler_ref
    _scheduler_ref = scheduler


def get_next_nightly_run_time() -> datetime | None:
    """Return the scheduler's next nightly_summary fire time, if active."""
    if _scheduler_ref is None:
        return None
    try:
        job = _scheduler_ref.get_job("nightly_summary")
        if job and job.next_run_time:
            return job.next_run_time
    except Exception:
        return None
    return None


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
    cutoff: datetime | str | None = None,
    llm_client: BaseLLMClient | None = None,
    destination_override: tuple[int, int | None] | None = None,
    dry_run: bool = False,
) -> tuple[bool, str]:
    """Execute end-to-end summarization pipeline for a single topic.

    When dry_run=True, performs read-only DB queries, generates a fresh summary,
    sends it (using destination_override if given), and NEVER writes to or deletes from the DB.

    Returns (success_bool, details_str).
    """
    settings = get_settings()

    if dry_run:
        group_cfg = await repository.get_group_settings(chat_id) or {
            "chat_id": chat_id,
            "mode": "blacklist",
            "destination_mode": "per_topic",
            "destination_thread_id": None,
            "enabled": 1,
            "min_messages": settings.MIN_MESSAGES_PER_TOPIC,
        }
    else:
        group_cfg = await repository.ensure_group_settings(chat_id, settings.MIN_MESSAGES_PER_TOPIC)

        if group_cfg.get("enabled", 1) == 0:
            return False, "سیستم خلاصه‌سازی این گروه غیرفعال است."

        if not await repository.is_topic_monitored(chat_id, thread_id):
            return False, "این تاپیک در تنظیمات فعلی گروه تحت نظارت نیست."

    topic_name = await repository.get_topic_name(chat_id, thread_id)
    if not topic_name:
        topic_name = "عمومی" if thread_id == 1 else f"تاپیک {thread_id}"

    # Determine cutoff ISO string
    cutoff_iso: str | None = None
    if cutoff is not None:
        if isinstance(cutoff, datetime):
            cutoff_iso = cutoff.astimezone(timezone.utc).isoformat()
        else:
            cutoff_iso = str(cutoff)
    elif is_nightly and not dry_run:
        cutoff_iso = compute_nightly_cutoff().isoformat()

    messages = await repository.fetch_unsummarized_messages(chat_id, thread_id, before=cutoff_iso)
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
        # Delete below-threshold messages only on nightly/cutoff runs, never on manual or dry_run
        if not dry_run and cutoff_iso is not None and mark_summarized and messages:
            msg_ids = [m["id"] for m in messages]
            deleted_cnt = await repository.delete_messages_by_ids(msg_ids)
            logger.info("Deleted %s below-threshold messages for topic '%s' to free storage.", deleted_cnt, topic_name)
        return False, f"تعداد پیام‌ها ({len(messages)}) کمتر از حداقل حد نصاب ({min_msgs}) بود."

    # Destination handling
    dest_mode = group_cfg.get("destination_mode", "per_topic")
    dest_tid = group_cfg.get("destination_thread_id")
    target_thread = dest_tid if dest_mode == "aggregate" and dest_tid is not None else thread_id

    if dry_run:
        try:
            summary_text = await summarize_topic_messages(
                topic_name=topic_name,
                messages=messages,
                chat_id=chat_id,
                thread_id=thread_id,
                min_messages=min_msgs,
                is_nightly=False,
                llm_client=llm_client,
            )
        except Exception as exc:
            logger.error("Dry-run summarization AI failed for topic '%s': %s", topic_name, exc, exc_info=True)
            return False, f"خطای هوش مصنوعی در تولید پیش‌نمایش: {exc}"

        if not summary_text:
            return False, "خلاصه معتبری توسط مدل تولید نشد."

        preview_text = (
            f"👁 <b>پیش‌نمایش خلاصه تاپیک «{escape_html(topic_name)}»</b>\n"
            "<i>(فقط برای شما ارسال شده و هیچ پیامی تغییر یا حذف نشد)</i>\n\n"
            f"{summary_text}"
        )
        try:
            delivered = await send_summary(
                bot=bot,
                chat_id=chat_id,
                thread_id=target_thread,
                text=preview_text,
                destination_override=destination_override,
                dry_run=True,
            )
        except Exception as exc:
            logger.error("Dry-run publishing failed for topic '%s': %s", topic_name, exc, exc_info=True)
            delivered = False

        if delivered:
            return True, "پیش‌نمایش خلاصه با موفقیت برای شما ارسال شد."
        return False, "خطا در ارسال پیش‌نمایش به تلگرام."

    # Derive day from earliest message's Tehran timestamp
    first_created = messages[0].get("created_at") or ""
    try:
        first_dt = datetime.fromisoformat(first_created)
        tehran_tz = ZoneInfo(settings.TIMEZONE)
        tehran_dt = (
            first_dt.astimezone(tehran_tz)
            if first_dt.tzinfo
            else first_dt.replace(tzinfo=ZoneInfo("UTC")).astimezone(tehran_tz)
        )
    except Exception:
        tehran_dt = datetime.now(ZoneInfo(settings.TIMEZONE))
    day = tehran_dt.strftime("%Y-%m-%d")

    # Check for existing summary in database to handle resumption and prevent duplicate posts
    existing_summary = await repository.get_summary_record(chat_id, thread_id, day)
    parts: list[str] = []
    sent_count = 0

    if existing_summary:
        try:
            parts = json.loads(existing_summary["parts_json"])
            sent_count = int(existing_summary["sent_count"])
        except Exception as parse_err:
            logger.warning("Failed parsing existing summary record for topic %s: %s", topic_name, parse_err)
            parts = []
            sent_count = 0

    if parts and sent_count >= len(parts):
        # Already completely delivered! Prevent duplicate posts
        logger.info("Summary for topic '%s' on %s was already fully delivered.", topic_name, day)
        if mark_summarized:
            msg_ids = [m["id"] for m in messages]
            await repository.delete_messages_by_ids(msg_ids)
        return True, "خلاصه با موفقیت ارسال شد."

    if not parts:
        # Generate summary with AI
        try:
            summary_text = await summarize_topic_messages(
                topic_name=topic_name,
                messages=messages,
                chat_id=chat_id,
                thread_id=thread_id,
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

        if dest_mode == "aggregate" and dest_tid is not None:
            content_to_send = f"📂 <b>تاپیک مبدا: {escape_html(topic_name)}</b>\n\n\n{summary_text}"
        else:
            content_to_send = summary_text

        parts = smart_split(content_to_send, limit=4000)
        if not parts:
            return False, "خلاصه پس از تقسیم خالی بود."

        sent_count = 0
        await repository.save_summary_record(chat_id, thread_id, day, parts, sent_count=0)

    # Publish remaining parts to Telegram
    async def _on_part_sent(new_sent: int) -> None:
        await repository.update_summary_sent_count(chat_id, thread_id, day, new_sent)

    pub_chat_id, pub_thread_id = (
        destination_override if destination_override is not None else (chat_id, target_thread)
    )

    try:
        delivered, sent_count = await send_summary_parts(
            bot=bot,
            chat_id=pub_chat_id,
            thread_id=pub_thread_id,
            parts=parts,
            start_index=sent_count,
            on_part_sent=_on_part_sent,
        )
    except Exception as exc:
        logger.error("Publishing summary failed: %s", exc, exc_info=True)
        delivered = False

    if delivered and sent_count >= len(parts):
        # Delete source messages only after ALL parts are delivered
        if mark_summarized:
            msg_ids = [m["id"] for m in messages]
            deleted_cnt = await repository.delete_messages_by_ids(msg_ids)
            logger.info("Deleted %s summarized messages in topic '%s' to free database space.", deleted_cnt, topic_name)
        return True, "خلاصه با موفقیت ارسال شد."
    else:
        logger.error(
            "Failed delivering summary for topic '%s' in chat %s (sent %s/%s parts).",
            topic_name,
            chat_id,
            sent_count,
            len(parts),
        )
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
            error_message=f"عدم موفقیت در ارسال بخش‌های پیام به تلگرام ({sent_count}/{len(parts)} بخش ارسال شد).",
        )
        return False, "خطا در ارسال پیام به تلگرام."


async def run_summary_for_group(
    bot: Bot,
    chat_id: int,
    mark_summarized: bool = True,
    is_nightly: bool = False,
    cutoff: datetime | str | None = None,
    llm_client: BaseLLMClient | None = None,
    progress_callback: Callable[[int, int, str], Awaitable[None]] | None = None,
) -> dict[str, int]:
    """Run summarization sequentially for all monitored topics in a group."""
    settings = get_settings()

    cutoff_iso: str | None = None
    if cutoff is not None:
        if isinstance(cutoff, datetime):
            cutoff_iso = cutoff.astimezone(timezone.utc).isoformat()
        else:
            cutoff_iso = str(cutoff)
    elif is_nightly:
        cutoff_iso = compute_nightly_cutoff().isoformat()

    distinct_threads = await repository.list_distinct_unsummarized_topics(chat_id, before=cutoff_iso)
    monitored_threads: list[int] = []
    for tid in distinct_threads:
        if await repository.is_topic_monitored(chat_id, tid):
            monitored_threads.append(tid)

    report = {"success": 0, "skipped": 0, "failed": 0}
    total = len(monitored_threads)

    for idx, tid in enumerate(monitored_threads, start=1):
        if progress_callback is not None:
            try:
                t_name = await repository.get_topic_name(chat_id, tid)
                if not t_name:
                    t_name = "عمومی" if tid == 1 else f"تاپیک {tid}"
                await progress_callback(idx, total, t_name)
            except Exception as exc:
                logger.debug("Progress callback failed for topic %s: %s", tid, exc)

        try:
            success, details = await run_summary_for_topic(
                bot=bot,
                chat_id=chat_id,
                thread_id=tid,
                mark_summarized=mark_summarized,
                is_nightly=is_nightly,
                cutoff=cutoff_iso,
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


async def nightly_summary_job(
    bot: Bot,
    llm_client: BaseLLMClient | None = None,
    run_time: datetime | None = None,
) -> None:
    """Nightly cron job executing at 00:01 Asia/Tehran or catch-up."""
    logger.info("Starting nightly summarization job...")
    lock = get_summary_lock()

    if lock.locked():
        logger.warning("Another summarization job is already active. Skipping duplicate run.")
        return

    async with lock:
        settings = get_settings()
        cutoff = compute_nightly_cutoff(run_time=run_time, tz_name=settings.TIMEZONE).isoformat()
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
                cutoff=cutoff,
                llm_client=llm_client,
            )
            total_success += group_report["success"]
            total_skipped += group_report["skipped"]
            total_failed += group_report["failed"]

        await repository.set_last_nightly_run(datetime.now(timezone.utc).isoformat())

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


def is_nightly_run_missed(now: datetime | None = None, last_run_iso: str | None = None) -> bool:
    """Determine if today's scheduled run time has passed without an execution."""
    settings = get_settings()
    tz = ZoneInfo(settings.TIMEZONE)
    if now is None:
        now = datetime.now(tz)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=tz)
    else:
        now = now.astimezone(tz)

    today_scheduled = now.replace(
        hour=settings.SUMMARY_TIME_HOUR,
        minute=settings.SUMMARY_TIME_MINUTE,
        second=0,
        microsecond=0,
    )
    if now < today_scheduled:
        return False

    if not last_run_iso:
        return True

    try:
        last_run_dt = datetime.fromisoformat(last_run_iso)
        if last_run_dt.tzinfo is None:
            last_run_dt = last_run_dt.replace(tzinfo=timezone.utc)
        last_run_tz = last_run_dt.astimezone(tz)
        return last_run_tz < today_scheduled
    except Exception:
        return True


async def run_catchup_if_needed(bot: Bot, llm_client: BaseLLMClient | None = None) -> bool:
    """Check if today's scheduled nightly run was missed; if so, execute it once."""
    last_run_iso = await repository.get_last_nightly_run()
    if is_nightly_run_missed(last_run_iso=last_run_iso):
        logger.info("Missed today's scheduled nightly summary run. Executing catch-up run now...")
        await nightly_summary_job(bot, llm_client=llm_client)
        return True
    return False


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

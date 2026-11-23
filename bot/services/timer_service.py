"""Timer scheduling, startup rehydration, safety sweeps, and simultaneous group+DM firing."""

import asyncio
from datetime import datetime, timedelta, timezone
import logging
from typing import Any
from zoneinfo import ZoneInfo
from aiogram import Bot
from aiogram.exceptions import TelegramBadRequest, TelegramForbiddenError, TelegramRetryAfter
from apscheduler.jobstores.base import JobLookupError
from apscheduler.triggers.date import DateTrigger
from bot.config import get_settings
from bot.database import repository
from bot.keyboards.timer import build_fired_alarm_kb
from bot.utils.dates import format_persian_date
from bot.utils.general_topic import to_send_thread_id
from bot.utils.html_utils import escape_html
from bot.utils.timer_parser import (
    MAX_TIMER_DAYS,
    format_duration_persian,
    parse_utc_iso,
    sanitize_note,
    to_persian_digits,
    to_utc_iso,
)

logger = logging.getLogger(__name__)

_bot_username_cache: str | None = None


def set_bot_username(username: str | None) -> None:
    """Cache the bot's Telegram username for building DM activation deep links."""
    global _bot_username_cache
    _bot_username_cache = username.strip().lstrip("@") if username and username.strip() else None


def get_bot_username() -> str | None:
    """Return cached bot username, if known."""
    return _bot_username_cache


async def ensure_bot_username(bot: Bot) -> str | None:
    """Return cached bot username, lazily querying bot.get_me() once if not yet cached."""
    global _bot_username_cache
    if _bot_username_cache:
        return _bot_username_cache
    try:
        me = await bot.get_me()
        if me and getattr(me, "username", None):
            set_bot_username(me.username)
    except Exception as exc:
        logger.debug("Could not fetch bot username via get_me(): %s", exc)
    return _bot_username_cache


def timer_job_id(timer_id: int) -> str:
    """Return canonical APScheduler job ID for a timer."""
    return f"timer_{timer_id}"


def build_topic_deep_link(chat_id: int, thread_id: int) -> str | None:
    """Build a direct Telegram topic link (https://t.me/c/<internal>/<thread>) for supergroups."""
    raw = str(chat_id)
    if not raw.startswith("-100") or len(raw) <= 4:
        return None
    internal_id = raw[4:]
    safe_tid = thread_id if thread_id and thread_id > 0 else 1
    return f"https://t.me/c/{internal_id}/{safe_tid}"


def format_user_mention(user_id: int, username: str | None, full_name: str) -> str:
    """Build safe HTML mention showing @username when present plus guaranteed tg://user link."""
    safe_name = escape_html((full_name or "").strip() or "کاربر")
    link = f'<a href="tg://user?id={user_id}">{safe_name}</a>'
    if username and username.strip():
        clean_u = escape_html(username.strip().lstrip("@"))
        return f"@{clean_u} ({link})"
    return link


def schedule_timer_job(bot: Bot, timer_row: dict[str, Any]) -> bool:
    """Register an exact one-shot DateTrigger job on the active AsyncIOScheduler."""
    from bot.scheduler.jobs import get_scheduler

    scheduler = get_scheduler()
    if scheduler is None:
        return False

    timer_id = int(timer_row["id"])
    fire_at_utc = parse_utc_iso(timer_row["fire_at"])
    try:
        scheduler.add_job(
            fire_timer,
            trigger=DateTrigger(run_date=fire_at_utc, timezone=timezone.utc),
            args=[bot, timer_id],
            id=timer_job_id(timer_id),
            name=f"User Timer #{timer_id}",
            replace_existing=True,
            misfire_grace_time=None,
            max_instances=1,
        )
        return True
    except Exception as exc:
        logger.error("Failed scheduling timer %s: %s", timer_id, exc, exc_info=True)
        return False


def unschedule_timer_job(timer_id: int) -> None:
    """Remove a pending timer job from APScheduler if present."""
    from bot.scheduler.jobs import get_scheduler

    scheduler = get_scheduler()
    if scheduler is None:
        return
    try:
        scheduler.remove_job(timer_job_id(timer_id))
    except JobLookupError:
        pass
    except Exception as exc:
        logger.debug("Ignoring error unscheduling timer %s: %s", timer_id, exc)


async def create_and_schedule_timer(
    bot: Bot,
    *,
    chat_id: int,
    thread_id: int,
    chat_type: str,
    user_id: int,
    username: str | None,
    full_name: str,
    note: str | None,
    timer_type: str,
    fire_at: datetime,
    now_utc: datetime | None = None,
    source_message_id: int | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Validate limits/bounds, persist a timer to SQLite, and schedule its APScheduler job.

    Returns (timer_row, None) on success, or (None, persian_error_message) on failure.
    """
    settings = get_settings()
    current_utc = now_utc.astimezone(timezone.utc) if now_utc else datetime.now(timezone.utc)
    fire_utc = (
        fire_at.astimezone(timezone.utc)
        if fire_at.tzinfo
        else fire_at.replace(tzinfo=timezone.utc)
    )

    delta = fire_utc - current_utc
    if delta.total_seconds() < 15:
        return None, "⚠️ زمان انتخابی باید در آینده باشه! یه زمان دیگه انتخاب کن."
    if delta > timedelta(days=MAX_TIMER_DAYS):
        return (
            None,
            f"⚠️ حداکثر بازه مجاز تایمر {to_persian_digits(MAX_TIMER_DAYS)} روزه (۱ سال) هست.",
        )

    active_count = await repository.count_active_timers_for_user(user_id)
    if active_count >= settings.TIMER_MAX_PER_USER:
        return (
            None,
            f"⚠️ به سقف {to_persian_digits(settings.TIMER_MAX_PER_USER)} تایمر فعال رسیدی! "
            "اول از بخش «📋 تایمرهای فعال من» یکی رو لغو کن.",
        )

    timer_row = await repository.create_timer(
        chat_id=chat_id,
        thread_id=thread_id,
        chat_type=chat_type,
        user_id=user_id,
        username=username,
        full_name=full_name,
        note=sanitize_note(note),
        timer_type=timer_type,
        fire_at_iso=to_utc_iso(fire_utc),
        created_at_iso=to_utc_iso(current_utc),
        source_message_id=source_message_id,
    )
    schedule_timer_job(bot, timer_row)
    return timer_row, None


def _build_delay_notice(fire_utc: datetime, now_utc: datetime) -> str:
    delay_seconds = (now_utc - fire_utc).total_seconds()
    if delay_seconds < 120:
        return ""
    delay_mins = max(1, int(delay_seconds // 60))
    return (
        f"\n⏱ <i>(ارسال با {format_duration_persian(delay_mins)} تأخیر "
        "به دلیل قطعی موقت ربات)</i>"
    )


def _build_group_alarm_text(
    timer: dict[str, Any],
    *,
    now_utc: datetime,
    tz_name: str,
) -> str:
    """Format the Persian HTML notification sent inside the original forum topic."""
    mention = format_user_mention(
        int(timer["user_id"]),
        timer.get("username"),
        str(timer.get("full_name") or "کاربر"),
    )
    fire_utc = parse_utc_iso(timer["fire_at"])
    delay_notice = _build_delay_notice(fire_utc, now_utc)
    note = timer.get("note")

    if note:
        return (
            "⏰ <b>دینگ‌دینگ! تایمرت به صدا در اومد!</b>\n\n"
            f"👤 {mention}\n"
            f"📝 <b>یادآور:</b>\n<blockquote>{escape_html(note)}</blockquote>\n"
            f"می‌تونی بری سراغ کارت! 🚀{delay_notice}"
        )

    return (
        "⏰ <b>دینگ‌دینگ! تایمرت به صدا در اومد!</b>\n\n"
        f"👤 {mention}\n"
        f"وقتت تموم شد، می‌تونی بری سراغ کارت! 🚀{delay_notice}"
    )


def _build_dm_alarm_text(
    timer: dict[str, Any],
    *,
    group_title: str | None,
    topic_name: str | None,
    now_utc: datetime,
    tz_name: str,
) -> str:
    """Format the Persian HTML notification sent simultaneously to the user's private DM."""
    tz = ZoneInfo(tz_name)
    fire_utc = parse_utc_iso(timer["fire_at"])
    fire_local = fire_utc.astimezone(tz)
    time_str = fire_local.strftime("%H:%M")
    date_str = format_persian_date(fire_local)
    delay_notice = _build_delay_notice(fire_utc, now_utc)
    note = timer.get("note")

    lines = ["🔔 <b>یادآور تایمرت به صدا در اومد!</b>", ""]
    if timer.get("chat_type") != "private":
        if group_title:
            lines.append(f"🏢 <b>گروه:</b> {escape_html(group_title)}")
        if topic_name:
            lines.append(f"📂 <b>تاپیک:</b> {escape_html(topic_name)}")
    if note:
        lines.append(f"📝 <b>یادآور:</b>\n<blockquote>{escape_html(note)}</blockquote>")
    lines.append(f"🕒 <b>زمان:</b> {time_str} — {date_str}")
    lines.append("")
    lines.append(f"می‌تونی بری سراغ کارت! 🚀{delay_notice}")
    return "\n".join(lines)


async def _send_group_alarm(
    bot: Bot,
    *,
    chat_id: int,
    thread_id: int,
    text: str,
    reply_markup: Any,
) -> bool:
    """Send alarm message into the target forum topic with General-topic fallback and retry."""
    target_thread = to_send_thread_id(thread_id)
    for attempt in range(2):
        try:
            await bot.send_message(
                chat_id=chat_id,
                message_thread_id=target_thread,
                text=text,
                reply_markup=reply_markup,
                parse_mode="HTML",
                disable_web_page_preview=True,
            )
            return True
        except TelegramRetryAfter as exc:
            if attempt == 0:
                await asyncio.sleep(min(exc.retry_after + 1, 15))
                continue
            logger.warning("Group alarm hit TelegramRetryAfter twice for chat %s", chat_id)
            return False
        except TelegramBadRequest as exc:
            err = str(exc).lower()
            if ("message thread not found" in err or "thread_not_found" in err) and target_thread is not None:
                logger.warning(
                    "Topic thread %s not found in chat %s for timer alarm. Falling back to General topic.",
                    target_thread,
                    chat_id,
                )
                target_thread = None
                continue
            logger.error("TelegramBadRequest sending group timer alarm to chat %s: %s", chat_id, exc)
            return False
        except Exception as exc:
            logger.error("Failed sending group timer alarm to chat %s: %s", chat_id, exc)
            return False
    return False


async def _send_dm_alarm(
    bot: Bot,
    *,
    user_id: int,
    username: str | None,
    full_name: str,
    text: str,
    reply_markup: Any,
) -> bool:
    """Best-effort send alarm to user's private DM; gracefully handles unstarted/blocked bots."""
    try:
        await bot.send_message(
            chat_id=user_id,
            text=text,
            reply_markup=reply_markup,
            parse_mode="HTML",
            disable_web_page_preview=True,
        )
        try:
            await repository.set_bot_user_dm_started(
                user_id,
                True,
                username=username,
                full_name=full_name,
            )
        except Exception:
            pass
        return True
    except TelegramForbiddenError:
        logger.debug("User %s has not started or blocked the bot; skipping DM timer alarm.", user_id)
        try:
            await repository.set_bot_user_dm_started(user_id, False)
        except Exception:
            pass
        return False
    except TelegramBadRequest as exc:
        logger.debug("TelegramBadRequest sending DM timer alarm to user %s: %s", user_id, exc)
        return False
    except Exception as exc:
        logger.warning("Unexpected error sending DM timer alarm to user %s: %s", user_id, exc)
        return False


async def fire_timer(
    bot: Bot,
    timer_id: int,
    *,
    now_utc: datetime | None = None,
) -> bool:
    """Atomically claim and fire a timer, notifying both the original topic and user DM simultaneously."""
    current_utc = now_utc.astimezone(timezone.utc) if now_utc else datetime.now(timezone.utc)
    timer = await repository.claim_timer_for_firing(timer_id, fired_at_iso=to_utc_iso(current_utc))
    if timer is None:
        return False

    settings = get_settings()
    tz_name = settings.TIMEZONE
    user_id = int(timer["user_id"])
    chat_id = int(timer["chat_id"])
    thread_id = int(timer["thread_id"])
    chat_type = str(timer.get("chat_type") or "supergroup")
    username = timer.get("username")
    full_name = str(timer.get("full_name") or "کاربر")

    if chat_type == "private":
        dm_text = _build_dm_alarm_text(
            timer,
            group_title=None,
            topic_name=None,
            now_utc=current_utc,
            tz_name=tz_name,
        )
        dm_kb = build_fired_alarm_kb(user_id, timer_id, topic_url=None)
        await _send_dm_alarm(
            bot,
            user_id=user_id,
            username=username,
            full_name=full_name,
            text=dm_text,
            reply_markup=dm_kb,
        )
        return True

    # Group timer -> resolve metadata for DM context and dispatch both simultaneously
    group_rec = await repository.get_known_group(chat_id)
    group_title = (group_rec.get("title") if group_rec else None) or f"گروه {chat_id}"
    topic_name = await repository.get_topic_name(chat_id, thread_id)
    if not topic_name:
        topic_name = "عمومی" if thread_id == 1 else f"تاپیک {thread_id}"

    topic_url = build_topic_deep_link(chat_id, thread_id)
    group_text = _build_group_alarm_text(timer, now_utc=current_utc, tz_name=tz_name)
    dm_text = _build_dm_alarm_text(
        timer,
        group_title=group_title,
        topic_name=topic_name,
        now_utc=current_utc,
        tz_name=tz_name,
    )
    group_kb = build_fired_alarm_kb(user_id, timer_id, topic_url=None)
    dm_kb = build_fired_alarm_kb(user_id, timer_id, topic_url=topic_url)

    await asyncio.gather(
        _send_group_alarm(
            bot,
            chat_id=chat_id,
            thread_id=thread_id,
            text=group_text,
            reply_markup=group_kb,
        ),
        _send_dm_alarm(
            bot,
            user_id=user_id,
            username=username,
            full_name=full_name,
            text=dm_text,
            reply_markup=dm_kb,
        ),
        return_exceptions=True,
    )
    return True


async def rehydrate_active_timers(
    bot: Bot,
    *,
    now_utc: datetime | None = None,
) -> int:
    """Restore all active timers from SQLite on startup, firing missed ones and scheduling future ones."""
    current_utc = now_utc.astimezone(timezone.utc) if now_utc else datetime.now(timezone.utc)
    active_timers = await repository.list_all_active_timers()
    processed = 0

    for t in active_timers:
        fire_utc = parse_utc_iso(t["fire_at"])
        if fire_utc <= current_utc:
            await fire_timer(bot, int(t["id"]), now_utc=current_utc)
        else:
            schedule_timer_job(bot, t)
        processed += 1

    if processed:
        logger.info("Rehydrated %s active timer(s) from database.", processed)
    return processed


async def sweep_due_timers(
    bot: Bot,
    *,
    now_utc: datetime | None = None,
) -> int:
    """Periodic safety-net sweep that fires any active timer whose fire_at has elapsed."""
    current_utc = now_utc.astimezone(timezone.utc) if now_utc else datetime.now(timezone.utc)
    due = await repository.list_due_active_timers(to_utc_iso(current_utc))
    fired = 0
    for t in due:
        if await fire_timer(bot, int(t["id"]), now_utc=current_utc):
            fired += 1
    return fired

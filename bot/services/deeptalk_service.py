"""DeepTalk lifecycle service: APScheduler jobs, topic cards, DM start pings, live extensions, and rehydration."""

import asyncio
from datetime import datetime, timedelta, timezone
import logging
from typing import Any
from zoneinfo import ZoneInfo
from aiogram import Bot
from aiogram.exceptions import TelegramBadRequest, TelegramForbiddenError, TelegramRetryAfter
from aiogram.types import InlineKeyboardMarkup, Message
from apscheduler.jobstores.base import JobLookupError
from apscheduler.triggers.date import DateTrigger
from bot.config import get_settings
from bot.database import repository
from bot.keyboards.deeptalk import (
    build_announcement_card_kb,
    build_dm_reminder_kb,
    build_end_card_kb,
    build_pv_participants_kb,
    build_start_card_kb,
)
from bot.services.timer_service import build_topic_deep_link, ensure_bot_username
from bot.texts.deeptalk_texts import (
    format_deeptalk_announcement_card,
    format_deeptalk_dm_reminder,
    format_deeptalk_end_card,
    format_deeptalk_participants_pv,
    format_deeptalk_start_card,
)
from bot.utils.general_topic import to_send_thread_id
from bot.utils.timer_parser import (
    MAX_TIMER_DAYS,
    format_duration_persian,
    parse_utc_iso,
    to_persian_digits,
    to_utc_iso,
)

logger = logging.getLogger(__name__)

MIN_DEEPTALK_DURATION_MINUTES = 5
MAX_DEEPTALK_DURATION_MINUTES = 360
MAX_SUBJECT_LENGTH = 200
MAX_DESCRIPTION_LENGTH = 600

# O(1) in-memory map of currently 'live' DeepTalk sessions: (chat_id, thread_id) -> session_id
_live_sessions_by_topic: dict[tuple[int, int], int] = {}


def clear_deeptalk_runtime_cache() -> None:
    """Clear in-memory live topic cache (useful for test isolation)."""
    _live_sessions_by_topic.clear()


def get_live_session_id_for_topic(chat_id: int, thread_id: int) -> int | None:
    """Return active 'live' DeepTalk session_id in (chat_id, thread_id), or None."""
    return _live_sessions_by_topic.get((chat_id, thread_id))


def _set_live_topic_session(chat_id: int, thread_id: int, session_id: int) -> None:
    _live_sessions_by_topic[(int(chat_id), int(thread_id))] = int(session_id)


def _clear_live_topic_session(chat_id: int, thread_id: int, session_id: int) -> None:
    key = (int(chat_id), int(thread_id))
    if _live_sessions_by_topic.get(key) == int(session_id):
        _live_sessions_by_topic.pop(key, None)


def deeptalk_start_job_id(session_id: int) -> str:
    """Canonical APScheduler job ID for starting a DeepTalk session."""
    return f"deeptalk_start_{session_id}"


def deeptalk_end_job_id(session_id: int) -> str:
    """Canonical APScheduler job ID for ending a DeepTalk session."""
    return f"deeptalk_end_{session_id}"


def schedule_deeptalk_start_job(bot: Bot, session_row: dict[str, Any]) -> bool:
    """Register exact DateTrigger job to fire start_deeptalk_session at start_at."""
    from bot.scheduler.jobs import get_scheduler

    scheduler = get_scheduler()
    if scheduler is None:
        return False

    session_id = int(session_row["id"])
    start_utc = parse_utc_iso(str(session_row["start_at"]))
    try:
        scheduler.add_job(
            start_deeptalk_session,
            trigger=DateTrigger(run_date=start_utc, timezone=timezone.utc),
            args=[bot, session_id],
            id=deeptalk_start_job_id(session_id),
            name=f"DeepTalk Start #{session_id}",
            replace_existing=True,
            misfire_grace_time=None,
            max_instances=1,
        )
        return True
    except Exception as exc:
        logger.error("Failed scheduling DeepTalk start %s: %s", session_id, exc, exc_info=True)
        return False


def schedule_deeptalk_end_job(bot: Bot, session_row: dict[str, Any]) -> bool:
    """Register exact DateTrigger job to fire end_deeptalk_session at end_at."""
    from bot.scheduler.jobs import get_scheduler

    scheduler = get_scheduler()
    if scheduler is None:
        return False

    session_id = int(session_row["id"])
    end_utc = parse_utc_iso(str(session_row["end_at"]))
    try:
        scheduler.add_job(
            end_deeptalk_session,
            trigger=DateTrigger(run_date=end_utc, timezone=timezone.utc),
            args=[bot, session_id],
            id=deeptalk_end_job_id(session_id),
            name=f"DeepTalk End #{session_id}",
            replace_existing=True,
            misfire_grace_time=None,
            max_instances=1,
        )
        return True
    except Exception as exc:
        logger.error("Failed scheduling DeepTalk end %s: %s", session_id, exc, exc_info=True)
        return False


def unschedule_deeptalk_jobs(session_id: int) -> None:
    """Remove pending start and end APScheduler jobs for a DeepTalk session."""
    from bot.scheduler.jobs import get_scheduler

    scheduler = get_scheduler()
    if scheduler is None:
        return
    for jid in (deeptalk_start_job_id(session_id), deeptalk_end_job_id(session_id)):
        try:
            scheduler.remove_job(jid)
        except JobLookupError:
            pass
        except Exception as exc:
            logger.debug("Ignoring error unscheduling %s: %s", jid, exc)


async def send_deeptalk_topic_message(
    bot: Bot,
    *,
    chat_id: int,
    thread_id: int,
    text: str,
    reply_markup: InlineKeyboardMarkup | None = None,
) -> Message | None:
    """Send an HTML message into the target forum topic with General fallback and flood retry."""
    target_thread = to_send_thread_id(thread_id)
    for attempt in range(3):
        try:
            return await bot.send_message(
                chat_id=chat_id,
                message_thread_id=target_thread,
                text=text,
                reply_markup=reply_markup,
                parse_mode="HTML",
                disable_web_page_preview=True,
            )
        except TelegramRetryAfter as exc:
            if attempt == 0:
                await asyncio.sleep(min(float(exc.retry_after) + 1.0, 15.0))
                continue
            logger.warning("DeepTalk topic send hit TelegramRetryAfter twice for chat %s", chat_id)
            return None
        except TelegramBadRequest as exc:
            err = str(exc).lower()
            if ("message thread not found" in err or "thread_not_found" in err) and target_thread is not None:
                logger.warning(
                    "DeepTalk topic %s not found in chat %s; falling back to General.",
                    target_thread,
                    chat_id,
                )
                target_thread = None
                continue
            logger.error("TelegramBadRequest sending DeepTalk message to chat %s: %s", chat_id, exc)
            return None
        except Exception as exc:
            logger.error("Failed sending DeepTalk message to chat %s: %s", chat_id, exc)
            return None
    return None


async def _safe_edit_topic_card(
    bot: Bot,
    *,
    chat_id: int,
    message_id: int | None,
    text: str,
    reply_markup: InlineKeyboardMarkup | None,
) -> bool:
    """Best-effort edit of an existing DeepTalk card in a group topic."""
    if not message_id:
        return False
    try:
        await bot.edit_message_text(
            chat_id=chat_id,
            message_id=int(message_id),
            text=text,
            reply_markup=reply_markup,
            parse_mode="HTML",
            disable_web_page_preview=True,
        )
        return True
    except TelegramBadRequest as exc:
        if "message is not modified" in str(exc).lower():
            return True
        logger.debug("Could not edit DeepTalk card %s/%s: %s", chat_id, message_id, exc)
        return False
    except Exception as exc:
        logger.debug("Unexpected error editing DeepTalk card %s/%s: %s", chat_id, message_id, exc)
        return False


async def refresh_announcement_card(
    bot: Bot,
    session_id: int,
    *,
    registered_count: int | None = None,
) -> None:
    """Update the live counter or status on a session's announcement message."""
    session = await repository.get_deeptalk_session(session_id)
    if session is None or not session.get("announce_message_id"):
        return

    settings = get_settings()
    if registered_count is None:
        stats = await repository.get_deeptalk_session_stats(session_id)
        registered_count = int(stats["registered_count"])

    bot_username = await ensure_bot_username(bot)
    text = format_deeptalk_announcement_card(
        session,
        registered_count=registered_count,
        tz_name=settings.TIMEZONE,
    )
    kb = build_announcement_card_kb(
        session_id,
        registered_count=registered_count,
        bot_username=bot_username,
        status=str(session.get("status") or "scheduled"),
    )
    await _safe_edit_topic_card(
        bot,
        chat_id=int(session["chat_id"]),
        message_id=int(session["announce_message_id"]),
        text=text,
        reply_markup=kb,
    )


async def refresh_start_card(
    bot: Bot,
    session_id: int,
    *,
    checked_in_count: int | None = None,
) -> None:
    """Update the live check-in counter, extended time, or status on a session's start message."""
    session = await repository.get_deeptalk_session(session_id)
    if session is None or not session.get("start_message_id"):
        return

    settings = get_settings()
    stats = await repository.get_deeptalk_session_stats(session_id)
    reg_cnt = int(stats["registered_count"])
    chk_cnt = int(checked_in_count) if checked_in_count is not None else int(stats["checked_in_count"])
    bot_username = await ensure_bot_username(bot)

    text = format_deeptalk_start_card(
        session,
        checked_in_count=chk_cnt,
        registered_count=reg_cnt,
        tz_name=settings.TIMEZONE,
    )
    kb = build_start_card_kb(
        session_id,
        checked_in_count=chk_cnt,
        bot_username=bot_username,
        status=str(session.get("status") or "live"),
    )
    await _safe_edit_topic_card(
        bot,
        chat_id=int(session["chat_id"]),
        message_id=int(session["start_message_id"]),
        text=text,
        reply_markup=kb,
    )


async def create_and_publish_deeptalk(
    bot: Bot,
    *,
    chat_id: int,
    thread_id: int,
    subject: str,
    description: str | None,
    created_by: int,
    creator_name: str,
    start_at: datetime,
    duration_minutes: int,
    now_utc: datetime | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Validate inputs, persist DeepTalk session, post announcement card to topic, and schedule start job."""
    clean_subj = (subject or "").strip()
    if not clean_subj:
        return None, "⚠️ لطفاً ابتدا محوریت گفتگو را مشخص کن!"
    if len(clean_subj) > MAX_SUBJECT_LENGTH:
        clean_subj = clean_subj[: MAX_SUBJECT_LENGTH - 1].rstrip() + "…"

    clean_desc = (description or "").strip() if description else None
    if clean_desc and len(clean_desc) > MAX_DESCRIPTION_LENGTH:
        clean_desc = clean_desc[: MAX_DESCRIPTION_LENGTH - 1].rstrip() + "…"

    if not (MIN_DEEPTALK_DURATION_MINUTES <= duration_minutes <= MAX_DEEPTALK_DURATION_MINUTES):
        return (
            None,
            f"⚠️ مدت گفتگو باید بین {to_persian_digits(MIN_DEEPTALK_DURATION_MINUTES)} تا "
            f"{to_persian_digits(MAX_DEEPTALK_DURATION_MINUTES)} دقیقه باشد.",
        )

    current_utc = now_utc.astimezone(timezone.utc) if now_utc else datetime.now(timezone.utc)
    start_utc = (
        start_at.astimezone(timezone.utc)
        if start_at.tzinfo
        else start_at.replace(tzinfo=timezone.utc)
    )
    delta = start_utc - current_utc
    if delta.total_seconds() < 15:
        return None, "⚠️ زمان شروع گفتگو باید در آینده باشد! لطفاً ساعت را جلوتر تنظیم کن."
    if delta > timedelta(days=MAX_TIMER_DAYS):
        return None, f"⚠️ حداکثر بازه زمانی مجاز {to_persian_digits(MAX_TIMER_DAYS)} روز است."

    end_utc = start_utc + timedelta(minutes=duration_minutes)
    session_row = await repository.create_deeptalk_session(
        chat_id=chat_id,
        thread_id=thread_id,
        subject=clean_subj,
        description=clean_desc,
        created_by=created_by,
        creator_name=creator_name,
        start_at_iso=to_utc_iso(start_utc),
        end_at_iso=to_utc_iso(end_utc),
        duration_minutes=duration_minutes,
        created_at_iso=to_utc_iso(current_utc),
    )
    session_id = int(session_row["id"])

    settings = get_settings()
    bot_username = await ensure_bot_username(bot)
    card_text = format_deeptalk_announcement_card(
        session_row,
        registered_count=0,
        now_tehran=current_utc.astimezone(ZoneInfo(settings.TIMEZONE)),
        tz_name=settings.TIMEZONE,
    )
    card_kb = build_announcement_card_kb(
        session_id,
        registered_count=0,
        bot_username=bot_username,
        status="scheduled",
    )

    sent = await send_deeptalk_topic_message(
        bot,
        chat_id=chat_id,
        thread_id=thread_id,
        text=card_text,
        reply_markup=card_kb,
    )
    if sent is None:
        await repository.cancel_deeptalk_session(session_id, ended_at_iso=to_utc_iso(current_utc))
        return None, "❌ خطا در ارسال فراخوان به تاپیک گروه (دسترسی ربات در گروه را بررسی کن)."

    msg_id = getattr(sent, "message_id", None)
    if isinstance(msg_id, int):
        await repository.update_deeptalk_message_ids(session_id, announce_message_id=msg_id)
        session_row["announce_message_id"] = msg_id

    schedule_deeptalk_start_job(bot, session_row)
    return session_row, None


async def _dispatch_start_dm_reminders(
    bot: Bot,
    session: dict[str, Any],
) -> None:
    """Send private start reminders to pre-registered users who have dm_started=1."""
    session_id = int(session["id"])
    targets = await repository.list_deeptalk_dm_targets(session_id)
    if not targets:
        return

    settings = get_settings()
    chat_id = int(session["chat_id"])
    thread_id = int(session["thread_id"])
    group_rec = await repository.get_known_group(chat_id)
    group_title = (group_rec.get("title") if group_rec else None) or None
    topic_name = await repository.get_topic_name(chat_id, thread_id)
    if not topic_name:
        topic_name = "عمومی" if thread_id == 1 else f"تاپیک {thread_id}"

    dm_text = format_deeptalk_dm_reminder(
        session,
        group_title=group_title,
        topic_name=topic_name,
        tz_name=settings.TIMEZONE,
    )
    dm_kb = build_dm_reminder_kb(build_topic_deep_link(chat_id, thread_id))

    for idx, t in enumerate(targets):
        uid = int(t["user_id"])
        try:
            await bot.send_message(
                chat_id=uid,
                text=dm_text,
                reply_markup=dm_kb,
                parse_mode="HTML",
                disable_web_page_preview=True,
            )
        except TelegramForbiddenError:
            try:
                await repository.set_bot_user_dm_started(uid, False)
            except Exception:
                pass
        except TelegramRetryAfter as exc:
            await asyncio.sleep(min(float(exc.retry_after) + 1.0, 10.0))
        except Exception as exc:
            logger.debug("Skipping DeepTalk DM reminder for user %s: %s", uid, exc)

        if idx > 0 and idx % 15 == 0:
            await asyncio.sleep(0.5)


async def start_deeptalk_session(
    bot: Bot,
    session_id: int,
    *,
    force_now: bool = False,
    now_utc: datetime | None = None,
) -> bool:
    """Atomically transition a scheduled DeepTalk to 'live', post start card, and send DM pings."""
    current_utc = now_utc.astimezone(timezone.utc) if now_utc else datetime.now(timezone.utc)
    new_end_iso: str | None = None

    if force_now:
        existing = await repository.get_deeptalk_session(session_id)
        if existing is None or str(existing.get("status")) != "scheduled":
            return False
        dur = int(existing.get("duration_minutes") or 45)
        new_end_iso = to_utc_iso(current_utc + timedelta(minutes=dur))
        unschedule_deeptalk_jobs(session_id)

    session = await repository.claim_deeptalk_for_start(
        session_id,
        started_at_iso=to_utc_iso(current_utc),
        new_end_at_iso=new_end_iso,
    )
    if session is None:
        return False

    chat_id = int(session["chat_id"])
    thread_id = int(session["thread_id"])
    _set_live_topic_session(chat_id, thread_id, session_id)
    schedule_deeptalk_end_job(bot, session)

    settings = get_settings()
    stats = await repository.get_deeptalk_session_stats(session_id)
    reg_cnt = int(stats["registered_count"])

    # 1. Update original announcement card status
    await refresh_announcement_card(bot, session_id, registered_count=reg_cnt)

    # 2. Send live Start Card with check-in button
    bot_username = await ensure_bot_username(bot)
    start_text = format_deeptalk_start_card(
        session,
        checked_in_count=0,
        registered_count=reg_cnt,
        tz_name=settings.TIMEZONE,
    )
    start_kb = build_start_card_kb(
        session_id,
        checked_in_count=0,
        bot_username=bot_username,
        status="live",
    )
    sent = await send_deeptalk_topic_message(
        bot,
        chat_id=chat_id,
        thread_id=thread_id,
        text=start_text,
        reply_markup=start_kb,
    )
    msg_id = getattr(sent, "message_id", None) if sent is not None else None
    if isinstance(msg_id, int):
        await repository.update_deeptalk_message_ids(session_id, start_message_id=msg_id)

    # 3. Best-effort DM reminders to pre-registered participants
    await _dispatch_start_dm_reminders(bot, session)
    return True


async def end_deeptalk_session(
    bot: Bot,
    session_id: int,
    *,
    now_utc: datetime | None = None,
) -> bool:
    """Atomically end a 'live' DeepTalk session and publish the wrap-up & stats card."""
    current_utc = now_utc.astimezone(timezone.utc) if now_utc else datetime.now(timezone.utc)
    session = await repository.claim_deeptalk_for_end(
        session_id,
        ended_at_iso=to_utc_iso(current_utc),
    )
    if session is None:
        return False

    chat_id = int(session["chat_id"])
    thread_id = int(session["thread_id"])
    unschedule_deeptalk_jobs(session_id)
    _clear_live_topic_session(chat_id, thread_id, session_id)

    stats = await repository.get_deeptalk_session_stats(session_id)

    # 1. Close buttons on start & announcement cards
    await refresh_start_card(bot, session_id, checked_in_count=int(stats["checked_in_count"]))
    await refresh_announcement_card(bot, session_id, registered_count=int(stats["registered_count"]))

    # 2. Send wrap-up card with fun session stats
    bot_username = await ensure_bot_username(bot)
    end_text = format_deeptalk_end_card(session, stats)
    end_kb = build_end_card_kb(session_id, bot_username=bot_username)
    sent = await send_deeptalk_topic_message(
        bot,
        chat_id=chat_id,
        thread_id=thread_id,
        text=end_text,
        reply_markup=end_kb,
    )
    msg_id = getattr(sent, "message_id", None) if sent is not None else None
    if isinstance(msg_id, int):
        await repository.update_deeptalk_message_ids(session_id, end_message_id=msg_id)

    return True


async def extend_live_deeptalk(
    bot: Bot,
    session_id: int,
    extra_minutes: int,
    *,
    now_utc: datetime | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Extend a currently 'live' DeepTalk session, reschedule end job, and notify the topic."""
    updated = await repository.extend_deeptalk_session(
        session_id,
        extra_minutes,
        now_utc=now_utc,
    )
    if updated is None:
        return None, "⚠️ این جلسه در حال حاضر در وضعیت «در حال برگزاری» نیست."

    schedule_deeptalk_end_job(bot, updated)
    await refresh_start_card(bot, session_id)

    settings = get_settings()
    tz = ZoneInfo(settings.TIMEZONE)
    end_local = parse_utc_iso(str(updated["end_at"])).astimezone(tz)
    end_clock = to_persian_digits(end_local.strftime("%H:%M"))
    dur_str = format_duration_persian(extra_minutes)

    notice = (
        f"🔥 <b>بحث داغه! زمان دیپ‌تاک {dur_str} تمدید شد!</b>\n"
        f"⏰ گفتگو تا ساعت <b>{end_clock}</b> ادامه داره — بترکونین! 🚀"
    )
    await send_deeptalk_topic_message(
        bot,
        chat_id=int(updated["chat_id"]),
        thread_id=int(updated["thread_id"]),
        text=notice,
    )
    return updated, None


async def cancel_deeptalk_flow(
    bot: Bot,
    session_id: int,
    *,
    now_utc: datetime | None = None,
) -> bool:
    """Cancel a scheduled or live DeepTalk session and update its topic cards."""
    current_utc = now_utc.astimezone(timezone.utc) if now_utc else datetime.now(timezone.utc)
    cancelled = await repository.cancel_deeptalk_session(
        session_id,
        ended_at_iso=to_utc_iso(current_utc),
    )
    if cancelled is None:
        return False

    chat_id = int(cancelled["chat_id"])
    thread_id = int(cancelled["thread_id"])
    unschedule_deeptalk_jobs(session_id)
    _clear_live_topic_session(chat_id, thread_id, session_id)

    await refresh_announcement_card(bot, session_id)
    await refresh_start_card(bot, session_id)
    return True


async def render_pv_participants_view(
    session_id: int,
) -> tuple[str, InlineKeyboardMarkup] | None:
    """Build the private-chat participant list view and refresh keyboard for a DeepTalk session."""
    session = await repository.get_deeptalk_session(session_id)
    if session is None:
        return None

    settings = get_settings()
    chat_id = int(session["chat_id"])
    thread_id = int(session["thread_id"])
    participants = await repository.list_deeptalk_participants(session_id)

    group_rec = await repository.get_known_group(chat_id)
    group_title = (group_rec.get("title") if group_rec else None) or None
    topic_name = await repository.get_topic_name(chat_id, thread_id)
    if not topic_name:
        topic_name = "عمومی" if thread_id == 1 else f"تاپیک {thread_id}"

    text = format_deeptalk_participants_pv(
        session,
        participants,
        group_title=group_title,
        topic_name=topic_name,
        tz_name=settings.TIMEZONE,
    )
    kb = build_pv_participants_kb(
        session_id,
        topic_url=build_topic_deep_link(chat_id, thread_id),
    )
    return text, kb


async def rehydrate_deeptalk_sessions(
    bot: Bot,
    *,
    now_utc: datetime | None = None,
) -> int:
    """Restore scheduled and live DeepTalk sessions on bot startup."""
    current_utc = now_utc.astimezone(timezone.utc) if now_utc else datetime.now(timezone.utc)
    sessions = await repository.list_active_or_scheduled_deeptalks()
    processed = 0

    for s in sessions:
        sid = int(s["id"])
        status = str(s["status"])
        start_utc = parse_utc_iso(str(s["start_at"]))
        end_utc = parse_utc_iso(str(s["end_at"]))

        if status == "live":
            if end_utc <= current_utc:
                await end_deeptalk_session(bot, sid, now_utc=current_utc)
            else:
                _set_live_topic_session(int(s["chat_id"]), int(s["thread_id"]), sid)
                schedule_deeptalk_end_job(bot, s)
        elif status == "scheduled":
            if end_utc <= current_utc:
                # Entire session window passed while offline -> cancel cleanly without spamming start+end
                await cancel_deeptalk_flow(bot, sid, now_utc=current_utc)
            elif start_utc <= current_utc:
                await start_deeptalk_session(bot, sid, now_utc=current_utc)
            else:
                schedule_deeptalk_start_job(bot, s)
        processed += 1

    if processed:
        logger.info("Rehydrated %s DeepTalk session(s) from database.", processed)
    return processed


async def sweep_due_deeptalks(
    bot: Bot,
    *,
    now_utc: datetime | None = None,
) -> int:
    """Periodic safety-net sweep firing any overdue DeepTalk starts or ends."""
    current_utc = now_utc.astimezone(timezone.utc) if now_utc else datetime.now(timezone.utc)
    now_iso = to_utc_iso(current_utc)
    triggered = 0

    due_starts = await repository.list_due_deeptalk_starts(now_iso)
    for s in due_starts:
        sid = int(s["id"])
        end_utc = parse_utc_iso(str(s["end_at"]))
        if end_utc <= current_utc:
            if await cancel_deeptalk_flow(bot, sid, now_utc=current_utc):
                triggered += 1
        else:
            if await start_deeptalk_session(bot, sid, now_utc=current_utc):
                triggered += 1

    due_ends = await repository.list_due_deeptalk_ends(now_iso)
    for s in due_ends:
        sid = int(s["id"])
        if await end_deeptalk_session(bot, sid, now_utc=current_utc):
            triggered += 1

    return triggered

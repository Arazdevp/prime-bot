"""Interactive /timer panel handlers, draft sessions, group-safe FSM inputs, and post-alarm actions."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import logging
import time
from typing import Any, Literal
from zoneinfo import ZoneInfo
from aiogram import Bot, Router
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command, Filter, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import CallbackQuery, InlineKeyboardMarkup, Message, User
import jdatetime
from bot.config import get_settings
from bot.database import repository
from bot.handlers.panel import safe_answer_callback, safe_edit_message
from bot.keyboards.panel import paginate
from bot.keyboards.timer import (
    TIMER_PAGE_SIZE,
    TimerCB,
    build_active_timers_kb,
    build_calendar_kb,
    build_cancel_all_confirm_kb,
    build_input_cancel_kb,
    build_quick_confirm_kb,
    build_relative_kb,
    build_saved_timer_kb,
    build_time_picker_kb,
    build_timer_home_kb,
)
from bot.services.access import UserRole, is_active_group_member, resolve_user_role
from bot.services.timer_service import (
    create_and_schedule_timer,
    ensure_bot_username,
    unschedule_timer_job,
)
from bot.utils.dates import format_persian_short_date, get_tehran_now
from bot.utils.general_topic import resolve_thread_id, to_send_thread_id
from bot.utils.html_utils import escape_html
from bot.utils.timer_parser import (
    MAX_NOTE_LENGTH,
    MAX_TIMER_MINUTES,
    MIN_TIMER_MINUTES,
    format_duration_persian,
    format_fire_time_persian,
    format_jalali_day_label,
    format_remaining_persian,
    jalali_to_tehran_dt,
    parse_duration_input,
    parse_jalali_date_time_input,
    parse_quick_timer_command,
    parse_time_input,
    parse_utc_iso,
    sanitize_note,
    to_persian_digits,
)

logger = logging.getLogger(__name__)

router = Router(name="timer_router")

DRAFT_TTL_SECONDS = 1800.0      # 30 minutes
FSM_INPUT_TTL_SECONDS = 120.0   # 2 minutes
GROUP_CMD_COOLDOWN_SECONDS = 3.0


class TimerStates(StatesGroup):
    """FSM states for optional custom text input in the /timer wizard."""

    waiting_note = State()
    waiting_custom_duration = State()
    waiting_custom_time = State()
    waiting_custom_datetime = State()


@dataclass
class TimerDraft:
    """In-memory wizard state scoped to a single /timer panel message."""

    owner_id: int
    chat_id: int
    thread_id: int
    chat_type: str
    mode: Literal["relative", "scheduled"] = "relative"
    duration_minutes: int = 30
    jyear: int = 1405
    jmonth: int = 1
    jday: int = 1
    cal_year: int = 1405
    cal_month: int = 1
    hour: int = 9
    minute: int = 0
    time_customized: bool = False
    note: str | None = None
    screen: str = "home"
    updated_at: float = field(default_factory=time.monotonic)


# (chat_id, panel_message_id) -> TimerDraft
_drafts: dict[tuple[int, int], TimerDraft] = {}
# (chat_id, user_id) -> last /timer monotonic timestamp
_group_cmd_cooldown: dict[tuple[int, int], float] = {}


def clear_timer_runtime_state() -> None:
    """Clear in-memory drafts and cooldowns (useful for test isolation)."""
    _drafts.clear()
    _group_cmd_cooldown.clear()


def _prune_stale_drafts() -> None:
    now = time.monotonic()
    expired = [k for k, v in _drafts.items() if (now - v.updated_at) > DRAFT_TTL_SECONDS]
    for k in expired:
        _drafts.pop(k, None)


def _default_today_time(now_tehran: datetime) -> tuple[int, int]:
    """Compute a convenient future clock time for today (next +1h rounded to 15m, or +15m near midnight)."""
    candidate = now_tehran + timedelta(minutes=60)
    rounded_min = (candidate.minute // 15) * 15
    candidate = candidate.replace(minute=rounded_min, second=0, microsecond=0)
    if candidate.date() != now_tehran.date():
        candidate = now_tehran + timedelta(minutes=15)
        if candidate.date() != now_tehran.date():
            return 23, 59
    return candidate.hour, candidate.minute


def _new_draft(
    owner_id: int,
    chat_id: int,
    thread_id: int,
    chat_type: str,
    now_tehran: datetime | None = None,
) -> TimerDraft:
    now = now_tehran or get_tehran_now()
    today_j = jdatetime.date.fromgregorian(date=now.date())
    h, m = _default_today_time(now)
    return TimerDraft(
        owner_id=owner_id,
        chat_id=chat_id,
        thread_id=thread_id,
        chat_type=chat_type,
        mode="relative",
        duration_minutes=30,
        jyear=today_j.year,
        jmonth=today_j.month,
        jday=today_j.day,
        cal_year=today_j.year,
        cal_month=today_j.month,
        hour=h,
        minute=m,
        time_customized=False,
        note=None,
        screen="home",
        updated_at=time.monotonic(),
    )


def _get_or_create_draft(
    callback: CallbackQuery,
    owner_id: int,
    *,
    create_if_missing: bool = True,
) -> TimerDraft | None:
    _prune_stale_drafts()
    msg = callback.message
    if msg is None or not getattr(msg, "chat", None):
        return None
    key = (msg.chat.id, msg.message_id)
    draft = _drafts.get(key)
    if draft is not None:
        draft.updated_at = time.monotonic()
        return draft
    if not create_if_missing:
        return None
    chat_type = "private" if msg.chat.type == "private" else "supergroup"
    thread_id = 1 if chat_type == "private" else resolve_thread_id(msg)
    draft = _new_draft(owner_id, msg.chat.id, thread_id, chat_type)
    _drafts[key] = draft
    return draft


def _is_allowed_timer_chat(chat: Any) -> bool:
    if not chat:
        return False
    if chat.type == "private":
        return True
    if chat.type != "supergroup":
        return False
    settings = get_settings()
    if settings.ALLOWED_CHAT_IDS and chat.id not in settings.ALLOWED_CHAT_IDS:
        return False
    return True


def _user_display_name(user: User) -> str:
    return (user.first_name or user.full_name or "دوست عزیز").strip() or "دوست عزیز"


async def _topic_label(chat_id: int, thread_id: int, chat_type: str) -> str:
    if chat_type == "private":
        return "پیوی شخصی"
    name = await repository.get_topic_name(chat_id, thread_id)
    if name:
        return name
    return "عمومی" if thread_id == 1 else f"تاپیک {thread_id}"


def _extract_safe_html(msg: Any) -> str:
    """Safely extract HTML text from a Telegram Message (falling back to escaped plain text)."""
    html_val = getattr(msg, "html_text", None)
    if isinstance(html_val, str) and html_val:
        return html_val
    raw = getattr(msg, "text", None)
    if isinstance(raw, str) and raw:
        return escape_html(raw)
    return ""


async def _try_delete_msg(message: Message | None) -> None:
    if message is None:
        return
    try:
        await message.delete()
    except Exception:
        pass


async def _send_in_origin(
    message: Message,
    text: str,
    reply_markup: InlineKeyboardMarkup | None = None,
) -> Message | None:
    """Send a message back into the exact origin (private chat or forum topic)."""
    if not message.chat or not message.bot:
        return None
    if message.chat.type == "private":
        try:
            return await message.answer(
                text=text,
                reply_markup=reply_markup,
                parse_mode="HTML",
                disable_web_page_preview=True,
            )
        except Exception as exc:
            logger.error("Failed sending private timer message: %s", exc)
            return None

    thread_id = resolve_thread_id(message)
    target_tid = to_send_thread_id(thread_id)
    try:
        return await message.bot.send_message(
            chat_id=message.chat.id,
            message_thread_id=target_tid,
            text=text,
            reply_markup=reply_markup,
            parse_mode="HTML",
            disable_web_page_preview=True,
        )
    except Exception as exc:
        logger.error("Failed sending group timer message: %s", exc)
        return None


async def _safe_edit_by_id(
    bot: Bot,
    chat_id: int,
    message_id: int,
    text: str,
    reply_markup: InlineKeyboardMarkup | None = None,
) -> bool:
    try:
        await bot.edit_message_text(
            chat_id=chat_id,
            message_id=message_id,
            text=text,
            reply_markup=reply_markup,
            parse_mode="HTML",
            disable_web_page_preview=True,
        )
        return True
    except TelegramBadRequest as exc:
        if "message is not modified" in str(exc).lower():
            return True
        logger.debug("Could not edit timer panel %s/%s: %s", chat_id, message_id, exc)
        return False
    except Exception as exc:
        logger.debug("Unexpected error editing timer panel %s/%s: %s", chat_id, message_id, exc)
        return False


# ---------------- Screen Renderers ----------------


async def render_timer_home(
    bot: Bot,
    draft: TimerDraft,
    user: User,
    *,
    now_tehran: datetime | None = None,
) -> tuple[str, InlineKeyboardMarkup]:
    """Render Screen 1: Main Timer Hub."""
    now = now_tehran or get_tehran_now()
    is_private = draft.chat_type == "private"
    active_count = await repository.count_active_timers_for_user(draft.owner_id)
    dm_started = True if is_private else await repository.is_bot_user_dm_started(draft.owner_id)
    bot_username = None if dm_started else await ensure_bot_username(bot)
    topic_name = await _topic_label(draft.chat_id, draft.thread_id, draft.chat_type)

    today_j = jdatetime.date.fromgregorian(date=now.date())
    date_line = format_jalali_day_label(today_j)
    time_line = now.strftime("%H:%M")

    if is_private:
        dm_line = "📬 <b>حالت اعلان:</b> پیوی شخصی ربات ✅"
    elif dm_started:
        dm_line = "📬 <b>اعلان پیوی:</b> فعال ✅ (همزمان در تاپیک و پیوی)"
    else:
        dm_line = "📬 <b>اعلان پیوی:</b> غیرفعال ⚠️ (برای دریافت در پیوی، دکمه زیر رو بزن)"

    text = (
        "⏱ <b>تنظیم تایمر و یادآور</b>\n"
        f"👤 کاربر: <b>{escape_html(_user_display_name(user))}</b>  |  📍 {escape_html(topic_name)}\n"
        f"🕒 ساعت تهران: <b>{time_line}</b> — {date_line}\n"
        f"{dm_line}\n\n"
        "یه زمان سریع انتخاب کن یا نوع تایمرت رو مشخص کن 👇\n"
        "<i>💡 میانبر سریع: <code>/timer 30m برم شام</code> یا <code>/timer فردا 09:00 جلسه</code></i>"
    )
    kb = build_timer_home_kb(
        draft.owner_id,
        active_count=active_count,
        dm_started=dm_started,
        bot_username=bot_username,
        is_private=is_private,
    )
    return text, kb


async def render_quick_confirm(
    draft: TimerDraft,
    *,
    now_tehran: datetime | None = None,
) -> tuple[str, InlineKeyboardMarkup]:
    """Render Quick Preset review & optional note screen."""
    now = now_tehran or get_tehran_now()
    target = now + timedelta(minutes=draft.duration_minutes)
    topic_name = await _topic_label(draft.chat_id, draft.thread_id, draft.chat_type)
    dest_label = "پیوی ربات" if draft.chat_type == "private" else f"تاپیک «{topic_name}» + پیوی ربات"
    note_line = (
        f"<b>{escape_html(draft.note)}</b>"
        if draft.note
        else "<i>ندارد (اختیاری)</i>"
    )

    text = (
        "⚡ <b>آماده ثبت تایمر!</b>\n\n"
        f"⏳ <b>مدت:</b> {format_duration_persian(draft.duration_minutes)}\n"
        f"🔔 <b>زمان زنگ خوردن:</b> {format_fire_time_persian(target, now)} (تهران)\n"
        f"📍 <b>مقصد اعلان:</b> {escape_html(dest_label)}\n"
        f"📝 <b>یادداشت:</b> {note_line}"
    )
    return text, build_quick_confirm_kb(draft.owner_id, has_note=bool(draft.note))


async def render_relative_builder(
    draft: TimerDraft,
    *,
    now_tehran: datetime | None = None,
) -> tuple[str, InlineKeyboardMarkup]:
    """Render Screen 2: Interactive countdown builder."""
    now = now_tehran or get_tehran_now()
    target = now + timedelta(minutes=draft.duration_minutes)
    note_line = (
        f"<b>{escape_html(draft.note)}</b>"
        if draft.note
        else "<i>ندارد (اختیاری)</i>"
    )

    text = (
        "⏳ <b>تایمر دقیقه‌ای / ساعتی</b>\n\n"
        f"⏱ <b>مدت انتخابی:</b> <b>{format_duration_persian(draft.duration_minutes)}</b>\n"
        f"🔔 <b>زمان زنگ خوردن:</b> {format_fire_time_persian(target, now)} (تهران)\n"
        f"📝 <b>یادداشت:</b> {note_line}\n\n"
        "با دکمه‌های زیر زمان رو دقیق تنظیم کن 👇"
    )
    return text, build_relative_kb(
        draft.owner_id,
        duration_minutes=draft.duration_minutes,
        has_note=bool(draft.note),
    )


async def render_calendar_picker(
    draft: TimerDraft,
    *,
    now_tehran: datetime | None = None,
) -> tuple[str, InlineKeyboardMarkup]:
    """Render Screen 3: Interactive Jalali Calendar date picker."""
    now = now_tehran or get_tehran_now()
    today_j = jdatetime.date.fromgregorian(date=now.date())
    selected_j = jdatetime.date(draft.jyear, draft.jmonth, draft.jday)
    topic_name = await _topic_label(draft.chat_id, draft.thread_id, draft.chat_type)

    text = (
        "📅 <b>تایمر روزی / تاریخی — انتخاب تاریخ</b>\n\n"
        f"📍 <b>مقصد:</b> {escape_html(topic_name)}\n"
        f"🕒 <b>الان:</b> {format_jalali_day_label(today_j)} — ساعت {now.strftime('%H:%M')}\n"
        f"📆 <b>تاریخ انتخابی فعلی:</b> {format_jalali_day_label(selected_j)}\n\n"
        "یه روز سریع بزن یا از تقویم شمسی انتخاب کن 👇"
    )
    kb = build_calendar_kb(
        draft.owner_id,
        cal_year=draft.cal_year,
        cal_month=draft.cal_month,
        selected_jdate=selected_j,
        today_jdate=today_j,
    )
    return text, kb


async def render_time_picker(
    draft: TimerDraft,
    *,
    now_tehran: datetime | None = None,
) -> tuple[str, InlineKeyboardMarkup]:
    """Render Screen 4: Interactive clock time picker for scheduled date timer."""
    settings = get_settings()
    now = now_tehran or get_tehran_now()
    selected_j = jdatetime.date(draft.jyear, draft.jmonth, draft.jday)
    target = jalali_to_tehran_dt(
        draft.jyear,
        draft.jmonth,
        draft.jday,
        draft.hour,
        draft.minute,
        settings.TIMEZONE,
    )
    delta = target - now
    is_valid_future = delta.total_seconds() >= 30

    if is_valid_future:
        status_line = f"⏳ <i>حدود {format_remaining_persian(delta)}</i>"
    else:
        status_line = "⚠️ <b>این ساعت گذشته! یه ساعت آینده انتخاب کن.</b>"

    note_line = (
        f"<b>{escape_html(draft.note)}</b>"
        if draft.note
        else "<i>ندارد (اختیاری)</i>"
    )

    text = (
        "🕐 <b>تایمر روزی — انتخاب ساعت</b>\n\n"
        f"📆 <b>تاریخ:</b> {format_jalali_day_label(selected_j)} ({format_persian_short_date(target)})\n"
        f"⏰ <b>ساعت انتخابی:</b> <b>{draft.hour:02d}:{draft.minute:02d}</b> (تهران)\n"
        f"{status_line}\n"
        f"📝 <b>یادداشت:</b> {note_line}"
    )
    kb = build_time_picker_kb(
        draft.owner_id,
        hour=draft.hour,
        minute=draft.minute,
        has_note=bool(draft.note),
        is_valid_future=is_valid_future,
    )
    return text, kb


async def render_saved_card(
    bot: Bot,
    timer_row: dict[str, Any],
    *,
    now_tehran: datetime | None = None,
) -> tuple[str, InlineKeyboardMarkup]:
    """Render compact confirmation card after successfully creating and scheduling a timer."""
    settings = get_settings()
    tz = ZoneInfo(settings.TIMEZONE)
    now = now_tehran or get_tehran_now()
    fire_tehran = parse_utc_iso(timer_row["fire_at"]).astimezone(tz)
    owner_id = int(timer_row["user_id"])
    timer_id = int(timer_row["id"])
    is_private = timer_row.get("chat_type") == "private"
    dm_started = True if is_private else await repository.is_bot_user_dm_started(owner_id)
    bot_username = None if dm_started else await ensure_bot_username(bot)

    remaining = format_remaining_persian(fire_tehran - now)
    when_str = format_fire_time_persian(fire_tehran, now, settings.TIMEZONE)
    note = timer_row.get("note")

    if is_private:
        dest_str = "پیوی ربات ✅"
    elif dm_started:
        dest_str = "همین تاپیک + پیوی ربات ✅"
    else:
        dest_str = "همین تاپیک (برای اعلان پیوی، دکمه زیر رو بزن 👇)"

    lines = [
        "✅ <b>تایمرت ثبت شد!</b>",
        "",
        f"👤 <b>کاربر:</b> {escape_html(timer_row.get('full_name') or 'کاربر')}",
        f"🔔 <b>زمان یادآوری:</b> {when_str} <i>({remaining})</i>",
    ]
    if note:
        lines.append(f"📝 <b>یادداشت:</b> {escape_html(note)}")
    lines.append(f"📍 <b>اعلان در:</b> {dest_str}")

    kb = build_saved_timer_kb(
        owner_id,
        timer_id,
        dm_started=dm_started,
        bot_username=bot_username,
        is_private=is_private,
    )
    return "\n".join(lines), kb


async def render_active_timers(
    owner_id: int,
    *,
    page: int = 0,
    now_tehran: datetime | None = None,
) -> tuple[str, InlineKeyboardMarkup]:
    """Render Screen 6: User's active timers with remaining times and 1-tap cancel buttons."""
    settings = get_settings()
    tz = ZoneInfo(settings.TIMEZONE)
    now = now_tehran or get_tehran_now()
    timers = await repository.list_active_timers_for_user(owner_id)

    if not timers:
        text = (
            "📋 <b>تایمرهای فعال من</b>\n\n"
            "الان هیچ تایمر فعالی نداری! 😌\n"
            "برای ساخت یه یادآور جدید، بزن روی «➕ تایمر جدید» 👇"
        )
        return text, build_active_timers_kb(owner_id, [], page=0, tz_info=tz)

    page_items, safe_page, _ = paginate(timers, page, TIMER_PAGE_SIZE)
    lines = [
        f"📋 <b>تایمرهای فعال من ({to_persian_digits(len(timers))} عدد):</b>\n"
    ]
    for idx, t in enumerate(page_items, start=safe_page * TIMER_PAGE_SIZE + 1):
        fire_tehran = parse_utc_iso(t["fire_at"]).astimezone(tz)
        when_str = format_fire_time_persian(fire_tehran, now, settings.TIMEZONE)
        rem_str = format_remaining_persian(fire_tehran - now)
        if t.get("chat_type") == "private":
            where_str = "پیوی شخصی"
        else:
            cid = int(t["chat_id"])
            tid = int(t["thread_id"])
            t_name = await _topic_label(cid, tid, "supergroup")
            g_rec = await repository.get_known_group(cid)
            g_title = (g_rec.get("title") if g_rec else None) or "گروه"
            where_str = f"{g_title} / {t_name}"
        note_line = f"\n   📝 {escape_html(t['note'])}" if t.get("note") else ""
        lines.append(
            f"<b>{to_persian_digits(idx)}.</b> 🔔 {when_str} <i>({rem_str})</i>\n"
            f"   📍 {escape_html(where_str)}{note_line}"
        )

    lines.append("\nبرای لغو هر تایمر، روی دکمه‌ش بزن 👇")
    kb = build_active_timers_kb(owner_id, timers, page=safe_page, tz_info=tz)
    return "\n".join(lines), kb


async def _render_draft_screen(
    bot: Bot,
    draft: TimerDraft,
    user: User,
    screen: str,
) -> tuple[str, InlineKeyboardMarkup]:
    draft.screen = screen
    if screen == "q":
        return await render_quick_confirm(draft)
    if screen == "rel":
        return await render_relative_builder(draft)
    if screen == "cal":
        return await render_calendar_picker(draft)
    if screen == "tm":
        return await render_time_picker(draft)
    return await render_timer_home(bot, draft, user)


# ---------------- Group-Safe FSM Filter ----------------


class ActiveTimerFsmFilter(Filter):
    """Match only non-command messages belonging to a live, non-expired TimerStates session in the same topic."""

    async def __call__(self, message: Message, state: FSMContext) -> bool:
        current = await state.get_state()
        if not current or not str(current).startswith("TimerStates:"):
            return False

        data = await state.get_data()
        expires_at = float(data.get("expires_at") or 0.0)
        if expires_at and time.monotonic() > expires_at:
            # Expired silently: clear state and let normal group messages pass through to listener!
            await state.clear()
            return False

        # Never hijack other slash commands (except /cancel handled separately)
        raw_text = (message.text or "").strip()
        if not raw_text or raw_text.startswith("/"):
            return False

        # In supergroups, only intercept messages sent in the exact same topic as the panel
        if message.chat and message.chat.type != "private":
            expected_tid = data.get("thread_id")
            if expected_tid is not None and resolve_thread_id(message) != int(expected_tid):
                return False

        return True


# ---------------- Command Handlers ----------------


@router.message(Command("cancel"), StateFilter(TimerStates))
async def cmd_timer_cancel_fsm(message: Message, bot: Bot, state: FSMContext) -> None:
    """Cancel an active timer text-input prompt and restore the previous panel screen."""
    user = message.from_user
    if user is None:
        await state.clear()
        return

    data = await state.get_data()
    await state.clear()

    panel_chat_id = data.get("panel_chat_id")
    panel_msg_id = data.get("panel_message_id")
    back_screen = str(data.get("back_screen") or "home")

    if message.chat and message.chat.type != "private":
        await _try_delete_msg(message)

    if panel_chat_id is not None and panel_msg_id is not None:
        draft = _drafts.get((int(panel_chat_id), int(panel_msg_id)))
        if draft is not None:
            text, kb = await _render_draft_screen(bot, draft, user, back_screen)
            await _safe_edit_by_id(bot, int(panel_chat_id), int(panel_msg_id), text, kb)
            return

    if message.chat and message.chat.type == "private":
        await message.answer("باشه، عملیات تایمر لغو شد 👍 برای شروع دوباره <code>/timer</code> رو بزن.", parse_mode="HTML")


@router.message(Command("timer"))
async def cmd_timer(message: Message, bot: Bot, state: FSMContext) -> None:
    """Entry point for /timer (interactive panel OR quick shorthand '/timer 30m note')."""
    try:
        if not message.chat or not _is_allowed_timer_chat(message.chat):
            return
        user = message.from_user
        if user is None or user.is_bot:
            return

        is_private = message.chat.type == "private"

        # Access check in private chat: admins + active group members allowed; outsiders -> stranger flow
        if is_private:
            role, _ = await resolve_user_role(bot, user.id)
            if role == UserRole.STRANGER:
                is_member = await is_active_group_member(bot, user.id)
                if not is_member:
                    from bot.handlers.private import handle_stranger_contact

                    await handle_stranger_contact(message, bot, is_start=False)
                    return
            await repository.set_bot_user_dm_started(
                user.id,
                True,
                username=user.username,
                full_name=user.full_name,
            )
        else:
            # Per-user cooldown in group chats to prevent accidental command spam
            now_mono = time.monotonic()
            _prune_stale_drafts()
            cd_key = (message.chat.id, user.id)
            last_ts = _group_cmd_cooldown.get(cd_key)
            if last_ts is not None and (now_mono - last_ts) < GROUP_CMD_COOLDOWN_SECONDS:
                return
            _group_cmd_cooldown[cd_key] = now_mono
            if len(_group_cmd_cooldown) > 5000:
                cutoff_cd = now_mono - max(60.0, GROUP_CMD_COOLDOWN_SECONDS * 10)
                stale_cd = [k for k, ts in _group_cmd_cooldown.items() if ts < cutoff_cd]
                for k in stale_cd:
                    _group_cmd_cooldown.pop(k, None)

        await state.clear()

        # Extract optional shorthand arguments after /timer
        raw_text = (message.text or "").strip()
        parts = raw_text.split(maxsplit=1)
        args_text = parts[1].strip() if len(parts) > 1 else ""

        chat_id = message.chat.id
        chat_type = "private" if is_private else "supergroup"
        thread_id = 1 if is_private else resolve_thread_id(message)

        if not is_private:
            await _try_delete_msg(message)

        # Shorthand mode: /timer 30m برم شام | /timer فردا 09:00 جلسه | /timer 18:30 تماس
        if args_text:
            settings = get_settings()
            now_tehran = get_tehran_now()
            parsed = parse_quick_timer_command(args_text, now_tehran, settings.TIMEZONE)
            if parsed is not None:
                timer_type, fire_at_tehran, note = parsed
                timer_row, err = await create_and_schedule_timer(
                    bot,
                    chat_id=chat_id,
                    thread_id=thread_id,
                    chat_type=chat_type,
                    user_id=user.id,
                    username=user.username,
                    full_name=user.full_name or user.first_name or "کاربر",
                    note=note,
                    timer_type=timer_type,
                    fire_at=fire_at_tehran,
                )
                if err is not None or timer_row is None:
                    await _send_in_origin(message, err or "❌ نتونستم تایمر رو ثبت کنم.")
                    return

                card_text, card_kb = await render_saved_card(bot, timer_row, now_tehran=now_tehran)
                await _send_in_origin(message, card_text, card_kb)
                return

            # Unrecognized shorthand -> open interactive panel with a friendly tip at top
            draft = _new_draft(user.id, chat_id, thread_id, chat_type, now_tehran=now_tehran)
            home_text, home_kb = await render_timer_home(bot, draft, user, now_tehran=now_tehran)
            tip_prefix = (
                "⚠️ فرمت میانبر رو متوجه نشدم، ولی پنل شیشه‌ای برات باز شد! 👇\n"
                "<i>نمونه میانبر: <code>/timer 30m شام</code> یا <code>/timer فردا 09:00 جلسه</code></i>\n\n"
            )
            sent = await _send_in_origin(message, tip_prefix + home_text, home_kb)
            if sent is not None:
                _drafts[(chat_id, sent.message_id)] = draft
            return

        # Standard interactive /timer panel
        draft = _new_draft(user.id, chat_id, thread_id, chat_type)
        home_text, home_kb = await render_timer_home(bot, draft, user)
        sent = await _send_in_origin(message, home_text, home_kb)
        if sent is not None:
            _drafts[(chat_id, sent.message_id)] = draft
    except Exception as exc:
        logger.error("Error in /timer command: %s", exc, exc_info=True)


# ---------------- Callback Handler ----------------


@router.callback_query(TimerCB.filter())
async def handle_timer_callback(
    callback: CallbackQuery,
    callback_data: TimerCB,
    bot: Bot,
    state: FSMContext,
) -> None:
    """Handle all interactive /timer inline keyboard interactions with strict owner lock."""
    try:
        user = callback.from_user
        if user is None:
            await safe_answer_callback(callback)
            return

        action = callback_data.action
        owner_id = callback_data.owner_id
        arg = callback_data.arg

        # Strict owner lock: prevent other group members from hijacking someone else's panel or alarm
        if owner_id and user.id != owner_id:
            await safe_answer_callback(
                callback,
                "این پنل تایمر مال تو نیست 😅 خودت /timer رو بزن!",
                show_alert=True,
            )
            return

        if action == "noop":
            await safe_answer_callback(callback)
            return

        # Clear pending FSM input whenever navigating via buttons (except when starting a new input)
        if action not in ("note", "rcus", "tcus", "dcus"):
            current_state = await state.get_state()
            if current_state and str(current_state).startswith("TimerStates:"):
                await state.clear()

        # Close / delete panel message
        if action == "cls":
            if callback.message:
                _drafts.pop((callback.message.chat.id, callback.message.message_id), None)
                try:
                    await callback.message.delete()
                except Exception:
                    await safe_edit_message(callback, "⏱ <i>پنل تایمر بسته شد.</i>", None)
            await safe_answer_callback(callback)
            return

        # Post-alarm Snooze (reads from SQLite so it survives restarts!)
        if action == "snz":
            parts = arg.split("_", 1)
            if len(parts) != 2:
                await safe_answer_callback(callback, "دکمه نامعتبره 😅", show_alert=True)
                return
            try:
                orig_id = int(parts[0])
                snooze_mins = int(parts[1])
            except ValueError:
                await safe_answer_callback(callback, "دکمه نامعتبره 😅", show_alert=True)
                return
            if not (1 <= snooze_mins <= 720):
                await safe_answer_callback(callback, "مدت تمدید نامعتبره 😅", show_alert=True)
                return

            orig = await repository.get_timer(orig_id)
            if orig is None or int(orig["user_id"]) != user.id:
                await safe_answer_callback(callback, "این تایمر پیدا نشد 😅", show_alert=True)
                return

            now_tehran = get_tehran_now()
            new_fire = now_tehran + timedelta(minutes=snooze_mins)
            new_timer, err = await create_and_schedule_timer(
                bot,
                chat_id=int(orig["chat_id"]),
                thread_id=int(orig["thread_id"]),
                chat_type=str(orig.get("chat_type") or "supergroup"),
                user_id=user.id,
                username=user.username or orig.get("username"),
                full_name=user.full_name or orig.get("full_name") or "کاربر",
                note=orig.get("note"),
                timer_type="relative",
                fire_at=new_fire,
            )
            if err is not None or new_timer is None:
                await safe_answer_callback(callback, err or "خطا در تمدید تایمر", show_alert=True)
                return

            new_id = int(new_timer["id"])
            time_str = new_fire.strftime("%H:%M")
            orig_html = _extract_safe_html(callback.message)
            updated_text = (
                f"{orig_html}\n\n"
                f"🔁 <i>برای {format_duration_persian(snooze_mins)} دیگه (ساعت {time_str}) تمدید شد!</i>"
            )
            cancel_kb = build_saved_timer_kb(
                user.id,
                new_id,
                dm_started=True,
                bot_username=None,
                is_private=(orig.get("chat_type") == "private"),
            )
            await safe_edit_message(callback, updated_text, cancel_kb)
            await safe_answer_callback(
                callback,
                f"🔁 {format_duration_persian(snooze_mins)} تمدید شد!",
            )
            return

        # Post-alarm Dismiss / Acknowledge
        if action == "ack":
            orig_html = _extract_safe_html(callback.message)
            if "دیده شد" not in orig_html:
                orig_html = f"{orig_html}\n\n✅ <i>دیده شد!</i>"
            await safe_edit_message(callback, orig_html, None)
            await safe_answer_callback(callback, "موفق باشی! 🚀")
            return

        # Active Timers List
        if action == "list":
            try:
                page = int(arg) if arg else 0
            except ValueError:
                page = 0
            text, kb = await render_active_timers(user.id, page=page)
            await safe_edit_message(callback, text, kb)
            await safe_answer_callback(callback)
            return

        # Cancel single timer
        if action == "del":
            parts = arg.split("_", 1)
            try:
                timer_id = int(parts[0])
            except ValueError:
                await safe_answer_callback(callback, "شناسه تایمر نامعتبره 😅", show_alert=True)
                return
            source_ctx = parts[1] if len(parts) > 1 else "0"
            cancelled = await repository.cancel_timer(timer_id, user.id)
            if cancelled:
                unschedule_timer_job(timer_id)
                if source_ctx == "c":
                    # Cancelled directly from the saved confirmation card
                    await safe_edit_message(
                        callback,
                        "🗑 <b>این تایمر لغو شد.</b>\n<i>برای تنظیم تایمر جدید دوباره <code>/timer</code> رو بزن.</i>",
                        None,
                    )
                    await safe_answer_callback(callback, "🗑 تایمر لغو شد!")
                    return
                try:
                    page = int(source_ctx)
                except ValueError:
                    page = 0
                text, kb = await render_active_timers(user.id, page=page)
                await safe_edit_message(callback, text, kb)
                await safe_answer_callback(callback, "🗑 تایمر لغو شد!")
                return

            await safe_answer_callback(
                callback,
                "این تایمر قبلاً اجرا یا لغو شده 😅",
                show_alert=True,
            )
            if source_ctx != "c":
                text, kb = await render_active_timers(user.id, page=0)
                await safe_edit_message(callback, text, kb)
            return

        # Cancel all active timers (2-step)
        if action == "dela":
            if arg == "ask":
                cnt = await repository.count_active_timers_for_user(user.id)
                if cnt == 0:
                    await safe_answer_callback(callback, "تایمر فعالی نداری! 😌")
                    return
                text = (
                    "🧹 <b>لغو همه‌ی تایمرهای فعال</b>\n\n"
                    f"مطمئنی می‌خوای هر <b>{to_persian_digits(cnt)}</b> تایمر فعالت رو یکجا لغو کنی؟"
                )
                await safe_edit_message(callback, text, build_cancel_all_confirm_kb(user.id))
                await safe_answer_callback(callback)
                return
            if arg == "yes":
                ids = await repository.cancel_all_user_timers(user.id)
                for tid in ids:
                    unschedule_timer_job(tid)
                text, kb = await render_active_timers(user.id, page=0)
                await safe_edit_message(callback, text, kb)
                await safe_answer_callback(
                    callback,
                    f"🧹 {to_persian_digits(len(ids))} تایمر لغو شد!",
                )
                return

        # From here on, wizard actions operate on a TimerDraft
        if action == "save":
            draft = _get_or_create_draft(callback, user.id, create_if_missing=False)
            if draft is None:
                await safe_answer_callback(
                    callback,
                    "نشست این پنل منقضی شده 😅 دوباره /timer رو بزن!",
                    show_alert=True,
                )
                return
            settings = get_settings()
            now_tehran = get_tehran_now()
            if draft.mode == "relative":
                fire_at = now_tehran + timedelta(minutes=draft.duration_minutes)
            else:
                fire_at = jalali_to_tehran_dt(
                    draft.jyear,
                    draft.jmonth,
                    draft.jday,
                    draft.hour,
                    draft.minute,
                    settings.TIMEZONE,
                )
                if (fire_at - now_tehran).total_seconds() < 30:
                    await safe_answer_callback(
                        callback,
                        "⚠️ این ساعت گذشته! لطفاً یه ساعت آینده انتخاب کن.",
                        show_alert=True,
                    )
                    text, kb = await render_time_picker(draft, now_tehran=now_tehran)
                    await safe_edit_message(callback, text, kb)
                    return

            timer_row, err = await create_and_schedule_timer(
                bot,
                chat_id=draft.chat_id,
                thread_id=draft.thread_id,
                chat_type=draft.chat_type,
                user_id=user.id,
                username=user.username,
                full_name=user.full_name or user.first_name or "کاربر",
                note=draft.note,
                timer_type=draft.mode,
                fire_at=fire_at,
                source_message_id=callback.message.message_id if callback.message else None,
            )
            if err is not None or timer_row is None:
                await safe_answer_callback(callback, err or "خطا در ثبت تایمر 😅", show_alert=True)
                return

            if callback.message:
                _drafts.pop((callback.message.chat.id, callback.message.message_id), None)
            card_text, card_kb = await render_saved_card(bot, timer_row, now_tehran=now_tehran)
            await safe_edit_message(callback, card_text, card_kb)
            await safe_answer_callback(callback, "✅ تایمرت ثبت شد!")
            return

        draft = _get_or_create_draft(callback, user.id, create_if_missing=True)
        if draft is None:
            await safe_answer_callback(callback)
            return

        if action == "home":
            text, kb = await _render_draft_screen(bot, draft, user, "home")
            await safe_edit_message(callback, text, kb)
            await safe_answer_callback(callback)
            return

        if action == "q":
            try:
                mins = int(arg)
            except ValueError:
                mins = 30
            draft.mode = "relative"
            draft.duration_minutes = max(MIN_TIMER_MINUTES, min(MAX_TIMER_MINUTES, mins))
            text, kb = await _render_draft_screen(bot, draft, user, "q")
            await safe_edit_message(callback, text, kb)
            await safe_answer_callback(callback)
            return

        if action == "rel":
            draft.mode = "relative"
            text, kb = await _render_draft_screen(bot, draft, user, "rel")
            await safe_edit_message(callback, text, kb)
            await safe_answer_callback(callback)
            return

        if action == "rset":
            try:
                mins = int(arg)
            except ValueError:
                await safe_answer_callback(callback, "مقدار نامعتبره 😅", show_alert=True)
                return
            draft.mode = "relative"
            draft.duration_minutes = max(MIN_TIMER_MINUTES, min(MAX_TIMER_MINUTES, mins))
            text, kb = await _render_draft_screen(bot, draft, user, "rel")
            await safe_edit_message(callback, text, kb)
            await safe_answer_callback(callback, f"⏱ {format_duration_persian(draft.duration_minutes)}")
            return

        if action == "radd":
            try:
                delta = int(arg)
            except ValueError:
                await safe_answer_callback(callback, "مقدار نامعتبره 😅", show_alert=True)
                return
            draft.mode = "relative"
            draft.duration_minutes = max(
                MIN_TIMER_MINUTES,
                min(MAX_TIMER_MINUTES, draft.duration_minutes + delta),
            )
            text, kb = await _render_draft_screen(bot, draft, user, "rel")
            await safe_edit_message(callback, text, kb)
            await safe_answer_callback(callback, f"⏱ {format_duration_persian(draft.duration_minutes)}")
            return

        if action == "cal":
            draft.mode = "scheduled"
            if arg and "-" in arg:
                try:
                    y_s, m_s = arg.split("-", 1)
                    y_val, m_val = int(y_s), int(m_s)
                    if 1300 <= y_val <= 1500 and 1 <= m_val <= 12:
                        draft.cal_year, draft.cal_month = y_val, m_val
                except ValueError:
                    pass
            else:
                draft.cal_year, draft.cal_month = draft.jyear, draft.jmonth
            text, kb = await _render_draft_screen(bot, draft, user, "cal")
            await safe_edit_message(callback, text, kb)
            await safe_answer_callback(callback)
            return

        if action == "day":
            try:
                y_s, m_s, d_s = arg.split("-", 2)
                chosen = jdatetime.date(int(y_s), int(m_s), int(d_s))
            except Exception:
                await safe_answer_callback(callback, "تاریخ نامعتبره 😅", show_alert=True)
                return
            now_tehran = get_tehran_now()
            today_j = jdatetime.date.fromgregorian(date=now_tehran.date())
            if chosen < today_j:
                await safe_answer_callback(callback, "این تاریخ گذشته! 😅", show_alert=True)
                return
            draft.mode = "scheduled"
            draft.jyear, draft.jmonth, draft.jday = chosen.year, chosen.month, chosen.day
            draft.cal_year, draft.cal_month = chosen.year, chosen.month
            if not draft.time_customized:
                if chosen == today_j:
                    draft.hour, draft.minute = _default_today_time(now_tehran)
                else:
                    draft.hour, draft.minute = 9, 0
            text, kb = await _render_draft_screen(bot, draft, user, "tm")
            await safe_edit_message(callback, text, kb)
            await safe_answer_callback(callback, f"📆 {format_jalali_day_label(chosen)}")
            return

        if action == "tm":
            draft.mode = "scheduled"
            text, kb = await _render_draft_screen(bot, draft, user, "tm")
            await safe_edit_message(callback, text, kb)
            await safe_answer_callback(callback)
            return

        if action == "tset":
            parsed = parse_time_input(arg)
            if parsed is None:
                await safe_answer_callback(callback, "ساعت نامعتبره 😅", show_alert=True)
                return
            draft.mode = "scheduled"
            draft.hour, draft.minute = parsed
            draft.time_customized = True
            text, kb = await _render_draft_screen(bot, draft, user, "tm")
            await safe_edit_message(callback, text, kb)
            await safe_answer_callback(callback, f"⏰ ساعت {draft.hour:02d}:{draft.minute:02d}")
            return

        if action == "hadd":
            try:
                delta = int(arg)
            except ValueError:
                await safe_answer_callback(callback)
                return
            draft.mode = "scheduled"
            draft.hour = (draft.hour + delta) % 24
            draft.time_customized = True
            text, kb = await _render_draft_screen(bot, draft, user, "tm")
            await safe_edit_message(callback, text, kb)
            await safe_answer_callback(callback, f"⏰ ساعت {draft.hour:02d}:{draft.minute:02d}")
            return

        if action == "madd":
            try:
                delta = int(arg)
            except ValueError:
                await safe_answer_callback(callback)
                return
            draft.mode = "scheduled"
            total_mins = (draft.hour * 60 + draft.minute + delta) % (24 * 60)
            draft.hour, draft.minute = divmod(total_mins, 60)
            draft.time_customized = True
            text, kb = await _render_draft_screen(bot, draft, user, "tm")
            await safe_edit_message(callback, text, kb)
            await safe_answer_callback(callback, f"⏰ ساعت {draft.hour:02d}:{draft.minute:02d}")
            return

        if action == "nclr":
            draft.note = None
            target_screen = arg if arg in ("q", "rel", "tm") else draft.screen
            text, kb = await _render_draft_screen(bot, draft, user, target_screen)
            await safe_edit_message(callback, text, kb)
            await safe_answer_callback(callback, "🗑 یادداشت حذف شد")
            return

        if action == "back":
            target_screen = arg if arg in ("home", "q", "rel", "cal", "tm") else "home"
            text, kb = await _render_draft_screen(bot, draft, user, target_screen)
            await safe_edit_message(callback, text, kb)
            await safe_answer_callback(callback)
            return

        # FSM text-input prompts
        if action in ("note", "rcus", "tcus", "dcus"):
            if callback.message is None:
                await safe_answer_callback(callback)
                return
            fsm_data = {
                "panel_chat_id": callback.message.chat.id,
                "panel_message_id": callback.message.message_id,
                "thread_id": draft.thread_id,
                "expires_at": time.monotonic() + FSM_INPUT_TTL_SECONDS,
            }
            if action == "note":
                back_screen = arg if arg in ("q", "rel", "tm") else draft.screen
                fsm_data["back_screen"] = back_screen
                await state.set_state(TimerStates.waiting_note)
                await state.update_data(**fsm_data)
                prompt = (
                    "📝 <b>نوشتن یادداشت یادآور</b>\n\n"
                    f"متنی که می‌خوای موقع زنگ خوردن تایمر بهت بگم رو همینجا بفرست (حداکثر {to_persian_digits(MAX_NOTE_LENGTH)} کاراکتر).\n"
                    "<i>مثلاً: برم شام بخورم یا به علی زنگ بزنم</i>\n\n"
                    "⏳ مهلت ارسال: ۲ دقیقه | برای انصراف دکمه زیر یا <code>/cancel</code> رو بزن 👇"
                )
            elif action == "rcus":
                back_screen = "rel"
                fsm_data["back_screen"] = back_screen
                await state.set_state(TimerStates.waiting_custom_duration)
                await state.update_data(**fsm_data)
                prompt = (
                    "⌨️ <b>وارد کردن مدت زمان دلخواه</b>\n\n"
                    "مدت زمان رو همینجا بفرست (عدد فارسی یا انگلیسی):\n"
                    "• فقط عدد (دقیقه): <code>35</code> یا <code>۴۵</code>\n"
                    "• ساعت و دقیقه: <code>1:45</code> یا <code>1h30m</code> یا <code>۲س ۱۵د</code>\n"
                    "• روز: <code>2d</code> یا <code>۲ روز</code>\n\n"
                    "⏳ مهلت ارسال: ۲ دقیقه | برای انصراف دکمه زیر رو بزن 👇"
                )
            elif action == "tcus":
                back_screen = "tm"
                fsm_data["back_screen"] = back_screen
                await state.set_state(TimerStates.waiting_custom_time)
                await state.update_data(**fsm_data)
                prompt = (
                    "⌨️ <b>وارد کردن ساعت دقیق (به وقت تهران)</b>\n\n"
                    "ساعت رو به فرمت <code>HH:MM</code> بفرست (فارسی یا انگلیسی):\n"
                    "• نمونه: <code>14:30</code> ، <code>۰۹:۱۵</code> یا <code>21</code>\n\n"
                    "⏳ مهلت ارسال: ۲ دقیقه | برای انصراف دکمه زیر رو بزن 👇"
                )
            else:  # dcus
                back_screen = "cal"
                fsm_data["back_screen"] = back_screen
                await state.set_state(TimerStates.waiting_custom_datetime)
                await state.update_data(**fsm_data)
                prompt = (
                    "⌨️ <b>وارد کردن تاریخ شمسی و ساعت</b>\n\n"
                    "تاریخ و ساعت رو همینجا بفرست (فارسی یا انگلیسی):\n"
                    "• <code>1405/09/10 14:30</code>\n"
                    "• <code>09/15 09:00</code>\n"
                    "• <code>فردا 08:30</code> یا <code>پس‌فردا 18:00</code>\n\n"
                    "⏳ مهلت ارسال: ۲ دقیقه | برای انصراف دکمه زیر رو بزن 👇"
                )

            await safe_edit_message(callback, prompt, build_input_cancel_kb(user.id, back_screen))
            await safe_answer_callback(callback)
            return

    except Exception as exc:
        logger.error("Unhandled error in timer callback handler: %s", exc, exc_info=True)
        await safe_answer_callback(
            callback,
            "یه خطای کوچیک پیش اومد 😅 دوباره امتحان کن!",
            show_alert=True,
        )


# ---------------- FSM Text Input Handlers ----------------


async def _load_fsm_draft(
    state: FSMContext,
) -> tuple[TimerDraft | None, int | None, int | None, str]:
    data = await state.get_data()
    p_chat = data.get("panel_chat_id")
    p_msg = data.get("panel_message_id")
    back_screen = str(data.get("back_screen") or "home")
    if p_chat is None or p_msg is None:
        return None, None, None, back_screen
    draft = _drafts.get((int(p_chat), int(p_msg)))
    return draft, int(p_chat), int(p_msg), back_screen


@router.message(ActiveTimerFsmFilter(), StateFilter(TimerStates.waiting_note))
async def handle_timer_note_input(message: Message, bot: Bot, state: FSMContext) -> None:
    """Capture user's custom reminder note and update the timer panel in-place."""
    user = message.from_user
    if user is None:
        await state.clear()
        return

    draft, p_chat, p_msg, back_screen = await _load_fsm_draft(state)
    await state.clear()
    if message.chat and message.chat.type != "private":
        await _try_delete_msg(message)

    if draft is None or p_chat is None or p_msg is None:
        return

    draft.note = sanitize_note(message.text)
    draft.updated_at = time.monotonic()
    text, kb = await _render_draft_screen(bot, draft, user, back_screen)
    edited = await _safe_edit_by_id(bot, p_chat, p_msg, text, kb)
    if not edited:
        sent = await _send_in_origin(message, text, kb)
        if sent is not None:
            _drafts[(sent.chat.id, sent.message_id)] = draft


@router.message(ActiveTimerFsmFilter(), StateFilter(TimerStates.waiting_custom_duration))
async def handle_timer_custom_duration_input(message: Message, bot: Bot, state: FSMContext) -> None:
    """Capture user's custom relative duration text (e.g. '35', '1:45', '2س 15د')."""
    user = message.from_user
    if user is None:
        await state.clear()
        return

    draft, p_chat, p_msg, back_screen = await _load_fsm_draft(state)
    if message.chat and message.chat.type != "private":
        await _try_delete_msg(message)

    if draft is None or p_chat is None or p_msg is None:
        await state.clear()
        return

    mins = parse_duration_input(message.text)
    if mins is None:
        err_prompt = (
            "⚠️ <b>مدت زمان واردشده معتبر نیست!</b>\n\n"
            "یه نمونه درست بفرست (بین ۱ دقیقه تا ۳۶۵ روز):\n"
            "• <code>35</code> (دقیقه)  |  <code>1:30</code>  |  <code>2h</code>  |  <code>۱س ۴۵د</code>\n\n"
            "برای انصراف دکمه زیر یا <code>/cancel</code> رو بزن 👇"
        )
        await state.update_data(expires_at=time.monotonic() + FSM_INPUT_TTL_SECONDS)
        await _safe_edit_by_id(bot, p_chat, p_msg, err_prompt, build_input_cancel_kb(user.id, back_screen))
        return

    await state.clear()
    draft.mode = "relative"
    draft.duration_minutes = mins
    draft.updated_at = time.monotonic()
    text, kb = await _render_draft_screen(bot, draft, user, "rel")
    edited = await _safe_edit_by_id(bot, p_chat, p_msg, text, kb)
    if not edited:
        sent = await _send_in_origin(message, text, kb)
        if sent is not None:
            _drafts[(sent.chat.id, sent.message_id)] = draft


@router.message(ActiveTimerFsmFilter(), StateFilter(TimerStates.waiting_custom_time))
async def handle_timer_custom_time_input(message: Message, bot: Bot, state: FSMContext) -> None:
    """Capture user's custom wall-clock time text (e.g. '14:30', '۰۹:۱۵')."""
    user = message.from_user
    if user is None:
        await state.clear()
        return

    draft, p_chat, p_msg, back_screen = await _load_fsm_draft(state)
    if message.chat and message.chat.type != "private":
        await _try_delete_msg(message)

    if draft is None or p_chat is None or p_msg is None:
        await state.clear()
        return

    parsed = parse_time_input(message.text)
    if parsed is None:
        err_prompt = (
            "⚠️ <b>فرمت ساعت معتبر نیست!</b>\n\n"
            "ساعت رو به شکل <code>14:30</code> یا <code>۰۹:۱۵</code> یا <code>21</code> بفرست.\n"
            "برای انصراف دکمه زیر یا <code>/cancel</code> رو بزن 👇"
        )
        await state.update_data(expires_at=time.monotonic() + FSM_INPUT_TTL_SECONDS)
        await _safe_edit_by_id(bot, p_chat, p_msg, err_prompt, build_input_cancel_kb(user.id, back_screen))
        return

    await state.clear()
    draft.mode = "scheduled"
    draft.hour, draft.minute = parsed
    draft.time_customized = True
    draft.updated_at = time.monotonic()
    text, kb = await _render_draft_screen(bot, draft, user, "tm")
    edited = await _safe_edit_by_id(bot, p_chat, p_msg, text, kb)
    if not edited:
        sent = await _send_in_origin(message, text, kb)
        if sent is not None:
            _drafts[(sent.chat.id, sent.message_id)] = draft


@router.message(ActiveTimerFsmFilter(), StateFilter(TimerStates.waiting_custom_datetime))
async def handle_timer_custom_datetime_input(message: Message, bot: Bot, state: FSMContext) -> None:
    """Capture user's custom Jalali date + time input (e.g. '1405/09/10 14:30', 'فردا 09:00')."""
    user = message.from_user
    if user is None:
        await state.clear()
        return

    draft, p_chat, p_msg, back_screen = await _load_fsm_draft(state)
    if message.chat and message.chat.type != "private":
        await _try_delete_msg(message)

    if draft is None or p_chat is None or p_msg is None:
        await state.clear()
        return

    settings = get_settings()
    now_tehran = get_tehran_now()
    target_dt = parse_jalali_date_time_input(message.text, now_tehran, settings.TIMEZONE)
    if target_dt is None:
        err_prompt = (
            "⚠️ <b>تاریخ یا ساعت معتبر نیست (یا در گذشته است)!</b>\n\n"
            "نمونه‌های درست (تا حداکثر ۱ سال آینده):\n"
            "• <code>1405/09/10 14:30</code>\n"
            "• <code>09/15 09:00</code>\n"
            "• <code>فردا 08:30</code>\n\n"
            "برای انصراف دکمه زیر یا <code>/cancel</code> رو بزن 👇"
        )
        await state.update_data(expires_at=time.monotonic() + FSM_INPUT_TTL_SECONDS)
        await _safe_edit_by_id(bot, p_chat, p_msg, err_prompt, build_input_cancel_kb(user.id, back_screen))
        return

    await state.clear()
    target_j = jdatetime.date.fromgregorian(date=target_dt.date())
    draft.mode = "scheduled"
    draft.jyear, draft.jmonth, draft.jday = target_j.year, target_j.month, target_j.day
    draft.cal_year, draft.cal_month = target_j.year, target_j.month
    draft.hour, draft.minute = target_dt.hour, target_dt.minute
    draft.time_customized = True
    draft.updated_at = time.monotonic()
    text, kb = await _render_draft_screen(bot, draft, user, "tm")
    edited = await _safe_edit_by_id(bot, p_chat, p_msg, text, kb)
    if not edited:
        sent = await _send_in_origin(message, text, kb)
        if sent is not None:
            _drafts[(sent.chat.id, sent.message_id)] = draft

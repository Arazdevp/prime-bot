"""Interactive /deeptalk handlers: admin management, session wizard, group cards, check-ins, and live controls."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import time
from typing import Any, Literal
from zoneinfo import ZoneInfo
from aiogram import Bot, Router
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command, Filter, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message, User
import jdatetime
from bot.config import get_settings
from bot.database import repository
from bot.handlers.panel import safe_answer_callback, safe_edit_message
from bot.keyboards.deeptalk import (
    DEEPTALK_ADMINS_PAGE_SIZE,
    DEEPTALK_SESSIONS_PAGE_SIZE,
    DeepTalkCB,
    build_deeptalk_admins_kb,
    build_deeptalk_calendar_kb,
    build_deeptalk_cancel_input_kb,
    build_deeptalk_confirm_action_kb,
    build_deeptalk_groups_kb,
    build_deeptalk_hub_kb,
    build_deeptalk_session_manage_kb,
    build_deeptalk_sessions_list_kb,
    build_deeptalk_time_kb,
    build_deeptalk_topics_kb,
    build_deeptalk_wizard_kb,
)
from bot.keyboards.panel import paginate
from bot.keyboards.timer import build_dm_timer_entry_kb
from bot.services.access import (
    can_manage_deeptalk,
    get_deeptalk_groups,
    is_active_group_member,
    is_global_admin,
    is_user_in_specific_group,
)
from bot.services.deeptalk_service import (
    MAX_DEEPTALK_DURATION_MINUTES,
    MIN_DEEPTALK_DURATION_MINUTES,
    cancel_deeptalk_flow,
    create_and_publish_deeptalk,
    end_deeptalk_session,
    extend_live_deeptalk,
    refresh_announcement_card,
    refresh_start_card,
    render_pv_participants_view,
    send_deeptalk_topic_message,
    start_deeptalk_session,
)
from bot.services.timer_service import build_topic_deep_link, ensure_bot_username
from bot.texts.deeptalk_texts import format_checkin_announcement
from bot.utils.dates import format_persian_short_date, get_tehran_now
from bot.utils.general_topic import resolve_thread_id, to_send_thread_id
from bot.utils.html_utils import escape_html
from bot.utils.link_parser import parse_topic_link
from bot.utils.timer_parser import (
    MAX_TIMER_DAYS,
    clamp_jalali_date,
    format_duration_persian,
    format_fire_time_persian,
    format_jalali_day_label,
    format_remaining_persian,
    jalali_to_tehran_dt,
    normalize_digits,
    parse_duration_input,
    parse_jalali_date_time_input,
    parse_utc_iso,
    to_persian_digits,
)

logger = logging.getLogger(__name__)

router = Router(name="deeptalk_router")

DRAFT_TTL_SECONDS = 1800.0
FSM_INPUT_TTL_SECONDS = 180.0

_PUBLIC_CALLBACK_ACTIONS = {"reg", "unreg", "chk", "plist", "pref"}
_OWNER_ONLY_ACTIONS = {"adm", "adma", "admd"}


class DeepTalkStates(StatesGroup):
    """FSM states for DeepTalk wizard inputs and DeepTalk admin registration."""

    waiting_subject = State()
    waiting_description = State()
    waiting_custom_start = State()
    waiting_custom_duration = State()
    waiting_new_admin = State()


@dataclass
class DeepTalkDraft:
    """In-memory wizard state scoped to a single /deeptalk panel message."""

    owner_id: int
    panel_chat_id: int
    panel_thread_id: int
    panel_chat_type: str
    target_chat_id: int
    target_thread_id: int
    subject: str | None = None
    description: str | None = None
    start_mode: Literal["relative", "scheduled"] = "scheduled"
    relative_minutes: int = 30
    jyear: int = 1405
    jmonth: int = 1
    jday: int = 1
    cal_year: int = 1405
    cal_month: int = 1
    hour: int = 21
    minute: int = 0
    duration_minutes: int = 45
    screen: str = "home"
    updated_at: float = field(default_factory=time.monotonic)


# (panel_chat_id, panel_message_id) -> DeepTalkDraft
_dt_drafts: dict[tuple[int, int], DeepTalkDraft] = {}


def clear_deeptalk_drafts() -> None:
    """Clear in-memory DeepTalk drafts (useful for test isolation)."""
    _dt_drafts.clear()


def _prune_stale_drafts() -> None:
    now = time.monotonic()
    expired = [k for k, v in _dt_drafts.items() if (now - v.updated_at) > DRAFT_TTL_SECONDS]
    for k in expired:
        _dt_drafts.pop(k, None)


def _default_start_clock(now_tehran: datetime) -> tuple[jdatetime.date, int, int]:
    """Compute a convenient future start date and clock time in Tehran."""
    candidate = now_tehran + timedelta(minutes=60)
    rounded_min = (candidate.minute // 15) * 15
    candidate = candidate.replace(minute=rounded_min, second=0, microsecond=0)
    if candidate <= now_tehran + timedelta(minutes=5):
        candidate = now_tehran + timedelta(minutes=30)
        candidate = candidate.replace(second=0, microsecond=0)
    jdate = jdatetime.date.fromgregorian(date=candidate.date())
    return jdate, candidate.hour, candidate.minute


def _compute_draft_start_dt(draft: DeepTalkDraft, now_tehran: datetime, tz_name: str) -> datetime:
    if draft.start_mode == "relative":
        return (now_tehran + timedelta(minutes=draft.relative_minutes)).replace(second=0, microsecond=0)
    return jalali_to_tehran_dt(
        draft.jyear,
        draft.jmonth,
        draft.jday,
        draft.hour,
        draft.minute,
        tz_name,
    )


async def _resolve_default_target(
    *,
    is_private: bool,
    origin_chat_id: int,
    origin_thread_id: int,
) -> tuple[int, int]:
    """Resolve initial (target_chat_id, target_thread_id) for a new DeepTalk draft."""
    if not is_private:
        detected = await repository.detect_deepwork_topic_id(origin_chat_id)
        return origin_chat_id, (detected if detected is not None else origin_thread_id)

    groups = await get_deeptalk_groups()
    if not groups:
        return 0, 1
    first_cid = int(groups[0]["chat_id"])
    detected = await repository.detect_deepwork_topic_id(first_cid)
    return first_cid, (detected if detected is not None else 1)


async def _new_draft(
    owner_id: int,
    panel_chat_id: int,
    panel_thread_id: int,
    panel_chat_type: str,
    *,
    now_tehran: datetime | None = None,
) -> DeepTalkDraft:
    now = now_tehran or get_tehran_now()
    jdate, h, m = _default_start_clock(now)
    target_cid, target_tid = await _resolve_default_target(
        is_private=(panel_chat_type == "private"),
        origin_chat_id=panel_chat_id,
        origin_thread_id=panel_thread_id,
    )
    default_dur = 45
    if target_cid:
        cfg = await repository.get_deeptalk_settings(target_cid)
        if cfg and cfg.get("default_duration_minutes"):
            default_dur = int(cfg["default_duration_minutes"])

    return DeepTalkDraft(
        owner_id=owner_id,
        panel_chat_id=panel_chat_id,
        panel_thread_id=panel_thread_id,
        panel_chat_type=panel_chat_type,
        target_chat_id=target_cid,
        target_thread_id=target_tid,
        subject=None,
        description=None,
        start_mode="scheduled",
        relative_minutes=30,
        jyear=jdate.year,
        jmonth=jdate.month,
        jday=jdate.day,
        cal_year=jdate.year,
        cal_month=jdate.month,
        hour=h,
        minute=m,
        duration_minutes=default_dur,
        screen="home",
        updated_at=time.monotonic(),
    )


async def _get_or_create_draft(callback: CallbackQuery, owner_id: int) -> DeepTalkDraft | None:
    _prune_stale_drafts()
    msg = callback.message
    if msg is None or not getattr(msg, "chat", None):
        return None
    key = (msg.chat.id, msg.message_id)
    existing = _dt_drafts.get(key)
    if existing is not None:
        existing.updated_at = time.monotonic()
        return existing
    chat_type = "private" if msg.chat.type == "private" else "supergroup"
    thread_id = 1 if chat_type == "private" else resolve_thread_id(msg)
    draft = await _new_draft(owner_id, msg.chat.id, thread_id, chat_type)
    _dt_drafts[key] = draft
    return draft


def _is_allowed_chat(chat: Any) -> bool:
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


async def _try_delete(message: Message | None) -> None:
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
            logger.error("Failed sending private DeepTalk message: %s", exc)
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
        logger.error("Failed sending group DeepTalk message: %s", exc)
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
        logger.debug("Could not edit DeepTalk panel %s/%s: %s", chat_id, message_id, exc)
        return False
    except Exception as exc:
        logger.debug("Unexpected error editing DeepTalk panel %s/%s: %s", chat_id, message_id, exc)
        return False


async def _group_and_topic_labels(chat_id: int, thread_id: int) -> tuple[str, str]:
    if not chat_id:
        return "بدون گروه فعال", "نامشخص"
    g_rec = await repository.get_known_group(chat_id)
    g_title = (g_rec.get("title") if g_rec else None) or f"گروه {chat_id}"
    t_name = await repository.get_topic_name(chat_id, thread_id)
    if not t_name:
        t_name = "عمومی" if thread_id == 1 else f"تاپیک {thread_id}"
    return str(g_title), str(t_name)


def _get_today_start_utc_iso() -> str:
    now = get_tehran_now()
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    return start.astimezone(ZoneInfo("UTC")).isoformat()


# ---------------- Screen Renderers ----------------


async def render_deeptalk_hub(
    draft: DeepTalkDraft,
    user: User,
) -> tuple[str, InlineKeyboardMarkup]:
    """Render the main /deeptalk control hub."""
    groups = await get_deeptalk_groups()
    if draft.target_chat_id == 0 and groups:
        draft.target_chat_id = int(groups[0]["chat_id"])
        detected = await repository.detect_deepwork_topic_id(draft.target_chat_id)
        if detected is not None:
            draft.target_thread_id = detected

    is_owner = is_global_admin(user.id)
    role_badge = "👑 مالک ربات" if is_owner else "🎙 ادمین دیپ‌تاک"
    sessions = (
        await repository.list_active_or_scheduled_deeptalks(draft.target_chat_id)
        if draft.target_chat_id
        else await repository.list_active_or_scheduled_deeptalks()
    )
    g_title, t_name = await _group_and_topic_labels(draft.target_chat_id, draft.target_thread_id)

    live_cnt = sum(1 for s in sessions if s.get("status") == "live")
    sched_cnt = sum(1 for s in sessions if s.get("status") == "scheduled")

    if draft.target_chat_id == 0:
        loc_block = "⚠️ <i>هنوز گروه فعالی در دیتابیس ثبت نشده است.</i>"
    else:
        loc_block = (
            f"🏢 <b>گروه هدف:</b> {escape_html(g_title)}\n"
            f"📂 <b>تاپیک دیپ‌تاک:</b> <b>«{escape_html(t_name)}»</b> (<code>{draft.target_thread_id}</code>)"
        )

    text = (
        "🎙 <b>پنل مدیریت دیپ‌تاک (DeepTalk)</b>\n"
        f"👤 کاربر: <b>{escape_html(user.first_name or user.full_name or 'ادمین')}</b> ({role_badge})\n\n"
        f"{loc_block}\n"
        f"📡 <b>وضعیت جلسات:</b> {to_persian_digits(live_cnt)} در حال برگزاری | "
        f"{to_persian_digits(sched_cnt)} در انتظار شروع\n\n"
        "از منوی شیشه‌ای زیر می‌تونی دیپ‌تاک جدید بسازی یا جلسات فعال رو مدیریت کنی 👇"
    )
    kb = build_deeptalk_hub_kb(
        draft.owner_id,
        active_sessions_count=len(sessions),
        is_owner=is_owner,
        multi_group=(len(groups) > 1),
        is_private=(draft.panel_chat_type == "private"),
    )
    return text, kb


async def render_deeptalk_wizard(
    draft: DeepTalkDraft,
    *,
    now_tehran: datetime | None = None,
) -> tuple[str, InlineKeyboardMarkup]:
    """Render the interactive session creation & preview screen."""
    settings = get_settings()
    now = now_tehran or get_tehran_now()
    g_title, t_name = await _group_and_topic_labels(draft.target_chat_id, draft.target_thread_id)

    start_dt = _compute_draft_start_dt(draft, now, settings.TIMEZONE)
    end_dt = start_dt + timedelta(minutes=draft.duration_minutes)
    delta = start_dt - now
    valid_future = delta.total_seconds() >= 20 and delta <= timedelta(days=MAX_TIMER_DAYS)

    subj_line = (
        f"<b>{escape_html(draft.subject)}</b>"
        if draft.subject
        else "⚠️ <i>هنوز وارد نشده (روی دکمه تعیین محوریت بزن)</i>"
    )
    desc_line = (
        f"<blockquote>{escape_html(draft.description)}</blockquote>"
        if draft.description
        else "<i>ندارد (اختیاری)</i>"
    )

    if valid_future:
        time_status = (
            f"<b>{to_persian_digits(format_fire_time_persian(start_dt, now, settings.TIMEZONE))}</b> "
            f"<i>({format_remaining_persian(delta)})</i>"
        )
    else:
        time_status = (
            f"⚠️ <b>{start_dt.strftime('%H:%M')} — این زمان گذشته! ساعت آینده انتخاب کن</b>"
        )

    can_publish = bool(draft.target_chat_id and draft.subject and valid_future)

    text = (
        "✨ <b>ساخت دیپ‌تاک جدید</b>\n\n"
        f"🏢 <b>گروه:</b> {escape_html(g_title)}\n"
        f"📂 <b>تاپیک مقصد:</b> <b>«{escape_html(t_name)}»</b>\n\n"
        f"🎯 <b>محوریت بحث:</b> {subj_line}\n"
        f"💡 <b>سوال / توضیحات:</b> {desc_line}\n"
        f"📅 <b>زمان شروع:</b> {time_status}\n"
        f"⏳ <b>مدت گفتگو:</b> <b>{format_duration_persian(draft.duration_minutes)}</b> "
        f"(پایان: ساعت {to_persian_digits(end_dt.strftime('%H:%M'))})\n\n"
        "تنظیمات رو کامل کن و در پایان روی <b>«✅ تأیید و ارسال فراخوان»</b> بزن 👇"
    )
    kb = build_deeptalk_wizard_kb(
        draft.owner_id,
        has_subject=bool(draft.subject),
        has_description=bool(draft.description),
        duration_minutes=draft.duration_minutes,
        can_publish=can_publish,
    )
    return text, kb


async def render_deeptalk_calendar(
    draft: DeepTalkDraft,
    *,
    now_tehran: datetime | None = None,
) -> tuple[str, InlineKeyboardMarkup]:
    now = now_tehran or get_tehran_now()
    today_j = jdatetime.date.fromgregorian(date=now.date())
    selected_j = clamp_jalali_date(draft.jyear, draft.jmonth, draft.jday)
    text = (
        "📅 <b>انتخاب تاریخ شروع دیپ‌تاک</b>\n\n"
        f"📆 <b>تاریخ انتخابی فعلی:</b> {format_jalali_day_label(selected_j)}\n"
        "یه روز از تقویم شمسی یا میانبرهای بالا انتخاب کن 👇"
    )
    kb = build_deeptalk_calendar_kb(
        draft.owner_id,
        cal_year=draft.cal_year,
        cal_month=draft.cal_month,
        selected_jdate=selected_j,
        today_jdate=today_j,
    )
    return text, kb


async def render_deeptalk_clock(
    draft: DeepTalkDraft,
    *,
    now_tehran: datetime | None = None,
) -> tuple[str, InlineKeyboardMarkup]:
    settings = get_settings()
    now = now_tehran or get_tehran_now()
    selected_j = clamp_jalali_date(draft.jyear, draft.jmonth, draft.jday)
    target = jalali_to_tehran_dt(
        selected_j.year,
        selected_j.month,
        selected_j.day,
        draft.hour,
        draft.minute,
        settings.TIMEZONE,
    )
    delta = target - now
    status_line = (
        f"⏳ <i>حدود {format_remaining_persian(delta)}</i>"
        if delta.total_seconds() >= 20
        else "⚠️ <b>این ساعت گذشته! ساعت جلوتر انتخاب کن.</b>"
    )
    text = (
        "🕐 <b>انتخاب ساعت شروع دیپ‌تاک</b>\n\n"
        f"📆 <b>تاریخ:</b> {format_jalali_day_label(selected_j)} ({format_persian_short_date(target)})\n"
        f"⏰ <b>ساعت انتخابی:</b> <b>{to_persian_digits(f'{draft.hour:02d}:{draft.minute:02d}')}</b> (تهران)\n"
        f"{status_line}"
    )
    kb = build_deeptalk_time_kb(draft.owner_id, hour=draft.hour, minute=draft.minute)
    return text, kb


async def render_deeptalk_sessions_list(
    draft: DeepTalkDraft,
    *,
    page: int = 0,
) -> tuple[str, InlineKeyboardMarkup]:
    settings = get_settings()
    tz = ZoneInfo(settings.TIMEZONE)
    now = get_tehran_now()
    sessions = (
        await repository.list_active_or_scheduled_deeptalks(draft.target_chat_id)
        if draft.target_chat_id
        else await repository.list_active_or_scheduled_deeptalks()
    )

    if not sessions:
        text = (
            "📋 <b>جلسات دیپ‌تاک فعال و پیش‌رو</b>\n\n"
            "در حال حاضر هیچ جلسه فعال یا زمان‌بندی‌شده‌ای وجود نداره 😌\n"
            "برای ساخت جلسه جدید بزن روی «➕ دیپ‌تاک جدید» 👇"
        )
        return text, build_deeptalk_sessions_list_kb(draft.owner_id, [], page=0, tz_info=tz)

    page_items, safe_page, _ = paginate(sessions, page, DEEPTALK_SESSIONS_PAGE_SIZE)
    lines = [f"📋 <b>جلسات فعال و پیش‌رو ({to_persian_digits(len(sessions))} مورد):</b>\n"]

    for idx, s in enumerate(page_items, start=safe_page * DEEPTALK_SESSIONS_PAGE_SIZE + 1):
        sid = int(s["id"])
        stats = await repository.get_deeptalk_session_stats(sid)
        st_local = parse_utc_iso(str(s["start_at"])).astimezone(tz)
        end_local = parse_utc_iso(str(s["end_at"])).astimezone(tz)
        if str(s["status"]) == "live":
            badge = f"🟢 در حال برگزاری (تا {to_persian_digits(end_local.strftime('%H:%M'))})"
            cnt_note = f"حاضرین: {to_persian_digits(stats['checked_in_count'])}"
        else:
            badge = f"🟡 شروع: {to_persian_digits(format_fire_time_persian(st_local, now, settings.TIMEZONE))}"
            cnt_note = f"ثبت‌نام: {to_persian_digits(stats['registered_count'])}"
        lines.append(
            f"<b>{to_persian_digits(idx)}.</b> 🎯 <b>{escape_html(str(s['subject']))}</b>\n"
            f"   {badge} — <i>{cnt_note}</i>"
        )

    lines.append("\nروی هر جلسه بزن تا دکمه‌های کنترل زنده (تمدید، شروع فوری، پایان یا لغو) باز بشن 👇")
    kb = build_deeptalk_sessions_list_kb(draft.owner_id, sessions, page=safe_page, tz_info=tz)
    return "\n".join(lines), kb


async def render_deeptalk_session_detail(
    bot: Bot,
    owner_id: int,
    session_id: int,
) -> tuple[str, InlineKeyboardMarkup] | None:
    session = await repository.get_deeptalk_session(session_id)
    if session is None:
        return None

    settings = get_settings()
    tz = ZoneInfo(settings.TIMEZONE)
    now = get_tehran_now()
    stats = await repository.get_deeptalk_session_stats(session_id)
    g_title, t_name = await _group_and_topic_labels(int(session["chat_id"]), int(session["thread_id"]))

    st_local = parse_utc_iso(str(session["start_at"])).astimezone(tz)
    end_local = parse_utc_iso(str(session["end_at"])).astimezone(tz)
    status = str(session["status"])

    status_labels = {
        "scheduled": "🟡 زمان‌بندی‌شده (در انتظار شروع)",
        "live": "🟢 در حال برگزاری (زنده)",
        "completed": "🏁 پایان یافته",
        "cancelled": "🔴 لغو شده",
    }

    lines = [
        "🎛 <b>کنترل جلسه دیپ‌تاک</b>",
        "",
        f"🎯 <b>محوریت:</b> <b>{escape_html(str(session['subject']))}</b>",
        f"📍 <b>مکان:</b> {escape_html(g_title)} / «{escape_html(t_name)}»",
        f"📡 <b>وضعیت:</b> {status_labels.get(status, status)}",
        f"🕒 <b>شروع:</b> {to_persian_digits(format_fire_time_persian(st_local, now, settings.TIMEZONE))}",
        f"⏰ <b>پایان:</b> ساعت {to_persian_digits(end_local.strftime('%H:%M'))} "
        f"({format_duration_persian(int(session['duration_minutes']))})",
        f"👥 <b>ثبت‌نام اولیه:</b> {to_persian_digits(stats['registered_count'])} نفر  |  "
        f"✋ <b>حاضرین:</b> {to_persian_digits(stats['checked_in_count'])} نفر  |  "
        f"💬 <b>پیام‌ها:</b> {to_persian_digits(stats['total_messages'])}",
    ]
    bot_username = await ensure_bot_username(bot)
    kb = build_deeptalk_session_manage_kb(
        owner_id,
        session_id,
        status=status,
        bot_username=bot_username,
    )
    return "\n".join(lines), kb


async def render_deeptalk_admins_screen(
    owner_id: int,
    *,
    page: int = 0,
) -> tuple[str, InlineKeyboardMarkup]:
    """Render owner-only DeepTalk admins management screen."""
    settings = get_settings()
    admins = await repository.list_deeptalk_admins()
    page_items, safe_page, _ = paginate(admins, page, DEEPTALK_ADMINS_PAGE_SIZE)

    owner_ids_str = " ، ".join(f"<code>{oid}</code>" for oid in settings.ADMIN_IDS) or "<i>تنظیم نشده</i>"
    lines = [
        "👑 <b>مدیریت ادمین‌های دیپ‌تاک</b>",
        "",
        f"🔐 <b>مالکین اصلی ربات:</b> {owner_ids_str}",
        "<i>فقط مالکین ربات و افراد معرفی‌شده در لیست زیر به پنل <code>/deeptalk</code> دسترسی دارند.</i>",
        "",
    ]

    if not admins:
        lines.append("فعلاً هیچ ادمین دیپ‌تاکی معرفی نشده است.")
    else:
        lines.append(f"🎙 <b>ادمین‌های دیپ‌تاک ({to_persian_digits(len(admins))} نفر):</b>")
        for idx, a in enumerate(page_items, start=safe_page * DEEPTALK_ADMINS_PAGE_SIZE + 1):
            uid = int(a["user_id"])
            name = escape_html(str(a.get("full_name") or uid))
            uname = f" (@{escape_html(a['username'])})" if a.get("username") else ""
            lines.append(f"{to_persian_digits(idx)}. <b>{name}</b>{uname} — <code>{uid}</code>")

    lines.append("\nبرای افزودن ادمین جدید یا لغو دسترسی روی دکمه‌های زیر بزن 👇")
    return "\n".join(lines), build_deeptalk_admins_kb(owner_id, admins, page=safe_page)


async def _render_screen_by_key(
    draft: DeepTalkDraft,
    user: User,
    screen: str,
) -> tuple[str, InlineKeyboardMarkup]:
    draft.screen = screen
    if screen == "new":
        return await render_deeptalk_wizard(draft)
    if screen == "cal":
        return await render_deeptalk_calendar(draft)
    if screen == "tm":
        return await render_deeptalk_clock(draft)
    if screen == "adm" and is_global_admin(user.id):
        return await render_deeptalk_admins_screen(draft.owner_id)
    if screen == "list":
        return await render_deeptalk_sessions_list(draft)
    return await render_deeptalk_hub(draft, user)


async def open_deeptalk_hub_from_panel(
    callback: CallbackQuery,
    bot: Bot,
    state: FSMContext,
) -> None:
    """Entry helper allowing the main /panel Home screen to jump straight into the DeepTalk Hub."""
    user = callback.from_user
    if user is None or callback.message is None:
        await safe_answer_callback(callback)
        return
    if not await can_manage_deeptalk(user.id):
        await safe_answer_callback(callback, "فقط مالک ربات و ادمین‌های دیپ‌تاک دسترسی دارن 🔒", show_alert=True)
        return

    await state.clear()
    chat_type = "private" if callback.message.chat.type == "private" else "supergroup"
    thread_id = 1 if chat_type == "private" else resolve_thread_id(callback.message)
    draft = await _new_draft(user.id, callback.message.chat.id, thread_id, chat_type)
    _dt_drafts[(callback.message.chat.id, callback.message.message_id)] = draft
    text, kb = await render_deeptalk_hub(draft, user)
    await safe_edit_message(callback, text, kb)
    await safe_answer_callback(callback)


# ---------------- Group-Safe FSM Filter ----------------


class ActiveDeepTalkFsmFilter(Filter):
    """Match only non-command messages belonging to a live DeepTalkStates session in the same topic."""

    async def __call__(self, message: Message, state: FSMContext) -> bool:
        current = await state.get_state()
        if not current or not str(current).startswith("DeepTalkStates:"):
            return False

        data = await state.get_data()
        expires_at = float(data.get("expires_at") or 0.0)
        if expires_at and time.monotonic() > expires_at:
            await state.clear()
            return False

        raw_text = (message.text or "").strip()
        if raw_text.startswith("/"):
            return False

        # Allow forwarded messages / replies when adding a new DeepTalk admin
        is_admin_state = str(current) == DeepTalkStates.waiting_new_admin.state
        has_forward = bool(getattr(message, "forward_origin", None) or getattr(message, "forward_from", None))
        has_reply_user = bool(
            is_admin_state
            and getattr(message, "reply_to_message", None)
            and getattr(message.reply_to_message, "from_user", None)
            and not getattr(message.reply_to_message, "forum_topic_created", None)
        )
        if not raw_text and not (is_admin_state and (has_forward or has_reply_user)):
            return False

        if message.chat and message.chat.type != "private":
            expected_tid = data.get("panel_thread_id")
            if expected_tid is not None and resolve_thread_id(message) != int(expected_tid):
                return False

        return True


# ---------------- Command Handlers ----------------


@router.message(Command("cancel"), StateFilter(DeepTalkStates))
async def cmd_deeptalk_cancel_fsm(message: Message, bot: Bot, state: FSMContext) -> None:
    """Cancel an active DeepTalk FSM prompt and restore the panel screen."""
    user = message.from_user
    data = await state.get_data()
    await state.clear()
    if user is None:
        return

    if message.chat and message.chat.type != "private":
        await _try_delete(message)

    p_cid = data.get("panel_chat_id")
    p_mid = data.get("panel_message_id")
    back_screen = str(data.get("back_screen") or "home")
    if p_cid is not None and p_mid is not None:
        draft = _dt_drafts.get((int(p_cid), int(p_mid)))
        if draft is not None:
            text, kb = await _render_screen_by_key(draft, user, back_screen)
            await _safe_edit_by_id(bot, int(p_cid), int(p_mid), text, kb)
            return

    if message.chat and message.chat.type == "private":
        await message.answer("باشه، عملیات لغو شد 👍 برای باز کردن پنل دیپ‌تاک <code>/deeptalk</code> رو بزن.", parse_mode="HTML")


@router.message(Command("set_deeptalk_topic"))
async def cmd_set_deeptalk_topic(message: Message) -> None:
    """Set current forum topic (or passed topic link/ID) as default DeepTalk destination."""
    if not message.chat or message.chat.type != "supergroup" or not _is_allowed_chat(message.chat):
        return
    user = message.from_user
    if user is None or user.is_bot:
        return

    await _try_delete(message)
    if not await can_manage_deeptalk(user.id):
        return

    chat_id = message.chat.id
    args = (message.text or "").split()[1:]
    if args:
        _, tid, err = parse_topic_link(args[0].strip(), current_chat_id=chat_id)
        if err or tid is None:
            await _send_in_origin(message, f"⚠️ {escape_html(err or 'شناسه تاپیک معتبر نیست.')}")
            return
        thread_id = tid
    else:
        thread_id = resolve_thread_id(message)

    await repository.upsert_known_group(
        chat_id=chat_id,
        title=getattr(message.chat, "title", None),
        is_forum=bool(getattr(message.chat, "is_forum", True)),
    )
    await repository.set_deeptalk_topic(chat_id, thread_id)
    _, t_name = await _group_and_topic_labels(chat_id, thread_id)
    await _send_in_origin(
        message,
        f"🎙 تاپیک پیش‌فرض دیپ‌تاک روی <b>«{escape_html(t_name)}»</b> (<code>{thread_id}</code>) تنظیم شد ✅",
    )


@router.message(Command("deeptalk"))
async def cmd_deeptalk(message: Message, bot: Bot, state: FSMContext) -> None:
    """Open the interactive /deeptalk panel (accessible ONLY to bot owners and DeepTalk admins)."""
    try:
        if not message.chat or not _is_allowed_chat(message.chat):
            return
        user = message.from_user
        if user is None or user.is_bot:
            return

        is_private = message.chat.type == "private"
        authorized = await can_manage_deeptalk(user.id)

        if not authorized:
            if not is_private:
                # Quietly delete unauthorized /deeptalk commands in group topics to prevent clutter
                await _try_delete(message)
                return

            if await is_active_group_member(bot, user.id):
                await message.answer(
                    "🔒 <b>دسترسی محدود</b>\n\n"
                    "پنل <code>/deeptalk</code> فقط مخصوص مالک ربات و ادمین‌های معرفی‌شده دیپ‌تاک هست!\n"
                    "اما می‌تونی از دستور <code>/timer</code> برای تایمر و یادآور شخصی استفاده کنی ⏱",
                    reply_markup=build_dm_timer_entry_kb(user.id),
                    parse_mode="HTML",
                )
                return

            from bot.handlers.private import handle_stranger_contact

            await handle_stranger_contact(message, bot, is_start=False)
            return

        await state.clear()
        _prune_stale_drafts()

        if is_private:
            await repository.set_bot_user_dm_started(
                user.id,
                True,
                username=user.username,
                full_name=user.full_name,
            )
        else:
            await repository.upsert_known_group(
                chat_id=message.chat.id,
                title=getattr(message.chat, "title", None),
                is_forum=bool(getattr(message.chat, "is_forum", True)),
            )
            await _try_delete(message)

        chat_type = "private" if is_private else "supergroup"
        thread_id = 1 if is_private else resolve_thread_id(message)
        draft = await _new_draft(user.id, message.chat.id, thread_id, chat_type)

        # Optional convenience: '/deeptalk محوریت بحث' jumps straight to wizard with subject pre-filled
        parts = (message.text or "").strip().split(maxsplit=1)
        if len(parts) > 1 and parts[1].strip():
            draft.subject = parts[1].strip()[:200]
            draft.screen = "new"
            text, kb = await render_deeptalk_wizard(draft)
        else:
            text, kb = await render_deeptalk_hub(draft, user)

        sent = await _send_in_origin(message, text, kb)
        if sent is not None:
            _dt_drafts[(message.chat.id, sent.message_id)] = draft
    except Exception as exc:
        logger.error("Error in /deeptalk command: %s", exc, exc_info=True)


# ---------------- Callback Query Handler ----------------


async def _start_fsm_prompt(
    callback: CallbackQuery,
    state: FSMContext,
    draft: DeepTalkDraft,
    *,
    target_state: State,
    back_screen: str,
    prompt_html: str,
) -> None:
    if callback.message is None:
        return
    await state.set_state(target_state)
    await state.update_data(
        panel_chat_id=callback.message.chat.id,
        panel_message_id=callback.message.message_id,
        panel_thread_id=draft.panel_thread_id,
        back_screen=back_screen,
        expires_at=time.monotonic() + FSM_INPUT_TTL_SECONDS,
    )
    await safe_edit_message(
        callback,
        prompt_html,
        build_deeptalk_cancel_input_kb(draft.owner_id, back_screen=back_screen),
    )
    await safe_answer_callback(callback)


@router.callback_query(DeepTalkCB.filter())
async def handle_deeptalk_callback(
    callback: CallbackQuery,
    callback_data: DeepTalkCB,
    bot: Bot,
    state: FSMContext,
) -> None:
    """Dispatch both public topic DeepTalk buttons and owner-locked admin panel actions."""
    try:
        user = callback.from_user
        if user is None:
            await safe_answer_callback(callback)
            return

        action = callback_data.action
        sid = int(callback_data.sid)
        arg = callback_data.arg

        if action == "noop":
            await safe_answer_callback(callback)
            return

        # ==================== 1. Public Member Actions (sid = session_id) ====================
        if action in _PUBLIC_CALLBACK_ACTIONS:
            session = await repository.get_deeptalk_session(sid)
            if session is None:
                await safe_answer_callback(callback, "این جلسه دیپ‌تاک پیدا نشد 😅", show_alert=True)
                return

            display_name = (user.full_name or user.first_name or "کاربر").strip()

            if action == "reg":
                status = str(session.get("status"))
                if status == "live":
                    await safe_answer_callback(
                        callback,
                        "🔥 دیپ‌تاک همین الان شروع شده! روی پیام شروع بزن «✋ اعلام حضور»!",
                        show_alert=True,
                    )
                    return
                if status != "scheduled":
                    await safe_answer_callback(callback, "مهلت ثبت‌نام این دیپ‌تاک تموم شده 😅", show_alert=True)
                    return

                newly_reg, total_reg = await repository.register_deeptalk_participant(
                    session_id=sid,
                    user_id=user.id,
                    username=user.username,
                    full_name=display_name,
                )
                if not newly_reg:
                    await safe_answer_callback(
                        callback,
                        "قبلاً اسمت توی لیست ثبت شده رفیق! 😉 سر ساعت می‌بینیمت 🔥",
                    )
                    return

                await refresh_announcement_card(bot, sid, registered_count=total_reg)
                dm_started = await repository.is_bot_user_dm_started(user.id)
                hint = (
                    "✅ اسمت ثبت شد! موقع شروع توی پیوی هم بهت خبر میدم 🔥"
                    if dm_started
                    else "✅ اسمت ثبت شد! برای یادآوری در پیوی، دکمه «مشاهده افراد در پیوی» رو هم بزن 🔔"
                )
                await safe_answer_callback(callback, hint, show_alert=not dm_started)
                return

            if action == "unreg":
                if str(session.get("status")) != "scheduled":
                    await safe_answer_callback(callback, "دیپ‌تاک شروع یا تمام شده!", show_alert=True)
                    return
                was_reg, total_reg = await repository.unregister_deeptalk_participant(
                    session_id=sid,
                    user_id=user.id,
                )
                if not was_reg:
                    await safe_answer_callback(callback, "هنوز ثبت‌نام نکرده بودی 😅")
                    return
                await refresh_announcement_card(bot, sid, registered_count=total_reg)
                await safe_answer_callback(callback, "انصرافت ثبت شد 👍 هر وقت نظرت عوض شد دوباره بزن!")
                return

            if action == "chk":
                res = await repository.checkin_deeptalk_participant_and_draw(
                    session_id=sid,
                    user_id=user.id,
                    username=user.username,
                    full_name=display_name,
                )
                if res is None:
                    await safe_answer_callback(
                        callback,
                        "⏰ مهلت اعلام حضور این دیپ‌تاک به پایان رسیده!",
                        show_alert=True,
                    )
                    return

                is_first, chosen_idx, total_chk = res
                if not is_first:
                    await safe_answer_callback(
                        callback,
                        "قبلاً اعلام حضور کردی قهرمان! الان نوبت ترکوندن توی بحثه 😎🔥",
                    )
                    return

                checkin_text = format_checkin_announcement(
                    chosen_idx,
                    user_id=user.id,
                    username=user.username,
                    full_name=display_name,
                )
                await send_deeptalk_topic_message(
                    bot,
                    chat_id=int(session["chat_id"]),
                    thread_id=int(session["thread_id"]),
                    text=checkin_text,
                )
                await refresh_start_card(bot, sid, checked_in_count=total_chk)
                await safe_answer_callback(callback, "🔥 اعلام حضورت ثبت شد! بپر وسط بحث!")
                return

            if action in ("pref", "plist"):
                chat_id = int(session["chat_id"])
                if not await is_user_in_specific_group(bot, chat_id, user.id):
                    await safe_answer_callback(callback, "فقط اعضای گروه می‌تونن لیست رو ببینن 🔒", show_alert=True)
                    return

                rendered = await render_pv_participants_view(sid)
                if rendered is None:
                    await safe_answer_callback(callback, "جلسه پیدا نشد 😅", show_alert=True)
                    return
                pv_text, pv_kb = rendered

                if action == "pref" and callback.message and callback.message.chat.type == "private":
                    await safe_edit_message(callback, pv_text, pv_kb)
                    await safe_answer_callback(callback, "لیست بروزرسانی شد ✅")
                    return

                # Fallback 'plist' when bot_username was not available
                try:
                    await bot.send_message(
                        chat_id=user.id,
                        text=pv_text,
                        reply_markup=pv_kb,
                        parse_mode="HTML",
                        disable_web_page_preview=True,
                    )
                    await repository.set_bot_user_dm_started(
                        user.id,
                        True,
                        username=user.username,
                        full_name=user.full_name,
                    )
                    await safe_answer_callback(callback, "📬 لیست شرکت‌کنندگان به پیوی ربات فرستاده شد!")
                except Exception:
                    await safe_answer_callback(
                        callback,
                        "⚠️ ابتدا ربات را در پیوی /start کن تا بتونم لیست رو بفرستم!",
                        show_alert=True,
                    )
                return

        # ==================== 2. Owner-Locked Panel Actions (sid = owner_id) ====================
        if sid and user.id != sid:
            await safe_answer_callback(
                callback,
                "این پنل دیپ‌تاک مال تو نیست 😅 خودت /deeptalk رو بزن!",
                show_alert=True,
            )
            return

        if action in _OWNER_ONLY_ACTIONS:
            if not is_global_admin(user.id):
                await safe_answer_callback(
                    callback,
                    "👑 فقط مالک اصلی ربات می‌تونه ادمین‌های دیپ‌تاک رو تغییر بده!",
                    show_alert=True,
                )
                return
        else:
            if not await can_manage_deeptalk(user.id):
                await safe_answer_callback(
                    callback,
                    "🔒 دسترسی شما به پنل دیپ‌تاک لغو شده یا فعال نیست.",
                    show_alert=True,
                )
                return

        if action not in ("subj", "desc", "scus", "adma") and not (action == "dur" and arg == "cus"):
            cur_st = await state.get_state()
            if cur_st and str(cur_st).startswith("DeepTalkStates:"):
                await state.clear()

        if action == "cls":
            if callback.message:
                _dt_drafts.pop((callback.message.chat.id, callback.message.message_id), None)
                await _try_delete(callback.message)
            await safe_answer_callback(callback)
            return

        draft = await _get_or_create_draft(callback, user.id)
        if draft is None:
            await safe_answer_callback(callback)
            return

        if action == "back":
            text, kb = await _render_screen_by_key(draft, user, arg or "new")
            await safe_edit_message(callback, text, kb)
            await safe_answer_callback(callback)
            return

        if action == "home":
            text, kb = await render_deeptalk_hub(draft, user)
            await safe_edit_message(callback, text, kb)
            await safe_answer_callback(callback)
            return

        if action == "new":
            if draft.target_chat_id == 0:
                await safe_answer_callback(callback, "⚠️ هنوز هیچ گروه فعالی برای ربات ثبت نشده!", show_alert=True)
                return
            draft.screen = "new"
            text, kb = await render_deeptalk_wizard(draft)
            await safe_edit_message(callback, text, kb)
            await safe_answer_callback(callback)
            return

        if action == "grp":
            groups = await get_deeptalk_groups()
            if arg.startswith("p"):
                page = int(arg[1:]) if arg[1:].isdigit() else 0
                g_text = "🏢 <b>انتخاب گروه برای برگزاری دیپ‌تاک:</b>"
                g_kb = build_deeptalk_groups_kb(
                    draft.owner_id,
                    groups,
                    selected_chat_id=draft.target_chat_id,
                    page=page,
                )
                await safe_edit_message(callback, g_text, g_kb)
                await safe_answer_callback(callback)
                return

            try:
                chosen_cid = int(arg)
            except ValueError:
                await safe_answer_callback(callback)
                return
            if any(int(g["chat_id"]) == chosen_cid for g in groups):
                draft.target_chat_id = chosen_cid
                detected = await repository.detect_deepwork_topic_id(chosen_cid)
                draft.target_thread_id = detected if detected is not None else 1
            text, kb = await render_deeptalk_hub(draft, user)
            await safe_edit_message(callback, text, kb)
            await safe_answer_callback(callback, "گروه هدف تغییر کرد ✅")
            return

        if action == "top":
            if draft.target_chat_id == 0:
                await safe_answer_callback(callback, "⚠️ ابتدا گروه را انتخاب کن!", show_alert=True)
                return
            return_to = "new" if arg.startswith("w") else "home"
            raw_p = arg[1:] if len(arg) > 1 else "0"
            page = int(raw_p) if raw_p.isdigit() else 0

            topics = await repository.list_topics_with_counts_today(
                draft.target_chat_id,
                _get_today_start_utc_iso(),
            )
            if not topics:
                topics = [{"thread_id": 1, "name": "عمومی", "today_count": 0}]
            g_title, _ = await _group_and_topic_labels(draft.target_chat_id, draft.target_thread_id)
            t_text = (
                f"📌 <b>انتخاب تاپیک مقصد دیپ‌تاک — «{escape_html(g_title)}»</b>\n\n"
                "تاپیک دیپ‌ورک (یا هر تاپیکی که می‌خوای فراخوان و بحث توش برگزار بشه) رو انتخاب کن 👇"
            )
            t_kb = build_deeptalk_topics_kb(
                draft.owner_id,
                topics,
                selected_thread_id=draft.target_thread_id,
                return_to=return_to,
                page=page,
            )
            await safe_edit_message(callback, t_text, t_kb)
            await safe_answer_callback(callback)
            return

        if action == "tops":
            parts = arg.split("_", 1)
            if len(parts) != 2:
                await safe_answer_callback(callback)
                return
            mode_prefix, raw_tid = parts
            try:
                tid = int(raw_tid)
            except ValueError:
                await safe_answer_callback(callback)
                return
            draft.target_thread_id = tid
            if draft.target_chat_id:
                await repository.set_deeptalk_topic(draft.target_chat_id, tid)
            target_screen = "new" if mode_prefix == "w" else "home"
            text, kb = await _render_screen_by_key(draft, user, target_screen)
            await safe_edit_message(callback, text, kb)
            await safe_answer_callback(callback, "📌 تاپیک مقصد دیپ‌تاک ذخیره شد! ✅")
            return

        if action == "subj":
            await _start_fsm_prompt(
                callback,
                state,
                draft,
                target_state=DeepTalkStates.waiting_subject,
                back_screen="new",
                prompt_html=(
                    "🎯 <b>تعیین محوریت گفتگو</b>\n\n"
                    "موضوع یا تیتر اصلی دیپ‌تاک رو بفرست (تا ۲۰۰ کاراکتر):\n"
                    "<i>مثال: چطور در ۳۰ ثانیه اول ویدیو مخاطب رو میخکوب کنیم؟</i>"
                ),
            )
            return

        if action == "desc":
            await _start_fsm_prompt(
                callback,
                state,
                draft,
                target_state=DeepTalkStates.waiting_description,
                back_screen="new",
                prompt_html=(
                    "💡 <b>توضیحات یا سوالات کلیدی (اختیاری)</b>\n\n"
                    "اگه دوست داری چند سوال راهنما یا توضیح کوتاه زیر فراخوان بیاد، الان بفرست:"
                ),
            )
            return

        if action == "dclr":
            draft.description = None
            text, kb = await render_deeptalk_wizard(draft)
            await safe_edit_message(callback, text, kb)
            await safe_answer_callback(callback, "توضیحات حذف شد 🗑")
            return

        if action == "qst":
            mins = int(arg) if arg.isdigit() else 30
            now = get_tehran_now()
            target = (now + timedelta(minutes=mins)).replace(second=0, microsecond=0)
            jdate = jdatetime.date.fromgregorian(date=target.date())
            draft.start_mode = "scheduled"
            draft.jyear, draft.jmonth, draft.jday = jdate.year, jdate.month, jdate.day
            draft.cal_year, draft.cal_month = jdate.year, jdate.month
            draft.hour, draft.minute = target.hour, target.minute
            text, kb = await render_deeptalk_wizard(draft, now_tehran=now)
            await safe_edit_message(callback, text, kb)
            await safe_answer_callback(callback, f"شروع روی {format_duration_persian(mins)} دیگه تنظیم شد ⚡")
            return

        if action == "cal":
            if arg and "-" in arg:
                ym = arg.split("-", 1)
                if len(ym) == 2 and ym[0].isdigit() and ym[1].isdigit():
                    draft.cal_year, draft.cal_month = int(ym[0]), int(ym[1])
            draft.screen = "cal"
            text, kb = await render_deeptalk_calendar(draft)
            await safe_edit_message(callback, text, kb)
            await safe_answer_callback(callback)
            return

        if action == "day":
            parts = arg.split("-")
            if len(parts) == 3 and all(p.isdigit() for p in parts):
                draft.start_mode = "scheduled"
                draft.jyear, draft.jmonth, draft.jday = int(parts[0]), int(parts[1]), int(parts[2])
                draft.cal_year, draft.cal_month = draft.jyear, draft.jmonth
            draft.screen = "tm"
            text, kb = await render_deeptalk_clock(draft)
            await safe_edit_message(callback, text, kb)
            await safe_answer_callback(callback)
            return

        if action == "tm":
            draft.screen = "tm"
            text, kb = await render_deeptalk_clock(draft)
            await safe_edit_message(callback, text, kb)
            await safe_answer_callback(callback)
            return

        if action == "tset":
            hm = arg.split(".")
            if len(hm) == 2 and hm[0].isdigit() and hm[1].isdigit():
                draft.start_mode = "scheduled"
                draft.hour = int(hm[0]) % 24
                draft.minute = int(hm[1]) % 60
            text, kb = await render_deeptalk_clock(draft)
            await safe_edit_message(callback, text, kb)
            await safe_answer_callback(callback)
            return

        if action == "hadd":
            try:
                delta_h = int(arg)
            except ValueError:
                delta_h = 0
            draft.start_mode = "scheduled"
            draft.hour = (draft.hour + delta_h) % 24
            text, kb = await render_deeptalk_clock(draft)
            await safe_edit_message(callback, text, kb)
            await safe_answer_callback(callback)
            return

        if action == "madd":
            try:
                delta_m = int(arg)
            except ValueError:
                delta_m = 0
            draft.start_mode = "scheduled"
            total_m = (draft.hour * 60 + draft.minute + delta_m) % (24 * 60)
            draft.hour, draft.minute = divmod(total_m, 60)
            text, kb = await render_deeptalk_clock(draft)
            await safe_edit_message(callback, text, kb)
            await safe_answer_callback(callback)
            return

        if action == "scus":
            await _start_fsm_prompt(
                callback,
                state,
                draft,
                target_state=DeepTalkStates.waiting_custom_start,
                back_screen="new",
                prompt_html=(
                    "⌨️ <b>ورود دستی زمان شروع دیپ‌تاک</b>\n\n"
                    "ساعت یا تاریخ و ساعت شروع رو بفرست:\n"
                    "• ساعت امروز/فردا: <code>21:00</code> یا <code>۲۱:۳۰</code>\n"
                    "• با تاریخ شمسی: <code>فردا 19:30</code> یا <code>1405/09/10 21:00</code>\n"
                    "• فاصله زمانی از الان: <code>45m</code> یا <code>1h30m</code>"
                ),
            )
            return

        if action == "dur":
            if arg == "cus":
                await _start_fsm_prompt(
                    callback,
                    state,
                    draft,
                    target_state=DeepTalkStates.waiting_custom_duration,
                    back_screen="new",
                    prompt_html=(
                        "⏱ <b>مدت زمان دلخواه گفتگو</b>\n\n"
                        f"مدت جلسه را بین <b>{to_persian_digits(MIN_DEEPTALK_DURATION_MINUTES)} تا "
                        f"{to_persian_digits(MAX_DEEPTALK_DURATION_MINUTES)} دقیقه</b> بفرست:\n"
                        "<i>مثال: <code>45</code> یا <code>1h30m</code> یا <code>۱ ساعت و ۱۵ دقیقه</code></i>"
                    ),
                )
                return

            if arg.isdigit():
                val = int(arg)
                if MIN_DEEPTALK_DURATION_MINUTES <= val <= MAX_DEEPTALK_DURATION_MINUTES:
                    draft.duration_minutes = val
            text, kb = await render_deeptalk_wizard(draft)
            await safe_edit_message(callback, text, kb)
            await safe_answer_callback(callback)
            return

        if action == "pub":
            if not draft.subject:
                await safe_answer_callback(callback, "⚠️ اول «محوریت بحث» رو مشخص کن!", show_alert=True)
                return
            settings = get_settings()
            now = get_tehran_now()
            start_dt = _compute_draft_start_dt(draft, now, settings.TIMEZONE)
            creator_name = (user.full_name or user.first_name or "ادمین دیپ‌تاک").strip()

            session_row, err = await create_and_publish_deeptalk(
                bot,
                chat_id=draft.target_chat_id,
                thread_id=draft.target_thread_id,
                subject=draft.subject,
                description=draft.description,
                created_by=user.id,
                creator_name=creator_name,
                start_at=start_dt,
                duration_minutes=draft.duration_minutes,
                now_utc=now,
            )
            if err is not None or session_row is None:
                await safe_answer_callback(callback, err or "❌ خطا در ثبت جلسه", show_alert=True)
                return

            await repository.set_deeptalk_topic(
                draft.target_chat_id,
                draft.target_thread_id,
                default_duration_minutes=draft.duration_minutes,
            )
            new_sid = int(session_row["id"])

            # If panel was opened inside the exact same group topic, remove the draft panel cleanly!
            if (
                callback.message
                and draft.panel_chat_type != "private"
                and draft.panel_chat_id == draft.target_chat_id
                and draft.panel_thread_id == draft.target_thread_id
            ):
                _dt_drafts.pop((callback.message.chat.id, callback.message.message_id), None)
                await _try_delete(callback.message)
                await safe_answer_callback(callback, "🚀 فراخوان دیپ‌تاک در تاپیک منتشر شد!")
                return

            # Reset draft subject/description and show session control screen
            draft.subject = None
            draft.description = None
            rendered = await render_deeptalk_session_detail(bot, draft.owner_id, new_sid)
            if rendered is not None:
                det_text, det_kb = rendered
                await safe_edit_message(
                    callback,
                    f"✅ <b>فراخوان دیپ‌تاک با موفقیت در تاپیک منتشر شد!</b> 🚀\n\n{det_text}",
                    det_kb,
                )
            await safe_answer_callback(callback, "فراخوان منتشر شد! 🚀")
            return

        if action == "list":
            page = int(arg) if arg.isdigit() else 0
            text, kb = await render_deeptalk_sessions_list(draft, page=page)
            await safe_edit_message(callback, text, kb)
            await safe_answer_callback(callback)
            return

        if action == "sess":
            if not arg.isdigit():
                await safe_answer_callback(callback)
                return
            rendered = await render_deeptalk_session_detail(bot, draft.owner_id, int(arg))
            if rendered is None:
                await safe_answer_callback(callback, "جلسه پیدا نشد 😅", show_alert=True)
                return
            await safe_edit_message(callback, rendered[0], rendered[1])
            await safe_answer_callback(callback)
            return

        if action == "ext":
            parts = arg.split("_", 1)
            if len(parts) != 2 or not parts[0].isdigit() or not parts[1].isdigit():
                await safe_answer_callback(callback)
                return
            target_sid, extra_m = int(parts[0]), int(parts[1])
            updated, err = await extend_live_deeptalk(bot, target_sid, extra_m)
            if err or updated is None:
                await safe_answer_callback(callback, err or "خطا در تمدید جلسه", show_alert=True)
                return
            rendered = await render_deeptalk_session_detail(bot, draft.owner_id, target_sid)
            if rendered is not None:
                await safe_edit_message(callback, rendered[0], rendered[1])
            await safe_answer_callback(callback, f"🔥 زمان دیپ‌تاک {to_persian_digits(extra_m)} دقیقه تمدید شد!")
            return

        if action == "snow":
            if not arg.isdigit():
                await safe_answer_callback(callback)
                return
            target_sid = int(arg)
            ok = await start_deeptalk_session(bot, target_sid, force_now=True)
            if not ok:
                await safe_answer_callback(callback, "این جلسه قبلاً شروع یا لغو شده است!", show_alert=True)
                return
            rendered = await render_deeptalk_session_detail(bot, draft.owner_id, target_sid)
            if rendered is not None:
                await safe_edit_message(callback, rendered[0], rendered[1])
            await safe_answer_callback(callback, "🔥 دیپ‌تاک همین الان استارت خورد!")
            return

        if action == "enow":
            parts = arg.split("_", 1)
            if len(parts) != 2 or not parts[0].isdigit():
                await safe_answer_callback(callback)
                return
            target_sid, step = int(parts[0]), parts[1]
            if step == "ask":
                kb = build_deeptalk_confirm_action_kb(
                    draft.owner_id,
                    confirm_action="enow",
                    session_id=target_sid,
                    confirm_label="⏹ بله، جلسه تمام شود",
                )
                await safe_edit_message(
                    callback,
                    "⏹ <b>پایان دستی دیپ‌تاک</b>\n\n"
                    "مطمئنی می‌خوای همین الان وقت دیپ‌تاک تمام بشه و پیام جمع‌بندی در تاپیک ارسال بشه؟",
                    kb,
                )
                await safe_answer_callback(callback)
                return

            ok = await end_deeptalk_session(bot, target_sid)
            if not ok:
                await safe_answer_callback(callback, "جلسه در حال برگزاری نیست!", show_alert=True)
            else:
                await safe_answer_callback(callback, "🏁 جلسه پایان یافت و پیام جمع‌بندی ارسال شد!")
            rendered = await render_deeptalk_session_detail(bot, draft.owner_id, target_sid)
            if rendered is not None:
                await safe_edit_message(callback, rendered[0], rendered[1])
            return

        if action == "canc":
            parts = arg.split("_", 1)
            if len(parts) != 2 or not parts[0].isdigit():
                await safe_answer_callback(callback)
                return
            target_sid, step = int(parts[0]), parts[1]
            if step == "ask":
                kb = build_deeptalk_confirm_action_kb(
                    draft.owner_id,
                    confirm_action="canc",
                    session_id=target_sid,
                    confirm_label="❌ بله، لغو کن",
                )
                await safe_edit_message(
                    callback,
                    "❌ <b>لغو دیپ‌تاک</b>\n\nمطمئنی می‌خوای این جلسه دیپ‌تاک رو کامل لغو کنی؟",
                    kb,
                )
                await safe_answer_callback(callback)
                return

            ok = await cancel_deeptalk_flow(bot, target_sid)
            toast = "🔴 دیپ‌تاک لغو شد." if ok else "قبلاً لغو یا تمام شده بود."
            text, kb = await render_deeptalk_sessions_list(draft)
            await safe_edit_message(callback, text, kb)
            await safe_answer_callback(callback, toast)
            return

        # ==================== 3. Owner-Only DeepTalk Admin Management ====================
        if action == "adm":
            page = int(arg) if arg.isdigit() else 0
            draft.screen = "adm"
            text, kb = await render_deeptalk_admins_screen(draft.owner_id, page=page)
            await safe_edit_message(callback, text, kb)
            await safe_answer_callback(callback)
            return

        if action == "adma":
            prompt_lines = [
                "👑 <b>افزودن ادمین دیپ‌تاک جدید</b>",
                "",
                "یکی از موارد زیر رو همینجا بفرست:",
                "1️⃣ <b>آیدی عددی کاربر</b> (مثلاً <code>123456789</code>)",
                "2️⃣ <b>یوزرنیم کاربر</b> (مثلاً <code>@username</code> — اگه قبلاً توی ربات یا گروه فعالیت داشته باشه)",
                "3️⃣ <b>فوروارد کردن یک پیام</b> از اون شخص (یا ریپلای روی پیامش در گروه)",
            ]
            await _start_fsm_prompt(
                callback,
                state,
                draft,
                target_state=DeepTalkStates.waiting_new_admin,
                back_screen="adm",
                prompt_html="\n".join(prompt_lines),
            )
            return

        if action == "admd":
            if not arg.isdigit():
                await safe_answer_callback(callback)
                return
            target_uid = int(arg)
            removed = await repository.remove_deeptalk_admin(target_uid)
            text, kb = await render_deeptalk_admins_screen(draft.owner_id)
            await safe_edit_message(callback, text, kb)
            toast = "🗑 دسترسی ادمین دیپ‌تاک حذف شد." if removed else "این کاربر در لیست نبود."
            await safe_answer_callback(callback, toast)
            return

    except Exception as exc:
        logger.error("Error handling DeepTalk callback: %s", exc, exc_info=True)
        await safe_answer_callback(callback, "یه خطای موقتی پیش اومد 😅 دوباره امتحان کن!", show_alert=True)


# ---------------- FSM Text / Forward Input Handlers ----------------


async def _resolve_admin_candidate(
    message: Message,
    bot: Bot,
) -> tuple[int | None, str | None, str | None, str | None]:
    """Resolve (user_id, username, full_name, error_msg) from forward, reply, numeric ID, or @username."""
    # 1. Check aiogram 3 forward_origin (MessageOriginUser) or legacy forward_from
    fwd_origin = getattr(message, "forward_origin", None)
    sender_user = getattr(fwd_origin, "sender_user", None) if fwd_origin is not None else None
    if sender_user is None:
        sender_user = getattr(message, "forward_from", None)
    if sender_user is not None and not getattr(sender_user, "is_bot", False):
        return (
            int(sender_user.id),
            getattr(sender_user, "username", None),
            getattr(sender_user, "full_name", None) or getattr(sender_user, "first_name", None),
            None,
        )
    if fwd_origin is not None and sender_user is None:
        return (
            None,
            None,
            None,
            "⚠️ حریم خصوصی فوروارد این کاربر قفله! لطفاً آیدی عددی یا <code>@username</code> اون رو بفرست.",
        )

    # 2. Check reply_to_message in group
    reply = getattr(message, "reply_to_message", None)
    if reply is not None and not getattr(reply, "forum_topic_created", None):
        r_user = getattr(reply, "from_user", None)
        if r_user is not None and not getattr(r_user, "is_bot", False):
            return (
                int(r_user.id),
                getattr(r_user, "username", None),
                getattr(r_user, "full_name", None) or getattr(r_user, "first_name", None),
                None,
            )

    raw = normalize_digits((message.text or "").strip())
    if not raw:
        return None, None, None, "⚠️ لطفاً آیدی عددی، <code>@username</code> یا یک پیام فورواردشده بفرست."

    # 3. Numeric user ID
    if raw.isdigit():
        uid = int(raw)
        if uid <= 0:
            return None, None, None, "⚠️ آیدی عددی معتبر نیست."
        known = await repository.get_bot_user(uid)
        if known:
            return uid, known.get("username"), known.get("full_name"), None
        try:
            chat = await bot.get_chat(uid)
            full_name = getattr(chat, "full_name", None) or getattr(chat, "first_name", None) or f"کاربر {uid}"
            return uid, getattr(chat, "username", None), full_name, None
        except Exception:
            return uid, None, f"کاربر {uid}", None

    # 4. @username lookup
    clean_u = raw.lstrip("@").strip()
    if clean_u and all(ch.isalnum() or ch == "_" for ch in clean_u):
        found = await repository.find_known_user_by_username(clean_u)
        if found is not None:
            return int(found["user_id"]), found.get("username"), found.get("full_name"), None
        try:
            chat = await bot.get_chat(f"@{clean_u}")
            if getattr(chat, "id", None) and int(chat.id) > 0:
                full_name = getattr(chat, "full_name", None) or getattr(chat, "first_name", None) or clean_u
                return int(chat.id), getattr(chat, "username", None) or clean_u, full_name, None
        except Exception:
            pass
        return (
            None,
            None,
            None,
            f"⚠️ یوزرنیم <code>@{escape_html(clean_u)}</code> هنوز توی دیتابیس ربات دیده نشده!\n"
            "لطفاً یا <b>آیدی عددی</b> کاربر رو بفرست، یا بگو یه بار به ربات <code>/start</code> بده، یا یه پیام ازش فوروارد کن 💡",
        )

    return None, None, None, "⚠️ ورودی نامعتبره! آیدی عددی، <code>@username</code> یا پیام فورواردشده بفرست."


@router.message(ActiveDeepTalkFsmFilter())
async def handle_deeptalk_fsm_input(message: Message, bot: Bot, state: FSMContext) -> None:
    """Process FSM text/forward inputs for the DeepTalk wizard and admin registration."""
    user = message.from_user
    if user is None:
        return

    current_state = await state.get_state()
    data = await state.get_data()
    p_cid = data.get("panel_chat_id")
    p_mid = data.get("panel_message_id")

    if message.chat and message.chat.type != "private":
        await _try_delete(message)

    if p_cid is None or p_mid is None:
        await state.clear()
        return

    draft = _dt_drafts.get((int(p_cid), int(p_mid)))
    if draft is None:
        await state.clear()
        return

    # Re-verify permissions
    if current_state == DeepTalkStates.waiting_new_admin.state:
        if not is_global_admin(user.id):
            await state.clear()
            return
    elif not await can_manage_deeptalk(user.id):
        await state.clear()
        return

    raw_text = (message.text or "").strip()

    if current_state == DeepTalkStates.waiting_subject.state:
        cleaned = " ".join(raw_text.split())
        if len(cleaned) < 2:
            await _safe_edit_by_id(
                bot,
                int(p_cid),
                int(p_mid),
                "⚠️ موضوع گفتگو خیلی کوتاهه! یه جمله یا عنوان واضح بفرست:",
                build_deeptalk_cancel_input_kb(draft.owner_id, "new"),
            )
            return
        draft.subject = cleaned[:200]
        await state.clear()
        text, kb = await render_deeptalk_wizard(draft)
        await _safe_edit_by_id(bot, int(p_cid), int(p_mid), text, kb)
        return

    if current_state == DeepTalkStates.waiting_description.state:
        draft.description = raw_text[:600] if raw_text else None
        await state.clear()
        text, kb = await render_deeptalk_wizard(draft)
        await _safe_edit_by_id(bot, int(p_cid), int(p_mid), text, kb)
        return

    if current_state == DeepTalkStates.waiting_custom_start.state:
        settings = get_settings()
        now = get_tehran_now()
        target_dt = parse_jalali_date_time_input(raw_text, now, settings.TIMEZONE)
        if target_dt is None:
            rel_mins = parse_duration_input(raw_text)
            if rel_mins is not None:
                target_dt = (now + timedelta(minutes=rel_mins)).replace(second=0, microsecond=0)

        if target_dt is None or (target_dt - now).total_seconds() < 20:
            await _safe_edit_by_id(
                bot,
                int(p_cid),
                int(p_mid),
                "⚠️ فرمت زمان معتبر نیست یا زمانش گذشته!\n"
                "مثال‌های معتبر: <code>21:00</code> ، <code>فردا 19:30</code> ، <code>45m</code>",
                build_deeptalk_cancel_input_kb(draft.owner_id, "new"),
            )
            return

        jdate = jdatetime.date.fromgregorian(date=target_dt.date())
        draft.start_mode = "scheduled"
        draft.jyear, draft.jmonth, draft.jday = jdate.year, jdate.month, jdate.day
        draft.cal_year, draft.cal_month = jdate.year, jdate.month
        draft.hour, draft.minute = target_dt.hour, target_dt.minute
        await state.clear()
        text, kb = await render_deeptalk_wizard(draft, now_tehran=now)
        await _safe_edit_by_id(bot, int(p_cid), int(p_mid), text, kb)
        return

    if current_state == DeepTalkStates.waiting_custom_duration.state:
        dur_mins = parse_duration_input(raw_text)
        if dur_mins is None or not (MIN_DEEPTALK_DURATION_MINUTES <= dur_mins <= MAX_DEEPTALK_DURATION_MINUTES):
            await _safe_edit_by_id(
                bot,
                int(p_cid),
                int(p_mid),
                f"⚠️ لطفاً عددی بین <b>{to_persian_digits(MIN_DEEPTALK_DURATION_MINUTES)} تا "
                f"{to_persian_digits(MAX_DEEPTALK_DURATION_MINUTES)} دقیقه</b> وارد کن (مثلاً <code>45</code> یا <code>1h30m</code>):",
                build_deeptalk_cancel_input_kb(draft.owner_id, "new"),
            )
            return

        draft.duration_minutes = dur_mins
        await state.clear()
        text, kb = await render_deeptalk_wizard(draft)
        await _safe_edit_by_id(bot, int(p_cid), int(p_mid), text, kb)
        return

    if current_state == DeepTalkStates.waiting_new_admin.state:
        uid, uname, fname, err = await _resolve_admin_candidate(message, bot)
        if err or uid is None:
            await _safe_edit_by_id(
                bot,
                int(p_cid),
                int(p_mid),
                f"{err}\n\nدوباره تلاش کن یا انصراف رو بزن:",
                build_deeptalk_cancel_input_kb(draft.owner_id, "adm"),
            )
            return

        await repository.add_deeptalk_admin(
            user_id=uid,
            username=uname,
            full_name=fname,
            added_by=user.id,
        )
        await state.clear()
        adm_text, adm_kb = await render_deeptalk_admins_screen(draft.owner_id)
        banner = (
            f"✅ <b>{escape_html(fname or str(uid))}</b> (<code>{uid}</code>) به عنوان ادمین دیپ‌تاک ثبت شد!\n\n"
        )
        await _safe_edit_by_id(bot, int(p_cid), int(p_mid), banner + adm_text, adm_kb)
        return

"""Private-chat commands, role-aware welcome/help, and stranger anti-spam responses."""

from datetime import datetime, timedelta, timezone
import logging
import time
from zoneinfo import ZoneInfo
from aiogram import Bot, Router
from aiogram.filters import Command, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import default_state
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup, Message, User
from bot.config import get_settings
from bot.database import repository
from bot.filters import PrivateChatFilter
from bot.handlers.panel import build_admin_help_text, render_home_or_dashboard
from bot.keyboards.panel import PanelCB, build_open_panel_kb
from bot.keyboards.timer import TimerCB, build_dm_timer_entry_kb
from bot.services.access import UserRole, is_active_group_member, resolve_user_role
from bot.texts.private_texts import pick
from bot.utils.dates import format_persian_date, format_persian_short_date
from bot.utils.html_utils import escape_html

logger = logging.getLogger(__name__)

router = Router(name="private_router")
router.message.filter(PrivateChatFilter())

STRANGER_WINDOW_SECONDS = 600       # 10 minutes
STRANGER_MAX_ATTEMPTS = 5           # 5 attempts in 10 min -> next triggers 1-hour ignore
STRANGER_IGNORE_SECONDS = 3600      # 1 hour
STRANGER_OTHER_COOLDOWN_SECONDS = 30.0  # Max 1 reply per 30s for non-/start stranger messages

# In-memory monotonic timestamps for the 30-second non-/start stranger reply cooldown
_stranger_last_reply: dict[int, float] = {}


def clear_stranger_cooldowns() -> None:
    """Clear in-memory stranger reply cooldown timestamps (useful for test isolation)."""
    _stranger_last_reply.clear()


def _parse_utc_iso(value: str | None) -> datetime | None:
    """Parse an ISO8601 timestamp string into timezone-aware UTC datetime."""
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _display_name(user: User) -> str:
    """Return friendly casual display name for a Telegram user."""
    return (user.first_name or user.full_name or "دوست عزیز").strip() or "دوست عزیز"


async def _safe_answer(
    message: Message,
    text: str,
    reply_markup=None,
) -> Message | None:
    """Safely send an HTML reply in private chat without raising."""
    try:
        return await message.answer(
            text=text,
            reply_markup=reply_markup,
            parse_mode="HTML",
            disable_web_page_preview=True,
        )
    except Exception as exc:
        logger.error("Failed sending private reply to user: %s", exc, exc_info=True)
        return None


async def notify_admins_unknown_user(
    bot: Bot,
    user: User,
    when_utc: datetime,
) -> None:
    """Notify all configured global admins once when an unknown user first contacts the bot."""
    settings = get_settings()
    if not settings.NOTIFY_ON_UNKNOWN_START or not settings.ADMIN_IDS:
        return

    tehran_dt = when_utc.astimezone(ZoneInfo(settings.TIMEZONE))
    jalali_full = format_persian_date(tehran_dt)
    jalali_short = format_persian_short_date(tehran_dt)
    time_str = tehran_dt.strftime("%H:%M:%S")

    username_str = f"@{escape_html(user.username)}" if user.username else "ندارد"
    full_name_str = escape_html(user.full_name or "نامشخص")

    alert_text = (
        "👤 <b>کاربر جدید به پیوی ربات پیام داد:</b>\n\n"
        f"🆔 <b>شناسه:</b> <code>{user.id}</code>\n"
        f"📛 <b>نام:</b> {full_name_str}\n"
        f"🔗 <b>یوزرنیم:</b> {username_str}\n"
        f"🕒 <b>زمان:</b> {jalali_full} ({jalali_short}) ساعت {time_str} (تهران)"
    )

    for admin_id in settings.ADMIN_IDS:
        try:
            await bot.send_message(
                chat_id=admin_id,
                text=alert_text,
                parse_mode="HTML",
                disable_web_page_preview=True,
            )
        except Exception as exc:
            logger.warning(
                "Failed sending unknown-user notification to admin %s: %s",
                admin_id,
                exc,
                exc_info=True,
            )


async def handle_stranger_contact(
    message: Message,
    bot: Bot,
    *,
    is_start: bool,
    now_utc: datetime | None = None,
) -> str | None:
    """Process a private message from an unauthorized user with pools, cooldown, and ignore window.

    Returns the sent text if a reply was sent, or None if suppressed/ignored.
    """
    user = message.from_user
    if user is None:
        return None

    settings = get_settings()
    current_utc = now_utc.astimezone(timezone.utc) if now_utc else datetime.now(timezone.utc)
    now_iso = current_utc.isoformat()

    user_rec = await repository.upsert_bot_user_seen(
        user_id=user.id,
        username=user.username,
        full_name=user.full_name,
        seen_at_iso=now_iso,
    )

    # 1. One-time first-contact alert to global admins
    if not user_rec.get("notified") and settings.NOTIFY_ON_UNKNOWN_START:
        await repository.mark_bot_user_notified(user.id)
        await notify_admins_unknown_user(bot, user, current_utc)

    # 2. Check active 1-hour ignore window
    ignored_until_dt = _parse_utc_iso(user_rec.get("ignored_until"))
    attempts_window_start = user_rec.get("attempts_window_start")
    attempts_count = int(user_rec.get("attempts_count") or 0)

    if ignored_until_dt is not None:
        if current_utc < ignored_until_dt:
            return None
        # Ignore period has elapsed -> reset counters cleanly
        attempts_window_start = None
        attempts_count = 0

    # 3. For non-/start messages, enforce max 1 reply per 30 seconds per user
    now_mono = time.monotonic()
    if not is_start:
        last_reply_mono = _stranger_last_reply.get(user.id)
        if last_reply_mono is not None and (now_mono - last_reply_mono) < STRANGER_OTHER_COOLDOWN_SECONDS:
            return None

    # 4. Evaluate 10-minute attempt window and select message pool
    window_start_dt = _parse_utc_iso(attempts_window_start)
    if (
        window_start_dt is None
        or (current_utc - window_start_dt) > timedelta(seconds=STRANGER_WINDOW_SECONDS)
    ):
        new_window_start = now_iso
        new_count = 1
        pool_key = "unauthorized"
        new_ignored_until: str | None = None
    else:
        new_window_start = attempts_window_start
        new_count = attempts_count + 1
        if new_count > STRANGER_MAX_ATTEMPTS:
            pool_key = "ignore_notice"
            new_ignored_until = (
                current_utc + timedelta(seconds=STRANGER_IGNORE_SECONDS)
            ).isoformat()
        else:
            pool_key = "repeat"
            new_ignored_until = None

    text, variant_key = pick(
        pool_key,
        user_id=user.id,
        exclude_last=user_rec.get("last_variant_key"),
        name=_display_name(user),
        owners=settings.OWNER_NAMES,
        escape=True,
    )

    await repository.update_bot_user_stranger_state(
        user_id=user.id,
        last_variant_key=variant_key,
        attempts_window_start=new_window_start,
        attempts_count=new_count,
        ignored_until=new_ignored_until,
    )
    _stranger_last_reply[user.id] = now_mono

    await _safe_answer(message, text)
    return text


@router.message(Command("cancel"))
async def cmd_private_cancel(message: Message, bot: Bot, state: FSMContext) -> None:
    """Clear any active FSM state in private chat."""
    try:
        user = message.from_user
        if user is None:
            return

        current_state = await state.get_state()
        await state.clear()

        if current_state is not None:
            await _safe_answer(
                message,
                "باشه، عملیات لغو شد 👍 هر وقت خواستی دوباره <code>/panel</code> رو بزن.",
            )
            return

        role, _groups = await resolve_user_role(bot, user.id)
        if role == UserRole.STRANGER:
            if await is_active_group_member(bot, user.id):
                await _safe_answer(
                    message,
                    "الان هیچ عملیاتی در جریان نیست 😄 برای تایمرهات <code>/timer</code> رو بزن.",
                    reply_markup=build_dm_timer_entry_kb(user.id),
                )
                return
            await handle_stranger_contact(message, bot, is_start=False)
            return

        await _safe_answer(
            message,
            "الان هیچ عملیاتی در جریان نیست که لغوش کنم 😄 برای مدیریت <code>/panel</code> رو بزن.",
        )
    except Exception as exc:
        logger.error("Error in private /cancel: %s", exc, exc_info=True)
        await _safe_answer(message, "یه خطای کوچیک پیش اومد 😅 دوباره امتحان کن.")


def _extract_start_payload(text: str | None) -> str:
    """Return optional deep-link payload from '/start <payload>'."""
    parts = (text or "").strip().split(maxsplit=1)
    return parts[1].strip().lower() if len(parts) > 1 else ""


@router.message(Command("start"))
async def cmd_private_start(message: Message, bot: Bot, state: FSMContext) -> None:
    """Handle /start in private chat based on user role and active group membership."""
    try:
        user = message.from_user
        if user is None:
            return

        await state.clear()
        payload = _extract_start_payload(message.text)
        role, _groups = await resolve_user_role(bot, user.id)

        if role == UserRole.STRANGER:
            if await is_active_group_member(bot, user.id):
                await repository.set_bot_user_dm_started(
                    user.id,
                    True,
                    username=user.username,
                    full_name=user.full_name,
                )
                safe_name = escape_html(_display_name(user))
                member_text = (
                    f"سلام <b>{safe_name}</b> جان! 👋\n\n"
                    "🔔 <b>اعلان‌های تایمر توی پیوی برات فعال شد ✅</b>\n"
                    "از این به بعد هر تایمری توی تاپیک‌های گروه بذاری، علاوه بر خود تاپیک، "
                    "همزمان همینجا هم بهت یادآوری می‌کنم!\n\n"
                    "همچنین می‌تونی با دستور <code>/timer</code> همینجا تایمرهای فعالت رو ببینی، "
                    "لغو کنی یا یادآور شخصی بذاری ⏱"
                )
                await _safe_answer(message, member_text, reply_markup=build_dm_timer_entry_kb(user.id))
                return

            await handle_stranger_contact(message, bot, is_start=True)
            return

        settings = get_settings()
        user_rec = await repository.upsert_bot_user_seen(
            user_id=user.id,
            username=user.username,
            full_name=user.full_name,
        )
        await repository.set_bot_user_dm_started(
            user.id,
            True,
            username=user.username,
            full_name=user.full_name,
        )

        if payload.startswith("timer"):
            safe_name = escape_html(_display_name(user))
            timer_kb = InlineKeyboardMarkup(
                inline_keyboard=[
                    [
                        InlineKeyboardButton(
                            text="⏱ باز کردن پنل تایمر",
                            callback_data=TimerCB(action="home", owner_id=user.id).pack(),
                        ),
                        InlineKeyboardButton(
                            text="📋 تایمرهای فعال من",
                            callback_data=TimerCB(action="list", owner_id=user.id, arg="0").pack(),
                        ),
                    ],
                    [
                        InlineKeyboardButton(
                            text="🎛 باز کردن پنل مدیریت",
                            callback_data=PanelCB(action="open").pack(),
                        )
                    ],
                ]
            )
            await _safe_answer(
                message,
                f"سلام <b>{safe_name}</b> جان! 👋\n\n"
                "🔔 <b>اعلان‌های تایمر توی پیوی برات فعال شد ✅</b>\n"
                "هر تایمری توی گروه بذاری، علاوه بر تاپیک گروه، همزمان همینجا هم بهت پیام میدم ⏱",
                reply_markup=timer_kb,
            )
            return

        pool_key = "admin_welcome" if role == UserRole.GLOBAL_ADMIN else "group_admin_welcome"
        text, variant_key = pick(
            pool_key,
            user_id=user.id,
            exclude_last=user_rec.get("last_variant_key"),
            name=_display_name(user),
            owners=settings.OWNER_NAMES,
            escape=True,
        )
        await repository.update_bot_user_last_variant(user.id, variant_key)
        await _safe_answer(message, text, reply_markup=build_open_panel_kb())
    except Exception as exc:
        logger.error("Error in private /start: %s", exc, exc_info=True)
        await _safe_answer(message, "یه خطای موقتی پیش اومد 😅 یه لحظه دیگه دوباره بزن.")


@router.message(Command("panel"))
async def cmd_private_panel(message: Message, bot: Bot, state: FSMContext) -> None:
    """Open the inline admin panel in private chat."""
    try:
        user = message.from_user
        if user is None:
            return

        await state.clear()
        rendered = await render_home_or_dashboard(bot, user.id, auto_enter_single=True)
        if rendered is None:
            await handle_stranger_contact(message, bot, is_start=False)
            return

        text, kb = rendered
        await _safe_answer(message, text, reply_markup=kb)
    except Exception as exc:
        logger.error("Error in private /panel: %s", exc, exc_info=True)
        await _safe_answer(message, "نتونستم پنل رو باز کنم 😅 یه لحظه دیگه دوباره /panel رو بزن.")


@router.message(Command("help"))
async def cmd_private_help(message: Message, bot: Bot) -> None:
    """Role-aware /help: unauthorized pool for strangers, full Persian guide for admins."""
    try:
        user = message.from_user
        if user is None:
            return

        role, _groups = await resolve_user_role(bot, user.id)
        if role == UserRole.STRANGER:
            if await is_active_group_member(bot, user.id):
                member_help = (
                    "⏱ <b>راهنمای تایمر و یادآور</b>\n\n"
                    "• در هر تاپیک گروه یا همینجا <code>/timer</code> رو بزن تا پنل شیشه‌ای تایمر باز بشه.\n"
                    "• می‌تونی <b>تایمر دقیقه‌ای / ساعتی</b> یا <b>تایمر روزی (تقویم شمسی)</b> تنظیم کنی و یادداشت بذاری.\n"
                    "• میانبر سریع: <code>/timer 30m برم شام</code> یا <code>/timer فردا 09:00 جلسه</code>\n"
                    "• وقتی تایمر به صدا در بیاد، هم توی همون تاپیک تگ می‌شی و هم همزمان همینجا بهت خبر میدم! 🔔"
                )
                await _safe_answer(message, member_help, reply_markup=build_dm_timer_entry_kb(user.id))
                return
            await handle_stranger_contact(message, bot, is_start=False)
            return

        text = build_admin_help_text(is_global=(role == UserRole.GLOBAL_ADMIN))
        await _safe_answer(message, text, reply_markup=build_open_panel_kb())
    except Exception as exc:
        logger.error("Error in private /help: %s", exc, exc_info=True)
        await _safe_answer(message, "یه مشکلی توی نمایش راهنما پیش اومد 😅 دوباره بزن.")


@router.message(Command("about"))
async def cmd_private_about(message: Message, bot: Bot) -> None:
    """Show casual Persian bot introduction with OWNER_NAMES."""
    try:
        user = message.from_user
        if user is None:
            return

        # Respect active 1-hour ignore window for ignored strangers
        role, _groups = await resolve_user_role(bot, user.id)
        if role == UserRole.STRANGER:
            existing = await repository.get_bot_user(user.id)
            if existing:
                ignored_until = _parse_utc_iso(existing.get("ignored_until"))
                if ignored_until is not None and datetime.now(timezone.utc) < ignored_until:
                    return

        settings = get_settings()
        owners = escape_html(settings.OWNER_NAMES)
        text = (
            "🤖 <b>درباره این ربات</b>\n\n"
            "من ربات خلاصه‌ساز روزانه‌ی تاپیک‌های گروه‌های تلگرامم! "
            "پیام‌های طول روز رو گوش میدم و هر شب با هوش مصنوعی یه خلاصه تمیز و خوانا به فارسی می‌فرستم.\n\n"
            f"👑 مدیریت و نگهداری: <b>{owners}</b> 🔥"
        )
        await _safe_answer(message, text)
    except Exception as exc:
        logger.error("Error in private /about: %s", exc, exc_info=True)
        await _safe_answer(message, "یه خطای کوچیک پیش اومد 😅 دوباره امتحان کن.")


@router.message(StateFilter(default_state))
async def handle_private_fallback(message: Message, bot: Bot) -> None:
    """Handle any other private message outside FSM states."""
    try:
        user = message.from_user
        if user is None:
            return

        role, _groups = await resolve_user_role(bot, user.id)
        if role == UserRole.STRANGER:
            if await is_active_group_member(bot, user.id):
                await _safe_answer(
                    message,
                    "برای تنظیم یا دیدن تایمرهات دستور <code>/timer</code> رو بزن یا از دکمه‌های زیر استفاده کن ⏱",
                    reply_markup=build_dm_timer_entry_kb(user.id),
                )
                return
            await handle_stranger_contact(message, bot, is_start=False)
            return

        settings = get_settings()
        user_rec = await repository.upsert_bot_user_seen(
            user_id=user.id,
            username=user.username,
            full_name=user.full_name,
        )
        text, variant_key = pick(
            "admin_hint",
            user_id=user.id,
            exclude_last=user_rec.get("last_variant_key"),
            name=_display_name(user),
            owners=settings.OWNER_NAMES,
            escape=True,
        )
        await repository.update_bot_user_last_variant(user.id, variant_key)
        await _safe_answer(message, text, reply_markup=build_open_panel_kb())
    except Exception as exc:
        logger.error("Error in private fallback handler: %s", exc, exc_info=True)
        await _safe_answer(message, "یه مشکلی پیش اومد 😅 /panel رو بزن تا پنل بیاد.")

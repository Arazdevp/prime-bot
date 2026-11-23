"""Private-chat inline admin panel handlers, FSM states, and screen renderers."""

import asyncio
from datetime import datetime, timedelta
import logging
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo
from aiogram import Bot, Router
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import CallbackQuery, InlineKeyboardMarkup, Message
from bot.config import get_settings
from bot.database import repository
from bot.filters import PrivateChatFilter
from bot.keyboards.panel import (
    PanelCB,
    build_back_to_group_kb,
    build_back_to_home_kb,
    build_dashboard_kb,
    build_dest_confirm_kb,
    build_dest_topics_kb,
    build_destination_kb,
    build_global_status_kb,
    build_home_kb,
    build_manual_run_confirm_kb,
    build_manual_run_kb,
    build_min_messages_kb,
    build_preview_topics_kb,
    build_topics_kb,
)
from bot.scheduler.jobs import (
    get_next_nightly_run_time,
    get_summary_lock,
    run_summary_for_group,
    run_summary_for_topic,
)
from bot.services.access import (
    UserRole,
    can_manage_group,
    is_global_admin,
    resolve_user_role,
)
from bot.texts.private_texts import pick
from bot.utils.dates import (
    format_persian_date,
    format_persian_short_date,
    get_tehran_now,
)
from bot.utils.html_utils import escape_html

logger = logging.getLogger(__name__)

router = Router(name="panel_router")
router.callback_query.filter(PrivateChatFilter())
router.message.filter(PrivateChatFilter())

_PERSIAN_DIGIT_TABLE = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")

_GLOBAL_ONLY_ACTIONS = {"stat"}
_GROUP_ACTIONS = {
    "grp",
    "tog",
    "mode",
    "tops",
    "topt",
    "topa",
    "dest",
    "dstp",
    "dstl",
    "dsts",
    "dstc",
    "min",
    "mind",
    "minf",
    "run",
    "runc",
    "rune",
    "prv",
    "prvr",
}
_GENERAL_ACTIONS = {"open", "home", "help", "noop"}


class PanelStates(StatesGroup):
    """FSM states used by the private admin panel."""

    waiting_min_messages = State()


def parse_min_messages_input(raw_text: str | None) -> int | None:
    """Parse a 1..100 integer from ASCII, Persian, or Arabic-Indic digits."""
    if not raw_text:
        return None
    normalized = raw_text.strip().translate(_PERSIAN_DIGIT_TABLE)
    if not normalized or not normalized.isdigit():
        return None
    val = int(normalized)
    if 1 <= val <= 100:
        return val
    return None


def _get_today_start_utc_iso() -> str:
    """Return start of today in Asia/Tehran converted to UTC ISO8601."""
    tehran_now = get_tehran_now()
    start_of_day = tehran_now.replace(hour=0, minute=0, second=0, microsecond=0)
    return start_of_day.astimezone(ZoneInfo("UTC")).isoformat()


def _format_next_nightly_run() -> str:
    """Format next scheduled nightly run time in Jalali + Tehran HH:MM."""
    cfg = get_settings()
    scheduled = get_next_nightly_run_time()
    if scheduled is not None:
        tehran_tz = ZoneInfo(cfg.TIMEZONE)
        target_time = (
            scheduled.astimezone(tehran_tz)
            if scheduled.tzinfo
            else scheduled.replace(tzinfo=tehran_tz)
        )
    else:
        tehran_now = get_tehran_now()
        target_time = tehran_now.replace(
            hour=cfg.SUMMARY_TIME_HOUR,
            minute=cfg.SUMMARY_TIME_MINUTE,
            second=0,
            microsecond=0,
        )
        if target_time <= tehran_now:
            target_time += timedelta(days=1)

    return f"{format_persian_short_date(target_time)} ساعت {target_time.strftime('%H:%M')} (تهران)"


def _format_ai_provider() -> str:
    """Return escaped display label for the configured AI provider and model."""
    cfg = get_settings()
    if cfg.AI_PROVIDER == "gemini":
        return f"Google Gemini ({escape_html(cfg.GEMINI_MODEL)})"
    return f"OpenAI ({escape_html(cfg.OPENAI_MODEL or 'نامشخص')})"


def _get_db_size_bytes(db_path: str) -> int:
    """Return total disk footprint in bytes for the SQLite DB file and WAL if present."""
    total = 0
    base = Path(db_path)
    for candidate in (base, Path(f"{db_path}-wal"), Path(f"{db_path}-shm")):
        if candidate.is_file():
            try:
                total += candidate.stat().st_size
            except OSError:
                pass
    return total


def _format_bytes(size_bytes: int) -> str:
    """Format byte size into human-readable Persian KB/MB string."""
    if size_bytes < 1024:
        return f"{size_bytes} بایت"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} کیلوبایت"
    return f"{size_bytes / (1024 * 1024):.2f} مگابایت"


async def _get_group_title(chat_id: int) -> str:
    """Fetch known group title or fallback to chat ID label."""
    group = await repository.get_known_group(chat_id)
    if group and group.get("title"):
        return str(group["title"])
    return f"گروه {chat_id}"


async def _get_topic_display_name(chat_id: int, thread_id: int) -> str:
    """Fetch stored topic name or standard Persian fallback."""
    name = await repository.get_topic_name(chat_id, thread_id)
    if name:
        return name
    return "عمومی" if thread_id == 1 else f"تاپیک {thread_id}"


def build_admin_help_text(is_global: bool) -> str:
    """Build casual Persian guide for admins in private chat."""
    role_note = (
        "تو <b>مدیر کل</b> رباتی، پس همه‌ی گروه‌ها و وضعیت کلی سرور رو می‌بینی 🔥"
        if is_global
        else "تو <b>ادمین گروه</b> هستی و هر گروهی که توش ادمین باشی برات توی پنل میاد 🔥"
    )
    return (
        "📖 <b>راهنمای پنل مدیریت ربات خلاصه‌ساز</b>\n\n"
        f"{role_note}\n\n"
        "<b>دستورات پیوی:</b>\n"
        "• <code>/panel</code> — باز کردن پنل مدیریتی دکمه‌ای\n"
        "• <code>/timer</code> — پنل شیشه‌ای تایمر و یادآور (دقیقه‌ای/ساعتی و تقویم شمسی)\n"
        "• <code>/start</code> — خوش‌آمدگویی و دکمه سریع پنل\n"
        "• <code>/help</code> — نمایش همین راهنما\n"
        "• <code>/about</code> — معرفی ربات و مدیرهاش\n"
        "• <code>/cancel</code> — لغو عملیات در حال انجام (مثل وارد کردن عدد)\n\n"
        "<b>امکانات پنل شیشه‌ای:</b>\n"
        "• روشن/خاموش کردن خلاصه‌سازی هر گروه\n"
        "• تغییر حالت لیست سیاه / لیست سفید\n"
        "• فعال و غیرفعال کردن تک‌تک تاپیک‌ها یا همه‌شون یکجا\n"
        "• تعیین مقصد خلاصه‌ها (هر تاپیک جدا یا تجمیعی)\n"
        "• تنظیم حداقل پیام (با دکمه یا وارد کردن عدد ۱ تا ۱۰۰)\n"
        "• اجرای دستی خلاصه (تک‌تاپیک یا کل گروه، با قابلیت keep)\n"
        "• 👁 پیش‌نمایش خلاصه فقط توی پیوی خودت بدون دست زدن به دیتابیس\n\n"
        "<i>راستی، تمام دستورات قدیمی توی خود گروه‌ها هم هنوز سر جاشونن و کار می‌کنن!</i>"
    )


async def render_home(
    groups: list[dict[str, Any]],
    is_global: bool,
    page: int = 0,
) -> tuple[str, InlineKeyboardMarkup]:
    """Render Home screen listing all manageable groups."""
    enriched: list[dict[str, Any]] = []
    for g in groups:
        cid = int(g["chat_id"])
        gs = await repository.get_group_settings(cid)
        enriched.append({**g, "enabled": gs.get("enabled", 1) if gs else 1})

    if enriched:
        text = (
            "🎛 <b>پنل مدیریت ربات خلاصه‌ساز</b>\n\n"
            f"تعداد گروه‌های قابل مدیریت: <b>{len(enriched)}</b>\n"
            "یه گروه رو انتخاب کن تا تنظیماتش رو ببینیم 👇"
        )
    else:
        text = (
            "🎛 <b>پنل مدیریت ربات خلاصه‌ساز</b>\n\n"
            "فعلاً هیچ گروهی ثبت نشده 😅\n"
            "من رو به یه سوپرگروه تاپیک‌دار اضافه کن و یه پیام توش بفرست تا همینجا پیداش بشه!"
        )

    return text, build_home_kb(enriched, is_global_admin=is_global, page=page)


async def render_group_dashboard(chat_id: int) -> tuple[str, InlineKeyboardMarkup]:
    """Render Screen 2: Group Dashboard."""
    cfg = get_settings()
    group_cfg = await repository.ensure_group_settings(chat_id, cfg.MIN_MESSAGES_PER_TOPIC)
    title = await _get_group_title(chat_id)

    is_enabled = group_cfg.get("enabled", 1) == 1
    status_str = "🟢 فعال" if is_enabled else "🔴 غیرفعال"

    mode = group_cfg.get("mode", "blacklist")
    mode_str = "لیست سیاه (Blacklist)" if mode == "blacklist" else "لیست سفید (Whitelist)"

    dest_mode = group_cfg.get("destination_mode", "per_topic")
    if dest_mode == "aggregate":
        dest_tid = group_cfg.get("destination_thread_id")
        dest_name = (await _get_topic_display_name(chat_id, dest_tid)) if dest_tid else "نامشخص"
        dest_str = f"تجمیعی در «{escape_html(dest_name)}» (<code>{dest_tid}</code>)"
    else:
        dest_str = "ارسال در همان تاپیک (Per Topic)"

    min_msgs = group_cfg.get("min_messages", cfg.MIN_MESSAGES_PER_TOPIC)
    unsummarized = await repository.count_unsummarized_messages(chat_id)
    next_run = _format_next_nightly_run()
    ai_info = _format_ai_provider()

    text = (
        f"🏢 <b>داشبورد گروه: {escape_html(title)}</b>\n"
        f"🆔 شناسه: <code>{chat_id}</code>\n\n"
        f"🔹 <b>وضعیت:</b> {status_str}\n"
        f"🔹 <b>حالت پایش:</b> {mode_str}\n"
        f"🔹 <b>مقصد خلاصه:</b> {dest_str}\n"
        f"🔹 <b>حداقل پیام:</b> {min_msgs} پیام\n"
        f"🔹 <b>پیام‌های تلخیص‌نشده:</b> {unsummarized} پیام\n"
        f"🔹 <b>اجرای شبانه بعدی:</b> {next_run}\n"
        f"🔹 <b>مدل هوش مصنوعی:</b> {ai_info}"
    )
    return text, build_dashboard_kb(chat_id, enabled=is_enabled, mode=mode)


async def render_home_or_dashboard(
    bot: Bot,
    user_id: int,
    *,
    auto_enter_single: bool = True,
    page: int = 0,
) -> tuple[str, InlineKeyboardMarkup] | None:
    """Open Home screen or directly enter Group Dashboard when exactly 1 group is manageable."""
    role, groups = await resolve_user_role(bot, user_id)
    if role == UserRole.STRANGER:
        return None
    if auto_enter_single and len(groups) == 1:
        return await render_group_dashboard(int(groups[0]["chat_id"]))
    return await render_home(groups, is_global=(role == UserRole.GLOBAL_ADMIN), page=page)


async def render_topics(chat_id: int, page: int = 0) -> tuple[str, InlineKeyboardMarkup]:
    """Render Screen 3: Paginated Topics management."""
    title = await _get_group_title(chat_id)
    group_cfg = await repository.ensure_group_settings(chat_id)
    mode = group_cfg.get("mode", "blacklist")
    mode_fa = "لیست سیاه" if mode == "blacklist" else "لیست سفید"

    topics = await repository.list_topics_with_counts_today(chat_id, _get_today_start_utc_iso())

    if not topics:
        text = (
            f"📂 <b>تاپیک‌های گروه «{escape_html(title)}»</b>\n"
            f"حالت فعلی: <b>{mode_fa}</b>\n\n"
            "هنوز هیچ تاپیکی توی دیتابیس ثبت نشده 😅\n"
            "یه پیام توی تاپیک‌های گروه فرستاده بشه، خودشون اینجا ظاهر میشن."
        )
    else:
        has_unknown = any(not t.get("has_known_name", True) for t in topics)
        hint = (
            "\n\n💡 <i>نکته: اسم بعضی تاپیک‌ها هنوز برام مشخص نیست؛ کافیه یه پیام توشون فرستاده بشه یا اسمشون ویرایش بشه تا یادش بگیرم.</i>"
            if has_unknown
            else ""
        )
        text = (
            f"📂 <b>تاپیک‌های گروه «{escape_html(title)}»</b>\n"
            f"حالت فعلی: <b>{mode_fa}</b> | تعداد کل: <b>{len(topics)}</b>\n"
            "روی هر تاپیک بزن تا وضعیتش (✅/⛔) عوض بشه — عدد داخل پرانتز پیام‌های امروزه:"
            f"{hint}"
        )

    return text, build_topics_kb(chat_id, topics, page=page)


async def render_destination(chat_id: int) -> tuple[str, InlineKeyboardMarkup]:
    """Render Screen 4a: Destination mode selector."""
    title = await _get_group_title(chat_id)
    group_cfg = await repository.ensure_group_settings(chat_id)
    dest_mode = group_cfg.get("destination_mode", "per_topic")
    if dest_mode == "aggregate":
        dest_tid = group_cfg.get("destination_thread_id")
        dest_name = (await _get_topic_display_name(chat_id, dest_tid)) if dest_tid else "نامشخص"
        current_str = f"تجمیعی در تاپیک <b>«{escape_html(dest_name)}»</b> (<code>{dest_tid}</code>)"
    else:
        current_str = "<b>ارسال در همان تاپیک (Per Topic)</b>"

    text = (
        f"📌 <b>مقصد خلاصه‌های گروه «{escape_html(title)}»</b>\n\n"
        f"تنظیم فعلی: {current_str}\n\n"
        "دوست داری خلاصه‌ها کجا ارسال بشن؟"
    )
    return text, build_destination_kb(chat_id, current_mode=dest_mode)


async def render_dest_topics(chat_id: int, page: int = 0) -> tuple[str, InlineKeyboardMarkup]:
    """Render Screen 4b: Paginated topic picker for aggregate destination."""
    title = await _get_group_title(chat_id)
    topics = await repository.list_topics_with_counts_today(chat_id, _get_today_start_utc_iso())
    if not topics:
        topics = [{"thread_id": 1, "name": "عمومی", "today_count": 0}]

    text = (
        f"📌 <b>انتخاب تاپیک مقصد تجمیعی برای «{escape_html(title)}»</b>\n\n"
        "تاپیکی که می‌خوای همه خلاصه‌ها توش جمع بشن رو انتخاب کن:"
    )
    return text, build_dest_topics_kb(chat_id, topics, page=page)


async def render_dest_confirm(
    chat_id: int,
    thread_id: int,
    page: int = 0,
) -> tuple[str, InlineKeyboardMarkup]:
    """Render Screen 4c: Confirm aggregate destination topic."""
    title = await _get_group_title(chat_id)
    topic_name = await _get_topic_display_name(chat_id, thread_id)
    text = (
        f"📌 <b>تأیید مقصد تجمیعی</b>\n\n"
        f"گروه: <b>{escape_html(title)}</b>\n"
        f"تاپیک مقصد: <b>«{escape_html(topic_name)}»</b> (<code>{thread_id}</code>)\n\n"
        "مطمئنی همه خلاصه‌های شبانه و دستی این گروه برن توی همین تاپیک؟"
    )
    return text, build_dest_confirm_kb(chat_id, thread_id, page=page)


async def render_min_messages(chat_id: int) -> tuple[str, InlineKeyboardMarkup]:
    """Render Screen 5: Min messages threshold editor."""
    cfg = get_settings()
    title = await _get_group_title(chat_id)
    group_cfg = await repository.ensure_group_settings(chat_id, cfg.MIN_MESSAGES_PER_TOPIC)
    current_min = group_cfg.get("min_messages", cfg.MIN_MESSAGES_PER_TOPIC)

    text = (
        f"🔢 <b>حداقل پیام برای خلاصه‌سازی — «{escape_html(title)}»</b>\n\n"
        f"مقدار فعلی: <b>{current_min} پیام</b>\n"
        "تاپیک‌هایی که تعداد پیامشون از این کمتر باشه خلاصه نمی‌شن.\n"
        "با دکمه‌های زیر کم و زیادش کن یا خودت عددش رو (۱ تا ۱۰۰) بفرست 👇"
    )
    return text, build_min_messages_kb(chat_id)


async def render_manual_run(chat_id: int, page: int = 0) -> tuple[str, InlineKeyboardMarkup]:
    """Render Screen 6a: Manual run target picker."""
    title = await _get_group_title(chat_id)
    topics = await repository.list_topics_with_counts_today(chat_id, _get_today_start_utc_iso())
    text = (
        f"▶️ <b>اجرای دستی خلاصه‌سازی — «{escape_html(title)}»</b>\n\n"
        "کدوم تاپیک رو الان خلاصه کنم؟ می‌تونی «همه تاپیک‌ها» یا یه تاپیک خاص رو انتخاب کنی 👇"
    )
    return text, build_manual_run_kb(chat_id, topics, page=page)


async def render_manual_run_confirm(
    chat_id: int,
    target_key: str,
    keep: bool,
    page: int = 0,
) -> tuple[str, InlineKeyboardMarkup]:
    """Render Screen 6b: Confirm manual run and keep flag."""
    title = await _get_group_title(chat_id)
    if target_key == "a":
        target_label = "همه تاپیک‌های فعال گروه"
    else:
        tid = int(target_key)
        t_name = await _get_topic_display_name(chat_id, tid)
        target_label = f"تاپیک «{t_name}»"

    keep_desc = (
        "حالت <b>keep روشن</b> — پیام‌ها بعد از خلاصه‌سازی توی دیتابیس باقی می‌مونن (مناسب تست)."
        if keep
        else "حالت <b>عادی</b> — پیام‌ها بعد از ارسال موفق خلاصه از دیتابیس پاک میشن."
    )
    text = (
        f"▶️ <b>تأیید اجرای دستی</b>\n\n"
        f"🏢 <b>گروه:</b> {escape_html(title)}\n"
        f"🎯 <b>هدف:</b> {escape_html(target_label)}\n"
        f"📦 <b>وضعیت پیام‌ها:</b> {keep_desc}\n\n"
        "اگه همه چیز اوکیه، بزن روی «شروع اجرا» 🔥"
    )
    return text, build_manual_run_confirm_kb(chat_id, target_key, keep, page=page)


async def render_preview_topics(chat_id: int, page: int = 0) -> tuple[str, InlineKeyboardMarkup]:
    """Render Screen 7: Preview topic picker."""
    title = await _get_group_title(chat_id)
    topics = await repository.list_topics_with_counts_today(chat_id, _get_today_start_utc_iso())
    if not topics:
        text = (
            f"👁 <b>پیش‌نمایش خلاصه — «{escape_html(title)}»</b>\n\n"
            "فعلاً هیچ تاپیکی برای پیش‌نمایش ثبت نشده 😅"
        )
    else:
        text = (
            f"👁 <b>پیش‌نمایش خلاصه — «{escape_html(title)}»</b>\n\n"
            "تاپیک رو انتخاب کن تا خلاصه‌ش <b>فقط همینجا توی پیوی خودت</b> فرستاده بشه "
            "(توی گروه چیزی ارسال نمی‌شه و هیچ پیامی هم پاک نمی‌شه) 👇"
        )
    return text, build_preview_topics_kb(chat_id, topics, page=page)


async def render_global_status() -> tuple[str, InlineKeyboardMarkup]:
    """Render Screen 8: Global status overview (global admins only)."""
    cfg = get_settings()
    groups = await repository.list_active_known_groups()

    group_lines: list[str] = []
    total_unsummarized = 0
    for g in groups:
        cid = int(g["chat_id"])
        title = g.get("title") or f"گروه {cid}"
        gs = await repository.get_group_settings(cid)
        enabled = (gs.get("enabled", 1) == 1) if gs else True
        cnt = await repository.count_unsummarized_messages(cid)
        total_unsummarized += cnt
        dot = "🟢" if enabled else "🔴"
        group_lines.append(f"• {dot} <b>{escape_html(title)}</b>: {cnt} پیام تلخیص‌نشده")

    groups_block = "\n".join(group_lines) if group_lines else "• هنوز هیچ گروه فعالی ثبت نشده."

    last_run_iso = await repository.get_last_nightly_run()
    if last_run_iso:
        try:
            dt = datetime.fromisoformat(last_run_iso)
            tehran_tz = ZoneInfo(cfg.TIMEZONE)
            tehran_dt = (
                dt.astimezone(tehran_tz)
                if dt.tzinfo
                else dt.replace(tzinfo=ZoneInfo("UTC")).astimezone(tehran_tz)
            )
            last_run_str = (
                f"{format_persian_date(tehran_dt)} ساعت {tehran_dt.strftime('%H:%M')} (تهران)"
            )
        except Exception:
            last_run_str = escape_html(last_run_iso)
    else:
        last_run_str = "هنوز اجرا نشده"

    next_run_str = _format_next_nightly_run()
    ai_info = _format_ai_provider()
    db_size = await asyncio.to_thread(_get_db_size_bytes, cfg.DB_PATH)

    text = (
        "📊 <b>وضعیت کلی ربات</b>\n\n"
        f"🏢 <b>گروه‌های فعال ({len(groups)}):</b>\n"
        f"{groups_block}\n\n"
        f"💬 <b>کل پیام‌های تلخیص‌نشده:</b> {total_unsummarized}\n"
        f"🌙 <b>آخرین اجرای شبانه:</b> {last_run_str}\n"
        f"⏰ <b>اجرای شبانه بعدی:</b> {next_run_str}\n"
        f"🧠 <b>سرویس هوش مصنوعی:</b> {ai_info}\n"
        f"💾 <b>حجم دیتابیس:</b> {_format_bytes(db_size)}"
    )
    return text, build_global_status_kb()


async def safe_answer_callback(
    callback: CallbackQuery,
    text: str | None = None,
    *,
    show_alert: bool = False,
) -> None:
    """Safely answer a callback query without raising on expired query IDs."""
    try:
        await callback.answer(text=text, show_alert=show_alert)
    except Exception as exc:
        logger.debug("Failed answering callback query: %s", exc)


async def safe_edit_message(
    callback: CallbackQuery,
    text: str,
    reply_markup: InlineKeyboardMarkup | None = None,
) -> bool:
    """Safely edit callback message text, ignoring 'message is not modified'."""
    msg = callback.message
    if msg is None or not hasattr(msg, "edit_text"):
        return False
    try:
        await msg.edit_text(
            text=text,
            reply_markup=reply_markup,
            parse_mode="HTML",
            disable_web_page_preview=True,
        )
        return True
    except TelegramBadRequest as exc:
        if "message is not modified" in str(exc).lower():
            return True
        logger.warning("TelegramBadRequest editing panel message: %s", exc, exc_info=True)
        return False
    except Exception as exc:
        logger.error("Unexpected error editing panel message: %s", exc, exc_info=True)
        return False


async def _deny_callback(callback: CallbackQuery) -> None:
    """Reject unauthorized callback with a random casual Persian alert."""
    user = callback.from_user
    uid = user.id if user else None
    name = (user.first_name or user.full_name or "دوست عزیز") if user else "دوست عزیز"
    text, _ = pick("callback_denied", user_id=uid, name=name, escape=False)
    await safe_answer_callback(callback, text, show_alert=True)


def _parse_run_arg(arg: str) -> tuple[str, bool] | None:
    """Parse manual-run arg '{target_key}_{keep_flag}' into (target_key, keep_bool)."""
    parts = arg.split("_", 1)
    if len(parts) != 2:
        return None
    target_key, keep_raw = parts
    if keep_raw not in ("0", "1"):
        return None
    if target_key != "a":
        try:
            int(target_key)
        except ValueError:
            return None
    return target_key, keep_raw == "1"


@router.callback_query(PanelCB.filter())
async def handle_panel_callback(
    callback: CallbackQuery,
    callback_data: PanelCB,
    bot: Bot,
    state: FSMContext,
) -> None:
    """Dispatch all PanelCB inline-keyboard interactions with strict permission re-checks."""
    try:
        user = callback.from_user
        if user is None:
            await safe_answer_callback(callback)
            return

        action = callback_data.action
        chat_id = callback_data.chat_id
        arg = callback_data.arg
        page = callback_data.page

        # Re-verify permissions on EVERY callback
        if action in _GLOBAL_ONLY_ACTIONS:
            if not is_global_admin(user.id):
                await _deny_callback(callback)
                return
        elif action in _GROUP_ACTIONS:
            if not await can_manage_group(bot, user.id, chat_id):
                await _deny_callback(callback)
                return
        elif action in _GENERAL_ACTIONS:
            role, _groups = await resolve_user_role(bot, user.id)
            if role == UserRole.STRANGER:
                await _deny_callback(callback)
                return
        else:
            await safe_answer_callback(
                callback,
                "این دکمه قدیمی یا نامعتبره 😅 با /panel یه پنل تازه باز کن",
                show_alert=True,
            )
            return

        # Clear any pending FSM state unless entering manual min-messages input
        if action != "minf":
            await state.clear()

        if action == "noop":
            await safe_answer_callback(callback)
            return

        if action == "open":
            rendered = await render_home_or_dashboard(bot, user.id, auto_enter_single=True)
            if rendered is None:
                await _deny_callback(callback)
                return
            text, kb = rendered
            await safe_edit_message(callback, text, kb)
            await safe_answer_callback(callback)
            return

        if action == "home":
            role, groups = await resolve_user_role(bot, user.id)
            if role == UserRole.STRANGER:
                await _deny_callback(callback)
                return
            text, kb = await render_home(groups, is_global=(role == UserRole.GLOBAL_ADMIN), page=page)
            await safe_edit_message(callback, text, kb)
            await safe_answer_callback(callback)
            return

        if action == "help":
            text = build_admin_help_text(is_global=is_global_admin(user.id))
            await safe_edit_message(callback, text, build_back_to_home_kb())
            await safe_answer_callback(callback)
            return

        if action == "stat":
            text, kb = await render_global_status()
            await safe_edit_message(callback, text, kb)
            await safe_answer_callback(callback)
            return

        if action == "grp":
            text, kb = await render_group_dashboard(chat_id)
            await safe_edit_message(callback, text, kb)
            await safe_answer_callback(callback)
            return

        if action == "tog":
            group_cfg = await repository.ensure_group_settings(chat_id)
            new_enabled = 0 if group_cfg.get("enabled", 1) == 1 else 1
            await repository.set_group_enabled(chat_id, new_enabled)
            text, kb = await render_group_dashboard(chat_id)
            await safe_edit_message(callback, text, kb)
            toast = "🟢 خلاصه‌سازی فعال شد!" if new_enabled == 1 else "🔴 خلاصه‌سازی غیرفعال شد!"
            await safe_answer_callback(callback, toast)
            return

        if action == "mode":
            group_cfg = await repository.ensure_group_settings(chat_id)
            new_mode = "whitelist" if group_cfg.get("mode", "blacklist") == "blacklist" else "blacklist"
            await repository.set_group_mode(chat_id, new_mode)
            text, kb = await render_group_dashboard(chat_id)
            await safe_edit_message(callback, text, kb)
            toast = "حالت شد لیست سفید ✅" if new_mode == "whitelist" else "حالت شد لیست سیاه ✅"
            await safe_answer_callback(callback, toast)
            return

        if action == "tops":
            text, kb = await render_topics(chat_id, page=page)
            await safe_edit_message(callback, text, kb)
            await safe_answer_callback(callback)
            return

        if action == "topt":
            try:
                tid = int(arg)
            except ValueError:
                await safe_answer_callback(callback, "شناسه تاپیک معتبر نیست 😅", show_alert=True)
                return

            # Toggle using exact group-command semantics
            group_cfg = await repository.ensure_group_settings(chat_id)
            topics = await repository.list_topics_with_counts_today(chat_id, _get_today_start_utc_iso())
            current = next((t for t in topics if int(t["thread_id"]) == tid), None)
            if current is not None:
                currently_monitored = bool(current["is_monitored"])
            else:
                currently_monitored = group_cfg.get("mode", "blacklist") == "blacklist"

            new_active = 0 if currently_monitored else 1
            await repository.set_topic_active_status(chat_id, tid, is_active=new_active)
            text, kb = await render_topics(chat_id, page=page)
            await safe_edit_message(callback, text, kb)
            await safe_answer_callback(callback, "✅ فعال شد" if new_active == 1 else "⛔ غیرفعال شد")
            return

        if action == "topa":
            if arg not in ("0", "1"):
                await safe_answer_callback(callback, "دستور نامعتبره 😅", show_alert=True)
                return
            new_active = int(arg)
            await repository.set_all_topics_active_status(chat_id, new_active)
            text, kb = await render_topics(chat_id, page=page)
            await safe_edit_message(callback, text, kb)
            toast = "✅ همه تاپیک‌ها فعال شدن!" if new_active == 1 else "⛔ همه تاپیک‌ها غیرفعال شدن!"
            await safe_answer_callback(callback, toast)
            return

        if action == "dest":
            text, kb = await render_destination(chat_id)
            await safe_edit_message(callback, text, kb)
            await safe_answer_callback(callback)
            return

        if action == "dstp":
            await repository.ensure_group_settings(chat_id)
            await repository.set_group_destination(chat_id, "per_topic", None)
            text, kb = await render_destination(chat_id)
            await safe_edit_message(callback, text, kb)
            await safe_answer_callback(callback, "مقصد برگشت روی هر تاپیک جدا ✅")
            return

        if action == "dstl":
            text, kb = await render_dest_topics(chat_id, page=page)
            await safe_edit_message(callback, text, kb)
            await safe_answer_callback(callback)
            return

        if action == "dsts":
            try:
                tid = int(arg)
            except ValueError:
                await safe_answer_callback(callback, "شناسه تاپیک معتبر نیست 😅", show_alert=True)
                return
            text, kb = await render_dest_confirm(chat_id, tid, page=page)
            await safe_edit_message(callback, text, kb)
            await safe_answer_callback(callback)
            return

        if action == "dstc":
            try:
                tid = int(arg)
            except ValueError:
                await safe_answer_callback(callback, "شناسه تاپیک معتبر نیست 😅", show_alert=True)
                return
            await repository.ensure_group_settings(chat_id)
            await repository.set_group_destination(chat_id, "aggregate", tid)
            text, kb = await render_destination(chat_id)
            await safe_edit_message(callback, text, kb)
            await safe_answer_callback(callback, "📌 مقصد تجمیعی ذخیره شد!")
            return

        if action == "min":
            text, kb = await render_min_messages(chat_id)
            await safe_edit_message(callback, text, kb)
            await safe_answer_callback(callback)
            return

        if action == "mind":
            try:
                delta = int(arg)
            except ValueError:
                await safe_answer_callback(callback, "مقدار نامعتبره 😅", show_alert=True)
                return
            cfg = get_settings()
            group_cfg = await repository.ensure_group_settings(chat_id, cfg.MIN_MESSAGES_PER_TOPIC)
            current = int(group_cfg.get("min_messages", cfg.MIN_MESSAGES_PER_TOPIC))
            updated = max(1, min(100, current + delta))
            await repository.set_group_min_messages(chat_id, updated)
            text, kb = await render_min_messages(chat_id)
            await safe_edit_message(callback, text, kb)
            await safe_answer_callback(callback, f"شد {updated} پیام ✅")
            return

        if action == "minf":
            await state.set_state(PanelStates.waiting_min_messages)
            panel_msg_id = callback.message.message_id if callback.message else None
            await state.update_data(chat_id=chat_id, panel_message_id=panel_msg_id)
            title = await _get_group_title(chat_id)
            prompt_text = (
                f"✏️ <b>تنظیم حداقل پیام برای «{escape_html(title)}»</b>\n\n"
                "یه عدد بین <b>۱ تا ۱۰۰</b> بفرست (عدد فارسی هم قبوله).\n"
                "اگه پشیمون شدی <code>/cancel</code> رو بزن 👇"
            )
            await safe_edit_message(callback, prompt_text, build_back_to_group_kb(chat_id))
            await safe_answer_callback(callback)
            return

        if action == "run":
            text, kb = await render_manual_run(chat_id, page=page)
            await safe_edit_message(callback, text, kb)
            await safe_answer_callback(callback)
            return

        if action == "runc":
            parsed = _parse_run_arg(arg)
            if parsed is None:
                await safe_answer_callback(callback, "پارامتر اجرا معتبر نیست 😅", show_alert=True)
                return
            target_key, keep = parsed
            text, kb = await render_manual_run_confirm(chat_id, target_key, keep, page=page)
            await safe_edit_message(callback, text, kb)
            await safe_answer_callback(callback)
            return

        if action == "rune":
            parsed = _parse_run_arg(arg)
            if parsed is None:
                await safe_answer_callback(callback, "پارامتر اجرا معتبر نیست 😅", show_alert=True)
                return
            target_key, keep = parsed

            lock = get_summary_lock()
            if lock.locked():
                await safe_answer_callback(
                    callback,
                    "⏳ یه اجرای دیگه در جریانه! یه کم صبر کن بعد دوباره بزن",
                    show_alert=True,
                )
                await safe_edit_message(
                    callback,
                    "⏳ <b>یه اجرای دیگه در جریانه!</b>\n\n"
                    "فعلاً یه خلاصه‌سازی دیگه داره انجام میشه؛ یه کم صبر کن و دوباره امتحان کن قربونت.",
                    build_back_to_group_kb(chat_id),
                )
                return

            await safe_answer_callback(callback, "🚀 اجرا شروع شد...")
            title = await _get_group_title(chat_id)

            async with lock:
                if target_key == "a":
                    await safe_edit_message(
                        callback,
                        f"⏳ <b>در حال شروع خلاصه‌سازی همه تاپیک‌های «{escape_html(title)}»...</b>",
                    )

                    async def _on_progress(idx: int, total: int, topic_name: str) -> None:
                        await safe_edit_message(
                            callback,
                            f"⏳ <b>خلاصه‌سازی گروه «{escape_html(title)}»</b>\n\n"
                            f"در حال پردازش تاپیک <b>{idx} از {total}</b>:\n"
                            f"📂 {escape_html(topic_name)}...",
                        )

                    try:
                        report = await run_summary_for_group(
                            bot=bot,
                            chat_id=chat_id,
                            mark_summarized=not keep,
                            is_nightly=False,
                            progress_callback=_on_progress,
                        )
                        mode_note = " (حالت keep)" if keep else ""
                        result_text = (
                            f"🏁 <b>گزارش اجرای دستی «{escape_html(title)}»{mode_note}</b>\n\n"
                            f"✅ موفق: <b>{report.get('success', 0)}</b> تاپیک\n"
                            f"⏭ کمتر از حد نصاب: <b>{report.get('skipped', 0)}</b> تاپیک\n"
                            f"❌ ناموفق: <b>{report.get('failed', 0)}</b> تاپیک"
                        )
                    except Exception as exc:
                        logger.error("Panel manual group run failed: %s", exc, exc_info=True)
                        result_text = (
                            f"❌ <b>خطا در اجرای خلاصه‌سازی گروه «{escape_html(title)}»</b>\n\n"
                            f"<code>{escape_html(exc)}</code>"
                        )

                    await safe_edit_message(callback, result_text, build_back_to_group_kb(chat_id))
                    return

                # Single topic manual run
                tid = int(target_key)
                t_name = await _get_topic_display_name(chat_id, tid)
                await safe_edit_message(
                    callback,
                    f"⏳ <b>در حال خلاصه‌سازی تاپیک «{escape_html(t_name)}»...</b>",
                )
                try:
                    success, details = await run_summary_for_topic(
                        bot=bot,
                        chat_id=chat_id,
                        thread_id=tid,
                        mark_summarized=not keep,
                        is_nightly=False,
                    )
                    if success:
                        mode_note = " (پیام‌ها در دیتابیس حفظ شدن)" if keep else ""
                        result_text = (
                            f"✅ <b>خلاصه تاپیک «{escape_html(t_name)}» ارسال شد!</b>{mode_note}"
                        )
                    else:
                        result_text = (
                            f"ℹ️ <b>خلاصه‌سازی تاپیک «{escape_html(t_name)}» انجام نشد:</b>\n\n"
                            f"{escape_html(details)}"
                        )
                except Exception as exc:
                    logger.error("Panel manual topic run failed: %s", exc, exc_info=True)
                    result_text = (
                        f"❌ <b>خطا در خلاصه‌سازی تاپیک «{escape_html(t_name)}»:</b>\n\n"
                        f"<code>{escape_html(exc)}</code>"
                    )

                await safe_edit_message(callback, result_text, build_back_to_group_kb(chat_id))
                return

        if action == "prv":
            text, kb = await render_preview_topics(chat_id, page=page)
            await safe_edit_message(callback, text, kb)
            await safe_answer_callback(callback)
            return

        if action == "prvr":
            try:
                tid = int(arg)
            except ValueError:
                await safe_answer_callback(callback, "شناسه تاپیک معتبر نیست 😅", show_alert=True)
                return

            t_name = await _get_topic_display_name(chat_id, tid)
            await safe_answer_callback(callback, "👁 در حال ساخت پیش‌نمایش...")
            await safe_edit_message(
                callback,
                f"⏳ <b>در حال آماده‌سازی پیش‌نمایش تاپیک «{escape_html(t_name)}»...</b>\n"
                "الان فقط توی پیوی خودت می‌فرستمش (هیچ پیامی هم پاک نمیشه) 👇",
            )
            try:
                success, details = await run_summary_for_topic(
                    bot=bot,
                    chat_id=chat_id,
                    thread_id=tid,
                    mark_summarized=False,
                    is_nightly=False,
                    destination_override=(user.id, None),
                    dry_run=True,
                )
                if success:
                    result_text = (
                        f"✅ <b>پیش‌نمایش تاپیک «{escape_html(t_name)}» برات فرستاده شد!</b>\n"
                        "<i>توی گروه چیزی ارسال نشد و دیتابیس هم اصلاً تغییر نکرد.</i>"
                    )
                else:
                    result_text = (
                        f"ℹ️ <b>پیش‌نمایش تاپیک «{escape_html(t_name)}» ساخته نشد:</b>\n\n"
                        f"{escape_html(details)}"
                    )
            except Exception as exc:
                logger.error("Panel preview failed: %s", exc, exc_info=True)
                result_text = (
                    f"❌ <b>خطا در ساخت پیش‌نمایش تاپیک «{escape_html(t_name)}»:</b>\n\n"
                    f"<code>{escape_html(exc)}</code>"
                )

            await safe_edit_message(callback, result_text, build_back_to_group_kb(chat_id))
            return

    except Exception as exc:
        logger.error("Unhandled error in panel callback handler: %s", exc, exc_info=True)
        await safe_answer_callback(
            callback,
            "یه خطای غیرمنتظره پیش اومد 😅 دوباره امتحان کن!",
            show_alert=True,
        )


@router.message(StateFilter(PanelStates.waiting_min_messages))
async def handle_min_messages_fsm_input(
    message: Message,
    bot: Bot,
    state: FSMContext,
) -> None:
    """Process manual min-messages numeric input (1-100, ASCII or Persian digits)."""
    try:
        user = message.from_user
        if user is None:
            return

        data = await state.get_data()
        chat_id = data.get("chat_id")
        if chat_id is None:
            await state.clear()
            await message.answer(
                "نشست ویرایش منقضی شده 😅 با /panel دوباره بازش کن.",
                parse_mode="HTML",
            )
            return

        if not await can_manage_group(bot, user.id, int(chat_id)):
            await state.clear()
            name = user.first_name or user.full_name or "دوست عزیز"
            denied_text, _ = pick("callback_denied", user_id=user.id, name=name, escape=True)
            await message.answer(denied_text, parse_mode="HTML")
            return

        parsed = parse_min_messages_input(message.text)
        if parsed is None:
            await message.answer(
                "⚠️ این عدد معتبر نیست! لطفاً یه عدد صحیح بین <b>۱ تا ۱۰۰</b> بفرست (فارسی یا انگلیسی).\n"
                "برای لغو هم <code>/cancel</code> رو بزن.",
                parse_mode="HTML",
            )
            return

        await repository.ensure_group_settings(int(chat_id))
        await repository.set_group_min_messages(int(chat_id), parsed)
        await state.clear()

        text, kb = await render_min_messages(int(chat_id))
        await message.answer(
            f"✅ حداقل تعداد پیام روی <b>{parsed}</b> تنظیم شد!\n\n{text}",
            reply_markup=kb,
            parse_mode="HTML",
            disable_web_page_preview=True,
        )
    except Exception as exc:
        logger.error("Error handling min_messages FSM input: %s", exc, exc_info=True)
        await state.clear()
        try:
            await message.answer(
                "❌ یه مشکلی توی ذخیره عدد پیش اومد 😅 دوباره با /panel امتحان کن.",
                parse_mode="HTML",
            )
        except Exception:
            pass


@router.callback_query()
async def handle_stale_callback(callback: CallbackQuery) -> None:
    """Gracefully handle unknown or stale callbacks without crashing."""
    await safe_answer_callback(
        callback,
        "این دکمه قدیمی یا نامعتبره 😅 با /panel یه پنل تازه باز کن",
        show_alert=True,
    )

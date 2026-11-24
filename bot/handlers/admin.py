"""Admin command handlers for managing group topics, settings, and manual runs."""

from datetime import datetime, timedelta
import logging
from zoneinfo import ZoneInfo
from aiogram import Router, F
from aiogram.filters import Command
from aiogram.types import Message
from bot.config import get_settings
from bot.database import repository
from bot.filters import AdminFilter, ForumGroupFilter
from bot.scheduler.jobs import (
    get_summary_lock,
    run_summary_for_group,
    run_summary_for_topic,
)
from bot.services.morning_service import send_morning_greeting_to_group
from bot.utils.dates import (
    format_persian_date,
    format_persian_short_date,
    get_tehran_now,
)
from bot.utils.general_topic import resolve_thread_id, to_send_thread_id
from bot.utils.html_utils import escape_html
from bot.utils.link_parser import parse_topic_link
from bot.utils.text_split import smart_split

logger = logging.getLogger(__name__)

router = Router(name="admin_router")
# Apply ForumGroupFilter at router level so only allowed supergroups evaluate handlers
router.message.filter(ForumGroupFilter())


async def try_delete_command(message: Message) -> None:
    """Best-effort deletion of the trigger command message."""
    try:
        await message.delete()
    except Exception:
        pass


async def send_topic_reply(message: Message, text: str) -> Message | None:
    """Send an HTML reply into the exact topic where the command was issued."""
    if not message.chat or not message.bot:
        return None
    thread_id = resolve_thread_id(message)
    target_thread_id = to_send_thread_id(thread_id)
    try:
        return await message.bot.send_message(
            chat_id=message.chat.id,
            message_thread_id=target_thread_id,
            text=text,
            parse_mode="HTML",
            disable_web_page_preview=True,
        )
    except Exception as exc:
        logger.error("Failed sending admin topic reply: %s", exc)
        return None


@router.message(Command("add_topic"), AdminFilter())
async def cmd_add_topic(message: Message) -> None:
    """Add current topic to active monitoring."""
    if not message.chat:
        return
    await try_delete_command(message)

    thread_id = resolve_thread_id(message)
    chat_id = message.chat.id

    topic_name = await repository.get_topic_name(chat_id, thread_id)
    if not topic_name:
        topic_name = "عمومی" if thread_id == 1 else f"تاپیک {thread_id}"

    await repository.set_topic_active_status(chat_id, thread_id, is_active=1, name=topic_name)
    await send_topic_reply(
        message,
        f"✅ تاپیک <b>«{escape_html(topic_name)}»</b> به لیست تاپیک‌های تحت نظارت اضافه و فعال شد.",
    )


@router.message(Command("remove_topic"), AdminFilter())
async def cmd_remove_topic(message: Message) -> None:
    """Remove current topic from active monitoring."""
    if not message.chat:
        return
    await try_delete_command(message)

    thread_id = resolve_thread_id(message)
    chat_id = message.chat.id

    topic_name = await repository.get_topic_name(chat_id, thread_id)
    if not topic_name:
        topic_name = "عمومی" if thread_id == 1 else f"تاپیک {thread_id}"

    await repository.set_topic_active_status(chat_id, thread_id, is_active=0, name=topic_name)
    await send_topic_reply(
        message,
        f"⛔ تاپیک <b>«{escape_html(topic_name)}»</b> از نظارت خارج و غیرفعال شد.",
    )


@router.message(Command("topics"), AdminFilter())
async def cmd_topics(message: Message) -> None:
    """List all known topics with monitoring status and today's message counts."""
    if not message.chat:
        return
    await try_delete_command(message)

    chat_id = message.chat.id
    settings = await repository.ensure_group_settings(chat_id)
    mode = settings.get("mode", "blacklist")
    mode_fa = "لیست سیاه (همه به جز حذفی‌ها)" if mode == "blacklist" else "لیست سفید (فقط افزوده‌شده‌ها)"

    # Compute start of today in Tehran converted to UTC ISO format
    tehran_now = get_tehran_now()
    tehran_start_of_day = tehran_now.replace(hour=0, minute=0, second=0, microsecond=0)
    utc_start_str = tehran_start_of_day.astimezone(ZoneInfo("UTC")).isoformat()

    topics = await repository.list_topics_with_counts_today(chat_id, utc_start_str)
    if not topics:
        await send_topic_reply(
            message,
            f"📋 <b>لیست تاپیک‌ها:</b>\n\nهنوز هیچ تاپیکی در پایگاه داده ثبت نشده است.\n"
            f"حالت جاری: <b>{mode_fa}</b>",
        )
        return

    lines = [
        f"📋 <b>وضعیت تاپیک‌های گروه</b> (حالت: <b>{mode_fa}</b>):\n"
    ]
    for t in topics:
        name = t["name"]
        tid = t["thread_id"]
        status_icon = "✅ فعال" if t["is_monitored"] else "⛔ غیرفعال"
        count = t["today_count"]
        lines.append(f"• <b>{escape_html(name)}</b> (شناسه: <code>{tid}</code>): {status_icon} — <i>{count} پیام امروز</i>")

    for part in smart_split("\n".join(lines), limit=4000):
        await send_topic_reply(message, part)


@router.message(Command("status"), AdminFilter())
async def cmd_status(message: Message) -> None:
    """Show detailed system status and configurations."""
    if not message.chat:
        return
    await try_delete_command(message)

    chat_id = message.chat.id
    cfg = get_settings()
    group_cfg = await repository.ensure_group_settings(chat_id, cfg.MIN_MESSAGES_PER_TOPIC)

    is_enabled = group_cfg.get("enabled", 1) == 1
    status_str = "🟢 فعال" if is_enabled else "🔴 غیرفعال"

    mode = group_cfg.get("mode", "blacklist")
    mode_str = "لیست سیاه (Blacklist)" if mode == "blacklist" else "لیست سفید (Whitelist)"

    dest_mode = group_cfg.get("destination_mode", "per_topic")
    if dest_mode == "aggregate":
        dest_tid = group_cfg.get("destination_thread_id")
        if dest_tid is not None:
            dest_name = (await repository.get_topic_name(chat_id, dest_tid)) or (
                "عمومی" if dest_tid == 1 else f"تاپیک {dest_tid}"
            )
        else:
            dest_name = "نامشخص"
        dest_str = f"تجمیعی در تاپیک «{escape_html(dest_name)}» (<code>{dest_tid}</code>)"
    else:
        dest_str = "ارسال در همان تاپیک (Per Topic)"

    min_msgs = group_cfg.get("min_messages", cfg.MIN_MESSAGES_PER_TOPIC)
    unsummarized_count = await repository.count_unsummarized_messages(chat_id)

    # Calculate next scheduled run time
    tehran_now = get_tehran_now()
    target_time = tehran_now.replace(
        hour=cfg.SUMMARY_TIME_HOUR,
        minute=cfg.SUMMARY_TIME_MINUTE,
        second=0,
        microsecond=0,
    )
    if target_time <= tehran_now:
        target_time += timedelta(days=1)

    next_run_str = (
        f"{format_persian_short_date(target_time)} ساعت "
        f"{target_time.strftime('%H:%M')} (تهران)"
    )

    ai_info = (
        f"Google Gemini ({escape_html(cfg.GEMINI_MODEL)})"
        if cfg.AI_PROVIDER == "gemini"
        else f"OpenAI ({escape_html(cfg.OPENAI_MODEL or 'نامشخص')})"
    )

    text = (
        f"⚙️ <b>وضعیت ربات خلاصه‌ساز:</b>\n\n"
        f"🔹 <b>وضعیت کلی:</b> {status_str}\n"
        f"🔹 <b>حالت پایش تاپیک‌ها:</b> {mode_str}\n"
        f"🔹 <b>مقصد ارسال خلاصه:</b> {dest_str}\n"
        f"🔹 <b>حداقل پیام برای خلاصه:</b> {min_msgs} پیام\n"
        f"🔹 <b>پیام‌های تلخیص‌نشده فعلی:</b> {unsummarized_count} پیام\n"
        f"🔹 <b>زمان اجرای شبانه بعدی:</b> {next_run_str}\n"
        f"🔹 <b>سرویس هوش مصنوعی:</b> {ai_info}\n"
    )
    await send_topic_reply(message, text)


@router.message(Command("mode"), AdminFilter())
async def cmd_mode(message: Message) -> None:
    """Switch monitoring mode between whitelist and blacklist."""
    if not message.chat:
        return
    await try_delete_command(message)

    args = (message.text or "").split()[1:]
    if not args or args[0].lower() not in ("whitelist", "blacklist"):
        await send_topic_reply(
            message,
            "⚠️ لطفاً حالت را مشخص کنید:\n"
            "<code>/mode whitelist</code> (فقط تاپیک‌های افزوده‌شده)\n"
            "یا\n"
            "<code>/mode blacklist</code> (تمام تاپیک‌ها به جز موارد حذفی)",
        )
        return

    new_mode = args[0].lower()
    await repository.set_group_mode(message.chat.id, new_mode)
    mode_fa = "لیست سفید (Whitelist)" if new_mode == "whitelist" else "لیست سیاه (Blacklist)"
    await send_topic_reply(
        message,
        f"⚙️ حالت پایش تاپیک‌های این گروه به <b>{mode_fa}</b> تغییر یافت.",
    )


@router.message(Command("set_min"), AdminFilter())
async def cmd_set_min(message: Message) -> None:
    """Set minimum messages threshold (1-100) to trigger summarization."""
    if not message.chat:
        return
    await try_delete_command(message)

    chat_id = message.chat.id
    args = (message.text or "").split()[1:]

    if not args:
        await send_topic_reply(
            message,
            "⚠️ لطفاً تعداد حداقل پیام را بین ۱ تا ۱۰۰ مشخص کنید:\n"
            "<code>/set_min 5</code>",
        )
        return

    try:
        min_val = int(args[0].strip())
        if not (1 <= min_val <= 100):
            raise ValueError()
    except ValueError:
        await send_topic_reply(
            message,
            "⚠️ عدد وارد شده نامعتبر است. لطفاً یک عدد صحیح بین <b>۱ تا ۱۰۰</b> وارد کنید:\n"
            "<code>/set_min 5</code>",
        )
        return

    await repository.ensure_group_settings(chat_id)
    await repository.set_group_min_messages(chat_id, min_val)
    await send_topic_reply(
        message,
        f"🔢 حداقل تعداد پیام برای خلاصه‌سازی تاپیک‌های این گروه به <b>{min_val}</b> پیام تنظیم شد.",
    )


@router.message(Command("set_destination"), AdminFilter())
async def cmd_set_destination(message: Message) -> None:
    """Set current topic or specified topic link/ID as aggregate summary destination."""
    if not message.chat:
        return
    await try_delete_command(message)

    chat_id = message.chat.id
    args = (message.text or "").split()[1:]

    if args:
        raw_arg = args[0].strip()
        _, thread_id, error = parse_topic_link(raw_arg, current_chat_id=chat_id)
        if error or thread_id is None:
            await send_topic_reply(
                message,
                f"⚠️ <b>خطا در تنظیم مقصد خلاصه:</b>\n\n{escape_html(error or 'شناسه تاپیک معتبر نیست.')}",
            )
            return
    else:
        thread_id = resolve_thread_id(message)

    topic_name = await repository.get_topic_name(chat_id, thread_id)
    if not topic_name:
        topic_name = "عمومی" if thread_id == 1 else f"تاپیک {thread_id}"

    await repository.set_group_destination(chat_id, "aggregate", thread_id)
    await send_topic_reply(
        message,
        f"📌 مقصد ارسال تمامی خلاصه‌های روزانه روی تاپیک <b>«{escape_html(topic_name)}»</b> (شناسه: <code>{thread_id}</code>) تنظیم شد.",
    )


@router.message(Command("reset_destination"), AdminFilter())
async def cmd_reset_destination(message: Message) -> None:
    """Reset summary destination back to per_topic."""
    if not message.chat:
        return
    await try_delete_command(message)

    await repository.set_group_destination(message.chat.id, "per_topic", None)
    await send_topic_reply(
        message,
        "🔄 مقصد خلاصه‌ها به حالت پیش‌فرض (<b>ارسال در هر تاپیک به صورت مجزا</b>) بازگردانده شد.",
    )


@router.message(Command("summary_on"), AdminFilter())
async def cmd_summary_on(message: Message) -> None:
    """Enable summary system for this group."""
    if not message.chat:
        return
    await try_delete_command(message)

    await repository.set_group_enabled(message.chat.id, 1)
    await send_topic_reply(
        message,
        "🟢 سیستم خلاصه‌سازی روزانه برای این گروه <b>فعال</b> شد.",
    )


@router.message(Command("summary_off"), AdminFilter())
async def cmd_summary_off(message: Message) -> None:
    """Disable summary system for this group."""
    if not message.chat:
        return
    await try_delete_command(message)

    await repository.set_group_enabled(message.chat.id, 0)
    await send_topic_reply(
        message,
        "🔴 سیستم خلاصه‌سازی روزانه برای این گروه <b>غیرفعال</b> شد. (پیام‌های جدید ثبت نمی‌شوند و اجرای شبانه انجام نمی‌گیرد)",
    )


@router.message(Command("summarize_now"), AdminFilter())
async def cmd_summarize_now(message: Message) -> None:
    """Manually trigger summarization for the current topic."""
    if not message.chat or not message.bot:
        return
    await try_delete_command(message)

    thread_id = resolve_thread_id(message)
    chat_id = message.chat.id

    args = (message.text or "").split()[1:]
    keep_mode = any(arg.lower() == "keep" for arg in args)

    lock = get_summary_lock()
    if lock.locked():
        await send_topic_reply(message, "اجرای دیگری در حال انجام است.")
        return

    async with lock:
        progress_msg = await send_topic_reply(
            message,
            "⏳ در حال پردازش و استخراج پیام‌ها برای خلاصه‌سازی این تاپیک...",
        )

        try:
            success, details = await run_summary_for_topic(
                bot=message.bot,
                chat_id=chat_id,
                thread_id=thread_id,
                mark_summarized=not keep_mode,
                is_nightly=False,
            )
            if success:
                mode_note = " (پیام‌ها بدون علامت‌گذاری باقی ماندند)" if keep_mode else ""
                await send_topic_reply(
                    message,
                    f"✅ خلاصه تاپیک با موفقیت تولید و ارسال شد.{mode_note}",
                )
            else:
                await send_topic_reply(
                    message,
                    f"ℹ️ خلاصه‌سازی انجام نشد: {escape_html(details)}",
                )
        except Exception as exc:
            logger.error("Error during manual topic summary: %s", exc, exc_info=True)
            await send_topic_reply(
                message,
                f"❌ بروز خطا در فرآیند خلاصه‌سازی تاپیک: <code>{escape_html(exc)}</code>",
            )
        finally:
            if progress_msg:
                try:
                    await progress_msg.delete()
                except Exception:
                    pass


@router.message(Command("summarize_all"), AdminFilter())
async def cmd_summarize_all(message: Message) -> None:
    """Manually trigger summarization for all monitored topics in the group."""
    if not message.chat or not message.bot:
        return
    await try_delete_command(message)

    chat_id = message.chat.id
    args = (message.text or "").split()[1:]
    keep_mode = any(arg.lower() == "keep" for arg in args)

    lock = get_summary_lock()
    if lock.locked():
        await send_topic_reply(message, "اجرای دیگری در حال انجام است.")
        return

    async with lock:
        progress_msg = await send_topic_reply(
            message,
            "⏳ در حال شروع خلاصه‌سازی دستی تمامی تاپیک‌های فعال گروه...",
        )

        try:
            report = await run_summary_for_group(
                bot=message.bot,
                chat_id=chat_id,
                mark_summarized=not keep_mode,
                is_nightly=False,
            )
            mode_note = " [حالت آزمایشی keep فعال]" if keep_mode else ""
            await send_topic_reply(
                message,
                f"🏁 <b>گزارش خلاصه‌سازی گروه{mode_note}:</b>\n\n"
                f"✅ موفق: {report.get('success', 0)} تاپیک\n"
                f"⏭ کمتر از حد نصاب: {report.get('skipped', 0)} تاپیک\n"
                f"❌ ناموفق: {report.get('failed', 0)} تاپیک",
            )
        except Exception as exc:
            logger.error("Error during manual group summary: %s", exc, exc_info=True)
            await send_topic_reply(
                message,
                f"❌ بروز خطا در اجرای خلاصه‌سازی کل گروه: <code>{escape_html(exc)}</code>",
            )
        finally:
            if progress_msg:
                try:
                    await progress_msg.delete()
                except Exception:
                    pass


@router.message(Command("set_morning_topic"), AdminFilter())
async def cmd_set_morning_topic(message: Message) -> None:
    """Set current topic or given topic link/ID as destination for daily 05:01 morning greeting."""
    if not message.chat:
        return
    await try_delete_command(message)

    chat_id = message.chat.id
    args = (message.text or "").split()[1:]

    if args:
        raw_arg = args[0].strip()
        _, thread_id, error = parse_topic_link(raw_arg, current_chat_id=chat_id)
        if error or thread_id is None:
            await send_topic_reply(
                message,
                f"⚠️ <b>خطا در تنظیم تاپیک صبح‌بخیر:</b>\n\n{escape_html(error or 'شناسه تاپیک معتبر نیست.')}",
            )
            return
    else:
        thread_id = resolve_thread_id(message)

    topic_name = await repository.get_topic_name(chat_id, thread_id)
    if not topic_name:
        topic_name = "عمومی" if thread_id == 1 else f"تاپیک {thread_id}"

    cfg = await repository.set_morning_topic(chat_id, thread_id, auto_enable_if_unset=True)
    status_note = "فعال ✅" if int(cfg.get("enabled", 0)) == 1 else "غیرفعال 🔴"
    settings = get_settings()
    await send_topic_reply(
        message,
        f"☀️ تاپیک مقصد پیام صبح‌بخیر روزانه هیلز (ساعت <code>{settings.MORNING_GREETING_HOUR:02d}:{settings.MORNING_GREETING_MINUTE:02d}</code> تهران) "
        f"روی <b>«{escape_html(topic_name)}»</b> (شناسه: <code>{thread_id}</code>) تنظیم شد.\n"
        f"وضعیت فعلی: <b>{status_note}</b>\n\n"
        "برای ارسال تستی فوری می‌توانید دستور <code>/morning_test</code> یا دکمه تست در <code>/panel</code> را بزنید.",
    )


@router.message(Command("morning_on"), AdminFilter())
async def cmd_morning_on(message: Message) -> None:
    """Enable daily 05:01 morning greeting for this group."""
    if not message.chat:
        return
    await try_delete_command(message)

    chat_id = message.chat.id
    cfg = await repository.ensure_morning_config(chat_id)
    if cfg.get("thread_id") is None:
        await send_topic_reply(
            message,
            "⚠️ هنوز تاپیک مقصد برای پیام صبح‌بخیر مشخص نشده است.\n"
            "ابتدا داخل تاپیک دلخواه دستور <code>/set_morning_topic</code> را بزنید یا از <code>/panel</code> انتخاب کنید.",
        )
        return

    await repository.set_morning_enabled(chat_id, 1)
    await send_topic_reply(message, "🟢 ارسال روزانه پیام صبح‌بخیر هیلز برای این گروه <b>فعال</b> شد.")


@router.message(Command("morning_off"), AdminFilter())
async def cmd_morning_off(message: Message) -> None:
    """Disable daily 05:01 morning greeting for this group."""
    if not message.chat:
        return
    await try_delete_command(message)

    await repository.set_morning_enabled(message.chat.id, 0)
    await send_topic_reply(message, "🔴 ارسال روزانه پیام صبح‌بخیر هیلز برای این گروه <b>غیرفعال</b> شد.")


@router.message(Command("morning_test"), AdminFilter())
async def cmd_morning_test(message: Message) -> None:
    """Admin-only test trigger sending a real morning greeting into the configured topic."""
    if not message.chat or not message.bot:
        return
    await try_delete_command(message)

    ok, details = await send_morning_greeting_to_group(
        bot=message.bot,
        chat_id=message.chat.id,
        is_test=True,
    )
    if not ok:
        await send_topic_reply(message, f"⚠️ {escape_html(details)}")


@router.message(Command("help"), AdminFilter())
async def cmd_help(message: Message) -> None:
    """Show Persian help with all available admin commands."""
    if not message.chat:
        return
    await try_delete_command(message)

    help_text = (
        "📖 <b>راهنمای دستورات مدیران ربات خلاصه‌ساز:</b>\n\n"
        "🔹 <code>/status</code> — مشاهده وضعیت کامل ربات، تاپیک‌ها و زمان اجرای بعدی\n"
        "🔹 <code>/topics</code> — فهرست تمامی تاپیک‌ها و تعداد پیام‌های ذخیره‌شده امروز\n"
        "🔹 <code>/add_topic</code> — افزودن تاپیک فعلی به پایش خلاصه‌سازی\n"
        "🔹 <code>/remove_topic</code> — خروج تاپیک فعلی از پایش خلاصه‌سازی\n"
        "🔹 <code>/mode whitelist|blacklist</code> — تغییر حالت پایش به لیست سفید یا سیاه\n"
        "🔹 <code>/set_min 1-100</code> — تنظیم حداقل تعداد پیام لازم برای خلاصه‌سازی هر تاپیک\n"
        "🔹 <code>/set_destination [لینک|شناسه]</code> — تنظیم تاپیک مقصد تجمیعی خلاصه‌ها (با ارسال در همان تاپیک یا ارسال لینک تاپیک)\n"
        "🔹 <code>/reset_destination</code> — بازگرداندن مقصد خلاصه‌ها به هر تاپیک مجزا\n"
        "🔹 <code>/summary_on</code> — فعال‌سازی موقت سیستم خلاصه‌سازی برای گروه\n"
        "🔹 <code>/summary_off</code> — غیرفعال‌سازی سیستم خلاصه‌سازی برای گروه\n"
        "🔹 <code>/summarize_now [keep]</code> — اجرای دستی خلاصه‌سازی برای تاپیک فعلی\n"
        "🔹 <code>/summarize_all [keep]</code> — اجرای دستی خلاصه‌سازی برای همه تاپیک‌های گروه\n"
        "🔹 <code>/set_morning_topic [لینک|شناسه]</code> — تعیین تاپیک مقصد پیام صبح‌بخیر روزانه هیلز (۰۵:۰۱)\n"
        "🔹 <code>/morning_on</code> / <code>/morning_off</code> — فعال یا غیرفعال کردن پیام صبح‌بخیر روزانه\n"
        "🔹 <code>/morning_test</code> — ارسال تستی پیام صبح‌بخیر در تاپیک مقصد (مخصوص ادمین)\n"
        "🔹 <code>/deeptalk</code> — پنل برگزاری و مدیریت جلسات دیپ‌تاک (مخصوص مالک ربات و ادمین‌های دیپ‌تاک)\n"
        "🔹 <code>/set_deeptalk_topic [لینک|شناسه]</code> — تعیین تاپیک پیش‌فرض دیپ‌تاک\n"
        "🔹 <code>/help</code> — نمایش همین پیام راهنما\n\n"
        "⏱ <b>تایمر و یادآور (برای همه اعضا):</b> هر عضو گروه می‌تواند با دستور <code>/timer</code> (یا میانبر <code>/timer 30m متن</code>) تایمر دقیقه‌ای/ساعتی یا تاریخی تنظیم کند تا رأس زمان مقرر در همان تاپیک و پیوی ربات به او یادآوری شود.\n\n"
        "🎛 <b>پنل مدیریتی در پیوی:</b> مدیران می‌توانند در پیوی ربات با دستور <code>/panel</code> تمام تنظیمات، تاپیک‌ها، پیش‌نمایش و اجرای دستی را به صورت دکمه‌ای مدیریت کنند.\n\n"
        "<i>نکته: پارامتر <code>keep</code> در دستورات دستی مانع از حذف پیام‌ها از دیتابیس می‌شود (مناسب تست). در حالت عادی، پس از خلاصه‌سازی موفق پیام‌ها از دیتابیس پاک می‌شوند تا فضا اشغال نشود.</i>"
    )
    await send_topic_reply(message, help_text)

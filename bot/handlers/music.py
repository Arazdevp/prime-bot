"""Private-chat Music Downloader handlers (/music, direct message link paste, and track list)."""

import logging
from aiogram import Bot, Router
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import BaseFilter, Command, StateFilter
from aiogram.fsm.state import default_state
from aiogram.types import CallbackQuery, InlineKeyboardMarkup, Message
from bot.database import repository
from bot.filters import PrivateChatFilter
from bot.handlers.private import handle_stranger_contact
from bot.keyboards.music import (
    MUSIC_PAGE_SIZE,
    MusicCB,
    build_music_home_kb,
    build_music_list_kb,
)
from bot.keyboards.panel import paginate
from bot.services.access import (
    can_manage_deeptalk,
    get_deeptalk_groups,
    is_active_group_member,
    is_global_admin,
    is_user_in_specific_group,
)
from bot.services.music_service import send_music_track_to_user
from bot.utils.link_parser import parse_message_link
from bot.utils.timer_parser import to_persian_digits

logger = logging.getLogger(__name__)

router = Router(name="music_router")
router.message.filter(PrivateChatFilter())
router.callback_query.filter(PrivateChatFilter())


class ContainsTelegramMessageLinkFilter(BaseFilter):
    """Match messages whose text or caption contains a parseable Telegram message link."""

    async def __call__(self, message: Message) -> bool:
        raw = message.text or message.caption or ""
        return parse_message_link(raw) is not None


async def _is_authorized_music_user(bot: Bot, user_id: int) -> bool:
    """Return True if user_id is a global admin, DeepTalk admin, or active allowed-group member."""
    if is_global_admin(user_id):
        return True
    if await can_manage_deeptalk(user_id):
        return True
    return await is_active_group_member(bot, user_id)


async def _get_user_member_chat_ids(bot: Bot, user_id: int) -> list[int]:
    """Return active allowed group chat_ids that user_id has access to."""
    groups = await get_deeptalk_groups()
    if not groups:
        return []
    if is_global_admin(user_id):
        return [int(g["chat_id"]) for g in groups]

    allowed_ids: list[int] = []
    for g in groups:
        cid = int(g["chat_id"])
        if await is_user_in_specific_group(bot, cid, user_id):
            allowed_ids.append(cid)
    return allowed_ids


async def _render_music_home(bot: Bot, user_id: int) -> tuple[str, InlineKeyboardMarkup]:
    """Render the Music Downloader home screen text and keyboard."""
    chat_ids = await _get_user_member_chat_ids(bot, user_id)
    total_tracks = await repository.count_music_tracks(chat_ids=chat_ids) if chat_ids else 0

    count_line = (
        f"📀 تعداد موزیک‌های ثبت‌شده تا الان: <b>{to_persian_digits(total_tracks)}</b> ترک\n\n"
        if total_tracks > 0
        else "📀 هنوز موزیک جدیدی بعد از فعال شدن این قابلیت ثبت نشده.\n\n"
    )

    text = (
        "🎵 <b>دانلودر موزیک گروه</b>\n\n"
        "چون ذخیره و فوروارد فایل‌ها توی گروه بسته است، کافیه:\n"
        "۱️⃣ روی موزیکِ داخل تاپیک موزیک کلیک کنی و <b>Copy Link</b> رو بزنی\n"
        "۲️⃣ همون لینک رو مستقیم همینجا برام بفرستی 🔗\n\n"
        "من همون لحظه فایل اصلی رو بدون محدودیت برات می‌فرستم تا راحت سیوش کنی! 💾✨\n\n"
        f"{count_line}"
        "<i>می‌تونی از دکمه زیر هم لیست آخرین موزیک‌های ثبت‌شده رو ببینی 👇</i>"
    )
    return text, build_music_home_kb(has_tracks=total_tracks > 0)


async def _safe_edit_music_message(
    callback: CallbackQuery,
    text: str,
    reply_markup: InlineKeyboardMarkup | None = None,
) -> None:
    """Safely edit a Music Downloader callback message."""
    msg = callback.message
    if msg is None or not hasattr(msg, "edit_text"):
        return
    try:
        await msg.edit_text(
            text=text,
            reply_markup=reply_markup,
            parse_mode="HTML",
            disable_web_page_preview=True,
        )
    except TelegramBadRequest as exc:
        if "message is not modified" not in str(exc).lower():
            logger.warning("TelegramBadRequest editing music message: %s", exc)
    except Exception as exc:
        logger.warning("Error editing music message: %s", exc)


@router.message(Command("music"))
async def cmd_private_music(message: Message, bot: Bot) -> None:
    """Show the Music Downloader guide and recent tracks entry in private chat."""
    try:
        user = message.from_user
        if user is None:
            return

        if not await _is_authorized_music_user(bot, user.id):
            await handle_stranger_contact(message, bot, is_start=False)
            return

        text, kb = await _render_music_home(bot, user.id)
        await message.answer(
            text=text,
            reply_markup=kb,
            parse_mode="HTML",
            disable_web_page_preview=True,
        )
    except Exception as exc:
        logger.error("Error in private /music command: %s", exc, exc_info=True)
        await message.answer("یه خطای کوچیک پیش اومد 😅 دوباره /music رو بزن.", parse_mode="HTML")


@router.message(StateFilter(default_state), ContainsTelegramMessageLinkFilter())
async def handle_music_link_message(message: Message, bot: Bot) -> None:
    """Handle pasted Telegram message links in private chat and deliver matching music files."""
    try:
        user = message.from_user
        if user is None:
            return

        parsed = parse_message_link(message.text or message.caption or "")
        if parsed is None:
            return

        track = None
        if parsed.chat_id is not None:
            has_access = is_global_admin(user.id) or await is_user_in_specific_group(
                bot, parsed.chat_id, user.id
            )
            if not has_access:
                if not await _is_authorized_music_user(bot, user.id):
                    await handle_stranger_contact(message, bot, is_start=False)
                    return
                await message.answer(
                    "⛔ این لینک متعلق به گروهی است که عضو آن نیستی.",
                    parse_mode="HTML",
                )
                return

            track = await repository.get_music_track_by_chat_and_message(
                parsed.chat_id, parsed.message_id
            )
        else:
            member_chat_ids = await _get_user_member_chat_ids(bot, user.id)
            if not member_chat_ids:
                await handle_stranger_contact(message, bot, is_start=False)
                return

            track = await repository.get_music_track_by_message_id_in_chats(
                parsed.message_id, member_chat_ids
            )

        if track is None:
            member_chat_ids = await _get_user_member_chat_ids(bot, user.id)
            has_tracks = (
                await repository.count_music_tracks(chat_ids=member_chat_ids) > 0
                if member_chat_ids
                else False
            )
            await message.answer(
                "🔍 <b>این موزیک توی آرشیوم پیدا نشد!</b>\n\n"
                "دلیلش می‌تونه یکی از اینا باشه:\n"
                "• این پیام <b>قبل از فعال شدن دانلودر موزیک</b> توی گروه فرستاده شده (ربات تلگرام به پیام‌های قبل از نصب دسترسی نداره)\n"
                "• پیام لینک‌شده فایل صوتی/موزیک نبوده یا توی تاپیک موزیک ارسال نشده\n\n"
                "<i>از این به بعد هر موزیکی توی تاپیک موزیک بیاد رو بلافاصله ذخیره می‌کنم! 🎧</i>",
                reply_markup=build_music_home_kb(has_tracks=has_tracks) if has_tracks else None,
                parse_mode="HTML",
                disable_web_page_preview=True,
            )
            return

        ok, err_msg = await send_music_track_to_user(bot, user.id, track)
        if not ok and err_msg:
            await message.answer(err_msg, parse_mode="HTML")
    except Exception as exc:
        logger.error("Error handling music link in private chat: %s", exc, exc_info=True)
        await message.answer(
            "❌ یه خطای غیرمنتظره موقع پیدا کردن موزیک پیش اومد 😅 دوباره امتحان کن.",
            parse_mode="HTML",
        )


@router.callback_query(MusicCB.filter())
async def handle_music_callback(
    callback: CallbackQuery,
    callback_data: MusicCB,
    bot: Bot,
) -> None:
    """Handle Music Downloader inline keyboard callbacks (home, paginated list, download)."""
    try:
        user = callback.from_user
        if user is None:
            await callback.answer()
            return

        if callback_data.action == "noop":
            await callback.answer()
            return

        if not await _is_authorized_music_user(bot, user.id):
            await callback.answer("⛔ دسترسی نداری.", show_alert=True)
            return

        if callback_data.action == "home":
            text, kb = await _render_music_home(bot, user.id)
            await _safe_edit_music_message(callback, text, kb)
            await callback.answer()
            return

        if callback_data.action == "list":
            chat_ids = await _get_user_member_chat_ids(bot, user.id)
            all_tracks = (
                await repository.list_recent_music_tracks(chat_ids, limit=100)
                if chat_ids
                else []
            )
            if not all_tracks:
                text, kb = await _render_music_home(bot, user.id)
                await _safe_edit_music_message(callback, text, kb)
                await callback.answer("هنوز موزیکی ثبت نشده 😅")
                return

            page_tracks, safe_page, total_pages = paginate(
                all_tracks, callback_data.page, MUSIC_PAGE_SIZE
            )
            kb = build_music_list_kb(page_tracks, safe_page, total_pages)
            text = (
                "🎶 <b>آخرین موزیک‌های تاپیک موزیک</b>\n\n"
                f"تعداد کل ثبت‌شده: <b>{to_persian_digits(len(all_tracks))}</b> ترک\n"
                "روی هر کدوم بزنی همون لحظه برات می‌فرستمش 👇"
            )
            await _safe_edit_music_message(callback, text, kb)
            await callback.answer()
            return

        if callback_data.action == "dl":
            try:
                track_id = int(callback_data.arg)
            except ValueError:
                await callback.answer("شناسه موزیک نامعتبر است 😅", show_alert=True)
                return

            track = await repository.get_music_track_by_id(track_id)
            if track is None:
                await callback.answer("این موزیک دیگه توی دیتابیس نیست 😅", show_alert=True)
                return

            track_chat_id = int(track["chat_id"])
            if not (
                is_global_admin(user.id)
                or await is_user_in_specific_group(bot, track_chat_id, user.id)
            ):
                await callback.answer("⛔ به موزیک‌های این گروه دسترسی نداری.", show_alert=True)
                return

            ok, err_msg = await send_music_track_to_user(bot, user.id, track)
            if ok:
                await callback.answer("🎵 ارسال شد!")
            else:
                await callback.answer(err_msg or "خطا در ارسال موزیک", show_alert=True)
            return

        await callback.answer("این دکمه نامعتبره 😅", show_alert=True)
    except Exception as exc:
        logger.error("Error in MusicCB callback handler: %s", exc, exc_info=True)
        try:
            await callback.answer("یه خطا پیش اومد 😅 دوباره امتحان کن.", show_alert=True)
        except Exception:
            pass

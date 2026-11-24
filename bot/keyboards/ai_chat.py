"""Inline keyboards and CallbackData schema for /manage_ai control panel."""

from aiogram.filters.callback_data import CallbackData
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup


class AiChatCB(CallbackData, prefix="ai"):
    """Compact CallbackData for /manage_ai panel interactions (always <= 64 bytes)."""

    action: str
    owner_id: int = 0


def build_manage_ai_kb(
    enabled: bool,
    has_custom_prompt: bool,
    owner_id: int,
    provider: str = "gemini",
    access_mode: str = "public",
) -> InlineKeyboardMarkup:
    """Build main inline keyboard for /manage_ai panel."""
    toggle_label = "🔴 خاموش کردن هوش مصنوعی" if enabled else "🟢 روشن کردن هوش مصنوعی"
    provider_label = (
        "🔄 تغییر سرویس به OpenAI"
        if provider == "gemini"
        else "🔄 تغییر سرویس به Gemini"
    )
    access_label = (
        "🔒 محدود به مدیر ربات"
        if access_mode == "public"
        else "👥 باز کردن برای همه (۲۰/۲ساعت)"
    )
    rows: list[list[InlineKeyboardButton]] = [
        [
            InlineKeyboardButton(
                text=toggle_label,
                callback_data=AiChatCB(action="toggle", owner_id=owner_id).pack(),
            )
        ],
        [
            InlineKeyboardButton(
                text=provider_label,
                callback_data=AiChatCB(action="prov", owner_id=owner_id).pack(),
            ),
            InlineKeyboardButton(
                text=access_label,
                callback_data=AiChatCB(action="acc", owner_id=owner_id).pack(),
            ),
        ],
        [
            InlineKeyboardButton(
                text="🧹 پاک کردن حافظه مکالمه",
                callback_data=AiChatCB(action="clear", owner_id=owner_id).pack(),
            ),
            InlineKeyboardButton(
                text="🔄 بروزرسانی",
                callback_data=AiChatCB(action="refresh", owner_id=owner_id).pack(),
            ),
        ],
        [
            InlineKeyboardButton(
                text="📝 پرامپت فعلی",
                callback_data=AiChatCB(action="viewp", owner_id=owner_id).pack(),
            ),
            InlineKeyboardButton(
                text="✏️ تغییر پرامپت",
                callback_data=AiChatCB(action="setp", owner_id=owner_id).pack(),
            ),
        ],
        [
            InlineKeyboardButton(
                text="🧠 یادگیری از گفتگوی آزاد",
                callback_data=AiChatCB(action="learn", owner_id=owner_id).pack(),
            ),
            InlineKeyboardButton(
                text="👥 حافظه اعضا",
                callback_data=AiChatCB(action="mem", owner_id=owner_id).pack(),
            ),
        ],
    ]
    if has_custom_prompt:
        rows.append(
            [
                InlineKeyboardButton(
                    text="♻️ بازگشت به پرامپت پیش‌فرض",
                    callback_data=AiChatCB(action="resetp", owner_id=owner_id).pack(),
                )
            ]
        )
    rows.append(
        [
            InlineKeyboardButton(
                text="✖️ بستن",
                callback_data=AiChatCB(action="close", owner_id=owner_id).pack(),
            )
        ]
    )
    return InlineKeyboardMarkup(inline_keyboard=rows)


def build_ai_memory_view_kb(owner_id: int, has_memory: bool) -> InlineKeyboardMarkup:
    """Build inline keyboard shown below the learned group & member memory view."""
    rows: list[list[InlineKeyboardButton]] = [
        [
            InlineKeyboardButton(
                text="🧠 یادگیری الان",
                callback_data=AiChatCB(action="learn", owner_id=owner_id).pack(),
            ),
            InlineKeyboardButton(
                text="🔙 بازگشت به پنل",
                callback_data=AiChatCB(action="refresh", owner_id=owner_id).pack(),
            ),
        ]
    ]
    if has_memory:
        rows.append(
            [
                InlineKeyboardButton(
                    text="🗑 پاک کردن حافظه بلندمدت",
                    callback_data=AiChatCB(action="clrmem", owner_id=owner_id).pack(),
                )
            ]
        )
    return InlineKeyboardMarkup(inline_keyboard=rows)


def build_ai_prompt_cancel_kb(owner_id: int) -> InlineKeyboardMarkup:
    """Build cancel button shown while waiting for custom AI prompt input."""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="❌ انصراف",
                    callback_data=AiChatCB(action="cancel", owner_id=owner_id).pack(),
                )
            ]
        ]
    )


def build_ai_prompt_view_kb(owner_id: int) -> InlineKeyboardMarkup:
    """Build back button shown below the current prompt preview."""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="✏️ تغییر پرامپت",
                    callback_data=AiChatCB(action="setp", owner_id=owner_id).pack(),
                ),
                InlineKeyboardButton(
                    text="🔙 بازگشت به پنل",
                    callback_data=AiChatCB(action="refresh", owner_id=owner_id).pack(),
                ),
            ]
        ]
    )

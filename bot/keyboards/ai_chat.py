"""Inline keyboards and CallbackData schema for /manage_ai control panel."""

from aiogram.filters.callback_data import CallbackData
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup


class AiChatCB(CallbackData, prefix="ai"):
    """Compact CallbackData for /manage_ai panel interactions (always <= 64 bytes)."""

    action: str
    owner_id: int = 0
    arg: str = ""
    page: int = 0


def build_manage_ai_kb(
    enabled: bool,
    has_custom_prompt: bool,
    owner_id: int,
    provider: str = "9router",
    access_mode: str = "public",
    gateway_label: str = "9Router",
    quota_limit: int = 20,
    window_hours: int = 2,
) -> InlineKeyboardMarkup:
    """Build main inline keyboard for /manage_ai panel."""
    toggle_label = "🔴 خاموش کردن هوش مصنوعی" if enabled else "🟢 روشن کردن هوش مصنوعی"
    provider_label = (
        f"🔄 تغییر سرویس به {gateway_label}"
        if provider == "gemini"
        else "🔄 تغییر سرویس به Gemini"
    )
    if access_mode == "public":
        access_label = "🔒 محدود به مدیر ربات"
    else:
        if quota_limit < 0:
            quota_tag = "نامحدود"
        elif quota_limit == 0:
            quota_tag = "سهمیه بسته"
        else:
            quota_tag = f"{quota_limit}/{window_hours}ساعت"
        access_label = f"👥 باز کردن برای همه ({quota_tag})"

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
                text="📊 مدیریت سهمیه کاربران",
                callback_data=AiChatCB(action="quota", owner_id=owner_id).pack(),
            ),
            InlineKeyboardButton(
                text="🧹 پاک کردن حافظه مکالمه",
                callback_data=AiChatCB(action="clear", owner_id=owner_id).pack(),
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
                callback_data=AiChatCB(action="mem", owner_id=owner_id, page=0).pack(),
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
                text="🔄 بروزرسانی",
                callback_data=AiChatCB(action="refresh", owner_id=owner_id).pack(),
            ),
            InlineKeyboardButton(
                text="✖️ بستن",
                callback_data=AiChatCB(action="close", owner_id=owner_id).pack(),
            ),
        ]
    )
    return InlineKeyboardMarkup(inline_keyboard=rows)


def build_ai_quota_kb(owner_id: int, current_limit: int) -> InlineKeyboardMarkup:
    """Build inline keyboard for managing regular-user AI chat quota limits and resets."""
    presets = [
        (10, "۱۰"),
        (20, "۲۰"),
        (50, "۵۰"),
        (100, "۱۰۰"),
    ]
    preset_row = [
        InlineKeyboardButton(
            text=f"{label} ✓" if current_limit == val else label,
            callback_data=AiChatCB(action="qset", owner_id=owner_id, arg=str(val)).pack(),
        )
        for val, label in presets
    ]
    close_label = "🔒 بستن سهمیه (۰) ✓" if current_limit == 0 else "🔒 بستن سهمیه (۰)"
    unlim_label = "♾️ نامحدود ✓" if current_limit < 0 else "♾️ نامحدود"
    return InlineKeyboardMarkup(
        inline_keyboard=[
            preset_row,
            [
                InlineKeyboardButton(
                    text="✏️ عدد دلخواه",
                    callback_data=AiChatCB(action="qcustom", owner_id=owner_id).pack(),
                ),
                InlineKeyboardButton(
                    text="♻️ ریست سهمیه همه",
                    callback_data=AiChatCB(action="qreset", owner_id=owner_id).pack(),
                ),
            ],
            [
                InlineKeyboardButton(
                    text=close_label,
                    callback_data=AiChatCB(action="qset", owner_id=owner_id, arg="0").pack(),
                ),
                InlineKeyboardButton(
                    text=unlim_label,
                    callback_data=AiChatCB(action="qset", owner_id=owner_id, arg="-1").pack(),
                ),
            ],
            [
                InlineKeyboardButton(
                    text="🔙 بازگشت به پنل",
                    callback_data=AiChatCB(action="refresh", owner_id=owner_id).pack(),
                )
            ],
        ]
    )


def build_ai_quota_cancel_kb(owner_id: int) -> InlineKeyboardMarkup:
    """Build cancel button shown while waiting for custom quota limit input."""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="❌ انصراف",
                    callback_data=AiChatCB(action="qcancel", owner_id=owner_id).pack(),
                )
            ]
        ]
    )


def build_ai_memory_view_kb(
    owner_id: int,
    has_memory: bool,
    page: int = 0,
    total_pages: int = 1,
) -> InlineKeyboardMarkup:
    """Build inline keyboard shown below the paginated learned group & member memory view."""
    rows: list[list[InlineKeyboardButton]] = []
    safe_total = max(1, int(total_pages))
    safe_page = max(0, min(int(page), safe_total - 1))

    if safe_total > 1:
        nav_row: list[InlineKeyboardButton] = []
        if safe_total >= 4:
            if safe_page > 0:
                nav_row.append(
                    InlineKeyboardButton(
                        text="⏮",
                        callback_data=AiChatCB(action="mem", owner_id=owner_id, page=0).pack(),
                    )
                )
                nav_row.append(
                    InlineKeyboardButton(
                        text="⬅️",
                        callback_data=AiChatCB(action="mem", owner_id=owner_id, page=safe_page - 1).pack(),
                    )
                )
            nav_row.append(
                InlineKeyboardButton(
                    text=f"صفحه {safe_page + 1}/{safe_total}",
                    callback_data=AiChatCB(action="noop", owner_id=owner_id).pack(),
                )
            )
            if safe_page < safe_total - 1:
                nav_row.append(
                    InlineKeyboardButton(
                        text="➡️",
                        callback_data=AiChatCB(action="mem", owner_id=owner_id, page=safe_page + 1).pack(),
                    )
                )
                nav_row.append(
                    InlineKeyboardButton(
                        text="⏭",
                        callback_data=AiChatCB(action="mem", owner_id=owner_id, page=safe_total - 1).pack(),
                    )
                )
        else:
            if safe_page > 0:
                nav_row.append(
                    InlineKeyboardButton(
                        text="⬅️ قبلی",
                        callback_data=AiChatCB(action="mem", owner_id=owner_id, page=safe_page - 1).pack(),
                    )
                )
            nav_row.append(
                InlineKeyboardButton(
                    text=f"صفحه {safe_page + 1}/{safe_total}",
                    callback_data=AiChatCB(action="noop", owner_id=owner_id).pack(),
                )
            )
            if safe_page < safe_total - 1:
                nav_row.append(
                    InlineKeyboardButton(
                        text="بعدی ➡️",
                        callback_data=AiChatCB(action="mem", owner_id=owner_id, page=safe_page + 1).pack(),
                    )
                )
        rows.append(nav_row)

    rows.append(
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
    )
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

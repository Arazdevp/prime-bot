"""Inline keyboard factories and CallbackData schema for the Private Admin Panel."""

from typing import Any, Sequence, TypeVar
from aiogram.filters.callback_data import CallbackData
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

T = TypeVar("T")

PAGE_SIZE = 8


class PanelCB(CallbackData, prefix="pn"):
    """Compact CallbackData for all private admin panel actions (always <= 64 bytes)."""

    action: str
    chat_id: int = 0
    arg: str = ""
    page: int = 0


def paginate(
    items: Sequence[T],
    page: int,
    page_size: int = PAGE_SIZE,
) -> tuple[list[T], int, int]:
    """Slice items for the given 0-based page, clamping out-of-range page numbers.

    Returns (page_items, clamped_page, total_pages).
    """
    if page_size <= 0:
        page_size = PAGE_SIZE
    if not items:
        return [], 0, 1
    total_pages = (len(items) + page_size - 1) // page_size
    safe_page = max(0, min(page, total_pages - 1))
    start = safe_page * page_size
    return list(items[start : start + page_size]), safe_page, total_pages


def _truncate(text: str, max_len: int = 28) -> str:
    """Truncate plain button label cleanly with an ellipsis."""
    cleaned = (text or "").strip()
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max_len - 1] + "…"


def _pagination_row(
    action: str,
    chat_id: int,
    page: int,
    total_pages: int,
    arg: str = "",
) -> list[InlineKeyboardButton]:
    """Build prev / page-indicator / next row when total_pages > 1."""
    if total_pages <= 1:
        return []
    row: list[InlineKeyboardButton] = []
    if page > 0:
        row.append(
            InlineKeyboardButton(
                text="⬅️ قبلی",
                callback_data=PanelCB(action=action, chat_id=chat_id, arg=arg, page=page - 1).pack(),
            )
        )
    row.append(
        InlineKeyboardButton(
            text=f"صفحه {page + 1}/{total_pages}",
            callback_data=PanelCB(action="noop").pack(),
        )
    )
    if page < total_pages - 1:
        row.append(
            InlineKeyboardButton(
                text="بعدی ➡️",
                callback_data=PanelCB(action=action, chat_id=chat_id, arg=arg, page=page + 1).pack(),
            )
        )
    return row


def build_open_panel_kb() -> InlineKeyboardMarkup:
    """Single-button keyboard to launch the admin panel."""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="🎛 باز کردن پنل",
                    callback_data=PanelCB(action="open").pack(),
                )
            ]
        ]
    )


def build_home_kb(
    groups: list[dict[str, Any]],
    is_global_admin: bool,
    page: int = 0,
) -> InlineKeyboardMarkup:
    """Home screen keyboard: manageable group list, Global Status (global admins), Help."""
    rows: list[list[InlineKeyboardButton]] = []
    page_groups, safe_page, total_pages = paginate(groups, page, PAGE_SIZE)

    for g in page_groups:
        cid = int(g["chat_id"])
        enabled = g.get("enabled", 1) == 1
        dot = "🟢" if enabled else "🔴"
        title = g.get("title") or f"گروه {cid}"
        rows.append(
            [
                InlineKeyboardButton(
                    text=f"{dot} {_truncate(title, 32)}",
                    callback_data=PanelCB(action="grp", chat_id=cid).pack(),
                )
            ]
        )

    nav = _pagination_row("home", 0, safe_page, total_pages)
    if nav:
        rows.append(nav)

    footer: list[InlineKeyboardButton] = []
    if is_global_admin:
        rows.append(
            [
                InlineKeyboardButton(
                    text="🎙 پنل دیپ‌تاک و مدیریت ادمین‌ها",
                    callback_data=PanelCB(action="dtk").pack(),
                )
            ]
        )
        footer.append(
            InlineKeyboardButton(
                text="📊 وضعیت کلی",
                callback_data=PanelCB(action="stat").pack(),
            )
        )
    footer.append(
        InlineKeyboardButton(
            text="ℹ️ راهنما",
            callback_data=PanelCB(action="help").pack(),
        )
    )
    rows.append(footer)
    return InlineKeyboardMarkup(inline_keyboard=rows)


def build_dashboard_kb(chat_id: int, enabled: bool, mode: str) -> InlineKeyboardMarkup:
    """Group dashboard keyboard."""
    toggle_label = "🔴 غیرفعال کردن" if enabled else "🟢 فعال کردن"
    mode_label = "🔄 حالت: لیست سفید" if mode == "blacklist" else "🔄 حالت: لیست سیاه"

    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=toggle_label,
                    callback_data=PanelCB(action="tog", chat_id=chat_id).pack(),
                ),
                InlineKeyboardButton(
                    text=mode_label,
                    callback_data=PanelCB(action="mode", chat_id=chat_id).pack(),
                ),
            ],
            [
                InlineKeyboardButton(
                    text="📂 تاپیک‌ها",
                    callback_data=PanelCB(action="tops", chat_id=chat_id, page=0).pack(),
                ),
                InlineKeyboardButton(
                    text="📌 مقصد خلاصه",
                    callback_data=PanelCB(action="dest", chat_id=chat_id).pack(),
                ),
            ],
            [
                InlineKeyboardButton(
                    text="🔢 حداقل پیام",
                    callback_data=PanelCB(action="min", chat_id=chat_id).pack(),
                ),
                InlineKeyboardButton(
                    text="▶️ اجرای دستی",
                    callback_data=PanelCB(action="run", chat_id=chat_id, page=0).pack(),
                ),
            ],
            [
                InlineKeyboardButton(
                    text="☀️ صبح‌بخیر روزانه",
                    callback_data=PanelCB(action="mrg", chat_id=chat_id).pack(),
                ),
                InlineKeyboardButton(
                    text="👁 پیش‌نمایش",
                    callback_data=PanelCB(action="prv", chat_id=chat_id, page=0).pack(),
                ),
            ],
            [
                InlineKeyboardButton(
                    text="🔙 بازگشت",
                    callback_data=PanelCB(action="home").pack(),
                ),
            ],
        ]
    )


def build_topics_kb(
    chat_id: int,
    topics: list[dict[str, Any]],
    page: int = 0,
) -> InlineKeyboardMarkup:
    """Paginated topics keyboard (8/page) with per-topic toggles and bulk actions."""
    rows: list[list[InlineKeyboardButton]] = []
    page_topics, safe_page, total_pages = paginate(topics, page, PAGE_SIZE)

    for t in page_topics:
        tid = int(t["thread_id"])
        icon = "✅" if t.get("is_monitored") else "⛔"
        label = f"{icon} {_truncate(str(t.get('name', tid)), 24)} ({t.get('today_count', 0)})"
        rows.append(
            [
                InlineKeyboardButton(
                    text=label,
                    callback_data=PanelCB(
                        action="topt",
                        chat_id=chat_id,
                        arg=str(tid),
                        page=safe_page,
                    ).pack(),
                )
            ]
        )

    if topics:
        rows.append(
            [
                InlineKeyboardButton(
                    text="✅ همه فعال",
                    callback_data=PanelCB(action="topa", chat_id=chat_id, arg="1", page=safe_page).pack(),
                ),
                InlineKeyboardButton(
                    text="⛔ همه غیرفعال",
                    callback_data=PanelCB(action="topa", chat_id=chat_id, arg="0", page=safe_page).pack(),
                ),
            ]
        )

    nav = _pagination_row("tops", chat_id, safe_page, total_pages)
    if nav:
        rows.append(nav)

    rows.append(
        [
            InlineKeyboardButton(
                text="🔙 بازگشت",
                callback_data=PanelCB(action="grp", chat_id=chat_id).pack(),
            )
        ]
    )
    return InlineKeyboardMarkup(inline_keyboard=rows)


def build_destination_kb(chat_id: int, current_mode: str) -> InlineKeyboardMarkup:
    """Destination mode selection keyboard."""
    per_mark = " ✓" if current_mode == "per_topic" else ""
    agg_mark = " ✓" if current_mode == "aggregate" else ""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=f"📍 ارسال در هر تاپیک (پیش‌فرض){per_mark}",
                    callback_data=PanelCB(action="dstp", chat_id=chat_id).pack(),
                )
            ],
            [
                InlineKeyboardButton(
                    text=f"📌 ارسال تجمیعی در یک تاپیک{agg_mark}",
                    callback_data=PanelCB(action="dstl", chat_id=chat_id, page=0).pack(),
                )
            ],
            [
                InlineKeyboardButton(
                    text="🔙 بازگشت",
                    callback_data=PanelCB(action="grp", chat_id=chat_id).pack(),
                )
            ],
        ]
    )


def build_dest_topics_kb(
    chat_id: int,
    topics: list[dict[str, Any]],
    page: int = 0,
) -> InlineKeyboardMarkup:
    """Paginated topic picker for aggregate summary destination."""
    rows: list[list[InlineKeyboardButton]] = []
    page_topics, safe_page, total_pages = paginate(topics, page, PAGE_SIZE)

    for t in page_topics:
        tid = int(t["thread_id"])
        label = f"📌 {_truncate(str(t.get('name', tid)), 28)}"
        rows.append(
            [
                InlineKeyboardButton(
                    text=label,
                    callback_data=PanelCB(
                        action="dsts",
                        chat_id=chat_id,
                        arg=str(tid),
                        page=safe_page,
                    ).pack(),
                )
            ]
        )

    nav = _pagination_row("dstl", chat_id, safe_page, total_pages)
    if nav:
        rows.append(nav)

    rows.append(
        [
            InlineKeyboardButton(
                text="🔙 بازگشت",
                callback_data=PanelCB(action="dest", chat_id=chat_id).pack(),
            )
        ]
    )
    return InlineKeyboardMarkup(inline_keyboard=rows)


def build_dest_confirm_kb(chat_id: int, thread_id: int, page: int = 0) -> InlineKeyboardMarkup:
    """Confirmation keyboard for selecting an aggregate destination topic."""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="✅ تأیید مقصد",
                    callback_data=PanelCB(action="dstc", chat_id=chat_id, arg=str(thread_id)).pack(),
                ),
                InlineKeyboardButton(
                    text="🔙 انصراف",
                    callback_data=PanelCB(action="dstl", chat_id=chat_id, page=page).pack(),
                ),
            ]
        ]
    )


def build_min_messages_kb(chat_id: int) -> InlineKeyboardMarkup:
    """Min-messages adjustment keyboard (-5, -1, +1, +5, manual FSM entry, back)."""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="-5",
                    callback_data=PanelCB(action="mind", chat_id=chat_id, arg="-5").pack(),
                ),
                InlineKeyboardButton(
                    text="-1",
                    callback_data=PanelCB(action="mind", chat_id=chat_id, arg="-1").pack(),
                ),
                InlineKeyboardButton(
                    text="+1",
                    callback_data=PanelCB(action="mind", chat_id=chat_id, arg="1").pack(),
                ),
                InlineKeyboardButton(
                    text="+5",
                    callback_data=PanelCB(action="mind", chat_id=chat_id, arg="5").pack(),
                ),
            ],
            [
                InlineKeyboardButton(
                    text="✏️ وارد کردن عدد",
                    callback_data=PanelCB(action="minf", chat_id=chat_id).pack(),
                )
            ],
            [
                InlineKeyboardButton(
                    text="🔙 بازگشت",
                    callback_data=PanelCB(action="grp", chat_id=chat_id).pack(),
                )
            ],
        ]
    )


def build_manual_run_kb(
    chat_id: int,
    topics: list[dict[str, Any]],
    page: int = 0,
) -> InlineKeyboardMarkup:
    """Manual run target picker: All topics or specific topic (paginated 8/page)."""
    rows: list[list[InlineKeyboardButton]] = [
        [
            InlineKeyboardButton(
                text="🌐 همه تاپیک‌ها",
                callback_data=PanelCB(action="runc", chat_id=chat_id, arg="a_0", page=page).pack(),
            )
        ]
    ]
    page_topics, safe_page, total_pages = paginate(topics, page, PAGE_SIZE)

    for t in page_topics:
        tid = int(t["thread_id"])
        label = f"📂 {_truncate(str(t.get('name', tid)), 24)} ({t.get('today_count', 0)})"
        rows.append(
            [
                InlineKeyboardButton(
                    text=label,
                    callback_data=PanelCB(
                        action="runc",
                        chat_id=chat_id,
                        arg=f"{tid}_0",
                        page=safe_page,
                    ).pack(),
                )
            ]
        )

    nav = _pagination_row("run", chat_id, safe_page, total_pages)
    if nav:
        rows.append(nav)

    rows.append(
        [
            InlineKeyboardButton(
                text="🔙 بازگشت",
                callback_data=PanelCB(action="grp", chat_id=chat_id).pack(),
            )
        ]
    )
    return InlineKeyboardMarkup(inline_keyboard=rows)


def build_manual_run_confirm_kb(
    chat_id: int,
    target_key: str,
    keep: bool,
    page: int = 0,
) -> InlineKeyboardMarkup:
    """Manual run confirmation keyboard with keep-mode toggle."""
    keep_int = 1 if keep else 0
    flipped_int = 0 if keep else 1
    keep_label = "📦 حفظ پیام‌ها (keep): روشن ✅" if keep else "📦 حفظ پیام‌ها (keep): خاموش ❌"

    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=keep_label,
                    callback_data=PanelCB(
                        action="runc",
                        chat_id=chat_id,
                        arg=f"{target_key}_{flipped_int}",
                        page=page,
                    ).pack(),
                )
            ],
            [
                InlineKeyboardButton(
                    text="🚀 شروع اجرا",
                    callback_data=PanelCB(
                        action="rune",
                        chat_id=chat_id,
                        arg=f"{target_key}_{keep_int}",
                        page=page,
                    ).pack(),
                )
            ],
            [
                InlineKeyboardButton(
                    text="🔙 بازگشت",
                    callback_data=PanelCB(action="run", chat_id=chat_id, page=page).pack(),
                )
            ],
        ]
    )


def build_preview_topics_kb(
    chat_id: int,
    topics: list[dict[str, Any]],
    page: int = 0,
) -> InlineKeyboardMarkup:
    """Paginated topic picker for DM-only summary preview."""
    rows: list[list[InlineKeyboardButton]] = []
    page_topics, safe_page, total_pages = paginate(topics, page, PAGE_SIZE)

    for t in page_topics:
        tid = int(t["thread_id"])
        label = f"👁 {_truncate(str(t.get('name', tid)), 24)} ({t.get('today_count', 0)})"
        rows.append(
            [
                InlineKeyboardButton(
                    text=label,
                    callback_data=PanelCB(
                        action="prvr",
                        chat_id=chat_id,
                        arg=str(tid),
                        page=safe_page,
                    ).pack(),
                )
            ]
        )

    nav = _pagination_row("prv", chat_id, safe_page, total_pages)
    if nav:
        rows.append(nav)

    rows.append(
        [
            InlineKeyboardButton(
                text="🔙 بازگشت",
                callback_data=PanelCB(action="grp", chat_id=chat_id).pack(),
            )
        ]
    )
    return InlineKeyboardMarkup(inline_keyboard=rows)


def build_global_status_kb() -> InlineKeyboardMarkup:
    """Global status refresh and back buttons."""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="🔄 بروزرسانی",
                    callback_data=PanelCB(action="stat").pack(),
                ),
                InlineKeyboardButton(
                    text="🔙 بازگشت",
                    callback_data=PanelCB(action="home").pack(),
                ),
            ]
        ]
    )


def build_morning_kb(
    chat_id: int,
    *,
    enabled: bool,
    has_topic: bool,
) -> InlineKeyboardMarkup:
    """Morning greeting management screen keyboard with toggle, topic picker, and admin test button."""
    toggle_label = "🔴 غیرفعال کردن ارسال صبحانه" if enabled else "🟢 فعال کردن ارسال صبحانه"
    topic_label = "📌 تغییر تاپیک مقصد" if has_topic else "📌 انتخاب تاپیک مقصد"
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=toggle_label,
                    callback_data=PanelCB(action="mrgt", chat_id=chat_id).pack(),
                )
            ],
            [
                InlineKeyboardButton(
                    text=topic_label,
                    callback_data=PanelCB(action="mrgl", chat_id=chat_id, page=0).pack(),
                )
            ],
            [
                InlineKeyboardButton(
                    text="🧪 ارسال تستی در تاپیک مقصد",
                    callback_data=PanelCB(action="mrgtst", chat_id=chat_id).pack(),
                )
            ],
            [
                InlineKeyboardButton(
                    text="🔙 بازگشت به داشبورد",
                    callback_data=PanelCB(action="grp", chat_id=chat_id).pack(),
                )
            ],
        ]
    )


def build_morning_topics_kb(
    chat_id: int,
    topics: list[dict[str, Any]],
    page: int = 0,
) -> InlineKeyboardMarkup:
    """Paginated topic picker for the daily morning greeting target topic."""
    rows: list[list[InlineKeyboardButton]] = []
    page_topics, safe_page, total_pages = paginate(topics, page, PAGE_SIZE)

    for t in page_topics:
        tid = int(t["thread_id"])
        label = f"☀️ {_truncate(str(t.get('name', tid)), 28)}"
        rows.append(
            [
                InlineKeyboardButton(
                    text=label,
                    callback_data=PanelCB(
                        action="mrgs",
                        chat_id=chat_id,
                        arg=str(tid),
                        page=safe_page,
                    ).pack(),
                )
            ]
        )

    nav = _pagination_row("mrgl", chat_id, safe_page, total_pages)
    if nav:
        rows.append(nav)

    rows.append(
        [
            InlineKeyboardButton(
                text="🔙 بازگشت",
                callback_data=PanelCB(action="mrg", chat_id=chat_id).pack(),
            )
        ]
    )
    return InlineKeyboardMarkup(inline_keyboard=rows)


def build_back_to_group_kb(chat_id: int) -> InlineKeyboardMarkup:
    """Single button returning to a group's dashboard."""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="🔙 بازگشت به داشبورد",
                    callback_data=PanelCB(action="grp", chat_id=chat_id).pack(),
                )
            ]
        ]
    )


def build_back_to_home_kb() -> InlineKeyboardMarkup:
    """Single button returning to the Home screen."""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="🔙 بازگشت به خانه",
                    callback_data=PanelCB(action="home").pack(),
                )
            ]
        ]
    )

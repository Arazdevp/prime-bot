"""Inline keyboard factories and CallbackData schema for the interactive /timer panel."""

from datetime import datetime
from typing import Any
from aiogram.filters.callback_data import CallbackData
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup
import jdatetime
from bot.keyboards.panel import paginate
from bot.utils.dates import PERSIAN_MONTHS
from bot.utils.timer_parser import (
    MAX_TIMER_DAYS,
    add_jalali_months,
    jalali_month_days,
    parse_utc_iso,
    to_persian_digits,
)

TIMER_PAGE_SIZE = 5


class TimerCB(CallbackData, prefix="tm"):
    """Compact CallbackData for all interactive /timer actions (always <= 64 bytes)."""

    action: str
    owner_id: int = 0
    arg: str = ""


def _noop(owner_id: int) -> str:
    return TimerCB(action="noop", owner_id=owner_id).pack()


def _truncate(text: str, max_len: int = 24) -> str:
    cleaned = (text or "").strip()
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max_len - 1] + "…"


def _dm_activate_row(bot_username: str | None) -> list[InlineKeyboardButton] | None:
    if not bot_username:
        return None
    clean_username = bot_username.strip().lstrip("@")
    if not clean_username:
        return None
    return [
        InlineKeyboardButton(
            text="🔔 فعال‌سازی اعلان پیوی (استارت ربات)",
            url=f"https://t.me/{clean_username}?start=timer",
        )
    ]


def build_timer_home_kb(
    owner_id: int,
    *,
    active_count: int,
    dm_started: bool,
    bot_username: str | None,
    is_private: bool = False,
) -> InlineKeyboardMarkup:
    """Screen 1: Main Timer Hub with 1-tap quick timers, two modes, and active list."""
    list_label = (
        f"📋 تایمرهای فعال من ({to_persian_digits(active_count)})"
        if active_count > 0
        else "📋 تایمرهای فعال من"
    )
    rows: list[list[InlineKeyboardButton]] = [
        [
            InlineKeyboardButton(
                text="⚡ ۱۵ دقیقه",
                callback_data=TimerCB(action="q", owner_id=owner_id, arg="15").pack(),
            ),
            InlineKeyboardButton(
                text="⚡ ۳۰ دقیقه",
                callback_data=TimerCB(action="q", owner_id=owner_id, arg="30").pack(),
            ),
            InlineKeyboardButton(
                text="⚡ ۴۵ دقیقه",
                callback_data=TimerCB(action="q", owner_id=owner_id, arg="45").pack(),
            ),
            InlineKeyboardButton(
                text="⚡ ۱ ساعت",
                callback_data=TimerCB(action="q", owner_id=owner_id, arg="60").pack(),
            ),
        ],
        [
            InlineKeyboardButton(
                text="⏳ تایمر دقیقه‌ای / ساعتی",
                callback_data=TimerCB(action="rel", owner_id=owner_id).pack(),
            ),
            InlineKeyboardButton(
                text="📅 تایمر روزی / تاریخی",
                callback_data=TimerCB(action="cal", owner_id=owner_id).pack(),
            ),
        ],
        [
            InlineKeyboardButton(
                text=list_label,
                callback_data=TimerCB(action="list", owner_id=owner_id, arg="0").pack(),
            )
        ],
    ]

    if not is_private and not dm_started:
        dm_row = _dm_activate_row(bot_username)
        if dm_row:
            rows.append(dm_row)

    rows.append(
        [
            InlineKeyboardButton(
                text="❌ بستن پنل",
                callback_data=TimerCB(action="cls", owner_id=owner_id).pack(),
            )
        ]
    )
    return InlineKeyboardMarkup(inline_keyboard=rows)


def build_quick_confirm_kb(owner_id: int, *, has_note: bool) -> InlineKeyboardMarkup:
    """Quick 1-tap preset review screen (confirm, add/edit note, or customize time)."""
    note_row: list[InlineKeyboardButton] = [
        InlineKeyboardButton(
            text="📝 ویرایش یادداشت" if has_note else "📝 افزودن یادداشت",
            callback_data=TimerCB(action="note", owner_id=owner_id, arg="q").pack(),
        )
    ]
    if has_note:
        note_row.append(
            InlineKeyboardButton(
                text="🗑 حذف یادداشت",
                callback_data=TimerCB(action="nclr", owner_id=owner_id, arg="q").pack(),
            )
        )
    else:
        note_row.append(
            InlineKeyboardButton(
                text="⚙️ تنظیم دقیق زمان",
                callback_data=TimerCB(action="rel", owner_id=owner_id).pack(),
            )
        )

    rows: list[list[InlineKeyboardButton]] = [
        [
            InlineKeyboardButton(
                text="✅ ثبت و شروع تایمر",
                callback_data=TimerCB(action="save", owner_id=owner_id).pack(),
            )
        ],
        note_row,
    ]
    if has_note:
        rows.append(
            [
                InlineKeyboardButton(
                    text="⚙️ تنظیم دقیق زمان",
                    callback_data=TimerCB(action="rel", owner_id=owner_id).pack(),
                )
            ]
        )
    rows.append(
        [
            InlineKeyboardButton(
                text="🔙 بازگشت",
                callback_data=TimerCB(action="home", owner_id=owner_id).pack(),
            ),
            InlineKeyboardButton(
                text="❌ بستن",
                callback_data=TimerCB(action="cls", owner_id=owner_id).pack(),
            ),
        ]
    )
    return InlineKeyboardMarkup(inline_keyboard=rows)


def build_relative_kb(
    owner_id: int,
    *,
    duration_minutes: int,
    has_note: bool,
) -> InlineKeyboardMarkup:
    """Screen 2: Interactive countdown builder with minute/hour presets and fine steppers."""

    def _preset_btn(label: str, mins: int) -> InlineKeyboardButton:
        mark = "✓ " if duration_minutes == mins else ""
        return InlineKeyboardButton(
            text=f"{mark}{label}",
            callback_data=TimerCB(action="rset", owner_id=owner_id, arg=str(mins)).pack(),
        )

    rows: list[list[InlineKeyboardButton]] = [
        [
            _preset_btn("۵د", 5),
            _preset_btn("۱۰د", 10),
            _preset_btn("۱۵د", 15),
            _preset_btn("۳۰د", 30),
            _preset_btn("۴۵د", 45),
        ],
        [
            _preset_btn("۱س", 60),
            _preset_btn("۲س", 120),
            _preset_btn("۳س", 180),
            _preset_btn("۶س", 360),
            _preset_btn("۱۲س", 720),
        ],
        [
            InlineKeyboardButton(
                text="-۱۵د",
                callback_data=TimerCB(action="radd", owner_id=owner_id, arg="-15").pack(),
            ),
            InlineKeyboardButton(
                text="-۵د",
                callback_data=TimerCB(action="radd", owner_id=owner_id, arg="-5").pack(),
            ),
            InlineKeyboardButton(
                text="+۵د",
                callback_data=TimerCB(action="radd", owner_id=owner_id, arg="5").pack(),
            ),
            InlineKeyboardButton(
                text="+۱۵د",
                callback_data=TimerCB(action="radd", owner_id=owner_id, arg="15").pack(),
            ),
            InlineKeyboardButton(
                text="+۱س",
                callback_data=TimerCB(action="radd", owner_id=owner_id, arg="60").pack(),
            ),
        ],
        [
            InlineKeyboardButton(
                text="⌨️ عدد دلخواه",
                callback_data=TimerCB(action="rcus", owner_id=owner_id).pack(),
            ),
            InlineKeyboardButton(
                text="📝 ویرایش یادداشت" if has_note else "📝 افزودن یادداشت",
                callback_data=TimerCB(action="note", owner_id=owner_id, arg="rel").pack(),
            ),
        ],
    ]

    if has_note:
        rows.append(
            [
                InlineKeyboardButton(
                    text="🗑 حذف یادداشت",
                    callback_data=TimerCB(action="nclr", owner_id=owner_id, arg="rel").pack(),
                )
            ]
        )

    rows.extend(
        [
            [
                InlineKeyboardButton(
                    text="✅ ثبت و شروع تایمر",
                    callback_data=TimerCB(action="save", owner_id=owner_id).pack(),
                )
            ],
            [
                InlineKeyboardButton(
                    text="🔙 بازگشت",
                    callback_data=TimerCB(action="home", owner_id=owner_id).pack(),
                ),
                InlineKeyboardButton(
                    text="❌ بستن",
                    callback_data=TimerCB(action="cls", owner_id=owner_id).pack(),
                ),
            ],
        ]
    )
    return InlineKeyboardMarkup(inline_keyboard=rows)


def build_calendar_kb(
    owner_id: int,
    *,
    cal_year: int,
    cal_month: int,
    selected_jdate: jdatetime.date,
    today_jdate: jdatetime.date,
) -> InlineKeyboardMarkup:
    """Screen 3: Interactive Jalali Calendar date picker with quick shortcuts and month navigation."""
    max_jdate = today_jdate + jdatetime.timedelta(days=MAX_TIMER_DAYS)
    tomorrow_j = today_jdate + jdatetime.timedelta(days=1)
    after_tomorrow_j = today_jdate + jdatetime.timedelta(days=2)
    next_week_j = today_jdate + jdatetime.timedelta(days=7)

    def _day_arg(d: jdatetime.date) -> str:
        return f"{d.year:04d}-{d.month:02d}-{d.day:02d}"

    rows: list[list[InlineKeyboardButton]] = [
        [
            InlineKeyboardButton(
                text="امروز",
                callback_data=TimerCB(action="day", owner_id=owner_id, arg=_day_arg(today_jdate)).pack(),
            ),
            InlineKeyboardButton(
                text="فردا",
                callback_data=TimerCB(action="day", owner_id=owner_id, arg=_day_arg(tomorrow_j)).pack(),
            ),
            InlineKeyboardButton(
                text="پس‌فردا",
                callback_data=TimerCB(action="day", owner_id=owner_id, arg=_day_arg(after_tomorrow_j)).pack(),
            ),
            InlineKeyboardButton(
                text="هفته بعد",
                callback_data=TimerCB(action="day", owner_id=owner_id, arg=_day_arg(next_week_j)).pack(),
            ),
        ]
    ]

    # Month navigation header
    can_prev = (cal_year, cal_month) > (today_jdate.year, today_jdate.month)
    can_next = (cal_year, cal_month) < (max_jdate.year, max_jdate.month)
    prev_y, prev_m = add_jalali_months(cal_year, cal_month, -1)
    next_y, next_m = add_jalali_months(cal_year, cal_month, 1)

    month_title = f"{PERSIAN_MONTHS.get(cal_month, '')} {to_persian_digits(cal_year)}"
    rows.append(
        [
            InlineKeyboardButton(
                text="◀️" if can_prev else "·",
                callback_data=(
                    TimerCB(action="cal", owner_id=owner_id, arg=f"{prev_y:04d}-{prev_m:02d}").pack()
                    if can_prev
                    else _noop(owner_id)
                ),
            ),
            InlineKeyboardButton(
                text=f"📆 {month_title}",
                callback_data=_noop(owner_id),
            ),
            InlineKeyboardButton(
                text="▶️" if can_next else "·",
                callback_data=(
                    TimerCB(action="cal", owner_id=owner_id, arg=f"{next_y:04d}-{next_m:02d}").pack()
                    if can_next
                    else _noop(owner_id)
                ),
            ),
        ]
    )

    # Weekday header (Saturday to Friday)
    rows.append(
        [
            InlineKeyboardButton(text=wd, callback_data=_noop(owner_id))
            for wd in ("ش", "ی", "د", "س", "چ", "پ", "ج")
        ]
    )

    # Build calendar grid
    first_wd = jdatetime.date(cal_year, cal_month, 1).weekday()  # 0=Sat .. 6=Fri
    total_days = jalali_month_days(cal_year, cal_month)
    week_row: list[InlineKeyboardButton] = [
        InlineKeyboardButton(text=" ", callback_data=_noop(owner_id)) for _ in range(first_wd)
    ]

    for day in range(1, total_days + 1):
        d = jdatetime.date(cal_year, cal_month, day)
        if d < today_jdate or d > max_jdate:
            btn = InlineKeyboardButton(text="·", callback_data=_noop(owner_id))
        else:
            day_fa = to_persian_digits(day)
            if d == selected_jdate:
                label = f"✅{day_fa}"
            elif d == today_jdate:
                label = f"📍{day_fa}"
            else:
                label = day_fa
            btn = InlineKeyboardButton(
                text=label,
                callback_data=TimerCB(action="day", owner_id=owner_id, arg=_day_arg(d)).pack(),
            )
        week_row.append(btn)
        if len(week_row) == 7:
            rows.append(week_row)
            week_row = []

    if week_row:
        while len(week_row) < 7:
            week_row.append(InlineKeyboardButton(text=" ", callback_data=_noop(owner_id)))
        rows.append(week_row)

    rows.append(
        [
            InlineKeyboardButton(
                text="⌨️ تایپ تاریخ و ساعت",
                callback_data=TimerCB(action="dcus", owner_id=owner_id).pack(),
            ),
            InlineKeyboardButton(
                text="🔙 بازگشت",
                callback_data=TimerCB(action="home", owner_id=owner_id).pack(),
            ),
            InlineKeyboardButton(
                text="❌ بستن",
                callback_data=TimerCB(action="cls", owner_id=owner_id).pack(),
            ),
        ]
    )
    return InlineKeyboardMarkup(inline_keyboard=rows)


def build_time_picker_kb(
    owner_id: int,
    *,
    hour: int,
    minute: int,
    has_note: bool,
    is_valid_future: bool,
) -> InlineKeyboardMarkup:
    """Screen 4: Interactive clock time picker (presets, +/- hour, +/- minute, custom time, note)."""
    rows: list[list[InlineKeyboardButton]] = [
        [
            InlineKeyboardButton(
                text="🌅 08:00",
                callback_data=TimerCB(action="tset", owner_id=owner_id, arg="08.00").pack(),
            ),
            InlineKeyboardButton(
                text="☀️ 12:00",
                callback_data=TimerCB(action="tset", owner_id=owner_id, arg="12.00").pack(),
            ),
            InlineKeyboardButton(
                text="🌤 16:00",
                callback_data=TimerCB(action="tset", owner_id=owner_id, arg="16.00").pack(),
            ),
            InlineKeyboardButton(
                text="🌆 20:00",
                callback_data=TimerCB(action="tset", owner_id=owner_id, arg="20.00").pack(),
            ),
            InlineKeyboardButton(
                text="🌙 23:00",
                callback_data=TimerCB(action="tset", owner_id=owner_id, arg="23.00").pack(),
            ),
        ],
        [
            InlineKeyboardButton(
                text="-۳س",
                callback_data=TimerCB(action="hadd", owner_id=owner_id, arg="-3").pack(),
            ),
            InlineKeyboardButton(
                text="-۱س",
                callback_data=TimerCB(action="hadd", owner_id=owner_id, arg="-1").pack(),
            ),
            InlineKeyboardButton(
                text=f"ساعت {hour:02d}",
                callback_data=_noop(owner_id),
            ),
            InlineKeyboardButton(
                text="+۱س",
                callback_data=TimerCB(action="hadd", owner_id=owner_id, arg="1").pack(),
            ),
            InlineKeyboardButton(
                text="+۳س",
                callback_data=TimerCB(action="hadd", owner_id=owner_id, arg="3").pack(),
            ),
        ],
        [
            InlineKeyboardButton(
                text="-۱۵د",
                callback_data=TimerCB(action="madd", owner_id=owner_id, arg="-15").pack(),
            ),
            InlineKeyboardButton(
                text="-۵د",
                callback_data=TimerCB(action="madd", owner_id=owner_id, arg="-5").pack(),
            ),
            InlineKeyboardButton(
                text=f"دقیقه {minute:02d}",
                callback_data=_noop(owner_id),
            ),
            InlineKeyboardButton(
                text="+۵د",
                callback_data=TimerCB(action="madd", owner_id=owner_id, arg="5").pack(),
            ),
            InlineKeyboardButton(
                text="+۱۵د",
                callback_data=TimerCB(action="madd", owner_id=owner_id, arg="15").pack(),
            ),
        ],
        [
            InlineKeyboardButton(
                text="⌨️ تایپ ساعت",
                callback_data=TimerCB(action="tcus", owner_id=owner_id).pack(),
            ),
            InlineKeyboardButton(
                text="📝 ویرایش یادداشت" if has_note else "📝 افزودن یادداشت",
                callback_data=TimerCB(action="note", owner_id=owner_id, arg="tm").pack(),
            ),
        ],
    ]

    if has_note:
        rows.append(
            [
                InlineKeyboardButton(
                    text="🗑 حذف یادداشت",
                    callback_data=TimerCB(action="nclr", owner_id=owner_id, arg="tm").pack(),
                )
            ]
        )

    save_label = "✅ ثبت یادآور" if is_valid_future else "⚠️ ساعت انتخابی گذشته (تغییر بده)"
    rows.extend(
        [
            [
                InlineKeyboardButton(
                    text=save_label,
                    callback_data=TimerCB(action="save", owner_id=owner_id).pack(),
                )
            ],
            [
                InlineKeyboardButton(
                    text="📅 تغییر تاریخ",
                    callback_data=TimerCB(action="cal", owner_id=owner_id).pack(),
                ),
                InlineKeyboardButton(
                    text="🔙 منوی اصلی",
                    callback_data=TimerCB(action="home", owner_id=owner_id).pack(),
                ),
                InlineKeyboardButton(
                    text="❌ بستن",
                    callback_data=TimerCB(action="cls", owner_id=owner_id).pack(),
                ),
            ],
        ]
    )
    return InlineKeyboardMarkup(inline_keyboard=rows)


def build_input_cancel_kb(owner_id: int, back_action: str) -> InlineKeyboardMarkup:
    """Cancel button shown while waiting for FSM text input."""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="🔙 انصراف و بازگشت",
                    callback_data=TimerCB(action="back", owner_id=owner_id, arg=back_action).pack(),
                )
            ]
        ]
    )


def build_saved_timer_kb(
    owner_id: int,
    timer_id: int,
    *,
    dm_started: bool,
    bot_username: str | None,
    is_private: bool = False,
) -> InlineKeyboardMarkup:
    """Compact action row attached to the confirmation card of a newly saved timer."""
    rows: list[list[InlineKeyboardButton]] = [
        [
            InlineKeyboardButton(
                text="❌ لغو این تایمر",
                callback_data=TimerCB(action="del", owner_id=owner_id, arg=f"{timer_id}_c").pack(),
            ),
            InlineKeyboardButton(
                text="📋 تایمرهای من",
                callback_data=TimerCB(action="list", owner_id=owner_id, arg="0").pack(),
            ),
        ]
    ]
    if not is_private and not dm_started:
        dm_row = _dm_activate_row(bot_username)
        if dm_row:
            rows.append(dm_row)
    return InlineKeyboardMarkup(inline_keyboard=rows)


def build_active_timers_kb(
    owner_id: int,
    timers: list[dict[str, Any]],
    *,
    page: int = 0,
    tz_info: Any = None,
) -> InlineKeyboardMarkup:
    """Screen 6: Paginated active timers list with 1-tap cancel buttons."""
    rows: list[list[InlineKeyboardButton]] = []
    page_items, safe_page, total_pages = paginate(timers, page, TIMER_PAGE_SIZE)

    for idx, t in enumerate(page_items, start=safe_page * TIMER_PAGE_SIZE + 1):
        tid = int(t["id"])
        try:
            fire_dt = parse_utc_iso(t["fire_at"])
            if tz_info is not None:
                fire_dt = fire_dt.astimezone(tz_info)
            time_label = fire_dt.strftime("%H:%M")
        except Exception:
            time_label = ""
        note_part = f" • {_truncate(t['note'], 18)}" if t.get("note") else ""
        btn_text = f"🗑 لغو #{to_persian_digits(idx)} ({time_label}{note_part})"
        rows.append(
            [
                InlineKeyboardButton(
                    text=btn_text,
                    callback_data=TimerCB(action="del", owner_id=owner_id, arg=f"{tid}_{safe_page}").pack(),
                )
            ]
        )

    if total_pages > 1:
        nav: list[InlineKeyboardButton] = []
        if safe_page > 0:
            nav.append(
                InlineKeyboardButton(
                    text="⬅️ قبلی",
                    callback_data=TimerCB(action="list", owner_id=owner_id, arg=str(safe_page - 1)).pack(),
                )
            )
        nav.append(
            InlineKeyboardButton(
                text=f"صفحه {to_persian_digits(safe_page + 1)}/{to_persian_digits(total_pages)}",
                callback_data=_noop(owner_id),
            )
        )
        if safe_page < total_pages - 1:
            nav.append(
                InlineKeyboardButton(
                    text="بعدی ➡️",
                    callback_data=TimerCB(action="list", owner_id=owner_id, arg=str(safe_page + 1)).pack(),
                )
            )
        rows.append(nav)

    if len(timers) >= 2:
        rows.append(
            [
                InlineKeyboardButton(
                    text="🧹 لغو همه تایمرها",
                    callback_data=TimerCB(action="dela", owner_id=owner_id, arg="ask").pack(),
                )
            ]
        )

    rows.append(
        [
            InlineKeyboardButton(
                text="➕ تایمر جدید",
                callback_data=TimerCB(action="home", owner_id=owner_id).pack(),
            ),
            InlineKeyboardButton(
                text="❌ بستن",
                callback_data=TimerCB(action="cls", owner_id=owner_id).pack(),
            ),
        ]
    )
    return InlineKeyboardMarkup(inline_keyboard=rows)


def build_cancel_all_confirm_kb(owner_id: int) -> InlineKeyboardMarkup:
    """Two-step confirmation keyboard for cancelling all active timers."""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="✅ بله، همه رو لغو کن",
                    callback_data=TimerCB(action="dela", owner_id=owner_id, arg="yes").pack(),
                ),
                InlineKeyboardButton(
                    text="🔙 انصراف",
                    callback_data=TimerCB(action="list", owner_id=owner_id, arg="0").pack(),
                ),
            ]
        ]
    )


def build_fired_alarm_kb(
    owner_id: int,
    timer_id: int,
    *,
    topic_url: str | None = None,
) -> InlineKeyboardMarkup:
    """Post-alarm interactive buttons: optional topic jump link, Snooze (+5/+10/+30m), and Dismiss."""
    rows: list[list[InlineKeyboardButton]] = []
    if topic_url:
        rows.append(
            [
                InlineKeyboardButton(
                    text="🔗 رفتن به تاپیک در گروه",
                    url=topic_url,
                )
            ]
        )
    rows.append(
        [
            InlineKeyboardButton(
                text="🔁 +۵ دقیقه",
                callback_data=TimerCB(action="snz", owner_id=owner_id, arg=f"{timer_id}_5").pack(),
            ),
            InlineKeyboardButton(
                text="🔁 +۱۰ دقیقه",
                callback_data=TimerCB(action="snz", owner_id=owner_id, arg=f"{timer_id}_10").pack(),
            ),
            InlineKeyboardButton(
                text="🔁 +۳۰ دقیقه",
                callback_data=TimerCB(action="snz", owner_id=owner_id, arg=f"{timer_id}_30").pack(),
            ),
        ]
    )
    rows.append(
        [
            InlineKeyboardButton(
                text="✅ دیدم، انجام شد",
                callback_data=TimerCB(action="ack", owner_id=owner_id, arg=str(timer_id)).pack(),
            )
        ]
    )
    return InlineKeyboardMarkup(inline_keyboard=rows)


def build_dm_timer_entry_kb(owner_id: int) -> InlineKeyboardMarkup:
    """Quick entry button shown in private chat after activating timer DM notifications."""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="⏱ باز کردن پنل تایمر",
                    callback_data=TimerCB(action="home", owner_id=owner_id).pack(),
                ),
                InlineKeyboardButton(
                    text="📋 تایمرهای فعال من",
                    callback_data=TimerCB(action="list", owner_id=owner_id, arg="0").pack(),
                ),
            ]
        ]
    )

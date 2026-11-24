"""Inline keyboard factories and CallbackData schema for the DeepTalk (/deeptalk) feature."""

from typing import Any
from aiogram.filters.callback_data import CallbackData
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup
import jdatetime
from bot.keyboards.panel import PAGE_SIZE, paginate
from bot.utils.dates import PERSIAN_MONTHS
from bot.utils.timer_parser import (
    MAX_TIMER_DAYS,
    add_jalali_months,
    jalali_month_days,
    parse_utc_iso,
    to_persian_digits,
)

DEEPTALK_SESSIONS_PAGE_SIZE = 5
DEEPTALK_ADMINS_PAGE_SIZE = 6


class DeepTalkCB(CallbackData, prefix="dt"):
    """Compact CallbackData for all DeepTalk panel and topic actions (always <= 64 bytes)."""

    action: str
    sid: int = 0
    arg: str = ""


def _noop(sid: int = 0) -> str:
    return DeepTalkCB(action="noop", sid=sid).pack()


def _truncate(text: str, max_len: int = 28) -> str:
    cleaned = (text or "").strip()
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max_len - 1] + "…"


def _participants_pv_button(
    session_id: int,
    bot_username: str | None,
    label: str = "👀 مشاهده افراد (در پیوی)",
) -> InlineKeyboardButton:
    """Build deep-link URL button opening the bot PV participant list (with callback fallback)."""
    clean_u = (bot_username or "").strip().lstrip("@")
    if clean_u:
        return InlineKeyboardButton(
            text=label,
            url=f"https://t.me/{clean_u}?start=dt_{session_id}",
        )
    return InlineKeyboardButton(
        text=label,
        callback_data=DeepTalkCB(action="plist", sid=session_id).pack(),
    )


# ---------------- Topic Public Card Keyboards ----------------


def build_announcement_card_kb(
    session_id: int,
    *,
    registered_count: int,
    bot_username: str | None,
    status: str = "scheduled",
) -> InlineKeyboardMarkup:
    """Inline keyboard attached to the DeepTalk announcement message in the group topic."""
    rows: list[list[InlineKeyboardButton]] = []
    if status == "scheduled":
        cnt_str = f" ({to_persian_digits(registered_count)})" if registered_count > 0 else ""
        rows.append(
            [
                InlineKeyboardButton(
                    text=f"🙋‍♂️ منم هستم!{cnt_str}",
                    callback_data=DeepTalkCB(action="reg", sid=session_id).pack(),
                ),
                InlineKeyboardButton(
                    text="🏳️ انصراف",
                    callback_data=DeepTalkCB(action="unreg", sid=session_id).pack(),
                ),
            ]
        )

    if status != "cancelled":
        rows.append(
            [
                _participants_pv_button(
                    session_id,
                    bot_username,
                    label="👀 مشاهده افراد ثبت‌نام‌شده (در پیوی)",
                )
            ]
        )
    return InlineKeyboardMarkup(inline_keyboard=rows)


def build_start_card_kb(
    session_id: int,
    *,
    checked_in_count: int,
    bot_username: str | None,
    status: str = "live",
) -> InlineKeyboardMarkup:
    """Inline keyboard attached to the live 'DeepTalk Started!!' message in the group topic."""
    rows: list[list[InlineKeyboardButton]] = []
    if status == "live":
        cnt_str = f" ({to_persian_digits(checked_in_count)} نفر)" if checked_in_count > 0 else ""
        rows.append(
            [
                InlineKeyboardButton(
                    text=f"✋ اعلام حضور در دیپ‌تاک!{cnt_str}",
                    callback_data=DeepTalkCB(action="chk", sid=session_id).pack(),
                )
            ]
        )
    elif status == "completed":
        rows.append(
            [
                InlineKeyboardButton(
                    text=f"🏁 دیپ‌تاک پایان یافت ({to_persian_digits(checked_in_count)} حاضر)",
                    callback_data=_noop(session_id),
                )
            ]
        )

    rows.append(
        [
            _participants_pv_button(
                session_id,
                bot_username,
                label="👀 مشاهده لیست حاضرین و ثبت‌نامی‌ها (در پیوی)",
            )
        ]
    )
    return InlineKeyboardMarkup(inline_keyboard=rows)


def build_end_card_kb(
    session_id: int,
    *,
    bot_username: str | None,
) -> InlineKeyboardMarkup:
    """Inline keyboard attached to the DeepTalk wrap-up summary card."""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                _participants_pv_button(
                    session_id,
                    bot_username,
                    label="👀 لیست کامل شرکت‌کنندگان (در پیوی)",
                )
            ]
        ]
    )


def build_dm_reminder_kb(topic_url: str | None) -> InlineKeyboardMarkup | None:
    """Jump link button sent in private DM reminder when DeepTalk starts."""
    if not topic_url:
        return None
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="🔗 پرش مستقیم به تاپیک دیپ‌ورک",
                    url=topic_url,
                )
            ]
        ]
    )


def build_pv_participants_kb(
    session_id: int,
    *,
    topic_url: str | None = None,
) -> InlineKeyboardMarkup:
    """Refresh and topic-jump buttons displayed under the participant list in private chat."""
    rows: list[list[InlineKeyboardButton]] = [
        [
            InlineKeyboardButton(
                text="🔄 بروزرسانی لیست",
                callback_data=DeepTalkCB(action="pref", sid=session_id).pack(),
            )
        ]
    ]
    if topic_url:
        rows.append(
            [
                InlineKeyboardButton(
                    text="🔗 رفتن به تاپیک گفتگو در گروه",
                    url=topic_url,
                )
            ]
        )
    return InlineKeyboardMarkup(inline_keyboard=rows)


# ---------------- Admin Panel & Wizard Keyboards ----------------


def build_deeptalk_hub_kb(
    owner_id: int,
    *,
    active_sessions_count: int,
    is_owner: bool,
    multi_group: bool,
    is_private: bool,
) -> InlineKeyboardMarkup:
    """Main /deeptalk control hub keyboard."""
    sess_label = (
        f"📋 جلسات فعال و پیش‌رو ({to_persian_digits(active_sessions_count)})"
        if active_sessions_count > 0
        else "📋 جلسات فعال و پیش‌رو"
    )
    rows: list[list[InlineKeyboardButton]] = [
        [
            InlineKeyboardButton(
                text="➕ ساخت دیپ‌تاک جدید",
                callback_data=DeepTalkCB(action="new", sid=owner_id).pack(),
            )
        ],
        [
            InlineKeyboardButton(
                text=sess_label,
                callback_data=DeepTalkCB(action="list", sid=owner_id, arg="0").pack(),
            )
        ],
    ]

    cfg_row: list[InlineKeyboardButton] = [
        InlineKeyboardButton(
            text="📌 انتخاب تاپیک مقصد",
            callback_data=DeepTalkCB(action="top", sid=owner_id, arg="p0").pack(),
        )
    ]
    if is_private and multi_group:
        cfg_row.append(
            InlineKeyboardButton(
                text="🏢 تغییر گروه",
                callback_data=DeepTalkCB(action="grp", sid=owner_id, arg="p0").pack(),
            )
        )
    rows.append(cfg_row)

    if is_owner:
        rows.append(
            [
                InlineKeyboardButton(
                    text="👑 مدیریت ادمین‌های دیپ‌تاک",
                    callback_data=DeepTalkCB(action="adm", sid=owner_id, arg="0").pack(),
                )
            ]
        )

    rows.append(
        [
            InlineKeyboardButton(
                text="❌ بستن پنل",
                callback_data=DeepTalkCB(action="cls", sid=owner_id).pack(),
            )
        ]
    )
    return InlineKeyboardMarkup(inline_keyboard=rows)


def build_deeptalk_wizard_kb(
    owner_id: int,
    *,
    has_subject: bool,
    has_description: bool,
    duration_minutes: int,
    can_publish: bool,
) -> InlineKeyboardMarkup:
    """Interactive session creation wizard keyboard."""
    subj_label = "✏️ ویرایش محوریت بحث" if has_subject else "🎯 تعیین محوریت بحث (الزامی)"
    desc_label = "📝 ویرایش توضیحات/سوال" if has_description else "💡 افزودن سوال/توضیح (اختیاری)"

    rows: list[list[InlineKeyboardButton]] = [
        [
            InlineKeyboardButton(
                text=subj_label,
                callback_data=DeepTalkCB(action="subj", sid=owner_id).pack(),
            )
        ],
    ]

    desc_row = [
        InlineKeyboardButton(
            text=desc_label,
            callback_data=DeepTalkCB(action="desc", sid=owner_id).pack(),
        )
    ]
    if has_description:
        desc_row.append(
            InlineKeyboardButton(
                text="🗑 حذف توضیح",
                callback_data=DeepTalkCB(action="dclr", sid=owner_id).pack(),
            )
        )
    rows.append(desc_row)

    # Start time quick shortcuts + Calendar/Clock pickers
    rows.extend(
        [
            [
                InlineKeyboardButton(
                    text="⚡ ۱۵د دیگه",
                    callback_data=DeepTalkCB(action="qst", sid=owner_id, arg="15").pack(),
                ),
                InlineKeyboardButton(
                    text="⚡ ۳۰د دیگه",
                    callback_data=DeepTalkCB(action="qst", sid=owner_id, arg="30").pack(),
                ),
                InlineKeyboardButton(
                    text="⚡ ۱س دیگه",
                    callback_data=DeepTalkCB(action="qst", sid=owner_id, arg="60").pack(),
                ),
            ],
            [
                InlineKeyboardButton(
                    text="📅 تاریخ شروع (شمسی)",
                    callback_data=DeepTalkCB(action="cal", sid=owner_id).pack(),
                ),
                InlineKeyboardButton(
                    text="🕐 ساعت شروع",
                    callback_data=DeepTalkCB(action="tm", sid=owner_id).pack(),
                ),
            ],
        ]
    )

    # Duration presets
    def _dur_btn(label: str, mins: int) -> InlineKeyboardButton:
        mark = "✓ " if duration_minutes == mins else ""
        return InlineKeyboardButton(
            text=f"{mark}{label}",
            callback_data=DeepTalkCB(action="dur", sid=owner_id, arg=str(mins)).pack(),
        )

    rows.extend(
        [
            [
                _dur_btn("۲۰د", 20),
                _dur_btn("۳۰د", 30),
                _dur_btn("۴۵د", 45),
                _dur_btn("۱س", 60),
                _dur_btn("۹۰د", 90),
            ],
            [
                InlineKeyboardButton(
                    text="⏱ مدت دلخواه",
                    callback_data=DeepTalkCB(action="dur", sid=owner_id, arg="cus").pack(),
                ),
                InlineKeyboardButton(
                    text="📌 تغییر تاپیک مقصد",
                    callback_data=DeepTalkCB(action="top", sid=owner_id, arg="w0").pack(),
                ),
            ],
        ]
    )

    pub_label = (
        "✅ تأیید و ارسال فراخوان در تاپیک 🚀"
        if can_publish
        else "⚠️ ابتدا محوریت و زمان آینده را کامل کن"
    )
    rows.append(
        [
            InlineKeyboardButton(
                text=pub_label,
                callback_data=DeepTalkCB(action="pub", sid=owner_id).pack(),
            )
        ]
    )
    rows.append(
        [
            InlineKeyboardButton(
                text="🔙 منوی اصلی",
                callback_data=DeepTalkCB(action="home", sid=owner_id).pack(),
            ),
            InlineKeyboardButton(
                text="❌ بستن",
                callback_data=DeepTalkCB(action="cls", sid=owner_id).pack(),
            ),
        ]
    )
    return InlineKeyboardMarkup(inline_keyboard=rows)


def build_deeptalk_calendar_kb(
    owner_id: int,
    *,
    cal_year: int,
    cal_month: int,
    selected_jdate: jdatetime.date,
    today_jdate: jdatetime.date,
) -> InlineKeyboardMarkup:
    """Interactive Jalali Calendar date picker for DeepTalk start date."""
    max_jdate = today_jdate + jdatetime.timedelta(days=MAX_TIMER_DAYS)
    tomorrow_j = today_jdate + jdatetime.timedelta(days=1)
    after_tomorrow_j = today_jdate + jdatetime.timedelta(days=2)

    def _day_arg(d: jdatetime.date) -> str:
        return f"{d.year:04d}-{d.month:02d}-{d.day:02d}"

    rows: list[list[InlineKeyboardButton]] = [
        [
            InlineKeyboardButton(
                text="امروز",
                callback_data=DeepTalkCB(action="day", sid=owner_id, arg=_day_arg(today_jdate)).pack(),
            ),
            InlineKeyboardButton(
                text="فردا",
                callback_data=DeepTalkCB(action="day", sid=owner_id, arg=_day_arg(tomorrow_j)).pack(),
            ),
            InlineKeyboardButton(
                text="پس‌فردا",
                callback_data=DeepTalkCB(action="day", sid=owner_id, arg=_day_arg(after_tomorrow_j)).pack(),
            ),
        ]
    ]

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
                    DeepTalkCB(action="cal", sid=owner_id, arg=f"{prev_y:04d}-{prev_m:02d}").pack()
                    if can_prev
                    else _noop(owner_id)
                ),
            ),
            InlineKeyboardButton(text=f"📆 {month_title}", callback_data=_noop(owner_id)),
            InlineKeyboardButton(
                text="▶️" if can_next else "·",
                callback_data=(
                    DeepTalkCB(action="cal", sid=owner_id, arg=f"{next_y:04d}-{next_m:02d}").pack()
                    if can_next
                    else _noop(owner_id)
                ),
            ),
        ]
    )

    rows.append(
        [InlineKeyboardButton(text=wd, callback_data=_noop(owner_id)) for wd in ("ش", "ی", "د", "س", "چ", "پ", "ج")]
    )

    first_wd = jdatetime.date(cal_year, cal_month, 1).weekday()
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
                callback_data=DeepTalkCB(action="day", sid=owner_id, arg=_day_arg(d)).pack(),
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
                callback_data=DeepTalkCB(action="scus", sid=owner_id).pack(),
            ),
            InlineKeyboardButton(
                text="🔙 بازگشت به ویزارد",
                callback_data=DeepTalkCB(action="new", sid=owner_id).pack(),
            ),
        ]
    )
    return InlineKeyboardMarkup(inline_keyboard=rows)


def build_deeptalk_time_kb(
    owner_id: int,
    *,
    hour: int,
    minute: int,
) -> InlineKeyboardMarkup:
    """Interactive clock time picker for DeepTalk start time."""
    rows: list[list[InlineKeyboardButton]] = [
        [
            InlineKeyboardButton(
                text="🌤 17:00",
                callback_data=DeepTalkCB(action="tset", sid=owner_id, arg="17.00").pack(),
            ),
            InlineKeyboardButton(
                text="🌆 19:00",
                callback_data=DeepTalkCB(action="tset", sid=owner_id, arg="19.00").pack(),
            ),
            InlineKeyboardButton(
                text="🔥 21:00",
                callback_data=DeepTalkCB(action="tset", sid=owner_id, arg="21.00").pack(),
            ),
            InlineKeyboardButton(
                text="🌙 22:00",
                callback_data=DeepTalkCB(action="tset", sid=owner_id, arg="22.00").pack(),
            ),
            InlineKeyboardButton(
                text="🌌 23:00",
                callback_data=DeepTalkCB(action="tset", sid=owner_id, arg="23.00").pack(),
            ),
        ],
        [
            InlineKeyboardButton(
                text="-۳س",
                callback_data=DeepTalkCB(action="hadd", sid=owner_id, arg="-3").pack(),
            ),
            InlineKeyboardButton(
                text="-۱س",
                callback_data=DeepTalkCB(action="hadd", sid=owner_id, arg="-1").pack(),
            ),
            InlineKeyboardButton(
                text=f"ساعت {hour:02d}",
                callback_data=_noop(owner_id),
            ),
            InlineKeyboardButton(
                text="+۱س",
                callback_data=DeepTalkCB(action="hadd", sid=owner_id, arg="1").pack(),
            ),
            InlineKeyboardButton(
                text="+۳س",
                callback_data=DeepTalkCB(action="hadd", sid=owner_id, arg="3").pack(),
            ),
        ],
        [
            InlineKeyboardButton(
                text="-۱۵د",
                callback_data=DeepTalkCB(action="madd", sid=owner_id, arg="-15").pack(),
            ),
            InlineKeyboardButton(
                text="-۵د",
                callback_data=DeepTalkCB(action="madd", sid=owner_id, arg="-5").pack(),
            ),
            InlineKeyboardButton(
                text=f"دقیقه {minute:02d}",
                callback_data=_noop(owner_id),
            ),
            InlineKeyboardButton(
                text="+۵د",
                callback_data=DeepTalkCB(action="madd", sid=owner_id, arg="5").pack(),
            ),
            InlineKeyboardButton(
                text="+۱۵د",
                callback_data=DeepTalkCB(action="madd", sid=owner_id, arg="15").pack(),
            ),
        ],
        [
            InlineKeyboardButton(
                text="⌨️ تایپ ساعت دلخواه",
                callback_data=DeepTalkCB(action="scus", sid=owner_id).pack(),
            ),
            InlineKeyboardButton(
                text="📅 تغییر تاریخ",
                callback_data=DeepTalkCB(action="cal", sid=owner_id).pack(),
            ),
        ],
        [
            InlineKeyboardButton(
                text="✅ تأیید ساعت و بازگشت به ویزارد",
                callback_data=DeepTalkCB(action="new", sid=owner_id).pack(),
            )
        ],
    ]
    return InlineKeyboardMarkup(inline_keyboard=rows)


def build_deeptalk_cancel_input_kb(owner_id: int, back_screen: str = "new") -> InlineKeyboardMarkup:
    """Cancel button displayed during FSM text/forward prompts."""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="🔙 انصراف و بازگشت",
                    callback_data=DeepTalkCB(action="back", sid=owner_id, arg=back_screen).pack(),
                )
            ]
        ]
    )


def build_deeptalk_groups_kb(
    owner_id: int,
    groups: list[dict[str, Any]],
    *,
    selected_chat_id: int | None,
    page: int = 0,
) -> InlineKeyboardMarkup:
    """Paginated group selector when multiple groups exist."""
    rows: list[list[InlineKeyboardButton]] = []
    page_items, safe_page, total_pages = paginate(groups, page, PAGE_SIZE)

    for g in page_items:
        cid = int(g["chat_id"])
        title = g.get("title") or f"گروه {cid}"
        mark = "✅ " if selected_chat_id == cid else "🏢 "
        rows.append(
            [
                InlineKeyboardButton(
                    text=f"{mark}{_truncate(str(title), 30)}",
                    callback_data=DeepTalkCB(action="grp", sid=owner_id, arg=str(cid)).pack(),
                )
            ]
        )

    if total_pages > 1:
        nav: list[InlineKeyboardButton] = []
        if safe_page > 0:
            nav.append(
                InlineKeyboardButton(
                    text="⬅️ قبلی",
                    callback_data=DeepTalkCB(action="grp", sid=owner_id, arg=f"p{safe_page - 1}").pack(),
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
                    callback_data=DeepTalkCB(action="grp", sid=owner_id, arg=f"p{safe_page + 1}").pack(),
                )
            )
        rows.append(nav)

    rows.append(
        [
            InlineKeyboardButton(
                text="🔙 بازگشت",
                callback_data=DeepTalkCB(action="home", sid=owner_id).pack(),
            )
        ]
    )
    return InlineKeyboardMarkup(inline_keyboard=rows)


def build_deeptalk_topics_kb(
    owner_id: int,
    topics: list[dict[str, Any]],
    *,
    selected_thread_id: int | None,
    return_to: str = "home",
    page: int = 0,
) -> InlineKeyboardMarkup:
    """Paginated topic picker for choosing the DeepTalk target topic."""
    rows: list[list[InlineKeyboardButton]] = []
    page_items, safe_page, total_pages = paginate(topics, page, PAGE_SIZE)
    mode_prefix = "w" if return_to == "new" else "p"

    for t in page_items:
        tid = int(t["thread_id"])
        name = str(t.get("name") or tid)
        mark = "✅ " if selected_thread_id == tid else "📂 "
        rows.append(
            [
                InlineKeyboardButton(
                    text=f"{mark}{_truncate(name, 30)}",
                    callback_data=DeepTalkCB(
                        action="tops",
                        sid=owner_id,
                        arg=f"{mode_prefix}_{tid}",
                    ).pack(),
                )
            ]
        )

    if total_pages > 1:
        nav: list[InlineKeyboardButton] = []
        if safe_page > 0:
            nav.append(
                InlineKeyboardButton(
                    text="⬅️ قبلی",
                    callback_data=DeepTalkCB(
                        action="top",
                        sid=owner_id,
                        arg=f"{mode_prefix}{safe_page - 1}",
                    ).pack(),
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
                    callback_data=DeepTalkCB(
                        action="top",
                        sid=owner_id,
                        arg=f"{mode_prefix}{safe_page + 1}",
                    ).pack(),
                )
            )
        rows.append(nav)

    back_action = "new" if return_to == "new" else "home"
    rows.append(
        [
            InlineKeyboardButton(
                text="🔙 بازگشت",
                callback_data=DeepTalkCB(action=back_action, sid=owner_id).pack(),
            )
        ]
    )
    return InlineKeyboardMarkup(inline_keyboard=rows)


def build_deeptalk_sessions_list_kb(
    owner_id: int,
    sessions: list[dict[str, Any]],
    *,
    page: int = 0,
    tz_info: Any = None,
) -> InlineKeyboardMarkup:
    """Paginated list of active ('live') and upcoming ('scheduled') DeepTalk sessions."""
    rows: list[list[InlineKeyboardButton]] = []
    page_items, safe_page, total_pages = paginate(sessions, page, DEEPTALK_SESSIONS_PAGE_SIZE)

    for s in page_items:
        sid = int(s["id"])
        dot = "🟢" if str(s.get("status")) == "live" else "🟡"
        try:
            st_dt = parse_utc_iso(str(s["start_at"]))
            if tz_info is not None:
                st_dt = st_dt.astimezone(tz_info)
            time_str = st_dt.strftime("%H:%M")
        except Exception:
            time_str = ""
        subj = _truncate(str(s.get("subject") or "دیپ‌تاک"), 20)
        rows.append(
            [
                InlineKeyboardButton(
                    text=f"{dot} {time_str} • {subj}",
                    callback_data=DeepTalkCB(action="sess", sid=owner_id, arg=str(sid)).pack(),
                )
            ]
        )

    if total_pages > 1:
        nav: list[InlineKeyboardButton] = []
        if safe_page > 0:
            nav.append(
                InlineKeyboardButton(
                    text="⬅️ قبلی",
                    callback_data=DeepTalkCB(action="list", sid=owner_id, arg=str(safe_page - 1)).pack(),
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
                    callback_data=DeepTalkCB(action="list", sid=owner_id, arg=str(safe_page + 1)).pack(),
                )
            )
        rows.append(nav)

    rows.append(
        [
            InlineKeyboardButton(
                text="➕ دیپ‌تاک جدید",
                callback_data=DeepTalkCB(action="new", sid=owner_id).pack(),
            ),
            InlineKeyboardButton(
                text="🔙 بازگشت",
                callback_data=DeepTalkCB(action="home", sid=owner_id).pack(),
            ),
        ]
    )
    return InlineKeyboardMarkup(inline_keyboard=rows)


def build_deeptalk_session_manage_kb(
    owner_id: int,
    session_id: int,
    *,
    status: str,
    bot_username: str | None,
) -> InlineKeyboardMarkup:
    """Live session control keyboard (extend +10/+15/+30m, start early, end now, cancel)."""
    rows: list[list[InlineKeyboardButton]] = []
    if status == "live":
        rows.append(
            [
                InlineKeyboardButton(
                    text="⏱ +۱۰ دقیقه",
                    callback_data=DeepTalkCB(action="ext", sid=owner_id, arg=f"{session_id}_10").pack(),
                ),
                InlineKeyboardButton(
                    text="⏱ +۱۵ دقیقه",
                    callback_data=DeepTalkCB(action="ext", sid=owner_id, arg=f"{session_id}_15").pack(),
                ),
                InlineKeyboardButton(
                    text="⏱ +۳۰ دقیقه",
                    callback_data=DeepTalkCB(action="ext", sid=owner_id, arg=f"{session_id}_30").pack(),
                ),
            ]
        )
        rows.append(
            [
                InlineKeyboardButton(
                    text="⏹ پایان دستی و جمع‌بندی",
                    callback_data=DeepTalkCB(action="enow", sid=owner_id, arg=f"{session_id}_ask").pack(),
                )
            ]
        )
    elif status == "scheduled":
        rows.append(
            [
                InlineKeyboardButton(
                    text="🚀 شروع همین الان!",
                    callback_data=DeepTalkCB(action="snow", sid=owner_id, arg=str(session_id)).pack(),
                )
            ]
        )
        rows.append(
            [
                InlineKeyboardButton(
                    text="❌ لغو این دیپ‌تاک",
                    callback_data=DeepTalkCB(action="canc", sid=owner_id, arg=f"{session_id}_ask").pack(),
                )
            ]
        )

    rows.append([_participants_pv_button(session_id, bot_username, label="👀 لیست شرکت‌کنندگان (در پیوی)")])
    rows.append(
        [
            InlineKeyboardButton(
                text="🔄 بروزرسانی",
                callback_data=DeepTalkCB(action="sess", sid=owner_id, arg=str(session_id)).pack(),
            ),
            InlineKeyboardButton(
                text="🔙 لیست جلسات",
                callback_data=DeepTalkCB(action="list", sid=owner_id, arg="0").pack(),
            ),
        ]
    )
    return InlineKeyboardMarkup(inline_keyboard=rows)


def build_deeptalk_confirm_action_kb(
    owner_id: int,
    *,
    confirm_action: str,
    session_id: int,
    confirm_label: str,
) -> InlineKeyboardMarkup:
    """Two-step confirmation keyboard for manual end or session cancellation."""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=confirm_label,
                    callback_data=DeepTalkCB(
                        action=confirm_action,
                        sid=owner_id,
                        arg=f"{session_id}_yes",
                    ).pack(),
                ),
                InlineKeyboardButton(
                    text="🔙 انصراف",
                    callback_data=DeepTalkCB(action="sess", sid=owner_id, arg=str(session_id)).pack(),
                ),
            ]
        ]
    )


def build_deeptalk_admins_kb(
    owner_id: int,
    admins: list[dict[str, Any]],
    *,
    page: int = 0,
) -> InlineKeyboardMarkup:
    """Owner-only DeepTalk admin management keyboard (add & 1-tap revoke)."""
    rows: list[list[InlineKeyboardButton]] = [
        [
            InlineKeyboardButton(
                text="➕ افزودن ادمین دیپ‌تاک جدید",
                callback_data=DeepTalkCB(action="adma", sid=owner_id).pack(),
            )
        ]
    ]

    page_items, safe_page, total_pages = paginate(admins, page, DEEPTALK_ADMINS_PAGE_SIZE)
    for a in page_items:
        uid = int(a["user_id"])
        name = _truncate(str(a.get("full_name") or uid), 18)
        uname = f" (@{a['username']})" if a.get("username") else ""
        rows.append(
            [
                InlineKeyboardButton(
                    text=f"🗑 حذف دسترسی: {name}{uname}",
                    callback_data=DeepTalkCB(action="admd", sid=owner_id, arg=str(uid)).pack(),
                )
            ]
        )

    if total_pages > 1:
        nav: list[InlineKeyboardButton] = []
        if safe_page > 0:
            nav.append(
                InlineKeyboardButton(
                    text="⬅️ قبلی",
                    callback_data=DeepTalkCB(action="adm", sid=owner_id, arg=str(safe_page - 1)).pack(),
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
                    callback_data=DeepTalkCB(action="adm", sid=owner_id, arg=str(safe_page + 1)).pack(),
                )
            )
        rows.append(nav)

    rows.append(
        [
            InlineKeyboardButton(
                text="🔙 بازگشت به پنل دیپ‌تاک",
                callback_data=DeepTalkCB(action="home", sid=owner_id).pack(),
            )
        ]
    )
    return InlineKeyboardMarkup(inline_keyboard=rows)

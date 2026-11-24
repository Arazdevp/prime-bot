"""20 distinct Persian DeepTalk check-in messages, shuffle-bag draw, and HTML card formatters."""

from datetime import datetime
import random
from typing import Any
from zoneinfo import ZoneInfo
from bot.utils.dates import format_persian_date
from bot.utils.html_utils import escape_html
from bot.utils.timer_parser import (
    format_duration_persian,
    format_fire_time_persian,
    format_remaining_persian,
    parse_utc_iso,
    to_persian_digits,
)

# 20 distinct, energetic Persian check-in templates (every entry contains {user} and balanced Telegram HTML)
DEEPTALK_CHECKIN_MESSAGES: list[str] = [
    "🎙 {user} اعلام حضور کرد! خوش اومدی به دیپ‌تاک 🔥",
    "✅ {user} حضور خودش رو تأیید کرد؛ بفرما بالا رفیق! 🙌",
    "🚀 {user} وارد دیپ‌تاک شد! ببینیم چه ایده‌ای تو آستین داره 😎",
    "⚡ میکروفون باز شد؛ {user} هم به جمع دیپ‌تاک پیوست!",
    "🔥 به به! {user} با انرژیِ تمام پرید وسط میدان گفتگو!",
    "🪑 صندلی ردیف اول رزرو شد؛ {user} اعلام حضور کرد! ✨",
    "💡 {user} چراغ ورودش رو روشن کرد و وارد بحث شد!",
    "🎯 حضور {user} ثبت شد! آماده‌ایم نظرت رو بشنویم رفیق 👀",
    "🌊 موج دیپ‌تاک قوی‌تر شد؛ {user} هم به گفتگو اضافه شد! 🔥",
    "☕ {user} با دستِ پر وارد دیپ‌تاک شد! خوش اومدی 🌟",
    "🫡 گزارش از اتاق دیپ‌ورک: {user} رسماً وارد جلسه شد!",
    "🎬 سکانسِ گفتگو داغ‌تر شد؛ {user} اعلام حضور کرد! ⚡",
    "🧠 یه ذهن خلاق دیگه رسید! {user} حضورش رو در دیپ‌تاک ثبت کرد 🔥",
    "🙌 دستا بالا! {user} هم وارد حلقه دیپ‌تاک شد!",
    "🔔 دینگ! {user} کارت حضورش رو زد و اومد تو بحث 😎",
    "🌟 جمعِ دیپ‌تاک تکمیل‌تر شد؛ خوش اومدی {user} عزیز!",
    "⚡ با ورود {user} تنور این دیپ‌تاک رسماً داغ شد! 🔥",
    "🎤 نوبت تریبون رسید؛ {user} اعلام حضور کرد و آماده بحثه!",
    "🚀 {user} کمربندش رو بست و وارد مدار دیپ‌تاک شد! ✨",
    "🤝 ایول! {user} حضور خودش رو تأیید کرد و به جمع پیوست 🔥",
]


def draw_next_checkin_index(
    deck: list[int] | None,
    last_index: int | None = None,
) -> tuple[int, list[int]]:
    """Draw the next check-in message index from a shuffle-bag without repeating until all 20 are used."""
    total = len(DEEPTALK_CHECKIN_MESSAGES)
    valid_deck: list[int] = []
    if isinstance(deck, list):
        seen: set[int] = set()
        for item in deck:
            if isinstance(item, int) and 0 <= item < total and item not in seen:
                valid_deck.append(item)
                seen.add(item)

    if not valid_deck:
        valid_deck = list(range(total))
        random.shuffle(valid_deck)
        if last_index is not None and total > 1 and valid_deck[0] == last_index:
            swap_pos = random.randrange(1, total)
            valid_deck[0], valid_deck[swap_pos] = valid_deck[swap_pos], valid_deck[0]

    chosen = valid_deck.pop(0)
    return chosen, valid_deck


def format_deeptalk_user_mention(
    user_id: int,
    username: str | None,
    full_name: str | None,
) -> str:
    """Format safe HTML mention showing clickable display name and @username if available."""
    safe_name = escape_html((full_name or "").strip() or "کاربر")
    link = f'<a href="tg://user?id={user_id}">{safe_name}</a>'
    if username and username.strip():
        clean_u = escape_html(username.strip().lstrip("@"))
        return f"<b>{link}</b> (@{clean_u})"
    return f"<b>{link}</b>"


def format_checkin_announcement(
    template_index: int,
    *,
    user_id: int,
    username: str | None,
    full_name: str | None,
) -> str:
    """Render one of the 20 check-in messages for a participant."""
    safe_idx = template_index % len(DEEPTALK_CHECKIN_MESSAGES)
    mention = format_deeptalk_user_mention(user_id, username, full_name)
    return DEEPTALK_CHECKIN_MESSAGES[safe_idx].format(user=mention)


def format_deeptalk_announcement_card(
    session: dict[str, Any],
    *,
    registered_count: int,
    now_tehran: datetime | None = None,
    tz_name: str = "Asia/Tehran",
) -> str:
    """Render the DeepTalk pre-registration / announcement card in the target topic."""
    tz = ZoneInfo(tz_name)
    now = now_tehran.astimezone(tz) if now_tehran else datetime.now(tz)
    start_local = parse_utc_iso(str(session["start_at"])).astimezone(tz)
    end_local = parse_utc_iso(str(session["end_at"])).astimezone(tz)

    subject = escape_html(str(session.get("subject") or "گفتگوی آزاد"))
    desc = session.get("description")
    creator = escape_html(str(session.get("creator_name") or "ادمین دیپ‌تاک"))
    dur_str = format_duration_persian(int(session.get("duration_minutes") or 45))
    start_str = format_fire_time_persian(start_local, now, tz_name)
    end_clock = to_persian_digits(end_local.strftime("%H:%M"))
    date_full = format_persian_date(start_local)
    status = str(session.get("status") or "scheduled")

    if status == "live":
        header = "🟢 <b>فراخوان دیپ‌تاک — هم‌اکنون در حال برگزاری!</b>"
        footer = "🔥 <i>این دیپ‌تاک شروع شده! روی پیام شروع در پایین‌تر کلیک کن و اعلام حضور بزن.</i>"
    elif status == "completed":
        header = "🏁 <b>فراخوان دیپ‌تاک — برگزار شد</b>"
        footer = "✨ <i>این جلسه به پایان رسیده است.</i>"
    elif status == "cancelled":
        header = "🔴 <b>فراخوان دیپ‌تاک — لغو شد</b>"
        footer = "⚠️ <i>این جلسه توسط برگزارکننده لغو شد.</i>"
    else:
        rem = format_remaining_persian(start_local - now)
        header = "🎙 <b>فراخوان دیپ‌تاک جدید!</b>"
        footer = (
            f"⏳ <i>شروع تا حدود {rem}</i>\n\n"
            "اگه پایه شرکت در این گفتگو هستی، روی <b>«🙋‍♂️ منم هستم!»</b> بزن تا اسمت ثبت بشه 👇"
        )

    lines = [
        header,
        "",
        f"🎯 <b>محوریت گفتگو:</b>\n<b>{subject}</b>",
    ]
    if desc:
        lines.append(f"\n💡 <b>نکات / سوالات کلیدی:</b>\n<blockquote>{escape_html(str(desc))}</blockquote>")

    lines.extend(
        [
            "",
            f"📅 <b>زمان شروع:</b> {to_persian_digits(start_str)} <i>({date_full})</i>",
            f"⏳ <b>مدت گفتگو:</b> {dur_str} (تا ساعت {end_clock})",
            f"🎙 <b>میزبان:</b> {creator}",
            f"👥 <b>افراد ثبت‌نام‌شده:</b> <b>{to_persian_digits(registered_count)} نفر</b>",
            "",
            footer,
        ]
    )
    return "\n".join(lines)


def format_deeptalk_start_card(
    session: dict[str, Any],
    *,
    checked_in_count: int,
    registered_count: int,
    tz_name: str = "Asia/Tehran",
) -> str:
    """Render the live 'DeepTalk Started!!' card posted when start_at arrives."""
    tz = ZoneInfo(tz_name)
    end_local = parse_utc_iso(str(session["end_at"])).astimezone(tz)
    end_clock = to_persian_digits(end_local.strftime("%H:%M"))
    dur_str = format_duration_persian(int(session.get("duration_minutes") or 45))
    subject = escape_html(str(session.get("subject") or "گفتگوی آزاد"))
    desc = session.get("description")
    status = str(session.get("status") or "live")

    if status == "completed":
        header = "🏁 <b>این نوبت دیپ‌تاک به پایان رسید!</b>"
        cta = "✅ <i>مهلت اعلام حضور تمام شد؛ جمع‌بندی نهایی را در پیام پایین ببینید.</i>"
    elif status == "cancelled":
        header = "🔴 <b>این دیپ‌تاک متوقف / لغو شد.</b>"
        cta = ""
    else:
        header = "🔥 <b>بحث دیپ‌تاک رسماً شروع شد!!</b>"
        cta = "👇 <b>با دکمه شیشه‌ای زیر اعلام حضور کنید و در بحث شرکت کنید!!</b>"

    lines = [
        header,
        "",
        f"🎯 <b>محوریت بحث:</b> <b>{subject}</b>",
    ]
    if desc:
        lines.append(f"💡 <b>سوال / سرنخ بحث:</b>\n<blockquote>{escape_html(str(desc))}</blockquote>")

    lines.extend(
        [
            "",
            f"⏳ <b>زمان گفتگو:</b> {dur_str} (تا ساعت <b>{end_clock}</b> تهران)",
            f"👥 <b>ثبت‌نام اولیه:</b> {to_persian_digits(registered_count)} نفر  |  "
            f"✋ <b>حاضرین تا الان:</b> <b>{to_persian_digits(checked_in_count)} نفر</b>",
        ]
    )
    if cta:
        lines.extend(["", cta])
    return "\n".join(lines)


def format_deeptalk_dm_reminder(
    session: dict[str, Any],
    *,
    group_title: str | None,
    topic_name: str | None,
    tz_name: str = "Asia/Tehran",
) -> str:
    """Render the private DM reminder sent at start_at to pre-registered participants."""
    tz = ZoneInfo(tz_name)
    end_local = parse_utc_iso(str(session["end_at"])).astimezone(tz)
    end_clock = to_persian_digits(end_local.strftime("%H:%M"))
    subject = escape_html(str(session.get("subject") or "گفتگوی آزاد"))

    lines = [
        "🔔 <b>دیپ‌تاکی که توش ثبت‌نام کرده بودی شروع شد!!</b> 🔥",
        "",
    ]
    if group_title:
        lines.append(f"🏢 <b>گروه:</b> {escape_html(group_title)}")
    if topic_name:
        lines.append(f"📂 <b>تاپیک:</b> {escape_html(topic_name)}")
    lines.extend(
        [
            f"🎯 <b>موضوع:</b> <b>{subject}</b>",
            f"⏰ <b>مهلت بحث:</b> تا ساعت {end_clock} (تهران)",
            "",
            "بزن روی دکمه زیر، بپر توی تاپیک و روی <b>«✋ اعلام حضور»</b> کلیک کن! 🚀",
        ]
    )
    return "\n".join(lines)


def format_deeptalk_end_card(
    session: dict[str, Any],
    stats: dict[str, Any],
) -> str:
    """Render the closing wrap-up card with fun attendance & participation stats."""
    subject = escape_html(str(session.get("subject") or "گفتگوی آزاد"))
    dur_str = format_duration_persian(int(session.get("duration_minutes") or 45))

    reg_cnt = int(stats.get("registered_count") or 0)
    chk_cnt = int(stats.get("checked_in_count") or 0)
    on_time = int(stats.get("on_time_count") or 0)
    walk_in = int(stats.get("walk_in_count") or 0)
    total_msgs = int(stats.get("total_messages") or 0)
    first_resp = stats.get("first_responder")
    top_speakers: list[dict[str, Any]] = stats.get("top_speakers") or []

    lines = [
        "⏰ <b>وقت دیپ‌تاک به پایان رسید! می‌تونید جمع‌بندی کنید!</b> 🏁",
        "",
        f"🎯 <b>محوریت گفتگو:</b> <b>{subject}</b>",
        f"⏳ <b>مدت برگزاری:</b> {dur_str}",
        "",
        "📊 <b>گزارش بامزه این نوبت دیپ‌تاک:</b>",
        f"• 👥 <b>حاضرین جلسه:</b> <b>{to_persian_digits(chk_cnt)} نفر</b> "
        f"(از {to_persian_digits(reg_cnt)} ثبت‌نام اولیه)",
    ]

    if chk_cnt > 0:
        detail_bits: list[str] = []
        if on_time > 0:
            detail_bits.append(f"⭐ {to_persian_digits(on_time)} خوش‌قول")
        if walk_in > 0:
            detail_bits.append(f"⚡ {to_persian_digits(walk_in)} ورود یهویی")
        if detail_bits:
            lines.append(f"• 🎖 <b>ترکیب تیم:</b> {' + '.join(detail_bits)}")

    if first_resp:
        fr_mention = format_deeptalk_user_mention(
            int(first_resp["user_id"]),
            first_resp.get("username"),
            first_resp.get("full_name"),
        )
        lines.append(f"• 🥇 <b>سریع‌ترین اعلام حضور:</b> {fr_mention}")

    if total_msgs > 0:
        lines.append(f"• 💬 <b>پیام‌های رد و بدل شده در بحث:</b> <b>{to_persian_digits(total_msgs)} پیام</b>")

    if top_speakers:
        medals = ("🥇", "🥈", "🥉")
        speaker_parts: list[str] = []
        for i, sp in enumerate(top_speakers[:3]):
            medal = medals[i] if i < len(medals) else "•"
            sp_name = escape_html((sp.get("full_name") or "کاربر").strip())
            sp_cnt = to_persian_digits(int(sp.get("message_count") or 0))
            speaker_parts.append(f"{medal} <b>{sp_name}</b> ({sp_cnt} پیام)")
        lines.append("• 🏆 <b>پرمشارکت‌ترین‌های بحث:</b> " + " ، ".join(speaker_parts))

    lines.extend(
        [
            "",
            "دم همه‌ی بچه‌هایی که وقت گذاشتن و توی گفتگو شرکت کردن گرم! 🔥🙌\n"
            "📝 <b>نوبت جمع‌بندی:</b> اگه نکته طلایی، نتیجه‌گیری یا آورده‌ای از بحث داشتین، همین پایین بنویسین 👇",
        ]
    )
    return "\n".join(lines)


def _format_pv_person_line(idx: int, p: dict[str, Any], badge: str) -> str:
    uid = int(p["user_id"])
    safe_name = escape_html((p.get("full_name") or "کاربر").strip() or "کاربر")
    raw_u = (p.get("username") or "").strip().lstrip("@")
    u_str = f"@{escape_html(raw_u)}" if raw_u else "<i>(بدون یوزرنیم)</i>"
    msg_cnt = int(p.get("message_count") or 0)
    msg_note = f" • 💬 {to_persian_digits(msg_cnt)}" if msg_cnt > 0 else ""
    return (
        f"{to_persian_digits(idx)}. <a href=\"tg://user?id={uid}\"><b>{safe_name}</b></a> — "
        f"{u_str} {badge}{msg_note}"
    )


def format_deeptalk_participants_pv(
    session: dict[str, Any],
    participants: list[dict[str, Any]],
    *,
    group_title: str | None = None,
    topic_name: str | None = None,
    tz_name: str = "Asia/Tehran",
) -> str:
    """Render the detailed participant list with @usernames in private chat."""
    tz = ZoneInfo(tz_name)
    start_local = parse_utc_iso(str(session["start_at"])).astimezone(tz)
    start_str = f"{format_persian_date(start_local)} ساعت {to_persian_digits(start_local.strftime('%H:%M'))}"
    subject = escape_html(str(session.get("subject") or "گفتگوی آزاد"))
    status = str(session.get("status") or "scheduled")

    status_map = {
        "scheduled": "🟡 در انتظار شروع (پیش‌ثبت‌نام)",
        "live": "🟢 در حال برگزاری",
        "completed": "🏁 پایان یافته",
        "cancelled": "🔴 لغو شده",
    }
    status_label = status_map.get(status, status)

    lines = [
        "📋 <b>لیست شرکت‌کنندگان دیپ‌تاک</b>",
        "",
        f"🎯 <b>محوریت:</b> <b>{subject}</b>",
    ]
    if group_title:
        topic_part = f" / {escape_html(topic_name)}" if topic_name else ""
        lines.append(f"📍 <b>مکان:</b> {escape_html(group_title)}{topic_part}")
    lines.extend(
        [
            f"🕒 <b>زمان شروع:</b> {start_str}",
            f"📡 <b>وضعیت:</b> {status_label}",
            "",
        ]
    )

    if not participants:
        lines.append("فعلاً کسی ثبت‌نام یا اعلام حضور نکرده! اولین نفر باش 😎🔥")
        return "\n".join(lines)

    if status == "scheduled":
        reg_list = [p for p in participants if int(p.get("registered") or 0) == 1]
        lines.append(f"🙋‍♂️ <b>افراد ثبت‌نام‌شده ({to_persian_digits(len(reg_list))} نفر):</b>")
        for idx, p in enumerate(reg_list, start=1):
            lines.append(_format_pv_person_line(idx, p, ""))
        return "\n".join(lines)

    checked_list = [p for p in participants if int(p.get("checked_in") or 0) == 1]
    waiting_or_missed = [
        p for p in participants
        if int(p.get("registered") or 0) == 1 and int(p.get("checked_in") or 0) == 0
    ]

    if checked_list:
        lines.append(f"🟢 <b>حاضرین در گفتگو ({to_persian_digits(len(checked_list))} نفر):</b>")
        for idx, p in enumerate(checked_list, start=1):
            was_reg = int(p.get("registered") or 0) == 1
            badge = "<i>(⭐ خوش‌قول)</i>" if was_reg else "<i>(⚡ ورود یهویی)</i>"
            lines.append(_format_pv_person_line(idx, p, badge))
        lines.append("")

    if waiting_or_missed:
        if status == "live":
            lines.append(
                f"⏳ <b>ثبت‌نام‌شدگان منتظر اعلام حضور ({to_persian_digits(len(waiting_or_missed))} نفر):</b>"
            )
            badge = "<i>(منتظر ورود)</i>"
        else:
            lines.append(
                f"😴 <b>جامانده‌ها — ثبت‌نام کردن ولی نیومدن ({to_persian_digits(len(waiting_or_missed))} نفر):</b>"
            )
            badge = "<i>(غایب)</i>"
        for idx, p in enumerate(waiting_or_missed, start=1):
            lines.append(_format_pv_person_line(idx, p, badge))

    return "\n".join(lines).rstrip()

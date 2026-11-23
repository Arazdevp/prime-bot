"""Offline unit tests for the interactive /timer feature, Jalali calendar, scheduling, and firing."""

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from zoneinfo import ZoneInfo
import aiosqlite
import jdatetime
import pytest
from aiogram import Bot
from aiogram.exceptions import TelegramBadRequest, TelegramForbiddenError
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, Chat, ChatMemberMember, Message, User
from bot.database import repository
from bot.database.db import close_db, get_db, init_db
from bot.filters import clear_admin_cache
from bot.handlers.private import clear_stranger_cooldowns, cmd_private_start
from bot.handlers.timer import (
    ActiveTimerFsmFilter,
    TimerStates,
    clear_timer_runtime_state,
    cmd_timer,
    handle_timer_callback,
    handle_timer_note_input,
)
from bot.keyboards.timer import TimerCB, build_calendar_kb
from bot.scheduler.jobs import build_scheduler, set_scheduler
from bot.services.access import clear_member_cache
from bot.services.timer_service import (
    build_topic_deep_link,
    create_and_schedule_timer,
    fire_timer,
    format_user_mention,
    rehydrate_active_timers,
    set_bot_username,
    sweep_due_timers,
)
from bot.texts.private_texts import clear_last_picked_cache
from bot.utils.timer_parser import (
    format_duration_persian,
    jalali_month_days,
    parse_duration_input,
    parse_jalali_date_time_input,
    parse_quick_timer_command,
    parse_time_input,
    to_utc_iso,
)

TEHRAN_TZ = ZoneInfo("Asia/Tehran")


def _make_fsm_memory() -> tuple[FSMContext, dict]:
    store: dict = {"state": None, "data": {}}
    fsm = MagicMock(spec=FSMContext)

    async def _set_state(s):
        store["state"] = s.state if hasattr(s, "state") else s

    async def _get_state():
        return store["state"]

    async def _clear():
        store["state"] = None
        store["data"] = {}

    async def _update_data(**kwargs):
        store["data"].update(kwargs)

    async def _get_data():
        return dict(store["data"])

    fsm.set_state = AsyncMock(side_effect=_set_state)
    fsm.get_state = AsyncMock(side_effect=_get_state)
    fsm.clear = AsyncMock(side_effect=_clear)
    fsm.update_data = AsyncMock(side_effect=_update_data)
    fsm.get_data = AsyncMock(side_effect=_get_data)
    return fsm, store


def _make_group_msg(
    *,
    chat_id: int = -1001234567890,
    thread_id: int = 42,
    user_id: int = 5001,
    username: str | None = "arash_dev",
    first_name: str = "آرش",
    full_name: str = "آرش رضایی",
    text: str = "/timer",
    bot: Bot | None = None,
) -> Message:
    msg = MagicMock(spec=Message)
    msg.message_id = 100
    msg.chat = MagicMock(spec=Chat)
    msg.chat.id = chat_id
    msg.chat.type = "supergroup"
    msg.is_topic_message = thread_id != 1
    msg.message_thread_id = None if thread_id == 1 else thread_id
    msg.text = text

    user = MagicMock(spec=User)
    user.id = user_id
    user.is_bot = False
    user.username = username
    user.first_name = first_name
    user.full_name = full_name
    msg.from_user = user

    msg.bot = bot or AsyncMock(spec=Bot)
    msg.delete = AsyncMock()
    msg.answer = AsyncMock()
    return msg


def _make_callback(
    *,
    user_id: int = 5001,
    chat_id: int = -1001234567890,
    message_id: int = 200,
    thread_id: int = 42,
    chat_type: str = "supergroup",
    username: str | None = "arash_dev",
    first_name: str = "آرش",
    full_name: str = "آرش رضایی",
    html_text: str = "⏰ تایمرت به صدا در اومد!",
) -> CallbackQuery:
    cb = MagicMock(spec=CallbackQuery)
    cb.from_user = MagicMock(spec=User)
    cb.from_user.id = user_id
    cb.from_user.is_bot = False
    cb.from_user.username = username
    cb.from_user.first_name = first_name
    cb.from_user.full_name = full_name
    cb.answer = AsyncMock()

    msg = MagicMock(spec=Message)
    msg.message_id = message_id
    msg.chat = MagicMock(spec=Chat)
    msg.chat.id = chat_id
    msg.chat.type = chat_type
    msg.is_topic_message = (chat_type == "supergroup" and thread_id != 1)
    msg.message_thread_id = None if (chat_type == "private" or thread_id == 1) else thread_id
    msg.html_text = html_text
    msg.text = html_text
    msg.edit_text = AsyncMock()
    msg.delete = AsyncMock()
    cb.message = msg
    return cb


# ---------------- 1. Callback Data & Calendar Grid ----------------


def test_timer_callback_data_under_64_bytes_worst_case():
    """Every TimerCB action with 64-bit owner_id and worst-case arg MUST stay <= 64 bytes."""
    worst_uid = 9223372036854775807  # 19 digits
    cases = [
        TimerCB(action="home", owner_id=worst_uid, arg=""),
        TimerCB(action="q", owner_id=worst_uid, arg="60"),
        TimerCB(action="rel", owner_id=worst_uid, arg=""),
        TimerCB(action="rset", owner_id=worst_uid, arg="720"),
        TimerCB(action="radd", owner_id=worst_uid, arg="-15"),
        TimerCB(action="rcus", owner_id=worst_uid, arg=""),
        TimerCB(action="cal", owner_id=worst_uid, arg="1406-12"),
        TimerCB(action="day", owner_id=worst_uid, arg="1406-12-29"),
        TimerCB(action="dcus", owner_id=worst_uid, arg=""),
        TimerCB(action="tm", owner_id=worst_uid, arg=""),
        TimerCB(action="tset", owner_id=worst_uid, arg="23.00"),
        TimerCB(action="hadd", owner_id=worst_uid, arg="-3"),
        TimerCB(action="madd", owner_id=worst_uid, arg="-15"),
        TimerCB(action="tcus", owner_id=worst_uid, arg=""),
        TimerCB(action="note", owner_id=worst_uid, arg="rel"),
        TimerCB(action="nclr", owner_id=worst_uid, arg="tm"),
        TimerCB(action="save", owner_id=worst_uid, arg=""),
        TimerCB(action="list", owner_id=worst_uid, arg="999"),
        TimerCB(action="del", owner_id=worst_uid, arg="999999999_99"),
        TimerCB(action="dela", owner_id=worst_uid, arg="yes"),
        TimerCB(action="snz", owner_id=worst_uid, arg="999999999_30"),
        TimerCB(action="ack", owner_id=worst_uid, arg="999999999"),
        TimerCB(action="back", owner_id=worst_uid, arg="home"),
        TimerCB(action="cls", owner_id=worst_uid, arg=""),
        TimerCB(action="noop", owner_id=worst_uid, arg=""),
    ]
    for cb in cases:
        packed = cb.pack()
        assert len(packed.encode("utf-8")) <= 64, f"{packed!r} exceeded 64 bytes"


def test_jalali_month_days_and_calendar_keyboard_bounds():
    """Verify leap/non-leap Jalali month lengths and Telegram keyboard button limits."""
    assert jalali_month_days(1405, 1) == 31
    assert jalali_month_days(1405, 7) == 30
    assert jalali_month_days(1405, 12) == 29  # 1405 is non-leap
    assert jalali_month_days(1403, 12) == 30  # 1403 is leap

    today = jdatetime.date(1405, 9, 2)
    selected = jdatetime.date(1405, 9, 5)
    kb = build_calendar_kb(5001, cal_year=1405, cal_month=9, selected_jdate=selected, today_jdate=today)

    total_buttons = sum(len(row) for row in kb.inline_keyboard)
    assert total_buttons <= 100
    assert all(len(row) <= 7 for row in kb.inline_keyboard)

    # Prev month arrow must be disabled on current month
    nav_row = kb.inline_keyboard[1]
    assert nav_row[0].text == "·"
    assert nav_row[2].text == "▶️"

    # Day 1 is in the past -> disabled; Day 2 is today -> 📍; Day 5 is selected -> ✅
    all_texts = [btn.text for row in kb.inline_keyboard for btn in row]
    assert "📍۲" in all_texts
    assert "✅۵" in all_texts
    assert "۱" not in all_texts


# ---------------- 2. Pure Parser Tests ----------------


def test_duration_time_datetime_and_quick_command_parsers():
    """Verify ASCII/Persian/Arabic parsing for durations, times, Jalali dates, and quick commands."""
    # Durations
    assert parse_duration_input("30") == 30
    assert parse_duration_input("۴۵") == 45
    assert parse_duration_input("30m") == 30
    assert parse_duration_input("۳۰د") == 30
    assert parse_duration_input("2h") == 120
    assert parse_duration_input("۲س") == 120
    assert parse_duration_input("1h30m") == 90
    assert parse_duration_input("۱س ۳۰د") == 90
    assert parse_duration_input("۲ ساعت و ۱۵ دقیقه") == 135
    assert parse_duration_input("1:45") == 105
    assert parse_duration_input("2d") == 2880
    assert parse_duration_input("0") is None
    assert parse_duration_input("-10") is None
    assert parse_duration_input("30درخت") is None
    assert parse_duration_input("abc") is None

    assert format_duration_persian(90) == "۱ ساعت و ۳۰ دقیقه"

    # Times
    assert parse_time_input("14:30") == (14, 30)
    assert parse_time_input("۰۹:۰۵") == (9, 5)
    assert parse_time_input("9.15") == (9, 15)
    assert parse_time_input("21") == (21, 0)
    assert parse_time_input("25:00") is None
    assert parse_time_input("14:60") is None

    # Fixed Tehran "now": Monday 1405/09/02 (2026-11-23) 12:30 Tehran
    now_tehran = datetime(2026, 11, 23, 12, 30, 0, tzinfo=TEHRAN_TZ)

    # Jalali date + time input
    dt1 = parse_jalali_date_time_input("1405/09/10 14:30", now_tehran)
    assert dt1 is not None and dt1.hour == 14 and dt1.minute == 30
    assert jdatetime.date.fromgregorian(date=dt1.date()) == jdatetime.date(1405, 9, 10)

    dt2 = parse_jalali_date_time_input("فردا 09:00", now_tehran)
    assert dt2 is not None and dt2.date() == datetime(2026, 11, 24).date() and dt2.hour == 9

    dt3 = parse_jalali_date_time_input("پس فردا 18:15", now_tehran)
    assert dt3 is not None and dt3.date() == datetime(2026, 11, 25).date() and dt3.hour == 18

    # Time-only today future vs past rollover to tomorrow
    dt_future_today = parse_jalali_date_time_input("18:30", now_tehran)
    assert dt_future_today is not None and dt_future_today.date() == now_tehran.date()
    dt_rollover = parse_jalali_date_time_input("08:00", now_tehran)
    assert dt_rollover is not None and dt_rollover.date() == (now_tehran + timedelta(days=1)).date()

    # Past explicit date rejected
    assert parse_jalali_date_time_input("1405/09/01 10:00", now_tehran) is None

    # Quick shorthand commands
    q1 = parse_quick_timer_command("30m برم شام بخورم", now_tehran)
    assert q1 is not None and q1[0] == "relative" and q1[1] == now_tehran + timedelta(minutes=30)
    assert q1[2] == "برم شام بخورم"

    q2 = parse_quick_timer_command("۱س ۳۰د جلسه با تیم", now_tehran)
    assert q2 is not None and q2[0] == "relative" and q2[1] == now_tehran + timedelta(minutes=90)
    assert q2[2] == "جلسه با تیم"

    q3 = parse_quick_timer_command("45 ورزش", now_tehran)
    assert q3 is not None and q3[0] == "relative" and q3[1] == now_tehran + timedelta(minutes=45)
    assert q3[2] == "ورزش"

    q4 = parse_quick_timer_command("18:30 تماس با دکتر", now_tehran)
    assert q4 is not None and q4[0] == "scheduled" and q4[1].hour == 18 and q4[1].minute == 30
    assert q4[2] == "تماس با دکتر"

    q5 = parse_quick_timer_command("فردا 09:00 تحویل پروژه", now_tehran)
    assert q5 is not None and q5[0] == "scheduled" and q5[1].date() == datetime(2026, 11, 24).date()
    assert q5[2] == "تحویل پروژه"

    q6 = parse_quick_timer_command("1405/09/10 14:00 امتحان", now_tehran)
    assert q6 is not None and q6[0] == "scheduled" and q6[2] == "امتحان"

    assert parse_quick_timer_command("سلام چطوری", now_tehran) is None


# ---------------- 3. DB Migration v3 -> v4 & Atomic Concurrency ----------------


@pytest.mark.asyncio
async def test_migration_v3_to_v4_creates_timers_and_dm_started_column(tmp_path: Path):
    """Upgrading from user_version=3 adds timers table and bot_users.dm_started preserving rows."""
    db_file = tmp_path / "upgrade_v3_to_v4.db"
    conn = await aiosqlite.connect(str(db_file))
    await conn.executescript(
        """
        CREATE TABLE bot_users (
            user_id INTEGER PRIMARY KEY,
            username TEXT,
            full_name TEXT,
            first_seen TEXT NOT NULL,
            last_seen TEXT NOT NULL,
            last_variant_key TEXT,
            attempts_window_start TEXT,
            attempts_count INTEGER NOT NULL DEFAULT 0,
            ignored_until TEXT,
            notified INTEGER NOT NULL DEFAULT 0
        );
        INSERT INTO bot_users (user_id, username, full_name, first_seen, last_seen)
        VALUES (7001, 'old_user', 'کاربر قدیمی', '2026-11-20T00:00:00+00:00', '2026-11-20T00:00:00+00:00');
        PRAGMA user_version = 3;
        """
    )
    await conn.commit()
    await conn.close()

    try:
        await init_db(str(db_file))
        db = get_db()
        cur = await db.execute("PRAGMA user_version;")
        assert (await cur.fetchone())[0] == 4

        user = await repository.get_bot_user(7001)
        assert user is not None
        assert user["username"] == "old_user"
        assert user["dm_started"] == 0

        await repository.set_bot_user_dm_started(7001, True)
        assert await repository.is_bot_user_dm_started(7001) is True
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_atomic_claim_timer_never_fires_twice_and_enforces_user_limit(tmp_path: Path):
    """Concurrent claim_timer_for_firing calls must yield exactly ONE winner; per-user cap is enforced."""
    await init_db(str(tmp_path / "atomic_timer.db"))
    try:
        fire_utc = datetime.now(timezone.utc) + timedelta(minutes=10)
        created = await repository.create_timer(
            chat_id=-100111,
            thread_id=10,
            chat_type="supergroup",
            user_id=5001,
            username="arash",
            full_name="آرش",
            note="تست همزمانی",
            timer_type="relative",
            fire_at_iso=to_utc_iso(fire_utc),
        )

        # 10 concurrent claim attempts -> strictly 1 succeeds!
        results = await asyncio.gather(
            *(repository.claim_timer_for_firing(created["id"]) for _ in range(10))
        )
        winners = [r for r in results if r is not None]
        assert len(winners) == 1
        assert winners[0]["status"] == "fired"

        # Cancel by wrong user fails; cancel by owner succeeds
        t2 = await repository.create_timer(
            chat_id=-100111,
            thread_id=10,
            chat_type="supergroup",
            user_id=5001,
            username="arash",
            full_name="آرش",
            note=None,
            timer_type="relative",
            fire_at_iso=to_utc_iso(fire_utc),
        )
        assert await repository.cancel_timer(t2["id"], 9999) is False
        assert await repository.cancel_timer(t2["id"], 5001) is True
        assert await repository.cancel_timer(t2["id"], 5001) is False

        # Per-user max limit check
        bot = AsyncMock(spec=Bot)
        with patch("bot.services.timer_service.get_settings") as mock_cfg:
            mock_cfg.return_value.TIMER_MAX_PER_USER = 2
            mock_cfg.return_value.TIMEZONE = "Asia/Tehran"
            r1, err1 = await create_and_schedule_timer(
                bot,
                chat_id=-100111,
                thread_id=10,
                chat_type="supergroup",
                user_id=6001,
                username="u1",
                full_name="کاربر ۱",
                note="یک",
                timer_type="relative",
                fire_at=fire_utc,
            )
            r2, err2 = await create_and_schedule_timer(
                bot,
                chat_id=-100111,
                thread_id=10,
                chat_type="supergroup",
                user_id=6001,
                username="u1",
                full_name="کاربر ۱",
                note="دو",
                timer_type="relative",
                fire_at=fire_utc,
            )
            r3, err3 = await create_and_schedule_timer(
                bot,
                chat_id=-100111,
                thread_id=10,
                chat_type="supergroup",
                user_id=6001,
                username="u1",
                full_name="کاربر ۱",
                note="سه",
                timer_type="relative",
                fire_at=fire_utc,
            )
            assert r1 is not None and err1 is None
            assert r2 is not None and err2 is None
            assert r3 is None and err3 is not None and "سقف" in err3
    finally:
        await close_db()


# ---------------- 4. Simultaneous Group + DM Firing, Fallbacks & Rehydration ----------------


@pytest.mark.asyncio
async def test_fire_timer_sends_simultaneously_to_topic_and_dm_with_escaping(tmp_path: Path):
    """fire_timer sends to exact topic and DM simultaneously, escapes HTML, links topic, and handles General topic."""
    await init_db(str(tmp_path / "fire_test.db"))
    try:
        chat_id = -1009876543210
        await repository.upsert_known_group(chat_id, "گروه R&D <x>", is_forum=True)
        await repository.upsert_topic(chat_id, 55, name="تاپیک <گفتگو>")

        now_utc = datetime(2026, 11, 23, 10, 0, 0, tzinfo=timezone.utc)
        t_topic = await repository.create_timer(
            chat_id=chat_id,
            thread_id=55,
            chat_type="supergroup",
            user_id=5001,
            username="arash_dev",
            full_name="آرش <A&B>",
            note="برم شام <مهم> & کار",
            timer_type="relative",
            fire_at_iso=to_utc_iso(now_utc - timedelta(minutes=5)),  # 5m delayed -> delay notice
        )

        bot = AsyncMock(spec=Bot)
        bot.send_message = AsyncMock()

        with patch("bot.services.timer_service.get_settings") as mock_cfg:
            mock_cfg.return_value.TIMEZONE = "Asia/Tehran"
            fired = await fire_timer(bot, t_topic["id"], now_utc=now_utc)

        assert fired is True
        assert bot.send_message.call_count == 2

        calls_by_chat = {c.kwargs["chat_id"]: c.kwargs for c in bot.send_message.call_args_list}
        assert chat_id in calls_by_chat
        assert 5001 in calls_by_chat

        grp_call = calls_by_chat[chat_id]
        assert grp_call["message_thread_id"] == 55
        assert "@arash_dev" in grp_call["text"]
        assert "آرش &lt;A&amp;B&gt;" in grp_call["text"]
        assert "برم شام &lt;مهم&gt; &amp; کار" in grp_call["text"]
        assert "تأخیر" in grp_call["text"]

        dm_call = calls_by_chat[5001]
        assert "گروه R&amp;D &lt;x&gt;" in dm_call["text"]
        assert "تاپیک &lt;گفتگو&gt;" in dm_call["text"]
        dm_kb = dm_call["reply_markup"]
        assert dm_kb.inline_keyboard[0][0].url == "https://t.me/c/9876543210/55"
        assert await repository.is_bot_user_dm_started(5001) is True

        # Second fire attempt is a no-op!
        bot.send_message.reset_mock()
        assert await fire_timer(bot, t_topic["id"], now_utc=now_utc) is False
        bot.send_message.assert_not_called()

        # General topic (thread_id=1 -> message_thread_id=None) + unstarted DM (TelegramForbiddenError)
        t_gen = await repository.create_timer(
            chat_id=chat_id,
            thread_id=1,
            chat_type="supergroup",
            user_id=5002,
            username=None,
            full_name="پیت",
            note=None,
            timer_type="relative",
            fire_at_iso=to_utc_iso(now_utc),
        )

        async def _send_side_effect(**kwargs):
            if kwargs["chat_id"] == 5002:
                raise TelegramForbiddenError(method=MagicMock(), message="Forbidden: bot was blocked by the user")
            return MagicMock()

        bot.send_message = AsyncMock(side_effect=_send_side_effect)
        with patch("bot.services.timer_service.get_settings") as mock_cfg:
            mock_cfg.return_value.TIMEZONE = "Asia/Tehran"
            assert await fire_timer(bot, t_gen["id"], now_utc=now_utc) is True

        grp_calls = [c.kwargs for c in bot.send_message.call_args_list if c.kwargs["chat_id"] == chat_id]
        assert len(grp_calls) == 1
        assert grp_calls[0]["message_thread_id"] is None
        assert 'tg://user?id=5002' in grp_calls[0]["text"]
        assert await repository.is_bot_user_dm_started(5002) is False
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_fire_timer_falls_back_to_general_when_topic_deleted(tmp_path: Path):
    """If the original forum topic was deleted, group alarm retries cleanly in General (None)."""
    await init_db(str(tmp_path / "deleted_topic.db"))
    try:
        now_utc = datetime.now(timezone.utc)
        t = await repository.create_timer(
            chat_id=-100555,
            thread_id=777,
            chat_type="supergroup",
            user_id=5001,
            username="u",
            full_name="کاربر",
            note="تست",
            timer_type="relative",
            fire_at_iso=to_utc_iso(now_utc),
        )
        bot = AsyncMock(spec=Bot)
        seen_threads: list[int | None] = []

        async def _side_effect(**kwargs):
            if kwargs["chat_id"] == -100555:
                seen_threads.append(kwargs.get("message_thread_id"))
                if kwargs.get("message_thread_id") == 777:
                    raise TelegramBadRequest(method=MagicMock(), message="Bad Request: message thread not found")
            return MagicMock()

        bot.send_message = AsyncMock(side_effect=_side_effect)
        with patch("bot.services.timer_service.get_settings") as mock_cfg:
            mock_cfg.return_value.TIMEZONE = "Asia/Tehran"
            assert await fire_timer(bot, t["id"], now_utc=now_utc) is True

        assert seen_threads == [777, None]
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_rehydrate_and_sweep_timers(tmp_path: Path):
    """rehydrate_active_timers fires overdue timers immediately and schedules future ones on APScheduler."""
    await init_db(str(tmp_path / "rehydrate.db"))
    bot = AsyncMock(spec=Bot)
    bot.send_message = AsyncMock()
    scheduler = build_scheduler(bot)
    set_scheduler(scheduler)
    scheduler.start()
    try:
        now_utc = datetime.now(timezone.utc)
        overdue = await repository.create_timer(
            chat_id=5001,
            thread_id=1,
            chat_type="private",
            user_id=5001,
            username="arash",
            full_name="آرش",
            note="جامانده",
            timer_type="relative",
            fire_at_iso=to_utc_iso(now_utc - timedelta(minutes=3)),
        )
        future = await repository.create_timer(
            chat_id=5001,
            thread_id=1,
            chat_type="private",
            user_id=5001,
            username="arash",
            full_name="آرش",
            note="آینده",
            timer_type="relative",
            fire_at_iso=to_utc_iso(now_utc + timedelta(hours=2)),
        )

        with patch("bot.services.timer_service.get_settings") as mock_cfg:
            mock_cfg.return_value.TIMEZONE = "Asia/Tehran"
            count = await rehydrate_active_timers(bot, now_utc=now_utc)

        assert count == 2
        assert (await repository.get_timer(overdue["id"]))["status"] == "fired"
        assert (await repository.get_timer(future["id"]))["status"] == "active"
        assert scheduler.get_job(f"timer_{future['id']}") is not None
        # Private timer sends only once (to DM, not twice)
        assert bot.send_message.call_count == 1

        # Safety sweep catches anything due in the future when clock advances
        with patch("bot.services.timer_service.get_settings") as mock_cfg:
            mock_cfg.return_value.TIMEZONE = "Asia/Tehran"
            swept = await sweep_due_timers(bot, now_utc=now_utc + timedelta(hours=3))
        assert swept == 1
        assert (await repository.get_timer(future["id"]))["status"] == "fired"
    finally:
        scheduler.shutdown(wait=False)
        set_scheduler(None)
        await close_db()


# ---------------- 5. Interactive UI Wizard, Owner Lock, FSM Safety & Snooze ----------------


@pytest.mark.asyncio
async def test_interactive_timer_ui_wizard_owner_lock_fsm_and_snooze(tmp_path: Path):
    """End-to-end test of /timer command, owner lock, relative builder, note FSM, save, and snooze."""
    clear_timer_runtime_state()
    set_bot_username("my_test_bot")
    await init_db(str(tmp_path / "ui_wizard.db"))
    try:
        chat_id = -1001234567890
        await repository.upsert_topic(chat_id, 42, name="کافه")
        bot = AsyncMock(spec=Bot)
        sent_panel = MagicMock(spec=Message)
        sent_panel.message_id = 200
        sent_panel.chat = MagicMock(spec=Chat)
        sent_panel.chat.id = chat_id
        bot.send_message = AsyncMock(return_value=sent_panel)
        bot.edit_message_text = AsyncMock()
        fsm, store = _make_fsm_memory()

        with patch("bot.handlers.timer.get_settings") as mock_h_cfg, patch(
            "bot.services.timer_service.get_settings"
        ) as mock_s_cfg:
            for m in (mock_h_cfg, mock_s_cfg):
                m.return_value.ALLOWED_CHAT_IDS = []
                m.return_value.TIMEZONE = "Asia/Tehran"
                m.return_value.TIMER_MAX_PER_USER = 15

            # 1. Open /timer panel in topic 42
            msg = _make_group_msg(chat_id=chat_id, thread_id=42, user_id=5001, text="/timer", bot=bot)
            await cmd_timer(msg, bot, fsm)
            msg.delete.assert_called_once()
            bot.send_message.assert_called_once()
            home_text = bot.send_message.call_args.kwargs["text"]
            assert "تنظیم تایمر و یادآور" in home_text
            assert "کافه" in home_text
            # Unstarted DM shows deep-link activation button
            home_kb = bot.send_message.call_args.kwargs["reply_markup"]
            urls = [b.url for r in home_kb.inline_keyboard for b in r if b.url]
            assert "https://t.me/my_test_bot?start=timer" in urls

            # 2. Another user (9999) clicks button -> rejected by Owner Lock!
            intruder_cb = _make_callback(user_id=9999, chat_id=chat_id, message_id=200, thread_id=42)
            await handle_timer_callback(intruder_cb, TimerCB(action="rel", owner_id=5001), bot, fsm)
            intruder_cb.answer.assert_called_once()
            assert intruder_cb.answer.call_args.kwargs["show_alert"] is True
            intruder_cb.message.edit_text.assert_not_called()

            # 3. Owner opens relative builder, sets 30m, adds +15m -> 45m
            owner_cb = _make_callback(user_id=5001, chat_id=chat_id, message_id=200, thread_id=42)
            await handle_timer_callback(owner_cb, TimerCB(action="rel", owner_id=5001), bot, fsm)
            await handle_timer_callback(owner_cb, TimerCB(action="rset", owner_id=5001, arg="30"), bot, fsm)
            await handle_timer_callback(owner_cb, TimerCB(action="radd", owner_id=5001, arg="15"), bot, fsm)
            assert "۴۵ دقیقه" in owner_cb.message.edit_text.call_args.kwargs["text"]

            # 4. Owner clicks Add Note -> enters TimerStates.waiting_note
            await handle_timer_callback(owner_cb, TimerCB(action="note", owner_id=5001, arg="rel"), bot, fsm)
            assert store["state"] == TimerStates.waiting_note.state

            # 4a. Group-safe FSM filter ignores message from a DIFFERENT topic (thread 99)
            other_topic_msg = _make_group_msg(chat_id=chat_id, thread_id=99, user_id=5001, text="سلام بچه‌ها", bot=bot)
            assert await ActiveTimerFsmFilter()(other_topic_msg, fsm) is False

            # 4b. Group-safe FSM filter accepts message in SAME topic (thread 42)
            note_msg = _make_group_msg(chat_id=chat_id, thread_id=42, user_id=5001, text="برم باشگاه", bot=bot)
            assert await ActiveTimerFsmFilter()(note_msg, fsm) is True
            await handle_timer_note_input(note_msg, bot, fsm)
            assert store["state"] is None
            assert "برم باشگاه" in bot.edit_message_text.call_args.kwargs["text"]

            # 5. Owner saves timer!
            await handle_timer_callback(owner_cb, TimerCB(action="save", owner_id=5001), bot, fsm)
            active = await repository.list_active_timers_for_user(5001)
            assert len(active) == 1
            assert active[0]["note"] == "برم باشگاه"
            assert active[0]["thread_id"] == 42
            assert active[0]["timer_type"] == "relative"

            # 6. Simulate timer firing and user clicking Snooze (+10m)
            t_id = active[0]["id"]
            await repository.claim_timer_for_firing(t_id)
            snz_cb = _make_callback(user_id=5001, chat_id=chat_id, message_id=300, thread_id=42)
            await handle_timer_callback(snz_cb, TimerCB(action="snz", owner_id=5001, arg=f"{t_id}_10"), bot, fsm)
            active_after_snz = await repository.list_active_timers_for_user(5001)
            assert len(active_after_snz) == 1
            assert active_after_snz[0]["id"] != t_id
            assert active_after_snz[0]["note"] == "برم باشگاه"
            assert "تمدید شد" in snz_cb.message.edit_text.call_args.kwargs["text"]

            # 7. Dismiss / Ack removes buttons
            ack_cb = _make_callback(user_id=5001, chat_id=chat_id, message_id=301, thread_id=42)
            await handle_timer_callback(ack_cb, TimerCB(action="ack", owner_id=5001, arg=str(t_id)), bot, fsm)
            assert "دیده شد" in ack_cb.message.edit_text.call_args.kwargs["text"]
            assert ack_cb.message.edit_text.call_args.kwargs["reply_markup"] is None
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_shorthand_command_and_fsm_timeout_safety(tmp_path: Path):
    """Shorthand '/timer 30m note' saves immediately; expired FSM never hijacks group chat."""
    clear_timer_runtime_state()
    await init_db(str(tmp_path / "shorthand.db"))
    try:
        bot = AsyncMock(spec=Bot)
        bot.send_message = AsyncMock(return_value=MagicMock(message_id=500))
        fsm, store = _make_fsm_memory()

        with patch("bot.handlers.timer.get_settings") as mock_h_cfg, patch(
            "bot.services.timer_service.get_settings"
        ) as mock_s_cfg:
            for m in (mock_h_cfg, mock_s_cfg):
                m.return_value.ALLOWED_CHAT_IDS = []
                m.return_value.TIMEZONE = "Asia/Tehran"
                m.return_value.TIMER_MAX_PER_USER = 15

            msg = _make_group_msg(chat_id=-100222, thread_id=10, user_id=5005, text="/timer 30m جلسه مهم", bot=bot)
            await cmd_timer(msg, bot, fsm)
            timers = await repository.list_active_timers_for_user(5005)
            assert len(timers) == 1
            assert timers[0]["note"] == "جلسه مهم"
            assert "تایمرت ثبت شد" in bot.send_message.call_args.kwargs["text"]

        # Expired FSM filter returns False AND clears state so normal group chat flows to listener
        await fsm.set_state(TimerStates.waiting_note)
        await fsm.update_data(thread_id=10, expires_at=1.0)  # already expired
        chat_msg = _make_group_msg(chat_id=-100222, thread_id=10, user_id=5005, text="چت عادی گروه", bot=bot)
        assert await ActiveTimerFsmFilter()(chat_msg, fsm) is False
        assert store["state"] is None
    finally:
        await close_db()


# ---------------- 6. Private Chat DM Activation (Group Member vs Outsider) ----------------


@pytest.mark.asyncio
async def test_private_start_activates_dm_for_group_member_without_stranger_alert(tmp_path: Path):
    """Non-admin member of an active group gets DM timer activation on /start; outsiders remain blocked."""
    clear_admin_cache()
    clear_member_cache()
    clear_last_picked_cache()
    clear_stranger_cooldowns()
    await init_db(str(tmp_path / "dm_member.db"))
    try:
        await repository.upsert_known_group(-100888, "گروه ما", is_forum=True)

        bot = AsyncMock(spec=Bot)
        bot.send_message = AsyncMock()

        async def _get_member(chat_id: int, user_id: int):
            m = MagicMock(spec=ChatMemberMember)
            m.status = "member" if (chat_id == -100888 and user_id == 7777) else "left"
            return m

        bot.get_chat_member = AsyncMock(side_effect=_get_member)
        fsm, _ = _make_fsm_memory()

        def _make_priv(uid: int, text: str) -> Message:
            m = MagicMock(spec=Message)
            m.chat = MagicMock(spec=Chat)
            m.chat.id = uid
            m.chat.type = "private"
            m.text = text
            m.from_user = MagicMock(spec=User)
            m.from_user.id = uid
            m.from_user.username = f"u_{uid}"
            m.from_user.first_name = "علی"
            m.from_user.full_name = "علی تست"
            m.answer = AsyncMock()
            return m

        with patch("bot.services.access.get_settings") as mock_acc, patch(
            "bot.handlers.private.get_settings"
        ) as mock_priv:
            for cfg in (mock_acc, mock_priv):
                cfg.return_value.ADMIN_IDS = [100]
                cfg.return_value.ALLOWED_CHAT_IDS = []
                cfg.return_value.NOTIFY_ON_UNKNOWN_START = True
                cfg.return_value.OWNER_NAMES = "آرش و پیت"
                cfg.return_value.TIMEZONE = "Asia/Tehran"

            # 1. Group member (7777) clicks /start timer -> DM activated, NO admin stranger alert!
            member_msg = _make_priv(7777, "/start timer")
            await cmd_private_start(member_msg, bot, fsm)
            assert "اعلان‌های تایمر توی پیوی برات فعال شد" in member_msg.answer.call_args.kwargs["text"]
            assert await repository.is_bot_user_dm_started(7777) is True
            bot.send_message.assert_not_called()  # No false stranger alert to admin 100!

            # 2. True outsider (8888) clicks /start -> stranger anti-spam + admin alert fires!
            outsider_msg = _make_priv(8888, "/start")
            await cmd_private_start(outsider_msg, bot, fsm)
            assert "اعلان‌های تایمر" not in outsider_msg.answer.call_args.kwargs["text"]
            assert bot.send_message.call_count == 1  # Admin 100 notified!
    finally:
        await close_db()


def test_topic_deep_link_and_user_mention_helpers():
    """Verify deep-link generation and mention HTML formatting."""
    assert build_topic_deep_link(-1001234567890, 42) == "https://t.me/c/1234567890/42"
    assert build_topic_deep_link(-1001234567890, 1) == "https://t.me/c/1234567890/1"
    assert build_topic_deep_link(12345, 1) is None

    assert format_user_mention(10, "arash", "آرش") == '@arash (<a href="tg://user?id=10">آرش</a>)'
    assert format_user_mention(10, None, "آرش <1>") == '<a href="tg://user?id=10">آرش &lt;1&gt;</a>'


@pytest.mark.asyncio
async def test_scheduled_jalali_calendar_time_picker_and_past_time_guard(tmp_path: Path):
    """Test Jalali calendar -> day -> time picker presets/steppers -> past-time guard -> save & cancel."""
    clear_timer_runtime_state()
    await init_db(str(tmp_path / "calendar_flow.db"))
    try:
        chat_id = -100333
        bot = AsyncMock(spec=Bot)
        fsm, _ = _make_fsm_memory()
        fixed_now = datetime(2026, 11, 23, 14, 0, 0, tzinfo=TEHRAN_TZ)  # 1405/09/02 14:00 Tehran

        with patch("bot.handlers.timer.get_settings") as mock_h_cfg, patch(
            "bot.services.timer_service.get_settings"
        ) as mock_s_cfg, patch("bot.handlers.timer.get_tehran_now", return_value=fixed_now):
            for m in (mock_h_cfg, mock_s_cfg):
                m.return_value.ALLOWED_CHAT_IDS = []
                m.return_value.TIMEZONE = "Asia/Tehran"
                m.return_value.TIMER_MAX_PER_USER = 15

            cb = _make_callback(user_id=5010, chat_id=chat_id, message_id=400, thread_id=12)

            # 1. Open calendar -> select today (1405-09-02) -> set past time 08.00 -> save blocked!
            await handle_timer_callback(cb, TimerCB(action="cal", owner_id=5010), bot, fsm)
            await handle_timer_callback(cb, TimerCB(action="day", owner_id=5010, arg="1405-09-02"), bot, fsm)
            await handle_timer_callback(cb, TimerCB(action="tset", owner_id=5010, arg="08.00"), bot, fsm)
            assert "این ساعت گذشته" in cb.message.edit_text.call_args.kwargs["text"]

            await handle_timer_callback(cb, TimerCB(action="save", owner_id=5010), bot, fsm)
            assert await repository.count_active_timers_for_user(5010) == 0

            # 2. Pick tomorrow (1405-09-03) -> preset 08.00 -> +1h -> +15m => 09:15 -> save!
            await handle_timer_callback(cb, TimerCB(action="day", owner_id=5010, arg="1405-09-03"), bot, fsm)
            await handle_timer_callback(cb, TimerCB(action="tset", owner_id=5010, arg="08.00"), bot, fsm)
            await handle_timer_callback(cb, TimerCB(action="hadd", owner_id=5010, arg="1"), bot, fsm)
            await handle_timer_callback(cb, TimerCB(action="madd", owner_id=5010, arg="15"), bot, fsm)
            assert "09:15" in cb.message.edit_text.call_args.kwargs["text"]

            await handle_timer_callback(cb, TimerCB(action="save", owner_id=5010), bot, fsm)
            active = await repository.list_active_timers_for_user(5010)
            assert len(active) == 1
            assert active[0]["timer_type"] == "scheduled"
            fire_tehran = datetime.fromisoformat(active[0]["fire_at"]).astimezone(TEHRAN_TZ)
            assert (fire_tehran.hour, fire_tehran.minute) == (9, 15)
            assert jdatetime.date.fromgregorian(date=fire_tehran.date()) == jdatetime.date(1405, 9, 3)

            # 3. View list & cancel timer from list
            await handle_timer_callback(cb, TimerCB(action="list", owner_id=5010, arg="0"), bot, fsm)
            assert "تایمرهای فعال من" in cb.message.edit_text.call_args.kwargs["text"]
            await handle_timer_callback(cb, TimerCB(action="del", owner_id=5010, arg=f"{active[0]['id']}_0"), bot, fsm)
            assert await repository.count_active_timers_for_user(5010) == 0
    finally:
        await close_db()


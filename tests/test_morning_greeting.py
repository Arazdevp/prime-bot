"""Offline unit tests for the Daily Hills/Prime Morning Greeting system."""

from datetime import date, datetime
from pathlib import Path
import re
from unittest.mock import AsyncMock, MagicMock, patch
from zoneinfo import ZoneInfo
import aiosqlite
import pytest
from aiogram import Bot
from aiogram.exceptions import TelegramBadRequest
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, Chat, ChatMemberMember, Message, User
from bot.database import repository
from bot.database.db import close_db, get_db, init_db
from bot.filters import clear_admin_cache
from bot.handlers.admin import (
    cmd_morning_off,
    cmd_morning_on,
    cmd_morning_test,
    cmd_set_morning_topic,
)
from bot.handlers.panel import handle_panel_callback, render_morning_settings
from bot.keyboards.panel import PanelCB
from bot.scheduler.jobs import build_scheduler
from bot.services.morning_service import (
    morning_greeting_job,
    run_morning_catchup_if_needed,
    send_morning_greeting_to_group,
)
from bot.texts.morning_texts import (
    MORNING_GREETINGS,
    compute_hills_day,
    draw_next_greeting_index,
    format_morning_greeting,
)


def _is_html_balanced(text: str) -> bool:
    tag_re = re.compile(r"<(/)?([a-zA-Z0-9_-]+)(?:\s+[^>]*?)?>")
    stack: list[str] = []
    for m in tag_re.finditer(text):
        is_close = bool(m.group(1))
        tag = m.group(2).lower()
        if not is_close:
            stack.append(tag)
        else:
            if not stack or stack.pop() != tag:
                return False
    return not stack


def test_morning_greetings_pool_has_100_unique_valid_html_templates():
    """Verify exactly 100 completely unique Persian templates, all containing {day} and valid Telegram HTML."""
    assert len(MORNING_GREETINGS) == 100
    assert len(set(MORNING_GREETINGS)) == 100

    sample_dt = datetime(2026, 11, 23, 5, 1, 0, tzinfo=ZoneInfo("Asia/Tehran"))
    rendered_all: set[str] = set()
    for idx, tpl in enumerate(MORNING_GREETINGS):
        assert "{day}" in tpl, f"Template #{idx} missing {{day}} placeholder"
        rendered = format_morning_greeting(idx, day_number=88, now_tehran=sample_dt)
        assert "۸۸" in rendered
        assert "📅" in rendered
        assert _is_html_balanced(rendered), f"Unbalanced HTML in template #{idx}: {rendered}"
        rendered_all.add(rendered)
    assert len(rendered_all) == 100


def test_compute_hills_day_exact_1_based_from_aug_28_2026():
    """Verify 2026-08-28 is Day 1, 2026-08-29 is Day 2, and 2026-11-23 is Day 88."""
    tz = ZoneInfo("Asia/Tehran")
    assert compute_hills_day(date(2026, 8, 28)) == 1
    assert compute_hills_day(date(2026, 8, 29)) == 2
    assert compute_hills_day(date(2026, 9, 27)) == 31
    assert compute_hills_day(datetime(2026, 11, 23, 5, 1, 0, tzinfo=tz)) == 88
    # Before start date safely clamps to 1
    assert compute_hills_day(date(2026, 8, 1)) == 1


def test_shuffle_bag_guarantees_zero_repeats_across_full_cycles():
    """Verify 200 sequential draws produce two complete duplicate-free cycles of 100 with no boundary repeat."""
    deck: list[int] | None = None
    last_idx: int | None = None
    draws: list[int] = []

    for _ in range(200):
        chosen, deck = draw_next_greeting_index(deck, last_idx)
        draws.append(chosen)
        last_idx = chosen

    first_cycle = draws[:100]
    second_cycle = draws[100:200]
    assert len(set(first_cycle)) == 100
    assert set(first_cycle) == set(range(100))
    assert len(set(second_cycle)) == 100
    assert set(second_cycle) == set(range(100))
    assert first_cycle[-1] != second_cycle[0]


@pytest.mark.asyncio
async def test_migration_v4_to_v5_creates_morning_greetings_table(tmp_path: Path):
    """Upgrading from user_version=4 adds morning_greetings table while preserving existing rows."""
    db_file = tmp_path / "v4_to_v5.db"
    conn = await aiosqlite.connect(str(db_file))
    await conn.executescript(
        """
        CREATE TABLE group_settings (
            chat_id INTEGER PRIMARY KEY,
            mode TEXT NOT NULL DEFAULT 'blacklist',
            destination_mode TEXT NOT NULL DEFAULT 'per_topic',
            destination_thread_id INTEGER,
            enabled INTEGER NOT NULL DEFAULT 1,
            min_messages INTEGER NOT NULL DEFAULT 5
        );
        INSERT INTO group_settings (chat_id, mode, destination_mode, destination_thread_id, enabled, min_messages)
        VALUES (-100555, 'blacklist', 'per_topic', NULL, 1, 5);
        PRAGMA user_version = 4;
        """
    )
    await conn.commit()
    await conn.close()

    try:
        await init_db(str(db_file))
        db = get_db()
        cur = await db.execute("PRAGMA user_version;")
        assert (await cur.fetchone())[0] == 5

        gs = await repository.get_group_settings(-100555)
        assert gs is not None and gs["min_messages"] == 5

        cfg = await repository.set_morning_topic(-100555, 42)
        assert cfg["thread_id"] == 42
        assert cfg["enabled"] == 1
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_morning_send_idempotency_rollback_and_test_button_non_blocking(tmp_path: Path):
    """Verify Test sends real greetings without blocking 05:01 auto-send, auto-send fires once/day, and failures rollback."""
    await init_db(str(tmp_path / "morning_send.db"))
    try:
        chat_id = -100777
        thread_id = 42
        await repository.upsert_known_group(chat_id, "هیلز پرایم")
        await repository.upsert_topic(chat_id, thread_id, "گفتگوی آزاد")
        await repository.set_morning_topic(chat_id, thread_id)

        bot = AsyncMock(spec=Bot)
        bot.send_message = AsyncMock()
        tz = ZoneInfo("Asia/Tehran")
        day_dt = datetime(2026, 11, 23, 5, 1, 0, tzinfo=tz)

        # 1. Three consecutive admin test sends produce 3 distinct live messages and keep last_sent_day=None
        test_texts: list[str] = []
        for _ in range(3):
            ok, details = await send_morning_greeting_to_group(bot, chat_id, is_test=True, now_tehran=day_dt)
            assert ok is True
            assert "۸۸" in details
            call_kw = bot.send_message.call_args.kwargs
            assert call_kw["chat_id"] == chat_id
            assert call_kw["message_thread_id"] == thread_id
            assert "۸۸" in call_kw["text"]
            assert call_kw["text"].startswith("🧪 <b>تست</b>\n\n")
            test_texts.append(call_kw["text"])

        assert len(set(test_texts)) == 3
        assert (await repository.get_morning_config(chat_id))["last_sent_day"] is None

        # 2. Scheduled 05:01 job STILL fires today and sends a 4th distinct message (without test prefix)
        mock_cfg = MagicMock()
        mock_cfg.ALLOWED_CHAT_IDS = []
        mock_cfg.TIMEZONE = "Asia/Tehran"
        mock_cfg.HILLS_START_DATE = "2026-08-28"
        mock_cfg.SEND_DELAY_SECONDS = 0.0

        with patch("bot.services.morning_service.get_settings", return_value=mock_cfg):
            report1 = await morning_greeting_job(bot, now_tehran=day_dt)
            assert report1 == {"sent": 1, "skipped": 0, "failed": 0}
            auto_text = bot.send_message.call_args.kwargs["text"]
            assert not auto_text.startswith("🧪 <b>تست</b>")
            assert f"🧪 <b>تست</b>\n\n{auto_text}" not in test_texts
            assert (await repository.get_morning_config(chat_id))["last_sent_day"] == "2026-11-23"

            # 3. Duplicate run same day is skipped
            report2 = await morning_greeting_job(bot, now_tehran=day_dt)
            assert report2 == {"sent": 0, "skipped": 1, "failed": 0}

            # 4. Next day failure rolls back last_sent_day so retry succeeds
            next_day_dt = datetime(2026, 11, 24, 5, 1, 0, tzinfo=tz)
            bot.send_message.side_effect = TelegramBadRequest(method="sendMessage", message="Chat not accessible")
            ok_fail, _ = await send_morning_greeting_to_group(bot, chat_id, is_test=False, now_tehran=next_day_dt)
            assert ok_fail is False
            assert (await repository.get_morning_config(chat_id))["last_sent_day"] == "2026-11-23"

            # Now recover and deliver Day 89
            bot.send_message.side_effect = None
            ok_ok, _ = await send_morning_greeting_to_group(bot, chat_id, is_test=False, now_tehran=next_day_dt)
            assert ok_ok is True
            assert "۸۹" in bot.send_message.call_args.kwargs["text"]
            assert (await repository.get_morning_config(chat_id))["last_sent_day"] == "2026-11-24"
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_morning_general_topic_and_deleted_topic_fallback(tmp_path: Path):
    """Verify General topic sends with message_thread_id=None and deleted topic falls back to General."""
    await init_db(str(tmp_path / "morning_general.db"))
    try:
        chat_id = -100888
        await repository.set_morning_topic(chat_id, 1)

        bot = AsyncMock(spec=Bot)
        bot.send_message = AsyncMock()
        now = datetime(2026, 11, 23, 5, 1, 0, tzinfo=ZoneInfo("Asia/Tehran"))

        ok, _ = await send_morning_greeting_to_group(bot, chat_id, is_test=True, now_tehran=now)
        assert ok is True
        assert bot.send_message.call_args.kwargs["message_thread_id"] is None

        # Deleted topic 77 -> falls back to General (None)
        await repository.set_morning_topic(chat_id, 77)
        calls: list[int | None] = []

        async def _thread_not_found_once(**kwargs):
            calls.append(kwargs["message_thread_id"])
            if kwargs["message_thread_id"] == 77:
                raise TelegramBadRequest(method="sendMessage", message="Bad Request: message thread not found")
            return AsyncMock()

        bot.send_message = AsyncMock(side_effect=_thread_not_found_once)
        ok2, _ = await send_morning_greeting_to_group(bot, chat_id, is_test=True, now_tehran=now)
        assert ok2 is True
        assert calls == [77, None]
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_morning_catchup_respects_morning_window(tmp_path: Path):
    """Verify startup catch-up sends only between 05:01 and MORNING_CATCHUP_END_HOUR (never at night)."""
    await init_db(str(tmp_path / "morning_catchup.db"))
    try:
        chat_id = -100999
        await repository.upsert_known_group(chat_id, "گروه پرایم")
        await repository.set_morning_topic(chat_id, 10)

        bot = AsyncMock(spec=Bot)
        bot.send_message = AsyncMock()
        tz = ZoneInfo("Asia/Tehran")

        mock_cfg = MagicMock()
        mock_cfg.ALLOWED_CHAT_IDS = []
        mock_cfg.TIMEZONE = "Asia/Tehran"
        mock_cfg.HILLS_START_DATE = "2026-08-28"
        mock_cfg.MORNING_GREETING_HOUR = 5
        mock_cfg.MORNING_GREETING_MINUTE = 1
        mock_cfg.MORNING_CATCHUP_END_HOUR = 11
        mock_cfg.SEND_DELAY_SECONDS = 0.0

        with patch("bot.services.morning_service.get_settings", return_value=mock_cfg):
            # Too early (04:55) -> 0 sent
            assert await run_morning_catchup_if_needed(bot, now_tehran=datetime(2026, 11, 23, 4, 55, tzinfo=tz)) == 0
            # Afternoon (14:30) -> 0 sent
            assert await run_morning_catchup_if_needed(bot, now_tehran=datetime(2026, 11, 23, 14, 30, tzinfo=tz)) == 0
            # Morning window (07:30) -> sends 1
            assert await run_morning_catchup_if_needed(bot, now_tehran=datetime(2026, 11, 23, 7, 30, tzinfo=tz)) == 1
            # Repeat at 08:00 -> already sent today, 0 sent
            assert await run_morning_catchup_if_needed(bot, now_tehran=datetime(2026, 11, 23, 8, 0, tzinfo=tz)) == 0
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_panel_morning_flow_security_callback_sizes_and_group_commands(tmp_path: Path):
    """Verify PanelCB sizes <= 64 bytes, admin-only access, panel topic/test flow, and group commands."""
    clear_admin_cache()
    await init_db(str(tmp_path / "morning_panel.db"))
    try:
        # 1. Worst-case PanelCB byte length check
        worst_chat_id = -1009999999999999999
        for cb in (
            PanelCB(action="mrg", chat_id=worst_chat_id),
            PanelCB(action="mrgt", chat_id=worst_chat_id),
            PanelCB(action="mrgl", chat_id=worst_chat_id, page=9999),
            PanelCB(action="mrgs", chat_id=worst_chat_id, arg="2147483647", page=9999),
            PanelCB(action="mrgtst", chat_id=worst_chat_id),
        ):
            assert len(cb.pack().encode("utf-8")) <= 64

        chat_id = -100505
        await repository.upsert_known_group(chat_id, "پرایم هیلز")
        await repository.upsert_topic(chat_id, 42, "تاپیک اصلی")

        bot = AsyncMock(spec=Bot)
        bot.send_message = AsyncMock()
        member_non_admin = MagicMock(spec=ChatMemberMember)
        member_non_admin.status = "member"
        bot.get_chat_member = AsyncMock(return_value=member_non_admin)

        def _make_cb(uid: int) -> CallbackQuery:
            cb = MagicMock(spec=CallbackQuery)
            cb.from_user = MagicMock(spec=User)
            cb.from_user.id = uid
            cb.from_user.first_name = "کاربر"
            cb.from_user.full_name = "کاربر"
            cb.answer = AsyncMock()
            cb.message = MagicMock(spec=Message)
            cb.message.message_id = 10
            cb.message.chat = MagicMock(spec=Chat)
            cb.message.chat.id = uid
            cb.message.chat.type = "private"
            cb.message.edit_text = AsyncMock()
            return cb

        fsm = MagicMock(spec=FSMContext)
        fsm.clear = AsyncMock()

        mock_cfg = MagicMock()
        mock_cfg.ADMIN_IDS = [100]
        mock_cfg.ALLOWED_CHAT_IDS = []
        mock_cfg.TIMEZONE = "Asia/Tehran"
        mock_cfg.HILLS_START_DATE = "2026-08-28"
        mock_cfg.MORNING_GREETING_HOUR = 5
        mock_cfg.MORNING_GREETING_MINUTE = 1
        mock_cfg.SEND_DELAY_SECONDS = 0.0

        with patch("bot.services.access.get_settings", return_value=mock_cfg), patch(
            "bot.handlers.panel.get_settings", return_value=mock_cfg
        ), patch("bot.services.morning_service.get_settings", return_value=mock_cfg):
            # 2. Non-admin is rejected on mrgtst
            cb_stranger = _make_cb(999)
            await handle_panel_callback(cb_stranger, PanelCB(action="mrgtst", chat_id=chat_id), bot, fsm)
            assert cb_stranger.answer.call_args.kwargs["show_alert"] is True
            bot.send_message.assert_not_called()

            # 3. Admin cannot test before picking a topic
            cb_admin = _make_cb(100)
            await handle_panel_callback(cb_admin, PanelCB(action="mrgtst", chat_id=chat_id), bot, fsm)
            assert "انتخاب" in cb_admin.answer.call_args.kwargs["text"]
            bot.send_message.assert_not_called()

            # 4. Admin selects topic 42 via panel -> auto-enables
            await handle_panel_callback(cb_admin, PanelCB(action="mrgs", chat_id=chat_id, arg="42"), bot, fsm)
            cfg = await repository.get_morning_config(chat_id)
            assert cfg["thread_id"] == 42 and cfg["enabled"] == 1

            # 5. Admin clicks Test button -> sends real morning greeting into topic 42!
            await handle_panel_callback(cb_admin, PanelCB(action="mrgtst", chat_id=chat_id), bot, fsm)
            assert bot.send_message.call_count == 1
            assert bot.send_message.call_args.kwargs["chat_id"] == chat_id
            assert bot.send_message.call_args.kwargs["message_thread_id"] == 42
            assert "✅" in cb_admin.answer.call_args.kwargs["text"]

            # Panel screen displays diversity progress (1 of 100 used)
            screen_text, _ = await render_morning_settings(chat_id)
            assert "۱ از ۱۰۰" in screen_text

            # 6. Group admin commands (/set_morning_topic, /morning_test, /morning_off, /morning_on)
            msg = MagicMock(spec=Message)
            msg.chat = MagicMock(spec=Chat)
            msg.chat.id = chat_id
            msg.chat.type = "supergroup"
            msg.is_topic_message = True
            msg.message_thread_id = 55
            msg.text = "/set_morning_topic"
            msg.delete = AsyncMock()
            msg.bot = bot

            bot.send_message.reset_mock()
            await cmd_set_morning_topic(msg)
            assert (await repository.get_morning_config(chat_id))["thread_id"] == 55

            bot.send_message.reset_mock()
            msg.text = "/morning_test"
            await cmd_morning_test(msg)
            assert bot.send_message.call_count == 1
            assert bot.send_message.call_args.kwargs["message_thread_id"] == 55

            await cmd_morning_off(msg)
            assert (await repository.get_morning_config(chat_id))["enabled"] == 0

            await cmd_morning_on(msg)
            assert (await repository.get_morning_config(chat_id))["enabled"] == 1
    finally:
        await close_db()


def test_scheduler_registers_daily_morning_greeting_job():
    """Verify build_scheduler registers daily_morning_greeting CronTrigger at configured 05:01."""
    bot = AsyncMock(spec=Bot)
    sched = build_scheduler(bot)
    job = sched.get_job("daily_morning_greeting")
    assert job is not None
    assert "05:01" in str(job.trigger) or ("hour='5'" in str(job.trigger) and "minute='1'" in str(job.trigger))

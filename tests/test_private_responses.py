"""Offline unit tests for private-chat responses, text pools, stranger anti-spam, throttle, and membership."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import aiosqlite
import pytest
from aiogram import Bot
from aiogram.fsm.context import FSMContext
from aiogram.types import Chat, ChatMemberAdministrator, ChatMemberLeft, ChatMemberMember, ChatMemberUpdated, Message, User
from bot.database import repository
from bot.database.db import close_db, get_db, init_db
from bot.filters import clear_admin_cache
from bot.handlers.listener import (
    clear_known_group_cache,
    on_my_chat_member,
)
from bot.handlers.private import (
    clear_stranger_cooldowns,
    cmd_private_about,
    cmd_private_cancel,
    cmd_private_help,
    cmd_private_start,
    handle_private_fallback,
    handle_stranger_contact,
)
from bot.middlewares.throttle import PrivateThrottleMiddleware
from bot.texts.private_texts import POOLS, clear_last_picked_cache, pick


def _make_private_msg(
    user_id: int,
    text: str = "/start",
    *,
    first_name: str = "علی",
    full_name: str = "علی رضایی",
    username: str | None = "ali_user",
) -> Message:
    msg = MagicMock(spec=Message)
    msg.chat = MagicMock(spec=Chat)
    msg.chat.id = user_id
    msg.chat.type = "private"
    msg.text = text

    user = MagicMock(spec=User)
    user.id = user_id
    user.first_name = first_name
    user.full_name = full_name
    user.username = username
    msg.from_user = user

    msg.answer = AsyncMock()
    return msg


def _make_fsm(initial_state: str | None = None) -> FSMContext:
    state = MagicMock(spec=FSMContext)
    state.get_state = AsyncMock(return_value=initial_state)
    state.clear = AsyncMock()
    state.set_state = AsyncMock()
    state.get_data = AsyncMock(return_value={})
    state.update_data = AsyncMock()
    return state


def test_pick_never_repeats_last_variant_and_formats_placeholders():
    """pick() must never repeat exclude_last (when pool size > 1) and must format & escape placeholders."""
    clear_last_picked_cache()

    for pool_key in ("unauthorized", "repeat", "admin_welcome", "group_admin_welcome", "admin_hint", "callback_denied"):
        assert len(POOLS[pool_key]) >= 2
        last_key = None
        for _ in range(30):
            text, key = pick(
                pool_key,
                user_id=1001,
                exclude_last=last_key,
                name="R&D <x>",
                owners="آرش & <پیت>",
                escape=True,
            )
            assert key != last_key
            assert "{name}" not in text
            assert "{owners}" not in text
            assert "R&D <x>" not in text
            assert "<پیت>" not in text
            last_key = key

    # Plain-text alert mode (escape=False) keeps raw chars without HTML entities
    alert_text, _ = pick(
        "callback_denied",
        user_id=1002,
        exclude_last="callback_denied:1",
        name="R&D <x>",
        owners="آرش و پیت",
        escape=False,
    )
    assert "{name}" not in alert_text
    assert "&lt;" not in alert_text


@pytest.mark.asyncio
async def test_migration_v2_to_v3_creates_tables_and_backfills_known_groups(tmp_path: Path):
    """Upgrading from user_version=2 creates bot_users & known_groups and backfills existing group_settings."""
    db_file = tmp_path / "upgrade_v2_to_v3.db"
    conn = await aiosqlite.connect(str(db_file))
    await conn.executescript(
        """
        CREATE TABLE topics (
            chat_id INTEGER NOT NULL,
            thread_id INTEGER NOT NULL,
            name TEXT,
            is_active INTEGER,
            updated_at TEXT,
            PRIMARY KEY(chat_id, thread_id)
        );
        CREATE TABLE group_settings (
            chat_id INTEGER PRIMARY KEY,
            mode TEXT NOT NULL DEFAULT 'blacklist',
            destination_mode TEXT NOT NULL DEFAULT 'per_topic',
            destination_thread_id INTEGER,
            enabled INTEGER NOT NULL DEFAULT 1,
            min_messages INTEGER NOT NULL DEFAULT 5
        );
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE summaries (
            chat_id INTEGER NOT NULL,
            thread_id INTEGER NOT NULL,
            day TEXT NOT NULL,
            parts_json TEXT NOT NULL,
            sent_count INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            PRIMARY KEY(chat_id, thread_id, day)
        );
        INSERT INTO group_settings (chat_id, mode, destination_mode, destination_thread_id, enabled, min_messages)
        VALUES (-100777888, 'blacklist', 'per_topic', NULL, 1, 5);
        PRAGMA user_version = 2;
        """
    )
    await conn.commit()
    await conn.close()

    try:
        await init_db(str(db_file))
        db = get_db()
        cur = await db.execute("PRAGMA user_version;")
        ver = (await cur.fetchone())[0]
        assert ver == 3

        groups = await repository.list_active_known_groups()
        assert len(groups) == 1
        assert groups[0]["chat_id"] == -100777888
        assert groups[0]["left_at"] is None
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_stranger_flow_first_notify_repeat_pool_5_attempt_ignore_and_resume(tmp_path: Path):
    """Verify first-contact admin notification once, repeat pool, ignore notice after 5 attempts, 1h silence, and resume."""
    clear_last_picked_cache()
    clear_stranger_cooldowns()
    clear_admin_cache()
    await init_db(str(tmp_path / "stranger_flow.db"))

    try:
        bot = AsyncMock(spec=Bot)
        bot.send_message = AsyncMock()
        t0 = datetime(2026, 11, 22, 12, 0, 0, tzinfo=timezone.utc)

        msg = _make_private_msg(
            55501,
            "/start",
            first_name="R&D <x>",
            full_name="R&D <x> User",
            username="rd_user",
        )

        with patch("bot.handlers.private.get_settings") as mock_cfg, patch(
            "bot.texts.private_texts.get_settings"
        ) as mock_txt_cfg:
            mock_cfg.return_value.ADMIN_IDS = [9001, 9002]
            mock_cfg.return_value.NOTIFY_ON_UNKNOWN_START = True
            mock_cfg.return_value.OWNER_NAMES = "آرش و پیت"
            mock_cfg.return_value.TIMEZONE = "Asia/Tehran"
            mock_txt_cfg.return_value.OWNER_NAMES = "آرش و پیت"

            # Attempt 1: unauthorized pool + admin notified ONCE (2 admins)
            reply1 = await handle_stranger_contact(msg, bot, is_start=True, now_utc=t0)
            assert reply1 is not None
            assert "R&amp;D &lt;x&gt;" in reply1 or "آرش و پیت" in reply1
            assert "R&D <x>" not in reply1
            assert bot.send_message.call_count == 2
            notify_text = bot.send_message.call_args_list[0].kwargs["text"]
            assert "55501" in notify_text
            assert "@rd_user" in notify_text
            assert "R&amp;D &lt;x&gt; User" in notify_text
            assert "R&D <x>" not in notify_text

            # Attempts 2..5 within 10 minutes: repeat pool, no duplicate admin notifications, never repeating last variant
            prev_variant = (await repository.get_bot_user(55501))["last_variant_key"]
            assert prev_variant.startswith("unauthorized:")

            for attempt_idx in range(2, 6):
                reply = await handle_stranger_contact(
                    msg,
                    bot,
                    is_start=True,
                    now_utc=t0 + timedelta(seconds=attempt_idx * 10),
                )
                assert reply is not None
                cur_variant = (await repository.get_bot_user(55501))["last_variant_key"]
                assert cur_variant.startswith("repeat:")
                assert cur_variant != prev_variant
                prev_variant = cur_variant

            # Admin notification count must still be 2 (notified only once!)
            assert bot.send_message.call_count == 2

            # Attempt 6 (after 5 attempts in 10 min): sends ignore_notice and sets 1-hour ignore
            reply6 = await handle_stranger_contact(
                msg,
                bot,
                is_start=True,
                now_utc=t0 + timedelta(minutes=2),
            )
            assert reply6 is not None
            assert "تا یه ساعت جواب نمیدم" in reply6
            user_row = await repository.get_bot_user(55501)
            assert user_row["ignored_until"] is not None

            # Attempt 7 (30 minutes later, still inside 1h ignore): silently ignored!
            msg.answer.reset_mock()
            reply7 = await handle_stranger_contact(
                msg,
                bot,
                is_start=True,
                now_utc=t0 + timedelta(minutes=32),
            )
            assert reply7 is None
            msg.answer.assert_not_called()

            # Attempt 8 (65 minutes after ignore_notice -> expired!): replies again from unauthorized pool
            reply8 = await handle_stranger_contact(
                msg,
                bot,
                is_start=True,
                now_utc=t0 + timedelta(minutes=65),
            )
            assert reply8 is not None
            user_after = await repository.get_bot_user(55501)
            assert user_after["last_variant_key"].startswith("unauthorized:")
            assert user_after["ignored_until"] is None
            assert user_after["attempts_count"] == 1
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_stranger_non_start_30s_cooldown(tmp_path: Path):
    """Non-/start stranger messages receive at most one reply per 30 seconds."""
    clear_last_picked_cache()
    clear_stranger_cooldowns()
    await init_db(str(tmp_path / "stranger_30s.db"))

    try:
        bot = AsyncMock(spec=Bot)
        msg = _make_private_msg(55502, "سلام ربات")
        t0 = datetime(2026, 11, 22, 13, 0, 0, tzinfo=timezone.utc)

        with patch("bot.handlers.private.get_settings") as mock_cfg, patch(
            "bot.handlers.private.time.monotonic"
        ) as mock_mono:
            mock_cfg.return_value.ADMIN_IDS = []
            mock_cfg.return_value.NOTIFY_ON_UNKNOWN_START = False
            mock_cfg.return_value.OWNER_NAMES = "آرش و پیت"
            mock_cfg.return_value.TIMEZONE = "Asia/Tehran"

            mock_mono.return_value = 100.0
            r1 = await handle_stranger_contact(msg, bot, is_start=False, now_utc=t0)
            assert r1 is not None

            # 10 seconds later -> suppressed!
            mock_mono.return_value = 110.0
            r2 = await handle_stranger_contact(msg, bot, is_start=False, now_utc=t0 + timedelta(seconds=10))
            assert r2 is None

            # 31 seconds later -> allowed again (uses repeat pool within 10 min)!
            mock_mono.return_value = 131.0
            r3 = await handle_stranger_contact(msg, bot, is_start=False, now_utc=t0 + timedelta(seconds=31))
            assert r3 is not None
            assert (await repository.get_bot_user(55502))["last_variant_key"].startswith("repeat:")
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_private_commands_role_aware_and_admin_welcome_hint_about_cancel(tmp_path: Path):
    """Verify /start, /help, /about, /cancel, and unknown text behavior across roles."""
    clear_last_picked_cache()
    clear_stranger_cooldowns()
    clear_admin_cache()
    await init_db(str(tmp_path / "private_cmds.db"))

    try:
        bot = AsyncMock(spec=Bot)
        state = _make_fsm()

        with patch("bot.services.access.get_settings") as mock_acc_cfg, patch(
            "bot.handlers.private.get_settings"
        ) as mock_priv_cfg:
            for m in (mock_acc_cfg, mock_priv_cfg):
                m.return_value.ADMIN_IDS = [100]
                m.return_value.ALLOWED_CHAT_IDS = []
                m.return_value.OWNER_NAMES = "آرش و پیت"
                m.return_value.NOTIFY_ON_UNKNOWN_START = False
                m.return_value.TIMEZONE = "Asia/Tehran"

            # Global admin /start -> admin_welcome + panel button
            admin_msg = _make_private_msg(100, "/start", first_name="آرش")
            await cmd_private_start(admin_msg, bot, state)
            admin_msg.answer.assert_called_once()
            assert admin_msg.answer.call_args.kwargs["reply_markup"] is not None
            kb_text = admin_msg.answer.call_args.kwargs["reply_markup"].inline_keyboard[0][0].text
            assert "باز کردن پنل" in kb_text

            # Global admin unknown text -> admin_hint
            hint_msg = _make_private_msg(100, "چطوری؟", first_name="آرش")
            await handle_private_fallback(hint_msg, bot)
            assert "/panel" in hint_msg.answer.call_args.kwargs["text"]

            # Global admin /help -> Persian guide
            help_msg = _make_private_msg(100, "/help", first_name="آرش")
            await cmd_private_help(help_msg, bot)
            assert "راهنمای پنل مدیریت" in help_msg.answer.call_args.kwargs["text"]

            # Stranger /help -> unauthorized pool (NOT admin guide)
            stranger_help = _make_private_msg(999, "/help", first_name="غریبه")
            await cmd_private_help(stranger_help, bot)
            sent = stranger_help.answer.call_args.kwargs["text"]
            assert "راهنمای پنل مدیریت" not in sent

            # /about -> mentions OWNER_NAMES
            about_msg = _make_private_msg(100, "/about")
            await cmd_private_about(about_msg, bot)
            assert "آرش و پیت" in about_msg.answer.call_args.kwargs["text"]

            # /cancel when inside FSM clears state
            active_state = _make_fsm(initial_state="PanelStates:waiting_min_messages")
            cancel_msg = _make_private_msg(100, "/cancel")
            await cmd_private_cancel(cancel_msg, bot, active_state)
            active_state.clear.assert_called_once()
            assert "لغو شد" in cancel_msg.answer.call_args.kwargs["text"]
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_private_throttle_middleware():
    """PrivateThrottleMiddleware enforces 1 update/sec in private chat and ignores supergroups."""
    mw = PrivateThrottleMiddleware(rate_limit_seconds=1.0)
    handler = AsyncMock(return_value="ok")

    priv_msg = _make_private_msg(777, "hi")
    group_msg = MagicMock(spec=Message)
    group_msg.chat = MagicMock(spec=Chat)
    group_msg.chat.type = "supergroup"
    group_msg.from_user = MagicMock(spec=User)
    group_msg.from_user.id = 777

    with patch("bot.middlewares.throttle.time.monotonic") as mock_mono:
        mock_mono.return_value = 50.0
        assert await mw(handler, priv_msg, {}) == "ok"
        assert handler.call_count == 1

        # 0.3s later -> silently dropped
        mock_mono.return_value = 50.3
        assert await mw(handler, priv_msg, {}) is None
        assert handler.call_count == 1

        # Group message is NEVER throttled
        assert await mw(handler, group_msg, {}) == "ok"
        assert handler.call_count == 2

        # 1.1s after first private message -> allowed
        mock_mono.return_value = 51.1
        assert await mw(handler, priv_msg, {}) == "ok"
        assert handler.call_count == 3


@pytest.mark.asyncio
async def test_my_chat_member_add_remove_and_auto_leave(tmp_path: Path):
    """Verify my_chat_member upserts/removes known_groups, welcomes, and enforces AUTO_LEAVE_UNKNOWN_GROUPS."""
    clear_known_group_cache()
    await init_db(str(tmp_path / "membership.db"))

    try:
        bot = AsyncMock(spec=Bot)
        bot.send_message = AsyncMock()
        bot.leave_chat = AsyncMock()

        def _make_member_event(chat_id: int, title: str, old_status: str, new_status: str) -> ChatMemberUpdated:
            ev = MagicMock(spec=ChatMemberUpdated)
            ev.chat = MagicMock(spec=Chat)
            ev.chat.id = chat_id
            ev.chat.type = "supergroup"
            ev.chat.title = title
            ev.chat.is_forum = True
            ev.old_chat_member = MagicMock(spec=ChatMemberLeft)
            ev.old_chat_member.status = old_status
            ev.new_chat_member = MagicMock(spec=ChatMemberAdministrator)
            ev.new_chat_member.status = new_status
            return ev

        # 1. Added to an allowed group
        with patch("bot.handlers.listener.get_settings") as mock_cfg:
            mock_cfg.return_value.AUTO_LEAVE_UNKNOWN_GROUPS = True
            mock_cfg.return_value.ALLOWED_CHAT_IDS = [-100111]
            mock_cfg.return_value.OWNER_NAMES = "آرش و پیت"

            await on_my_chat_member(_make_member_event(-100111, "گروه مجاز", "left", "administrator"), bot)
            grp = await repository.get_known_group(-100111)
            assert grp is not None and grp["title"] == "گروه مجاز" and grp["left_at"] is None
            bot.leave_chat.assert_not_called()
            assert "ربات خلاصهساز تاپیکام" in bot.send_message.call_args.kwargs["text"]

            # 2. Added to an unauthorized group with AUTO_LEAVE_UNKNOWN_GROUPS=True
            bot.send_message.reset_mock()
            bot.leave_chat.reset_mock()
            await on_my_chat_member(_make_member_event(-100999, "گروه غیرمجاز", "left", "member"), bot)
            assert "توی لیست مجاز من نیست" in bot.send_message.call_args.kwargs["text"]
            bot.leave_chat.assert_called_once_with(-100999)

            # 3. Removed from allowed group -> left_at populated
            await on_my_chat_member(_make_member_event(-100111, "گروه مجاز", "administrator", "kicked"), bot)
            grp_after = await repository.get_known_group(-100111)
            assert grp_after is not None and grp_after["left_at"] is not None
    finally:
        await close_db()

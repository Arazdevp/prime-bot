"""Offline unit tests for the Private Admin Panel, access roles, callbacks, FSM, lock, and dry-run preview."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from aiogram import Bot
from aiogram.fsm.context import FSMContext
from aiogram.types import (
    CallbackQuery,
    Chat,
    ChatMemberAdministrator,
    ChatMemberMember,
    Message,
    User,
)
from bot.database import repository
from bot.database.db import close_db, get_db, init_db
from bot.filters import AdminFilter, clear_admin_cache, get_cached_admin_status
from bot.handlers.panel import (
    PanelStates,
    handle_min_messages_fsm_input,
    handle_panel_callback,
    handle_stale_callback,
    parse_min_messages_input,
    render_group_dashboard,
    render_topics,
)
from bot.keyboards.panel import PanelCB, paginate
from bot.scheduler.jobs import get_summary_lock, run_summary_for_topic
from bot.services.access import UserRole, resolve_user_role
from bot.services.llm_client import BaseLLMClient


def _make_callback(user_id: int, first_name: str = "آرش") -> CallbackQuery:
    cb = MagicMock(spec=CallbackQuery)
    cb.from_user = MagicMock(spec=User)
    cb.from_user.id = user_id
    cb.from_user.first_name = first_name
    cb.from_user.full_name = first_name
    cb.answer = AsyncMock()

    msg = MagicMock(spec=Message)
    msg.message_id = 42
    msg.chat = MagicMock(spec=Chat)
    msg.chat.id = user_id
    msg.chat.type = "private"
    msg.edit_text = AsyncMock()
    cb.message = msg
    return cb


def _make_fsm_memory() -> tuple[FSMContext, dict]:
    store: dict = {"state": None, "data": {}}
    fsm = MagicMock(spec=FSMContext)

    async def _set_state(s):
        store["state"] = s

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


def test_callback_data_size_under_64_bytes_worst_case():
    """Every PanelCB action with worst-case 64-bit chat_id, thread_id, and page MUST be <= 64 bytes."""
    worst_chat_id = -1009999999999999999  # 20 chars
    worst_cases = [
        PanelCB(action="home", chat_id=0, arg="", page=9999),
        PanelCB(action="open", chat_id=0, arg="", page=0),
        PanelCB(action="grp", chat_id=worst_chat_id, arg="", page=0),
        PanelCB(action="tog", chat_id=worst_chat_id, arg="", page=0),
        PanelCB(action="mode", chat_id=worst_chat_id, arg="", page=0),
        PanelCB(action="tops", chat_id=worst_chat_id, arg="", page=9999),
        PanelCB(action="topt", chat_id=worst_chat_id, arg="2147483647", page=9999),
        PanelCB(action="topa", chat_id=worst_chat_id, arg="1", page=9999),
        PanelCB(action="dest", chat_id=worst_chat_id, arg="", page=0),
        PanelCB(action="dstp", chat_id=worst_chat_id, arg="", page=0),
        PanelCB(action="dstl", chat_id=worst_chat_id, arg="", page=9999),
        PanelCB(action="dsts", chat_id=worst_chat_id, arg="2147483647", page=9999),
        PanelCB(action="dstc", chat_id=worst_chat_id, arg="2147483647", page=0),
        PanelCB(action="min", chat_id=worst_chat_id, arg="", page=0),
        PanelCB(action="mind", chat_id=worst_chat_id, arg="-5", page=0),
        PanelCB(action="minf", chat_id=worst_chat_id, arg="", page=0),
        PanelCB(action="run", chat_id=worst_chat_id, arg="", page=9999),
        PanelCB(action="runc", chat_id=worst_chat_id, arg="2147483647_1", page=9999),
        PanelCB(action="rune", chat_id=worst_chat_id, arg="2147483647_1", page=9999),
        PanelCB(action="prv", chat_id=worst_chat_id, arg="", page=9999),
        PanelCB(action="prvr", chat_id=worst_chat_id, arg="2147483647", page=9999),
        PanelCB(action="stat", chat_id=0, arg="", page=0),
        PanelCB(action="help", chat_id=0, arg="", page=0),
        PanelCB(action="noop", chat_id=0, arg="", page=0),
    ]
    for cb in worst_cases:
        packed = cb.pack()
        assert len(packed.encode("utf-8")) <= 64, f"{packed!r} exceeded 64 bytes ({len(packed.encode('utf-8'))})"


def test_pagination_edges():
    """Verify paginate() handles empty lists, exact multiples, overflow/negative pages."""
    assert paginate([], 0, 8) == ([], 0, 1)
    assert paginate([], -3, 8) == ([], 0, 1)
    assert paginate([1], 0, 8) == ([1], 0, 1)

    items_8 = list(range(8))
    assert paginate(items_8, 0, 8) == (items_8, 0, 1)
    assert paginate(items_8, 5, 8) == (items_8, 0, 1)

    items_9 = list(range(9))
    assert paginate(items_9, 0, 8) == (list(range(8)), 0, 2)
    assert paginate(items_9, 1, 8) == ([8], 1, 2)
    assert paginate(items_9, 99, 8) == ([8], 1, 2)
    assert paginate(items_9, -5, 8) == (list(range(8)), 0, 2)


def test_min_messages_parsing_persian_arabic_and_bounds():
    """parse_min_messages_input must accept ASCII/Persian/Arabic digits in 1..100 and reject invalid input."""
    assert parse_min_messages_input("1") == 1
    assert parse_min_messages_input("100") == 100
    assert parse_min_messages_input("  ۱۵ ") == 15
    assert parse_min_messages_input("۴۲") == 42
    assert parse_min_messages_input("١٠٠") == 100
    assert parse_min_messages_input("۰") is None
    assert parse_min_messages_input("0") is None
    assert parse_min_messages_input("101") is None
    assert parse_min_messages_input("۱۰۱") is None
    assert parse_min_messages_input("-5") is None
    assert parse_min_messages_input("abc") is None
    assert parse_min_messages_input("") is None
    assert parse_min_messages_input(None) is None


@pytest.mark.asyncio
async def test_role_resolution_and_shared_ttl_cache_with_admin_filter(tmp_path: Path):
    """Verify GLOBAL_ADMIN / GROUP_ADMIN / STRANGER resolution and shared 5-min TTL cache with AdminFilter."""
    clear_admin_cache()
    await init_db(str(tmp_path / "roles.db"))

    try:
        await repository.upsert_known_group(-10010, "گروه اول", is_forum=True)
        await repository.upsert_known_group(-10020, "گروه دوم", is_forum=True)

        bot = AsyncMock(spec=Bot)

        async def _get_member(chat_id: int, user_id: int):
            if chat_id == -10010 and user_id == 200:
                m = MagicMock(spec=ChatMemberAdministrator)
                m.status = "administrator"
                return m
            m = MagicMock(spec=ChatMemberMember)
            m.status = "member"
            return m

        bot.get_chat_member = AsyncMock(side_effect=_get_member)

        with patch("bot.services.access.get_settings") as mock_acc, patch(
            "bot.filters.get_settings"
        ) as mock_flt, patch("bot.filters.time.monotonic") as mock_mono:
            mock_acc.return_value.ADMIN_IDS = [100]
            mock_acc.return_value.ALLOWED_CHAT_IDS = []
            mock_flt.return_value.ADMIN_IDS = [100]
            mock_mono.return_value = 1000.0

            # 1. Global admin sees ALL active groups without calling get_chat_member
            role_g, groups_g = await resolve_user_role(bot, 100)
            assert role_g == UserRole.GLOBAL_ADMIN
            assert [g["chat_id"] for g in groups_g] == [-10010, -10020]
            assert bot.get_chat_member.call_count == 0

            # 2. Group admin sees only group -10010, populates shared TTL cache
            role_a, groups_a = await resolve_user_role(bot, 200)
            assert role_a == UserRole.GROUP_ADMIN
            assert [g["chat_id"] for g in groups_a] == [-10010]
            assert bot.get_chat_member.call_count == 2
            assert get_cached_admin_status(-10010, 200) is True
            assert get_cached_admin_status(-10020, 200) is False

            # 3. Group AdminFilter reuses the exact same cache (0 additional API calls!)
            grp_msg = MagicMock(spec=Message)
            grp_msg.chat = MagicMock(spec=Chat)
            grp_msg.chat.id = -10010
            grp_msg.sender_chat = None
            grp_msg.from_user = MagicMock(spec=User)
            grp_msg.from_user.id = 200
            assert await AdminFilter()(grp_msg, bot) is True
            assert bot.get_chat_member.call_count == 2

            # 4. After TTL expires (>300s), get_chat_member is queried again
            mock_mono.return_value = 1305.0
            role_a2, _ = await resolve_user_role(bot, 200)
            assert role_a2 == UserRole.GROUP_ADMIN
            assert bot.get_chat_member.call_count == 4

            # 5. Stranger gets STRANGER role and empty group list
            role_s, groups_s = await resolve_user_role(bot, 300)
            assert role_s == UserRole.STRANGER
            assert groups_s == []
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_panel_toggles_match_group_commands_and_html_escaping(tmp_path: Path):
    """Toggling enabled, mode, topic active, destination, and min_messages via panel updates DB identically to commands."""
    clear_admin_cache()
    await init_db(str(tmp_path / "panel_toggles.db"))

    try:
        chat_id = -100500
        await repository.upsert_known_group(chat_id, "گروه R&D <x>", is_forum=True)
        await repository.ensure_group_settings(chat_id, default_min_messages=5)
        await repository.upsert_topic(chat_id, 10, name="تاپیک R&D <x>")
        await repository.upsert_topic(chat_id, 20, name="تاپیک دوم")

        bot = AsyncMock(spec=Bot)
        fsm, _ = _make_fsm_memory()
        cb = _make_callback(user_id=100, first_name="R&D <x>")

        with patch("bot.services.access.get_settings") as mock_acc, patch(
            "bot.handlers.panel.get_settings"
        ) as mock_pan:
            for m in (mock_acc, mock_pan):
                m.return_value.ADMIN_IDS = [100]
                m.return_value.ALLOWED_CHAT_IDS = []
                m.return_value.MIN_MESSAGES_PER_TOPIC = 5
                m.return_value.AI_PROVIDER = "gemini"
                m.return_value.GEMINI_MODEL = "gemini-3.8-flash"
                m.return_value.SUMMARY_TIME_HOUR = 0
                m.return_value.SUMMARY_TIME_MINUTE = 1
                m.return_value.TIMEZONE = "Asia/Tehran"
                m.return_value.DB_PATH = str(tmp_path / "panel_toggles.db")

            # Verify dashboard escapes R&D <x>
            dash_text, _ = await render_group_dashboard(chat_id)
            assert "R&amp;D &lt;x&gt;" in dash_text
            assert "R&D <x>" not in dash_text

            # 1. Toggle enabled: 1 -> 0 -> 1
            await handle_panel_callback(cb, PanelCB(action="tog", chat_id=chat_id), bot, fsm)
            assert (await repository.ensure_group_settings(chat_id))["enabled"] == 0
            await handle_panel_callback(cb, PanelCB(action="tog", chat_id=chat_id), bot, fsm)
            assert (await repository.ensure_group_settings(chat_id))["enabled"] == 1

            # 2. Toggle mode: blacklist -> whitelist
            await handle_panel_callback(cb, PanelCB(action="mode", chat_id=chat_id), bot, fsm)
            assert (await repository.ensure_group_settings(chat_id))["mode"] == "whitelist"
            # Topic 10 has NULL is_active -> NOT monitored in whitelist mode
            assert not await repository.is_topic_monitored(chat_id, 10)

            # 3. Toggle topic 10 on via panel -> identical to /add_topic (is_active=1)
            await handle_panel_callback(cb, PanelCB(action="topt", chat_id=chat_id, arg="10", page=0), bot, fsm)
            assert await repository.is_topic_monitored(chat_id, 10)

            # Toggle topic 10 off -> identical to /remove_topic (is_active=0)
            await handle_panel_callback(cb, PanelCB(action="topt", chat_id=chat_id, arg="10", page=0), bot, fsm)
            assert not await repository.is_topic_monitored(chat_id, 10)

            # Bulk enable all -> both monitored
            await handle_panel_callback(cb, PanelCB(action="topa", chat_id=chat_id, arg="1", page=0), bot, fsm)
            assert await repository.is_topic_monitored(chat_id, 10)
            assert await repository.is_topic_monitored(chat_id, 20)

            # 4. Destination aggregate confirm -> identical to /set_destination
            await handle_panel_callback(cb, PanelCB(action="dstc", chat_id=chat_id, arg="10"), bot, fsm)
            gs = await repository.ensure_group_settings(chat_id)
            assert gs["destination_mode"] == "aggregate"
            assert gs["destination_thread_id"] == 10

            # Destination per_topic -> identical to /reset_destination
            await handle_panel_callback(cb, PanelCB(action="dstp", chat_id=chat_id), bot, fsm)
            gs = await repository.ensure_group_settings(chat_id)
            assert gs["destination_mode"] == "per_topic"
            assert gs["destination_thread_id"] is None

            # 5. Min messages stepper + FSM Persian digit input
            await handle_panel_callback(cb, PanelCB(action="mind", chat_id=chat_id, arg="5"), bot, fsm)
            assert (await repository.ensure_group_settings(chat_id))["min_messages"] == 10

            await handle_panel_callback(cb, PanelCB(action="minf", chat_id=chat_id), bot, fsm)
            assert await fsm.get_state() == PanelStates.waiting_min_messages

            # Invalid out-of-range input keeps state
            bad_msg = MagicMock(spec=Message)
            bad_msg.from_user = cb.from_user
            bad_msg.text = "۹۹۹"
            bad_msg.answer = AsyncMock()
            await handle_min_messages_fsm_input(bad_msg, bot, fsm)
            assert await fsm.get_state() == PanelStates.waiting_min_messages
            assert (await repository.ensure_group_settings(chat_id))["min_messages"] == 10

            # Valid Persian digits update DB and clear state
            good_msg = MagicMock(spec=Message)
            good_msg.from_user = cb.from_user
            good_msg.text = "۱۸"
            good_msg.answer = AsyncMock()
            await handle_min_messages_fsm_input(good_msg, bot, fsm)
            assert await fsm.get_state() is None
            assert (await repository.ensure_group_settings(chat_id))["min_messages"] == 18

            # Verify topics screen escapes group name
            topics_text, _ = await render_topics(chat_id, page=0)
            assert "R&amp;D &lt;x&gt;" in topics_text
            assert "R&D <x>" not in topics_text
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_manual_run_blocked_when_summary_lock_held(tmp_path: Path):
    """Manual run from panel must reject execution with friendly Persian notice when summary lock is held."""
    clear_admin_cache()
    await init_db(str(tmp_path / "panel_lock.db"))

    try:
        chat_id = -100600
        await repository.upsert_known_group(chat_id, "گروه لاک", is_forum=True)
        await repository.ensure_group_settings(chat_id)

        bot = AsyncMock(spec=Bot)
        fsm, _ = _make_fsm_memory()
        cb = _make_callback(user_id=100)
        lock = get_summary_lock()

        with patch("bot.services.access.get_settings") as mock_acc, patch(
            "bot.handlers.panel.run_summary_for_group"
        ) as mock_run:
            mock_acc.return_value.ADMIN_IDS = [100]
            mock_acc.return_value.ALLOWED_CHAT_IDS = []

            async with lock:
                await handle_panel_callback(
                    cb,
                    PanelCB(action="rune", chat_id=chat_id, arg="a_0", page=0),
                    bot,
                    fsm,
                )
                mock_run.assert_not_called()
                assert "در جریانه" in cb.answer.call_args.kwargs["text"]
                assert cb.answer.call_args.kwargs["show_alert"] is True
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_preview_sends_only_to_admin_dm_and_never_modifies_db(tmp_path: Path):
    """Preview (dry_run=True + destination_override) sends split summary ONLY to admin DM and never alters DB."""
    clear_admin_cache()
    await init_db(str(tmp_path / "preview_dry_run.db"))

    try:
        chat_id = -100700
        thread_id = 42
        admin_user_id = 100

        await repository.upsert_known_group(chat_id, "گروه تست پیش‌نمایش", is_forum=True)
        await repository.ensure_group_settings(chat_id, default_min_messages=2)
        await repository.upsert_topic(chat_id, thread_id, name="R&D <x>", is_active=1)

        for i in (1, 2):
            await repository.insert_message(
                chat_id=chat_id,
                thread_id=thread_id,
                message_id=i,
                user_id=50 + i,
                sender_name=f"User {i}",
                text=f"این یک پیام مهم تستی شماره {i} است.",
                media_type=None,
                reply_to_message_id=None,
                reply_to_name=None,
                reply_to_snippet=None,
                created_at="2026-11-22T10:00:00+00:00",
            )

        async def _snapshot_db() -> dict:
            db = get_db()
            snap = {}
            for table in ("messages", "topics", "group_settings", "summaries", "meta", "bot_users", "known_groups"):
                cur = await db.execute(f"SELECT * FROM {table};")
                snap[table] = [tuple(r) for r in await cur.fetchall()]
            return snap

        before = await _snapshot_db()

        bot = AsyncMock(spec=Bot)
        bot.send_message = AsyncMock()
        mock_llm = AsyncMock(spec=BaseLLMClient)
        mock_llm.generate.return_value = "<b>خلاصه آزمایشی</b> مباحث روز."

        success, details = await run_summary_for_topic(
            bot=bot,
            chat_id=chat_id,
            thread_id=thread_id,
            mark_summarized=True,  # Even if caller passed True, dry_run MUST NOT delete or mark!
            is_nightly=True,
            llm_client=mock_llm,
            destination_override=(admin_user_id, None),
            dry_run=True,
        )

        assert success is True
        assert "پیش‌نمایش" in details

        # Sent ONLY to admin DM (chat_id=admin_user_id, message_thread_id=None), NEVER to group chat_id!
        assert bot.send_message.call_count == 1
        call_kwargs = bot.send_message.call_args.kwargs
        assert call_kwargs["chat_id"] == admin_user_id
        assert call_kwargs["message_thread_id"] is None
        assert "R&amp;D &lt;x&gt;" in call_kwargs["text"]
        assert "R&D <x>" not in call_kwargs["text"]

        # Entire database must be bit-for-bit identical before and after preview!
        after = await _snapshot_db()
        assert after == before
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_global_status_restricted_to_global_admins_and_stale_callback(tmp_path: Path):
    """Non-global admins cannot view global status; stale callbacks answer gracefully."""
    clear_admin_cache()
    await init_db(str(tmp_path / "global_stat.db"))

    try:
        bot = AsyncMock(spec=Bot)
        fsm, _ = _make_fsm_memory()

        # Non-global user attempts to trigger stat callback
        cb_denied = _make_callback(user_id=200, first_name="ادمین گروه")
        with patch("bot.services.access.get_settings") as mock_acc:
            mock_acc.return_value.ADMIN_IDS = [100]
            mock_acc.return_value.ALLOWED_CHAT_IDS = []

            await handle_panel_callback(cb_denied, PanelCB(action="stat"), bot, fsm)
            cb_denied.answer.assert_called_once()
            assert cb_denied.answer.call_args.kwargs["show_alert"] is True
            cb_denied.message.edit_text.assert_not_called()

        # Stale / unknown callback answers gracefully
        cb_stale = _make_callback(user_id=100)
        await handle_stale_callback(cb_stale)
        cb_stale.answer.assert_called_once()
        assert "قدیمی" in cb_stale.answer.call_args.kwargs["text"]
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_manual_run_live_progress_and_final_report_and_group_admin_isolation(tmp_path: Path):
    """Verify live progress edits + final report on manual group run, single-group auto-dashboard, and cross-group isolation."""
    clear_admin_cache()
    await init_db(str(tmp_path / "panel_progress_and_iso.db"))

    try:
        chat_a = -100801
        chat_b = -100802
        await repository.upsert_known_group(chat_a, "گروه الف", is_forum=True)
        await repository.upsert_known_group(chat_b, "گروه ب", is_forum=True)
        await repository.ensure_group_settings(chat_a)
        await repository.ensure_group_settings(chat_b)

        bot = AsyncMock(spec=Bot)

        async def _get_member(chat_id: int, user_id: int):
            if chat_id == chat_a and user_id == 200:
                m = MagicMock(spec=ChatMemberAdministrator)
                m.status = "administrator"
                return m
            m = MagicMock(spec=ChatMemberMember)
            m.status = "member"
            return m

        bot.get_chat_member = AsyncMock(side_effect=_get_member)
        fsm, _ = _make_fsm_memory()

        with patch("bot.services.access.get_settings") as mock_acc, patch(
            "bot.handlers.panel.get_settings"
        ) as mock_pan:
            for m in (mock_acc, mock_pan):
                m.return_value.ADMIN_IDS = [100]
                m.return_value.ALLOWED_CHAT_IDS = []
                m.return_value.MIN_MESSAGES_PER_TOPIC = 5
                m.return_value.AI_PROVIDER = "gemini"
                m.return_value.GEMINI_MODEL = "gemini-3.8-flash"
                m.return_value.SUMMARY_TIME_HOUR = 0
                m.return_value.SUMMARY_TIME_MINUTE = 1
                m.return_value.TIMEZONE = "Asia/Tehran"
                m.return_value.DB_PATH = str(tmp_path / "panel_progress_and_iso.db")

            # Group admin (200) has exactly 1 manageable group (chat_a) -> "open" jumps straight to dashboard
            cb_ga = _make_callback(user_id=200, first_name="ادمین الف")
            await handle_panel_callback(cb_ga, PanelCB(action="open"), bot, fsm)
            dash_sent = cb_ga.message.edit_text.call_args.kwargs["text"]
            assert "داشبورد گروه" in dash_sent and "گروه الف" in dash_sent

            # Group admin (200) attempting to toggle chat_b MUST be denied!
            cb_ga.answer.reset_mock()
            cb_ga.message.edit_text.reset_mock()
            await handle_panel_callback(cb_ga, PanelCB(action="tog", chat_id=chat_b), bot, fsm)
            assert cb_ga.answer.call_args.kwargs["show_alert"] is True
            cb_ga.message.edit_text.assert_not_called()

            # Manual run for all topics invokes progress_callback and renders final report
            async def _fake_group_run(*, progress_callback=None, **kwargs):
                if progress_callback:
                    await progress_callback(1, 2, "R&D <x>")
                    await progress_callback(2, 2, "عمومی")
                return {"success": 1, "skipped": 1, "failed": 0}

            cb_run = _make_callback(user_id=100, first_name="آرش")
            with patch("bot.handlers.panel.run_summary_for_group", side_effect=_fake_group_run):
                await handle_panel_callback(
                    cb_run,
                    PanelCB(action="rune", chat_id=chat_a, arg="a_1", page=0),
                    bot,
                    fsm,
                )

            edits = [c.kwargs["text"] for c in cb_run.message.edit_text.call_args_list]
            # Initial progress + 2 topic progress updates (with escaped R&D <x>) + final report
            assert any("1 از 2" in t and "R&amp;D &lt;x&gt;" in t for t in edits)
            assert any("2 از 2" in t for t in edits)
            assert "گزارش اجرای دستی" in edits[-1]
            assert "(حالت keep)" in edits[-1]
            assert "موفق: <b>1</b>" in edits[-1]
            assert "کمتر از حد نصاب: <b>1</b>" in edits[-1]
    finally:
        await close_db()

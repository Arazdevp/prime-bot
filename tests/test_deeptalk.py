"""Comprehensive offline unit & integration tests for the /deeptalk system."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import re
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from zoneinfo import ZoneInfo
import aiosqlite
import pytest
from aiogram import Bot
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, Chat, ChatMemberAdministrator, ChatMemberLeft, ChatMemberMember, Message, User
from bot.database import repository
from bot.database.db import close_db, get_db, init_db
from bot.filters import clear_admin_cache
from bot.handlers.deeptalk import (
    DeepTalkStates,
    clear_deeptalk_drafts,
    cmd_deeptalk,
    cmd_set_deeptalk_topic,
    handle_deeptalk_callback,
    handle_deeptalk_fsm_input,
)
from bot.handlers.listener import capture_message, clear_known_group_cache
from bot.handlers.private import clear_stranger_cooldowns, cmd_private_start
from bot.keyboards.deeptalk import DeepTalkCB
from bot.scheduler.jobs import build_scheduler, set_scheduler
from bot.services.access import clear_member_cache
from bot.services.deeptalk_service import (
    clear_deeptalk_runtime_cache,
    create_and_publish_deeptalk,
    end_deeptalk_session,
    rehydrate_deeptalk_sessions,
    start_deeptalk_session,
    sweep_due_deeptalks,
)
from bot.services.timer_service import set_bot_username
from bot.texts.deeptalk_texts import (
    DEEPTALK_CHECKIN_MESSAGES,
    draw_next_checkin_index,
    format_checkin_announcement,
)
from bot.texts.private_texts import clear_last_picked_cache
from bot.utils.timer_parser import to_utc_iso

TEHRAN_TZ = ZoneInfo("Asia/Tehran")


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


def _make_fsm() -> tuple[FSMContext, dict]:
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


def _make_msg(
    *,
    chat_id: int,
    chat_type: str = "supergroup",
    thread_id: int = 77,
    message_id: int = 500,
    user_id: int = 1001,
    username: str | None = "arash_owner",
    first_name: str = "آرش",
    full_name: str = "آرش مالک",
    text: str | None = "/deeptalk",
    bot: Bot | None = None,
    forward_origin: object | None = None,
) -> Message:
    msg = MagicMock(spec=Message)
    msg.message_id = message_id
    msg.date = datetime(2026, 11, 24, 18, 0, 0, tzinfo=timezone.utc)
    msg.chat = MagicMock(spec=Chat)
    msg.chat.id = chat_id
    msg.chat.type = chat_type
    msg.chat.title = "Hills Prime" if chat_type != "private" else None
    msg.chat.is_forum = chat_type == "supergroup"
    msg.is_topic_message = chat_type == "supergroup" and thread_id != 1
    msg.message_thread_id = None if (chat_type == "private" or thread_id == 1) else thread_id
    msg.text = text
    msg.caption = None
    msg.photo = None
    msg.video = None
    msg.document = None
    msg.voice = None
    msg.audio = None
    msg.video_note = None
    msg.animation = None
    msg.sticker = None
    msg.sender_chat = None
    msg.reply_to_message = None
    msg.forum_topic_created = None
    msg.forum_topic_edited = None
    msg.forward_origin = forward_origin
    msg.forward_from = None

    user = MagicMock(spec=User)
    user.id = user_id
    user.is_bot = False
    user.username = username
    user.first_name = first_name
    user.full_name = full_name
    msg.from_user = user

    msg.bot = bot or AsyncMock(spec=Bot)
    msg.delete = AsyncMock()
    msg.answer = AsyncMock(side_effect=lambda *args, **kw: SimpleNamespace(message_id=900))
    return msg


def _make_cb(
    *,
    user_id: int,
    username: str | None = "user_x",
    first_name: str = "کاربر",
    full_name: str = "کاربر تست",
    chat_id: int = -1001234567890,
    chat_type: str = "supergroup",
    thread_id: int = 77,
    message_id: int = 900,
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
    msg.is_topic_message = chat_type == "supergroup" and thread_id != 1
    msg.message_thread_id = None if (chat_type == "private" or thread_id == 1) else thread_id
    msg.edit_text = AsyncMock()
    msg.delete = AsyncMock()
    cb.message = msg
    return cb


def test_deeptalk_checkin_pool_has_20_unique_valid_html_templates():
    """Verify exactly 20 unique Persian check-in templates with valid balanced HTML and escaping."""
    assert len(DEEPTALK_CHECKIN_MESSAGES) == 20
    assert len(set(DEEPTALK_CHECKIN_MESSAGES)) == 20

    rendered_set: set[str] = set()
    for idx, tpl in enumerate(DEEPTALK_CHECKIN_MESSAGES):
        assert "{user}" in tpl, f"Template #{idx} missing {{user}}"
        rendered = format_checkin_announcement(
            idx,
            user_id=777,
            username="dev_user",
            full_name="Ali <R&D>",
        )
        assert "Ali &lt;R&amp;D&gt;" in rendered
        assert "Ali <R&D>" not in rendered
        assert "@dev_user" in rendered
        assert _is_html_balanced(rendered), f"Unbalanced HTML in #{idx}: {rendered}"
        rendered_set.add(rendered)

    assert len(rendered_set) == 20


def test_deeptalk_shuffle_bag_guarantees_20_unique_per_cycle_and_no_boundary_repeat():
    """Verify 40 sequential check-in draws yield two full 20-item duplicate-free cycles with no boundary repeat."""
    deck: list[int] | None = None
    last_idx: int | None = None
    draws: list[int] = []

    for _ in range(40):
        chosen, deck = draw_next_checkin_index(deck, last_idx)
        draws.append(chosen)
        last_idx = chosen

    c1 = draws[:20]
    c2 = draws[20:40]
    assert len(set(c1)) == 20
    assert set(c1) == set(range(20))
    assert len(set(c2)) == 20
    assert set(c2) == set(range(20))
    assert c1[-1] != c2[0]


def test_deeptalk_callback_data_under_64_bytes_worst_case():
    """Verify all DeepTalkCB payloads remain strictly <= 64 bytes."""
    worst_uid = 9999999999
    worst_cid = -1009999999999999
    cases = [
        DeepTalkCB(action="reg", sid=999999),
        DeepTalkCB(action="unreg", sid=999999),
        DeepTalkCB(action="chk", sid=999999),
        DeepTalkCB(action="plist", sid=999999),
        DeepTalkCB(action="pref", sid=999999),
        DeepTalkCB(action="home", sid=worst_uid),
        DeepTalkCB(action="new", sid=worst_uid),
        DeepTalkCB(action="grp", sid=worst_uid, arg=str(worst_cid)),
        DeepTalkCB(action="top", sid=worst_uid, arg="w99"),
        DeepTalkCB(action="tops", sid=worst_uid, arg="w_2147483647"),
        DeepTalkCB(action="cal", sid=worst_uid, arg="1405-12"),
        DeepTalkCB(action="day", sid=worst_uid, arg="1405-12-29"),
        DeepTalkCB(action="ext", sid=worst_uid, arg="999999_30"),
        DeepTalkCB(action="enow", sid=worst_uid, arg="999999_yes"),
        DeepTalkCB(action="canc", sid=worst_uid, arg="999999_yes"),
        DeepTalkCB(action="admd", sid=worst_uid, arg=str(worst_uid)),
    ]
    for cb in cases:
        packed = cb.pack()
        assert len(packed.encode("utf-8")) <= 64, f"{packed!r} exceeded 64 bytes"


@pytest.mark.asyncio
async def test_migration_v6_to_v7_creates_deeptalk_tables_and_preserves_data(tmp_path: Path):
    """Upgrading from user_version=6 creates all 4 DeepTalk tables without losing existing data."""
    db_file = tmp_path / "v6_to_v7.db"
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
        VALUES (-100888, 'blacklist', 'per_topic', NULL, 1, 7);
        PRAGMA user_version = 6;
        """
    )
    await conn.commit()
    await conn.close()

    try:
        await init_db(str(db_file))
        db = get_db()
        cur = await db.execute("PRAGMA user_version;")
        assert (await cur.fetchone())[0] >= 7

        gs = await repository.get_group_settings(-100888)
        assert gs is not None and gs["min_messages"] == 7

        adm = await repository.add_deeptalk_admin(
            user_id=2002,
            username="@captain_prime",
            full_name="کاپیتان پرایم",
            added_by=1001,
        )
        assert adm["username"] == "captain_prime"
        assert await repository.is_deeptalk_admin(2002) is True
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_owner_admin_management_and_strict_deeptalk_access_control(tmp_path: Path):
    """Only bot owners can manage DeepTalk admins, and ONLY owners + DeepTalk admins can open /deeptalk."""
    clear_deeptalk_drafts()
    clear_deeptalk_runtime_cache()
    clear_admin_cache()
    clear_member_cache()
    clear_stranger_cooldowns()
    clear_last_picked_cache()
    await init_db(str(tmp_path / "dt_access.db"))

    owner_id = 1001
    tg_group_admin_id = 2002   # Telegram group admin, NOT owner, NOT DeepTalk admin
    dt_admin_by_id = 3003
    dt_admin_by_uname = 3004
    dt_admin_by_fwd = 3005
    chat_id = -1001234567890

    try:
        await repository.upsert_known_group(chat_id, "Hills Prime", is_forum=True)
        await repository.upsert_topic(chat_id, 77, name="Deep Work | دیپ‌ورک")
        await repository.upsert_bot_user_seen(dt_admin_by_uname, "parsa_host", "پارسا")

        bot = AsyncMock(spec=Bot)

        async def _get_member(chat_id: int, user_id: int):
            if user_id == tg_group_admin_id:
                m = MagicMock(spec=ChatMemberAdministrator)
                m.status = "administrator"
                return m
            m = MagicMock(spec=ChatMemberMember)
            m.status = "member"
            return m

        bot.get_chat_member = AsyncMock(side_effect=_get_member)
        bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=901))
        bot.edit_message_text = AsyncMock()
        bot.get_chat = AsyncMock(return_value=SimpleNamespace(id=dt_admin_by_id, username="ali_dt", full_name="علی"))

        with patch("bot.services.access.get_settings") as mock_acc, patch(
            "bot.handlers.deeptalk.get_settings"
        ) as mock_dt, patch("bot.handlers.private.get_settings") as mock_prv:
            for m in (mock_acc, mock_dt, mock_prv):
                m.return_value.ADMIN_IDS = [owner_id]
                m.return_value.ALLOWED_CHAT_IDS = []
                m.return_value.TIMEZONE = "Asia/Tehran"
                m.return_value.OWNER_NAMES = "آرش و پیت"
                m.return_value.NOTIFY_ON_UNKNOWN_START = False

            fsm, _ = _make_fsm()

            # 1. Ordinary Telegram Group Admin CANNOT open /deeptalk in group (deleted silently)
            grp_msg_unauth = _make_msg(
                chat_id=chat_id,
                user_id=tg_group_admin_id,
                text="/deeptalk",
                bot=bot,
            )
            await cmd_deeptalk(grp_msg_unauth, bot, fsm)
            grp_msg_unauth.delete.assert_awaited_once()
            bot.send_message.assert_not_called()

            # Same group admin in PV gets polite restriction notice
            pv_msg_unauth = _make_msg(
                chat_id=tg_group_admin_id,
                chat_type="private",
                user_id=tg_group_admin_id,
                text="/deeptalk",
                bot=bot,
            )
            await cmd_deeptalk(pv_msg_unauth, bot, fsm)
            unauth_reply_text = (
                pv_msg_unauth.answer.call_args.args[0]
                if pv_msg_unauth.answer.call_args.args
                else pv_msg_unauth.answer.call_args.kwargs["text"]
            )
            assert "فقط مخصوص مالک ربات و ادمین‌های معرفی‌شده دیپ‌تاک" in unauth_reply_text

            # 2. Bot Owner opens /deeptalk in PV and adds 3 DeepTalk admins (by Persian ID, @username, forward)
            owner_pv = _make_msg(
                chat_id=owner_id,
                chat_type="private",
                user_id=owner_id,
                text="/deeptalk",
                bot=bot,
            )
            await cmd_deeptalk(owner_pv, bot, fsm)
            owner_pv.answer.assert_awaited_once()

            owner_cb = _make_cb(user_id=owner_id, chat_id=owner_id, chat_type="private", message_id=900)

            # Add via Persian numeric ID
            await handle_deeptalk_callback(owner_cb, DeepTalkCB(action="adma", sid=owner_id), bot, fsm)
            assert await fsm.get_state() == DeepTalkStates.waiting_new_admin.state
            id_input = _make_msg(chat_id=owner_id, chat_type="private", user_id=owner_id, text="۳۰۰۳", bot=bot)
            await handle_deeptalk_fsm_input(id_input, bot, fsm)
            assert await repository.is_deeptalk_admin(dt_admin_by_id) is True

            # Add via @username
            await handle_deeptalk_callback(owner_cb, DeepTalkCB(action="adma", sid=owner_id), bot, fsm)
            uname_input = _make_msg(chat_id=owner_id, chat_type="private", user_id=owner_id, text="@parsa_host", bot=bot)
            await handle_deeptalk_fsm_input(uname_input, bot, fsm)
            assert await repository.is_deeptalk_admin(dt_admin_by_uname) is True

            # Add via forwarded message
            await handle_deeptalk_callback(owner_cb, DeepTalkCB(action="adma", sid=owner_id), bot, fsm)
            fwd_user = SimpleNamespace(id=dt_admin_by_fwd, is_bot=False, username="sara_mod", full_name="سارا")
            fwd_origin = SimpleNamespace(sender_user=fwd_user)
            fwd_input = _make_msg(
                chat_id=owner_id,
                chat_type="private",
                user_id=owner_id,
                text="",
                bot=bot,
                forward_origin=fwd_origin,
            )
            await handle_deeptalk_fsm_input(fwd_input, bot, fsm)
            assert await repository.is_deeptalk_admin(dt_admin_by_fwd) is True

            # 3. Newly added DeepTalk admin CAN open /deeptalk and /start in PV, but CANNOT open 'adm' screen!
            dt_fsm, _ = _make_fsm()
            dt_start = _make_msg(
                chat_id=dt_admin_by_uname,
                chat_type="private",
                user_id=dt_admin_by_uname,
                username="parsa_host",
                text="/start",
                bot=bot,
            )
            await cmd_private_start(dt_start, bot, dt_fsm)
            assert "ادمین دیپ‌تاک" in dt_start.answer.call_args.kwargs["text"]

            dt_open = _make_msg(
                chat_id=dt_admin_by_uname,
                chat_type="private",
                user_id=dt_admin_by_uname,
                username="parsa_host",
                text="/deeptalk",
                bot=bot,
            )
            await cmd_deeptalk(dt_open, bot, dt_fsm)
            assert "پنل مدیریت دیپ‌تاک" in dt_open.answer.call_args.kwargs["text"]

            # DeepTalk admin trying owner-only 'adm' action is rejected
            dt_cb = _make_cb(
                user_id=dt_admin_by_uname,
                chat_id=dt_admin_by_uname,
                chat_type="private",
                message_id=900,
            )
            await handle_deeptalk_callback(dt_cb, DeepTalkCB(action="adm", sid=dt_admin_by_uname), bot, dt_fsm)
            assert "فقط مالک اصلی ربات" in dt_cb.answer.call_args.kwargs["text"]

            # 4. Owner revokes DeepTalk admin -> access immediately revoked
            await handle_deeptalk_callback(
                owner_cb,
                DeepTalkCB(action="admd", sid=owner_id, arg=str(dt_admin_by_uname)),
                bot,
                fsm,
            )
            assert await repository.is_deeptalk_admin(dt_admin_by_uname) is False
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_end_to_end_deeptalk_lifecycle_group_pv_checkins_extension_and_stats(tmp_path: Path):
    """Full lifecycle: publish -> register/unreg -> PV list -> start + DM ping -> 20-pool check-in -> extend -> end stats."""
    clear_deeptalk_drafts()
    clear_deeptalk_runtime_cache()
    clear_known_group_cache()
    clear_member_cache()
    set_bot_username("prime_summarizer_bot")
    await init_db(str(tmp_path / "dt_lifecycle.db"))

    owner_id = 1001
    dt_admin_id = 2001
    user_a = 4001  # Pre-registered + dm_started + checks in first + sends 3 messages
    user_b = 4002  # Pre-registered + misses check-in ('جامانده')
    user_c = 4003  # Regular group member viewing PV list via deep link
    user_d = 4004  # Walk-in (not pre-registered, checks in live + sends 1 message)
    outsider_id = 9999

    chat_id = -1001234567890
    deepwork_tid = 77

    sent_messages: list[dict] = []
    next_mid = 1000

    async def _fake_send_message(**kwargs):
        nonlocal next_mid
        next_mid += 1
        sent_messages.append({"message_id": next_mid, **kwargs})
        return SimpleNamespace(message_id=next_mid)

    bot = AsyncMock(spec=Bot)
    bot.send_message = AsyncMock(side_effect=_fake_send_message)
    bot.edit_message_text = AsyncMock()

    async def _fake_get_member(chat_id: int, user_id: int):
        if user_id == outsider_id:
            m = MagicMock(spec=ChatMemberLeft)
            m.status = "left"
            return m
        m = MagicMock(spec=ChatMemberMember)
        m.status = "member"
        return m

    bot.get_chat_member = AsyncMock(side_effect=_fake_get_member)
    scheduler = build_scheduler(bot)
    set_scheduler(scheduler)

    try:
        await repository.upsert_known_group(chat_id, "Hills Prime", is_forum=True)
        await repository.ensure_group_settings(chat_id)
        await repository.upsert_topic(chat_id, deepwork_tid, name="Deep Work | دیپ‌ورک")
        await repository.add_deeptalk_admin(
            user_id=dt_admin_id,
            username="captain_dt",
            full_name="کاپیتان دیپ‌تاک",
            added_by=owner_id,
        )
        await repository.set_bot_user_dm_started(user_a, True, username="ali_creator", full_name="علی کریتور")

        with patch("bot.services.access.get_settings") as mock_acc, patch(
            "bot.handlers.deeptalk.get_settings"
        ) as mock_dt, patch("bot.services.deeptalk_service.get_settings") as mock_svc, patch(
            "bot.handlers.private.get_settings"
        ) as mock_prv, patch("bot.handlers.listener.get_settings") as mock_lst:
            for m in (mock_acc, mock_dt, mock_svc, mock_prv, mock_lst):
                m.return_value.ADMIN_IDS = [owner_id]
                m.return_value.ALLOWED_CHAT_IDS = []
                m.return_value.TIMEZONE = "Asia/Tehran"
                m.return_value.OWNER_NAMES = "آرش و پیت"
                m.return_value.NOTIFY_ON_UNKNOWN_START = False
                m.return_value.MIN_MESSAGE_LENGTH = 3

            # 0. Verify auto-detection of 'Deep Work | دیپ‌ورک' topic
            assert await repository.detect_deepwork_topic_id(chat_id) == deepwork_tid

            # Also verify /set_deeptalk_topic command works for DeepTalk admin
            set_cmd = _make_msg(
                chat_id=chat_id,
                thread_id=deepwork_tid,
                user_id=dt_admin_id,
                text="/set_deeptalk_topic",
                bot=bot,
            )
            await cmd_set_deeptalk_topic(set_cmd)
            assert (await repository.get_deeptalk_settings(chat_id))["thread_id"] == deepwork_tid
            sent_messages.clear()

            # 1. DeepTalk admin opens '/deeptalk ساخت هوک یوتیوب' directly inside Deep Work topic
            fsm, _ = _make_fsm()
            open_msg = _make_msg(
                chat_id=chat_id,
                thread_id=deepwork_tid,
                user_id=dt_admin_id,
                username="captain_dt",
                full_name="کاپیتان دیپ‌تاک",
                text="/deeptalk ساخت هوک ۳۰ ثانیه اول در یوتیوب",
                bot=bot,
            )
            await cmd_deeptalk(open_msg, bot, fsm)
            open_msg.delete.assert_awaited_once()
            assert len(sent_messages) == 1
            wizard_mid = sent_messages[0]["message_id"]
            assert "ساخت هوک ۳۰ ثانیه اول در یوتیوب" in sent_messages[0]["text"]

            # Set quick start 30m from now and duration 45m, then publish!
            admin_cb = _make_cb(
                user_id=dt_admin_id,
                username="captain_dt",
                full_name="کاپیتان دیپ‌تاک",
                chat_id=chat_id,
                thread_id=deepwork_tid,
                message_id=wizard_mid,
            )
            await handle_deeptalk_callback(admin_cb, DeepTalkCB(action="qst", sid=dt_admin_id, arg="30"), bot, fsm)
            await handle_deeptalk_callback(admin_cb, DeepTalkCB(action="dur", sid=dt_admin_id, arg="45"), bot, fsm)
            sent_messages.clear()

            await handle_deeptalk_callback(admin_cb, DeepTalkCB(action="pub", sid=dt_admin_id), bot, fsm)
            # Wizard message inside the topic was deleted so only official announcement card remains!
            admin_cb.message.delete.assert_awaited_once()
            assert len(sent_messages) == 1
            ann_post = sent_messages[0]
            assert "فراخوان دیپ‌تاک جدید" in ann_post["text"]
            assert "ساخت هوک ۳۰ ثانیه اول در یوتیوب" in ann_post["text"]

            sessions = await repository.list_active_or_scheduled_deeptalks(chat_id)
            assert len(sessions) == 1
            sid = int(sessions[0]["id"])

            # Check deep-link URL button on announcement card
            ann_kb = ann_post["reply_markup"]
            pv_btn = ann_kb.inline_keyboard[1][0]
            assert pv_btn.url == f"https://t.me/prime_summarizer_bot?start=dt_{sid}"

            # 2. Pre-registration by User A and User B
            cb_a = _make_cb(user_id=user_a, username="ali_creator", full_name="علی کریتور", chat_id=chat_id)
            await handle_deeptalk_callback(cb_a, DeepTalkCB(action="reg", sid=sid), bot, fsm)
            # Duplicate tap by User A
            await handle_deeptalk_callback(cb_a, DeepTalkCB(action="reg", sid=sid), bot, fsm)

            cb_b = _make_cb(user_id=user_b, username="sara_editor", full_name="سارا ادیتور", chat_id=chat_id)
            await handle_deeptalk_callback(cb_b, DeepTalkCB(action="reg", sid=sid), bot, fsm)
            # User B unregisters then registers again
            await handle_deeptalk_callback(cb_b, DeepTalkCB(action="unreg", sid=sid), bot, fsm)
            assert (await repository.get_deeptalk_session_stats(sid))["registered_count"] == 1
            await handle_deeptalk_callback(cb_b, DeepTalkCB(action="reg", sid=sid), bot, fsm)
            assert (await repository.get_deeptalk_session_stats(sid))["registered_count"] == 2

            # 3. Regular group member User C clicks deep-link -> '/start dt_<sid>' in PV
            c_fsm, _ = _make_fsm()
            c_start = _make_msg(
                chat_id=user_c,
                chat_type="private",
                user_id=user_c,
                username="reza_member",
                full_name="رضا عضو",
                text=f"/start dt_{sid}",
                bot=bot,
            )
            await cmd_private_start(c_start, bot, c_fsm)
            pv_reply_text = c_start.answer.call_args.kwargs["text"]
            assert "@ali_creator" in pv_reply_text
            assert "@sara_editor" in pv_reply_text
            assert await repository.is_bot_user_dm_started(user_c) is True

            # Outsider NOT in group trying '/start dt_<sid>' is rejected!
            out_fsm, _ = _make_fsm()
            out_start = _make_msg(
                chat_id=outsider_id,
                chat_type="private",
                user_id=outsider_id,
                username="hacker",
                text=f"/start dt_{sid}",
                bot=bot,
            )
            await cmd_private_start(out_start, bot, out_fsm)
            out_reply = out_start.answer.call_args.kwargs["text"]
            assert "@ali_creator" not in out_reply

            # 4. Start time arrives!
            sent_messages.clear()
            t_start = datetime(2026, 11, 24, 18, 30, 0, tzinfo=timezone.utc)
            started = await start_deeptalk_session(bot, sid, now_utc=t_start)
            assert started is True

            # Topic received Start Card AND User A (dm_started=1) received DM reminder!
            topic_starts = [m for m in sent_messages if m["chat_id"] == chat_id]
            dm_pings = [m for m in sent_messages if m["chat_id"] == user_a]
            assert len(topic_starts) == 1
            assert "بحث دیپ‌تاک رسماً شروع شد" in topic_starts[0]["text"]
            assert len(dm_pings) == 1
            assert "دیپ‌تاکی که توش ثبت‌نام کرده بودی شروع شد" in dm_pings[0]["text"]

            # 5. Live Check-ins: User A (first responder, خوش‌قول) and User D (walk-in, ورود یهویی)
            sent_messages.clear()
            await handle_deeptalk_callback(cb_a, DeepTalkCB(action="chk", sid=sid), bot, fsm)
            assert len(sent_messages) == 1
            first_checkin_msg = sent_messages[0]["text"]
            assert "@ali_creator" in first_checkin_msg

            # Duplicate check-in tap by User A must NOT post another topic message!
            await handle_deeptalk_callback(cb_a, DeepTalkCB(action="chk", sid=sid), bot, fsm)
            assert len(sent_messages) == 1

            cb_d = _make_cb(user_id=user_d, username="omid_walkin", full_name="امید یهویی", chat_id=chat_id)
            await handle_deeptalk_callback(cb_d, DeepTalkCB(action="chk", sid=sid), bot, fsm)
            assert len(sent_messages) == 2
            second_checkin_msg = sent_messages[1]["text"]
            assert "@omid_walkin" in second_checkin_msg

            # 6. Live discussion messages captured by listener -> increments message_count!
            for i in range(3):
                m_a = _make_msg(
                    chat_id=chat_id,
                    thread_id=deepwork_tid,
                    message_id=2000 + i,
                    user_id=user_a,
                    username="ali_creator",
                    full_name="علی کریتور",
                    text=f"نکته شماره {i + 1} درباره هوک ویدیو اینه که تعلیق بسازیم",
                    bot=bot,
                )
                await capture_message(m_a)

            m_d = _make_msg(
                chat_id=chat_id,
                thread_id=deepwork_tid,
                message_id=2010,
                user_id=user_d,
                username="omid_walkin",
                full_name="امید یهویی",
                text="کاملاً موافقم ریتم ادیت ثانیه‌های اول هم خیلی مهمه",
                bot=bot,
            )
            await capture_message(m_d)

            # 7. DeepTalk admin extends live session by +15 minutes from PV panel!
            sent_messages.clear()
            pv_admin_cb = _make_cb(
                user_id=dt_admin_id,
                chat_id=dt_admin_id,
                chat_type="private",
                message_id=950,
            )
            await handle_deeptalk_callback(
                pv_admin_cb,
                DeepTalkCB(action="ext", sid=dt_admin_id, arg=f"{sid}_15"),
                bot,
                fsm,
            )
            assert len(sent_messages) == 1
            assert "تمدید شد" in sent_messages[0]["text"]
            updated_sess = await repository.get_deeptalk_session(sid)
            assert updated_sess["duration_minutes"] == 60

            # 8. End session -> Wrap-up card with fun stats!
            sent_messages.clear()
            ended = await end_deeptalk_session(bot, sid, now_utc=t_start + timedelta(minutes=60))
            assert ended is True
            assert len(sent_messages) == 1
            end_card = sent_messages[0]["text"]
            assert "وقت دیپ‌تاک به پایان رسید! می‌تونید جمع‌بندی کنید!" in end_card
            assert "سریع‌ترین اعلام حضور" in end_card
            assert "@ali_creator" in end_card
            assert "۴ پیام" in end_card
            assert "علی کریتور" in end_card
            assert "امید یهویی" in end_card

            # 9. PV refresh ('pref') after end shows خوش‌قول, ورود یهویی, and جامانده‌ها!
            pv_c_cb = _make_cb(user_id=user_c, chat_id=user_c, chat_type="private", message_id=960)
            await handle_deeptalk_callback(pv_c_cb, DeepTalkCB(action="pref", sid=sid), bot, c_fsm)
            refreshed_pv = pv_c_cb.message.edit_text.call_args.kwargs["text"]
            assert "خوش‌قول" in refreshed_pv
            assert "ورود یهویی" in refreshed_pv
            assert "جامانده‌ها" in refreshed_pv
            assert "@sara_editor" in refreshed_pv
    finally:
        set_scheduler(None)
        await close_db()


@pytest.mark.asyncio
async def test_rehydrate_and_sweep_deeptalk_sessions(tmp_path: Path):
    """Verify startup rehydration starts in-window overdue sessions, ends expired live ones, and cancels stale ones."""
    clear_deeptalk_runtime_cache()
    await init_db(str(tmp_path / "dt_rehydrate.db"))

    bot = AsyncMock(spec=Bot)
    bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=700))
    bot.edit_message_text = AsyncMock()

    chat_id = -10055555
    now_utc = datetime(2026, 11, 24, 20, 0, 0, tzinfo=timezone.utc)

    try:
        await repository.upsert_known_group(chat_id, "Prime Group", is_forum=True)

        # Session 1: Scheduled, window completely passed while bot was offline -> cancelled silently
        s_stale = await repository.create_deeptalk_session(
            chat_id=chat_id,
            thread_id=10,
            subject="جلسه قدیمی",
            description=None,
            created_by=1001,
            creator_name="آرش",
            start_at_iso=to_utc_iso(now_utc - timedelta(hours=3)),
            end_at_iso=to_utc_iso(now_utc - timedelta(hours=2)),
            duration_minutes=60,
        )

        # Session 2: Scheduled, start_at passed 10m ago, end_at in 35m -> starts now!
        s_in_window = await repository.create_deeptalk_session(
            chat_id=chat_id,
            thread_id=10,
            subject="جلسه در جریان",
            description=None,
            created_by=1001,
            creator_name="آرش",
            start_at_iso=to_utc_iso(now_utc - timedelta(minutes=10)),
            end_at_iso=to_utc_iso(now_utc + timedelta(minutes=35)),
            duration_minutes=45,
        )

        # Session 3: Live, end_at passed 5m ago -> ends now!
        s_live_due = await repository.create_deeptalk_session(
            chat_id=chat_id,
            thread_id=20,
            subject="جلسه تمام‌شده",
            description=None,
            created_by=1001,
            creator_name="آرش",
            start_at_iso=to_utc_iso(now_utc - timedelta(minutes=50)),
            end_at_iso=to_utc_iso(now_utc - timedelta(minutes=5)),
            duration_minutes=45,
        )
        await repository.claim_deeptalk_for_start(int(s_live_due["id"]))

        with patch("bot.services.deeptalk_service.get_settings") as mock_svc:
            mock_svc.return_value.TIMEZONE = "Asia/Tehran"

            count = await rehydrate_deeptalk_sessions(bot, now_utc=now_utc)
            assert count == 3

            assert (await repository.get_deeptalk_session(int(s_stale["id"])))["status"] == "cancelled"
            assert (await repository.get_deeptalk_session(int(s_in_window["id"])))["status"] == "live"
            assert (await repository.get_deeptalk_session(int(s_live_due["id"])))["status"] == "completed"

            # Advance 40 minutes -> periodic sweep ends Session 2!
            swept = await sweep_due_deeptalks(bot, now_utc=now_utc + timedelta(minutes=40))
            assert swept == 1
            assert (await repository.get_deeptalk_session(int(s_in_window["id"])))["status"] == "completed"
    finally:
        await close_db()

"""Offline unit tests for the Music Downloader feature."""

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import aiosqlite
import pytest
from aiogram import Bot
from aiogram.fsm.context import FSMContext
from aiogram.types import (
    Audio,
    CallbackQuery,
    Chat,
    ChatMemberLeft,
    ChatMemberMember,
    Document,
    ForumTopicCreated,
    Message,
    User,
    Voice,
)
from bot.database import repository
from bot.database.db import close_db, get_db, init_db
from bot.filters import clear_admin_cache
from bot.handlers.admin import cmd_music_off, cmd_music_on, cmd_set_music_topic
from bot.handlers.listener import capture_message, clear_known_group_cache
from bot.handlers.music import (
    ContainsTelegramMessageLinkFilter,
    cmd_private_music,
    handle_music_callback,
    handle_music_link_message,
)
from bot.handlers.panel import handle_panel_callback
from bot.handlers.private import clear_stranger_cooldowns
from bot.keyboards.music import MusicCB
from bot.keyboards.panel import PanelCB
from bot.services.access import clear_member_cache
from bot.utils.link_parser import ParsedMessageLink, parse_message_link


def _make_group_media_msg(
    *,
    chat_id: int,
    thread_id: int | None,
    message_id: int,
    user_id: int = 500,
    full_name: str = "علی رضایی",
    is_bot: bool = False,
    caption: str | None = None,
    audio: Audio | None = None,
    document: Document | None = None,
    voice: Voice | None = None,
    topic_created_name: str | None = None,
) -> Message:
    msg = MagicMock(spec=Message)
    msg.chat = MagicMock(spec=Chat)
    msg.chat.id = chat_id
    msg.chat.type = "supergroup"
    msg.chat.title = "Prime"
    msg.chat.is_forum = True
    msg.message_id = message_id
    msg.message_thread_id = thread_id
    msg.date = datetime(2026, 11, 26, 12, 0, 0, tzinfo=timezone.utc)

    msg.from_user = MagicMock(spec=User)
    msg.from_user.id = user_id
    msg.from_user.full_name = full_name
    msg.from_user.first_name = full_name
    msg.from_user.username = "ali_user"
    msg.from_user.is_bot = is_bot
    msg.sender_chat = None

    msg.text = None
    msg.caption = caption
    msg.animation = None
    msg.sticker = None
    msg.photo = None
    msg.video = None
    msg.video_note = None
    msg.audio = audio
    msg.document = document
    msg.voice = voice

    if topic_created_name is not None:
        reply = MagicMock(spec=Message)
        reply.message_id = thread_id or 1
        reply.forum_topic_created = MagicMock(spec=ForumTopicCreated)
        reply.forum_topic_created.name = topic_created_name
        reply.text = None
        reply.caption = None
        msg.reply_to_message = reply
    else:
        msg.reply_to_message = None

    return msg


def _make_private_msg(user_id: int, text: str) -> Message:
    msg = MagicMock(spec=Message)
    msg.chat = MagicMock(spec=Chat)
    msg.chat.id = user_id
    msg.chat.type = "private"
    msg.text = text
    msg.caption = None
    msg.from_user = MagicMock(spec=User)
    msg.from_user.id = user_id
    msg.from_user.first_name = "امیر"
    msg.from_user.full_name = "امیر حسینی"
    msg.from_user.username = "amir_h"
    msg.from_user.is_bot = False
    msg.answer = AsyncMock()
    return msg


def _make_callback(user_id: int) -> CallbackQuery:
    cb = MagicMock(spec=CallbackQuery)
    cb.from_user = MagicMock(spec=User)
    cb.from_user.id = user_id
    cb.from_user.first_name = "امیر"
    cb.from_user.full_name = "امیر حسینی"
    cb.answer = AsyncMock()
    msg = MagicMock(spec=Message)
    msg.message_id = 99
    msg.chat = MagicMock(spec=Chat)
    msg.chat.id = user_id
    msg.chat.type = "private"
    msg.edit_text = AsyncMock()
    cb.message = msg
    return cb


def _make_fsm() -> FSMContext:
    fsm = MagicMock(spec=FSMContext)
    fsm.get_state = AsyncMock(return_value=None)
    fsm.clear = AsyncMock()
    fsm.set_state = AsyncMock()
    fsm.get_data = AsyncMock(return_value={})
    fsm.update_data = AsyncMock()
    return fsm


def test_parse_message_link_variants() -> None:
    """parse_message_link handles private, public, query-param, embedded, and invalid links."""
    # Private topic message link
    res = parse_message_link("https://t.me/c/2458921841/42/999")
    assert res == ParsedMessageLink(
        chat_id=-1002458921841,
        username=None,
        thread_id=42,
        message_id=999,
    )

    # Private direct message link without thread
    res2 = parse_message_link("t.me/c/2458921841/999")
    assert res2 == ParsedMessageLink(
        chat_id=-1002458921841,
        username=None,
        thread_id=None,
        message_id=999,
    )

    # Private link with ?thread=42&single
    res3 = parse_message_link("https://t.me/c/2458921841/999?thread=42&single")
    assert res3 == ParsedMessageLink(
        chat_id=-1002458921841,
        username=None,
        thread_id=42,
        message_id=999,
    )

    # Public group topic link & direct link
    assert parse_message_link("https://t.me/PrimeGroup/15/880") == ParsedMessageLink(
        chat_id=None,
        username="primegroup",
        thread_id=15,
        message_id=880,
    )
    assert parse_message_link("https://t.me/PrimeGroup/880?thread=15") == ParsedMessageLink(
        chat_id=None,
        username="primegroup",
        thread_id=15,
        message_id=880,
    )

    # Embedded in Persian prose
    embedded = parse_message_link("این آهنگ رو بفرست بیاد https://t.me/c/112233/7/456 ممنون!")
    assert embedded == ParsedMessageLink(
        chat_id=-100112233,
        username=None,
        thread_id=7,
        message_id=456,
    )

    # Non-message links / invite links / invalid inputs
    assert parse_message_link("https://t.me/+AbCdEfG12345") is None
    assert parse_message_link("https://t.me/joinchat/AAAAAE") is None
    assert parse_message_link("https://t.me/c/12345") is None
    assert parse_message_link("https://t.me/PrimeGroup") is None
    assert parse_message_link("سلام ربات") is None
    assert parse_message_link("") is None


def test_music_callback_data_under_64_bytes() -> None:
    """All MusicCB and new PanelCB actions must pack to <= 64 bytes."""
    worst_chat = -1009999999999999999
    for cb in (
        MusicCB(action="home"),
        MusicCB(action="noop"),
        MusicCB(action="list", page=9999),
        MusicCB(action="dl", arg="999999999", page=9999),
        PanelCB(action="mus", chat_id=worst_chat),
        PanelCB(action="must", chat_id=worst_chat),
        PanelCB(action="musl", chat_id=worst_chat, page=9999),
        PanelCB(action="muss", chat_id=worst_chat, arg="2147483647", page=9999),
        PanelCB(action="musa", chat_id=worst_chat),
    ):
        assert len(cb.pack().encode("utf-8")) <= 64


@pytest.mark.asyncio
async def test_migration_v8_to_v9_and_repository_crud(tmp_path: Path) -> None:
    """Upgrading from user_version=8 creates music_settings & music_tracks and repository CRUD works."""
    db_file = tmp_path / "upgrade_v8_to_v9.db"
    conn = await aiosqlite.connect(str(db_file))
    await conn.executescript("PRAGMA user_version = 8;")
    await conn.commit()
    await conn.close()

    try:
        await init_db(str(db_file))
        db = get_db()
        cur = await db.execute("PRAGMA user_version;")
        ver = (await cur.fetchone())[0]
        assert ver >= 9

        chat_id = -100123456
        cfg = await repository.ensure_music_settings(chat_id)
        assert cfg["enabled"] == 1
        assert cfg["thread_id"] is None

        await repository.set_music_topic(chat_id, 42, auto_enable=True)
        cfg2 = await repository.get_music_settings(chat_id)
        assert cfg2 is not None
        assert cfg2["thread_id"] == 42
        assert cfg2["enabled"] == 1

        tid1 = await repository.insert_music_track(
            chat_id=chat_id,
            thread_id=42,
            message_id=101,
            file_id="FILE_A",
            media_type="audio",
            title="Night Drive",
            performer="Artist X",
            sender_name="آرش",
            created_at="2026-11-26T10:00:00+00:00",
        )
        assert tid1 is not None

        # Duplicate (chat_id, message_id) is ignored
        dup = await repository.insert_music_track(
            chat_id=chat_id,
            thread_id=42,
            message_id=101,
            file_id="FILE_A",
        )
        assert dup is None

        fetched = await repository.get_music_track_by_chat_and_message(chat_id, 101)
        assert fetched is not None
        assert fetched["title"] == "Night Drive"
        assert fetched["download_count"] == 0

        by_msg = await repository.get_music_track_by_message_id_in_chats(101, [chat_id])
        assert by_msg is not None and by_msg["id"] == tid1
        assert await repository.get_music_track_by_message_id_in_chats(101, [-100999]) is None

        await repository.increment_music_download_count(tid1)
        fetched2 = await repository.get_music_track_by_id(tid1)
        assert fetched2 is not None and fetched2["download_count"] == 1
        assert await repository.count_music_tracks(chat_id=chat_id) == 1
        assert len(await repository.list_recent_music_tracks([chat_id])) == 1
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_listener_auto_captures_music_without_caption_and_respects_config(tmp_path: Path) -> None:
    """Listener indexes audio/mp3-document/voice in music topic (even without caption) and respects settings."""
    clear_known_group_cache()
    await init_db(str(tmp_path / "listener_music.db"))

    try:
        chat_id = -100555111
        music_tid = 4
        other_tid = 9
        await repository.upsert_topic(chat_id, music_tid, name="Muzik | موزیک")
        await repository.upsert_topic(chat_id, other_tid, name="The Hub | گفتگوی آزاد")

        audio_obj = MagicMock(spec=Audio)
        audio_obj.file_id = "AUDIO_FILE_1"
        audio_obj.file_unique_id = "UNIQ_1"
        audio_obj.title = "Midnight"
        audio_obj.performer = "Lofi"
        audio_obj.file_name = "midnight.mp3"
        audio_obj.duration = 185
        audio_obj.file_size = 4_000_000
        audio_obj.mime_type = "audio/mpeg"

        # 1. Audio WITHOUT caption in auto-detected Muzik topic -> captured!
        msg1 = _make_group_media_msg(
            chat_id=chat_id,
            thread_id=music_tid,
            message_id=701,
            caption=None,
            audio=audio_obj,
        )
        await capture_message(msg1)
        t1 = await repository.get_music_track_by_chat_and_message(chat_id, 701)
        assert t1 is not None
        assert t1["file_id"] == "AUDIO_FILE_1"
        assert t1["media_type"] == "audio"

        # 2. Audio document (.mp3) in Muzik topic -> captured!
        mp3_doc = MagicMock(spec=Document)
        mp3_doc.file_id = "DOC_MP3_2"
        mp3_doc.file_unique_id = "UNIQ_2"
        mp3_doc.file_name = "track.mp3"
        mp3_doc.file_size = 5_000_000
        mp3_doc.mime_type = "application/octet-stream"
        msg2 = _make_group_media_msg(
            chat_id=chat_id,
            thread_id=music_tid,
            message_id=702,
            document=mp3_doc,
        )
        await capture_message(msg2)
        t2 = await repository.get_music_track_by_chat_and_message(chat_id, 702)
        assert t2 is not None and t2["media_type"] == "document"

        # 3. Voice in Muzik topic -> captured!
        voice_obj = MagicMock(spec=Voice)
        voice_obj.file_id = "VOICE_3"
        voice_obj.file_unique_id = "UNIQ_3"
        voice_obj.duration = 25
        voice_obj.file_size = 50_000
        voice_obj.mime_type = "audio/ogg"
        msg3 = _make_group_media_msg(
            chat_id=chat_id,
            thread_id=music_tid,
            message_id=703,
            voice=voice_obj,
        )
        await capture_message(msg3)
        assert await repository.get_music_track_by_chat_and_message(chat_id, 703) is not None

        # 4. Non-audio PDF document in Muzik topic -> ignored!
        pdf_doc = MagicMock(spec=Document)
        pdf_doc.file_id = "PDF_FILE"
        pdf_doc.file_unique_id = "UNIQ_PDF"
        pdf_doc.file_name = "notes.pdf"
        pdf_doc.file_size = 100_000
        pdf_doc.mime_type = "application/pdf"
        msg4 = _make_group_media_msg(
            chat_id=chat_id,
            thread_id=music_tid,
            message_id=704,
            document=pdf_doc,
        )
        await capture_message(msg4)
        assert await repository.get_music_track_by_chat_and_message(chat_id, 704) is None

        # 5. Audio in non-music topic -> ignored!
        msg5 = _make_group_media_msg(
            chat_id=chat_id,
            thread_id=other_tid,
            message_id=705,
            audio=audio_obj,
        )
        await capture_message(msg5)
        assert await repository.get_music_track_by_chat_and_message(chat_id, 705) is None

        # 6. Explicit configured topic overrides name matching
        await repository.set_music_topic(chat_id, other_tid, auto_enable=True)
        msg6 = _make_group_media_msg(
            chat_id=chat_id,
            thread_id=other_tid,
            message_id=706,
            audio=audio_obj,
        )
        await capture_message(msg6)
        assert await repository.get_music_track_by_chat_and_message(chat_id, 706) is not None

        # 7. Disabled -> skips capture
        await repository.set_music_enabled(chat_id, 0)
        msg7 = _make_group_media_msg(
            chat_id=chat_id,
            thread_id=other_tid,
            message_id=707,
            audio=audio_obj,
        )
        await capture_message(msg7)
        assert await repository.get_music_track_by_chat_and_message(chat_id, 707) is None
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_private_music_link_download_access_and_callbacks(tmp_path: Path) -> None:
    """Authorized group members receive audio without protect_content; strangers and outsiders are blocked."""
    clear_admin_cache()
    clear_member_cache()
    clear_stranger_cooldowns()
    await init_db(str(tmp_path / "music_pv.db"))

    try:
        chat_a = -1002458921841
        chat_b = -1009998887771
        await repository.upsert_known_group(chat_a, "Prime", is_forum=True)
        await repository.upsert_known_group(chat_b, "Other House", is_forum=True)

        track_id = await repository.insert_music_track(
            chat_id=chat_a,
            thread_id=4,
            message_id=555,
            file_id="TG_AUDIO_555",
            media_type="audio",
            title="Track <1>",
            performer="DJ & Co",
            sender_name="پیت",
            created_at="2026-11-26T10:00:00+00:00",
        )
        assert track_id is not None

        await repository.insert_music_track(
            chat_id=chat_b,
            thread_id=2,
            message_id=888,
            file_id="TG_AUDIO_888",
            media_type="audio",
            title="Secret Track",
            sender_name="عضو ب",
            created_at="2026-11-26T10:05:00+00:00",
        )

        bot = AsyncMock(spec=Bot)
        bot.send_audio = AsyncMock()
        bot.send_message = AsyncMock()

        async def _get_member(chat_id: int, user_id: int):
            if user_id == 2001 and chat_id == chat_a:
                m = MagicMock(spec=ChatMemberMember)
                m.status = "member"
                return m
            m = MagicMock(spec=ChatMemberLeft)
            m.status = "left"
            return m

        bot.get_chat_member = AsyncMock(side_effect=_get_member)

        with patch("bot.services.access.get_settings") as mock_acc, patch(
            "bot.handlers.private.get_settings"
        ) as mock_priv:
            mock_acc.return_value.ADMIN_IDS = [100]
            mock_acc.return_value.ALLOWED_CHAT_IDS = []
            mock_priv.return_value.ADMIN_IDS = [100]
            mock_priv.return_value.NOTIFY_ON_UNKNOWN_START = False
            mock_priv.return_value.OWNER_NAMES = "آرش و پیت"

            # 1. Link filter matches message link
            flt = ContainsTelegramMessageLinkFilter()
            assert await flt(_make_private_msg(2001, "https://t.me/c/2458921841/4/555")) is True
            assert await flt(_make_private_msg(2001, "سلام")) is False

            # 2. Authorized member pastes link -> receives audio with protect_content=False
            msg_ok = _make_private_msg(2001, "https://t.me/c/2458921841/4/555")
            await handle_music_link_message(msg_ok, bot)
            bot.send_audio.assert_awaited_once()
            kwargs = bot.send_audio.call_args.kwargs
            assert kwargs["chat_id"] == 2001
            assert kwargs["audio"] == "TG_AUDIO_555"
            assert kwargs["protect_content"] is False
            assert "DJ &amp; Co - Track &lt;1&gt;" in kwargs["caption"]
            assert (await repository.get_music_track_by_id(track_id))["download_count"] == 1

            # 3. Member of group A pasting link from group B -> denied
            msg_other = _make_private_msg(2001, "https://t.me/c/9998887771/2/888")
            await handle_music_link_message(msg_other, bot)
            assert "عضو آن نیستی" in msg_other.answer.call_args.kwargs.get("text", msg_other.answer.call_args.args[0])

            # 4. Missing track -> friendly Persian explanation
            msg_missing = _make_private_msg(2001, "https://t.me/c/2458921841/4/999999")
            await handle_music_link_message(msg_missing, bot)
            assert "پیدا نشد" in msg_missing.answer.call_args.args[0]

            # 5. Stranger pasting link -> routed to stranger handler, NOT sent audio
            bot.send_audio.reset_mock()
            msg_stranger = _make_private_msg(9999, "https://t.me/c/2458921841/4/555")
            await handle_music_link_message(msg_stranger, bot)
            bot.send_audio.assert_not_called()
            msg_stranger.answer.assert_awaited_once()

            # 6. /music command + list + dl callback
            msg_cmd = _make_private_msg(2001, "/music")
            await cmd_private_music(msg_cmd, bot)
            assert "دانلودر موزیک گروه" in msg_cmd.answer.call_args.kwargs["text"]

            cb = _make_callback(2001)
            await handle_music_callback(cb, MusicCB(action="list", page=0), bot)
            assert "آخرین موزیک‌های تاپیک موزیک" in cb.message.edit_text.call_args.kwargs["text"]

            bot.send_audio.reset_mock()
            await handle_music_callback(cb, MusicCB(action="dl", arg=str(track_id), page=0), bot)
            bot.send_audio.assert_awaited_once()
            assert (await repository.get_music_track_by_id(track_id))["download_count"] == 2
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_panel_and_admin_music_commands(tmp_path: Path) -> None:
    """Admin panel mus/must/musl/muss/musa and /set_music_topic, /music_on, /music_off update DB."""
    clear_admin_cache()
    await init_db(str(tmp_path / "panel_music.db"))

    try:
        chat_id = -100333222
        await repository.upsert_known_group(chat_id, "Prime", is_forum=True)
        await repository.upsert_topic(chat_id, 4, name="Muzik | موزیک")

        bot = AsyncMock(spec=Bot)
        bot.send_message = AsyncMock()
        fsm = _make_fsm()
        cb = _make_callback(100)

        with patch("bot.services.access.get_settings") as mock_acc:
            mock_acc.return_value.ADMIN_IDS = [100]
            mock_acc.return_value.ALLOWED_CHAT_IDS = []

            await handle_panel_callback(cb, PanelCB(action="mus", chat_id=chat_id), bot, fsm)
            assert "تنظیمات دانلودر موزیک" in cb.message.edit_text.call_args.kwargs["text"]

            await handle_panel_callback(cb, PanelCB(action="must", chat_id=chat_id), bot, fsm)
            assert (await repository.get_music_settings(chat_id))["enabled"] == 0

            await handle_panel_callback(cb, PanelCB(action="muss", chat_id=chat_id, arg="4", page=0), bot, fsm)
            cfg = await repository.get_music_settings(chat_id)
            assert cfg["enabled"] == 1
            assert cfg["thread_id"] == 4

            await handle_panel_callback(cb, PanelCB(action="musa", chat_id=chat_id), bot, fsm)
            assert (await repository.get_music_settings(chat_id))["thread_id"] is None

        # Group admin commands
        admin_msg = _make_group_media_msg(chat_id=chat_id, thread_id=4, message_id=900, user_id=100)
        admin_msg.text = "/set_music_topic"
        admin_msg.bot = bot
        admin_msg.delete = AsyncMock()

        await cmd_set_music_topic(admin_msg)
        assert (await repository.get_music_settings(chat_id))["thread_id"] == 4

        admin_msg.text = "/music_off"
        await cmd_music_off(admin_msg)
        assert (await repository.get_music_settings(chat_id))["enabled"] == 0

        admin_msg.text = "/music_on"
        await cmd_music_on(admin_msg)
        assert (await repository.get_music_settings(chat_id))["enabled"] == 1
    finally:
        await close_db()

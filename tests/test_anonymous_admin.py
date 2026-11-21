"""Tests for anonymous admin message handling and sender naming."""

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock
import pytest
from aiogram.types import Chat, Message, User
from bot.database.db import close_db, init_db
from bot.database.repository import ensure_group_settings, fetch_unsummarized_messages
from bot.handlers.listener import capture_message


def _create_mock_msg(
    chat_id: int,
    thread_id: int,
    message_id: int,
    text: str,
    is_bot: bool = False,
    sender_chat_id: int | None = None,
    sender_chat_title: str | None = None,
    reply_msg: Message | None = None,
) -> Message:
    """Build mock message with customizable sender_chat and from_user."""
    msg = MagicMock(spec=Message)
    msg.chat = MagicMock(spec=Chat)
    msg.chat.id = chat_id
    msg.chat.type = "supergroup"
    msg.chat.is_forum = True

    msg.message_id = message_id
    msg.is_topic_message = True
    msg.message_thread_id = thread_id
    msg.text = text
    msg.caption = None
    msg.animation = None
    msg.sticker = None
    msg.photo = None
    msg.video = None
    msg.document = None
    msg.voice = None
    msg.audio = None
    msg.video_note = None
    msg.reply_to_message = reply_msg
    msg.date = datetime.now(timezone.utc)

    user = MagicMock(spec=User)
    user.id = 1087968824 if is_bot else 12345
    user.full_name = "GroupAnonymousBot" if is_bot else "کاربر عادی"
    user.is_bot = is_bot
    msg.from_user = user

    if sender_chat_id is not None:
        sender_chat = MagicMock(spec=Chat)
        sender_chat.id = sender_chat_id
        sender_chat.title = sender_chat_title
        msg.sender_chat = sender_chat
    else:
        msg.sender_chat = None

    return msg


@pytest.mark.asyncio
async def test_regular_bot_is_skipped(tmp_path: Path) -> None:
    """A regular bot without sender_chat must be skipped."""
    db_file = tmp_path / "test_bot_skip.db"
    await init_db(str(db_file))

    try:
        chat_id = -100111
        thread_id = 10
        await ensure_group_settings(chat_id)

        msg = _create_mock_msg(
            chat_id=chat_id,
            thread_id=thread_id,
            message_id=1,
            text="پیام ارسالی توسط یک ربات معمولی تلگرام",
            is_bot=True,
            sender_chat_id=None,
        )
        await capture_message(msg)

        msgs = await fetch_unsummarized_messages(chat_id, thread_id)
        assert len(msgs) == 0

    finally:
        await close_db()


@pytest.mark.asyncio
async def test_anonymous_admin_captured_with_persian_name(tmp_path: Path) -> None:
    """Anonymous admin message (is_bot=True, sender_chat.id == chat.id) must be captured as 'ادمین ناشناس'."""
    db_file = tmp_path / "test_anon_admin.db"
    await init_db(str(db_file))

    try:
        chat_id = -100222
        thread_id = 20
        await ensure_group_settings(chat_id)

        # Anonymous admin message
        msg = _create_mock_msg(
            chat_id=chat_id,
            thread_id=thread_id,
            message_id=10,
            text="اطلاعیه مهم مدیریت گروه در خصوص قوانین جدید",
            is_bot=True,
            sender_chat_id=chat_id,  # sender_chat.id == chat.id
            sender_chat_title="گروه برنامه نویسی",
        )
        await capture_message(msg)

        msgs = await fetch_unsummarized_messages(chat_id, thread_id)
        assert len(msgs) == 1
        assert msgs[0]["message_id"] == 10
        assert msgs[0]["sender_name"] == "ادمین ناشناس"
        assert msgs[0]["user_id"] == chat_id

    finally:
        await close_db()


@pytest.mark.asyncio
async def test_linked_channel_message_captured_with_channel_title(tmp_path: Path) -> None:
    """Message from a channel (sender_chat.id != chat.id) uses channel title."""
    db_file = tmp_path / "test_channel_msg.db"
    await init_db(str(db_file))

    try:
        chat_id = -100222
        thread_id = 20
        channel_id = -100333
        await ensure_group_settings(chat_id)

        msg = _create_mock_msg(
            chat_id=chat_id,
            thread_id=thread_id,
            message_id=15,
            text="پست ارسال شده از کانال متصل به گروه",
            is_bot=True,
            sender_chat_id=channel_id,
            sender_chat_title="کانال اخبار هوش مصنوعی",
        )
        await capture_message(msg)

        msgs = await fetch_unsummarized_messages(chat_id, thread_id)
        assert len(msgs) == 1
        assert msgs[0]["message_id"] == 15
        assert msgs[0]["sender_name"] == "کانال اخبار هوش مصنوعی"
        assert msgs[0]["user_id"] == channel_id

    finally:
        await close_db()


@pytest.mark.asyncio
async def test_reply_to_anonymous_admin(tmp_path: Path) -> None:
    """Replying to an anonymous admin message extracts reply_to_name as 'ادمین ناشناس'."""
    db_file = tmp_path / "test_anon_reply.db"
    await init_db(str(db_file))

    try:
        chat_id = -100222
        thread_id = 20
        await ensure_group_settings(chat_id)

        # Mock parent anonymous admin message
        parent_msg = _create_mock_msg(
            chat_id=chat_id,
            thread_id=thread_id,
            message_id=50,
            text="پیام والد از طرف ادمین ناشناس",
            is_bot=True,
            sender_chat_id=chat_id,
        )
        parent_msg.forum_topic_created = None

        # Child message replying to parent
        child_msg = _create_mock_msg(
            chat_id=chat_id,
            thread_id=thread_id,
            message_id=51,
            text="پاسخ من به پیام ادمین ناشناس",
            is_bot=False,
            reply_msg=parent_msg,
        )
        await capture_message(child_msg)

        msgs = await fetch_unsummarized_messages(chat_id, thread_id)
        assert len(msgs) == 1
        assert msgs[0]["message_id"] == 51
        assert msgs[0]["reply_to_message_id"] == 50
        assert msgs[0]["reply_to_name"] == "ادمین ناشناس"

    finally:
        await close_db()

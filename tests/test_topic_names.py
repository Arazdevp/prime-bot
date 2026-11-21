"""Tests for learning topic names from reply_to_message.forum_topic_created.name."""

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock
import pytest
from aiogram.types import Chat, ForumTopicCreated, Message, User
from bot.database.db import close_db, init_db
from bot.database.repository import ensure_group_settings, get_topic_name
from bot.handlers.listener import capture_message


def _create_mock_message(
    chat_id: int,
    thread_id: int,
    text: str,
    reply_topic_name: str | None = None,
    reply_message_id: int = 100,
    is_bot: bool = False,
) -> Message:
    """Helper to build a mock Message object with optional reply_to_message forum_topic_created."""
    msg = MagicMock(spec=Message)
    msg.chat = MagicMock(spec=Chat)
    msg.chat.id = chat_id
    msg.chat.type = "supergroup"
    msg.chat.is_forum = True

    msg.message_id = 999
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
    msg.sender_chat = None
    msg.date = datetime.now(timezone.utc)

    user = MagicMock(spec=User)
    user.id = 12345
    user.full_name = "تست کننده"
    user.is_bot = is_bot
    msg.from_user = user

    if reply_topic_name is not None:
        reply_msg = MagicMock(spec=Message)
        reply_msg.message_id = reply_message_id
        reply_msg.from_user = user
        reply_msg.sender_chat = None
        reply_msg.text = None
        reply_msg.caption = None

        topic_created = MagicMock(spec=ForumTopicCreated)
        topic_created.name = reply_topic_name
        reply_msg.forum_topic_created = topic_created

        msg.reply_to_message = reply_msg
    else:
        msg.reply_to_message = None

    return msg


@pytest.mark.asyncio
async def test_topic_name_upserted_from_reply_to_message(tmp_path: Path) -> None:
    """Test that topic name is learned from reply_to_message.forum_topic_created.name."""
    db_file = tmp_path / "test_topic_name.db"
    await init_db(str(db_file))

    try:
        chat_id = -100123456
        thread_id = 42
        await ensure_group_settings(chat_id)

        # Initially, topic name does not exist
        assert await get_topic_name(chat_id, thread_id) is None

        # 1. Incoming message carries reply_to_message with topic name
        msg = _create_mock_message(
            chat_id=chat_id,
            thread_id=thread_id,
            text="این یک پیام معتبر در تاپیک است.",
            reply_topic_name="مباحث پایتون",
            reply_message_id=thread_id,
        )
        await capture_message(msg)

        # Verify topic name is now stored
        assert await get_topic_name(chat_id, thread_id) == "مباحث پایتون"

        # 2. Incoming message with changed topic name
        msg2 = _create_mock_message(
            chat_id=chat_id,
            thread_id=thread_id,
            text="پیام دوم بعد از تغییر نام تاپیک.",
            reply_topic_name="مباحث پیشرفته پایتون و هوش مصنوعی",
            reply_message_id=thread_id,
        )
        await capture_message(msg2)

        # Verify topic name was updated
        assert await get_topic_name(chat_id, thread_id) == "مباحث پیشرفته پایتون و هوش مصنوعی"

    finally:
        await close_db()


@pytest.mark.asyncio
async def test_topic_name_upserted_even_on_noisy_message(tmp_path: Path) -> None:
    """Topic name must be captured before noise check, even if the message itself is noise."""
    db_file = tmp_path / "test_topic_name_noise.db"
    await init_db(str(db_file))

    try:
        chat_id = -100123456
        thread_id = 77
        await ensure_group_settings(chat_id)

        assert await get_topic_name(chat_id, thread_id) is None

        # Message is pure noise ("سلام"), which will be dropped by noise filter
        msg = _create_mock_message(
            chat_id=chat_id,
            thread_id=thread_id,
            text="سلام",
            reply_topic_name="تاپیک خوش‌آمدگویی",
            reply_message_id=thread_id,
        )
        await capture_message(msg)

        # Topic name must still have been learned!
        assert await get_topic_name(chat_id, thread_id) == "تاپیک خوش‌آمدگویی"

    finally:
        await close_db()

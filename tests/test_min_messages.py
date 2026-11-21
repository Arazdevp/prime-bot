"""Tests for min_messages configuration and the /set_min admin command."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
import pytest
from aiogram.types import Chat, Message, User
from bot.config import get_settings
from bot.database.db import close_db, init_db
from bot.database.repository import ensure_group_settings, set_group_min_messages
from bot.handlers.admin import cmd_set_min


def _create_admin_message(chat_id: int, text: str) -> tuple[Message, AsyncMock]:
    """Helper to create a mock Message and mock bot for admin commands."""
    msg = MagicMock(spec=Message)
    msg.chat = MagicMock(spec=Chat)
    msg.chat.id = chat_id
    msg.chat.type = "supergroup"
    msg.chat.is_forum = True

    msg.message_id = 500
    msg.is_topic_message = True
    msg.message_thread_id = 1
    msg.text = text
    msg.delete = AsyncMock()

    user = MagicMock(spec=User)
    user.id = 9999
    user.full_name = "مدیر"
    msg.from_user = user

    bot = AsyncMock()
    bot.send_message = AsyncMock()
    msg.bot = bot

    return msg, bot.send_message


@pytest.mark.asyncio
async def test_ensure_group_settings_defaults_to_config(tmp_path: Path) -> None:
    """ensure_group_settings must default to get_settings().MIN_MESSAGES_PER_TOPIC."""
    db_file = tmp_path / "test_min_cfg.db"
    await init_db(str(db_file))

    try:
        chat_id = -100444
        cfg = get_settings()

        settings = await ensure_group_settings(chat_id)
        assert settings["min_messages"] == cfg.MIN_MESSAGES_PER_TOPIC

        # Calling again retrieves existing
        settings2 = await ensure_group_settings(chat_id)
        assert settings2["min_messages"] == cfg.MIN_MESSAGES_PER_TOPIC

        # Test set_group_min_messages
        await set_group_min_messages(chat_id, 15)
        settings3 = await ensure_group_settings(chat_id)
        assert settings3["min_messages"] == 15

    finally:
        await close_db()


@pytest.mark.asyncio
async def test_cmd_set_min_valid_and_invalid(tmp_path: Path) -> None:
    """Test /set_min validation and replies."""
    db_file = tmp_path / "test_cmd_set_min.db"
    await init_db(str(db_file))

    try:
        chat_id = -100555
        await ensure_group_settings(chat_id)

        # 1. Valid command: /set_min 12
        msg, send_mock = _create_admin_message(chat_id, "/set_min 12")
        await cmd_set_min(msg)

        send_mock.assert_called_once()
        sent_text = send_mock.call_args[1]["text"]
        assert "12" in sent_text
        assert "حداقل تعداد پیام برای خلاصه‌سازی" in sent_text

        # Verify DB updated
        gs = await ensure_group_settings(chat_id)
        assert gs["min_messages"] == 12

        # 2. Invalid command: Missing argument
        send_mock.reset_mock()
        msg_missing, send_mock = _create_admin_message(chat_id, "/set_min")
        await cmd_set_min(msg_missing)
        sent_text = send_mock.call_args[1]["text"]
        assert "بین ۱ تا ۱۰۰" in sent_text

        # 3. Invalid command: Out of bounds (> 100)
        send_mock.reset_mock()
        msg_out_high, send_mock = _create_admin_message(chat_id, "/set_min 150")
        await cmd_set_min(msg_out_high)
        sent_text = send_mock.call_args[1]["text"]
        assert "نامعتبر" in sent_text

        # 4. Invalid command: Out of bounds (< 1)
        send_mock.reset_mock()
        msg_out_low, send_mock = _create_admin_message(chat_id, "/set_min 0")
        await cmd_set_min(msg_out_low)
        sent_text = send_mock.call_args[1]["text"]
        assert "نامعتبر" in sent_text

        # 5. Invalid command: Non-numeric
        send_mock.reset_mock()
        msg_nan, send_mock = _create_admin_message(chat_id, "/set_min test")
        await cmd_set_min(msg_nan)
        sent_text = send_mock.call_args[1]["text"]
        assert "نامعتبر" in sent_text

        # Verify DB was NOT corrupted by invalid attempts
        gs = await ensure_group_settings(chat_id)
        assert gs["min_messages"] == 12

    finally:
        await close_db()

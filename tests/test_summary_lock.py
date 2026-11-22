"""Unit tests for get_summary_lock concurrency protection across manual and nightly summaries."""

from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from aiogram import Bot
from aiogram.types import Chat, Message, User
from bot.handlers.admin import cmd_summarize_all, cmd_summarize_now
from bot.scheduler.jobs import get_summary_lock, nightly_summary_job


def _build_admin_cmd_msg(chat_id: int, text: str) -> tuple[Message, AsyncMock]:
    msg = MagicMock(spec=Message)
    msg.chat = MagicMock(spec=Chat)
    msg.chat.id = chat_id
    msg.chat.type = "supergroup"
    msg.chat.is_forum = True

    msg.message_id = 900
    msg.is_topic_message = True
    msg.message_thread_id = 1
    msg.text = text
    msg.delete = AsyncMock()

    user = MagicMock(spec=User)
    user.id = 8888
    msg.from_user = user

    bot = AsyncMock(spec=Bot)
    bot.send_message = AsyncMock()
    msg.bot = bot

    return msg, bot.send_message


@pytest.mark.asyncio
async def test_summarize_now_rejected_when_locked():
    """Verify /summarize_now rejects execution if another summary is already running."""
    lock = get_summary_lock()
    chat_id = -100555
    msg, send_mock = _build_admin_cmd_msg(chat_id, "/summarize_now")

    async with lock:
        with patch("bot.handlers.admin.run_summary_for_topic") as mock_run:
            await cmd_summarize_now(msg)
            mock_run.assert_not_called()

            send_mock.assert_called_once()
            sent_text = send_mock.call_args.kwargs["text"]
            assert "اجرای دیگری در حال انجام است" in sent_text


@pytest.mark.asyncio
async def test_summarize_all_rejected_when_locked():
    """Verify /summarize_all rejects execution if another summary is already running."""
    lock = get_summary_lock()
    chat_id = -100666
    msg, send_mock = _build_admin_cmd_msg(chat_id, "/summarize_all")

    async with lock:
        with patch("bot.handlers.admin.run_summary_for_group") as mock_run:
            await cmd_summarize_all(msg)
            mock_run.assert_not_called()

            send_mock.assert_called_once()
            sent_text = send_mock.call_args.kwargs["text"]
            assert "اجرای دیگری در حال انجام است" in sent_text


@pytest.mark.asyncio
async def test_nightly_job_skipped_when_manual_summary_holds_lock():
    """Verify nightly_summary_job skips when manual summary holds the lock."""
    lock = get_summary_lock()
    bot = AsyncMock(spec=Bot)

    async with lock:
        with patch("bot.scheduler.jobs.repository.list_enabled_groups") as mock_list:
            await nightly_summary_job(bot)
            mock_list.assert_not_called()


@pytest.mark.asyncio
async def test_summarize_now_acquires_and_releases_lock():
    """Verify /summarize_now holds the lock during execution and releases it afterwards."""
    lock = get_summary_lock()
    assert not lock.locked()

    chat_id = -100777
    msg, send_mock = _build_admin_cmd_msg(chat_id, "/summarize_now")

    lock_state_during_run = []

    async def fake_run_topic(*args, **kwargs):
        lock_state_during_run.append(lock.locked())
        return True, "خلاصه با موفقیت ارسال شد."

    with patch("bot.handlers.admin.run_summary_for_topic", side_effect=fake_run_topic):
        await cmd_summarize_now(msg)

    assert lock_state_during_run == [True]
    assert not lock.locked()

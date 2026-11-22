"""Unit tests for partial delivery persistence, resumption, and duplicate post prevention."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch
import pytest
from aiogram import Bot
from aiogram.exceptions import TelegramBadRequest
from bot.database.db import close_db, init_db
from bot.database import repository
from bot.scheduler.jobs import run_summary_for_topic
from bot.services.llm_client import BaseLLMClient
from bot.services.publisher import send_summary_parts


@pytest.mark.asyncio
async def test_summaries_table_crud(tmp_path: Path):
    """Verify repository functions for summaries table."""
    db_file = tmp_path / "test_summaries_crud.db"
    await init_db(str(db_file))

    try:
        chat_id = -100999
        thread_id = 42
        day = "2026-11-22"
        parts = ["بخش ۱ خلاصه", "بخش ۲ خلاصه"]

        # Initial check
        assert await repository.get_summary_record(chat_id, thread_id, day) is None

        # Save record
        await repository.save_summary_record(chat_id, thread_id, day, parts, sent_count=0)
        rec = await repository.get_summary_record(chat_id, thread_id, day)
        assert rec is not None
        assert rec["chat_id"] == chat_id
        assert rec["thread_id"] == thread_id
        assert rec["day"] == day
        assert json.loads(rec["parts_json"]) == parts
        assert rec["sent_count"] == 0

        # Update sent count
        await repository.update_summary_sent_count(chat_id, thread_id, day, 1)
        rec_updated = await repository.get_summary_record(chat_id, thread_id, day)
        assert rec_updated["sent_count"] == 1

        # Delete record
        await repository.delete_summary(chat_id, thread_id, day)
        assert await repository.get_summary_record(chat_id, thread_id, day) is None

    finally:
        await close_db()


@pytest.mark.asyncio
async def test_send_summary_parts_start_index_and_callback():
    """Verify send_summary_parts resumes from start_index and calls progress callback."""
    bot = AsyncMock(spec=Bot)
    parts = ["Part 1", "Part 2", "Part 3"]

    callback_counts = []

    async def _on_sent(sent_count: int):
        callback_counts.append(sent_count)

    # Resume from start_index = 1 (skipping Part 1)
    all_delivered, final_count = await send_summary_parts(
        bot=bot,
        chat_id=-1001,
        thread_id=1,
        parts=parts,
        start_index=1,
        on_part_sent=_on_sent,
    )

    assert all_delivered is True
    assert final_count == 3
    # bot.send_message called twice (for Part 2 and Part 3)
    assert bot.send_message.call_count == 2
    # callback invoked for 2 and 3
    assert callback_counts == [2, 3]


@pytest.mark.asyncio
async def test_partial_delivery_failure_resumption_and_no_duplicates(tmp_path: Path):
    """Verify that failure after part k preserves state, next run resumes from part k+1, and deletes source messages only when complete."""
    db_file = tmp_path / "test_partial_delivery.db"
    await init_db(str(db_file))

    try:
        chat_id = -100777
        thread_id = 7
        day = "2026-11-21"

        await repository.ensure_group_settings(chat_id, default_min_messages=2)
        await repository.upsert_topic(chat_id, thread_id, "تاپیک تست", is_active=1)

        # Insert 2 messages
        for i in (1, 2):
            await repository.insert_message(
                chat_id=chat_id,
                thread_id=thread_id,
                message_id=i,
                user_id=10,
                sender_name=f"User {i}",
                text=f"پیام شماره {i}",
                media_type=None,
                reply_to_message_id=None,
                reply_to_name=None,
                reply_to_snippet=None,
                created_at="2026-11-21T10:00:00+00:00",
            )

        mock_llm = AsyncMock(spec=BaseLLMClient)
        mock_llm.generate.return_value = "خلاصه مباحث تاپیک"

        # Mock smart_split to return 3 parts
        fake_parts = ["Part 1 content", "Part 2 content", "Part 3 content"]

        bot = AsyncMock(spec=Bot)

        # 1. Attempt 1: Part 1 succeeds, Part 2 fails with fatal Telegram error
        async def _send_side_effect(**kwargs):
            text = kwargs.get("text", "")
            if "Part 2" in text:
                raise TelegramBadRequest(method="sendMessage", message="Chat not accessible")
            return AsyncMock()

        bot.send_message.side_effect = _send_side_effect

        with patch("bot.scheduler.jobs.smart_split", return_value=fake_parts):
            success1, details1 = await run_summary_for_topic(
                bot=bot,
                chat_id=chat_id,
                thread_id=thread_id,
                mark_summarized=True,
                is_nightly=False,
                llm_client=mock_llm,
            )

        assert success1 is False

        # Verify summary record in DB has sent_count = 1
        record1 = await repository.get_summary_record(chat_id, thread_id, day)
        assert record1 is not None
        assert record1["sent_count"] == 1
        assert json.loads(record1["parts_json"]) == fake_parts

        # Source messages MUST NOT be deleted yet!
        msgs_remaining = await repository.fetch_unsummarized_messages(chat_id, thread_id)
        assert len(msgs_remaining) == 2

        # 2. Attempt 2: All parts succeed now! (keep mode to test duplicate trigger next)
        bot.send_message.reset_mock()
        bot.send_message.side_effect = None

        with patch("bot.scheduler.jobs.smart_split", return_value=fake_parts):
            success2, details2 = await run_summary_for_topic(
                bot=bot,
                chat_id=chat_id,
                thread_id=thread_id,
                mark_summarized=False,
                is_nightly=False,
                llm_client=mock_llm,
            )

        assert success2 is True
        # Part 1 was already sent, so bot.send_message should only be called for Part 2 and Part 3 (2 calls)
        assert bot.send_message.call_count == 2
        calls = [c.kwargs.get("text") for c in bot.send_message.call_args_list]
        assert calls == ["Part 2 content", "Part 3 content"]

        # Now all parts are sent: sent_count in DB should be 3
        record2 = await repository.get_summary_record(chat_id, thread_id, day)
        assert record2["sent_count"] == 3

        # 3. Attempt 3: Another run for the same day (duplicate trigger)
        bot.send_message.reset_mock()
        with patch("bot.scheduler.jobs.smart_split", return_value=fake_parts):
            success3, details3 = await run_summary_for_topic(
                bot=bot,
                chat_id=chat_id,
                thread_id=thread_id,
                mark_summarized=True,
                is_nightly=False,
                llm_client=mock_llm,
            )

        assert success3 is True
        # No duplicate messages sent to Telegram!
        assert bot.send_message.call_count == 0

        # Source messages MUST NOW BE DELETED
        msgs_after_complete = await repository.fetch_unsummarized_messages(chat_id, thread_id)
        assert len(msgs_after_complete) == 0

    finally:
        await close_db()

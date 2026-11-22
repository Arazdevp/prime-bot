"""Unit tests verifying data loss prevention on below-threshold messages."""

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock
import pytest
from aiogram import Bot
from bot.database.db import close_db, init_db
from bot.database.repository import (
    ensure_group_settings,
    fetch_unsummarized_messages,
    insert_message,
    set_group_min_messages,
    upsert_topic,
)
from bot.scheduler.jobs import run_summary_for_topic


@pytest.mark.asyncio
async def test_manual_run_never_deletes_below_threshold_messages(tmp_path: Path):
    """Verify manual run (is_nightly=False) preserves unsummarized messages when below threshold."""
    db_file = tmp_path / "test_data_loss.db"
    await init_db(str(db_file))

    try:
        chat_id = -100888
        thread_id = 10

        await ensure_group_settings(chat_id, 5)
        await set_group_min_messages(chat_id, 5)
        await upsert_topic(chat_id, thread_id, "تاپیک تست", is_active=1)

        # Insert 2 messages (< min 5)
        now_str = datetime.now(timezone.utc).isoformat()
        await insert_message(
            chat_id=chat_id,
            thread_id=thread_id,
            message_id=101,
            user_id=1,
            sender_name="کاربر ۱",
            text="پیام ۱",
            media_type=None,
            reply_to_message_id=None,
            reply_to_name=None,
            reply_to_snippet=None,
            created_at=now_str,
        )
        await insert_message(
            chat_id=chat_id,
            thread_id=thread_id,
            message_id=102,
            user_id=2,
            sender_name="کاربر ۲",
            text="پیام ۲",
            media_type=None,
            reply_to_message_id=None,
            reply_to_name=None,
            reply_to_snippet=None,
            created_at=now_str,
        )

        bot = AsyncMock(spec=Bot)

        # 1. Manual run: is_nightly=False, mark_summarized=True
        success, details = await run_summary_for_topic(
            bot=bot,
            chat_id=chat_id,
            thread_id=thread_id,
            mark_summarized=True,
            is_nightly=False,
        )

        assert success is False
        assert "کمتر از حداقل حد نصاب" in details

        # Verify messages are STILL in DB
        remaining = await fetch_unsummarized_messages(chat_id, thread_id)
        assert len(remaining) == 2

        # 2. Nightly run: is_nightly=True, mark_summarized=True
        success_nightly, details_nightly = await run_summary_for_topic(
            bot=bot,
            chat_id=chat_id,
            thread_id=thread_id,
            mark_summarized=True,
            is_nightly=True,
        )

        assert success_nightly is False
        assert "کمتر از حداقل حد نصاب" in details_nightly

        # Verify messages ARE cleaned up on nightly run
        remaining_nightly = await fetch_unsummarized_messages(chat_id, thread_id)
        assert len(remaining_nightly) == 0

    finally:
        await close_db()

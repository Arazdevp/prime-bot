"""Unit tests verifying repository query chunking by 500 parameters."""

from pathlib import Path
import pytest
from bot.database.db import close_db, init_db
from bot.database.repository import (
    delete_messages_by_ids,
    fetch_unsummarized_messages,
    insert_message,
    mark_messages_summarized,
)


@pytest.mark.asyncio
async def test_chunking_mark_and_delete_over_1000_messages(tmp_path: Path):
    """Verify mark_messages_summarized and delete_messages_by_ids work with >1000 items in chunks of 500."""
    db_file = tmp_path / "test_chunking.db"
    await init_db(str(db_file))

    try:
        chat_id = -100555
        thread_id = 1
        total_messages = 1250

        # Insert 1,250 messages
        for i in range(1, total_messages + 1):
            await insert_message(
                chat_id=chat_id,
                thread_id=thread_id,
                message_id=i,
                user_id=10,
                sender_name="User",
                text=f"Message {i}",
                media_type=None,
                reply_to_message_id=None,
                reply_to_name=None,
                reply_to_snippet=None,
                created_at="2026-11-20T10:00:00+00:00",
            )

        messages = await fetch_unsummarized_messages(chat_id, thread_id)
        assert len(messages) == total_messages
        all_db_ids = [m["id"] for m in messages]

        # 1. Mark first 750 as summarized (should be chunked: 500 + 250)
        first_batch = all_db_ids[:750]
        await mark_messages_summarized(first_batch)

        remaining_unsummarized = await fetch_unsummarized_messages(chat_id, thread_id)
        assert len(remaining_unsummarized) == 500

        # 2. Delete all 1,250 messages by primary IDs (should be chunked: 500 + 500 + 250)
        deleted_count = await delete_messages_by_ids(all_db_ids)
        assert deleted_count == total_messages

        # Verify database is now empty
        messages_after_delete = await fetch_unsummarized_messages(chat_id, thread_id)
        assert len(messages_after_delete) == 0

        # 3. Test empty list handling
        assert await delete_messages_by_ids([]) == 0
        await mark_messages_summarized([])

    finally:
        await close_db()

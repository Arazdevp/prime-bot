"""Tests for database repository message deletion and cleanup."""

import pytest
from pathlib import Path
from bot.database.db import close_db, init_db
from bot.database.repository import (
    delete_messages_by_ids,
    fetch_unsummarized_messages,
    insert_message,
)


@pytest.mark.asyncio
async def test_delete_messages_by_ids(tmp_path: Path) -> None:
    db_file = tmp_path / "test_cleanup.db"
    await init_db(str(db_file))

    try:
        chat_id = -1002458921841
        thread_id = 42

        # Insert 3 messages
        await insert_message(
            chat_id=chat_id,
            thread_id=thread_id,
            message_id=1,
            user_id=10,
            sender_name="کاربر ۱",
            text="پیام اول",
            media_type=None,
            reply_to_message_id=None,
            reply_to_name=None,
            reply_to_snippet=None,
            created_at="2026-11-21T10:00:00+00:00",
        )
        await insert_message(
            chat_id=chat_id,
            thread_id=thread_id,
            message_id=2,
            user_id=11,
            sender_name="کاربر ۲",
            text="پیام دوم",
            media_type=None,
            reply_to_message_id=None,
            reply_to_name=None,
            reply_to_snippet=None,
            created_at="2026-11-21T10:01:00+00:00",
        )
        await insert_message(
            chat_id=chat_id,
            thread_id=thread_id,
            message_id=3,
            user_id=12,
            sender_name="کاربر ۳",
            text="پیام سوم",
            media_type=None,
            reply_to_message_id=None,
            reply_to_name=None,
            reply_to_snippet=None,
            created_at="2026-11-21T10:02:00+00:00",
        )

        initial_messages = await fetch_unsummarized_messages(chat_id, thread_id)
        assert len(initial_messages) == 3

        # Delete the first 2 messages by primary ID
        ids_to_delete = [initial_messages[0]["id"], initial_messages[1]["id"]]
        deleted_count = await delete_messages_by_ids(ids_to_delete)
        assert deleted_count == 2

        # Remaining messages should be exactly 1
        remaining_messages = await fetch_unsummarized_messages(chat_id, thread_id)
        assert len(remaining_messages) == 1
        assert remaining_messages[0]["message_id"] == 3

        # Deleting empty list returns 0
        assert await delete_messages_by_ids([]) == 0

    finally:
        await close_db()

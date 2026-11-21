"""Verification script for Phase 1."""

import asyncio
import os
import sys
import tempfile
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bot.config import Settings, get_settings
from bot.database.db import init_db, get_db, close_db
from bot.database import repository


async def verify_phase1() -> None:
    print("[1/4] Verifying Config parsing and validation...")
    os.environ["BOT_TOKEN"] = "123456789:ABC_TEST_TOKEN"
    os.environ["AI_PROVIDER"] = "gemini"
    os.environ["GEMINI_API_KEY"] = "AIza_TEST_KEY"
    os.environ["ADMIN_IDS"] = "11111, 22222,33333"
    os.environ["ALLOWED_CHAT_IDS"] = "-100123456789"
    os.environ["PROXY_URL"] = ""

    s = Settings()
    assert s.BOT_TOKEN == "123456789:ABC_TEST_TOKEN"
    assert s.ADMIN_IDS == [11111, 22222, 33333]
    assert s.ALLOWED_CHAT_IDS == [-100123456789]
    assert s.PROXY_URL is None
    assert s.AI_PROVIDER == "gemini"
    assert s.GEMINI_API_KEY == "AIza_TEST_KEY"
    print(" Config validation passed.")

    print("[2/4] Initializing temporary database...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_db_path = os.path.join(tmp_dir, "test.db")
        conn = await init_db(test_db_path)

        # Verify PRAGMAs
        async with conn.execute("PRAGMA journal_mode;") as cur:
            row = await cur.fetchone()
            journal_mode = row[0].upper()
            assert journal_mode == "WAL", f"Expected WAL, got {journal_mode}"

        async with conn.execute("PRAGMA foreign_keys;") as cur:
            row = await cur.fetchone()
            fk = row[0]
            assert fk == 1, f"Expected foreign_keys=1, got {fk}"

        print(" PRAGMA WAL & foreign keys verified.")

        print("[3/4] Checking tables and indexes...")
        async with conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
        ) as cur:
            tables = [r[0] for r in await cur.fetchall()]
        print(f"  Tables found: {tables}")
        assert "messages" in tables, "messages table missing"
        assert "topics" in tables, "topics table missing"
        assert "group_settings" in tables, "group_settings table missing"

        async with conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' ORDER BY name;"
        ) as cur:
            indexes = [r[0] for r in await cur.fetchall()]
        print(f"  Indexes found: {indexes}")
        assert "idx_messages_chat_thread_sum_created" in indexes
        assert "idx_messages_created_at" in indexes

        print("[4/4] Testing repository operations...")
        chat_id = -1001999999
        # Group settings
        gs = await repository.ensure_group_settings(chat_id)
        assert gs["mode"] == "blacklist"
        assert gs["enabled"] == 1

        # Test whitelist vs blacklist monitoring
        # In blacklist mode, topic 100 is monitored by default
        assert await repository.is_topic_monitored(chat_id, 100) is True
        # Blacklist topic 100
        await repository.set_topic_active_status(chat_id, 100, is_active=0, name="Testing Topic")
        assert await repository.is_topic_monitored(chat_id, 100) is False

        # Switch to whitelist mode
        await repository.set_group_mode(chat_id, "whitelist")
        assert await repository.is_topic_monitored(chat_id, 200) is False
        await repository.set_topic_active_status(chat_id, 200, is_active=1, name="Whitelisted Topic")
        assert await repository.is_topic_monitored(chat_id, 200) is True

        # Insert message & duplicate ignore
        inserted = await repository.insert_message(
            chat_id=chat_id,
            thread_id=200,
            message_id=1,
            user_id=123,
            sender_name="Alice",
            text="Hello world!",
            media_type=None,
            reply_to_message_id=None,
            reply_to_name=None,
            reply_to_snippet=None,
            created_at="2026-11-21T10:00:00Z",
        )
        assert inserted is True
        # Duplicate insert
        inserted_dup = await repository.insert_message(
            chat_id=chat_id,
            thread_id=200,
            message_id=1,
            user_id=123,
            sender_name="Alice",
            text="Hello world duplicate!",
            media_type=None,
            reply_to_message_id=None,
            reply_to_name=None,
            reply_to_snippet=None,
            created_at="2026-11-21T10:00:00Z",
        )
        assert inserted_dup is False

        # Fetch unsummarized messages
        msgs = await repository.fetch_unsummarized_messages(chat_id, 200)
        assert len(msgs) == 1
        msg_db_id = msgs[0]["id"]

        # Mark summarized
        await repository.mark_messages_summarized([msg_db_id])
        msgs_after = await repository.fetch_unsummarized_messages(chat_id, 200)
        assert len(msgs_after) == 0

        await close_db()
        print(" Repository operations tested and passed.")

    print("\n>>> ALL PHASE 1 CHECKS PASSED SUCCESSFULLY! <<<")


if __name__ == "__main__":
    asyncio.run(verify_phase1())

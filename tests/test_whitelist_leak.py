"""Tests for whitelist leak fix, nullable is_active, and PRAGMA user_version migration."""

import pytest
from pathlib import Path
import aiosqlite
from bot.database.db import close_db, init_db, get_db
from bot.database.repository import (
    ensure_group_settings,
    is_topic_monitored,
    list_topics_with_counts_today,
    set_group_mode,
    set_topic_active_status,
    upsert_topic,
)


@pytest.mark.asyncio
async def test_migration_recreates_table_preserves_data_and_allows_null(tmp_path: Path) -> None:
    """Test that PRAGMA user_version migration recreates legacy table with nullable is_active."""
    db_file = tmp_path / "legacy_test.db"

    # 1. Create a legacy database where topics.is_active is NOT NULL DEFAULT 1 and user_version = 0
    conn = await aiosqlite.connect(str(db_file))
    await conn.execute("""
        CREATE TABLE topics (
            chat_id INTEGER NOT NULL,
            thread_id INTEGER NOT NULL,
            name TEXT,
            is_active INTEGER NOT NULL DEFAULT 1,
            updated_at TEXT,
            PRIMARY KEY(chat_id, thread_id)
        );
    """)
    await conn.execute(
        "INSERT INTO topics (chat_id, thread_id, name, is_active, updated_at) VALUES (?, ?, ?, ?, ?);",
        (-1001, 10, "Existing Topic", 1, "2026-11-20T00:00:00+00:00"),
    )
    await conn.execute(
        "INSERT INTO topics (chat_id, thread_id, name, is_active, updated_at) VALUES (?, ?, ?, ?, ?);",
        (-1001, 20, "Disabled Topic", 0, "2026-11-20T00:00:00+00:00"),
    )
    await conn.execute("PRAGMA user_version = 0;")
    await conn.commit()
    await conn.close()

    # 2. Run init_db which should apply _apply_migrations
    try:
        await init_db(str(db_file))
        db = get_db()

        # Check PRAGMA user_version
        cursor = await db.execute("PRAGMA user_version;")
        row = await cursor.fetchone()
        assert row[0] == 1

        # Check existing data was preserved
        cursor = await db.execute("SELECT thread_id, name, is_active FROM topics ORDER BY thread_id ASC;")
        rows = await cursor.fetchall()
        assert len(rows) == 2
        assert rows[0]["thread_id"] == 10
        assert rows[0]["name"] == "Existing Topic"
        assert rows[0]["is_active"] == 1

        assert rows[1]["thread_id"] == 20
        assert rows[1]["name"] == "Disabled Topic"
        assert rows[1]["is_active"] == 0

        # Check that nullable insertion works (is_active = NULL)
        await upsert_topic(-1001, 30, name="New Topic Without Status")
        cursor = await db.execute("SELECT thread_id, name, is_active FROM topics WHERE thread_id = 30;")
        new_row = await cursor.fetchone()
        assert new_row["is_active"] is None
        assert new_row["name"] == "New Topic Without Status"

    finally:
        await close_db()


@pytest.mark.asyncio
async def test_whitelist_mode_topic_creation_rename_and_general_does_not_whitelist(tmp_path: Path) -> None:
    """In whitelist mode, a topic-created event, rename, or first General message must NOT make topic monitored."""
    db_file = tmp_path / "whitelist_test.db"
    await init_db(str(db_file))

    try:
        chat_id = -100999
        await ensure_group_settings(chat_id)
        await set_group_mode(chat_id, "whitelist")

        # 1. Forum topic created event (upsert with name only, is_active not passed)
        await upsert_topic(chat_id, 100, name="Python Discussions")
        # In whitelist mode, it must NOT be monitored
        assert not await is_topic_monitored(chat_id, 100)

        db = get_db()
        cursor = await db.execute("SELECT is_active FROM topics WHERE chat_id = ? AND thread_id = ?;", (chat_id, 100))
        row = await cursor.fetchone()
        assert row["is_active"] is None

        # 2. Forum topic rename event (upsert with new name only)
        await upsert_topic(chat_id, 100, name="Python & AI Discussions")
        # Still must NOT be monitored
        assert not await is_topic_monitored(chat_id, 100)

        cursor = await db.execute("SELECT is_active, name FROM topics WHERE chat_id = ? AND thread_id = ?;", (chat_id, 100))
        row = await cursor.fetchone()
        assert row["is_active"] is None
        assert row["name"] == "Python & AI Discussions"

        # 3. First General message registers "عمومی" with is_active not passed
        await upsert_topic(chat_id, 1, name="عمومی")
        # General must NOT be monitored in whitelist mode
        assert not await is_topic_monitored(chat_id, 1)

        cursor = await db.execute("SELECT is_active FROM topics WHERE chat_id = ? AND thread_id = 1;", (chat_id,))
        row = await cursor.fetchone()
        assert row["is_active"] is None

        # 4. Explicit admin activation (/add_topic)
        await set_topic_active_status(chat_id, 100, is_active=1)
        assert await is_topic_monitored(chat_id, 100)

        # 5. Subsequent rename must preserve the admin's is_active=1
        await upsert_topic(chat_id, 100, name="Python, AI & Data")
        assert await is_topic_monitored(chat_id, 100)
        cursor = await db.execute("SELECT is_active, name FROM topics WHERE chat_id = ? AND thread_id = ?;", (chat_id, 100))
        row = await cursor.fetchone()
        assert row["is_active"] == 1
        assert row["name"] == "Python, AI & Data"

    finally:
        await close_db()


@pytest.mark.asyncio
async def test_blacklist_mode_semantics(tmp_path: Path) -> None:
    """In blacklist mode, NULL is_active means monitored unless explicitly set to 0."""
    db_file = tmp_path / "blacklist_test.db"
    await init_db(str(db_file))

    try:
        chat_id = -100888
        await ensure_group_settings(chat_id)
        await set_group_mode(chat_id, "blacklist")

        # Name-only upsert (is_active is NULL)
        await upsert_topic(chat_id, 50, name="General Discussion")
        # In blacklist mode, NULL is monitored
        assert await is_topic_monitored(chat_id, 50)

        # Explicitly remove topic (/remove_topic -> is_active=0)
        await set_topic_active_status(chat_id, 50, is_active=0)
        assert not await is_topic_monitored(chat_id, 50)

        # Name edit must NOT un-blacklist it
        await upsert_topic(chat_id, 50, name="General Discussion (Archived)")
        assert not await is_topic_monitored(chat_id, 50)

    finally:
        await close_db()


@pytest.mark.asyncio
async def test_list_topics_with_counts_today_semantics(tmp_path: Path) -> None:
    """Verify list_topics_with_counts_today correctly reflects is_monitored for NULL is_active."""
    db_file = tmp_path / "list_topics_test.db"
    await init_db(str(db_file))

    try:
        chat_id = -100777
        await ensure_group_settings(chat_id)

        # Insert topic with NULL is_active
        await upsert_topic(chat_id, 10, name="Topic Ten")
        # Insert topic with is_active = 1
        await set_topic_active_status(chat_id, 20, is_active=1, name="Topic Twenty")
        # Insert topic with is_active = 0
        await set_topic_active_status(chat_id, 30, is_active=0, name="Topic Thirty")

        today_utc = "2026-11-21T00:00:00+00:00"

        # 1. Blacklist mode
        await set_group_mode(chat_id, "blacklist")
        topics = await list_topics_with_counts_today(chat_id, today_utc)
        topic_map = {t["thread_id"]: t for t in topics}

        assert topic_map[10]["is_monitored"] is True   # NULL is monitored in blacklist
        assert topic_map[20]["is_monitored"] is True   # 1 is monitored in blacklist
        assert topic_map[30]["is_monitored"] is False  # 0 is unmonitored in blacklist

        # 2. Whitelist mode
        await set_group_mode(chat_id, "whitelist")
        topics_wl = await list_topics_with_counts_today(chat_id, today_utc)
        topic_map_wl = {t["thread_id"]: t for t in topics_wl}

        assert topic_map_wl[10]["is_monitored"] is False  # NULL is NOT monitored in whitelist
        assert topic_map_wl[20]["is_monitored"] is True   # 1 is monitored in whitelist
        assert topic_map_wl[30]["is_monitored"] is False  # 0 is NOT monitored in whitelist

    finally:
        await close_db()

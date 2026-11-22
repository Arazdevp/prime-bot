"""Database repository providing parameterized queries for all operations."""

from datetime import datetime, timezone, timedelta
import json
from typing import Any
import aiosqlite
from bot.database.db import get_db, get_write_lock


async def insert_message(
    chat_id: int,
    thread_id: int,
    message_id: int,
    user_id: int | None,
    sender_name: str,
    text: str,
    media_type: str | None,
    reply_to_message_id: int | None,
    reply_to_name: str | None,
    reply_to_snippet: str | None,
    created_at: str,
) -> bool:
    """Insert a new message, ignoring duplicates (UNIQUE constraint on chat_id, message_id)."""
    conn = get_db()
    lock = get_write_lock()
    sql = """
    INSERT OR IGNORE INTO messages (
        chat_id, thread_id, message_id, user_id, sender_name,
        text, media_type, reply_to_message_id, reply_to_name,
        reply_to_snippet, created_at, is_summarized
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0);
    """
    async with lock:
        cursor = await conn.execute(
            sql,
            (
                chat_id,
                thread_id,
                message_id,
                user_id,
                sender_name,
                text,
                media_type,
                reply_to_message_id,
                reply_to_name,
                reply_to_snippet,
                created_at,
            ),
        )
        await conn.commit()
        return cursor.rowcount > 0


async def upsert_topic(
    chat_id: int,
    thread_id: int,
    name: str | None = None,
    is_active: int | None = None,
    updated_at: str | None = None,
) -> None:
    """Upsert topic details. Update name or is_active if specified."""
    conn = get_db()
    lock = get_write_lock()
    if updated_at is None:
        updated_at = datetime.now(timezone.utc).isoformat()

    sql = """
    INSERT INTO topics (chat_id, thread_id, name, is_active, updated_at)
    VALUES (?, ?, ?, ?, ?)
    ON CONFLICT(chat_id, thread_id) DO UPDATE SET
        name = COALESCE(excluded.name, topics.name),
        is_active = COALESCE(excluded.is_active, topics.is_active),
        updated_at = excluded.updated_at;
    """
    async with lock:
        await conn.execute(
            sql,
            (chat_id, thread_id, name, is_active, updated_at),
        )
        await conn.commit()


async def set_topic_active_status(
    chat_id: int,
    thread_id: int,
    is_active: int,
    name: str | None = None,
) -> None:
    """Set the is_active flag for a specific topic (used by /add_topic and /remove_topic)."""
    conn = get_db()
    lock = get_write_lock()
    updated_at = datetime.now(timezone.utc).isoformat()

    sql = """
    INSERT INTO topics (chat_id, thread_id, name, is_active, updated_at)
    VALUES (?, ?, COALESCE(?, ''), ?, ?)
    ON CONFLICT(chat_id, thread_id) DO UPDATE SET
        is_active = excluded.is_active,
        name = CASE WHEN excluded.name != '' THEN excluded.name ELSE topics.name END,
        updated_at = excluded.updated_at;
    """
    async with lock:
        await conn.execute(
            sql,
            (chat_id, thread_id, name, is_active, updated_at),
        )
        await conn.commit()


async def get_topic_name(chat_id: int, thread_id: int) -> str | None:
    """Get the stored name of a topic."""
    conn = get_db()
    cursor = await conn.execute(
        "SELECT name FROM topics WHERE chat_id = ? AND thread_id = ?;",
        (chat_id, thread_id),
    )
    row = await cursor.fetchone()
    if row and row["name"]:
        return str(row["name"])
    return None


async def ensure_group_settings(chat_id: int, default_min_messages: int | None = None) -> dict[str, Any]:
    """Retrieve group settings or initialize defaults if missing."""
    if default_min_messages is None:
        from bot.config import get_settings
        default_min_messages = get_settings().MIN_MESSAGES_PER_TOPIC

    conn = get_db()
    lock = get_write_lock()
    cursor = await conn.execute(
        "SELECT chat_id, mode, destination_mode, destination_thread_id, enabled, min_messages "
        "FROM group_settings WHERE chat_id = ?;",
        (chat_id,),
    )
    row = await cursor.fetchone()
    if row is not None:
        return dict(row)

    async with lock:
        # Re-check inside write lock
        cursor = await conn.execute(
            "SELECT chat_id, mode, destination_mode, destination_thread_id, enabled, min_messages "
            "FROM group_settings WHERE chat_id = ?;",
            (chat_id,),
        )
        row = await cursor.fetchone()
        if row is not None:
            return dict(row)

        await conn.execute(
            """
            INSERT INTO group_settings (
                chat_id, mode, destination_mode, destination_thread_id, enabled, min_messages
            ) VALUES (?, 'blacklist', 'per_topic', NULL, 1, ?);
            """,
            (chat_id, default_min_messages),
        )
        await conn.commit()

    return {
        "chat_id": chat_id,
        "mode": "blacklist",
        "destination_mode": "per_topic",
        "destination_thread_id": None,
        "enabled": 1,
        "min_messages": default_min_messages,
    }


async def set_group_mode(chat_id: int, mode: str) -> None:
    """Set topic filtering mode ('blacklist' or 'whitelist')."""
    conn = get_db()
    lock = get_write_lock()
    async with lock:
        await conn.execute(
            "UPDATE group_settings SET mode = ? WHERE chat_id = ?;",
            (mode, chat_id),
        )
        await conn.commit()


async def set_group_destination(
    chat_id: int,
    destination_mode: str,
    destination_thread_id: int | None,
) -> None:
    """Set destination mode ('per_topic' or 'aggregate') and aggregate thread ID."""
    conn = get_db()
    lock = get_write_lock()
    async with lock:
        await conn.execute(
            "UPDATE group_settings SET destination_mode = ?, destination_thread_id = ? WHERE chat_id = ?;",
            (destination_mode, destination_thread_id, chat_id),
        )
        await conn.commit()


async def set_group_enabled(chat_id: int, enabled: int) -> None:
    """Enable or disable summarization for a group."""
    conn = get_db()
    lock = get_write_lock()
    async with lock:
        await conn.execute(
            "UPDATE group_settings SET enabled = ? WHERE chat_id = ?;",
            (enabled, chat_id),
        )
        await conn.commit()


async def set_group_min_messages(chat_id: int, min_messages: int) -> None:
    """Set the minimum messages threshold for group summarization."""
    conn = get_db()
    lock = get_write_lock()
    async with lock:
        await conn.execute(
            "UPDATE group_settings SET min_messages = ? WHERE chat_id = ?;",
            (min_messages, chat_id),
        )
        await conn.commit()


async def is_topic_monitored(chat_id: int, thread_id: int) -> bool:
    """Check if topic is monitored based on group mode and topic status."""
    settings = await ensure_group_settings(chat_id)
    if settings.get("enabled", 1) == 0:
        return False

    mode = settings.get("mode", "blacklist")
    conn = get_db()
    cursor = await conn.execute(
        "SELECT is_active FROM topics WHERE chat_id = ? AND thread_id = ?;",
        (chat_id, thread_id),
    )
    row = await cursor.fetchone()

    if mode == "whitelist":
        # In whitelist mode, monitored ONLY IF row exists and is_active == 1
        return row is not None and row["is_active"] == 1
    else:
        # In blacklist mode, monitored UNLESS row exists and is_active == 0
        return row is None or row["is_active"] != 0


async def list_topics_with_counts_today(
    chat_id: int,
    today_start_utc: str,
) -> list[dict[str, Any]]:
    """List known topics along with their active status and today's stored message count."""
    settings = await ensure_group_settings(chat_id)
    mode = settings.get("mode", "blacklist")

    conn = get_db()
    sql = """
    SELECT 
        COALESCE(t.thread_id, m.thread_id) AS thread_id,
        COALESCE(t.name, CASE WHEN COALESCE(t.thread_id, m.thread_id) = 1 THEN 'عمومی' ELSE 'تاپیک ' || COALESCE(t.thread_id, m.thread_id) END) AS name,
        t.is_active,
        COUNT(m.id) AS today_count
    FROM (
        SELECT DISTINCT thread_id FROM topics WHERE chat_id = ?
        UNION
        SELECT DISTINCT thread_id FROM messages WHERE chat_id = ?
    ) all_threads
    LEFT JOIN topics t ON t.chat_id = ? AND t.thread_id = all_threads.thread_id
    LEFT JOIN messages m ON m.chat_id = ? AND m.thread_id = all_threads.thread_id AND m.created_at >= ?
    GROUP BY all_threads.thread_id
    ORDER BY all_threads.thread_id ASC;
    """
    cursor = await conn.execute(sql, (chat_id, chat_id, chat_id, chat_id, today_start_utc))
    rows = await cursor.fetchall()

    results: list[dict[str, Any]] = []
    for r in rows:
        tid = r["thread_id"]
        name = r["name"]
        raw_is_active = r["is_active"]

        if mode == "whitelist":
            is_monitored = raw_is_active is not None and raw_is_active == 1
        else:
            is_monitored = raw_is_active is None or raw_is_active != 0

        results.append({
            "thread_id": tid,
            "name": name,
            "is_active": raw_is_active,
            "is_monitored": is_monitored,
            "today_count": r["today_count"],
        })
    return results


async def fetch_unsummarized_messages(
    chat_id: int,
    thread_id: int,
    before: str | None = None,
) -> list[dict[str, Any]]:
    """Fetch unsummarized messages for a topic, optionally created before cutoff, ordered chronologically."""
    conn = get_db()
    if before:
        sql = """
        SELECT 
            id, chat_id, thread_id, message_id, user_id, sender_name,
            text, media_type, reply_to_message_id, reply_to_name,
            reply_to_snippet, created_at, is_summarized
        FROM messages
        WHERE chat_id = ? AND thread_id = ? AND is_summarized = 0 AND created_at < ?
        ORDER BY created_at ASC, message_id ASC;
        """
        cursor = await conn.execute(sql, (chat_id, thread_id, before))
    else:
        sql = """
        SELECT 
            id, chat_id, thread_id, message_id, user_id, sender_name,
            text, media_type, reply_to_message_id, reply_to_name,
            reply_to_snippet, created_at, is_summarized
        FROM messages
        WHERE chat_id = ? AND thread_id = ? AND is_summarized = 0
        ORDER BY created_at ASC, message_id ASC;
        """
        cursor = await conn.execute(sql, (chat_id, thread_id))
    rows = await cursor.fetchall()
    return [dict(r) for r in rows]


async def list_distinct_unsummarized_topics(
    chat_id: int,
    before: str | None = None,
) -> list[int]:
    """List distinct thread IDs having unsummarized messages for a group, optionally before cutoff."""
    conn = get_db()
    if before:
        sql = (
            "SELECT DISTINCT thread_id FROM messages "
            "WHERE chat_id = ? AND is_summarized = 0 AND created_at < ? "
            "ORDER BY thread_id ASC;"
        )
        cursor = await conn.execute(sql, (chat_id, before))
    else:
        sql = (
            "SELECT DISTINCT thread_id FROM messages "
            "WHERE chat_id = ? AND is_summarized = 0 "
            "ORDER BY thread_id ASC;"
        )
        cursor = await conn.execute(sql, (chat_id,))
    rows = await cursor.fetchall()
    return [r["thread_id"] for r in rows]


async def list_enabled_groups() -> list[dict[str, Any]]:
    """List all groups with enabled=1 in group_settings."""
    conn = get_db()
    cursor = await conn.execute(
        "SELECT chat_id, mode, destination_mode, destination_thread_id, enabled, min_messages "
        "FROM group_settings WHERE enabled = 1;"
    )
    rows = await cursor.fetchall()
    return [dict(r) for r in rows]


async def mark_messages_summarized(message_ids: list[int]) -> None:
    """Mark the given message IDs as summarized in batches of 500."""
    if not message_ids:
        return

    conn = get_db()
    lock = get_write_lock()
    async with lock:
        for i in range(0, len(message_ids), 500):
            chunk = message_ids[i : i + 500]
            placeholders = ",".join("?" for _ in chunk)
            sql = f"UPDATE messages SET is_summarized = 1 WHERE id IN ({placeholders});"
            await conn.execute(sql, chunk)
        await conn.commit()


async def delete_messages_by_ids(message_ids: list[int]) -> int:
    """Delete messages permanently by their database primary IDs in batches of 500 to reclaim disk space."""
    if not message_ids:
        return 0

    conn = get_db()
    lock = get_write_lock()
    total_deleted = 0
    async with lock:
        for i in range(0, len(message_ids), 500):
            chunk = message_ids[i : i + 500]
            placeholders = ",".join("?" for _ in chunk)
            sql = f"DELETE FROM messages WHERE id IN ({placeholders});"
            cursor = await conn.execute(sql, chunk)
            total_deleted += cursor.rowcount
        await conn.commit()
    return total_deleted


async def delete_old_messages(retention_days: int) -> int:
    """Delete messages older than retention_days (UTC). Returns count of deleted messages."""
    conn = get_db()
    lock = get_write_lock()
    cutoff = (datetime.now(timezone.utc) - timedelta(days=retention_days)).isoformat()
    sql = "DELETE FROM messages WHERE created_at < ?;"

    async with lock:
        cursor = await conn.execute(sql, (cutoff,))
        await conn.commit()
        return cursor.rowcount


async def count_today_messages(chat_id: int, today_start_utc: str) -> int:
    """Count messages saved for this chat since today_start_utc."""
    conn = get_db()
    cursor = await conn.execute(
        "SELECT COUNT(*) as cnt FROM messages WHERE chat_id = ? AND created_at >= ?;",
        (chat_id, today_start_utc),
    )
    row = await cursor.fetchone()
    return row["cnt"] if row else 0


async def count_unsummarized_messages(chat_id: int) -> int:
    """Count total unsummarized messages for a chat."""
    conn = get_db()
    cursor = await conn.execute(
        "SELECT COUNT(*) as cnt FROM messages WHERE chat_id = ? AND is_summarized = 0;",
        (chat_id,),
    )
    row = await cursor.fetchone()
    return row["cnt"] if row else 0


# ---------------- Meta Table Operations ----------------


async def get_meta_value(key: str) -> str | None:
    """Retrieve string value for a metadata key."""
    conn = get_db()
    cursor = await conn.execute("SELECT value FROM meta WHERE key = ?;", (key,))
    row = await cursor.fetchone()
    return str(row["value"]) if row and row["value"] is not None else None


async def set_meta_value(key: str, value: str) -> None:
    """Store or update metadata key-value pair."""
    conn = get_db()
    lock = get_write_lock()
    sql = """
    INSERT INTO meta (key, value) VALUES (?, ?)
    ON CONFLICT(key) DO UPDATE SET value = excluded.value;
    """
    async with lock:
        await conn.execute(sql, (key, value))
        await conn.commit()


async def get_last_nightly_run() -> str | None:
    """Get the ISO timestamp of the last executed nightly summary."""
    return await get_meta_value("last_nightly_run")


async def set_last_nightly_run(timestamp_iso: str) -> None:
    """Set the ISO timestamp of the last executed nightly summary."""
    await set_meta_value("last_nightly_run", timestamp_iso)


# ---------------- Summaries Table Operations ----------------


async def get_summary_record(chat_id: int, thread_id: int, day: str) -> dict[str, Any] | None:
    """Retrieve existing summary record for a topic and day."""
    conn = get_db()
    cursor = await conn.execute(
        "SELECT chat_id, thread_id, day, parts_json, sent_count, created_at "
        "FROM summaries WHERE chat_id = ? AND thread_id = ? AND day = ?;",
        (chat_id, thread_id, day),
    )
    row = await cursor.fetchone()
    return dict(row) if row else None


async def save_summary_record(
    chat_id: int,
    thread_id: int,
    day: str,
    parts: list[str],
    sent_count: int = 0,
) -> None:
    """Upsert summary record with parts_json and sent_count."""
    conn = get_db()
    lock = get_write_lock()
    now_utc = datetime.now(timezone.utc).isoformat()
    parts_json = json.dumps(parts, ensure_ascii=False)
    sql = """
    INSERT INTO summaries (chat_id, thread_id, day, parts_json, sent_count, created_at)
    VALUES (?, ?, ?, ?, ?, ?)
    ON CONFLICT(chat_id, thread_id, day) DO UPDATE SET
        parts_json = excluded.parts_json,
        sent_count = excluded.sent_count,
        created_at = excluded.created_at;
    """
    async with lock:
        await conn.execute(sql, (chat_id, thread_id, day, parts_json, sent_count, now_utc))
        await conn.commit()


async def update_summary_sent_count(
    chat_id: int,
    thread_id: int,
    day: str,
    sent_count: int,
) -> None:
    """Update sent_count progress for a summary record."""
    conn = get_db()
    lock = get_write_lock()
    sql = "UPDATE summaries SET sent_count = ? WHERE chat_id = ? AND thread_id = ? AND day = ?;"
    async with lock:
        await conn.execute(sql, (sent_count, chat_id, thread_id, day))
        await conn.commit()


async def delete_summary(chat_id: int, thread_id: int, day: str) -> None:
    """Delete a summary record by topic and day."""
    conn = get_db()
    lock = get_write_lock()
    async with lock:
        await conn.execute(
            "DELETE FROM summaries WHERE chat_id = ? AND thread_id = ? AND day = ?;",
            (chat_id, thread_id, day),
        )
        await conn.commit()

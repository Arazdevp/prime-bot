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


async def set_all_topics_active_status(
    chat_id: int,
    is_active: int,
) -> int:
    """Set is_active for all known topics in a group (across topics and messages tables)."""
    conn = get_db()
    lock = get_write_lock()
    updated_at = datetime.now(timezone.utc).isoformat()

    async with lock:
        cursor = await conn.execute(
            """
            SELECT DISTINCT thread_id FROM topics WHERE chat_id = ?
            UNION
            SELECT DISTINCT thread_id FROM messages WHERE chat_id = ?;
            """,
            (chat_id, chat_id),
        )
        rows = await cursor.fetchall()
        thread_ids = [int(r["thread_id"]) for r in rows]

        for tid in thread_ids:
            await conn.execute(
                """
                INSERT INTO topics (chat_id, thread_id, name, is_active, updated_at)
                VALUES (?, ?, NULL, ?, ?)
                ON CONFLICT(chat_id, thread_id) DO UPDATE SET
                    is_active = excluded.is_active,
                    updated_at = excluded.updated_at;
                """,
                (chat_id, tid, is_active, updated_at),
            )
        await conn.commit()
    return len(thread_ids)


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


async def get_group_settings(chat_id: int) -> dict[str, Any] | None:
    """Read-only lookup of group settings; returns None if not initialized (never writes to DB)."""
    conn = get_db()
    cursor = await conn.execute(
        "SELECT chat_id, mode, destination_mode, destination_thread_id, enabled, min_messages "
        "FROM group_settings WHERE chat_id = ?;",
        (chat_id,),
    )
    row = await cursor.fetchone()
    return dict(row) if row is not None else None


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


async def _ensure_group_settings_row_locked(conn: aiosqlite.Connection, chat_id: int) -> None:
    """Insert default group_settings row if missing (caller MUST hold get_write_lock())."""
    from bot.config import get_settings

    default_min = get_settings().MIN_MESSAGES_PER_TOPIC
    await conn.execute(
        """
        INSERT OR IGNORE INTO group_settings (
            chat_id, mode, destination_mode, destination_thread_id, enabled, min_messages
        ) VALUES (?, 'blacklist', 'per_topic', NULL, 1, ?);
        """,
        (chat_id, default_min),
    )


async def set_group_mode(chat_id: int, mode: str) -> None:
    """Set topic filtering mode ('blacklist' or 'whitelist')."""
    conn = get_db()
    lock = get_write_lock()
    async with lock:
        await _ensure_group_settings_row_locked(conn, chat_id)
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
        await _ensure_group_settings_row_locked(conn, chat_id)
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
        await _ensure_group_settings_row_locked(conn, chat_id)
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
        await _ensure_group_settings_row_locked(conn, chat_id)
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
        all_threads.thread_id AS thread_id,
        t.name AS raw_name,
        COALESCE(NULLIF(t.name, ''), CASE WHEN all_threads.thread_id = 1 THEN 'عمومی' ELSE 'تاپیک ' || all_threads.thread_id END) AS name,
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
        raw_name = (r["raw_name"] or "").strip()
        name = r["name"]
        raw_is_active = r["is_active"]
        has_known_name = (tid == 1) or bool(raw_name and raw_name != f"تاپیک {tid}")

        if mode == "whitelist":
            is_monitored = raw_is_active is not None and raw_is_active == 1
        else:
            is_monitored = raw_is_active is None or raw_is_active != 0

        results.append({
            "thread_id": tid,
            "name": name,
            "has_known_name": has_known_name,
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
    """List all active groups (not left in known_groups) with enabled=1 in group_settings."""
    conn = get_db()
    cursor = await conn.execute(
        """
        SELECT gs.chat_id, gs.mode, gs.destination_mode, gs.destination_thread_id, gs.enabled, gs.min_messages
        FROM group_settings gs
        LEFT JOIN known_groups kg ON kg.chat_id = gs.chat_id
        WHERE gs.enabled = 1 AND kg.left_at IS NULL;
        """
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


async def delete_old_summaries(retention_days: int) -> int:
    """Delete persisted summary records older than retention_days (UTC)."""
    conn = get_db()
    lock = get_write_lock()
    cutoff = (datetime.now(timezone.utc) - timedelta(days=retention_days)).isoformat()
    async with lock:
        cursor = await conn.execute(
            "DELETE FROM summaries WHERE created_at < ?;",
            (cutoff,),
        )
        await conn.commit()
        return cursor.rowcount


# ---------------- Map-Reduce Chunk Cache Operations ----------------


async def get_cached_map_chunk(chat_id: int, thread_id: int, chunk_hash: str) -> str | None:
    """Fetch a previously completed Map chunk summary by deterministic content hash."""
    conn = get_db()
    cursor = await conn.execute(
        "SELECT summary_text FROM map_chunk_cache "
        "WHERE chat_id = ? AND thread_id = ? AND chunk_hash = ?;",
        (chat_id, thread_id, chunk_hash),
    )
    row = await cursor.fetchone()
    if row and row["summary_text"]:
        return str(row["summary_text"])
    return None


async def save_cached_map_chunk(
    chat_id: int,
    thread_id: int,
    chunk_hash: str,
    summary_text: str,
) -> None:
    """Persist a completed Map chunk summary so retries resume without re-paying API costs."""
    conn = get_db()
    lock = get_write_lock()
    now_utc = datetime.now(timezone.utc).isoformat()
    sql = """
    INSERT INTO map_chunk_cache (chat_id, thread_id, chunk_hash, summary_text, created_at)
    VALUES (?, ?, ?, ?, ?)
    ON CONFLICT(chat_id, thread_id, chunk_hash) DO UPDATE SET
        summary_text = excluded.summary_text,
        created_at = excluded.created_at;
    """
    async with lock:
        await conn.execute(sql, (chat_id, thread_id, chunk_hash, summary_text, now_utc))
        await conn.commit()


async def clear_cached_map_chunks(chat_id: int, thread_id: int) -> int:
    """Delete all cached Map chunk summaries for a topic once its final summary is persisted."""
    conn = get_db()
    lock = get_write_lock()
    async with lock:
        cursor = await conn.execute(
            "DELETE FROM map_chunk_cache WHERE chat_id = ? AND thread_id = ?;",
            (chat_id, thread_id),
        )
        await conn.commit()
        return cursor.rowcount


async def delete_old_cached_map_chunks(retention_days: int) -> int:
    """Delete stale cached Map chunk summaries older than retention_days (UTC)."""
    conn = get_db()
    lock = get_write_lock()
    cutoff = (datetime.now(timezone.utc) - timedelta(days=retention_days)).isoformat()
    async with lock:
        cursor = await conn.execute(
            "DELETE FROM map_chunk_cache WHERE created_at < ?;",
            (cutoff,),
        )
        await conn.commit()
        return cursor.rowcount


# ---------------- Bot Users Table Operations ----------------


async def get_bot_user(user_id: int) -> dict[str, Any] | None:
    """Fetch a private-chat user record by Telegram user ID."""
    conn = get_db()
    cursor = await conn.execute(
        """
        SELECT user_id, username, full_name, first_seen, last_seen,
               last_variant_key, attempts_window_start, attempts_count,
               ignored_until, notified, dm_started
        FROM bot_users WHERE user_id = ?;
        """,
        (user_id,),
    )
    row = await cursor.fetchone()
    return dict(row) if row is not None else None


async def upsert_bot_user_seen(
    user_id: int,
    username: str | None,
    full_name: str | None,
    seen_at_iso: str | None = None,
) -> dict[str, Any]:
    """Insert or update a user's basic metadata and last_seen timestamp, returning the full row."""
    conn = get_db()
    lock = get_write_lock()
    now_iso = seen_at_iso or datetime.now(timezone.utc).isoformat()
    sql = """
    INSERT INTO bot_users (
        user_id, username, full_name, first_seen, last_seen,
        last_variant_key, attempts_window_start, attempts_count, ignored_until, notified, dm_started
    ) VALUES (?, ?, ?, ?, ?, NULL, NULL, 0, NULL, 0, 0)
    ON CONFLICT(user_id) DO UPDATE SET
        username = excluded.username,
        full_name = excluded.full_name,
        last_seen = excluded.last_seen;
    """
    async with lock:
        await conn.execute(sql, (user_id, username, full_name, now_iso, now_iso))
        await conn.commit()
        cursor = await conn.execute(
            """
            SELECT user_id, username, full_name, first_seen, last_seen,
                   last_variant_key, attempts_window_start, attempts_count,
                   ignored_until, notified, dm_started
            FROM bot_users WHERE user_id = ?;
            """,
            (user_id,),
        )
        row = await cursor.fetchone()
    return dict(row) if row is not None else {
        "user_id": user_id,
        "username": username,
        "full_name": full_name,
        "first_seen": now_iso,
        "last_seen": now_iso,
        "last_variant_key": None,
        "attempts_window_start": None,
        "attempts_count": 0,
        "ignored_until": None,
        "notified": 0,
        "dm_started": 0,
    }


async def set_bot_user_dm_started(
    user_id: int,
    dm_started: bool,
    *,
    username: str | None = None,
    full_name: str | None = None,
) -> None:
    """Record whether a user has an active private DM session with the bot."""
    conn = get_db()
    lock = get_write_lock()
    now_iso = datetime.now(timezone.utc).isoformat()
    flag = 1 if dm_started else 0
    sql = """
    INSERT INTO bot_users (
        user_id, username, full_name, first_seen, last_seen,
        last_variant_key, attempts_window_start, attempts_count, ignored_until, notified, dm_started
    ) VALUES (?, ?, ?, ?, ?, NULL, NULL, 0, NULL, 0, ?)
    ON CONFLICT(user_id) DO UPDATE SET
        username = COALESCE(excluded.username, bot_users.username),
        full_name = COALESCE(excluded.full_name, bot_users.full_name),
        last_seen = excluded.last_seen,
        dm_started = excluded.dm_started;
    """
    async with lock:
        await conn.execute(sql, (user_id, username, full_name, now_iso, now_iso, flag))
        await conn.commit()


async def is_bot_user_dm_started(user_id: int) -> bool:
    """Return True if the user is known to have started the bot in private chat."""
    conn = get_db()
    cursor = await conn.execute(
        "SELECT dm_started FROM bot_users WHERE user_id = ?;",
        (user_id,),
    )
    row = await cursor.fetchone()
    return bool(row and row["dm_started"] == 1)


async def update_bot_user_stranger_state(
    user_id: int,
    *,
    last_variant_key: str | None,
    attempts_window_start: str | None,
    attempts_count: int,
    ignored_until: str | None,
) -> None:
    """Persist anti-spam attempt window, ignore window, and last variant for a user."""
    conn = get_db()
    lock = get_write_lock()
    sql = """
    UPDATE bot_users
    SET last_variant_key = ?,
        attempts_window_start = ?,
        attempts_count = ?,
        ignored_until = ?
    WHERE user_id = ?;
    """
    async with lock:
        await conn.execute(
            sql,
            (last_variant_key, attempts_window_start, attempts_count, ignored_until, user_id),
        )
        await conn.commit()


async def update_bot_user_last_variant(user_id: int, last_variant_key: str) -> None:
    """Persist only the last_variant_key for a user (used for admin welcome/hint rotation)."""
    conn = get_db()
    lock = get_write_lock()
    async with lock:
        await conn.execute(
            "UPDATE bot_users SET last_variant_key = ? WHERE user_id = ?;",
            (last_variant_key, user_id),
        )
        await conn.commit()


async def mark_bot_user_notified(user_id: int) -> None:
    """Mark a stranger user as already reported to global admins."""
    conn = get_db()
    lock = get_write_lock()
    async with lock:
        await conn.execute(
            "UPDATE bot_users SET notified = 1 WHERE user_id = ?;",
            (user_id,),
        )
        await conn.commit()


# ---------------- Known Groups Table Operations ----------------


async def upsert_known_group(
    chat_id: int,
    title: str | None = None,
    is_forum: bool = True,
    added_at_iso: str | None = None,
) -> None:
    """Register or refresh an active group in known_groups (clears left_at)."""
    conn = get_db()
    lock = get_write_lock()
    now_iso = added_at_iso or datetime.now(timezone.utc).isoformat()
    clean_title = title.strip() if isinstance(title, str) and title.strip() else None
    is_forum_int = 1 if is_forum else 0
    sql = """
    INSERT INTO known_groups (chat_id, title, is_forum, added_at, left_at)
    VALUES (?, ?, ?, ?, NULL)
    ON CONFLICT(chat_id) DO UPDATE SET
        title = COALESCE(excluded.title, known_groups.title),
        is_forum = excluded.is_forum,
        left_at = NULL;
    """
    async with lock:
        await conn.execute(sql, (chat_id, clean_title, is_forum_int, now_iso))
        await conn.commit()


async def mark_known_group_left(
    chat_id: int,
    left_at_iso: str | None = None,
) -> None:
    """Mark a group as left in known_groups."""
    conn = get_db()
    lock = get_write_lock()
    now_iso = left_at_iso or datetime.now(timezone.utc).isoformat()
    async with lock:
        await conn.execute(
            "UPDATE known_groups SET left_at = ? WHERE chat_id = ?;",
            (now_iso, chat_id),
        )
        await conn.commit()


async def get_known_group(chat_id: int) -> dict[str, Any] | None:
    """Retrieve a known_groups entry by chat_id."""
    conn = get_db()
    cursor = await conn.execute(
        "SELECT chat_id, title, is_forum, added_at, left_at FROM known_groups WHERE chat_id = ?;",
        (chat_id,),
    )
    row = await cursor.fetchone()
    return dict(row) if row is not None else None


async def list_active_known_groups() -> list[dict[str, Any]]:
    """List all groups where the bot is currently active (left_at IS NULL)."""
    conn = get_db()
    cursor = await conn.execute(
        """
        SELECT chat_id, title, is_forum, added_at, left_at
        FROM known_groups
        WHERE left_at IS NULL
        ORDER BY added_at ASC, chat_id ASC;
        """
    )
    rows = await cursor.fetchall()
    return [dict(r) for r in rows]


# ---------------- Timers Table Operations ----------------

_TIMER_SELECT_COLS = (
    "id, chat_id, thread_id, chat_type, user_id, username, full_name, "
    "note, timer_type, fire_at, created_at, status, fired_at, source_message_id"
)


async def create_timer(
    *,
    chat_id: int,
    thread_id: int,
    chat_type: str,
    user_id: int,
    username: str | None,
    full_name: str,
    note: str | None,
    timer_type: str,
    fire_at_iso: str,
    created_at_iso: str | None = None,
    source_message_id: int | None = None,
) -> dict[str, Any]:
    """Insert a new active timer and return the created row."""
    conn = get_db()
    lock = get_write_lock()
    now_iso = created_at_iso or datetime.now(timezone.utc).isoformat()
    clean_note = note.strip() if isinstance(note, str) and note.strip() else None
    clean_username = username.strip().lstrip("@") if isinstance(username, str) and username.strip() else None
    clean_full_name = (full_name or "").strip() or "کاربر"

    sql = """
    INSERT INTO timers (
        chat_id, thread_id, chat_type, user_id, username, full_name,
        note, timer_type, fire_at, created_at, status, fired_at, source_message_id
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', NULL, ?);
    """
    async with lock:
        cursor = await conn.execute(
            sql,
            (
                chat_id,
                thread_id,
                chat_type,
                user_id,
                clean_username,
                clean_full_name,
                clean_note,
                timer_type,
                fire_at_iso,
                now_iso,
                source_message_id,
            ),
        )
        timer_id = int(cursor.lastrowid)
        await conn.commit()

    return {
        "id": timer_id,
        "chat_id": chat_id,
        "thread_id": thread_id,
        "chat_type": chat_type,
        "user_id": user_id,
        "username": clean_username,
        "full_name": clean_full_name,
        "note": clean_note,
        "timer_type": timer_type,
        "fire_at": fire_at_iso,
        "created_at": now_iso,
        "status": "active",
        "fired_at": None,
        "source_message_id": source_message_id,
    }


async def get_timer(timer_id: int) -> dict[str, Any] | None:
    """Fetch a single timer record by primary ID."""
    conn = get_db()
    cursor = await conn.execute(
        f"SELECT {_TIMER_SELECT_COLS} FROM timers WHERE id = ?;",
        (timer_id,),
    )
    row = await cursor.fetchone()
    return dict(row) if row is not None else None


async def count_active_timers_for_user(user_id: int) -> int:
    """Return total active timers owned by user_id across all chats."""
    conn = get_db()
    cursor = await conn.execute(
        "SELECT COUNT(*) AS cnt FROM timers WHERE user_id = ? AND status = 'active';",
        (user_id,),
    )
    row = await cursor.fetchone()
    return int(row["cnt"]) if row else 0


async def list_active_timers_for_user(
    user_id: int,
    chat_id: int | None = None,
) -> list[dict[str, Any]]:
    """List active timers for a user ordered chronologically by fire_at."""
    conn = get_db()
    if chat_id is not None:
        cursor = await conn.execute(
            f"SELECT {_TIMER_SELECT_COLS} FROM timers "
            "WHERE user_id = ? AND chat_id = ? AND status = 'active' "
            "ORDER BY fire_at ASC, id ASC;",
            (user_id, chat_id),
        )
    else:
        cursor = await conn.execute(
            f"SELECT {_TIMER_SELECT_COLS} FROM timers "
            "WHERE user_id = ? AND status = 'active' "
            "ORDER BY fire_at ASC, id ASC;",
            (user_id,),
        )
    rows = await cursor.fetchall()
    return [dict(r) for r in rows]


async def list_all_active_timers() -> list[dict[str, Any]]:
    """Return all currently active timers across all users (used for startup rehydration)."""
    conn = get_db()
    cursor = await conn.execute(
        f"SELECT {_TIMER_SELECT_COLS} FROM timers "
        "WHERE status = 'active' ORDER BY fire_at ASC, id ASC;"
    )
    rows = await cursor.fetchall()
    return [dict(r) for r in rows]


async def list_due_active_timers(now_utc_iso: str) -> list[dict[str, Any]]:
    """Return active timers whose fire_at timestamp is at or before now_utc_iso."""
    conn = get_db()
    cursor = await conn.execute(
        f"SELECT {_TIMER_SELECT_COLS} FROM timers "
        "WHERE status = 'active' AND fire_at <= ? ORDER BY fire_at ASC, id ASC;",
        (now_utc_iso,),
    )
    rows = await cursor.fetchall()
    return [dict(r) for r in rows]


async def claim_timer_for_firing(
    timer_id: int,
    fired_at_iso: str | None = None,
) -> dict[str, Any] | None:
    """Atomically transition an active timer to 'fired' and return it, or None if already claimed."""
    conn = get_db()
    lock = get_write_lock()
    ts = fired_at_iso or datetime.now(timezone.utc).isoformat()
    async with lock:
        cursor = await conn.execute(
            "UPDATE timers SET status = 'fired', fired_at = ? WHERE id = ? AND status = 'active';",
            (ts, timer_id),
        )
        if cursor.rowcount != 1:
            await conn.commit()
            return None
        fetch = await conn.execute(
            f"SELECT {_TIMER_SELECT_COLS} FROM timers WHERE id = ?;",
            (timer_id,),
        )
        row = await fetch.fetchone()
        await conn.commit()
    return dict(row) if row is not None else None


async def cancel_timer(timer_id: int, user_id: int) -> bool:
    """Cancel an active timer owned by user_id. Returns True if cancelled."""
    conn = get_db()
    lock = get_write_lock()
    async with lock:
        cursor = await conn.execute(
            "UPDATE timers SET status = 'cancelled' WHERE id = ? AND user_id = ? AND status = 'active';",
            (timer_id, user_id),
        )
        await conn.commit()
        return cursor.rowcount == 1


async def cancel_all_user_timers(
    user_id: int,
    chat_id: int | None = None,
) -> list[int]:
    """Cancel all active timers for user_id (optionally scoped to chat_id) and return their IDs."""
    conn = get_db()
    lock = get_write_lock()
    async with lock:
        if chat_id is not None:
            cur = await conn.execute(
                "SELECT id FROM timers WHERE user_id = ? AND chat_id = ? AND status = 'active';",
                (user_id, chat_id),
            )
            ids = [int(r["id"]) for r in await cur.fetchall()]
            if ids:
                await conn.execute(
                    "UPDATE timers SET status = 'cancelled' "
                    "WHERE user_id = ? AND chat_id = ? AND status = 'active';",
                    (user_id, chat_id),
                )
        else:
            cur = await conn.execute(
                "SELECT id FROM timers WHERE user_id = ? AND status = 'active';",
                (user_id,),
            )
            ids = [int(r["id"]) for r in await cur.fetchall()]
            if ids:
                await conn.execute(
                    "UPDATE timers SET status = 'cancelled' WHERE user_id = ? AND status = 'active';",
                    (user_id,),
                )
        await conn.commit()
    return ids


async def delete_old_completed_timers(retention_days: int = 30) -> int:
    """Delete fired or cancelled timers older than retention_days."""
    conn = get_db()
    lock = get_write_lock()
    cutoff = (datetime.now(timezone.utc) - timedelta(days=retention_days)).isoformat()
    async with lock:
        cursor = await conn.execute(
            "DELETE FROM timers WHERE status IN ('fired', 'cancelled') "
            "AND COALESCE(fired_at, fire_at, created_at) < ?;",
            (cutoff,),
        )
        await conn.commit()
        return cursor.rowcount


# ---------------- Morning Greetings Table Operations ----------------

_MORNING_SELECT_COLS = (
    "chat_id, enabled, thread_id, last_sent_day, deck_json, last_index, updated_at"
)


async def get_morning_config(chat_id: int) -> dict[str, Any] | None:
    """Read-only lookup of morning greeting configuration for a group."""
    conn = get_db()
    cursor = await conn.execute(
        f"SELECT {_MORNING_SELECT_COLS} FROM morning_greetings WHERE chat_id = ?;",
        (chat_id,),
    )
    row = await cursor.fetchone()
    return dict(row) if row is not None else None


async def ensure_morning_config(chat_id: int) -> dict[str, Any]:
    """Retrieve morning greeting configuration for chat_id, inserting disabled defaults if absent."""
    existing = await get_morning_config(chat_id)
    if existing is not None:
        return existing

    conn = get_db()
    lock = get_write_lock()
    now_iso = datetime.now(timezone.utc).isoformat()
    async with lock:
        cursor = await conn.execute(
            f"SELECT {_MORNING_SELECT_COLS} FROM morning_greetings WHERE chat_id = ?;",
            (chat_id,),
        )
        row = await cursor.fetchone()
        if row is not None:
            return dict(row)

        await conn.execute(
            """
            INSERT INTO morning_greetings (
                chat_id, enabled, thread_id, last_sent_day, deck_json, last_index, updated_at
            ) VALUES (?, 0, NULL, NULL, NULL, NULL, ?);
            """,
            (chat_id, now_iso),
        )
        await conn.commit()

    return {
        "chat_id": chat_id,
        "enabled": 0,
        "thread_id": None,
        "last_sent_day": None,
        "deck_json": None,
        "last_index": None,
        "updated_at": now_iso,
    }


async def set_morning_topic(
    chat_id: int,
    thread_id: int,
    *,
    auto_enable_if_unset: bool = True,
) -> dict[str, Any]:
    """Set target forum topic thread_id for daily morning greetings (auto-enables on first setup if requested)."""
    conn = get_db()
    lock = get_write_lock()
    now_iso = datetime.now(timezone.utc).isoformat()
    safe_tid = int(thread_id) if thread_id and int(thread_id) > 0 else 1

    async with lock:
        cursor = await conn.execute(
            f"SELECT {_MORNING_SELECT_COLS} FROM morning_greetings WHERE chat_id = ?;",
            (chat_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            new_enabled = 1 if auto_enable_if_unset else 0
            await conn.execute(
                """
                INSERT INTO morning_greetings (
                    chat_id, enabled, thread_id, last_sent_day, deck_json, last_index, updated_at
                ) VALUES (?, ?, ?, NULL, NULL, NULL, ?);
                """,
                (chat_id, new_enabled, safe_tid, now_iso),
            )
        else:
            prev_tid = row["thread_id"]
            prev_enabled = int(row["enabled"])
            new_enabled = 1 if (auto_enable_if_unset and prev_tid is None) else prev_enabled
            await conn.execute(
                """
                UPDATE morning_greetings
                SET thread_id = ?, enabled = ?, updated_at = ?
                WHERE chat_id = ?;
                """,
                (safe_tid, new_enabled, now_iso, chat_id),
            )
        await conn.commit()

        fetch = await conn.execute(
            f"SELECT {_MORNING_SELECT_COLS} FROM morning_greetings WHERE chat_id = ?;",
            (chat_id,),
        )
        updated = await fetch.fetchone()
    return dict(updated)


async def set_morning_enabled(chat_id: int, enabled: int) -> None:
    """Enable or disable daily morning greeting for chat_id."""
    conn = get_db()
    lock = get_write_lock()
    now_iso = datetime.now(timezone.utc).isoformat()
    flag = 1 if enabled else 0
    async with lock:
        await conn.execute(
            """
            INSERT INTO morning_greetings (
                chat_id, enabled, thread_id, last_sent_day, deck_json, last_index, updated_at
            ) VALUES (?, ?, NULL, NULL, NULL, NULL, ?)
            ON CONFLICT(chat_id) DO UPDATE SET
                enabled = excluded.enabled,
                updated_at = excluded.updated_at;
            """,
            (chat_id, flag, now_iso),
        )
        await conn.commit()


async def list_enabled_morning_groups() -> list[dict[str, Any]]:
    """List all active groups with morning greetings enabled and a target topic configured."""
    conn = get_db()
    cursor = await conn.execute(
        """
        SELECT mg.chat_id, mg.enabled, mg.thread_id, mg.last_sent_day,
               mg.deck_json, mg.last_index, mg.updated_at
        FROM morning_greetings mg
        LEFT JOIN known_groups kg ON kg.chat_id = mg.chat_id
        WHERE mg.enabled = 1 AND mg.thread_id IS NOT NULL AND kg.left_at IS NULL
        ORDER BY mg.chat_id ASC;
        """
    )
    rows = await cursor.fetchall()
    return [dict(r) for r in rows]


async def claim_morning_greeting_draw(
    chat_id: int,
    *,
    today_day_str: str,
    is_test: bool = False,
) -> tuple[int, str | None, int] | None:
    """Atomically draw the next non-repeating template index from SQLite shuffle-bag.

    For automatic runs (is_test=False), enforces at-most-once delivery per today_day_str by
    checking and updating last_sent_day inside the write lock. Returns None if already sent today.
    For test runs (is_test=True), draws a fresh index from the deck without modifying last_sent_day.

    Returns (chosen_index, previous_last_sent_day, remaining_in_deck).
    """
    from bot.texts.morning_texts import draw_next_greeting_index

    conn = get_db()
    lock = get_write_lock()
    now_iso = datetime.now(timezone.utc).isoformat()

    async with lock:
        cursor = await conn.execute(
            f"SELECT {_MORNING_SELECT_COLS} FROM morning_greetings WHERE chat_id = ?;",
            (chat_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None

        prev_last_day = row["last_sent_day"]
        if not is_test and prev_last_day == today_day_str:
            return None

        raw_deck = row["deck_json"]
        deck: list[int] | None = None
        if raw_deck:
            try:
                parsed = json.loads(raw_deck)
                if isinstance(parsed, list):
                    deck = parsed
            except Exception:
                deck = None

        last_idx = row["last_index"]
        chosen_idx, remaining_deck = draw_next_greeting_index(
            deck,
            int(last_idx) if last_idx is not None else None,
        )
        new_deck_json = json.dumps(remaining_deck)
        new_last_day = prev_last_day if is_test else today_day_str

        await conn.execute(
            """
            UPDATE morning_greetings
            SET last_sent_day = ?, deck_json = ?, last_index = ?, updated_at = ?
            WHERE chat_id = ?;
            """,
            (new_last_day, new_deck_json, chosen_idx, now_iso, chat_id),
        )
        await conn.commit()

    return chosen_idx, prev_last_day, len(remaining_deck)


async def rollback_morning_last_sent_day(chat_id: int, previous_day: str | None) -> None:
    """Restore last_sent_day if an automatic morning greeting delivery failed."""
    conn = get_db()
    lock = get_write_lock()
    now_iso = datetime.now(timezone.utc).isoformat()
    async with lock:
        await conn.execute(
            "UPDATE morning_greetings SET last_sent_day = ?, updated_at = ? WHERE chat_id = ?;",
            (previous_day, now_iso, chat_id),
        )
        await conn.commit()


# ---------------- DeepTalk Admins & Settings Operations ----------------

_DEEPTALK_SESSION_COLS = (
    "id, chat_id, thread_id, subject, description, created_by, creator_name, "
    "start_at, end_at, duration_minutes, status, announce_message_id, "
    "start_message_id, end_message_id, checkin_deck_json, last_checkin_idx, "
    "created_at, started_at, ended_at"
)


def _clean_username(username: str | None) -> str | None:
    if not isinstance(username, str):
        return None
    cleaned = username.strip().lstrip("@").strip()
    return cleaned if cleaned else None


async def add_deeptalk_admin(
    *,
    user_id: int,
    username: str | None,
    full_name: str | None,
    added_by: int,
    added_at_iso: str | None = None,
) -> dict[str, Any]:
    """Register or refresh a designated DeepTalk admin."""
    conn = get_db()
    lock = get_write_lock()
    now_iso = added_at_iso or datetime.now(timezone.utc).isoformat()
    clean_u = _clean_username(username)
    clean_name = (full_name or "").strip() or f"کاربر {user_id}"

    sql = """
    INSERT INTO deeptalk_admins (user_id, username, full_name, added_by, added_at)
    VALUES (?, ?, ?, ?, ?)
    ON CONFLICT(user_id) DO UPDATE SET
        username = COALESCE(excluded.username, deeptalk_admins.username),
        full_name = CASE
            WHEN excluded.full_name != '' AND excluded.full_name != ('کاربر ' || excluded.user_id)
            THEN excluded.full_name
            ELSE COALESCE(deeptalk_admins.full_name, excluded.full_name)
        END,
        added_by = excluded.added_by,
        added_at = excluded.added_at;
    """
    async with lock:
        await conn.execute(sql, (user_id, clean_u, clean_name, added_by, now_iso))
        await conn.commit()
        cursor = await conn.execute(
            "SELECT user_id, username, full_name, added_by, added_at FROM deeptalk_admins WHERE user_id = ?;",
            (user_id,),
        )
        row = await cursor.fetchone()
    return dict(row) if row is not None else {
        "user_id": user_id,
        "username": clean_u,
        "full_name": clean_name,
        "added_by": added_by,
        "added_at": now_iso,
    }


async def remove_deeptalk_admin(user_id: int) -> bool:
    """Remove a designated DeepTalk admin by user_id. Returns True if removed."""
    conn = get_db()
    lock = get_write_lock()
    async with lock:
        cursor = await conn.execute(
            "DELETE FROM deeptalk_admins WHERE user_id = ?;",
            (user_id,),
        )
        await conn.commit()
        return cursor.rowcount > 0


async def is_deeptalk_admin(user_id: int) -> bool:
    """Return True if user_id is explicitly registered in deeptalk_admins."""
    conn = get_db()
    cursor = await conn.execute(
        "SELECT 1 FROM deeptalk_admins WHERE user_id = ? LIMIT 1;",
        (user_id,),
    )
    return (await cursor.fetchone()) is not None


async def list_deeptalk_admins() -> list[dict[str, Any]]:
    """Return all designated DeepTalk admins ordered by added_at ASC."""
    conn = get_db()
    cursor = await conn.execute(
        "SELECT user_id, username, full_name, added_by, added_at "
        "FROM deeptalk_admins ORDER BY added_at ASC, user_id ASC;"
    )
    rows = await cursor.fetchall()
    return [dict(r) for r in rows]


async def find_known_user_by_username(username: str) -> dict[str, Any] | None:
    """Look up a user's user_id and display info case-insensitively across known tables."""
    clean_u = _clean_username(username)
    if not clean_u:
        return None

    conn = get_db()
    # 1. Check bot_users (most up-to-date private interactions)
    cur = await conn.execute(
        "SELECT user_id, username, full_name FROM bot_users "
        "WHERE LOWER(LTRIM(username, '@')) = LOWER(?) ORDER BY last_seen DESC LIMIT 1;",
        (clean_u,),
    )
    row = await cur.fetchone()
    if row is not None:
        return {
            "user_id": int(row["user_id"]),
            "username": _clean_username(row["username"]) or clean_u,
            "full_name": row["full_name"] or clean_u,
        }

    # 2. Check deeptalk_participants
    cur = await conn.execute(
        "SELECT user_id, username, full_name FROM deeptalk_participants "
        "WHERE LOWER(LTRIM(username, '@')) = LOWER(?) ORDER BY session_id DESC LIMIT 1;",
        (clean_u,),
    )
    row = await cur.fetchone()
    if row is not None:
        return {
            "user_id": int(row["user_id"]),
            "username": _clean_username(row["username"]) or clean_u,
            "full_name": row["full_name"] or clean_u,
        }

    # 3. Check timers
    cur = await conn.execute(
        "SELECT user_id, username, full_name FROM timers "
        "WHERE LOWER(LTRIM(username, '@')) = LOWER(?) ORDER BY id DESC LIMIT 1;",
        (clean_u,),
    )
    row = await cur.fetchone()
    if row is not None:
        return {
            "user_id": int(row["user_id"]),
            "username": _clean_username(row["username"]) or clean_u,
            "full_name": row["full_name"] or clean_u,
        }

    return None


async def get_deeptalk_settings(chat_id: int) -> dict[str, Any] | None:
    """Fetch per-group DeepTalk configuration if set."""
    conn = get_db()
    cursor = await conn.execute(
        "SELECT chat_id, thread_id, default_duration_minutes, updated_at "
        "FROM deeptalk_settings WHERE chat_id = ?;",
        (chat_id,),
    )
    row = await cursor.fetchone()
    return dict(row) if row is not None else None


async def set_deeptalk_topic(
    chat_id: int,
    thread_id: int,
    default_duration_minutes: int | None = None,
) -> dict[str, Any]:
    """Persist target forum topic thread_id (and optional default duration) for a group's DeepTalks."""
    conn = get_db()
    lock = get_write_lock()
    now_iso = datetime.now(timezone.utc).isoformat()
    safe_tid = int(thread_id) if thread_id and int(thread_id) > 0 else 1
    dur = max(5, min(360, int(default_duration_minutes))) if default_duration_minutes is not None else 45

    async with lock:
        if default_duration_minutes is not None:
            await conn.execute(
                """
                INSERT INTO deeptalk_settings (chat_id, thread_id, default_duration_minutes, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(chat_id) DO UPDATE SET
                    thread_id = excluded.thread_id,
                    default_duration_minutes = excluded.default_duration_minutes,
                    updated_at = excluded.updated_at;
                """,
                (chat_id, safe_tid, dur, now_iso),
            )
        else:
            await conn.execute(
                """
                INSERT INTO deeptalk_settings (chat_id, thread_id, default_duration_minutes, updated_at)
                VALUES (?, ?, 45, ?)
                ON CONFLICT(chat_id) DO UPDATE SET
                    thread_id = excluded.thread_id,
                    updated_at = excluded.updated_at;
                """,
                (chat_id, safe_tid, now_iso),
            )
        await conn.commit()
        cur = await conn.execute(
            "SELECT chat_id, thread_id, default_duration_minutes, updated_at "
            "FROM deeptalk_settings WHERE chat_id = ?;",
            (chat_id,),
        )
        updated = await cur.fetchone()
    return dict(updated)


async def detect_deepwork_topic_id(chat_id: int) -> int | None:
    """Return configured DeepTalk thread_id, or auto-detect a 'Deep Work / دیپ‌ورک' topic if present."""
    cfg = await get_deeptalk_settings(chat_id)
    if cfg and cfg.get("thread_id") is not None:
        return int(cfg["thread_id"])

    conn = get_db()
    cursor = await conn.execute(
        "SELECT thread_id, name FROM topics WHERE chat_id = ? ORDER BY thread_id ASC;",
        (chat_id,),
    )
    rows = await cursor.fetchall()
    keywords = ("deep work", "deepwork", "دیپ‌ورک", "دیپ ورک", "دیپورک", "deeptalk", "دیپ‌تاک", "دیپ تاک")
    for r in rows:
        raw_name = (r["name"] or "").strip().lower()
        normalized = raw_name.replace("\u200c", "")
        for kw in keywords:
            kw_norm = kw.replace("\u200c", "")
            if kw in raw_name or kw_norm in normalized:
                return int(r["thread_id"])
    return None


# ---------------- DeepTalk Sessions & Participants Operations ----------------


async def create_deeptalk_session(
    *,
    chat_id: int,
    thread_id: int,
    subject: str,
    description: str | None,
    created_by: int,
    creator_name: str,
    start_at_iso: str,
    end_at_iso: str,
    duration_minutes: int,
    created_at_iso: str | None = None,
) -> dict[str, Any]:
    """Insert a new scheduled DeepTalk session and return its record."""
    conn = get_db()
    lock = get_write_lock()
    now_iso = created_at_iso or datetime.now(timezone.utc).isoformat()
    clean_subject = subject.strip()
    clean_desc = description.strip() if isinstance(description, str) and description.strip() else None
    clean_creator = (creator_name or "").strip() or "ادمین دیپ‌تاک"
    safe_tid = int(thread_id) if thread_id and int(thread_id) > 0 else 1

    sql = """
    INSERT INTO deeptalk_sessions (
        chat_id, thread_id, subject, description, created_by, creator_name,
        start_at, end_at, duration_minutes, status,
        announce_message_id, start_message_id, end_message_id,
        checkin_deck_json, last_checkin_idx, created_at, started_at, ended_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'scheduled', NULL, NULL, NULL, NULL, NULL, ?, NULL, NULL);
    """
    async with lock:
        cursor = await conn.execute(
            sql,
            (
                chat_id,
                safe_tid,
                clean_subject,
                clean_desc,
                created_by,
                clean_creator,
                start_at_iso,
                end_at_iso,
                int(duration_minutes),
                now_iso,
            ),
        )
        session_id = int(cursor.lastrowid)
        await conn.commit()
        cur = await conn.execute(
            f"SELECT {_DEEPTALK_SESSION_COLS} FROM deeptalk_sessions WHERE id = ?;",
            (session_id,),
        )
        row = await cur.fetchone()
    return dict(row)


async def get_deeptalk_session(session_id: int) -> dict[str, Any] | None:
    """Fetch a single DeepTalk session by primary ID."""
    conn = get_db()
    cursor = await conn.execute(
        f"SELECT {_DEEPTALK_SESSION_COLS} FROM deeptalk_sessions WHERE id = ?;",
        (session_id,),
    )
    row = await cursor.fetchone()
    return dict(row) if row is not None else None


async def update_deeptalk_message_ids(
    session_id: int,
    *,
    announce_message_id: int | None = None,
    start_message_id: int | None = None,
    end_message_id: int | None = None,
) -> None:
    """Store Telegram message IDs for announcement, start, or end cards."""
    conn = get_db()
    lock = get_write_lock()
    updates: list[str] = []
    params: list[Any] = []
    if announce_message_id is not None:
        updates.append("announce_message_id = ?")
        params.append(int(announce_message_id))
    if start_message_id is not None:
        updates.append("start_message_id = ?")
        params.append(int(start_message_id))
    if end_message_id is not None:
        updates.append("end_message_id = ?")
        params.append(int(end_message_id))
    if not updates:
        return
    params.append(session_id)
    sql = f"UPDATE deeptalk_sessions SET {', '.join(updates)} WHERE id = ?;"
    async with lock:
        await conn.execute(sql, tuple(params))
        await conn.commit()


async def list_active_or_scheduled_deeptalks(chat_id: int | None = None) -> list[dict[str, Any]]:
    """List 'live' and 'scheduled' DeepTalk sessions ('live' first, then by start_at ASC)."""
    conn = get_db()
    if chat_id is not None:
        cursor = await conn.execute(
            f"SELECT {_DEEPTALK_SESSION_COLS} FROM deeptalk_sessions "
            "WHERE chat_id = ? AND status IN ('live', 'scheduled') "
            "ORDER BY CASE WHEN status = 'live' THEN 0 ELSE 1 END ASC, start_at ASC, id ASC;",
            (chat_id,),
        )
    else:
        cursor = await conn.execute(
            f"SELECT {_DEEPTALK_SESSION_COLS} FROM deeptalk_sessions "
            "WHERE status IN ('live', 'scheduled') "
            "ORDER BY CASE WHEN status = 'live' THEN 0 ELSE 1 END ASC, start_at ASC, id ASC;"
        )
    rows = await cursor.fetchall()
    return [dict(r) for r in rows]


async def list_due_deeptalk_starts(now_utc_iso: str) -> list[dict[str, Any]]:
    """Return scheduled DeepTalk sessions whose start_at is at or before now_utc_iso."""
    conn = get_db()
    cursor = await conn.execute(
        f"SELECT {_DEEPTALK_SESSION_COLS} FROM deeptalk_sessions "
        "WHERE status = 'scheduled' AND start_at <= ? ORDER BY start_at ASC, id ASC;",
        (now_utc_iso,),
    )
    rows = await cursor.fetchall()
    return [dict(r) for r in rows]


async def list_due_deeptalk_ends(now_utc_iso: str) -> list[dict[str, Any]]:
    """Return live DeepTalk sessions whose end_at is at or before now_utc_iso."""
    conn = get_db()
    cursor = await conn.execute(
        f"SELECT {_DEEPTALK_SESSION_COLS} FROM deeptalk_sessions "
        "WHERE status = 'live' AND end_at <= ? ORDER BY end_at ASC, id ASC;",
        (now_utc_iso,),
    )
    rows = await cursor.fetchall()
    return [dict(r) for r in rows]


async def claim_deeptalk_for_start(
    session_id: int,
    *,
    started_at_iso: str | None = None,
    new_end_at_iso: str | None = None,
) -> dict[str, Any] | None:
    """Atomically transition a 'scheduled' session to 'live', returning the updated row or None."""
    conn = get_db()
    lock = get_write_lock()
    ts = started_at_iso or datetime.now(timezone.utc).isoformat()
    async with lock:
        if new_end_at_iso is not None:
            cursor = await conn.execute(
                "UPDATE deeptalk_sessions SET status = 'live', started_at = ?, end_at = ? "
                "WHERE id = ? AND status = 'scheduled';",
                (ts, new_end_at_iso, session_id),
            )
        else:
            cursor = await conn.execute(
                "UPDATE deeptalk_sessions SET status = 'live', started_at = ? "
                "WHERE id = ? AND status = 'scheduled';",
                (ts, session_id),
            )
        if cursor.rowcount != 1:
            await conn.commit()
            return None
        cur = await conn.execute(
            f"SELECT {_DEEPTALK_SESSION_COLS} FROM deeptalk_sessions WHERE id = ?;",
            (session_id,),
        )
        row = await cur.fetchone()
        await conn.commit()
    return dict(row) if row is not None else None


async def claim_deeptalk_for_end(
    session_id: int,
    *,
    ended_at_iso: str | None = None,
) -> dict[str, Any] | None:
    """Atomically transition a 'live' session to 'completed', returning the updated row or None."""
    conn = get_db()
    lock = get_write_lock()
    ts = ended_at_iso or datetime.now(timezone.utc).isoformat()
    async with lock:
        cursor = await conn.execute(
            "UPDATE deeptalk_sessions SET status = 'completed', ended_at = ? "
            "WHERE id = ? AND status = 'live';",
            (ts, session_id),
        )
        if cursor.rowcount != 1:
            await conn.commit()
            return None
        cur = await conn.execute(
            f"SELECT {_DEEPTALK_SESSION_COLS} FROM deeptalk_sessions WHERE id = ?;",
            (session_id,),
        )
        row = await cur.fetchone()
        await conn.commit()
    return dict(row) if row is not None else None


async def extend_deeptalk_session(
    session_id: int,
    extra_minutes: int,
    *,
    now_utc: datetime | None = None,
) -> dict[str, Any] | None:
    """Extend a 'live' DeepTalk session's end_at and duration_minutes atomically."""
    if extra_minutes <= 0:
        return None
    conn = get_db()
    lock = get_write_lock()
    current_utc = now_utc.astimezone(timezone.utc) if now_utc else datetime.now(timezone.utc)

    async with lock:
        cur = await conn.execute(
            f"SELECT {_DEEPTALK_SESSION_COLS} FROM deeptalk_sessions WHERE id = ? AND status = 'live';",
            (session_id,),
        )
        row = await cur.fetchone()
        if row is None:
            return None

        raw_end = str(row["end_at"])
        end_dt = datetime.fromisoformat(raw_end)
        if end_dt.tzinfo is None:
            end_dt = end_dt.replace(tzinfo=timezone.utc)
        else:
            end_dt = end_dt.astimezone(timezone.utc)

        base_dt = max(end_dt, current_utc)
        new_end_dt = (base_dt + timedelta(minutes=extra_minutes)).replace(microsecond=0)
        new_dur = int(row["duration_minutes"]) + int(extra_minutes)

        await conn.execute(
            "UPDATE deeptalk_sessions SET end_at = ?, duration_minutes = ? WHERE id = ? AND status = 'live';",
            (new_end_dt.isoformat(), new_dur, session_id),
        )
        await conn.commit()
        cur2 = await conn.execute(
            f"SELECT {_DEEPTALK_SESSION_COLS} FROM deeptalk_sessions WHERE id = ?;",
            (session_id,),
        )
        updated = await cur2.fetchone()
    return dict(updated) if updated is not None else None


async def cancel_deeptalk_session(
    session_id: int,
    *,
    ended_at_iso: str | None = None,
) -> dict[str, Any] | None:
    """Atomically cancel a 'scheduled' or 'live' DeepTalk session."""
    conn = get_db()
    lock = get_write_lock()
    ts = ended_at_iso or datetime.now(timezone.utc).isoformat()
    async with lock:
        cursor = await conn.execute(
            "UPDATE deeptalk_sessions SET status = 'cancelled', ended_at = ? "
            "WHERE id = ? AND status IN ('scheduled', 'live');",
            (ts, session_id),
        )
        if cursor.rowcount != 1:
            await conn.commit()
            return None
        cur = await conn.execute(
            f"SELECT {_DEEPTALK_SESSION_COLS} FROM deeptalk_sessions WHERE id = ?;",
            (session_id,),
        )
        row = await cur.fetchone()
        await conn.commit()
    return dict(row) if row is not None else None


async def delete_old_deeptalk_sessions(retention_days: int = 30) -> int:
    """Delete completed or cancelled DeepTalk sessions older than retention_days."""
    conn = get_db()
    lock = get_write_lock()
    cutoff = (datetime.now(timezone.utc) - timedelta(days=retention_days)).isoformat()
    async with lock:
        cursor = await conn.execute(
            "DELETE FROM deeptalk_sessions WHERE status IN ('completed', 'cancelled') "
            "AND COALESCE(ended_at, end_at, created_at) < ?;",
            (cutoff,),
        )
        await conn.commit()
        return cursor.rowcount


async def register_deeptalk_participant(
    *,
    session_id: int,
    user_id: int,
    username: str | None,
    full_name: str,
    now_iso: str | None = None,
) -> tuple[bool, int]:
    """Pre-register a user for a DeepTalk session. Returns (newly_registered, total_registered)."""
    conn = get_db()
    lock = get_write_lock()
    ts = now_iso or datetime.now(timezone.utc).isoformat()
    clean_u = _clean_username(username)
    clean_name = (full_name or "").strip() or "کاربر"

    async with lock:
        cur = await conn.execute(
            "SELECT registered FROM deeptalk_participants WHERE session_id = ? AND user_id = ?;",
            (session_id, user_id),
        )
        existing = await cur.fetchone()
        already_reg = bool(existing and int(existing["registered"]) == 1)

        await conn.execute(
            """
            INSERT INTO deeptalk_participants (
                session_id, user_id, username, full_name,
                registered, registered_at, checked_in, checked_in_at, message_count
            ) VALUES (?, ?, ?, ?, 1, ?, 0, NULL, 0)
            ON CONFLICT(session_id, user_id) DO UPDATE SET
                username = excluded.username,
                full_name = excluded.full_name,
                registered = 1,
                registered_at = COALESCE(deeptalk_participants.registered_at, excluded.registered_at);
            """,
            (session_id, user_id, clean_u, clean_name, ts),
        )
        cnt_cur = await conn.execute(
            "SELECT COUNT(*) AS cnt FROM deeptalk_participants WHERE session_id = ? AND registered = 1;",
            (session_id,),
        )
        cnt_row = await cnt_cur.fetchone()
        total = int(cnt_row["cnt"]) if cnt_row else 0
        await conn.commit()

    return (not already_reg), total


async def unregister_deeptalk_participant(
    *,
    session_id: int,
    user_id: int,
) -> tuple[bool, int]:
    """Cancel pre-registration for a user. Returns (was_registered, total_registered)."""
    conn = get_db()
    lock = get_write_lock()

    async with lock:
        cur = await conn.execute(
            "SELECT registered, checked_in, message_count FROM deeptalk_participants "
            "WHERE session_id = ? AND user_id = ?;",
            (session_id, user_id),
        )
        row = await cur.fetchone()
        if row is None or int(row["registered"]) == 0:
            cnt_cur = await conn.execute(
                "SELECT COUNT(*) AS cnt FROM deeptalk_participants WHERE session_id = ? AND registered = 1;",
                (session_id,),
            )
            cnt_row = await cnt_cur.fetchone()
            return False, (int(cnt_row["cnt"]) if cnt_row else 0)

        if int(row["checked_in"]) == 0 and int(row["message_count"]) == 0:
            await conn.execute(
                "DELETE FROM deeptalk_participants WHERE session_id = ? AND user_id = ?;",
                (session_id, user_id),
            )
        else:
            await conn.execute(
                "UPDATE deeptalk_participants SET registered = 0, registered_at = NULL "
                "WHERE session_id = ? AND user_id = ?;",
                (session_id, user_id),
            )

        cnt_cur = await conn.execute(
            "SELECT COUNT(*) AS cnt FROM deeptalk_participants WHERE session_id = ? AND registered = 1;",
            (session_id,),
        )
        cnt_row = await cnt_cur.fetchone()
        total = int(cnt_row["cnt"]) if cnt_row else 0
        await conn.commit()

    return True, total


async def checkin_deeptalk_participant_and_draw(
    *,
    session_id: int,
    user_id: int,
    username: str | None,
    full_name: str,
    now_iso: str | None = None,
) -> tuple[bool, int, int] | None:
    """Atomically check in a user to a 'live' DeepTalk and draw a unique check-in template index.

    Returns:
      - None if the session does not exist or is not currently 'live'.
      - (False, -1, total_checked_in) if user already checked in earlier.
      - (True, chosen_template_idx, total_checked_in) on first check-in.
    """
    from bot.texts.deeptalk_texts import draw_next_checkin_index

    conn = get_db()
    lock = get_write_lock()
    ts = now_iso or datetime.now(timezone.utc).isoformat()
    clean_u = _clean_username(username)
    clean_name = (full_name or "").strip() or "کاربر"

    async with lock:
        sess_cur = await conn.execute(
            "SELECT status, checkin_deck_json, last_checkin_idx FROM deeptalk_sessions WHERE id = ?;",
            (session_id,),
        )
        sess_row = await sess_cur.fetchone()
        if sess_row is None or str(sess_row["status"]) != "live":
            return None

        part_cur = await conn.execute(
            "SELECT checked_in FROM deeptalk_participants WHERE session_id = ? AND user_id = ?;",
            (session_id, user_id),
        )
        part_row = await part_cur.fetchone()
        if part_row is not None and int(part_row["checked_in"]) == 1:
            await conn.execute(
                "UPDATE deeptalk_participants SET username = ?, full_name = ? "
                "WHERE session_id = ? AND user_id = ?;",
                (clean_u, clean_name, session_id, user_id),
            )
            cnt_cur = await conn.execute(
                "SELECT COUNT(*) AS cnt FROM deeptalk_participants WHERE session_id = ? AND checked_in = 1;",
                (session_id,),
            )
            cnt_row = await cnt_cur.fetchone()
            await conn.commit()
            return False, -1, (int(cnt_row["cnt"]) if cnt_row else 1)

        raw_deck = sess_row["checkin_deck_json"]
        deck: list[int] | None = None
        if raw_deck:
            try:
                parsed = json.loads(raw_deck)
                if isinstance(parsed, list):
                    deck = parsed
            except Exception:
                deck = None

        last_idx = sess_row["last_checkin_idx"]
        chosen_idx, remaining_deck = draw_next_checkin_index(
            deck,
            int(last_idx) if last_idx is not None else None,
        )

        await conn.execute(
            """
            INSERT INTO deeptalk_participants (
                session_id, user_id, username, full_name,
                registered, registered_at, checked_in, checked_in_at, message_count
            ) VALUES (?, ?, ?, ?, 0, NULL, 1, ?, 0)
            ON CONFLICT(session_id, user_id) DO UPDATE SET
                username = excluded.username,
                full_name = excluded.full_name,
                checked_in = 1,
                checked_in_at = COALESCE(deeptalk_participants.checked_in_at, excluded.checked_in_at);
            """,
            (session_id, user_id, clean_u, clean_name, ts),
        )
        await conn.execute(
            "UPDATE deeptalk_sessions SET checkin_deck_json = ?, last_checkin_idx = ? WHERE id = ?;",
            (json.dumps(remaining_deck), chosen_idx, session_id),
        )
        cnt_cur = await conn.execute(
            "SELECT COUNT(*) AS cnt FROM deeptalk_participants WHERE session_id = ? AND checked_in = 1;",
            (session_id,),
        )
        cnt_row = await cnt_cur.fetchone()
        total_checked_in = int(cnt_row["cnt"]) if cnt_row else 1
        await conn.commit()

    return True, chosen_idx, total_checked_in


async def increment_deeptalk_message_count(
    *,
    session_id: int,
    user_id: int,
    username: str | None,
    full_name: str,
) -> None:
    """Increment live discussion message count for a participant in an active DeepTalk."""
    conn = get_db()
    lock = get_write_lock()
    clean_u = _clean_username(username)
    clean_name = (full_name or "").strip() or "کاربر"

    async with lock:
        await conn.execute(
            """
            INSERT INTO deeptalk_participants (
                session_id, user_id, username, full_name,
                registered, registered_at, checked_in, checked_in_at, message_count
            ) VALUES (?, ?, ?, ?, 0, NULL, 0, NULL, 1)
            ON CONFLICT(session_id, user_id) DO UPDATE SET
                username = COALESCE(excluded.username, deeptalk_participants.username),
                full_name = excluded.full_name,
                message_count = deeptalk_participants.message_count + 1;
            """,
            (session_id, user_id, clean_u, clean_name),
        )
        await conn.commit()


async def list_deeptalk_participants(session_id: int) -> list[dict[str, Any]]:
    """List all registered or checked-in participants for a DeepTalk session."""
    conn = get_db()
    cursor = await conn.execute(
        "SELECT session_id, user_id, username, full_name, registered, registered_at, "
        "checked_in, checked_in_at, message_count "
        "FROM deeptalk_participants "
        "WHERE session_id = ? AND (registered = 1 OR checked_in = 1) "
        "ORDER BY checked_in DESC, COALESCE(checked_in_at, registered_at) ASC, user_id ASC;",
        (session_id,),
    )
    rows = await cursor.fetchall()
    return [dict(r) for r in rows]


async def list_registered_deeptalk_participants(session_id: int) -> list[dict[str, Any]]:
    """Return all pre-registered participants (registered = 1) ordered by registered_at ASC."""
    conn = get_db()
    cursor = await conn.execute(
        "SELECT user_id, username, full_name, registered_at "
        "FROM deeptalk_participants "
        "WHERE session_id = ? AND registered = 1 "
        "ORDER BY registered_at ASC, user_id ASC;",
        (session_id,),
    )
    rows = await cursor.fetchall()
    return [dict(r) for r in rows]


async def list_deeptalk_dm_targets(session_id: int) -> list[dict[str, Any]]:
    """Return pre-registered participants who have started the bot in private chat (dm_started = 1)."""
    conn = get_db()
    cursor = await conn.execute(
        """
        SELECT p.user_id, p.username, p.full_name
        FROM deeptalk_participants p
        INNER JOIN bot_users u ON u.user_id = p.user_id
        WHERE p.session_id = ? AND p.registered = 1 AND u.dm_started = 1
        ORDER BY p.registered_at ASC;
        """,
        (session_id,),
    )
    rows = await cursor.fetchall()
    return [dict(r) for r in rows]


async def get_deeptalk_session_stats(session_id: int) -> dict[str, Any]:
    """Compute attendance, first-responder, message count, and top speakers for a DeepTalk session."""
    conn = get_db()
    agg_cur = await conn.execute(
        """
        SELECT
            COALESCE(SUM(CASE WHEN registered = 1 THEN 1 ELSE 0 END), 0) AS registered_count,
            COALESCE(SUM(CASE WHEN checked_in = 1 THEN 1 ELSE 0 END), 0) AS checked_in_count,
            COALESCE(SUM(CASE WHEN registered = 1 AND checked_in = 1 THEN 1 ELSE 0 END), 0) AS on_time_count,
            COALESCE(SUM(CASE WHEN registered = 0 AND checked_in = 1 THEN 1 ELSE 0 END), 0) AS walk_in_count,
            COALESCE(SUM(CASE WHEN registered = 1 AND checked_in = 0 THEN 1 ELSE 0 END), 0) AS missed_count,
            COALESCE(SUM(message_count), 0) AS total_messages
        FROM deeptalk_participants
        WHERE session_id = ?;
        """,
        (session_id,),
    )
    agg = await agg_cur.fetchone()

    first_cur = await conn.execute(
        "SELECT user_id, username, full_name, checked_in_at FROM deeptalk_participants "
        "WHERE session_id = ? AND checked_in = 1 "
        "ORDER BY checked_in_at ASC, user_id ASC LIMIT 1;",
        (session_id,),
    )
    first_row = await first_cur.fetchone()

    top_cur = await conn.execute(
        "SELECT user_id, username, full_name, message_count FROM deeptalk_participants "
        "WHERE session_id = ? AND message_count > 0 "
        "ORDER BY message_count DESC, checked_in DESC, user_id ASC LIMIT 3;",
        (session_id,),
    )
    top_rows = await top_cur.fetchall()

    return {
        "registered_count": int(agg["registered_count"]) if agg else 0,
        "checked_in_count": int(agg["checked_in_count"]) if agg else 0,
        "on_time_count": int(agg["on_time_count"]) if agg else 0,
        "walk_in_count": int(agg["walk_in_count"]) if agg else 0,
        "missed_count": int(agg["missed_count"]) if agg else 0,
        "total_messages": int(agg["total_messages"]) if agg else 0,
        "first_responder": dict(first_row) if first_row is not None else None,
        "top_speakers": [dict(r) for r in top_rows],
    }

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

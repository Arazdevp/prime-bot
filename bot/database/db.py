"""Database connection management and schema initialization."""

import asyncio
from pathlib import Path
import aiosqlite

_db_connection: aiosqlite.Connection | None = None
_write_lock: asyncio.Lock = asyncio.Lock()

INIT_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER NOT NULL,
    thread_id INTEGER NOT NULL,
    message_id INTEGER NOT NULL,
    user_id INTEGER,
    sender_name TEXT NOT NULL,
    text TEXT NOT NULL,
    media_type TEXT,
    reply_to_message_id INTEGER,
    reply_to_name TEXT,
    reply_to_snippet TEXT,
    created_at TEXT NOT NULL, /* ISO8601 UTC */
    is_summarized INTEGER NOT NULL DEFAULT 0,
    UNIQUE(chat_id, message_id)
);

CREATE INDEX IF NOT EXISTS idx_messages_chat_thread_sum_created 
    ON messages(chat_id, thread_id, is_summarized, created_at);

CREATE INDEX IF NOT EXISTS idx_messages_created_at 
    ON messages(created_at);

CREATE TABLE IF NOT EXISTS topics (
    chat_id INTEGER NOT NULL,
    thread_id INTEGER NOT NULL,
    name TEXT,
    is_active INTEGER,
    updated_at TEXT,
    PRIMARY KEY(chat_id, thread_id)
);

CREATE TABLE IF NOT EXISTS group_settings (
    chat_id INTEGER PRIMARY KEY,
    mode TEXT NOT NULL DEFAULT 'blacklist',
    destination_mode TEXT NOT NULL DEFAULT 'per_topic',
    destination_thread_id INTEGER,
    enabled INTEGER NOT NULL DEFAULT 1,
    min_messages INTEGER NOT NULL DEFAULT 5
);
"""


async def _apply_migrations(conn: aiosqlite.Connection) -> None:
    """Run database schema migrations based on PRAGMA user_version."""
    cursor = await conn.execute("PRAGMA user_version;")
    row = await cursor.fetchone()
    current_version = row[0] if row else 0

    if current_version < 1:
        cursor = await conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='topics';"
        )
        table_row = await cursor.fetchone()
        if table_row and table_row[0]:
            # Recreate topics table with nullable is_active column while preserving all data
            await conn.executescript("""
                CREATE TABLE IF NOT EXISTS topics_new (
                    chat_id INTEGER NOT NULL,
                    thread_id INTEGER NOT NULL,
                    name TEXT,
                    is_active INTEGER,
                    updated_at TEXT,
                    PRIMARY KEY(chat_id, thread_id)
                );
                INSERT INTO topics_new (chat_id, thread_id, name, is_active, updated_at)
                SELECT chat_id, thread_id, name, is_active, updated_at FROM topics;
                DROP TABLE topics;
                ALTER TABLE topics_new RENAME TO topics;
            """)
        await conn.execute("PRAGMA user_version = 1;")
        await conn.commit()


async def init_db(db_path: str = "data/bot.db") -> aiosqlite.Connection:
    """Initialize database directory, connection, PRAGMAs, and tables."""
    global _db_connection

    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if _db_connection is not None:
        try:
            await _db_connection.close()
        except Exception:
            pass

    conn = await aiosqlite.connect(str(path))
    conn.row_factory = aiosqlite.Row

    # Enforce SQLite PRAGMAs
    await conn.execute("PRAGMA journal_mode=WAL;")
    await conn.execute("PRAGMA synchronous=NORMAL;")
    await conn.execute("PRAGMA foreign_keys=ON;")

    # Apply schema scripts and migrations
    async with _write_lock:
        await conn.executescript(INIT_SCHEMA_SQL)
        await _apply_migrations(conn)
        await conn.commit()

    _db_connection = conn
    return _db_connection


def get_db() -> aiosqlite.Connection:
    """Return active shared database connection."""
    if _db_connection is None:
        raise RuntimeError("Database is not initialized. Call init_db() first.")
    return _db_connection


def get_write_lock() -> asyncio.Lock:
    """Return lock for serializing write transactions."""
    return _write_lock


async def close_db() -> None:
    """Close active database connection."""
    global _db_connection
    if _db_connection is not None:
        try:
            await _db_connection.close()
        finally:
            _db_connection = None

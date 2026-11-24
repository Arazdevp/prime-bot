"""Database connection management and schema initialization."""

import asyncio
from datetime import datetime, timezone
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

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS summaries (
    chat_id INTEGER NOT NULL,
    thread_id INTEGER NOT NULL,
    day TEXT NOT NULL,
    parts_json TEXT NOT NULL,
    sent_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    PRIMARY KEY(chat_id, thread_id, day)
);

CREATE TABLE IF NOT EXISTS bot_users (
    user_id INTEGER PRIMARY KEY,
    username TEXT,
    full_name TEXT,
    first_seen TEXT NOT NULL,
    last_seen TEXT NOT NULL,
    last_variant_key TEXT,
    attempts_window_start TEXT,
    attempts_count INTEGER NOT NULL DEFAULT 0,
    ignored_until TEXT,
    notified INTEGER NOT NULL DEFAULT 0,
    dm_started INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS known_groups (
    chat_id INTEGER PRIMARY KEY,
    title TEXT,
    is_forum INTEGER NOT NULL DEFAULT 1,
    added_at TEXT NOT NULL,
    left_at TEXT
);

CREATE TABLE IF NOT EXISTS timers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER NOT NULL,
    thread_id INTEGER NOT NULL DEFAULT 1,
    chat_type TEXT NOT NULL DEFAULT 'supergroup',
    user_id INTEGER NOT NULL,
    username TEXT,
    full_name TEXT NOT NULL,
    note TEXT,
    timer_type TEXT NOT NULL,
    fire_at TEXT NOT NULL,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    fired_at TEXT,
    source_message_id INTEGER
);

CREATE INDEX IF NOT EXISTS idx_timers_status_fire_at
    ON timers(status, fire_at);

CREATE INDEX IF NOT EXISTS idx_timers_user_status
    ON timers(user_id, status, fire_at);

CREATE TABLE IF NOT EXISTS morning_greetings (
    chat_id INTEGER PRIMARY KEY,
    enabled INTEGER NOT NULL DEFAULT 0,
    thread_id INTEGER,
    last_sent_day TEXT,
    deck_json TEXT,
    last_index INTEGER,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS map_chunk_cache (
    chat_id INTEGER NOT NULL,
    thread_id INTEGER NOT NULL,
    chunk_hash TEXT NOT NULL,
    summary_text TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY(chat_id, thread_id, chunk_hash)
);

CREATE INDEX IF NOT EXISTS idx_map_chunk_cache_created
    ON map_chunk_cache(created_at);

CREATE TABLE IF NOT EXISTS deeptalk_admins (
    user_id INTEGER PRIMARY KEY,
    username TEXT,
    full_name TEXT,
    added_by INTEGER NOT NULL,
    added_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS deeptalk_settings (
    chat_id INTEGER PRIMARY KEY,
    thread_id INTEGER,
    default_duration_minutes INTEGER NOT NULL DEFAULT 45,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS deeptalk_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER NOT NULL,
    thread_id INTEGER NOT NULL,
    subject TEXT NOT NULL,
    description TEXT,
    created_by INTEGER NOT NULL,
    creator_name TEXT NOT NULL,
    start_at TEXT NOT NULL,
    end_at TEXT NOT NULL,
    duration_minutes INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'scheduled',
    announce_message_id INTEGER,
    start_message_id INTEGER,
    end_message_id INTEGER,
    checkin_deck_json TEXT,
    last_checkin_idx INTEGER,
    created_at TEXT NOT NULL,
    started_at TEXT,
    ended_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_deeptalk_sessions_status_times
    ON deeptalk_sessions(status, start_at, end_at);

CREATE INDEX IF NOT EXISTS idx_deeptalk_sessions_chat_status
    ON deeptalk_sessions(chat_id, status, start_at);

CREATE TABLE IF NOT EXISTS deeptalk_participants (
    session_id INTEGER NOT NULL,
    user_id INTEGER NOT NULL,
    username TEXT,
    full_name TEXT NOT NULL,
    registered INTEGER NOT NULL DEFAULT 0,
    registered_at TEXT,
    checked_in INTEGER NOT NULL DEFAULT 0,
    checked_in_at TEXT,
    message_count INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY(session_id, user_id),
    FOREIGN KEY(session_id) REFERENCES deeptalk_sessions(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_deeptalk_participants_session
    ON deeptalk_participants(session_id, checked_in, registered);

CREATE TABLE IF NOT EXISTS ai_chat_memory_sources (
    chat_id INTEGER NOT NULL,
    thread_id INTEGER NOT NULL,
    topic_name TEXT,
    last_learned_message_id INTEGER NOT NULL DEFAULT 0,
    last_learned_at TEXT,
    created_at TEXT NOT NULL,
    PRIMARY KEY(chat_id, thread_id)
);

CREATE TABLE IF NOT EXISTS ai_member_profiles (
    chat_id INTEGER NOT NULL,
    user_id INTEGER NOT NULL,
    display_name TEXT NOT NULL,
    personality_summary TEXT NOT NULL,
    speaking_style TEXT,
    interests_and_topics TEXT,
    notable_facts TEXT,
    message_count INTEGER NOT NULL DEFAULT 0,
    last_seen_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY(chat_id, user_id)
);

CREATE INDEX IF NOT EXISTS idx_ai_member_profiles_chat_updated
    ON ai_member_profiles(chat_id, message_count DESC, updated_at DESC);

CREATE TABLE IF NOT EXISTS ai_group_memory (
    chat_id INTEGER PRIMARY KEY,
    group_title TEXT,
    culture_summary TEXT NOT NULL DEFAULT '',
    inside_jokes TEXT NOT NULL DEFAULT '',
    ongoing_topics TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL
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

    if current_version < 2:
        await conn.executescript("""
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS summaries (
                chat_id INTEGER NOT NULL,
                thread_id INTEGER NOT NULL,
                day TEXT NOT NULL,
                parts_json TEXT NOT NULL,
                sent_count INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                PRIMARY KEY(chat_id, thread_id, day)
            );
        """)
        await conn.execute("PRAGMA user_version = 2;")
        await conn.commit()

    if current_version < 3:
        await conn.executescript("""
            CREATE TABLE IF NOT EXISTS bot_users (
                user_id INTEGER PRIMARY KEY,
                username TEXT,
                full_name TEXT,
                first_seen TEXT NOT NULL,
                last_seen TEXT NOT NULL,
                last_variant_key TEXT,
                attempts_window_start TEXT,
                attempts_count INTEGER NOT NULL DEFAULT 0,
                ignored_until TEXT,
                notified INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS known_groups (
                chat_id INTEGER PRIMARY KEY,
                title TEXT,
                is_forum INTEGER NOT NULL DEFAULT 1,
                added_at TEXT NOT NULL,
                left_at TEXT
            );
        """)
        now_utc = datetime.now(timezone.utc).isoformat()
        await conn.execute(
            """
            INSERT OR IGNORE INTO known_groups (chat_id, title, is_forum, added_at, left_at)
            SELECT chat_id, NULL, 1, ?, NULL FROM group_settings;
            """,
            (now_utc,),
        )
        await conn.execute("PRAGMA user_version = 3;")
        await conn.commit()

    if current_version < 4:
        await conn.executescript("""
            CREATE TABLE IF NOT EXISTS timers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER NOT NULL,
                thread_id INTEGER NOT NULL DEFAULT 1,
                chat_type TEXT NOT NULL DEFAULT 'supergroup',
                user_id INTEGER NOT NULL,
                username TEXT,
                full_name TEXT NOT NULL,
                note TEXT,
                timer_type TEXT NOT NULL,
                fire_at TEXT NOT NULL,
                created_at TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'active',
                fired_at TEXT,
                source_message_id INTEGER
            );
            CREATE INDEX IF NOT EXISTS idx_timers_status_fire_at
                ON timers(status, fire_at);
            CREATE INDEX IF NOT EXISTS idx_timers_user_status
                ON timers(user_id, status, fire_at);
        """)
        cursor = await conn.execute("PRAGMA table_info(bot_users);")
        cols = await cursor.fetchall()
        col_names = {c[1] for c in cols}
        if "dm_started" not in col_names:
            await conn.execute(
                "ALTER TABLE bot_users ADD COLUMN dm_started INTEGER NOT NULL DEFAULT 0;"
            )
        await conn.execute("PRAGMA user_version = 4;")
        await conn.commit()

    if current_version < 5:
        await conn.executescript("""
            CREATE TABLE IF NOT EXISTS morning_greetings (
                chat_id INTEGER PRIMARY KEY,
                enabled INTEGER NOT NULL DEFAULT 0,
                thread_id INTEGER,
                last_sent_day TEXT,
                deck_json TEXT,
                last_index INTEGER,
                updated_at TEXT NOT NULL
            );
        """)
        await conn.execute("PRAGMA user_version = 5;")
        await conn.commit()

    if current_version < 6:
        await conn.executescript("""
            CREATE TABLE IF NOT EXISTS map_chunk_cache (
                chat_id INTEGER NOT NULL,
                thread_id INTEGER NOT NULL,
                chunk_hash TEXT NOT NULL,
                summary_text TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY(chat_id, thread_id, chunk_hash)
            );
            CREATE INDEX IF NOT EXISTS idx_map_chunk_cache_created
                ON map_chunk_cache(created_at);
        """)
        await conn.execute("PRAGMA user_version = 6;")
        await conn.commit()

    if current_version < 7:
        await conn.executescript("""
            CREATE TABLE IF NOT EXISTS deeptalk_admins (
                user_id INTEGER PRIMARY KEY,
                username TEXT,
                full_name TEXT,
                added_by INTEGER NOT NULL,
                added_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS deeptalk_settings (
                chat_id INTEGER PRIMARY KEY,
                thread_id INTEGER,
                default_duration_minutes INTEGER NOT NULL DEFAULT 45,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS deeptalk_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER NOT NULL,
                thread_id INTEGER NOT NULL,
                subject TEXT NOT NULL,
                description TEXT,
                created_by INTEGER NOT NULL,
                creator_name TEXT NOT NULL,
                start_at TEXT NOT NULL,
                end_at TEXT NOT NULL,
                duration_minutes INTEGER NOT NULL,
                status TEXT NOT NULL DEFAULT 'scheduled',
                announce_message_id INTEGER,
                start_message_id INTEGER,
                end_message_id INTEGER,
                checkin_deck_json TEXT,
                last_checkin_idx INTEGER,
                created_at TEXT NOT NULL,
                started_at TEXT,
                ended_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_deeptalk_sessions_status_times
                ON deeptalk_sessions(status, start_at, end_at);
            CREATE INDEX IF NOT EXISTS idx_deeptalk_sessions_chat_status
                ON deeptalk_sessions(chat_id, status, start_at);
            CREATE TABLE IF NOT EXISTS deeptalk_participants (
                session_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                username TEXT,
                full_name TEXT NOT NULL,
                registered INTEGER NOT NULL DEFAULT 0,
                registered_at TEXT,
                checked_in INTEGER NOT NULL DEFAULT 0,
                checked_in_at TEXT,
                message_count INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY(session_id, user_id),
                FOREIGN KEY(session_id) REFERENCES deeptalk_sessions(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_deeptalk_participants_session
                ON deeptalk_participants(session_id, checked_in, registered);
        """)
        await conn.execute("PRAGMA user_version = 7;")
        await conn.commit()

    if current_version < 8:
        await conn.executescript("""
            CREATE TABLE IF NOT EXISTS ai_chat_memory_sources (
                chat_id INTEGER NOT NULL,
                thread_id INTEGER NOT NULL,
                topic_name TEXT,
                last_learned_message_id INTEGER NOT NULL DEFAULT 0,
                last_learned_at TEXT,
                created_at TEXT NOT NULL,
                PRIMARY KEY(chat_id, thread_id)
            );
            CREATE TABLE IF NOT EXISTS ai_member_profiles (
                chat_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                display_name TEXT NOT NULL,
                personality_summary TEXT NOT NULL,
                speaking_style TEXT,
                interests_and_topics TEXT,
                notable_facts TEXT,
                message_count INTEGER NOT NULL DEFAULT 0,
                last_seen_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(chat_id, user_id)
            );
            CREATE INDEX IF NOT EXISTS idx_ai_member_profiles_chat_updated
                ON ai_member_profiles(chat_id, message_count DESC, updated_at DESC);
            CREATE TABLE IF NOT EXISTS ai_group_memory (
                chat_id INTEGER PRIMARY KEY,
                group_title TEXT,
                culture_summary TEXT NOT NULL DEFAULT '',
                inside_jokes TEXT NOT NULL DEFAULT '',
                ongoing_topics TEXT NOT NULL DEFAULT '',
                updated_at TEXT NOT NULL
            );
        """)
        await conn.execute("PRAGMA user_version = 8;")
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
    await conn.execute("PRAGMA busy_timeout=5000;")

    # Apply schema scripts and migrations
    async with _write_lock:
        await conn.executescript(INIT_SCHEMA_SQL)
        await _apply_migrations(conn)
        await conn.commit()

    _db_connection = conn
    return _db_connection


def is_db_initialized() -> bool:
    """Return True if the shared database connection is currently active."""
    return _db_connection is not None


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

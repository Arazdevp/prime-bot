# CLAUDE.md — Telegram Forum Daily Summarizer Bot Guide

This file provides architectural context, commands, and operational guidelines for AI coding assistants (such as Claude Code, Cursor, and Cline) working on this codebase.

---

## 1. Project Overview

A production-ready Telegram bot tailored for **Forum Supergroups (topic-enabled groups)**.
Telegram Bot API does not permit querying past message history, so this bot listens to messages **in real time**, stores them in an async SQLite database, and every night at **00:01 Asia/Tehran** (or via manual admin trigger), summarizes each topic's discussions in fluent **Persian** using LLM providers (Google Gemini or OpenAI-compatible gateways like 9Router).

- **Repository:** [https://github.com/Arazdevp/prime-bot](https://github.com/Arazdevp/prime-bot)
- **Primary Language:** Python 3.11+
- **Frameworks:** `aiogram 3.x`, `aiosqlite`, `APScheduler`, `pydantic-settings`, `httpx[socks]`, `aiohttp-socks`, `jdatetime`

---

## 2. Common Developer Commands

```bash
# Activate virtual environment
source .venv/bin/activate       # Linux/macOS
.venv\Scripts\activate          # Windows PowerShell

# Install dependencies
pip install -r requirements.txt

# Run the bot
python -m bot.main

# Run test suite
python -m pytest -v

# Syntax verification across all Python files
python -m py_compile $(git ls-files '*.py')   # Linux
python -m py_compile (Get-ChildItem -Recurse -Filter *.py).FullName # Windows
```

---

## 3. Architecture & Core Modules

```
tg-daily-summarizer/
├── bot/
│   ├── main.py              # Lifecycle entrypoint, session creation with proxy, shutdown
│   ├── config.py            # Pydantic Settings with multi-path .env resolution
│   ├── filters.py           # AdminFilter (creator/administrator/anonymous/ADMIN_IDS), ForumGroupFilter
│   ├── database/
│   │   ├── db.py            # aiosqlite connection with PRAGMA WAL, foreign keys, write lock
│   │   └── repository.py    # All parameterized async queries (messages, topics, settings)
│   ├── handlers/
│   │   ├── listener.py      # Captures text, captions, replies, topic creation/rename
│   │   └── admin.py         # Full Persian admin command suite (/status, /topics, /mode, etc.)
│   ├── services/
│   │   ├── llm_client.py    # BaseLLMClient, GeminiClient (REST) & OpenAIClient (for 9Router)
│   │   ├── summarizer.py    # Transcript formatting, map-reduce chunking, Telegram HTML sanitize
│   │   └── publisher.py     # Safe sender with smart tag-aware splitting and parse error recovery
│   ├── scheduler/
│   │   └── jobs.py          # APScheduler nightly job (00:01 Tehran), daily retention cleanup
│   └── utils/
│       ├── general_topic.py # Normalizes General topic (None/1 -> 1; send target -> None)
│       ├── noise_filter.py  # Strips greetings, filler, emojis, punctuation, normalizes ي/ك
│       ├── text_split.py    # Tag-aware smart message chunker (<= 4000 chars)
│       ├── dates.py         # Jalali (jdatetime) & Gregorian helpers in Asia/Tehran
│       ├── retry.py         # Async exponential backoff with jitter on 429/5xx/network errors
│       └── logging_setup.py # Rotating UTF-8 file handler (5MB x 5) & console handler
├── tests/                   # Complete offline unit tests (17 passing tests)
├── deploy/                  # Linux systemd unit (tg-summarizer.service)
├── .env.example             # Documented template with all 22 environment keys
└── README.md                # Comprehensive Persian documentation
```

---

## 4. Key Implementation Rules

1. **Strict Async:** Never introduce blocking I/O (no `requests`, `time.sleep`, or standard `sqlite3`). Use `asyncio.sleep`, `aiosqlite`, `httpx.AsyncClient`.
2. **Database Concurrency:** All write operations in `bot/database/repository.py` MUST be guarded by `async with get_write_lock():`. All SQL queries MUST be parameterized (never format strings into SQL).
3. **General Topic Normalization:**
   - In Telegram, the General topic thread ID can be `None` or `1`. Always normalize both to `1` using `bot.utils.general_topic.resolve_thread_id`.
   - When sending to General via Telegram Bot API, use `message_thread_id=None` using `bot.utils.general_topic.to_send_thread_id`.
4. **HTML Tag Balance:** Telegram parse mode `HTML` rejects malformed or unclosed tags. `bot/utils/text_split.py` ensures any split inside open tags closes them at the chunk boundary and reopens them in the next chunk.
5. **Map-Reduce Transcript Chunking:** When transcripts exceed `CHUNK_MAX_CHARS` (24,000), lines are chunked cleanly and processed via `SYSTEM_PROMPT_CHUNK` then merged with `SYSTEM_PROMPT_MERGE`.
6. **Languages:**
   - All code, identifiers, comments, and docstrings MUST be in **English**.
   - All user-facing bot messages, admin replies, and AI system prompts MUST be in fluent **Persian**.

---

## 5. Configuration & Environment (`.env`)

### Telegram Connection:
- `BOT_TOKEN`: Required Telegram bot token from `@BotFather`.
- `PROXY_URL`: If running in Iran or behind a firewall, set `PROXY_URL` (e.g. `socks5://127.0.0.1:10808` or `http://127.0.0.1:7890`). This is automatically passed to both `AiohttpSession` (via `aiohttp-socks`) and `httpx.AsyncClient`.

### AI Providers:
- **Google Gemini (Direct API):**
  ```ini
  AI_PROVIDER=gemini
  GEMINI_API_KEY=AIzaSy...
  GEMINI_MODEL=gemini-2.5-flash
  ```
- **9Router / OpenAI-Compatible (Local Gateway for Gemini 3.8):**
  ```ini
  AI_PROVIDER=openai
  OPENAI_BASE_URL=http://localhost:20128/v1
  OPENAI_API_KEY=local-9router-key
  OPENAI_MODEL=gemini-3.8-flash
  ```

---

## 6. Troubleshooting Common Issues

- **`TokenValidationError: Token is invalid!`**
  The `.env` file still contains placeholder text like `YOUR_TELEGRAM_BOT_TOKEN_HERE`. Put your real token from `@BotFather`.
- **`ClientConnectorError: Cannot connect to host api.telegram.org:443 ssl:default`**
  Telegram is filtered in your network. Set `PROXY_URL=socks5://127.0.0.1:PORT` (v2ray / clash / sing-box) in `.env`.
- **`RuntimeError: Polling is not started` on shutdown**
  Handled gracefully in `bot/main.py` inside the `finally` block.

"""Interactive AI chat service: persistent toggle cache, trigger matching, and multi-turn Gemini memory."""

import asyncio
from dataclasses import dataclass, field
import logging
import re
import time
from aiogram import Bot
from aiogram.types import Message
from bot.config import get_settings
from bot.database import repository
from bot.services.llm_client import GeminiClient, LLMError, get_ai_chat_client
from bot.services.summarizer import sanitize_telegram_html
from bot.services.timer_service import get_bot_username
from bot.texts.ai_chat_texts import DEFAULT_AI_CHAT_SYSTEM_PROMPT, build_runtime_system_prompt
from bot.utils.html_utils import escape_html
from bot.utils.retry import async_retry

logger = logging.getLogger(__name__)

CONVERSATION_IDLE_TTL_SECONDS = 7200.0  # 2 hours


@dataclass
class _ConversationState:
    """Rolling multi-turn message buffer for a single chat or topic."""

    messages: list[dict[str, str]] = field(default_factory=list)
    updated_at: float = field(default_factory=time.monotonic)


_ai_chat_enabled_cache: bool = False
_bot_id_cache: int | None = None
_bot_username_cache: str | None = None
_conversations: dict[tuple[int, int], _ConversationState] = {}
_conv_locks: dict[tuple[int, int], asyncio.Lock] = {}


def reset_ai_chat_state() -> None:
    """Reset in-memory AI chat toggle, identity, and conversation caches (for test isolation)."""
    global _ai_chat_enabled_cache, _bot_id_cache, _bot_username_cache
    _ai_chat_enabled_cache = False
    _bot_id_cache = None
    _bot_username_cache = None
    _conversations.clear()
    _conv_locks.clear()


def is_ai_chat_enabled_cached() -> bool:
    """Fast O(1) read of the cached AI chat enabled flag."""
    return _ai_chat_enabled_cache


async def init_ai_chat_state() -> bool:
    """Load persisted ai_chat_enabled flag from SQLite into the in-memory cache."""
    global _ai_chat_enabled_cache
    _ai_chat_enabled_cache = await repository.get_ai_chat_enabled()
    return _ai_chat_enabled_cache


async def set_ai_chat_state(enabled: bool) -> bool:
    """Persist ai_chat_enabled flag to SQLite and update the in-memory cache."""
    global _ai_chat_enabled_cache
    await repository.set_ai_chat_enabled(enabled)
    _ai_chat_enabled_cache = bool(enabled)
    return _ai_chat_enabled_cache


def set_ai_bot_identity(bot_id: int | None, username: str | None) -> None:
    """Cache the bot's Telegram user ID and username for group reply/mention detection."""
    global _bot_id_cache, _bot_username_cache
    if isinstance(bot_id, int) and bot_id > 0:
        _bot_id_cache = bot_id
    if isinstance(username, str) and username.strip():
        _bot_username_cache = username.strip().lstrip("@")


async def ensure_ai_bot_identity(bot: Bot) -> tuple[int | None, str | None]:
    """Return cached (bot_id, bot_username), lazily resolving via bot.id / bot.get_me()."""
    global _bot_id_cache, _bot_username_cache
    if _bot_id_cache is None:
        raw_id = getattr(bot, "id", None)
        if isinstance(raw_id, int) and raw_id > 0:
            _bot_id_cache = raw_id
    if _bot_username_cache is None:
        cached_uname = get_bot_username()
        if cached_uname:
            _bot_username_cache = cached_uname

    if _bot_id_cache is None or _bot_username_cache is None:
        try:
            me = await bot.get_me()
            if me is not None:
                set_ai_bot_identity(getattr(me, "id", None), getattr(me, "username", None))
        except Exception as exc:
            logger.debug("Could not resolve bot identity via get_me(): %s", exc)

    return _bot_id_cache, _bot_username_cache


def is_message_addressing_bot(
    message: Message,
    bot_id: int | None,
    bot_username: str | None,
) -> bool:
    """Return True if message replies to the bot or explicitly mentions @bot_username."""
    reply = getattr(message, "reply_to_message", None)
    if reply is not None and not getattr(reply, "forum_topic_created", None):
        reply_from = getattr(reply, "from_user", None)
        if reply_from is not None and bot_id is not None and getattr(reply_from, "id", None) == bot_id:
            return True

    if bot_username:
        raw_text = getattr(message, "text", None) or getattr(message, "caption", None) or ""
        pattern = rf"(?<![A-Za-z0-9_])@{re.escape(bot_username)}(?![A-Za-z0-9_])"
        if re.search(pattern, raw_text, flags=re.IGNORECASE):
            return True

    return False


def strip_bot_mention(text: str, bot_username: str | None) -> str:
    """Remove @bot_username mention from user message and collapse extra spaces."""
    cleaned = (text or "").strip()
    if bot_username:
        pattern = rf"(?<![A-Za-z0-9_])@{re.escape(bot_username)}(?![A-Za-z0-9_])"
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()
    return cleaned


def _prune_stale_conversations(now: float | None = None) -> None:
    """Evict idle conversations older than CONVERSATION_IDLE_TTL_SECONDS."""
    current = now if now is not None else time.monotonic()
    expired = [
        key
        for key, state in _conversations.items()
        if (current - state.updated_at) > CONVERSATION_IDLE_TTL_SECONDS
    ]
    for key in expired:
        _conversations.pop(key, None)
        _conv_locks.pop(key, None)


def clear_conversation_history(chat_id: int | None = None, thread_id: int | None = None) -> int:
    """Clear in-memory AI chat history for a single chat/topic or across all conversations."""
    if chat_id is None:
        count = len(_conversations)
        _conversations.clear()
        _conv_locks.clear()
        return count

    if thread_id is not None:
        existed = _conversations.pop((chat_id, thread_id), None)
        return 1 if existed is not None else 0

    keys = [k for k in _conversations if k[0] == chat_id]
    for k in keys:
        _conversations.pop(k, None)
    return len(keys)


def count_active_conversations() -> int:
    """Return number of currently cached non-expired conversation threads."""
    _prune_stale_conversations()
    return len(_conversations)


def _get_conv_lock(key: tuple[int, int]) -> asyncio.Lock:
    lock = _conv_locks.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _conv_locks[key] = lock
    return lock


def _normalize_turns(turns: list[dict[str, str]], max_turns: int) -> list[dict[str, str]]:
    """Trim history to max_turns while ensuring the sequence starts with a 'user' turn."""
    trimmed = list(turns[-max_turns:]) if max_turns > 0 else list(turns)
    while trimmed and trimmed[0].get("role") != "user":
        trimmed.pop(0)
    return trimmed


async def generate_ai_chat_reply(
    *,
    chat_id: int,
    thread_id: int,
    chat_type: str,
    speaker_name: str,
    user_text: str,
    topic_name: str | None = None,
    reply_context: str | None = None,
    client: GeminiClient | None = None,
) -> str:
    """Generate a natural Iranian Persian conversational reply and update rolling memory."""
    settings = get_settings()
    active_client = client or get_ai_chat_client(settings)
    key = (chat_id, thread_id)
    lock = _get_conv_lock(key)

    async with lock:
        now = time.monotonic()
        _prune_stale_conversations(now)
        state = _conversations.get(key)
        if state is None:
            state = _ConversationState(messages=[], updated_at=now)
            _conversations[key] = state

        prompt_input = user_text.strip()
        if reply_context and reply_context.strip():
            prompt_input = f"[در پاسخ به پیام قبلی: «{reply_context.strip()}»]\n{prompt_input}"

        history_limit = max(2, int(settings.AI_CHAT_HISTORY_LIMIT))
        pending_turns = _normalize_turns(
            [*state.messages, {"role": "user", "text": prompt_input}],
            max_turns=history_limit,
        )

        custom_prompt = await repository.get_ai_chat_custom_prompt()
        base_prompt = custom_prompt or DEFAULT_AI_CHAT_SYSTEM_PROMPT
        runtime_system = build_runtime_system_prompt(
            base_prompt,
            speaker_name=speaker_name,
            chat_type=chat_type,
            topic_name=topic_name,
        )

        raw_fallbacks = getattr(settings, "AI_CHAT_FALLBACK_MODELS", None) or []
        candidate_models: list[str] = []
        for candidate in [settings.AI_CHAT_MODEL, *raw_fallbacks]:
            cleaned = str(candidate).strip()
            if cleaned and cleaned not in candidate_models:
                candidate_models.append(cleaned)
        if not candidate_models:
            candidate_models = ["gemini-3.5-flash-lite"]

        raw_reply: str | None = None
        for idx, candidate_model in enumerate(candidate_models):
            is_last = idx == len(candidate_models) - 1
            try:
                raw_reply = await async_retry(
                    lambda m=candidate_model: active_client.generate_chat(
                        system_prompt=runtime_system,
                        messages=pending_turns,
                        temperature=settings.AI_CHAT_TEMPERATURE,
                        max_output_tokens=settings.AI_CHAT_MAX_OUTPUT_TOKENS,
                        model=m,
                    ),
                    retries=1 if is_last else 0,
                )
                break
            except LLMError as exc:
                if not is_last and exc.status_code in (404, 429, 503):
                    logger.warning(
                        "AI chat model '%s' failed (status=%s); falling back to next model...",
                        candidate_model,
                        exc.status_code,
                    )
                    continue
                raise

        if raw_reply is None:
            raise LLMError("All AI chat model candidates failed", status_code=503, is_retryable=True)

        sanitized = sanitize_telegram_html(raw_reply)
        if not sanitized:
            sanitized = escape_html(raw_reply.strip())

        state.messages = _normalize_turns(
            [
                *state.messages,
                {"role": "user", "text": prompt_input},
                {"role": "model", "text": sanitized},
            ],
            max_turns=history_limit,
        )
        state.updated_at = time.monotonic()
        return sanitized

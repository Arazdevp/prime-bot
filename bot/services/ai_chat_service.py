"""Interactive AI chat service: persistent toggle cache, trigger matching, and multi-turn Gemini memory."""

import asyncio
from collections import deque
from dataclasses import dataclass, field
import logging
import re
import time
from aiogram import Bot
from aiogram.types import Message
from bot.config import (
    get_settings,
    is_openai_compatible_provider,
    normalize_ai_provider,
    resolve_openai_chat_fallback_models,
    resolve_openai_compat_config,
)
from bot.database import repository
from bot.services.access import is_global_admin
from bot.services.ai_media_reaction_service import (
    ParsedAiDirectives,
    canonical_media_tag,
    format_directive_history_placeholder,
    get_ai_chat_media_counts_cached,
    init_ai_media_state,
    parse_ai_chat_directives,
    reset_ai_media_caches,
)
from bot.services.ai_memory_service import build_live_and_learned_context, reset_ai_memory_caches
from bot.services.llm_client import GeminiClient, LLMError, OpenAIClient, get_ai_chat_client
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
_ai_chat_provider_cache: str | None = None
_ai_chat_access_mode_cache: str = "public"
_ai_chat_user_limit_cache: int | None = None
_bot_id_cache: int | None = None
_bot_username_cache: str | None = None
_conversations: dict[tuple[int, int], _ConversationState] = {}
_conv_locks: dict[tuple[int, int], asyncio.Lock] = {}
_user_quota_timestamps: dict[int, deque[float]] = {}
_user_quota_notice_ts: dict[int, float] = {}


def reset_ai_chat_state() -> None:
    """Reset in-memory AI chat toggle, provider, access mode, quotas, and conversation caches."""
    global _ai_chat_enabled_cache, _ai_chat_provider_cache, _ai_chat_access_mode_cache
    global _ai_chat_user_limit_cache, _bot_id_cache, _bot_username_cache
    _ai_chat_enabled_cache = False
    _ai_chat_provider_cache = None
    _ai_chat_access_mode_cache = "public"
    _ai_chat_user_limit_cache = None
    _bot_id_cache = None
    _bot_username_cache = None
    _conversations.clear()
    _conv_locks.clear()
    _user_quota_timestamps.clear()
    _user_quota_notice_ts.clear()
    reset_ai_memory_caches()
    reset_ai_media_caches()


def is_ai_chat_enabled_cached() -> bool:
    """Fast O(1) read of the cached AI chat enabled flag."""
    return _ai_chat_enabled_cache


def get_effective_ai_chat_provider() -> str:
    """Return active interactive AI chat provider ('9router', 'openai', or 'gemini')."""
    if _ai_chat_provider_cache in ("9router", "openai", "gemini"):
        return _ai_chat_provider_cache
    raw = getattr(get_settings(), "AI_CHAT_PROVIDER", "9router")
    normalized = normalize_ai_provider(raw, default="9router")
    return normalized if normalized in ("9router", "openai", "gemini") else "9router"


def get_effective_ai_chat_access_mode() -> str:
    """Return active AI chat access mode ('public' or 'owner_only')."""
    return _ai_chat_access_mode_cache if _ai_chat_access_mode_cache in ("public", "owner_only") else "public"


async def init_ai_chat_state() -> bool:
    """Load persisted AI chat toggle, provider, access mode, and user quota limit from SQLite."""
    global _ai_chat_enabled_cache, _ai_chat_provider_cache, _ai_chat_access_mode_cache
    global _ai_chat_user_limit_cache
    _ai_chat_enabled_cache = await repository.get_ai_chat_enabled()
    _ai_chat_provider_cache = await repository.get_ai_chat_provider()
    _ai_chat_access_mode_cache = await repository.get_ai_chat_access_mode()
    _ai_chat_user_limit_cache = await repository.get_ai_chat_user_limit()
    await init_ai_media_state()
    return _ai_chat_enabled_cache


async def set_ai_chat_state(enabled: bool) -> bool:
    """Persist ai_chat_enabled flag to SQLite and update the in-memory cache."""
    global _ai_chat_enabled_cache
    await repository.set_ai_chat_enabled(enabled)
    _ai_chat_enabled_cache = bool(enabled)
    return _ai_chat_enabled_cache


async def set_ai_chat_provider_state(provider: str) -> str:
    """Persist ai_chat_provider ('9router', 'openai', or 'gemini') to SQLite and update cache."""
    global _ai_chat_provider_cache
    cleaned = normalize_ai_provider(provider, default="")
    if cleaned not in ("9router", "openai", "gemini"):
        raise ValueError(f"Unsupported AI chat provider: {provider}")
    await repository.set_ai_chat_provider(cleaned)
    _ai_chat_provider_cache = cleaned
    return cleaned


async def set_ai_chat_access_mode_state(mode: str) -> str:
    """Persist ai_chat_access_mode ('public' or 'owner_only') to SQLite and update cache."""
    global _ai_chat_access_mode_cache
    cleaned = mode.strip().lower()
    if cleaned not in ("public", "owner_only"):
        raise ValueError(f"Unsupported AI chat access mode: {mode}")
    await repository.set_ai_chat_access_mode(cleaned)
    _ai_chat_access_mode_cache = cleaned
    return cleaned


def get_effective_ai_chat_user_limit() -> int:
    """Return active regular-user AI chat limit (-1=unlimited, 0=closed, >=1=per-window cap)."""
    if _ai_chat_user_limit_cache is not None:
        return _ai_chat_user_limit_cache
    return max(1, int(getattr(get_settings(), "AI_CHAT_USER_LIMIT_COUNT", 20)))


async def set_ai_chat_user_limit_state(limit: int | None) -> int:
    """Persist regular-user AI chat quota limit (-1=unlimited, 0=closed, 1..10000) and update cache."""
    global _ai_chat_user_limit_cache
    if limit is not None:
        parsed = int(limit)
        if not (parsed == -1 or 0 <= parsed <= repository.AI_CHAT_USER_LIMIT_MAX):
            raise ValueError(f"Unsupported AI chat user limit: {limit}")
        limit = parsed
    await repository.set_ai_chat_user_limit(limit)
    _ai_chat_user_limit_cache = limit
    _user_quota_notice_ts.clear()
    return get_effective_ai_chat_user_limit()


def count_active_quota_users(*, now: float | None = None) -> int:
    """Prune expired quota timestamps and return count of regular users with active usage."""
    settings = get_settings()
    window_sec = max(60, int(getattr(settings, "AI_CHAT_USER_LIMIT_WINDOW_SECONDS", 7200)))
    current = now if now is not None else time.monotonic()
    cutoff = current - window_sec
    stale_users: list[int] = []
    for uid, dq in _user_quota_timestamps.items():
        while dq and dq[0] <= cutoff:
            dq.popleft()
        if not dq:
            stale_users.append(uid)
    for uid in stale_users:
        _user_quota_timestamps.pop(uid, None)
    return len(_user_quota_timestamps)


def reset_all_user_quotas(*, now: float | None = None) -> int:
    """Clear all non-manager AI chat sliding-window quotas and notice cooldowns."""
    active_count = count_active_quota_users(now=now)
    _user_quota_timestamps.clear()
    _user_quota_notice_ts.clear()
    return active_count


def check_and_consume_user_quota(
    user_id: int,
    *,
    now: float | None = None,
) -> tuple[bool, int, int]:
    """Enforce sliding-window AI chat quota for non-managers.

    Returns (allowed, remaining_count_or_-1_for_unlimited, wait_minutes_if_blocked_or_-1_if_closed).
    """
    if is_global_admin(user_id):
        return True, -1, 0

    limit_count = get_effective_ai_chat_user_limit()
    if limit_count < 0:
        return True, -1, 0
    if limit_count == 0:
        return False, 0, -1

    settings = get_settings()
    window_sec = max(60, int(getattr(settings, "AI_CHAT_USER_LIMIT_WINDOW_SECONDS", 7200)))
    current = now if now is not None else time.monotonic()

    dq = _user_quota_timestamps.get(user_id)
    if dq is None:
        dq = deque()
        _user_quota_timestamps[user_id] = dq

    cutoff = current - window_sec
    while dq and dq[0] <= cutoff:
        dq.popleft()

    if len(dq) >= limit_count:
        wait_sec = max(1.0, window_sec - (current - dq[0]))
        wait_mins = max(1, int((wait_sec + 59) // 60))
        return False, 0, wait_mins

    dq.append(current)
    return True, limit_count - len(dq), 0


def refund_user_quota(user_id: int) -> None:
    """Refund the most recently consumed quota slot when an LLM reply fails."""
    if is_global_admin(user_id) or get_effective_ai_chat_user_limit() <= 0:
        return
    dq = _user_quota_timestamps.get(user_id)
    if dq:
        dq.pop()


def should_send_quota_notice(
    user_id: int,
    *,
    cooldown_seconds: float = 600.0,
    now: float | None = None,
) -> bool:
    """Allow at most one rate-limit notice per user within cooldown_seconds to avoid spam."""
    current = now if now is not None else time.monotonic()
    last_ts = _user_quota_notice_ts.get(user_id)
    if last_ts is not None and (current - last_ts) < cooldown_seconds:
        return False
    _user_quota_notice_ts[user_id] = current
    return True


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
    speaker_user_id: int | None = None,
    topic_name: str | None = None,
    reply_context: str | None = None,
    client: GeminiClient | OpenAIClient | None = None,
) -> str:
    """Generate a natural Iranian Persian conversational reply and update rolling memory."""
    settings = get_settings()
    provider = get_effective_ai_chat_provider()
    active_client = client or get_ai_chat_client(settings, provider=provider)
    key = (chat_id, thread_id)
    lock = _get_conv_lock(key)

    async with lock:
        now = time.monotonic()
        _prune_stale_conversations(now)
        state = _conversations.get(key)
        existing_messages = state.messages if state is not None else []

        prompt_input = user_text.strip()
        if reply_context and reply_context.strip():
            prompt_input = f"[در پاسخ به پیام قبلی: «{reply_context.strip()}»]\n{prompt_input}"
        if chat_type != "private":
            clean_speaker = (speaker_name or "عضو").strip() or "عضو"
            prompt_input = f"{clean_speaker}: {prompt_input}"

        history_limit = max(2, int(settings.AI_CHAT_HISTORY_LIMIT))
        pending_turns = _normalize_turns(
            [*existing_messages, {"role": "user", "text": prompt_input}],
            max_turns=history_limit,
        )

        custom_prompt = await repository.get_ai_chat_custom_prompt()
        base_prompt = custom_prompt or DEFAULT_AI_CHAT_SYSTEM_PROMPT

        mem_ctx: dict = {}
        try:
            mem_ctx = await build_live_and_learned_context(
                chat_id=chat_id,
                thread_id=thread_id,
                chat_type=chat_type,
                speaker_user_id=speaker_user_id,
            )
        except Exception as ctx_exc:
            logger.debug("Skipping live/learned context injection due to error: %s", ctx_exc)

        media_counts: dict[str, int] = {}
        try:
            media_counts = await get_ai_chat_media_counts_cached()
        except Exception as media_exc:
            logger.debug("Skipping AI chat media availability lookup: %s", media_exc)
        available_media_types = [k for k, v in media_counts.items() if v > 0]

        runtime_system = build_runtime_system_prompt(
            base_prompt,
            speaker_name=speaker_name,
            chat_type=chat_type,
            topic_name=topic_name,
            group_title=mem_ctx.get("group_title"),
            group_memory=mem_ctx.get("group_memory"),
            member_profiles=mem_ctx.get("member_profiles"),
            recent_messages_text=mem_ctx.get("recent_messages_text"),
            available_media_types=available_media_types,
        )

        raw_reply: str | None = None
        if isinstance(active_client, OpenAIClient):
            _, _, primary_model, _ = resolve_openai_compat_config(settings, for_chat=True, provider=provider)
            fallbacks = resolve_openai_chat_fallback_models(settings)
            candidate_models: list[str] = []
            for candidate in [primary_model, *fallbacks]:
                cleaned = str(candidate or "").strip()
                if cleaned and cleaned not in candidate_models:
                    candidate_models.append(cleaned)

            if len(candidate_models) <= 1:
                target_model = candidate_models[0] if candidate_models else None
                raw_reply = await async_retry(
                    lambda: active_client.generate_chat(
                        system_prompt=runtime_system,
                        messages=pending_turns,
                        temperature=settings.AI_CHAT_TEMPERATURE,
                        max_output_tokens=settings.AI_CHAT_MAX_OUTPUT_TOKENS,
                        model=target_model,
                    ),
                    retries=1,
                )
            else:
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
                                "9Router/OpenAI chat model '%s' failed (status=%s); falling back to next model...",
                                candidate_model,
                                exc.status_code,
                            )
                            continue
                        raise
        else:
            raw_fallbacks = getattr(settings, "AI_CHAT_FALLBACK_MODELS", None) or []
            candidate_models: list[str] = []
            for candidate in [settings.AI_CHAT_MODEL, *raw_fallbacks]:
                cleaned = str(candidate).strip()
                if cleaned and cleaned not in candidate_models:
                    candidate_models.append(cleaned)
            if not candidate_models:
                candidate_models = ["gemini-3.5-flash-lite"]

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

        parsed = parse_ai_chat_directives(raw_reply)
        sanitized_text = ""
        if parsed.text:
            sanitized_text = sanitize_telegram_html(parsed.text)
            if not sanitized_text:
                sanitized_text = escape_html(parsed.text.strip())

        history_text = format_directive_history_placeholder(
            ParsedAiDirectives(
                text=sanitized_text,
                reaction=parsed.reaction,
                media_type=parsed.media_type,
                media_mood=parsed.media_mood,
            )
        )

        if state is None:
            state = _ConversationState(messages=[], updated_at=now)
            _conversations[key] = state

        state.messages = _normalize_turns(
            [
                *existing_messages,
                {"role": "user", "text": prompt_input},
                {"role": "model", "text": history_text or sanitized_text},
            ],
            max_turns=history_limit,
        )
        state.updated_at = time.monotonic()

        if not parsed.reaction and not parsed.media_type:
            return sanitized_text
        if parsed.reaction and not sanitized_text and not parsed.media_type:
            return f"[REACT:{parsed.reaction}]"

        out_parts: list[str] = []
        if parsed.reaction:
            out_parts.append(f"[REACT:{parsed.reaction}]")
        if sanitized_text:
            out_parts.append(sanitized_text)
        if parsed.media_type:
            out_parts.append(canonical_media_tag(parsed.media_type, parsed.media_mood))
        return "\n".join(out_parts)

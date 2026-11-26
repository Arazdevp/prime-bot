"""Live group awareness and persistent long-term personality/culture learning for AI chat."""

from datetime import datetime
import json
import logging
import re
from typing import Any
from zoneinfo import ZoneInfo
from bot.config import (
    get_settings,
    resolve_openai_chat_fallback_models,
    resolve_openai_compat_config,
)
from bot.database import repository
from bot.services.llm_client import GeminiClient, LLMError, OpenAIClient, get_ai_chat_client
from bot.texts.ai_chat_texts import SYSTEM_PROMPT_LEARN_MEMORY
from bot.utils.retry import async_retry

logger = logging.getLogger(__name__)

_FREE_CHAT_KEYWORDS = ("گفتگوی آزاد", "گفتگو آزاد", "the hub")
_LEARN_BATCH_MAX_CHARS = 16000
_LEARN_MAX_BATCHES_PER_RUN = 5
AI_MEMORY_SWEEP_MIN_MESSAGES = 25
AI_MEMORY_PRE_PURGE_MIN_MESSAGES = 5

# Fast in-memory cache of chat_id -> allowed free-chat thread_id
_allowed_topic_cache: dict[int, int] = {}


def reset_ai_memory_caches() -> None:
    """Clear cached allowed-topic bindings (useful for test isolation)."""
    _allowed_topic_cache.clear()


def _normalize_topic_label(name: str | None) -> str:
    if not name:
        return ""
    cleaned = name.replace("ي", "ی").replace("ك", "ک").replace("\u200c", " ")
    return re.sub(r"\s+", " ", cleaned).strip().lower()


def is_free_chat_topic_name(name: str | None) -> bool:
    """Return True if topic name matches the community's free-chat / Hub topic."""
    norm = _normalize_topic_label(name)
    if not norm:
        return False
    return any(kw in norm for kw in _FREE_CHAT_KEYWORDS)


def _extract_json_object(raw_text: str) -> dict[str, Any] | None:
    """Extract and parse the outermost JSON object from an LLM response."""
    if not raw_text or not raw_text.strip():
        return None
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw_text.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        parsed = json.loads(cleaned[start : end + 1])
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def _format_tehran_hhmm(raw_created: str | None, tz_name: str) -> str:
    if not raw_created:
        return "00:00"
    try:
        dt = datetime.fromisoformat(raw_created)
        tz = ZoneInfo(tz_name)
        local_dt = dt.astimezone(tz) if dt.tzinfo else dt.replace(tzinfo=ZoneInfo("UTC")).astimezone(tz)
        return local_dt.strftime("%H:%M")
    except Exception:
        return "00:00"


async def resolve_memory_group(chat_id: int | None, chat_type: str) -> tuple[int | None, str | None]:
    """Resolve the target group chat_id and title for live context and long-term memory."""
    if chat_type != "private" and chat_id is not None:
        known = await repository.get_known_group(chat_id)
        title = str(known["title"]).strip() if known and known.get("title") else None
        return chat_id, title

    settings = get_settings()
    allowed = set(settings.ALLOWED_CHAT_IDS) if settings.ALLOWED_CHAT_IDS else None
    active_groups = [
        g for g in await repository.list_active_known_groups()
        if allowed is None or g["chat_id"] in allowed
    ]
    if not active_groups:
        return None, None

    sources = await repository.list_ai_memory_sources()
    active_ids = {int(g["chat_id"]): g for g in active_groups}
    for src in sources:
        cid = int(src["chat_id"])
        if cid in active_ids:
            title = str(active_ids[cid]["title"]).strip() if active_ids[cid].get("title") else None
            return cid, title

    for g in active_groups:
        cid = int(g["chat_id"])
        if await repository.count_ai_member_profiles(cid) > 0:
            title = str(g["title"]).strip() if g.get("title") else None
            return cid, title

    first = active_groups[0]
    title = str(first["title"]).strip() if first.get("title") else None
    return int(first["chat_id"]), title


async def ensure_memory_source_for_group(chat_id: int) -> dict[str, Any] | None:
    """Return existing memory source for chat_id, or auto-detect and register 'گفتگوی آزاد' / 'The Hub'."""
    sources = await repository.list_ai_memory_sources(chat_id)
    if sources:
        return sources[0]

    free_topic = await repository.find_free_chat_topic(chat_id)
    if not free_topic:
        return None

    return await repository.upsert_ai_memory_source(
        chat_id=chat_id,
        thread_id=int(free_topic["thread_id"]),
        topic_name=free_topic.get("name"),
    )


async def bind_memory_source_topic(
    chat_id: int,
    thread_id: int,
    topic_name: str | None = None,
) -> dict[str, Any]:
    """Explicitly bind a group topic as the AI memory learning source and active chat topic."""
    resolved_name = topic_name or await repository.get_topic_name(chat_id, thread_id)
    if not resolved_name:
        resolved_name = "عمومی" if thread_id == 1 else f"تاپیک {thread_id}"
    src = await repository.upsert_ai_memory_source(
        chat_id=chat_id,
        thread_id=thread_id,
        topic_name=resolved_name,
    )
    _allowed_topic_cache[chat_id] = int(thread_id)
    return src


async def is_allowed_ai_chat_topic(chat_id: int, thread_id: int) -> bool:
    """Return True ONLY when thread_id matches the group's free-chat ('گفتگوی آزاد') / bound topic."""
    cached_tid = _allowed_topic_cache.get(chat_id)
    if cached_tid is not None:
        return cached_tid == int(thread_id)

    source = await ensure_memory_source_for_group(chat_id)
    if source is not None:
        bound_tid = int(source["thread_id"])
        _allowed_topic_cache[chat_id] = bound_tid
        return bound_tid == int(thread_id)

    return False


def format_recent_group_messages(
    messages: list[dict[str, Any]],
    max_chars: int = 1300,
) -> str:
    """Format recent DB messages chronologically in Tehran time within max_chars."""
    if not messages:
        return ""
    tz_name = get_settings().TIMEZONE
    lines: list[str] = []
    for msg in messages:
        text = re.sub(r"\s+", " ", (msg.get("text") or "").strip())
        if not text:
            continue
        hhmm = _format_tehran_hhmm(msg.get("created_at"), tz_name)
        sender = (msg.get("sender_name") or "کاربر").strip()
        reply_name = (msg.get("reply_to_name") or "").strip()
        reply_part = f" (در پاسخ به {reply_name})" if reply_name else ""
        lines.append(f"[{hhmm}] {sender}{reply_part}: {text[:180]}")

    if not lines:
        return ""

    # Keep the newest lines when trimming to max_chars
    kept: list[str] = []
    total = 0
    for line in reversed(lines):
        add = len(line) + (1 if kept else 0)
        if kept and total + add > max_chars:
            break
        kept.append(line)
        total += add
    kept.reverse()
    return "\n".join(kept)


async def build_live_and_learned_context(
    *,
    chat_id: int,
    thread_id: int,
    chat_type: str,
    speaker_user_id: int | None,
) -> dict[str, Any]:
    """Load real-time recent messages plus persistent group and member memory for prompt injection."""
    mem_chat_id, group_title = await resolve_memory_group(chat_id, chat_type)
    if mem_chat_id is None:
        return {
            "memory_chat_id": None,
            "group_title": None,
            "group_memory": None,
            "member_profiles": [],
            "recent_messages_text": "",
        }

    priority_ids: list[int] = []
    if speaker_user_id is not None and speaker_user_id > 0:
        priority_ids.append(speaker_user_id)

    recent_text = ""
    if chat_type != "private":
        recent_msgs = await repository.fetch_recent_messages_for_topic(chat_id, thread_id, limit=14)
        for m in reversed(recent_msgs):
            uid = m.get("user_id")
            if isinstance(uid, int) and uid > 0 and uid not in priority_ids:
                priority_ids.append(uid)
        recent_text = format_recent_group_messages(recent_msgs, max_chars=1300)

    group_memory = await repository.get_ai_group_memory(mem_chat_id)
    member_profiles = await repository.list_ai_member_profiles(
        mem_chat_id,
        priority_user_ids=priority_ids,
        limit=8,
    )

    return {
        "memory_chat_id": mem_chat_id,
        "group_title": group_title,
        "group_memory": group_memory,
        "member_profiles": member_profiles,
        "recent_messages_text": recent_text,
    }


def _format_learning_batch(
    messages: list[dict[str, Any]],
    tz_name: str,
    max_chars: int = _LEARN_BATCH_MAX_CHARS,
) -> list[list[dict[str, Any]]]:
    """Group messages into chronological batches whose formatted transcript fits max_chars."""
    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_len = 0

    for msg in messages:
        text = re.sub(r"\s+", " ", (msg.get("text") or "").strip())
        if not text:
            continue
        uid = msg.get("user_id")
        uid_tag = f"[uid:{uid}] " if isinstance(uid, int) and uid > 0 else ""
        hhmm = _format_tehran_hhmm(msg.get("created_at"), tz_name)
        sender = (msg.get("sender_name") or "کاربر").strip()
        line_len = len(f"{uid_tag}[{hhmm}] {sender}: {text[:400]}") + 1
        if current and current_len + line_len > max_chars:
            batches.append(current)
            current = [msg]
            current_len = line_len
        else:
            current.append(msg)
            current_len += line_len

    if current:
        batches.append(current)
    return batches


async def _call_memory_learner(
    active_client: GeminiClient | OpenAIClient,
    user_content: str,
) -> str:
    """Invoke the active chat LLM provider with low temperature to extract structured JSON memory."""
    settings = get_settings()
    if isinstance(active_client, OpenAIClient):
        _, _, primary_model, _ = resolve_openai_compat_config(settings, for_chat=True)
        fallbacks = resolve_openai_chat_fallback_models(settings)
        candidate_models: list[str] = []
        for candidate in [primary_model, *fallbacks]:
            cleaned = str(candidate or "").strip()
            if cleaned and cleaned not in candidate_models:
                candidate_models.append(cleaned)

        if len(candidate_models) <= 1:
            target_model = candidate_models[0] if candidate_models else None
            return await async_retry(
                lambda: active_client.generate_chat(
                    system_prompt=SYSTEM_PROMPT_LEARN_MEMORY,
                    messages=[{"role": "user", "text": user_content}],
                    temperature=0.3,
                    max_output_tokens=2048,
                    model=target_model,
                ),
                retries=1,
            )

        for idx, candidate_model in enumerate(candidate_models):
            is_last = idx == len(candidate_models) - 1
            try:
                return await async_retry(
                    lambda m=candidate_model: active_client.generate_chat(
                        system_prompt=SYSTEM_PROMPT_LEARN_MEMORY,
                        messages=[{"role": "user", "text": user_content}],
                        temperature=0.3,
                        max_output_tokens=2048,
                        model=m,
                    ),
                    retries=1 if is_last else 0,
                )
            except LLMError as exc:
                if not is_last and exc.status_code in (404, 429, 503):
                    continue
                raise

        raise LLMError("All AI memory learning model candidates failed", status_code=503, is_retryable=True)

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
            return await async_retry(
                lambda m=candidate_model: active_client.generate_chat(
                    system_prompt=SYSTEM_PROMPT_LEARN_MEMORY,
                    messages=[{"role": "user", "text": user_content}],
                    temperature=0.3,
                    max_output_tokens=2048,
                    model=m,
                ),
                retries=1 if is_last else 0,
            )
        except LLMError as exc:
            if not is_last and exc.status_code in (404, 429, 503):
                continue
            raise

    raise LLMError("All AI memory learning model candidates failed", status_code=503, is_retryable=True)


async def learn_from_topic_messages(
    chat_id: int,
    thread_id: int,
    *,
    messages: list[dict[str, Any]] | None = None,
    topic_name: str | None = None,
    client: GeminiClient | OpenAIClient | None = None,
    force_full_scan: bool = False,
) -> dict[str, Any]:
    """Analyze topic messages and incrementally persist group culture and member profiles."""
    from bot.services.ai_chat_service import get_effective_ai_chat_provider

    resolved_name = topic_name or await repository.get_topic_name(chat_id, thread_id)
    source = await repository.upsert_ai_memory_source(
        chat_id=chat_id,
        thread_id=thread_id,
        topic_name=resolved_name,
    )
    last_id = 0 if force_full_scan else int(source.get("last_learned_message_id") or 0)

    if messages is None:
        target_messages = await repository.fetch_messages_after_id(
            chat_id,
            thread_id,
            after_message_id=last_id,
            limit=500,
        )
    else:
        target_messages = [
            m for m in messages
            if int(m.get("message_id") or 0) > last_id
        ]

    if not target_messages:
        return {
            "messages_processed": 0,
            "members_updated": 0,
            "group_updated": False,
            "topic_name": source.get("topic_name") or resolved_name or f"تاپیک {thread_id}",
        }

    settings = get_settings()
    provider = get_effective_ai_chat_provider()
    active_client = client or get_ai_chat_client(settings, provider=provider)
    known = await repository.get_known_group(chat_id)
    group_title = str(known["title"]).strip() if known and known.get("title") else None

    batches = _format_learning_batch(target_messages, settings.TIMEZONE)[:_LEARN_MAX_BATCHES_PER_RUN]
    total_messages_processed = 0
    updated_user_ids: set[int] = set()
    group_updated = False

    for batch in batches:
        batch_user_counts: dict[int, int] = {}
        batch_user_names: dict[int, str] = {}
        batch_user_last_seen: dict[int, str] = {}
        batch_lines: list[str] = []
        max_msg_id = last_id

        for msg in batch:
            mid = int(msg.get("message_id") or 0)
            if mid > max_msg_id:
                max_msg_id = mid
            uid = msg.get("user_id")
            sender = (msg.get("sender_name") or "کاربر").strip()
            text = re.sub(r"\s+", " ", (msg.get("text") or "").strip())
            if not text:
                continue
            hhmm = _format_tehran_hhmm(msg.get("created_at"), settings.TIMEZONE)
            if isinstance(uid, int) and uid > 0:
                batch_user_counts[uid] = batch_user_counts.get(uid, 0) + 1
                batch_user_names[uid] = sender
                if msg.get("created_at"):
                    batch_user_last_seen[uid] = str(msg["created_at"])
                batch_lines.append(f"[uid:{uid}] [{hhmm}] {sender}: {text[:400]}")
            else:
                batch_lines.append(f"[{hhmm}] {sender}: {text[:400]}")

        if not batch_lines:
            if max_msg_id > last_id:
                await repository.update_ai_memory_source_progress(chat_id, thread_id, max_msg_id, resolved_name)
                last_id = max_msg_id
            continue

        existing_group = await repository.get_ai_group_memory(chat_id)
        existing_profiles = await repository.list_ai_member_profiles(
            chat_id,
            priority_user_ids=list(batch_user_counts.keys()),
            limit=max(20, len(batch_user_counts)),
        )
        existing_by_uid = {
            int(p["user_id"]): {
                "user_id": int(p["user_id"]),
                "display_name": p["display_name"],
                "personality_summary": p["personality_summary"],
                "speaking_style": p.get("speaking_style") or "",
                "interests_and_topics": p.get("interests_and_topics") or "",
                "notable_facts": p.get("notable_facts") or "",
            }
            for p in existing_profiles
            if int(p["user_id"]) in batch_user_counts
        }

        existing_context = {
            "group_memory": {
                "group_culture": existing_group.get("culture_summary", "") if existing_group else "",
                "inside_jokes": existing_group.get("inside_jokes", "") if existing_group else "",
                "ongoing_topics": existing_group.get("ongoing_topics", "") if existing_group else "",
            },
            "existing_members": list(existing_by_uid.values()),
        }

        user_content = (
            f"حافظه فعلی:\n{json.dumps(existing_context, ensure_ascii=False)}\n\n"
            f"پیام‌های جدید تاپیک «{source.get('topic_name') or resolved_name or 'گفتگوی آزاد'}»:\n"
            + "\n".join(batch_lines)
        )

        raw_out = await _call_memory_learner(active_client, user_content)
        parsed = _extract_json_object(raw_out)
        if parsed is None:
            logger.warning("AI memory learning returned invalid JSON for chat %s thread %s; skipping batch.", chat_id, thread_id)
            break

        prev_culture = existing_group.get("culture_summary", "") if existing_group else ""
        prev_jokes = existing_group.get("inside_jokes", "") if existing_group else ""
        prev_topics = existing_group.get("ongoing_topics", "") if existing_group else ""

        new_culture = str(parsed.get("group_culture") or "").strip() or prev_culture
        new_jokes = str(parsed.get("inside_jokes") or "").strip() or prev_jokes
        new_topics = str(parsed.get("ongoing_topics") or "").strip() or prev_topics

        if new_culture or new_jokes or new_topics:
            await repository.upsert_ai_group_memory(
                chat_id=chat_id,
                group_title=group_title,
                culture_summary=new_culture,
                inside_jokes=new_jokes,
                ongoing_topics=new_topics,
            )
            group_updated = True

        raw_members = parsed.get("members")
        if isinstance(raw_members, list):
            for item in raw_members:
                if not isinstance(item, dict):
                    continue
                try:
                    uid = int(item.get("user_id"))
                except (TypeError, ValueError):
                    continue
                if uid not in batch_user_counts:
                    continue
                personality = str(item.get("personality_summary") or "").strip()
                if not personality:
                    continue
                display_name = str(item.get("display_name") or batch_user_names.get(uid) or "عضو").strip()
                await repository.upsert_ai_member_profile(
                    chat_id=chat_id,
                    user_id=uid,
                    display_name=display_name,
                    personality_summary=personality,
                    speaking_style=str(item.get("speaking_style") or "").strip() or None,
                    interests_and_topics=str(item.get("interests_and_topics") or "").strip() or None,
                    notable_facts=str(item.get("notable_facts") or "").strip() or None,
                    add_message_count=batch_user_counts.get(uid, 0),
                    last_seen_at=batch_user_last_seen.get(uid),
                )
                updated_user_ids.add(uid)

        if max_msg_id > last_id:
            await repository.update_ai_memory_source_progress(chat_id, thread_id, max_msg_id, resolved_name)
            last_id = max_msg_id
        total_messages_processed += len(batch)

    return {
        "messages_processed": total_messages_processed,
        "members_updated": len(updated_user_ids),
        "group_updated": group_updated,
        "topic_name": source.get("topic_name") or resolved_name or f"تاپیک {thread_id}",
    }


async def maybe_learn_before_purge(
    chat_id: int,
    thread_id: int,
    topic_name: str | None,
    messages: list[dict[str, Any]],
) -> bool:
    """Learn from free-chat / bound memory topic messages right before nightly deletion."""
    if not messages or len(messages) < AI_MEMORY_PRE_PURGE_MIN_MESSAGES:
        return False

    from bot.services.ai_chat_service import is_ai_chat_enabled_cached

    existing_source = await repository.get_ai_memory_source(chat_id, thread_id)
    if existing_source is None:
        if not (is_ai_chat_enabled_cached() and is_free_chat_topic_name(topic_name)):
            return False

    try:
        result = await learn_from_topic_messages(
            chat_id,
            thread_id,
            messages=messages,
            topic_name=topic_name,
        )
        return int(result.get("messages_processed") or 0) > 0
    except Exception as exc:
        logger.warning("Pre-purge AI memory learning skipped for chat %s thread %s: %s", chat_id, thread_id, exc)
        return False


async def ai_memory_learn_sweep() -> None:
    """Periodic background sweep learning new messages from bound/free-chat topics when AI chat is ON."""
    from bot.services.ai_chat_service import is_ai_chat_enabled_cached

    if not is_ai_chat_enabled_cached():
        return

    settings = get_settings()
    allowed = set(settings.ALLOWED_CHAT_IDS) if settings.ALLOWED_CHAT_IDS else None
    groups = [
        g for g in await repository.list_active_known_groups()
        if allowed is None or g["chat_id"] in allowed
    ]
    for group in groups:
        chat_id = int(group["chat_id"])
        try:
            source = await ensure_memory_source_for_group(chat_id)
            if not source:
                continue
            thread_id = int(source["thread_id"])
            last_id = int(source.get("last_learned_message_id") or 0)
            pending_cnt = await repository.count_messages_after_id(chat_id, thread_id, after_message_id=last_id)
            if pending_cnt >= AI_MEMORY_SWEEP_MIN_MESSAGES:
                await learn_from_topic_messages(chat_id, thread_id, topic_name=source.get("topic_name"))
        except Exception as exc:
            logger.warning("AI memory sweep failed for chat %s: %s", chat_id, exc)

"""Listener handler to capture group messages in real time."""

from datetime import timezone
import logging
import time
from aiogram import Bot, Router, F
from aiogram.types import ChatMemberUpdated, Message
from bot.config import get_settings
from bot.database import repository
from bot.filters import ForumGroupFilter
from bot.services.deeptalk_service import get_live_session_id_for_topic
from bot.texts.private_texts import pick
from bot.utils.general_topic import resolve_thread_id
from bot.utils.noise_filter import is_noise

logger = logging.getLogger(__name__)

router = Router(name="listener_router")
# Apply ForumGroupFilter to message events only (my_chat_member sees all groups for auto-leave/tracking)
router.message.filter(ForumGroupFilter())

# Throttled in-memory cache for known_groups upserts: chat_id -> (title, is_forum, last_ts)
_known_group_seen: dict[int, tuple[str | None, bool, float]] = {}
KNOWN_GROUP_UPSERT_TTL_SECONDS: float = 300.0


def clear_known_group_cache() -> None:
    """Clear the throttled known_groups upsert cache (useful for test isolation)."""
    _known_group_seen.clear()


def _extract_chat_meta(chat: object) -> tuple[str | None, bool]:
    """Safely extract clean title string and is_forum boolean from a Chat-like object."""
    raw_title = getattr(chat, "title", None)
    title = raw_title.strip() if isinstance(raw_title, str) and raw_title.strip() else None
    is_forum = bool(getattr(chat, "is_forum", False))
    return title, is_forum


async def _maybe_upsert_known_group(chat: object) -> None:
    """Upsert group metadata into known_groups on first seen, metadata change, or TTL expiry."""
    chat_id = getattr(chat, "id", None)
    if not isinstance(chat_id, int):
        return

    title, is_forum = _extract_chat_meta(chat)
    now = time.monotonic()
    cached = _known_group_seen.get(chat_id)
    if (
        cached is not None
        and cached[0] == title
        and cached[1] == is_forum
        and (now - cached[2]) < KNOWN_GROUP_UPSERT_TTL_SECONDS
    ):
        return

    await repository.upsert_known_group(chat_id=chat_id, title=title, is_forum=is_forum)
    _known_group_seen[chat_id] = (title, is_forum, now)


@router.my_chat_member()
async def on_my_chat_member(event: ChatMemberUpdated, bot: Bot) -> None:
    """Track bot group additions/removals, send welcome, and enforce AUTO_LEAVE_UNKNOWN_GROUPS."""
    try:
        chat = event.chat
        if not chat or chat.type not in ("group", "supergroup"):
            return

        old_status = event.old_chat_member.status
        new_status = event.new_chat_member.status
        title, is_forum = _extract_chat_meta(chat)

        was_outside = old_status in ("left", "kicked")
        is_inside = new_status in ("member", "administrator")
        is_outside = new_status in ("left", "kicked")

        if was_outside and is_inside:
            settings = get_settings()
            if (
                settings.AUTO_LEAVE_UNKNOWN_GROUPS
                and settings.ALLOWED_CHAT_IDS
                and chat.id not in settings.ALLOWED_CHAT_IDS
            ):
                leave_text, _ = pick(
                    "group_not_allowed",
                    owners=settings.OWNER_NAMES,
                    escape=True,
                )
                try:
                    await bot.send_message(
                        chat_id=chat.id,
                        text=leave_text,
                        parse_mode="HTML",
                        disable_web_page_preview=True,
                    )
                except Exception as exc:
                    logger.warning("Failed sending group_not_allowed to chat %s: %s", chat.id, exc, exc_info=True)
                try:
                    await bot.leave_chat(chat.id)
                except Exception as exc:
                    logger.error("Failed leaving unauthorized group %s: %s", chat.id, exc, exc_info=True)
                await repository.mark_known_group_left(chat.id)
                _known_group_seen.pop(chat.id, None)
                return

            await repository.upsert_known_group(chat_id=chat.id, title=title, is_forum=is_forum)
            _known_group_seen[chat.id] = (title, is_forum, time.monotonic())
            await repository.ensure_group_settings(chat.id)

            welcome_text, _ = pick("group_added", owners=settings.OWNER_NAMES, escape=True)
            try:
                await bot.send_message(
                    chat_id=chat.id,
                    text=welcome_text,
                    parse_mode="HTML",
                    disable_web_page_preview=True,
                )
            except Exception as exc:
                logger.warning("Failed sending group_added welcome to chat %s: %s", chat.id, exc, exc_info=True)
            return

        if is_inside:
            await repository.upsert_known_group(chat_id=chat.id, title=title, is_forum=is_forum)
            _known_group_seen[chat.id] = (title, is_forum, time.monotonic())
            return

        if is_outside:
            await repository.mark_known_group_left(chat.id)
            _known_group_seen.pop(chat.id, None)
    except Exception as exc:
        logger.error("Unexpected error handling my_chat_member: %s", exc, exc_info=True)


@router.message(F.forum_topic_created)
async def on_forum_topic_created(message: Message) -> None:
    """Learn topic name when a new forum topic is created."""
    try:
        if not message.chat or not message.forum_topic_created:
            return
        await _maybe_upsert_known_group(message.chat)
        thread_id = resolve_thread_id(message)
        topic_name = message.forum_topic_created.name
        await repository.upsert_topic(
            chat_id=message.chat.id,
            thread_id=thread_id,
            name=topic_name,
        )
        logger.info(
            "Registered new forum topic in chat %s: %s (id: %s)",
            message.chat.id,
            topic_name,
            thread_id,
        )
    except Exception as exc:
        logger.error("Failed handling forum_topic_created: %s", exc, exc_info=True)


@router.message(F.forum_topic_edited)
async def on_forum_topic_edited(message: Message) -> None:
    """Update topic name when a forum topic is edited."""
    try:
        if not message.chat or not message.forum_topic_edited:
            return
        await _maybe_upsert_known_group(message.chat)
        thread_id = resolve_thread_id(message)
        topic_name = message.forum_topic_edited.name
        if topic_name:
            await repository.upsert_topic(
                chat_id=message.chat.id,
                thread_id=thread_id,
                name=topic_name,
            )
            logger.info(
                "Updated forum topic in chat %s: %s (id: %s)",
                message.chat.id,
                topic_name,
                thread_id,
            )
    except Exception as exc:
        logger.error("Failed handling forum_topic_edited: %s", exc, exc_info=True)


@router.message()
async def capture_message(message: Message) -> None:
    """Capture and store qualifying group messages for later summarization."""
    try:
        if not message.chat:
            return

        await _maybe_upsert_known_group(message.chat)

        # Check for topic name in reply_to_message.forum_topic_created before noise/monitored checks
        if (
            message.reply_to_message
            and message.reply_to_message.forum_topic_created
            and message.reply_to_message.forum_topic_created.name
        ):
            raw_name = message.reply_to_message.forum_topic_created.name
            topic_name = raw_name.strip() if raw_name else None
            if topic_name:
                chat_id = message.chat.id
                thread_id = resolve_thread_id(message)
                if thread_id == 1 and message.reply_to_message.message_id:
                    thread_id = message.reply_to_message.message_id
                current_name = await repository.get_topic_name(chat_id, thread_id)
                if current_name != topic_name:
                    await repository.upsert_topic(
                        chat_id=chat_id,
                        thread_id=thread_id,
                        name=topic_name,
                    )

        # 1. Skip bots (unless sender_chat is set, e.g. GroupAnonymousBot)
        if message.from_user and message.from_user.is_bot and not message.sender_chat:
            return

        # 2. Skip animations, stickers, and purely decorative media
        if message.animation or message.sticker:
            return

        # 3. Extract text and identify media type
        text: str | None = None
        media_type: str | None = None

        if message.text:
            text = message.text
            media_type = None
        elif message.photo:
            text = message.caption
            media_type = "photo"
        elif message.video:
            text = message.caption
            media_type = "video"
        elif message.document:
            text = message.caption
            media_type = "document"
        elif message.voice:
            text = message.caption
            media_type = "voice"
        elif message.audio:
            text = message.caption
            media_type = "audio"
        elif message.video_note:
            text = message.caption
            media_type = "video_note"

        # If there is no text or caption, there is nothing to summarize
        if not text or not text.strip():
            return

        # 4. Skip bot commands
        if text.strip().startswith("/"):
            return

        # 4b. Track live DeepTalk participation stats (O(1) memory lookup; zero DB hit when idle)
        live_dt_sid = get_live_session_id_for_topic(message.chat.id, resolve_thread_id(message))
        if live_dt_sid is not None and message.from_user and not message.from_user.is_bot:
            try:
                await repository.increment_deeptalk_message_count(
                    session_id=live_dt_sid,
                    user_id=message.from_user.id,
                    username=message.from_user.username,
                    full_name=message.from_user.full_name or message.from_user.first_name or "کاربر",
                )
            except Exception as dt_exc:
                logger.debug("Failed incrementing DeepTalk message count: %s", dt_exc)

        # 5. Apply noise filter
        settings = get_settings()
        if is_noise(text, min_len=settings.MIN_MESSAGE_LENGTH):
            return

        # 6. Resolve thread ID and check if topic is monitored
        chat_id = message.chat.id
        thread_id = resolve_thread_id(message)

        # Ensure General topic has a name registered
        if thread_id == 1:
            existing_name = await repository.get_topic_name(chat_id, 1)
            if not existing_name:
                await repository.upsert_topic(chat_id, 1, name="عمومی")

        if not await repository.is_topic_monitored(chat_id, thread_id):
            return

        # 7. Extract sender name
        sender_name: str
        user_id: int | None = None
        if message.sender_chat:
            if message.chat and message.sender_chat.id == message.chat.id:
                sender_name = "ادمین ناشناس"
            else:
                sender_name = message.sender_chat.title or "کانال/گروه"
            user_id = message.sender_chat.id
        elif message.from_user:
            sender_name = message.from_user.full_name
            user_id = message.from_user.id
        else:
            sender_name = "کاربر"

        # 8. Extract reply context (ignoring topic root creation service messages)
        reply_to_message_id: int | None = None
        reply_to_name: str | None = None
        reply_to_snippet: str | None = None

        reply = message.reply_to_message
        if reply and not reply.forum_topic_created:
            reply_to_message_id = reply.message_id
            if reply.sender_chat and message.chat and reply.sender_chat.id == message.chat.id:
                reply_to_name = "ادمین ناشناس"
            elif reply.sender_chat:
                reply_to_name = reply.sender_chat.title or "کانال/گروه"
            elif reply.from_user:
                reply_to_name = reply.from_user.full_name
            else:
                reply_to_name = "کاربر"

            reply_text = reply.text or reply.caption or ""
            if reply_text:
                reply_to_snippet = reply_text.strip()[:80]

        # 9. Format timestamp as UTC ISO8601
        created_at = (
            message.date.astimezone(timezone.utc).isoformat()
            if message.date.tzinfo
            else message.date.replace(tzinfo=timezone.utc).isoformat()
        )

        # 10. Store message in database
        inserted = await repository.insert_message(
            chat_id=chat_id,
            thread_id=thread_id,
            message_id=message.message_id,
            user_id=user_id,
            sender_name=sender_name,
            text=text.strip(),
            media_type=media_type,
            reply_to_message_id=reply_to_message_id,
            reply_to_name=reply_to_name,
            reply_to_snippet=reply_to_snippet,
            created_at=created_at,
        )

        if inserted:
            logger.debug(
                "Captured message %s in chat %s, thread %s from %s",
                message.message_id,
                chat_id,
                thread_id,
                sender_name,
            )
    except Exception as exc:
        logger.error("Unexpected error in message listener: %s", exc, exc_info=True)

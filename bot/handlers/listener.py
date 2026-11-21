"""Listener handler to capture group messages in real time."""

from datetime import timezone
import logging
from aiogram import Router, F
from aiogram.types import Message
from bot.config import get_settings
from bot.database import repository
from bot.filters import ForumGroupFilter
from bot.utils.general_topic import resolve_thread_id
from bot.utils.noise_filter import is_noise

logger = logging.getLogger(__name__)

router = Router(name="listener_router")
# Apply ForumGroupFilter to ensure messages come from supergroups
router.message.filter(ForumGroupFilter())


@router.message(F.forum_topic_created)
async def on_forum_topic_created(message: Message) -> None:
    """Learn topic name when a new forum topic is created."""
    try:
        if not message.chat or not message.forum_topic_created:
            return
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
